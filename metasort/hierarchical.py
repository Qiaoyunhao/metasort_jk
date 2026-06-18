from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

from .metasort import MetaSortConfig, MetaSortResult, MetaSortSolver


@dataclass
class HierarchicalMetaSortConfig(MetaSortConfig):
    linkage_method: str = "average"
    coarse_group_count: int | None = None
    max_genes_per_stage: int | None = 150
    min_genes_per_stage: int = 30
    gene_score_epsilon: float = 1e-12
    single_cell_log1p: bool = True
    single_cell_equal_subject_weight: bool = True


def load_single_cell_hierarchy_inputs(
    data_root: str | Path,
    expr_name: str = "singleCellExpr.txt",
    labels_name: str = "singleCellLabels.txt",
    subjects_name: str = "singleCellSubjects.txt",
) -> tuple[np.ndarray, list[str], list[str] | None]:
    root = Path(data_root)
    expr_df = pd.read_csv(root / expr_name, sep="\t", index_col=0)
    labels = pd.read_csv(root / labels_name, sep="\t").iloc[:, 0].astype(str).to_list()
    subjects_path = root / subjects_name
    subjects = None
    if subjects_path.exists():
        subjects = pd.read_csv(subjects_path, sep="\t").iloc[:, 0].astype(str).to_list()
    return expr_df.to_numpy(dtype=float), labels, subjects


@dataclass
class HierarchyNode:
    name: str
    cell_types: list[str]
    children: list["HierarchyNode"] = field(default_factory=list)
    distance: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cell_types": list(self.cell_types),
            "children": [child.to_dict() for child in self.children],
            "distance": float(self.distance),
        }


@dataclass
class HierarchicalStageResult:
    stage: int
    visible_nodes: list[str]
    expanded_parents: list[str]
    selected_gene_count: int
    raw_proportions: dict[str, float]
    constrained_proportions: dict[str, float]
    result: MetaSortResult | None
    parent_results: dict[str, MetaSortResult] = field(default_factory=dict)
    selected_gene_count_by_parent: dict[str, int] = field(default_factory=dict)


@dataclass
class HierarchicalMetaSortResult:
    proportions: list[float]
    cell_types: list[str]
    hierarchy: dict[str, Any]
    hierarchy_source: str
    stage_results: list[HierarchicalStageResult]
    config: dict[str, Any]


class HierarchicalMetaSortSolver:
    def __init__(self, config: HierarchicalMetaSortConfig | None = None) -> None:
        self.config = config or HierarchicalMetaSortConfig()
        self.base_solver = MetaSortSolver(self.config)

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(values, dtype=float), 0.0, None)
        total = float(np.sum(values))
        if total <= 0.0:
            return np.full(values.shape, 1.0 / len(values), dtype=float)
        return values / total

    @staticmethod
    def _leaf_indices(node: HierarchyNode, cell_type_to_index: dict[str, int]) -> list[int]:
        return [cell_type_to_index[cell_type] for cell_type in node.cell_types]

    @staticmethod
    def _aggregate_signature(
        signature: np.ndarray,
        nodes: list[HierarchyNode],
        cell_type_to_index: dict[str, int],
    ) -> np.ndarray:
        columns = []
        for node in nodes:
            indices = HierarchicalMetaSortSolver._leaf_indices(node, cell_type_to_index)
            columns.append(np.mean(signature[:, indices], axis=1))
        return np.column_stack(columns)

    def build_hierarchy(
        self,
        signature: np.ndarray,
        cell_types: list[str],
    ) -> HierarchyNode:
        signature = np.asarray(signature, dtype=float)
        if signature.ndim != 2:
            raise ValueError("signature must be 2D.")
        if signature.shape[1] != len(cell_types):
            raise ValueError("cell_types length must match signature columns.")
        if len(cell_types) == 0:
            raise ValueError("cell_types must not be empty.")
        if len(set(cell_types)) != len(cell_types):
            raise ValueError("cell_types must be unique.")
        if len(cell_types) == 1:
            return HierarchyNode(name=cell_types[0], cell_types=[cell_types[0]])

        corr = np.corrcoef(signature.T)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        distance = np.clip(1.0 - corr, 0.0, 2.0)
        np.fill_diagonal(distance, 0.0)
        condensed = squareform(distance, checks=False)
        z_matrix = linkage(condensed, method=self.config.linkage_method)
        scipy_root = to_tree(z_matrix, rd=False)

        def convert(scipy_node: Any) -> HierarchyNode:
            if scipy_node.is_leaf():
                cell_type = cell_types[scipy_node.id]
                return HierarchyNode(name=cell_type, cell_types=[cell_type])
            left = convert(scipy_node.left)
            right = convert(scipy_node.right)
            merged_cell_types = left.cell_types + right.cell_types
            return HierarchyNode(
                name=f"__hierarchical_node_{scipy_node.id}",
                cell_types=merged_cell_types,
                children=[left, right],
                distance=float(scipy_node.dist),
            )

        return convert(scipy_root)

    def _single_cell_profiles(
        self,
        single_cell_expr: np.ndarray,
        single_cell_labels: list[str],
        cell_types: list[str],
        single_cell_subjects: list[str] | None = None,
    ) -> np.ndarray:
        expr = np.asarray(single_cell_expr, dtype=float)
        if expr.ndim != 2:
            raise ValueError("single_cell_expr must be 2D.")
        labels = np.asarray(single_cell_labels, dtype=str)
        if expr.shape[1] != labels.shape[0]:
            raise ValueError("single_cell_expr columns must match single_cell_labels length.")

        subjects = None
        if single_cell_subjects is not None:
            subjects = np.asarray(single_cell_subjects, dtype=str)
            if subjects.shape[0] != labels.shape[0]:
                raise ValueError("single_cell_subjects length must match single_cell_labels length.")

        if self.config.single_cell_log1p:
            expr = np.log1p(np.clip(expr, 0.0, None))

        profiles = []
        for cell_type in cell_types:
            cell_indices = np.flatnonzero(labels == cell_type)
            if len(cell_indices) == 0:
                raise ValueError(f"No single cells found for cell type {cell_type}.")

            if (
                subjects is not None
                and self.config.single_cell_equal_subject_weight
            ):
                subject_profiles = []
                for subject in np.unique(subjects[cell_indices]):
                    subject_indices = cell_indices[subjects[cell_indices] == subject]
                    if len(subject_indices) > 0:
                        subject_profiles.append(np.mean(expr[:, subject_indices], axis=1))
                profiles.append(np.mean(np.column_stack(subject_profiles), axis=1))
            else:
                profiles.append(np.mean(expr[:, cell_indices], axis=1))
        return np.column_stack(profiles)

    def build_hierarchy_from_single_cell(
        self,
        single_cell_expr: np.ndarray,
        single_cell_labels: list[str],
        cell_types: list[str],
        single_cell_subjects: list[str] | None = None,
    ) -> HierarchyNode:
        profiles = self._single_cell_profiles(
            single_cell_expr=single_cell_expr,
            single_cell_labels=single_cell_labels,
            cell_types=cell_types,
            single_cell_subjects=single_cell_subjects,
        )
        return self.build_hierarchy(profiles, cell_types)

    @staticmethod
    def _walk_hierarchy(node: HierarchyNode) -> list[HierarchyNode]:
        nodes = [node]
        for child in node.children:
            nodes.extend(HierarchicalMetaSortSolver._walk_hierarchy(child))
        return nodes

    @staticmethod
    def _validate_hierarchy(root: HierarchyNode, cell_types: list[str]) -> None:
        nodes = HierarchicalMetaSortSolver._walk_hierarchy(root)
        node_names = [node.name for node in nodes]
        if len(set(node_names)) != len(node_names):
            raise ValueError("Hierarchy node names must be unique.")

        leaf_names = [node.name for node in nodes if node.is_leaf]
        if len(set(leaf_names)) != len(leaf_names):
            raise ValueError("Hierarchy leaf names must be unique.")
        if set(leaf_names) != set(cell_types) or len(leaf_names) != len(cell_types):
            raise ValueError("Hierarchy leaves must match cell_types exactly.")

        for node in nodes:
            child_cell_types = []
            for child in node.children:
                child_cell_types.extend(child.cell_types)
            if node.children and set(child_cell_types) != set(node.cell_types):
                raise ValueError(f"Hierarchy node {node.name} has inconsistent child cell types.")

    def _coarse_group_count(self, n_cell_types: int) -> int:
        if n_cell_types <= 1:
            return 1
        if self.config.coarse_group_count is not None:
            if self.config.coarse_group_count < 2:
                raise ValueError("coarse_group_count must be at least 2 when provided.")
            return min(n_cell_types, int(self.config.coarse_group_count))
        return min(n_cell_types, max(2, int(np.ceil(np.sqrt(n_cell_types)))))

    def _cut_to_coarse_groups(self, root: HierarchyNode) -> list[HierarchyNode]:
        target_count = self._coarse_group_count(len(root.cell_types))
        visible_nodes = [root]
        while len(visible_nodes) < target_count:
            candidates = [node for node in visible_nodes if not node.is_leaf]
            if not candidates:
                break
            split_node = max(candidates, key=lambda node: (node.distance, len(node.cell_types)))
            split_index = visible_nodes.index(split_node)
            visible_nodes = (
                visible_nodes[:split_index]
                + split_node.children
                + visible_nodes[split_index + 1 :]
            )
        return visible_nodes

    def _select_stage_genes(
        self,
        signature: np.ndarray,
        visible_nodes: list[HierarchyNode],
        cell_type_to_index: dict[str, int],
    ) -> np.ndarray:
        n_genes = signature.shape[0]
        if self.config.max_genes_per_stage is None:
            return np.arange(n_genes)
        if self.config.max_genes_per_stage <= 0:
            raise ValueError("max_genes_per_stage must be positive or None.")

        aggregate_signature = self._aggregate_signature(signature, visible_nodes, cell_type_to_index)
        between = np.var(aggregate_signature, axis=1)

        within_parts = []
        for node in visible_nodes:
            indices = self._leaf_indices(node, cell_type_to_index)
            if len(indices) == 1:
                within_parts.append(np.zeros(n_genes, dtype=float))
            else:
                within_parts.append(np.var(signature[:, indices], axis=1))
        within = np.mean(np.vstack(within_parts), axis=0)
        score = between / (within + self.config.gene_score_epsilon)
        score = np.nan_to_num(score, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)

        n_select = min(
            n_genes,
            max(self.config.min_genes_per_stage, self.config.max_genes_per_stage),
        )
        if n_select >= n_genes:
            return np.arange(n_genes)
        return np.argsort(score)[-n_select:]

    @staticmethod
    def _expand_visible_nodes(
        visible_nodes: list[HierarchyNode],
    ) -> tuple[list[HierarchyNode], list[HierarchyNode]]:
        next_nodes: list[HierarchyNode] = []
        expanded_parents: list[HierarchyNode] = []
        for node in visible_nodes:
            if node.is_leaf:
                next_nodes.append(node)
            else:
                next_nodes.extend(node.children)
                expanded_parents.append(node)
        return next_nodes, expanded_parents

    def _solve_stage(
        self,
        signature: np.ndarray,
        bulk: np.ndarray,
        nodes: list[HierarchyNode],
        cell_type_to_index: dict[str, int],
    ) -> tuple[np.ndarray, MetaSortResult, dict[str, float]]:
        selected_genes = self._select_stage_genes(
            signature=signature,
            visible_nodes=nodes,
            cell_type_to_index=cell_type_to_index,
        )
        stage_signature = self._aggregate_signature(
            signature=signature[selected_genes, :],
            nodes=nodes,
            cell_type_to_index=cell_type_to_index,
        )
        stage_bulk = bulk[selected_genes]
        stage_names = [node.name for node in nodes]
        result = self.base_solver.solve(stage_signature, stage_bulk, cell_types=stage_names)
        raw_props = {
            node.name: float(prop)
            for node, prop in zip(nodes, result.proportions)
        }
        return selected_genes, result, raw_props

    def _build_parent_local_bulk(
        self,
        signature: np.ndarray,
        bulk: np.ndarray,
        parent: HierarchyNode,
        visible_nodes: list[HierarchyNode],
        masses: dict[str, float],
        cell_type_to_index: dict[str, int],
    ) -> np.ndarray:
        fitted_bulk = np.zeros_like(np.asarray(bulk, dtype=float))
        parent_profile = None
        for node in visible_nodes:
            node_profile = self._aggregate_signature(
                signature=signature,
                nodes=[node],
                cell_type_to_index=cell_type_to_index,
            )[:, 0]
            fitted_bulk = fitted_bulk + masses[node.name] * node_profile
            if node.name == parent.name:
                parent_profile = node_profile

        if parent_profile is None:
            raise ValueError(f"Parent node {parent.name} is not visible in the current stage.")

        parent_mass = max(float(masses[parent.name]), self.config.min_weight_floor)
        residual = np.asarray(bulk, dtype=float) - fitted_bulk
        local_bulk = parent_mass * (parent_profile + residual)
        return np.clip(local_bulk, self.config.min_weight_floor, None)

    def solve(
        self,
        signature: np.ndarray,
        bulk: np.ndarray,
        cell_types: list[str],
        hierarchy: HierarchyNode | None = None,
        single_cell_expr: np.ndarray | None = None,
        single_cell_labels: list[str] | None = None,
        single_cell_subjects: list[str] | None = None,
    ) -> HierarchicalMetaSortResult:
        signature = np.asarray(signature, dtype=float)
        bulk = np.asarray(bulk, dtype=float)
        if signature.ndim != 2 or bulk.ndim != 1:
            raise ValueError("signature must be 2D and bulk must be 1D.")
        if signature.shape[0] != bulk.shape[0]:
            raise ValueError("signature and bulk must have the same gene dimension.")
        if signature.shape[1] != len(cell_types):
            raise ValueError("cell_types length must match signature columns.")
        if len(set(cell_types)) != len(cell_types):
            raise ValueError("cell_types must be unique.")

        if hierarchy is None:
            raise ValueError(
                "Manual hierarchy must be provided. Use MetaSortSolver for direct deconvolution without a hierarchy."
            )
        root = hierarchy
        hierarchy_source = "manual"
        self._validate_hierarchy(root, cell_types)
        cell_type_to_index = {cell_type: idx for idx, cell_type in enumerate(cell_types)}
        masses: dict[str, float] = {root.name: 1.0}
        stage_results: list[HierarchicalStageResult] = []

        if root.is_leaf:
            return HierarchicalMetaSortResult(
                proportions=[1.0],
                cell_types=list(cell_types),
                hierarchy=root.to_dict(),
                hierarchy_source=hierarchy_source,
                stage_results=[],
                config=asdict(self.config),
            )

        visible_nodes = list(root.children)
        stage = 0
        selected_genes, result, raw_props = self._solve_stage(
            signature=signature,
            bulk=bulk,
            nodes=visible_nodes,
            cell_type_to_index=cell_type_to_index,
        )
        initial_props = self._normalize(np.asarray([raw_props[node.name] for node in visible_nodes], dtype=float))
        masses = {
            node.name: float(prop)
            for node, prop in zip(visible_nodes, initial_props)
        }
        constrained_props = {node.name: masses[node.name] for node in visible_nodes}
        stage_results.append(
            HierarchicalStageResult(
                stage=stage,
                visible_nodes=[node.name for node in visible_nodes],
                expanded_parents=[root.name],
                selected_gene_count=int(len(selected_genes)),
                raw_proportions=raw_props,
                constrained_proportions=constrained_props,
                result=result,
                parent_results={root.name: result},
                selected_gene_count_by_parent={root.name: int(len(selected_genes))},
            )
        )
        stage += 1

        while any(not node.is_leaf for node in visible_nodes):
            expanded_nodes, expanded_parents = self._expand_visible_nodes(visible_nodes)
            next_masses: dict[str, float] = {}
            raw_props: dict[str, float] = {}
            parent_results: dict[str, MetaSortResult] = {}
            selected_gene_count_by_parent: dict[str, int] = {}
            selected_gene_union: set[int] = set()

            for node in visible_nodes:
                parent_mass = masses[node.name]
                if node.is_leaf:
                    next_masses[node.name] = parent_mass
                    raw_props[node.name] = parent_mass
                    continue
                if len(node.children) == 1:
                    child = node.children[0]
                    next_masses[child.name] = parent_mass
                    raw_props[child.name] = 1.0
                    selected_gene_count_by_parent[node.name] = 0
                    continue

                local_bulk = self._build_parent_local_bulk(
                    signature=signature,
                    bulk=bulk,
                    parent=node,
                    visible_nodes=visible_nodes,
                    masses=masses,
                    cell_type_to_index=cell_type_to_index,
                )
                selected_genes, result, parent_raw_props = self._solve_stage(
                    signature=signature,
                    bulk=local_bulk,
                    nodes=node.children,
                    cell_type_to_index=cell_type_to_index,
                )
                parent_results[node.name] = result
                selected_gene_count_by_parent[node.name] = int(len(selected_genes))
                selected_gene_union.update(int(gene_idx) for gene_idx in selected_genes)
                raw_props.update(parent_raw_props)

                child_raw = np.asarray([parent_raw_props[child.name] for child in node.children], dtype=float)
                child_local = self._normalize(child_raw)
                for child, local_prop in zip(node.children, child_local):
                    next_masses[child.name] = float(parent_mass * local_prop)

            constrained_props = {node.name: next_masses[node.name] for node in expanded_nodes}
            stage_results.append(
                HierarchicalStageResult(
                    stage=stage,
                    visible_nodes=[node.name for node in expanded_nodes],
                    expanded_parents=[node.name for node in expanded_parents],
                    selected_gene_count=int(len(selected_gene_union)),
                    raw_proportions=raw_props,
                    constrained_proportions=constrained_props,
                    result=None,
                    parent_results=parent_results,
                    selected_gene_count_by_parent=selected_gene_count_by_parent,
                )
            )
            masses = next_masses
            visible_nodes = expanded_nodes
            stage += 1

        leaf_props = np.zeros(len(cell_types), dtype=float)
        for node in visible_nodes:
            if not node.is_leaf:
                raise RuntimeError("Hierarchy expansion stopped before all leaves were reached.")
            leaf_props[cell_type_to_index[node.name]] = masses[node.name]
        leaf_props = self._normalize(leaf_props)
        return HierarchicalMetaSortResult(
            proportions=leaf_props.tolist(),
            cell_types=list(cell_types),
            hierarchy=root.to_dict(),
            hierarchy_source=hierarchy_source,
            stage_results=stage_results,
            config=asdict(self.config),
        )

    @staticmethod
    def result_to_dict(result: HierarchicalMetaSortResult) -> dict[str, Any]:
        return asdict(result)
