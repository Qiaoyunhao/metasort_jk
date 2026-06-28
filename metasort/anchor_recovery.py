from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import nnls

from .algorithm import solve_simplex_constrained_ls
from .metasort import MetaSortSolver


@dataclass
class AnchorRecoveryConfig:
    lambda_reference: float = 1.0
    preprocess_anchor_selection: bool = True
    use_preprocessed_anchor_proportions: bool = True
    anchor_proportion_method: str = "simplex_ls"
    anchor_residual_metric: str = "absolute"
    anchor_fit_quantile: float = 0.25
    anchor_weight_quantile: float = 0.75
    min_anchor_genes: int = 20
    max_anchor_fraction: float = 0.5
    residual_epsilon: float = 1e-12
    ridge_jitter: float = 1e-10
    normalize_nnls_proportions: bool = True
    enforce_nonnegative_signature: bool = False


@dataclass
class AnchorRecoveryResult:
    proportions: list[list[float]]
    initial_proportions: list[list[float]]
    recovered_signature: list[list[float]]
    batch_shift: list[list[float]]
    anchor_mask: list[bool]
    anchor_genes: list[str]
    recovered_genes: list[str]
    cell_types: list[str]
    genes: list[str]
    gene_metrics: dict[str, list[float]]
    reconstruction_error: float
    initial_reconstruction_error: float
    config: dict[str, float | int | bool]


class AnchorRecoverySolver:
    def __init__(
        self,
        config: AnchorRecoveryConfig | None = None,
        initial_solver: MetaSortSolver | None = None,
    ) -> None:
        self.config = config or AnchorRecoveryConfig()
        self.initial_solver = initial_solver or MetaSortSolver()

    @staticmethod
    def _as_bulk_matrix(bulk: np.ndarray) -> np.ndarray:
        bulk = np.asarray(bulk, dtype=float)
        if bulk.ndim == 1:
            return bulk[:, None]
        if bulk.ndim == 2:
            return bulk
        raise ValueError("bulk must be 1D or 2D.")

    @staticmethod
    def _normalize_nonnegative(vector: np.ndarray) -> np.ndarray:
        vector = np.clip(np.asarray(vector, dtype=float), 0.0, None)
        total = float(np.sum(vector))
        if total <= 0.0:
            return np.full(vector.shape[0], 1.0 / vector.shape[0], dtype=float)
        return vector / total

    @staticmethod
    def _validate_quantile(value: float, name: str) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")

    @staticmethod
    def _normalize_columns_to_sum_one(matrix: np.ndarray, name: str) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        column_sums = np.sum(matrix, axis=0)
        if np.any(~np.isfinite(column_sums)) or np.any(column_sums <= 0.0):
            raise ValueError(f"{name} columns must have positive finite sums for normalization.")
        return matrix / column_sums[None, :]

    @staticmethod
    def _joint_gene_zscore_matrices(
        signature: np.ndarray,
        bulk_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        combined = np.column_stack(
            [
                np.asarray(signature, dtype=float),
                np.asarray(bulk_matrix, dtype=float),
            ]
        )
        row_means = np.mean(combined, axis=1, keepdims=True)
        row_stds = np.std(combined, axis=1, keepdims=True)
        row_stds = np.where(row_stds > 0.0, row_stds, 1.0)
        zscored = (combined - row_means) / row_stds
        n_cell_types = signature.shape[1]
        return zscored[:, :n_cell_types], zscored[:, n_cell_types:]

    def _anchor_selection_inputs(
        self,
        signature: np.ndarray,
        bulk_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.config.preprocess_anchor_selection:
            return signature, bulk_matrix

        normalized_signature = self._normalize_columns_to_sum_one(signature, "signature")
        normalized_bulk = self._normalize_columns_to_sum_one(bulk_matrix, "bulk")
        return self._joint_gene_zscore_matrices(normalized_signature, normalized_bulk)

    def _initial_metasort(
        self,
        signature: np.ndarray,
        bulk_matrix: np.ndarray,
        cell_types: list[str] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_cell_types = signature.shape[1]
        n_samples = bulk_matrix.shape[1]
        proportions = np.zeros((n_cell_types, n_samples), dtype=float)
        meta_weights = np.ones((signature.shape[0], n_samples), dtype=float)

        for sample_idx in range(n_samples):
            result = self.initial_solver.solve(
                signature,
                bulk_matrix[:, sample_idx],
                cell_types=cell_types,
            )
            proportions[:, sample_idx] = np.asarray(result.proportions, dtype=float)
            if result.meta_weights is not None:
                weights = np.asarray(result.meta_weights, dtype=float)
                if weights.shape[0] != signature.shape[0]:
                    raise ValueError("MetaSort returned meta_weights with the wrong gene dimension.")
                meta_weights[:, sample_idx] = weights

        return proportions, meta_weights

    def _select_anchor_genes(
        self,
        signature: np.ndarray,
        bulk_matrix: np.ndarray,
        initial_proportions: np.ndarray,
        meta_weights: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cfg = self.config
        fitted = signature @ initial_proportions
        residual = bulk_matrix - fitted
        residual_rms = np.sqrt(np.mean(residual ** 2, axis=1))
        scale = np.sqrt(np.mean(bulk_matrix ** 2, axis=1)) + np.sqrt(np.mean(fitted ** 2, axis=1))
        relative_residual = residual_rms / np.clip(scale, cfg.residual_epsilon, None)
        if cfg.anchor_residual_metric == "absolute":
            anchor_residual = residual_rms
        elif cfg.anchor_residual_metric == "relative":
            anchor_residual = relative_residual
        else:
            raise ValueError("anchor_residual_metric must be 'absolute' or 'relative'.")
        mean_meta_weight = np.mean(meta_weights, axis=1)

        weight_min = float(np.min(mean_meta_weight))
        weight_span = float(np.max(mean_meta_weight) - weight_min)
        normalized_weight = (
            np.ones_like(mean_meta_weight)
            if weight_span <= cfg.residual_epsilon
            else (mean_meta_weight - weight_min) / weight_span
        )
        anchor_score = normalized_weight / (anchor_residual + cfg.residual_epsilon)

        fit_threshold = float(np.quantile(anchor_residual, cfg.anchor_fit_quantile))
        weight_threshold = float(np.quantile(mean_meta_weight, cfg.anchor_weight_quantile))
        anchor_mask = (anchor_residual <= fit_threshold) & (mean_meta_weight >= weight_threshold)

        n_genes = signature.shape[0]
        min_anchor_genes = min(max(1, int(cfg.min_anchor_genes)), n_genes)
        max_anchor_genes = max(min_anchor_genes, int(np.ceil(cfg.max_anchor_fraction * n_genes)))
        max_anchor_genes = min(max_anchor_genes, n_genes)

        if int(np.sum(anchor_mask)) < min_anchor_genes:
            selected = np.argsort(anchor_score)[::-1][:min_anchor_genes]
            anchor_mask = np.zeros(n_genes, dtype=bool)
            anchor_mask[selected] = True
        elif int(np.sum(anchor_mask)) > max_anchor_genes:
            candidates = np.flatnonzero(anchor_mask)
            ranked = candidates[np.argsort(anchor_score[candidates])[::-1]]
            anchor_mask = np.zeros(n_genes, dtype=bool)
            anchor_mask[ranked[:max_anchor_genes]] = True

        metrics = {
            "anchor_residual": anchor_residual,
            "relative_residual": relative_residual,
            "mean_meta_weight": mean_meta_weight,
            "anchor_score": anchor_score,
            "initial_fitted_mean": np.mean(fitted, axis=1),
            "initial_residual_rms": residual_rms,
        }
        return anchor_mask, metrics

    def _solve_anchor_proportions(
        self,
        signature: np.ndarray,
        bulk_matrix: np.ndarray,
        anchor_mask: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        anchor_signature = signature[anchor_mask, :]
        anchor_bulk = bulk_matrix[anchor_mask, :]
        n_cell_types = signature.shape[1]
        n_samples = bulk_matrix.shape[1]
        proportions = np.zeros((n_cell_types, n_samples), dtype=float)

        for sample_idx in range(n_samples):
            if cfg.anchor_proportion_method == "simplex_ls":
                solution = solve_simplex_constrained_ls(
                    anchor_signature,
                    anchor_bulk[:, sample_idx],
                )
            elif cfg.anchor_proportion_method == "nnls":
                solution, _ = nnls(anchor_signature, anchor_bulk[:, sample_idx])
            else:
                raise ValueError("anchor_proportion_method must be 'simplex_ls' or 'nnls'.")
            if cfg.anchor_proportion_method == "nnls" and cfg.normalize_nnls_proportions:
                solution = self._normalize_nonnegative(solution)
            proportions[:, sample_idx] = solution

        return proportions

    def _recover_non_anchor_signature(
        self,
        signature: np.ndarray,
        bulk_matrix: np.ndarray,
        proportions: np.ndarray,
        anchor_mask: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        if cfg.lambda_reference <= 0.0:
            raise ValueError("lambda_reference must be positive.")
        recovered = np.asarray(signature, dtype=float).copy()
        non_anchor_mask = ~anchor_mask
        if not np.any(non_anchor_mask):
            return recovered

        n_cell_types = signature.shape[1]
        gram = proportions @ proportions.T
        regularized = gram + (cfg.lambda_reference + cfg.ridge_jitter) * np.eye(n_cell_types)
        rhs = bulk_matrix[non_anchor_mask, :] @ proportions.T
        rhs = rhs + cfg.lambda_reference * signature[non_anchor_mask, :]
        recovered_non_anchor = rhs @ np.linalg.pinv(regularized)

        if cfg.enforce_nonnegative_signature:
            recovered_non_anchor = np.clip(recovered_non_anchor, 0.0, None)

        recovered[non_anchor_mask, :] = recovered_non_anchor
        return recovered

    def solve(
        self,
        signature: np.ndarray,
        bulk: np.ndarray,
        cell_types: list[str] | None = None,
        genes: list[str] | None = None,
    ) -> AnchorRecoveryResult:
        cfg = self.config
        self._validate_quantile(cfg.anchor_fit_quantile, "anchor_fit_quantile")
        self._validate_quantile(cfg.anchor_weight_quantile, "anchor_weight_quantile")
        if cfg.max_anchor_fraction <= 0.0 or cfg.max_anchor_fraction > 1.0:
            raise ValueError("max_anchor_fraction must be in (0, 1].")

        signature = np.asarray(signature, dtype=float)
        bulk_matrix = self._as_bulk_matrix(bulk)
        if signature.ndim != 2:
            raise ValueError("signature must be 2D.")
        if signature.shape[0] != bulk_matrix.shape[0]:
            raise ValueError("signature and bulk must have the same gene dimension.")
        if signature.shape[1] == 0:
            raise ValueError("signature must have at least one cell type.")
        if cell_types is not None and len(cell_types) != signature.shape[1]:
            raise ValueError("cell_types length must match signature columns.")
        if genes is not None and len(genes) != signature.shape[0]:
            raise ValueError("genes length must match signature rows.")
        if np.any(~np.isfinite(signature)) or np.any(~np.isfinite(bulk_matrix)):
            raise ValueError("signature and bulk must be finite.")

        resolved_cell_types = (
            [f"cell_type_{idx}" for idx in range(signature.shape[1])]
            if cell_types is None
            else list(cell_types)
        )
        resolved_genes = (
            [f"gene_{idx}" for idx in range(signature.shape[0])]
            if genes is None
            else list(genes)
        )
        selection_signature, selection_bulk_matrix = self._anchor_selection_inputs(
            signature=signature,
            bulk_matrix=bulk_matrix,
        )

        initial_proportions, meta_weights = self._initial_metasort(
            signature=selection_signature,
            bulk_matrix=selection_bulk_matrix,
            cell_types=resolved_cell_types,
        )
        anchor_mask, gene_metrics_np = self._select_anchor_genes(
            signature=selection_signature,
            bulk_matrix=selection_bulk_matrix,
            initial_proportions=initial_proportions,
            meta_weights=meta_weights,
        )
        proportion_signature = (
            selection_signature if cfg.use_preprocessed_anchor_proportions else signature
        )
        proportion_bulk_matrix = (
            selection_bulk_matrix if cfg.use_preprocessed_anchor_proportions else bulk_matrix
        )
        proportions = self._solve_anchor_proportions(
            signature=proportion_signature,
            bulk_matrix=proportion_bulk_matrix,
            anchor_mask=anchor_mask,
        )
        recovered_signature = self._recover_non_anchor_signature(
            signature=signature,
            bulk_matrix=bulk_matrix,
            proportions=proportions,
            anchor_mask=anchor_mask,
        )
        batch_shift = recovered_signature - signature
        initial_residual = bulk_matrix - signature @ initial_proportions
        final_residual = bulk_matrix - recovered_signature @ proportions

        anchor_genes = [gene for gene, is_anchor in zip(resolved_genes, anchor_mask) if is_anchor]
        recovered_genes = [gene for gene, is_anchor in zip(resolved_genes, anchor_mask) if not is_anchor]
        gene_metrics = {name: values.tolist() for name, values in gene_metrics_np.items()}

        return AnchorRecoveryResult(
            proportions=proportions.tolist(),
            initial_proportions=initial_proportions.tolist(),
            recovered_signature=recovered_signature.tolist(),
            batch_shift=batch_shift.tolist(),
            anchor_mask=anchor_mask.tolist(),
            anchor_genes=anchor_genes,
            recovered_genes=recovered_genes,
            cell_types=resolved_cell_types,
            genes=resolved_genes,
            gene_metrics=gene_metrics,
            reconstruction_error=float(np.linalg.norm(final_residual)),
            initial_reconstruction_error=float(np.linalg.norm(initial_residual)),
            config=asdict(cfg),
        )

    @staticmethod
    def result_to_dict(result: AnchorRecoveryResult) -> dict[str, Any]:
        return asdict(result)
