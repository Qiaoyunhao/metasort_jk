from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import nnls


@dataclass
class LocalPseudoBulkConfig:
    n_pseudobulks: int = 128
    weight_epochs: int = 100
    outer_iter: int = 5
    update_fraction: float = 0.5
    gene_sigma: float = 0.08
    cell_sigma: float = 0.08
    pseudobulk_noise_sigma: float = 0.01
    ridge: float = 1e-4
    proportion_floor: float = 1e-4
    weight_floor: float = 1e-6
    lambda_log_weight: float = 0.02
    beta_relative: float = 0.5
    leaf_loss_weight: float = 1.0
    relative_tau: float = 0.01
    relative_gamma: float = 0.5
    relative_alpha_max: float = 20.0
    adam_lr: float = 0.05
    convergence_tol: float = 0.002
    seed: int = 123


@dataclass
class HierarchyLevel:
    name: str
    groups: list[str]
    merge_matrix: np.ndarray
    weight: float


@dataclass
class HierarchySpec:
    levels: list[HierarchyLevel]


def normalize_nonnegative(values: np.ndarray, floor: float = 0.0) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), floor, None)
    total = float(np.sum(values))
    if total <= 0.0:
        return np.full(values.shape, 1.0 / len(values), dtype=float)
    return values / total


def parse_level_weights(raw_weights: str | None, n_levels: int) -> list[float]:
    if n_levels == 0:
        return []
    if raw_weights is None:
        return [1.0] * n_levels
    weights = [float(value.strip()) for value in raw_weights.split(",") if value.strip()]
    if len(weights) != n_levels:
        raise ValueError(
            f"Expected {n_levels} hierarchy level weights, got {len(weights)}."
        )
    return weights


def load_hierarchy_spec(
    path: str | Path | None,
    cell_types: list[str],
    raw_level_weights: str | None = None,
) -> HierarchySpec | None:
    if path is None:
        return None

    mapping = pd.read_csv(path, sep=None, engine="python", dtype=str)
    if mapping.empty or mapping.shape[1] < 2:
        raise ValueError("Hierarchy map must contain a cell type column and at least one level column.")

    cell_type_col = "cell_type" if "cell_type" in mapping.columns else mapping.columns[0]
    level_cols = [col for col in mapping.columns if col != cell_type_col]
    level_weights = parse_level_weights(raw_level_weights, len(level_cols))

    mapping[cell_type_col] = mapping[cell_type_col].astype(str)
    duplicated = mapping[cell_type_col][mapping[cell_type_col].duplicated()].unique()
    if len(duplicated) > 0:
        raise ValueError(f"Hierarchy map contains duplicated cell types: {', '.join(duplicated)}")

    mapping = mapping.set_index(cell_type_col)
    missing = [cell_type for cell_type in cell_types if cell_type not in mapping.index]
    if missing:
        raise ValueError(f"Hierarchy map is missing cell types: {', '.join(missing)}")

    mapping = mapping.loc[cell_types, level_cols]
    levels: list[HierarchyLevel] = []
    for level_col, level_weight in zip(level_cols, level_weights):
        if level_weight <= 0:
            continue
        labels = mapping[level_col].astype(str).to_list()
        if any(label == "" or label.lower() == "nan" for label in labels):
            raise ValueError(f"Hierarchy level {level_col} contains empty labels.")
        groups = list(dict.fromkeys(labels))
        if len(groups) <= 1 or len(groups) >= len(cell_types):
            continue

        group_to_index = {group: idx for idx, group in enumerate(groups)}
        merge_matrix = np.zeros((len(cell_types), len(groups)), dtype=float)
        for cell_type_index, group in enumerate(labels):
            merge_matrix[cell_type_index, group_to_index[group]] = 1.0
        levels.append(
            HierarchyLevel(
                name=str(level_col),
                groups=groups,
                merge_matrix=merge_matrix,
                weight=float(level_weight),
            )
        )

    if not levels:
        return None
    return HierarchySpec(levels=levels)


def aggregate_signature_by_anchor(
    signature: np.ndarray,
    anchor: np.ndarray,
    merge_matrix: np.ndarray,
    floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    signature = np.asarray(signature, dtype=float)
    anchor = normalize_nonnegative(anchor, floor=floor)
    merge_matrix = np.asarray(merge_matrix, dtype=float)
    group_anchor = anchor @ merge_matrix

    group_profiles = []
    for group_index in range(merge_matrix.shape[1]):
        member_mask = merge_matrix[:, group_index] > 0
        group_mass = float(group_anchor[group_index])
        if group_mass > floor:
            member_weights = anchor[member_mask] / group_mass
        else:
            member_weights = np.full(np.sum(member_mask), 1.0 / np.sum(member_mask), dtype=float)
        group_profiles.append(signature[:, member_mask] @ member_weights)
    aggregate_signature = np.column_stack(group_profiles)
    return aggregate_signature, normalize_nonnegative(group_anchor, floor=floor)


def solve_nnls_proportions(
    signature: np.ndarray,
    bulk: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    signature = np.asarray(signature, dtype=float)
    bulk = np.asarray(bulk, dtype=float)
    if weights is None:
        solution, _ = nnls(signature, bulk)
    else:
        sqrt_w = np.sqrt(np.clip(np.asarray(weights, dtype=float), 1e-12, None))
        solution, _ = nnls(signature * sqrt_w[:, None], bulk * sqrt_w)
    return normalize_nonnegative(solution)


def generate_reference_perturbed_pseudobulks(
    signature: np.ndarray,
    anchor: np.ndarray,
    cfg: LocalPseudoBulkConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    signature = np.asarray(signature, dtype=float)
    anchor = normalize_nonnegative(anchor, floor=cfg.proportion_floor)
    n_genes, n_cell_types = signature.shape
    gene_shift = rng.normal(0.0, cfg.gene_sigma, size=(cfg.n_pseudobulks, n_genes, 1))
    cell_shift = rng.normal(0.0, cfg.cell_sigma, size=(cfg.n_pseudobulks, n_genes, n_cell_types))
    perturbed = signature[None, :, :] * np.exp(gene_shift + cell_shift)
    pseudobulks = np.einsum("ngk,k->ng", perturbed, anchor)
    if cfg.pseudobulk_noise_sigma > 0.0:
        noise = rng.normal(0.0, cfg.pseudobulk_noise_sigma, size=pseudobulks.shape)
        pseudobulks = pseudobulks * np.exp(noise)
    return np.clip(pseudobulks, 0.0, None)


def learn_weights_from_local_pseudobulks(
    signature: np.ndarray,
    anchor: np.ndarray,
    cfg: LocalPseudoBulkConfig,
    seed: int,
    hierarchy: HierarchySpec | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)
    pseudobulks = generate_reference_perturbed_pseudobulks(signature, anchor, cfg, rng)
    anchor = normalize_nonnegative(anchor, floor=cfg.proportion_floor)

    device = torch.device("cpu")
    a_tensor = torch.tensor(np.asarray(signature, dtype=np.float32), device=device)
    y_tensor = torch.tensor(pseudobulks.T.astype(np.float32), device=device)
    anchor_tensor = torch.tensor(anchor.astype(np.float32), device=device)
    n_genes, n_cell_types = signature.shape

    log_delta = torch.zeros(n_genes, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([log_delta], lr=cfg.adam_lr)
    identity = torch.eye(n_cell_types, dtype=torch.float32, device=device)

    level_tensors = []
    if hierarchy is not None:
        for level in hierarchy.levels:
            level_signature, level_anchor = aggregate_signature_by_anchor(
                signature=signature,
                anchor=anchor,
                merge_matrix=level.merge_matrix,
                floor=cfg.proportion_floor,
            )
            level_tensors.append(
                {
                    "name": level.name,
                    "weight": level.weight,
                    "signature": torch.tensor(level_signature.astype(np.float32), device=device),
                    "anchor": torch.tensor(level_anchor.astype(np.float32), device=device),
                    "identity": torch.eye(level_signature.shape[1], dtype=torch.float32, device=device),
                }
            )

    def proportion_loss(props: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha = 1.0 / torch.pow(target + cfg.relative_tau, cfg.relative_gamma)
        alpha = torch.clamp(alpha, max=cfg.relative_alpha_max)
        alpha = alpha / torch.mean(alpha)
        hellinger = torch.mean(
            torch.sum((torch.sqrt(props + eps) - torch.sqrt(target[None, :] + eps)) ** 2, dim=1)
        )
        relative = torch.mean(torch.sum(alpha[None, :] * (props - target[None, :]) ** 2, dim=1))
        return hellinger + cfg.beta_relative * relative, hellinger, relative

    def solve_weighted_props(
        level_signature: torch.Tensor,
        level_identity: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        weighted_signature = weights[:, None] * level_signature
        hessian = level_signature.T @ weighted_signature
        ridge_scale = torch.mean(torch.diag(hessian)).detach().clamp_min(eps)
        hessian = hessian + cfg.ridge * ridge_scale * level_identity
        rhs = level_signature.T @ (weights[:, None] * y_tensor)
        raw_props = torch.linalg.solve(hessian, rhs).T
        props = torch.relu(raw_props) + eps
        return props / torch.sum(props, dim=1, keepdim=True)

    best_weights: np.ndarray | None = None
    best_loss = float("inf")
    last_metrics: dict[str, float] = {}
    eps = 1e-8

    for _ in range(cfg.weight_epochs):
        optimizer.zero_grad()
        weights = torch.exp(log_delta)
        weights = torch.clamp(weights, min=cfg.weight_floor)
        weights = weights / torch.mean(weights)

        props = solve_weighted_props(a_tensor, identity, weights)
        leaf_loss, hellinger, rel_error = proportion_loss(props, anchor_tensor)
        hierarchy_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        hierarchy_metric_values: dict[str, float] = {}
        for level_data in level_tensors:
            level_props = solve_weighted_props(
                level_signature=level_data["signature"],
                level_identity=level_data["identity"],
                weights=weights,
            )
            level_loss, level_hellinger, level_relative = proportion_loss(level_props, level_data["anchor"])
            weighted_level_loss = level_data["weight"] * level_loss
            hierarchy_loss = hierarchy_loss + weighted_level_loss
            level_key = str(level_data["name"]).replace(" ", "_")
            hierarchy_metric_values[f"hierarchy_{level_key}_loss"] = float(level_loss.detach().item())
            hierarchy_metric_values[f"hierarchy_{level_key}_hellinger_loss"] = float(level_hellinger.detach().item())
            hierarchy_metric_values[f"hierarchy_{level_key}_relative_loss"] = float(level_relative.detach().item())

        reg_loss = torch.mean(log_delta**2)
        loss = cfg.leaf_loss_weight * leaf_loss + hierarchy_loss + cfg.lambda_log_weight * reg_loss
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().item())
        if loss_value < best_loss:
            with torch.no_grad():
                current_weights = torch.exp(log_delta)
                current_weights = torch.clamp(current_weights, min=cfg.weight_floor)
                current_weights = current_weights / torch.mean(current_weights)
                best_weights = current_weights.detach().cpu().numpy().astype(float)
                best_loss = loss_value
                last_metrics = {
                    "weight_loss": loss_value,
                    "leaf_loss": float(leaf_loss.detach().item()),
                    "hellinger_loss": float(hellinger.detach().item()),
                    "relative_loss": float(rel_error.detach().item()),
                    "hierarchy_loss": float(hierarchy_loss.detach().item()),
                    "weight_reg_loss": float(reg_loss.detach().item()),
                }
                last_metrics.update(hierarchy_metric_values)

    if best_weights is None:
        best_weights = np.ones(n_genes, dtype=float)
    return best_weights, last_metrics


def solve_local_pseudobulk_adaptive(
    signature: np.ndarray,
    bulk: np.ndarray,
    cfg: LocalPseudoBulkConfig,
    hierarchy: HierarchySpec | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    proportions = solve_nnls_proportions(signature, bulk)
    final_weights = np.ones(signature.shape[0], dtype=float)
    metrics: dict[str, float] = {}
    changes: list[float] = []

    for iteration in range(cfg.outer_iter):
        anchor = normalize_nonnegative(proportions, floor=cfg.proportion_floor)
        weights, train_metrics = learn_weights_from_local_pseudobulks(
            signature=signature,
            anchor=anchor,
            cfg=cfg,
            seed=cfg.seed + iteration,
            hierarchy=hierarchy,
        )
        weighted_props = solve_nnls_proportions(signature, bulk, weights=weights)
        next_props = normalize_nonnegative(
            (1.0 - cfg.update_fraction) * proportions + cfg.update_fraction * weighted_props
        )
        change = float(np.linalg.norm(next_props - proportions))
        changes.append(change)
        proportions = next_props
        final_weights = weights
        metrics = train_metrics | {
            "iterations": float(iteration + 1),
            "last_change": change,
            "weight_min": float(np.min(weights)),
            "weight_mean": float(np.mean(weights)),
            "weight_max": float(np.max(weights)),
        }
        if change < cfg.convergence_tol:
            break

    metrics["change_history"] = ";".join(f"{value:.6g}" for value in changes)
    return proportions, final_weights, metrics


def load_tissue_data(data_root: Path, tissue: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tissue_root = data_root / tissue
    signature = pd.read_csv(tissue_root / "signature.txt", sep="\t", index_col=0)
    bulk = pd.read_csv(tissue_root / "bulk.txt", sep="\t", index_col=0)
    truth = pd.read_csv(tissue_root / "bulkRatio.txt", sep="\t", index_col=0)
    common_genes = signature.index.intersection(bulk.index)
    return signature.loc[common_genes], bulk.loc[common_genes], truth


def score_prediction(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred = normalize_nonnegative(pred)
    truth = normalize_nonnegative(truth)
    l1 = float(np.sum(np.abs(pred - truth)))
    rare_mask = truth <= 0.05
    rare_mae = float(np.mean(np.abs(pred[rare_mask] - truth[rare_mask]))) if np.any(rare_mask) else float("nan")
    return {
        "l1": l1,
        "accuracy": 1.0 - 0.5 * l1,
        "rare_mae": rare_mae,
    }


def run_demo(args: argparse.Namespace) -> None:
    cfg = LocalPseudoBulkConfig(
        n_pseudobulks=args.n_pseudobulks,
        weight_epochs=args.weight_epochs,
        outer_iter=args.outer_iter,
        update_fraction=args.update_fraction,
        gene_sigma=args.gene_sigma,
        cell_sigma=args.cell_sigma,
        pseudobulk_noise_sigma=args.pseudobulk_noise_sigma,
        lambda_log_weight=args.lambda_log_weight,
        beta_relative=args.beta_relative,
        leaf_loss_weight=args.leaf_loss_weight,
        seed=args.seed,
    )
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_rows: list[dict[str, float | str]] = []
    config_row = asdict(cfg) | {
        "hierarchy_map": "" if args.hierarchy_map is None else str(args.hierarchy_map),
        "hierarchy_level_weights": "" if args.hierarchy_level_weights is None else args.hierarchy_level_weights,
    }
    for tissue in args.tissues:
        signature_df, bulk_df, truth_df = load_tissue_data(data_root, tissue)
        mixtures = list(bulk_df.columns[: args.n_mixtures])
        common_cell_types = [cell_type for cell_type in signature_df.columns if cell_type in truth_df.index]
        signature = signature_df.loc[:, common_cell_types].to_numpy(dtype=float)
        hierarchy = load_hierarchy_spec(
            path=args.hierarchy_map,
            cell_types=common_cell_types,
            raw_level_weights=args.hierarchy_level_weights,
        )
        hierarchy_levels = 0 if hierarchy is None else len(hierarchy.levels)

        for mixture_index, mixture in enumerate(mixtures):
            bulk = bulk_df.loc[:, mixture].to_numpy(dtype=float)
            truth = truth_df.loc[common_cell_types, mixture].to_numpy(dtype=float)
            nnls_props = solve_nnls_proportions(signature, bulk)
            adaptive_props, weights, metrics = solve_local_pseudobulk_adaptive(
                signature,
                bulk,
                cfg,
                hierarchy=hierarchy,
            )
            nnls_score = score_prediction(nnls_props, truth)
            adaptive_score = score_prediction(adaptive_props, truth)
            row: dict[str, float | str] = {
                "Tissue": tissue,
                "Mixture": mixture,
                "MixtureIndex": float(mixture_index + 1),
                "HierarchyLevels": float(hierarchy_levels),
                "NNLS_L1": nnls_score["l1"],
                "NNLS_Accuracy": nnls_score["accuracy"],
                "NNLS_RareMAE": nnls_score["rare_mae"],
                "Adaptive_L1": adaptive_score["l1"],
                "Adaptive_Accuracy": adaptive_score["accuracy"],
                "Adaptive_RareMAE": adaptive_score["rare_mae"],
                "Delta_L1": adaptive_score["l1"] - nnls_score["l1"],
                "Delta_Accuracy": adaptive_score["accuracy"] - nnls_score["accuracy"],
                "Delta_RareMAE": adaptive_score["rare_mae"] - nnls_score["rare_mae"],
                "WeightMin": float(np.min(weights)),
                "WeightMean": float(np.mean(weights)),
                "WeightMax": float(np.max(weights)),
            }
            row.update(metrics)
            detail_rows.append(row)
            print(
                f"{tissue} {mixture}: "
                f"NNLS acc={nnls_score['accuracy']:.4f}, "
                f"adaptive acc={adaptive_score['accuracy']:.4f}, "
                f"delta={adaptive_score['accuracy'] - nnls_score['accuracy']:+.4f}"
            )

    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby("Tissue", as_index=False)
        .agg(
            Mixtures=("Mixture", "count"),
            NNLS_MeanL1=("NNLS_L1", "mean"),
            Adaptive_MeanL1=("Adaptive_L1", "mean"),
            Delta_MeanL1=("Delta_L1", "mean"),
            NNLS_MeanAccuracy=("NNLS_Accuracy", "mean"),
            Adaptive_MeanAccuracy=("Adaptive_Accuracy", "mean"),
            Delta_MeanAccuracy=("Delta_Accuracy", "mean"),
            NNLS_MeanRareMAE=("NNLS_RareMAE", "mean"),
            Adaptive_MeanRareMAE=("Adaptive_RareMAE", "mean"),
            Delta_MeanRareMAE=("Delta_RareMAE", "mean"),
            MeanWeightMax=("WeightMax", "mean"),
        )
    )
    detail_path = output_dir / "local_pseudobulk_weight_detail.csv"
    summary_path = output_dir / "local_pseudobulk_weight_summary.csv"
    config_path = output_dir / "local_pseudobulk_weight_config.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame([config_row]).to_csv(config_path, index=False)
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nWrote {detail_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {config_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local pseudobulk gene-weight demo.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs/local_pseudobulk_weights_first10")
    parser.add_argument("--tissues", nargs="+", default=["Blood", "Eye", "Fat", "Lung"])
    parser.add_argument("--n-mixtures", type=int, default=10)
    parser.add_argument("--n-pseudobulks", type=int, default=128)
    parser.add_argument("--weight-epochs", type=int, default=100)
    parser.add_argument("--outer-iter", type=int, default=5)
    parser.add_argument("--update-fraction", type=float, default=0.5)
    parser.add_argument("--gene-sigma", type=float, default=0.08)
    parser.add_argument("--cell-sigma", type=float, default=0.08)
    parser.add_argument("--pseudobulk-noise-sigma", type=float, default=0.01)
    parser.add_argument("--lambda-log-weight", type=float, default=0.02)
    parser.add_argument("--beta-relative", type=float, default=0.5)
    parser.add_argument("--leaf-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--hierarchy-map",
        default=None,
        help=(
            "Optional CSV/TSV with a cell_type column followed by one or more "
            "hierarchy level columns. Each level column maps leaf cell types to merged groups."
        ),
    )
    parser.add_argument(
        "--hierarchy-level-weights",
        default=None,
        help="Comma-separated weights for hierarchy level columns. Defaults to 1.0 per level.",
    )
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
