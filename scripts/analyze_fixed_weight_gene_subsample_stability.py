from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metasort import MetaSortConfig, MetaSortSolver
from metasort.algorithm import (
    _joint_gene_zscore,
    _normalize_columns_to_sum_one,
    _normalize_vector_to_sum_one,
    solve_simplex_constrained_ls,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MetaSort once on all genes to get final gene weights, then repeatedly "
            "sample a fraction of genes and solve weighted deconvolution using only "
            "the sampled genes and the fixed all-gene weights."
        )
    )
    parser.add_argument(
        "--tissue-root",
        type=Path,
        default=Path(
            "/home/zhiyuan/nfs/omics_remote/yunhao/nar_dataset_sim/Tabula-Sapiens/Lymph_Node"
        ),
    )
    parser.add_argument("--mixture", default="Mixture1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-repeats", type=int, default=100)
    parser.add_argument("--sample-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--signature-name", default="signature.txt")
    parser.add_argument("--bulk-name", default="bulk.txt")
    parser.add_argument("--truth-name", default="bulkRatio.txt")
    parser.add_argument("--lambda-hessian", type=float, default=0.1)
    parser.add_argument("--lambda-residual", type=float, default=0.01)
    parser.add_argument("--lambda3", type=float, default=1e-4)
    parser.add_argument("--lambda4", type=float, default=1e-5)
    parser.add_argument("--meta-weight-baseline", type=float, default=10.0)
    parser.add_argument("--meta-weight-floor", type=float, default=1.0)
    parser.add_argument(
        "--sqrt-sphere-hessian",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--normalize-meta-weight-mean",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_nonnegative(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(np.sum(values))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(values.shape, 1.0 / len(values), dtype=float)
    return values / total


def pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    diff = np.asarray(prediction, dtype=float) - np.asarray(truth, dtype=float)
    l1 = float(np.sum(np.abs(diff)))
    return {
        "L1": l1,
        "Accuracy": 1.0 - l1 / 2.0,
        "MAE": float(np.mean(np.abs(diff))),
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "Pearson": pearson_safe(truth, prediction),
    }


def preprocess_inputs(
    tissue_root: Path,
    signature_name: str,
    bulk_name: str,
    truth_name: str,
    mixture: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    signature_df = pd.read_csv(tissue_root / signature_name, sep="\t", index_col=0)
    bulk_df = pd.read_csv(tissue_root / bulk_name, sep="\t", index_col=0)
    truth_df = pd.read_csv(tissue_root / truth_name, sep="\t", index_col=0)
    if mixture not in bulk_df.columns:
        raise ValueError(f"{mixture} is missing from {bulk_name}.")
    if mixture not in truth_df.columns:
        raise ValueError(f"{mixture} is missing from {truth_name}.")

    cell_types = signature_df.columns.astype(str).to_list()
    truth_df.index = truth_df.index.astype(str)
    missing_truth = sorted(set(cell_types) - set(truth_df.index))
    if missing_truth:
        raise ValueError(f"Truth is missing cell types: {missing_truth}")

    common_genes = bulk_df.index.intersection(signature_df.index)
    if len(common_genes) == 0:
        raise ValueError("No shared genes between signature and bulk.")

    signature = _normalize_columns_to_sum_one(
        signature_df.loc[common_genes].to_numpy(dtype=float),
        name="signature",
    )
    bulk = _normalize_vector_to_sum_one(
        bulk_df.loc[common_genes, mixture].to_numpy(dtype=float),
        name="bulk",
    )
    signature, bulk = _joint_gene_zscore(signature, bulk)
    truth = normalize_nonnegative(truth_df.loc[cell_types, mixture].to_numpy(dtype=float))
    return signature, bulk, truth, common_genes.astype(str).to_list(), cell_types


def run_full_metasort_with_weights(
    solver: MetaSortSolver,
    signature: np.ndarray,
    bulk: np.ndarray,
) -> dict[str, Any]:
    cfg = solver.config
    initial_solution = solver.solve_initial(signature, bulk)
    solution = initial_solution.copy()
    meta_weights = np.full(signature.shape[0], cfg.meta_weight_baseline, dtype=float)
    changes: list[float] = []
    final_metrics: dict[str, float] = {}
    converged = False
    start = time.time()

    for _ in range(cfg.max_iter):
        new_solution, meta_step, meta_metrics = solver.solve_meta_weighted_step(
            signature=signature,
            bulk=bulk,
            current_solution=solution,
            prev_meta_weights=meta_weights,
        )
        solution_average = (
            new_solution + cfg.averaging_old_weight * solution
        ) / float(cfg.averaging_old_weight + 1)
        change = float(np.linalg.norm(solution_average - solution))
        changes.append(change)
        solution = solution_average
        meta_weights = meta_step
        final_metrics = meta_metrics
        if change <= cfg.convergence_tol:
            converged = True
            break

    return {
        "prediction": normalize_nonnegative(solution),
        "initial_prediction": normalize_nonnegative(initial_solution),
        "meta_weights": meta_weights,
        "changes": changes,
        "iterations": len(changes),
        "converged": converged,
        "runtime_seconds": time.time() - start,
        "meta_metrics": final_metrics,
    }


def summarize_numeric(values: pd.Series) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "q05": float(values.quantile(0.05)),
        "q25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q75": float(values.quantile(0.75)),
        "q95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def main() -> int:
    args = parse_args()
    if not (0.0 < args.sample_fraction <= 1.0):
        raise ValueError("--sample-fraction must be in (0, 1].")
    if args.n_repeats <= 0:
        raise ValueError("--n-repeats must be positive.")

    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists and is not empty. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    signature, bulk, truth, genes, cell_types = preprocess_inputs(
        tissue_root=args.tissue_root,
        signature_name=args.signature_name,
        bulk_name=args.bulk_name,
        truth_name=args.truth_name,
        mixture=args.mixture,
    )
    config = MetaSortConfig(
        lambda_hessian=args.lambda_hessian,
        lambda_residual=args.lambda_residual,
        lambda3=args.lambda3,
        lambda4=args.lambda4,
        meta_weight_baseline=args.meta_weight_baseline,
        meta_weight_floor=args.meta_weight_floor,
        use_sqrt_sphere_hessian=args.sqrt_sphere_hessian,
        normalize_meta_weight_mean=args.normalize_meta_weight_mean,
    )
    solver = MetaSortSolver(config)

    full = run_full_metasort_with_weights(solver, signature, bulk)
    full_prediction = full["prediction"]
    meta_weights = np.asarray(full["meta_weights"], dtype=float)
    full_metrics = compute_metrics(truth, full_prediction)

    rng = np.random.default_rng(args.seed)
    n_genes = signature.shape[0]
    sample_size = max(1, int(round(n_genes * args.sample_fraction)))
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for repeat_idx in range(args.n_repeats):
        selected = np.sort(rng.choice(n_genes, size=sample_size, replace=False))
        prediction = solve_simplex_constrained_ls(
            signature[selected, :],
            bulk[selected],
            weights=meta_weights[selected],
            initial=full_prediction,
        )
        prediction = normalize_nonnegative(prediction)
        metrics = compute_metrics(truth, prediction)
        metrics.update(
            {
                "Repeat": repeat_idx,
                "SelectedGenes": int(len(selected)),
                "WeightMin": float(np.min(meta_weights[selected])),
                "WeightMean": float(np.mean(meta_weights[selected])),
                "WeightMax": float(np.max(meta_weights[selected])),
                "PredictionL1VsFull": float(np.sum(np.abs(prediction - full_prediction))),
                "PredictionL2VsFull": float(np.linalg.norm(prediction - full_prediction)),
            }
        )
        metric_rows.append(metrics)
        for cell_type, value, truth_value, full_value in zip(
            cell_types,
            prediction,
            truth,
            full_prediction,
        ):
            prediction_rows.append(
                {
                    "Repeat": repeat_idx,
                    "CellType": cell_type,
                    "Prediction": float(value),
                    "Truth": float(truth_value),
                    "FullAllGenePrediction": float(full_value),
                    "DeltaVsTruth": float(value - truth_value),
                    "DeltaVsFull": float(value - full_value),
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    metric_summary_rows = []
    for metric_name in [
        "Accuracy",
        "L1",
        "MAE",
        "RMSE",
        "Pearson",
        "PredictionL1VsFull",
        "PredictionL2VsFull",
        "WeightMean",
    ]:
        row = {"Metric": metric_name}
        row.update(summarize_numeric(metrics_df[metric_name]))
        metric_summary_rows.append(row)
    metric_summary_df = pd.DataFrame(metric_summary_rows)

    cell_rows = []
    for cell_type, group in predictions_df.groupby("CellType", sort=False):
        pred_summary = summarize_numeric(group["Prediction"])
        cell_rows.append(
            {
                "CellType": cell_type,
                "Truth": float(group["Truth"].iloc[0]),
                "FullAllGenePrediction": float(group["FullAllGenePrediction"].iloc[0]),
                "MeanPrediction": pred_summary["mean"],
                "StdPrediction": pred_summary["std"],
                "MinPrediction": pred_summary["min"],
                "Q05Prediction": pred_summary["q05"],
                "MedianPrediction": pred_summary["median"],
                "Q95Prediction": pred_summary["q95"],
                "MaxPrediction": pred_summary["max"],
                "MeanDeltaVsFull": float(group["DeltaVsFull"].mean()),
                "AbsMeanDeltaVsFull": float(group["DeltaVsFull"].abs().mean()),
                "AbsMeanDeltaVsTruth": float(group["DeltaVsTruth"].abs().mean()),
            }
        )
    cell_stability_df = pd.DataFrame(cell_rows)

    metrics_df.to_csv(output_dir / "subsample_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "subsample_predictions.csv", index=False)
    metric_summary_df.to_csv(output_dir / "metric_summary.csv", index=False)
    cell_stability_df.to_csv(output_dir / "cell_type_stability.csv", index=False)
    pd.DataFrame(
        {
            "Gene": genes,
            "MetaWeight": meta_weights,
            "SelectedFloor": meta_weights <= args.meta_weight_floor + 1e-9,
        }
    ).to_csv(output_dir / "full_gene_meta_weights.csv", index=False)

    run_info = {
        "tissue_root": str(args.tissue_root),
        "mixture": args.mixture,
        "n_genes": int(n_genes),
        "sample_fraction": float(args.sample_fraction),
        "sample_size": int(sample_size),
        "n_repeats": int(args.n_repeats),
        "seed": int(args.seed),
        "cell_types": cell_types,
        "config": asdict(config),
        "full_all_gene": {
            "metrics": full_metrics,
            "prediction": {
                cell_type: float(value)
                for cell_type, value in zip(cell_types, full_prediction)
            },
            "initial_prediction": {
                cell_type: float(value)
                for cell_type, value in zip(cell_types, full["initial_prediction"])
            },
            "iterations": int(full["iterations"]),
            "converged": bool(full["converged"]),
            "runtime_seconds": float(full["runtime_seconds"]),
            "meta_weight_min": float(np.min(meta_weights)),
            "meta_weight_mean": float(np.mean(meta_weights)),
            "meta_weight_max": float(np.max(meta_weights)),
            "meta_weight_floor_fraction": float(
                np.mean(meta_weights <= args.meta_weight_floor + 1e-9)
            ),
            "meta_metrics": full["meta_metrics"],
        },
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))

    print(f"Wrote {output_dir}")
    print(
        "Full all-gene: "
        f"Accuracy={full_metrics['Accuracy']:.6f} "
        f"L1={full_metrics['L1']:.6f} "
        f"iterations={full['iterations']} "
        f"weight_min/mean/max={np.min(meta_weights):.4g}/"
        f"{np.mean(meta_weights):.4g}/{np.max(meta_weights):.4g}"
    )
    acc = summarize_numeric(metrics_df["Accuracy"])
    l1 = summarize_numeric(metrics_df["L1"])
    delta = summarize_numeric(metrics_df["PredictionL1VsFull"])
    print(
        "Subsample50 fixed-weight: "
        f"Accuracy mean±sd={acc['mean']:.6f}±{acc['std']:.6f}, "
        f"range={acc['min']:.6f}-{acc['max']:.6f}; "
        f"L1 mean±sd={l1['mean']:.6f}±{l1['std']:.6f}; "
        f"L1_vs_full mean±sd={delta['mean']:.6f}±{delta['std']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
