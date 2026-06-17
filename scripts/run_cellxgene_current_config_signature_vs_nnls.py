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
from scipy.optimize import nnls

from metasort import MetaSortConfig, MetaSortSolver
from metasort.algorithm import (
    _joint_gene_zscore,
    _normalize_columns_to_sum_one,
    _normalize_vector_to_sum_one,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run current MetaSort config and pure NNLS on the first N CELlxGENE "
            "mixtures for every tissue using signature.txt."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/zhiyuan/nfs/omics_remote/yunhao/nar_dataset_sim/CELlxGENE"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cellxgene_current_config_signature_first10_vs_nnls"),
    )
    parser.add_argument("--n-mixtures", type=int, default=10)
    parser.add_argument("--signature-name", default="signature.txt")
    parser.add_argument("--bulk-name", default="bulk.txt")
    parser.add_argument("--truth-name", default="bulkRatio.txt")
    parser.add_argument(
        "--normalize-meta-weight-mean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize optimized meta weights to mean 1 after each LBFGS round.",
    )
    parser.add_argument("--lambda-hessian", type=float, default=1.0)
    parser.add_argument("--lambda3", type=float, default=0.01)
    parser.add_argument("--lambda4", type=float, default=0.001)
    parser.add_argument(
        "--tissues",
        nargs="+",
        default=None,
        help="Optional tissue names to run. Defaults to every tissue under data-root.",
    )
    parser.add_argument(
        "--mixtures",
        nargs="+",
        default=None,
        help="Optional mixture names to run within each selected tissue. Defaults to the first n mixtures.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_tissues(data_root: Path) -> list[Path]:
    tissues = []
    for tissue_root in sorted(data_root.iterdir()):
        if not tissue_root.is_dir():
            continue
        if (
            (tissue_root / "signature.txt").exists()
            and (tissue_root / "bulk.txt").exists()
            and (tissue_root / "bulkRatio.txt").exists()
        ):
            tissues.append(tissue_root)
    return tissues


def preprocess_inputs(
    signature_df: pd.DataFrame,
    bulk_df: pd.DataFrame,
    mixture_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    common_genes = bulk_df.index.intersection(signature_df.index)
    if len(common_genes) == 0:
        raise ValueError("No shared genes between signature and bulk.")

    signature = _normalize_columns_to_sum_one(
        signature_df.loc[common_genes].to_numpy(dtype=float),
        name="signature",
    )
    bulk = _normalize_vector_to_sum_one(
        bulk_df.loc[common_genes, mixture_name].to_numpy(dtype=float),
        name="bulk",
    )
    signature, bulk = _joint_gene_zscore(signature, bulk)
    return signature, bulk, common_genes.to_list()


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
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    diff = prediction - truth
    l1 = float(np.sum(np.abs(diff)))
    rare_mask = (truth > 0.0) & (truth < 0.05)
    return {
        "L1": l1,
        "Accuracy": 1.0 - l1 / 2.0,
        "MAE": float(np.mean(np.abs(diff))),
        "RMSE": float(np.sqrt(np.mean(diff**2))),
        "Pearson": pearson_safe(truth, prediction),
        "RareMAE": float(np.mean(np.abs(diff[rare_mask]))) if np.any(rare_mask) else float("nan"),
    }


def summarize_detail(detail_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = detail_df.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        N=("Mixture", "count"),
        MeanL1=("L1", "mean"),
        MedianL1=("L1", "median"),
        MeanAccuracy=("Accuracy", "mean"),
        MedianAccuracy=("Accuracy", "median"),
        MeanMAE=("MAE", "mean"),
        MeanRMSE=("RMSE", "mean"),
        MeanPearson=("Pearson", "mean"),
        MeanRareMAE=("RareMAE", "mean"),
        MeanRuntimeSec=("RuntimeSec", "mean"),
        Errors=("Error", lambda values: int(values.notna().sum())),
    ).reset_index()

    if "Converged" in detail_df.columns:
        converged = grouped["Converged"].sum(min_count=1).reset_index(name="Converged")
        iterations = grouped["Iterations"].mean().reset_index(name="MeanIterations")
        summary = summary.merge(converged, on=group_cols, how="left")
        summary = summary.merge(iterations, on=group_cols, how="left")
    return summary


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    if args.n_mixtures <= 0:
        raise ValueError("--n-mixtures must be positive.")

    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists and is not empty. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = MetaSortConfig(
        lambda_hessian=args.lambda_hessian,
        lambda3=args.lambda3,
        lambda4=args.lambda4,
        normalize_meta_weight_mean=args.normalize_meta_weight_mean,
    )
    solver = MetaSortSolver(config)
    tissues = discover_tissues(args.data_root)
    if args.tissues is not None:
        selected = set(args.tissues)
        tissues = [tissue_root for tissue_root in tissues if tissue_root.name in selected]
        missing = sorted(selected - {tissue_root.name for tissue_root in tissues})
        if missing:
            raise ValueError(f"Requested tissues were not found or are missing inputs: {missing}")
    if not tissues:
        raise ValueError(f"No tissues found under {args.data_root}.")

    detail_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    run_start = time.time()

    for tissue_index, tissue_root in enumerate(tissues, start=1):
        tissue_name = tissue_root.name
        signature_df = pd.read_csv(tissue_root / args.signature_name, sep="\t", index_col=0)
        bulk_df = pd.read_csv(tissue_root / args.bulk_name, sep="\t", index_col=0)
        truth_df = pd.read_csv(tissue_root / args.truth_name, sep="\t", index_col=0)

        cell_types = signature_df.columns.astype(str).to_list()
        missing_truth = sorted(set(cell_types) - set(truth_df.index.astype(str)))
        if missing_truth:
            raise ValueError(f"{tissue_name}: truth is missing cell types: {missing_truth}")
        truth_df.index = truth_df.index.astype(str)
        if args.mixtures is None:
            mixture_names = bulk_df.columns.astype(str).to_list()[: args.n_mixtures]
        else:
            requested_mixtures = set(args.mixtures)
            mixture_names = [
                mixture_name
                for mixture_name in bulk_df.columns.astype(str).to_list()
                if mixture_name in requested_mixtures
            ]
            missing_mixtures = sorted(requested_mixtures - set(mixture_names))
            if missing_mixtures:
                raise ValueError(f"{tissue_name}: requested mixtures were not found: {missing_mixtures}")

        print(
            f"[{tissue_index}/{len(tissues)}] {tissue_name}: "
            f"{len(cell_types)} cell types, {len(mixture_names)} mixtures",
            flush=True,
        )

        for mixture_index, mixture_name in enumerate(mixture_names, start=1):
            if mixture_name not in truth_df.columns:
                raise ValueError(f"{tissue_name}: {mixture_name} is missing from {args.truth_name}.")
            truth = truth_df.loc[cell_types, mixture_name].to_numpy(dtype=float)
            truth = normalize_nonnegative(truth)
            signature, bulk, genes = preprocess_inputs(signature_df, bulk_df, mixture_name)

            methods: list[tuple[str, np.ndarray, dict[str, Any], float, str | None]] = []

            start = time.time()
            metasort_error = None
            metasort_extra: dict[str, Any] = {}
            try:
                result = solver.solve(signature, bulk, cell_types=cell_types)
                metasort_prediction = normalize_nonnegative(np.asarray(result.proportions, dtype=float))
                metasort_extra = {
                    "Converged": bool(result.converged),
                    "Iterations": int(result.iterations),
                    "SelectedJ": int(result.selected_j),
                    "SelectedMultiplier": float(result.selected_multiplier),
                    "MetaWeightMin": float(result.meta_weight_min),
                    "MetaWeightMean": float(result.meta_weight_mean),
                    "MetaWeightMax": float(result.meta_weight_max),
                    "TotalWeightMin": float(result.total_weight_min),
                    "TotalWeightMean": float(result.total_weight_mean),
                    "TotalWeightMax": float(result.total_weight_max),
                }
            except Exception as exc:
                metasort_prediction = np.full(len(cell_types), np.nan, dtype=float)
                metasort_error = repr(exc)
                metasort_extra = {
                    "Converged": False,
                    "Iterations": np.nan,
                    "SelectedJ": np.nan,
                    "SelectedMultiplier": np.nan,
                    "MetaWeightMin": np.nan,
                    "MetaWeightMean": np.nan,
                    "MetaWeightMax": np.nan,
                    "TotalWeightMin": np.nan,
                    "TotalWeightMean": np.nan,
                    "TotalWeightMax": np.nan,
                }
            runtime = time.time() - start
            methods.append(("MetaSortCurrent", metasort_prediction, metasort_extra, runtime, metasort_error))

            start = time.time()
            nnls_error = None
            try:
                nnls_coef, nnls_residual = nnls(signature, bulk)
                nnls_prediction = normalize_nonnegative(nnls_coef)
                nnls_extra = {"NNLSResidual": float(nnls_residual)}
            except Exception as exc:
                nnls_prediction = np.full(len(cell_types), np.nan, dtype=float)
                nnls_error = repr(exc)
                nnls_extra = {"NNLSResidual": np.nan}
            runtime = time.time() - start
            methods.append(("NNLS", nnls_prediction, nnls_extra, runtime, nnls_error))

            for method_name, prediction, extra, runtime, error in methods:
                if np.any(~np.isfinite(prediction)):
                    metrics = {
                        "L1": np.nan,
                        "Accuracy": np.nan,
                        "MAE": np.nan,
                        "RMSE": np.nan,
                        "Pearson": np.nan,
                        "RareMAE": np.nan,
                    }
                else:
                    metrics = compute_metrics(truth, prediction)
                    for cell_type, truth_value, predicted_value in zip(cell_types, truth, prediction):
                        prediction_rows.append(
                            {
                                "Tissue": tissue_name,
                                "Mixture": mixture_name,
                                "Method": method_name,
                                "CellType": cell_type,
                                "Truth": float(truth_value),
                                "Prediction": float(predicted_value),
                                "AbsError": float(abs(predicted_value - truth_value)),
                                "SignedError": float(predicted_value - truth_value),
                            }
                        )

                detail_rows.append(
                    {
                        "Tissue": tissue_name,
                        "Mixture": mixture_name,
                        "Method": method_name,
                        "NGenes": int(len(genes)),
                        "NCellTypes": int(len(cell_types)),
                        "RuntimeSec": float(runtime),
                        "Error": error,
                        **metrics,
                        **extra,
                    }
                )

            print(
                f"  {mixture_index:02d}/{len(mixture_names)} {mixture_name} done",
                flush=True,
            )

    detail_df = pd.DataFrame(detail_rows)
    prediction_df = pd.DataFrame(prediction_rows)

    by_tissue = summarize_detail(detail_df, ["Method", "Tissue"])
    overall = summarize_detail(detail_df, ["Method"])

    wide_tissue = by_tissue.pivot(index="Tissue", columns="Method")
    comparison_rows = []
    for tissue_name in sorted(detail_df["Tissue"].unique()):
        row = {"Tissue": tissue_name}
        for metric in ["MeanAccuracy", "MeanL1", "MeanMAE", "MeanRMSE", "MeanPearson", "MeanRuntimeSec"]:
            row[f"{metric}_MetaSortCurrent"] = float(wide_tissue.loc[tissue_name, (metric, "MetaSortCurrent")])
            row[f"{metric}_NNLS"] = float(wide_tissue.loc[tissue_name, (metric, "NNLS")])
        row["DeltaAccuracy_MetaSortMinusNNLS"] = (
            row["MeanAccuracy_MetaSortCurrent"] - row["MeanAccuracy_NNLS"]
        )
        row["DeltaL1_MetaSortMinusNNLS"] = row["MeanL1_MetaSortCurrent"] - row["MeanL1_NNLS"]
        comparison_rows.append(row)
    comparison_by_tissue = pd.DataFrame(comparison_rows)

    overall_wide = overall.set_index("Method")
    comparison_overall = pd.DataFrame(
        [
            {
                "N": int(overall_wide.loc["MetaSortCurrent", "N"]),
                "MeanAccuracy_MetaSortCurrent": float(overall_wide.loc["MetaSortCurrent", "MeanAccuracy"]),
                "MeanAccuracy_NNLS": float(overall_wide.loc["NNLS", "MeanAccuracy"]),
                "DeltaAccuracy_MetaSortMinusNNLS": float(
                    overall_wide.loc["MetaSortCurrent", "MeanAccuracy"]
                    - overall_wide.loc["NNLS", "MeanAccuracy"]
                ),
                "MeanL1_MetaSortCurrent": float(overall_wide.loc["MetaSortCurrent", "MeanL1"]),
                "MeanL1_NNLS": float(overall_wide.loc["NNLS", "MeanL1"]),
                "DeltaL1_MetaSortMinusNNLS": float(
                    overall_wide.loc["MetaSortCurrent", "MeanL1"]
                    - overall_wide.loc["NNLS", "MeanL1"]
                ),
                "MeanRuntimeSec_MetaSortCurrent": float(overall_wide.loc["MetaSortCurrent", "MeanRuntimeSec"]),
                "MeanRuntimeSec_NNLS": float(overall_wide.loc["NNLS", "MeanRuntimeSec"]),
            }
        ]
    )

    write_csv(detail_df, output_dir / "detail.csv")
    write_csv(prediction_df, output_dir / "predictions_long.csv")
    write_csv(by_tissue, output_dir / "summary_by_tissue.csv")
    write_csv(overall, output_dir / "summary_overall.csv")
    write_csv(comparison_by_tissue, output_dir / "comparison_by_tissue.csv")
    write_csv(comparison_overall, output_dir / "comparison_overall.csv")

    config_payload = {
        **asdict(config),
        "data_root": str(args.data_root),
        "n_mixtures_per_tissue": args.n_mixtures,
        "signature_name": args.signature_name,
        "bulk_name": args.bulk_name,
        "truth_name": args.truth_name,
        "method_metasort": "MetaSortSolver(MetaSortConfig())",
        "method_nnls": "scipy.optimize.nnls on the same preprocessed signature and bulk, normalized to proportions",
        "preprocessing": (
            "shared_genes -> signature column sum-to-one, bulk vector sum-to-one -> "
            "joint gene-row z-score of signature plus selected bulk"
        ),
        "elapsed_sec": time.time() - run_start,
        "python": sys.executable,
    }
    pd.DataFrame([config_payload]).to_csv(output_dir / "config.csv", index=False)
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True) + "\n")

    print(f"Wrote outputs to {output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
