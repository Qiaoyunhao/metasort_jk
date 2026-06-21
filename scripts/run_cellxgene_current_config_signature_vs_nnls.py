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

from metasort import (
    HierarchicalMetaSortConfig,
    HierarchicalMetaSortSolver,
    HierarchyNode,
    MetaSortConfig,
    MetaSortSolver,
)
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
    parser.add_argument(
        "--hierarchy-file",
        type=Path,
        default=None,
        help=(
            "Optional manual hierarchy JSON file. If provided, the script uses "
            "HierarchicalMetaSortSolver; otherwise it uses plain MetaSortSolver."
        ),
    )
    parser.add_argument("--lambda-hessian", type=float, default=1.0)
    parser.add_argument("--lambda-residual", type=float, default=0.005)
    parser.add_argument("--convergence-tol", type=float, default=0.02)
    parser.add_argument(
        "--meta-weight-baseline",
        type=float,
        default=1.0,
        help=(
            "Initial gene meta weight and regularization center. "
            "Use 10 to keep the optimized weights centered around 10."
        ),
    )
    parser.add_argument("--meta-weight-floor", type=float, default=1e-2)
    parser.add_argument(
        "--sqrt-sphere-hessian",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compute the Hessian spectrum after mapping proportions with sqrt(p) "
            "onto the sphere and projecting onto the current sphere tangent space."
        ),
    )
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


def hierarchy_node_from_dict(data: Any) -> HierarchyNode:
    if not isinstance(data, dict):
        raise ValueError("Hierarchy JSON nodes must be objects.")

    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Every hierarchy node must have a non-empty string name.")

    children_data = data.get("children", [])
    if children_data is None:
        children_data = []
    if not isinstance(children_data, list):
        raise ValueError(f"Hierarchy node {name} has non-list children.")

    children = [hierarchy_node_from_dict(child_data) for child_data in children_data]
    if children:
        inferred_cell_types = [
            cell_type
            for child in children
            for cell_type in child.cell_types
        ]
        provided_cell_types = data.get("cell_types")
        if provided_cell_types is not None:
            if not isinstance(provided_cell_types, list) or not all(
                isinstance(value, str)
                for value in provided_cell_types
            ):
                raise ValueError(f"Hierarchy node {name} has invalid cell_types.")
            if (
                set(provided_cell_types) != set(inferred_cell_types)
                or len(provided_cell_types) != len(inferred_cell_types)
            ):
                raise ValueError(f"Hierarchy node {name} cell_types do not match its children.")
        cell_types = inferred_cell_types
    else:
        provided_cell_types = data.get("cell_types")
        if provided_cell_types is None:
            cell_types = [name]
        elif (
            isinstance(provided_cell_types, list)
            and len(provided_cell_types) == 1
            and provided_cell_types[0] == name
        ):
            cell_types = [name]
        else:
            raise ValueError(
                f"Leaf hierarchy node {name} must omit cell_types or set it to [{name!r}]."
            )

    distance = data.get("distance", 0.0)
    return HierarchyNode(
        name=name,
        cell_types=cell_types,
        children=children,
        distance=float(distance),
    )


def load_hierarchy_file(path: Path) -> HierarchyNode:
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise ValueError(f"Could not read hierarchy file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Hierarchy file {path} is not valid JSON: {exc}") from exc
    return hierarchy_node_from_dict(payload)


def base_metasort_extra(result: Any) -> dict[str, Any]:
    return {
        "Converged": bool(result.converged),
        "Iterations": int(result.iterations),
        "MetaWeightMin": float(result.meta_weight_min),
        "MetaWeightMean": float(result.meta_weight_mean),
        "MetaWeightMax": float(result.meta_weight_max),
        "TotalWeightMin": float(result.total_weight_min),
        "TotalWeightMean": float(result.total_weight_mean),
        "TotalWeightMax": float(result.total_weight_max),
        "HierarchyEnabled": False,
        "HierarchySource": "",
        "HierarchyStages": 0,
    }


def empty_metasort_extra(hierarchical: bool) -> dict[str, Any]:
    return {
        "Converged": False,
        "Iterations": np.nan,
        "MetaWeightMin": np.nan,
        "MetaWeightMean": np.nan,
        "MetaWeightMax": np.nan,
        "TotalWeightMin": np.nan,
        "TotalWeightMean": np.nan,
        "TotalWeightMax": np.nan,
        "HierarchyEnabled": bool(hierarchical),
        "HierarchySource": "",
        "HierarchyStages": np.nan,
    }


def hierarchical_metasort_extra(result: Any) -> dict[str, Any]:
    base_results = []
    for stage_result in result.stage_results:
        if stage_result.result is not None:
            base_results.append(stage_result.result)
        else:
            base_results.extend(stage_result.parent_results.values())

    converged = all(base_result.converged for base_result in base_results) if base_results else True
    iterations = sum(int(base_result.iterations) for base_result in base_results)
    hierarchy_source = "manual" if result.hierarchy_source == "provided" else result.hierarchy_source
    return {
        "Converged": bool(converged),
        "Iterations": int(iterations),
        "MetaWeightMin": np.nan,
        "MetaWeightMean": np.nan,
        "MetaWeightMax": np.nan,
        "TotalWeightMin": np.nan,
        "TotalWeightMean": np.nan,
        "TotalWeightMax": np.nan,
        "HierarchyEnabled": True,
        "HierarchySource": hierarchy_source,
        "HierarchyStages": int(len(result.stage_results)),
    }


def main() -> int:
    args = parse_args()
    if args.n_mixtures <= 0:
        raise ValueError("--n-mixtures must be positive.")

    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists and is not empty. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    hierarchy = load_hierarchy_file(args.hierarchy_file) if args.hierarchy_file is not None else None
    use_hierarchy = hierarchy is not None
    config_class = HierarchicalMetaSortConfig if use_hierarchy else MetaSortConfig
    config = config_class(
        convergence_tol=args.convergence_tol,
        lambda_hessian=args.lambda_hessian,
        lambda_residual=args.lambda_residual,
        use_sqrt_sphere_hessian=args.sqrt_sphere_hessian,
        lambda3=args.lambda3,
        lambda4=args.lambda4,
        meta_weight_baseline=args.meta_weight_baseline,
        meta_weight_floor=args.meta_weight_floor,
        normalize_meta_weight_mean=args.normalize_meta_weight_mean,
    )
    solver = HierarchicalMetaSortSolver(config) if use_hierarchy else MetaSortSolver(config)
    primary_method_name = "HierarchicalMetaSortCurrent" if use_hierarchy else "MetaSortCurrent"
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
                if use_hierarchy:
                    result = solver.solve(
                        signature,
                        bulk,
                        cell_types=cell_types,
                        hierarchy=hierarchy,
                    )
                    metasort_extra = hierarchical_metasort_extra(result)
                else:
                    result = solver.solve(signature, bulk, cell_types=cell_types)
                    metasort_extra = base_metasort_extra(result)
                metasort_prediction = normalize_nonnegative(np.asarray(result.proportions, dtype=float))
            except Exception as exc:
                metasort_prediction = np.full(len(cell_types), np.nan, dtype=float)
                metasort_error = repr(exc)
                metasort_extra = empty_metasort_extra(use_hierarchy)
            runtime = time.time() - start
            methods.append((primary_method_name, metasort_prediction, metasort_extra, runtime, metasort_error))

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
            row[f"{metric}_{primary_method_name}"] = float(wide_tissue.loc[tissue_name, (metric, primary_method_name)])
            row[f"{metric}_NNLS"] = float(wide_tissue.loc[tissue_name, (metric, "NNLS")])
        row[f"DeltaAccuracy_{primary_method_name}MinusNNLS"] = (
            row[f"MeanAccuracy_{primary_method_name}"] - row["MeanAccuracy_NNLS"]
        )
        row[f"DeltaL1_{primary_method_name}MinusNNLS"] = row[f"MeanL1_{primary_method_name}"] - row["MeanL1_NNLS"]
        comparison_rows.append(row)
    comparison_by_tissue = pd.DataFrame(comparison_rows)

    overall_wide = overall.set_index("Method")
    comparison_overall = pd.DataFrame(
        [
            {
                "N": int(overall_wide.loc[primary_method_name, "N"]),
                f"MeanAccuracy_{primary_method_name}": float(overall_wide.loc[primary_method_name, "MeanAccuracy"]),
                "MeanAccuracy_NNLS": float(overall_wide.loc["NNLS", "MeanAccuracy"]),
                f"DeltaAccuracy_{primary_method_name}MinusNNLS": float(
                    overall_wide.loc[primary_method_name, "MeanAccuracy"]
                    - overall_wide.loc["NNLS", "MeanAccuracy"]
                ),
                f"MeanL1_{primary_method_name}": float(overall_wide.loc[primary_method_name, "MeanL1"]),
                "MeanL1_NNLS": float(overall_wide.loc["NNLS", "MeanL1"]),
                f"DeltaL1_{primary_method_name}MinusNNLS": float(
                    overall_wide.loc[primary_method_name, "MeanL1"]
                    - overall_wide.loc["NNLS", "MeanL1"]
                ),
                f"MeanRuntimeSec_{primary_method_name}": float(overall_wide.loc[primary_method_name, "MeanRuntimeSec"]),
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

    config_payload = asdict(config)
    if use_hierarchy:
        for unused_tree_key in (
            "coarse_group_count",
            "linkage_method",
            "single_cell_equal_subject_weight",
            "single_cell_log1p",
        ):
            config_payload.pop(unused_tree_key, None)
    config_payload.update(
        {
            "data_root": str(args.data_root),
            "n_mixtures_per_tissue": args.n_mixtures,
            "signature_name": args.signature_name,
            "bulk_name": args.bulk_name,
            "truth_name": args.truth_name,
            "hierarchical": bool(use_hierarchy),
            "hierarchy_file": "" if args.hierarchy_file is None else str(args.hierarchy_file),
            "hierarchy_source": "manual" if use_hierarchy else "",
            "method_metasort": (
                "HierarchicalMetaSortSolver(HierarchicalMetaSortConfig())"
                if use_hierarchy
                else "MetaSortSolver(MetaSortConfig())"
            ),
            "method_nnls": (
                "scipy.optimize.nnls on the same preprocessed signature and bulk, "
                "normalized to proportions"
            ),
            "preprocessing": (
                "shared_genes -> signature column sum-to-one, bulk vector sum-to-one -> "
                "joint gene-row z-score of signature plus selected bulk"
            ),
            "elapsed_sec": time.time() - run_start,
            "python": sys.executable,
        }
    )
    pd.DataFrame([config_payload]).to_csv(output_dir / "config.csv", index=False)
    (output_dir / "config.json").write_text(json.dumps(config_payload, indent=2, sort_keys=True) + "\n")

    print(f"Wrote outputs to {output_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
