from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from metasort import MetaSortConfig, MetaSortSolver
from metasort.algorithm import (
    _joint_gene_zscore,
    _normalize_columns_to_sum_one,
    _normalize_vector_to_sum_one,
)


DEFAULT_DATA_ROOT = Path("/home/zhiyuan/nfs/omics_remote/yunhao/nar_dataset_sim")
DEFAULT_OUTPUT_DIR = Path("/home/zhiyuan/metasort_jk/outputs/nar30_first10_compare")
DATASETS = ("Tabula-Sapiens", "CELlxGENE")
BASELINE_METHODS = ("NNLS", "DWLS", "HiDecon", "LinDeconSeq", "MuSic", "bayesprism")


def normalize_nonnegative(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(np.sum(values))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(values.shape, 1.0 / len(values), dtype=float)
    return values / total


def find_case_dirs(data_root: Path, datasets: tuple[str, ...]) -> list[Path]:
    case_dirs: list[Path] = []
    for dataset in datasets:
        dataset_dir = data_root / dataset
        for case_dir in sorted(dataset_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            if all((case_dir / name).exists() for name in ("signature.txt", "bulk.txt", "bulkRatio.txt")):
                case_dirs.append(case_dir)
    return case_dirs


def load_case(case_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signature = pd.read_csv(case_dir / "signature.txt", sep="\t", index_col=0)
    bulk = pd.read_csv(case_dir / "bulk.txt", sep="\t", index_col=0)
    truth = pd.read_csv(case_dir / "bulkRatio.txt", sep="\t", index_col=0)
    signature.columns = signature.columns.astype(str)
    truth.index = truth.index.astype(str)
    return signature, bulk, truth


def preprocess_inputs(
    signature_df: pd.DataFrame,
    bulk_df: pd.DataFrame,
    mixture: str,
    cell_types: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    common_genes = bulk_df.index.intersection(signature_df.index)
    if len(common_genes) == 0:
        raise ValueError("No shared genes between signature and bulk.")
    signature = _normalize_columns_to_sum_one(
        signature_df.loc[common_genes, cell_types].to_numpy(dtype=float),
        name="signature",
    )
    bulk = _normalize_vector_to_sum_one(
        bulk_df.loc[common_genes, mixture].to_numpy(dtype=float),
        name="bulk",
    )
    signature, bulk = _joint_gene_zscore(signature, bulk)
    return signature, bulk, common_genes.to_list()


def score_vectors(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred = normalize_nonnegative(pred)
    truth = normalize_nonnegative(truth)
    diff = pred - truth
    l1 = float(np.sum(np.abs(diff)))
    if len(pred) > 1 and np.std(pred) > 0.0 and np.std(truth) > 0.0:
        pearson = float(np.corrcoef(pred, truth)[0, 1])
    else:
        pearson = float("nan")
    return {
        "l1": l1,
        "accuracy": 1.0 - 0.5 * l1,
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "pearson": pearson,
    }


def run_metasort_case(
    case_dir: Path,
    output_dir: Path,
    cfg: MetaSortConfig,
    n_mixtures: int,
) -> tuple[dict[str, Any], pd.DataFrame, Path]:
    dataset = case_dir.parent.name
    tissue = case_dir.name
    signature_df, bulk_df, truth_df = load_case(case_dir)
    cell_types = [cell_type for cell_type in signature_df.columns if cell_type in truth_df.index]
    if not cell_types:
        raise ValueError(f"{case_dir}: no common cell types between signature and truth")
    mixtures = bulk_df.columns.astype(str).to_list()[:n_mixtures]
    solver = MetaSortSolver(cfg)
    result = pd.DataFrame(index=mixtures, columns=cell_types, dtype=float)
    metric_rows: list[dict[str, Any]] = []
    start_case = time.time()

    for mixture_idx, mixture in enumerate(mixtures, start=1):
        signature, bulk, genes = preprocess_inputs(signature_df, bulk_df, mixture, cell_types)
        start = time.time()
        solve_result = solver.solve(signature, bulk, cell_types=cell_types)
        runtime = time.time() - start
        props = normalize_nonnegative(np.asarray(solve_result.proportions, dtype=float))
        result.loc[mixture, :] = props
        metric_rows.append(
            {
                "dataset": dataset,
                "tissue": tissue,
                "case": f"{dataset}/{tissue}",
                "mixture": mixture,
                "mixture_index": mixture_idx,
                "genes": len(genes),
                "cell_types": len(cell_types),
                "runtime_sec": runtime,
                "converged": bool(solve_result.converged),
                "iterations": int(solve_result.iterations),
                "change_history": ";".join(f"{value:.6g}" for value in solve_result.change_history),
                "meta_weight_min": float(solve_result.meta_weight_min),
                "meta_weight_mean": float(solve_result.meta_weight_mean),
                "meta_weight_max": float(solve_result.meta_weight_max),
                "total_weight_min": float(solve_result.total_weight_min),
                "total_weight_mean": float(solve_result.total_weight_mean),
                "total_weight_max": float(solve_result.total_weight_max),
            }
        )
        print(
            f"{dataset}/{tissue} {mixture_idx}/{len(mixtures)} {mixture}: "
            f"iter={solve_result.iterations} converged={solve_result.converged}",
            flush=True,
        )

    prediction_dir = output_dir / "predictions" / dataset / tissue
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = prediction_dir / "MetaSort.txt"
    result.to_csv(prediction_file, sep="\t")
    success_row = {
        "dataset": dataset,
        "tissue": tissue,
        "case": f"{dataset}/{tissue}",
        "status": "ok",
        "mixtures": len(mixtures),
        "cell_types": len(cell_types),
        "elapsed_sec": time.time() - start_case,
        "out_file": str(prediction_file),
    }
    return success_row, pd.DataFrame(metric_rows), prediction_file


def method_prediction(method_file: Path, truth: pd.DataFrame, mixtures: list[str]) -> pd.DataFrame:
    pred = pd.read_csv(method_file, sep="\t", index_col=0)
    pred.columns = pred.columns.astype(str)
    pred.index = pred.index.astype(str)
    common_cell_types = [cell_type for cell_type in truth.index if cell_type in pred.columns]
    if not common_cell_types:
        common_cell_types = [cell_type for cell_type in truth.index if cell_type in pred.index]
        if common_cell_types:
            pred = pred.T
    if not common_cell_types:
        raise ValueError(f"{method_file}: no cell type overlap with truth")
    if all(mixture in pred.index for mixture in mixtures):
        aligned = pred.loc[mixtures, common_cell_types]
    elif pred.shape[0] >= len(mixtures):
        aligned = pred.iloc[: len(mixtures), :].copy()
        aligned.index = mixtures
        aligned = aligned.loc[:, common_cell_types]
    else:
        raise ValueError(f"{method_file}: only {pred.shape[0]} rows for {len(mixtures)} mixtures")
    return aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def evaluate_case(
    case_dir: Path,
    methods: tuple[str, ...],
    n_mixtures: int,
    metasort_file: Path | None,
) -> pd.DataFrame:
    _, bulk, truth = load_case(case_dir)
    mixtures = bulk.columns.astype(str).to_list()[:n_mixtures]
    rows: list[dict[str, Any]] = []
    for method in methods:
        method_file = metasort_file if method == "MetaSort" else case_dir / "result" / f"{method}.txt"
        if method_file is None or not method_file.exists():
            rows.append(
                {
                    "dataset": case_dir.parent.name,
                    "tissue": case_dir.name,
                    "case": f"{case_dir.parent.name}/{case_dir.name}",
                    "method": method,
                    "status": "missing",
                }
            )
            continue
        try:
            pred = method_prediction(method_file, truth, mixtures)
            common_cell_types = [cell_type for cell_type in truth.index if cell_type in pred.columns]
            for mixture_index, mixture in enumerate(mixtures, start=1):
                rows.append(
                    {
                        "dataset": case_dir.parent.name,
                        "tissue": case_dir.name,
                        "case": f"{case_dir.parent.name}/{case_dir.name}",
                        "method": method,
                        "mixture": mixture,
                        "mixture_index": mixture_index,
                        "n_cell_types": len(common_cell_types),
                        "status": "ok",
                        **score_vectors(
                            pred.loc[mixture, common_cell_types].to_numpy(dtype=float),
                            truth.loc[common_cell_types, mixture].to_numpy(dtype=float),
                        ),
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "dataset": case_dir.parent.name,
                    "tissue": case_dir.name,
                    "case": f"{case_dir.parent.name}/{case_dir.name}",
                    "method": method,
                    "status": "failed",
                    "error": repr(exc),
                }
            )
    return pd.DataFrame(rows)


def write_summaries(evaluation: pd.DataFrame, output_dir: Path, print_tables: bool = False) -> None:
    ok = evaluation.loc[evaluation["status"] == "ok"].copy()
    method_summary = (
        ok.groupby(["method"], as_index=False)
        .agg(
            datasets=("dataset", "nunique"),
            cases=("case", "nunique"),
            mixtures=("mixture", "count"),
            mean_l1=("l1", "mean"),
            median_l1=("l1", "median"),
            mean_accuracy=("accuracy", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_pearson=("pearson", "mean"),
        )
        .sort_values(["mean_l1", "mean_rmse"], kind="mergesort")
    )
    dataset_method_summary = (
        ok.groupby(["dataset", "method"], as_index=False)
        .agg(
            cases=("case", "nunique"),
            mixtures=("mixture", "count"),
            mean_l1=("l1", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_pearson=("pearson", "mean"),
        )
        .sort_values(["dataset", "mean_l1"], kind="mergesort")
    )
    tissue_method_summary = (
        ok.groupby(["dataset", "tissue", "case", "method"], as_index=False)
        .agg(
            mixtures=("mixture", "count"),
            mean_l1=("l1", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            mean_pearson=("pearson", "mean"),
        )
        .sort_values(["dataset", "tissue", "mean_l1"], kind="mergesort")
    )
    pivot = tissue_method_summary.pivot_table(
        index=["dataset", "tissue", "case"],
        columns="method",
        values="mean_l1",
        aggfunc="first",
    )
    method_order = [method for method in BASELINE_METHODS + ("MetaSort",) if method in pivot.columns]
    pivot = pivot[method_order]
    pivot["best_method"] = pivot[method_order].idxmin(axis=1)
    pivot["best_l1"] = pivot[method_order].min(axis=1)
    pivot["metasort_rank"] = pivot[method_order].rank(axis=1, method="min", ascending=True)["MetaSort"].astype(int)

    evaluation.to_csv(output_dir / "method_mixture_metrics.csv", index=False)
    method_summary.to_csv(output_dir / "method_summary.csv", index=False)
    dataset_method_summary.to_csv(output_dir / "dataset_method_summary.csv", index=False)
    tissue_method_summary.to_csv(output_dir / "tissue_method_summary.csv", index=False)
    pivot.reset_index().to_csv(output_dir / "tissue_mean_l1_pivot.csv", index=False)

    if print_tables:
        print("\nOverall method summary:")
        print(method_summary.to_string(index=False))
        print("\nDataset method summary:")
        print(dataset_method_summary.to_string(index=False))


def process_case(
    case_dir: Path,
    output_dir: Path,
    cfg: MetaSortConfig,
    n_mixtures: int,
    methods: tuple[str, ...],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, pd.DataFrame | None, pd.DataFrame]:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    dataset = case_dir.parent.name
    tissue = case_dir.name
    metasort_file: Path | None = None
    success_row = None
    failed_row = None
    metric_df = None
    try:
        success_row, metric_df, metasort_file = run_metasort_case(case_dir, output_dir, cfg, n_mixtures)
        print(
            f"  wrote {success_row['out_file']} "
            f"({success_row['mixtures']} x {success_row['cell_types']})",
            flush=True,
        )
    except Exception as exc:
        failed_row = {
            "dataset": dataset,
            "tissue": tissue,
            "case": f"{dataset}/{tissue}",
            "status": "failed",
            "error": repr(exc),
        }
        print(f"  FAILED MetaSort {dataset}/{tissue}: {exc!r}", flush=True)
    evaluation = evaluate_case(case_dir, methods, n_mixtures, metasort_file)
    return success_row, failed_row, metric_df, evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--n-mixtures", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--lambda-hessian", type=float, default=1.0)
    parser.add_argument("--lambda-residual", type=float, default=0.005)
    parser.add_argument("--convergence-tol", type=float, default=0.02)
    parser.add_argument(
        "--averaging-old-weight",
        type=int,
        default=4,
        help=(
            "Weight assigned to the previous proportion vector when averaging with "
            "the new weighted solve. Set to 0 to disable old/new proportion smoothing."
        ),
    )
    parser.add_argument("--meta-weight-baseline", type=float, default=1.0)
    parser.add_argument("--meta-weight-floor", type=float, default=1e-2)
    parser.add_argument("--lambda3", type=float, default=0.01)
    parser.add_argument("--lambda4", type=float, default=0.001)
    parser.add_argument("--sqrt-sphere-hessian", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--normalize-meta-weight-mean", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.averaging_old_weight < 0:
        parser.error("--averaging-old-weight must be nonnegative")

    cfg = MetaSortConfig(
        convergence_tol=args.convergence_tol,
        averaging_old_weight=args.averaging_old_weight,
        lambda_hessian=args.lambda_hessian,
        lambda_residual=args.lambda_residual,
        use_sqrt_sphere_hessian=args.sqrt_sphere_hessian,
        lambda3=args.lambda3,
        lambda4=args.lambda4,
        meta_weight_baseline=args.meta_weight_baseline,
        meta_weight_floor=args.meta_weight_floor,
        normalize_meta_weight_mean=args.normalize_meta_weight_mean,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = BASELINE_METHODS + ("MetaSort",)
    case_dirs = find_case_dirs(args.data_root, tuple(args.datasets))
    if args.max_cases is not None:
        case_dirs = case_dirs[: args.max_cases]
    print(f"cases: {len(case_dirs)}", flush=True)

    with (args.output_dir / "run_config.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(cfg).keys()) + ["datasets", "n_mixtures", "methods", "workers"],
        )
        writer.writeheader()
        writer.writerow(
            asdict(cfg)
            | {
                "datasets": ",".join(args.datasets),
                "n_mixtures": args.n_mixtures,
                "methods": ",".join(methods),
                "workers": args.workers,
            }
        )

    success_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    all_metasort_metrics: list[pd.DataFrame] = []
    all_evaluations: list[pd.DataFrame] = []

    def record_result(
        success_row: dict[str, Any] | None,
        failed_row: dict[str, Any] | None,
        metric_df: pd.DataFrame | None,
        evaluation: pd.DataFrame,
    ) -> None:
        if success_row is not None:
            success_rows.append(success_row)
        if failed_row is not None:
            failed_rows.append(failed_row)
        if metric_df is not None:
            all_metasort_metrics.append(metric_df)
        all_evaluations.append(evaluation)
        pd.DataFrame(success_rows).to_csv(args.output_dir / "run_success.csv", index=False)
        pd.DataFrame(failed_rows).to_csv(args.output_dir / "run_failed.csv", index=False)
        if all_metasort_metrics:
            pd.concat(all_metasort_metrics, ignore_index=True).to_csv(
                args.output_dir / "metasort_mixture_metrics.csv",
                index=False,
            )
        if all_evaluations:
            write_summaries(pd.concat(all_evaluations, ignore_index=True), args.output_dir)

    if args.workers <= 1:
        for idx, case_dir in enumerate(case_dirs, start=1):
            print(f"[{idx}/{len(case_dirs)}] {case_dir.parent.name}/{case_dir.name}", flush=True)
            record_result(*process_case(case_dir, args.output_dir, cfg, args.n_mixtures, methods))
    else:
        print(f"workers: {args.workers}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for idx, case_dir in enumerate(case_dirs, start=1):
                print(f"[submit {idx}/{len(case_dirs)}] {case_dir.parent.name}/{case_dir.name}", flush=True)
                futures[
                    executor.submit(process_case, case_dir, args.output_dir, cfg, args.n_mixtures, methods)
                ] = case_dir
            completed = 0
            for future in as_completed(futures):
                case_dir = futures[future]
                completed += 1
                try:
                    record_result(*future.result())
                    print(f"[done {completed}/{len(case_dirs)}] {case_dir.parent.name}/{case_dir.name}", flush=True)
                except Exception as exc:
                    failed_rows.append(
                        {
                            "dataset": case_dir.parent.name,
                            "tissue": case_dir.name,
                            "case": f"{case_dir.parent.name}/{case_dir.name}",
                            "status": "failed",
                            "error": repr(exc),
                        }
                    )
                    pd.DataFrame(failed_rows).to_csv(args.output_dir / "run_failed.csv", index=False)
                    print(
                        f"[done {completed}/{len(case_dirs)}] "
                        f"{case_dir.parent.name}/{case_dir.name} FAILED: {exc!r}",
                        flush=True,
                    )

    if all_evaluations:
        write_summaries(pd.concat(all_evaluations, ignore_index=True), args.output_dir, print_tables=True)
    print(f"completed: {len(success_rows)}", flush=True)
    print(f"failed: {len(failed_rows)}", flush=True)
    print(f"output_dir: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
