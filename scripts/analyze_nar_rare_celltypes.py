from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = ("CELlxGENE", "Tabula-Sapiens")
BASELINES = ("NNLS", "DWLS", "HiDecon", "LinDeconSeq", "MuSic", "bayesprism")
METHODS = BASELINES + ("MetaSort_jk",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MetaSort_jk with NAR baselines for low-abundance cell types."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/zhiyuan/nfs/omics_remote/yunhao/nar_dataset_sim"),
    )
    parser.add_argument("--metasort-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rare-threshold", type=float, default=0.05)
    parser.add_argument("--min-rare-mixtures", type=int, default=5)
    parser.add_argument(
        "--write-case-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write each completed MetaSort matrix to tissue/result/MetaSort_jk.txt.",
    )
    return parser.parse_args()


def normalize_rows(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    values = np.clip(values, 0.0, None)
    totals = values.sum(axis=1)
    bad = (~np.isfinite(totals)) | (totals <= 0.0)
    totals[bad] = 1.0
    values = values / totals[:, None]
    if np.any(bad):
        values[bad, :] = 1.0 / values.shape[1]
    return pd.DataFrame(values, index=frame.index, columns=frame.columns)


def discover_cases(data_root: Path) -> list[Path]:
    cases: list[Path] = []
    for dataset in DATASETS:
        for case_dir in sorted((data_root / dataset).iterdir()):
            if all((case_dir / name).exists() for name in ("bulkRatio.txt", "bulk.txt", "signature.txt")):
                cases.append(case_dir)
    return cases


def align_prediction(
    path: Path,
    truth: pd.DataFrame,
    expected_mixtures: list[str],
) -> pd.DataFrame:
    pred = pd.read_csv(path, sep="\t", index_col=0)
    pred.index = pred.index.astype(str)
    pred.columns = pred.columns.astype(str)
    truth_cell_types = truth.index.astype(str).to_list()

    column_overlap = len(set(pred.columns) & set(truth_cell_types))
    index_overlap = len(set(pred.index) & set(truth_cell_types))
    if index_overlap > column_overlap:
        pred = pred.T
        pred.index = pred.index.astype(str)
        pred.columns = pred.columns.astype(str)

    missing_cell_types = sorted(set(truth_cell_types) - set(pred.columns))
    if missing_cell_types:
        raise ValueError(f"{path}: missing cell types {missing_cell_types}")
    pred = pred.loc[:, truth_cell_types]

    exact = [mixture for mixture in expected_mixtures if mixture in pred.index]
    if exact:
        pred = pred.loc[exact]
    else:
        numbered: dict[int, str] = {}
        for label in pred.index:
            match = re.search(r"(\d+)$", label)
            if match:
                numbered[int(match.group(1))] = label
        ordered_labels = []
        ordered_mixtures = []
        for position, mixture in enumerate(expected_mixtures, start=1):
            if position in numbered:
                ordered_labels.append(numbered[position])
                ordered_mixtures.append(mixture)
        if ordered_labels:
            pred = pred.loc[ordered_labels].copy()
            pred.index = ordered_mixtures
        elif len(pred) <= len(expected_mixtures):
            pred = pred.iloc[: len(expected_mixtures)].copy()
            pred.index = expected_mixtures[: len(pred)]
        else:
            raise ValueError(f"{path}: cannot align mixture rows")
    return normalize_rows(pred)


def method_path(case_dir: Path, metasort_output_dir: Path, method: str) -> Path:
    if method == "MetaSort_jk":
        return (
            metasort_output_dir
            / "predictions"
            / case_dir.parent.name
            / case_dir.name
            / "MetaSort.txt"
        )
    return case_dir / "result" / f"{method}.txt"


def add_winner_columns(summary: pd.DataFrame, error_col: str) -> pd.DataFrame:
    keys = ["dataset", "tissue", "case", "cell_type"]
    pivot = summary.pivot(index=keys, columns="method", values=error_col)
    for method in METHODS:
        if method not in pivot.columns:
            pivot[method] = np.nan
    pivot = pivot[list(METHODS)]
    method_values = pivot[list(METHODS)].to_numpy(dtype=float)
    ranked = np.sort(method_values, axis=1)
    ranked_method_indices = np.argsort(method_values, axis=1)
    pivot["best_method"] = pivot[list(METHODS)].idxmin(axis=1)
    pivot["second_best_method"] = [METHODS[index] for index in ranked_method_indices[:, 1]]
    pivot["best_mae"] = ranked[:, 0]
    pivot["second_best_mae"] = ranked[:, 1]
    pivot["margin_vs_second"] = ranked[:, 1] - ranked[:, 0]
    pivot["relative_margin_vs_second"] = np.where(
        ranked[:, 1] > 0.0,
        (ranked[:, 1] - ranked[:, 0]) / ranked[:, 1],
        np.nan,
    )
    pivot["metasort_rank"] = (
        pivot[list(METHODS)].rank(axis=1, method="min", ascending=True)["MetaSort_jk"].astype(int)
    )
    pivot["metasort_best"] = pivot["best_method"].eq("MetaSort_jk")
    return pivot.reset_index()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.rare_threshold < 1.0:
        raise ValueError("--rare-threshold must be between 0 and 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detail_parts: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    for case_dir in discover_cases(args.data_root):
        dataset = case_dir.parent.name
        tissue = case_dir.name
        truth = pd.read_csv(case_dir / "bulkRatio.txt", sep="\t", index_col=0)
        truth.index = truth.index.astype(str)
        truth.columns = truth.columns.astype(str)
        expected_mixtures = truth.columns.to_list()

        predictions: dict[str, pd.DataFrame] = {}
        for method in METHODS:
            path = method_path(case_dir, args.metasort_output_dir, method)
            if not path.exists():
                raise FileNotFoundError(path)
            predictions[method] = align_prediction(path, truth, expected_mixtures)
            coverage_rows.append(
                {
                    "dataset": dataset,
                    "tissue": tissue,
                    "case": f"{dataset}/{tissue}",
                    "method": method,
                    "available_mixtures": len(predictions[method]),
                }
            )

        common = [
            mixture
            for mixture in expected_mixtures
            if all(mixture in predictions[method].index for method in METHODS)
        ]
        if not common:
            raise ValueError(f"{dataset}/{tissue}: no mixtures common to every method")

        if args.write_case_results:
            destination = case_dir / "result" / "MetaSort_jk.txt"
            predictions["MetaSort_jk"].to_csv(destination, sep="\t")

        truth_long = (
            truth.loc[:, common]
            .T.rename_axis(index="mixture", columns="cell_type")
            .stack()
            .rename("truth")
            .reset_index()
        )
        for method in METHODS:
            pred_long = (
                predictions[method]
                .loc[common]
                .rename_axis(index="mixture", columns="cell_type")
                .stack()
                .rename("prediction")
                .reset_index()
            )
            part = truth_long.merge(pred_long, on=["mixture", "cell_type"], validate="one_to_one")
            part.insert(0, "method", method)
            part.insert(0, "case", f"{dataset}/{tissue}")
            part.insert(0, "tissue", tissue)
            part.insert(0, "dataset", dataset)
            part["abs_error"] = (part["prediction"] - part["truth"]).abs()
            part["signed_error"] = part["prediction"] - part["truth"]
            part["rare_observation"] = (part["truth"] > 0.0) & (
                part["truth"] < args.rare_threshold
            )
            detail_parts.append(part)
        print(f"{dataset}/{tissue}: common mixtures={len(common)}", flush=True)

    detail = pd.concat(detail_parts, ignore_index=True)
    coverage = pd.DataFrame(coverage_rows)
    keys = ["dataset", "tissue", "case", "cell_type", "method"]

    cell_summary = (
        detail.groupby(keys, as_index=False)
        .agg(
            n_mixtures=("mixture", "size"),
            mean_truth=("truth", "mean"),
            median_truth=("truth", "median"),
            mae_all=("abs_error", "mean"),
            median_ae_all=("abs_error", "median"),
        )
    )
    rare = detail.loc[detail["rare_observation"]].copy()
    rare_summary = (
        rare.groupby(keys, as_index=False)
        .agg(
            rare_n=("mixture", "size"),
            rare_mean_truth=("truth", "mean"),
            rare_median_truth=("truth", "median"),
            rare_mae=("abs_error", "mean"),
            rare_median_ae=("abs_error", "median"),
        )
    )
    cell_summary = cell_summary.merge(rare_summary, on=keys, how="left")

    rare_eligible = rare_summary.loc[rare_summary["rare_n"] >= args.min_rare_mixtures].copy()
    rare_winners = add_winner_columns(rare_eligible, "rare_mae")
    rare_metadata = (
        rare_eligible.loc[rare_eligible["method"] == "MetaSort_jk", keys[:-1] + ["rare_n", "rare_mean_truth", "rare_median_truth"]]
    )
    rare_winners = rare_winners.merge(rare_metadata, on=keys[:-1], how="left")
    rare_winners = rare_winners.sort_values(
        ["metasort_best", "margin_vs_second", "case", "cell_type"],
        ascending=[False, False, True, True],
    )

    low_celltypes = cell_summary.loc[cell_summary["mean_truth"] < args.rare_threshold].copy()
    low_winners = add_winner_columns(low_celltypes, "mae_all")
    low_metadata = low_celltypes.loc[
        low_celltypes["method"] == "MetaSort_jk",
        keys[:-1] + ["n_mixtures", "mean_truth", "median_truth"],
    ]
    low_winners = low_winners.merge(low_metadata, on=keys[:-1], how="left")
    low_winners = low_winners.sort_values(
        ["metasort_best", "margin_vs_second", "case", "cell_type"],
        ascending=[False, False, True, True],
    )

    rare_method_summary = (
        rare.groupby("method", as_index=False)
        .agg(n=("abs_error", "size"), mae=("abs_error", "mean"), median_ae=("abs_error", "median"))
        .sort_values("mae")
    )

    detail.to_csv(args.output_dir / "celltype_mixture_method_detail.csv", index=False)
    coverage.to_csv(args.output_dir / "method_coverage.csv", index=False)
    cell_summary.to_csv(args.output_dir / "tissue_celltype_method_summary.csv", index=False)
    rare_winners.to_csv(args.output_dir / "rare_observation_winners.csv", index=False)
    low_winners.to_csv(args.output_dir / "low_mean_proportion_celltype_winners.csv", index=False)
    rare_method_summary.to_csv(args.output_dir / "rare_observation_method_summary.csv", index=False)

    meta_rare = rare_winners.loc[rare_winners["metasort_best"]]
    meta_low = low_winners.loc[low_winners["metasort_best"]]
    def winner_table(frame: pd.DataFrame, n_col: str, truth_col: str) -> list[str]:
        lines = [
            "| Dataset / tissue | Cell type | n | Mean truth | MetaSort MAE | Runner-up (MAE) | Improvement |",
            "|---|---|---:|---:|---:|---|---:|",
        ]
        for row in frame.itertuples(index=False):
            lines.append(
                f"| {row.dataset} / {row.tissue} | {row.cell_type} | "
                f"{int(getattr(row, n_col))} | {getattr(row, truth_col):.2%} | "
                f"{row.MetaSort_jk:.5f} | {row.second_best_method} ({row.second_best_mae:.5f}) | "
                f"{row.relative_margin_vs_second:.1%} |"
            )
        return lines

    report = [
        "# MetaSort_jk low-abundance cell-type comparison",
        "",
        f"Rare threshold: `0 < truth < {args.rare_threshold:g}`; minimum rare observations per tissue/cell type: {args.min_rare_mixtures}.",
        "Every winner comparison uses the intersection of mixtures available for all seven methods within that tissue.",
        "",
        f"- Eligible tissue/cell-type pairs under the rare-observation definition: {len(rare_winners)}",
        f"- MetaSort_jk winners under the rare-observation definition: {len(meta_rare)}",
        f"- Cell types with mean truth below threshold: {len(low_winners)}",
        f"- MetaSort_jk winners among low-mean-proportion cell types: {len(meta_low)}",
        "",
        "## Low-mean-proportion cell types won by MetaSort_jk",
        "",
        "Here a cell type is rare when its mean true proportion across common mixtures is below the threshold; MAE uses all common mixtures.",
        "",
        *winner_table(meta_low, "n_mixtures", "mean_truth"),
        "",
        "## Rare-observation comparisons won by MetaSort_jk",
        "",
        "Here MAE is restricted to mixtures where that cell type has a positive true proportion below the threshold.",
        "",
        *winner_table(meta_rare, "rare_n", "rare_mean_truth"),
        "",
        "## Overall MAE on rare observations",
        "",
        "| Method | n | MAE | Median absolute error |",
        "|---|---:|---:|---:|",
        *[
            f"| {row.method} | {int(row.n)} | {row.mae:.5f} | {row.median_ae:.5f} |"
            for row in rare_method_summary.itertuples(index=False)
        ],
        "",
        "The tissue/cell-type wins are local findings; MetaSort_jk is not the best method on rare observations overall.",
        "Tabula-Sapiens/Thymus uses 10 common mixtures because its MuSic result contains only 10 rows; all other tissues use 100.",
        "",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(report) + "\n")
    (args.output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "data_root": str(args.data_root),
                "metasort_output_dir": str(args.metasort_output_dir),
                "rare_threshold": args.rare_threshold,
                "min_rare_mixtures": args.min_rare_mixtures,
                "methods": METHODS,
                "winner_metric_rare_observation": "mean absolute error over 0 < truth < threshold",
                "winner_metric_low_mean_celltype": "mean absolute error over all common mixtures",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"MetaSort rare-observation winners: {len(meta_rare)}/{len(rare_winners)}")
    print(f"MetaSort low-mean-proportion winners: {len(meta_low)}/{len(low_winners)}")


if __name__ == "__main__":
    main()
