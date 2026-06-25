from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import fisher_exact, mannwhitneyu, norm, pearsonr, rankdata, spearmanr, wilcoxon

from metasort import MetaSortConfig, MetaSortSolver
from scripts.run_nar30_first10_compare import load_case, preprocess_inputs


TARGETS = (
    ("Tabula-Sapiens", "Pancreas", "B_cell"),
    ("Tabula-Sapiens", "Salivary_Gland", "mature_NK_T_cell"),
    ("Tabula-Sapiens", "Skin", "CD1c_positive_myeloid_dendritic_cell"),
    ("Tabula-Sapiens", "Skin", "naive_thymus_derived_CD8_positive_alpha_beta_T_cell"),
    ("CELlxGENE", "Primary_Somatosensory_Cortex", "oligodendrocyte_precursor_cell"),
    ("CELlxGENE", "Primary_Visual_Cortex", "chandelier_pvalb_gabaergic_cortical_interneuron"),
)


class WeightRecordingSolver(MetaSortSolver):
    def __init__(self, config: MetaSortConfig) -> None:
        super().__init__(config)
        self.last_meta_weights: np.ndarray | None = None

    def solve_meta_weighted_step(self, *args, **kwargs):
        result = super().solve_meta_weighted_step(*args, **kwargs)
        self.last_meta_weights = np.asarray(result[1], dtype=float).copy()
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/zhiyuan/nfs/omics_remote/yunhao/nar_dataset_sim"),
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("outputs/nar30_all100_sqrt_sphere_tol0p01_no_averaging/predictions"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-log2fc", type=float, default=1.0)
    parser.add_argument("--max-fdr", type=float, default=0.05)
    parser.add_argument("--min-pct-in", type=float, default=0.10)
    parser.add_argument("--pseudocount", type=float, default=0.1)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def safe_correlations(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    if len(x) < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan, np.nan, np.nan, np.nan
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return (
        float(pearson.statistic),
        float(pearson.pvalue),
        float(spearman.statistic),
        float(spearman.pvalue),
    )


def run_de(
    case_dir: Path,
    min_log2fc: float,
    max_fdr: float,
    min_pct_in: float,
    pseudocount: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signature = pd.read_csv(case_dir / "signature.txt", sep="\t", index_col=0)
    expression = pd.read_csv(case_dir / "singleCellExpr.txt", sep="\t", index_col=0)
    labels = pd.read_csv(case_dir / "singleCellLabels.txt", sep="\t").iloc[:, 0].astype(str)
    if expression.shape[1] != len(labels):
        raise ValueError(f"{case_dir}: expression columns and labels differ")
    genes = signature.index.astype(str).to_list()
    missing = sorted(set(genes) - set(expression.index.astype(str)))
    if missing:
        raise ValueError(f"{case_dir}: single-cell expression is missing {len(missing)} signature genes")
    expression.index = expression.index.astype(str)
    matrix = expression.loc[genes].to_numpy(dtype=float)
    ranks = rankdata(matrix, axis=1, method="average")
    n_total = matrix.shape[1]
    tie_terms = np.empty(matrix.shape[0], dtype=float)
    for gene_index, values in enumerate(matrix):
        counts = np.unique(values, return_counts=True)[1].astype(float)
        tie_terms[gene_index] = np.sum(counts**3 - counts)
    existing_markers = pd.read_csv(case_dir / "markers.txt", sep="\t")
    existing_pairs = set(zip(existing_markers["cell_type"].astype(str), existing_markers["gene"].astype(str)))

    rows: list[pd.DataFrame] = []
    for cell_type in signature.columns.astype(str):
        inside = labels.eq(cell_type).to_numpy()
        outside = ~inside
        inside_values = matrix[:, inside]
        outside_values = matrix[:, outside]
        mean_in = inside_values.mean(axis=1)
        mean_out = outside_values.mean(axis=1)
        pct_in = (inside_values > 0.0).mean(axis=1)
        pct_out = (outside_values > 0.0).mean(axis=1)
        n_in = int(inside.sum())
        n_out = int(outside.sum())
        rank_sum_in = ranks[:, inside].sum(axis=1)
        u_statistic = rank_sum_in - n_in * (n_in + 1) / 2.0
        u_mean = n_in * n_out / 2.0
        u_variance = (n_in * n_out / 12.0) * (
            (n_total + 1.0) - tie_terms / (n_total * (n_total - 1.0))
        )
        valid_variance = u_variance > 0.0
        z = np.zeros_like(u_statistic)
        z[valid_variance] = (
            u_statistic[valid_variance] - u_mean - 0.5
        ) / np.sqrt(u_variance[valid_variance])
        pvalue = np.ones_like(u_statistic)
        pvalue[valid_variance] = norm.sf(z[valid_variance])
        fdr = bh_adjust(pvalue)
        log2fc = np.log2((mean_in + pseudocount) / (mean_out + pseudocount))
        frame = pd.DataFrame(
            {
                "dataset": case_dir.parent.name,
                "tissue": case_dir.name,
                "cell_type": cell_type,
                "gene": genes,
                "n_cells_in": n_in,
                "n_cells_out": n_out,
                "mean_in": mean_in,
                "mean_out": mean_out,
                "log2fc": log2fc,
                "pct_in": pct_in,
                "pct_out": pct_out,
                "pct_difference": pct_in - pct_out,
                "u_statistic": u_statistic,
                "pvalue": pvalue,
                "fdr": fdr,
            }
        )
        frame["is_marker"] = (
            (frame["fdr"] < max_fdr)
            & (frame["log2fc"] > min_log2fc)
            & (frame["pct_in"] >= min_pct_in)
        )
        frame["in_existing_markers_file"] = [
            (cell_type, gene) in existing_pairs for gene in genes
        ]
        rows.append(frame)

    de = pd.concat(rows, ignore_index=True)
    marker_de = de.loc[de["is_marker"]].sort_values(
        ["gene", "log2fc", "pct_difference"], ascending=[True, False, False]
    )
    primary = marker_de.drop_duplicates("gene", keep="first")[
        ["gene", "cell_type", "log2fc", "fdr"]
    ].rename(
        columns={
            "cell_type": "primary_marker_cell_type",
            "log2fc": "primary_marker_log2fc",
            "fdr": "primary_marker_fdr",
        }
    )
    assignment = pd.DataFrame({"gene": genes}).merge(primary, on="gene", how="left")
    assignment.insert(0, "tissue", case_dir.name)
    assignment.insert(0, "dataset", case_dir.parent.name)
    assignment["is_any_marker"] = assignment["primary_marker_cell_type"].notna()
    return de, assignment


def extract_weights(
    case_dir: Path,
    prediction_root: Path,
    weights_dir: Path,
) -> tuple[pd.DataFrame, float]:
    signature_df, bulk_df, truth_df = load_case(case_dir)
    cell_types = [cell_type for cell_type in signature_df.columns if cell_type in truth_df.index]
    mixtures = bulk_df.columns.astype(str).to_list()
    config = MetaSortConfig(
        convergence_tol=0.01,
        averaging_old_weight=0,
        lambda_hessian=1.0,
        lambda_residual=0.005,
        use_sqrt_sphere_hessian=True,
        lambda3=0.01,
        lambda4=0.001,
        meta_weight_baseline=1.0,
        meta_weight_floor=0.01,
        normalize_meta_weight_mean=True,
    )
    solver = WeightRecordingSolver(config)
    weight_columns: dict[str, np.ndarray] = {}
    prediction_columns: dict[str, np.ndarray] = {}
    genes_expected: list[str] | None = None
    for index, mixture in enumerate(mixtures, start=1):
        signature, bulk, genes = preprocess_inputs(signature_df, bulk_df, mixture, cell_types)
        if genes_expected is None:
            genes_expected = genes
        elif genes != genes_expected:
            raise ValueError(f"{case_dir}: shared gene order changed between mixtures")
        solver.last_meta_weights = None
        result = solver.solve(signature, bulk, cell_types=cell_types)
        if solver.last_meta_weights is None:
            raise RuntimeError(f"{case_dir}/{mixture}: no meta weights were recorded")
        weight_columns[mixture] = solver.last_meta_weights
        prediction_columns[mixture] = np.asarray(result.proportions, dtype=float)
        if index % 20 == 0 or index == len(mixtures):
            print(f"weights {case_dir.parent.name}/{case_dir.name}: {index}/{len(mixtures)}", flush=True)

    if genes_expected is None:
        raise ValueError(f"{case_dir}: no mixtures")
    weights = pd.DataFrame(weight_columns, index=genes_expected)
    weights.index.name = "gene"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights.to_csv(
        weights_dir / f"{case_dir.parent.name}__{case_dir.name}__meta_weights.tsv.gz",
        sep="\t",
        compression="gzip",
    )

    rerun_prediction = pd.DataFrame(prediction_columns, index=cell_types).T
    saved_prediction = pd.read_csv(
        prediction_root / case_dir.parent.name / case_dir.name / "MetaSort.txt",
        sep="\t",
        index_col=0,
    )
    saved_prediction = saved_prediction.loc[rerun_prediction.index, rerun_prediction.columns]
    max_difference = float(
        np.max(np.abs(rerun_prediction.to_numpy() - saved_prediction.to_numpy()))
    )

    long = weights.rename_axis(columns="mixture").stack().rename("meta_weight").reset_index()
    long.insert(0, "tissue", case_dir.name)
    long.insert(0, "dataset", case_dir.parent.name)
    return long, max_difference


def analyze_targets(
    de: pd.DataFrame,
    weights: pd.DataFrame,
    data_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    gene_rows: list[pd.DataFrame] = []
    mixture_rows: list[pd.DataFrame] = []
    for dataset, tissue, cell_type in TARGETS:
        target_de = de.loc[
            (de["dataset"] == dataset)
            & (de["tissue"] == tissue)
            & (de["cell_type"] == cell_type)
        ].copy()
        target_weights = weights.loc[
            (weights["dataset"] == dataset) & (weights["tissue"] == tissue)
        ].copy()
        gene_weight = (
            target_weights.groupby("gene", as_index=False)
            .agg(
                mean_meta_weight=("meta_weight", "mean"),
                median_meta_weight=("meta_weight", "median"),
                sd_meta_weight=("meta_weight", "std"),
                fraction_weight_gt_1=("meta_weight", lambda values: float(np.mean(values > 1.0))),
            )
        )
        gene_table = target_de.merge(gene_weight, on="gene", validate="one_to_one")
        gene_table["target_cell_type"] = cell_type
        gene_table["mean_weight_amplified"] = gene_table["mean_meta_weight"] > 1.0
        top_threshold = float(gene_table["mean_meta_weight"].quantile(0.90))
        gene_table["top_10pct_mean_weight"] = gene_table["mean_meta_weight"] >= top_threshold
        gene_rows.append(gene_table)

        markers = gene_table.loc[gene_table["is_marker"]]
        nonmarkers = gene_table.loc[~gene_table["is_marker"]]
        if markers.empty:
            raise ValueError(f"No markers passed the thresholds for {dataset}/{tissue}/{cell_type}")
        mw = mannwhitneyu(
            markers["mean_meta_weight"],
            nonmarkers["mean_meta_weight"],
            alternative="greater",
            method="asymptotic",
        )
        top_marker = int((markers["top_10pct_mean_weight"]).sum())
        top_nonmarker = int((nonmarkers["top_10pct_mean_weight"]).sum())
        not_top_marker = len(markers) - top_marker
        not_top_nonmarker = len(nonmarkers) - top_nonmarker
        fisher = fisher_exact(
            [[top_marker, not_top_marker], [top_nonmarker, not_top_nonmarker]],
            alternative="greater",
        )

        marker_genes = set(markers["gene"])
        target_weights["is_target_marker"] = target_weights["gene"].isin(marker_genes)
        per_mixture = (
            target_weights.groupby(["mixture", "is_target_marker"], as_index=False)["meta_weight"]
            .mean()
            .pivot(index="mixture", columns="is_target_marker", values="meta_weight")
            .rename(columns={False: "nonmarker_mean_weight", True: "marker_mean_weight"})
            .reset_index()
        )
        truth = pd.read_csv(
            data_root / dataset / tissue / "bulkRatio.txt", sep="\t", index_col=0
        )
        truth.index = truth.index.astype(str)
        per_mixture["truth_proportion"] = [
            float(truth.loc[cell_type, mixture]) for mixture in per_mixture["mixture"]
        ]
        per_mixture.insert(0, "cell_type", cell_type)
        per_mixture.insert(0, "tissue", tissue)
        per_mixture.insert(0, "dataset", dataset)
        mixture_rows.append(per_mixture)

        pearson_r, pearson_p, spearman_r, spearman_p = safe_correlations(
            per_mixture["truth_proportion"].to_numpy(dtype=float),
            per_mixture["marker_mean_weight"].to_numpy(dtype=float),
        )
        paired = wilcoxon(
            per_mixture["marker_mean_weight"],
            per_mixture["nonmarker_mean_weight"],
            alternative="greater",
            zero_method="wilcox",
        )
        summary_rows.append(
            {
                "dataset": dataset,
                "tissue": tissue,
                "cell_type": cell_type,
                "n_target_cells": int(target_de["n_cells_in"].iloc[0]),
                "n_signature_genes": len(gene_table),
                "n_target_markers": len(markers),
                "marker_mean_weight": float(markers["mean_meta_weight"].mean()),
                "marker_median_weight": float(markers["mean_meta_weight"].median()),
                "nonmarker_mean_weight": float(nonmarkers["mean_meta_weight"].mean()),
                "marker_to_nonmarker_weight_ratio": float(
                    markers["mean_meta_weight"].mean() / nonmarkers["mean_meta_weight"].mean()
                ),
                "marker_genes_mean_weight_gt_1": int(markers["mean_weight_amplified"].sum()),
                "marker_fraction_mean_weight_gt_1": float(markers["mean_weight_amplified"].mean()),
                "marker_weight_mannwhitney_p": float(mw.pvalue),
                "top10pct_marker_genes": top_marker,
                "top10pct_enrichment_odds_ratio": float(fisher.statistic),
                "top10pct_enrichment_p": float(fisher.pvalue),
                "mean_paired_marker_minus_nonmarker_weight": float(
                    (per_mixture["marker_mean_weight"] - per_mixture["nonmarker_mean_weight"]).mean()
                ),
                "paired_mixture_wilcoxon_p": float(paired.pvalue),
                "truth_vs_marker_weight_pearson_r": pearson_r,
                "truth_vs_marker_weight_pearson_p": pearson_p,
                "truth_vs_marker_weight_spearman_r": spearman_r,
                "truth_vs_marker_weight_spearman_p": spearman_p,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary["marker_weight_mannwhitney_fdr"] = bh_adjust(
        summary["marker_weight_mannwhitney_p"].to_numpy()
    )
    summary["top10pct_enrichment_fdr"] = bh_adjust(summary["top10pct_enrichment_p"].to_numpy())
    summary["paired_mixture_wilcoxon_fdr"] = bh_adjust(
        summary["paired_mixture_wilcoxon_p"].to_numpy()
    )
    return summary, pd.concat(gene_rows, ignore_index=True), pd.concat(mixture_rows, ignore_index=True)


def make_plots(
    summary: pd.DataFrame,
    target_genes: pd.DataFrame,
    mixture_summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(TARGETS), 3, figsize=(17, 4.1 * len(TARGETS)), squeeze=False)
    for row_index, (dataset, tissue, cell_type) in enumerate(TARGETS):
        genes = target_genes.loc[
            (target_genes["dataset"] == dataset)
            & (target_genes["tissue"] == tissue)
            & (target_genes["target_cell_type"] == cell_type)
        ]
        mixtures = mixture_summary.loc[
            (mixture_summary["dataset"] == dataset)
            & (mixture_summary["tissue"] == tissue)
            & (mixture_summary["cell_type"] == cell_type)
        ]
        row = summary.loc[
            (summary["dataset"] == dataset)
            & (summary["tissue"] == tissue)
            & (summary["cell_type"] == cell_type)
        ].iloc[0]

        ax = axes[row_index, 0]
        colors = np.where(genes["is_marker"], "#D55E00", "#B8B8B8")
        ax.scatter(
            genes["log2fc"],
            -np.log10(np.maximum(genes["fdr"], 1e-300)),
            c=colors,
            s=np.where(genes["is_marker"], 18, 9),
            alpha=0.72,
            linewidths=0,
        )
        ax.axvline(1.0, color="#333333", linestyle="--", linewidth=0.8)
        ax.axhline(-math.log10(0.05), color="#333333", linestyle="--", linewidth=0.8)
        ax.set_xlabel("one-vs-rest log2FC")
        ax.set_ylabel("-log10(FDR)")
        ax.set_title(f"DE markers: {int(row.n_target_markers)}")
        ax.grid(alpha=0.2)

        ax = axes[row_index, 1]
        marker_values = genes.loc[genes["is_marker"], "mean_meta_weight"].to_numpy()
        nonmarker_values = genes.loc[~genes["is_marker"], "mean_meta_weight"].to_numpy()
        boxes = ax.boxplot(
            [nonmarker_values, marker_values],
            tick_labels=["Non-marker", "Target marker"],
            patch_artist=True,
            showfliers=False,
        )
        boxes["boxes"][0].set_facecolor("#A7C7E7")
        boxes["boxes"][1].set_facecolor("#E69F00")
        rng = np.random.default_rng(1000 + row_index)
        for position, values, color in (
            (1, nonmarker_values, "#4C78A8"),
            (2, marker_values, "#D55E00"),
        ):
            sampled = values if len(values) <= 250 else rng.choice(values, 250, replace=False)
            ax.scatter(
                np.full(len(sampled), position) + rng.normal(0, 0.045, len(sampled)),
                sampled,
                s=9,
                alpha=0.45,
                color=color,
                linewidths=0,
            )
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Mean MetaSort gene weight")
        ax.set_title(
            f"marker/non-marker={row.marker_to_nonmarker_weight_ratio:.3f}; "
            f"FDR={row.marker_weight_mannwhitney_fdr:.2g}"
        )
        ax.grid(axis="y", alpha=0.2)

        ax = axes[row_index, 2]
        ax.scatter(
            mixtures["truth_proportion"],
            mixtures["marker_mean_weight"],
            s=24,
            alpha=0.7,
            color="#D55E00",
            edgecolors="white",
            linewidths=0.3,
        )
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_xlabel("True target-cell proportion")
        ax.set_ylabel("Mean target-marker weight")
        ax.set_title(
            f"Pearson r={row.truth_vs_marker_weight_pearson_r:.3f}; "
            f"Spearman ρ={row.truth_vs_marker_weight_spearman_r:.3f}"
        )
        ax.grid(alpha=0.2)

        axes[row_index, 0].text(
            -0.28,
            0.5,
            textwrap.fill(f"{dataset} / {tissue}\n{cell_type}", width=34),
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    fig.suptitle(
        "Single-cell DE markers and MetaSort gene-weight assessment",
        fontsize=16,
        fontweight="bold",
        y=1.002,
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "target_marker_weight_assessment.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(plot_dir / "target_marker_weight_assessment.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, output_dir: Path, config: dict[str, object]) -> None:
    lines = [
        "# Signature marker and MetaSort gene-weight analysis",
        "",
        "Marker definition: one-vs-rest Mann–Whitney test on raw single-cell counts, "
        f"BH-FDR < {config['max_fdr']}, log2FC > {config['min_log2fc']}, "
        f"and target-cell detection fraction >= {config['min_pct_in']:.0%}.",
        "MetaSort weights use sqrt-sphere Hessian, convergence_tol=0.01, and no old/new proportion averaging.",
        "Weights are normalized to mean 1 within each mixture; values above 1 are amplified.",
        "",
        "| Dataset / tissue | Rare cell type | Cells | Markers | Marker mean weight | Non-marker mean | Ratio | Marker genes >1 | Top-10% enrichment FDR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} / {row.tissue} | {row.cell_type} | {row.n_target_cells} | "
            f"{row.n_target_markers} | {row.marker_mean_weight:.3f} | "
            f"{row.nonmarker_mean_weight:.3f} | {row.marker_to_nonmarker_weight_ratio:.3f} | "
            f"{row.marker_genes_mean_weight_gt_1}/{row.n_target_markers} | "
            f"{row.top10pct_enrichment_fdr:.2g} |"
        )
    lines.extend(
        [
            "",
            "Interpretation rule: marker amplification is supported when marker mean weight exceeds 1, "
            "the marker/non-marker ratio exceeds 1, and enrichment/paired tests support the difference.",
            "Because the optimization is not cell-type-aware, marker upweighting is an empirical outcome, not an algorithmic constraint.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tissue_pairs = list(dict.fromkeys((dataset, tissue) for dataset, tissue, _ in TARGETS))

    de_parts: list[pd.DataFrame] = []
    assignment_parts: list[pd.DataFrame] = []
    weight_parts: list[pd.DataFrame] = []
    validation_rows: list[dict[str, object]] = []
    for dataset, tissue in tissue_pairs:
        case_dir = args.data_root / dataset / tissue
        print(f"DE {dataset}/{tissue}", flush=True)
        de, assignment = run_de(
            case_dir,
            min_log2fc=args.min_log2fc,
            max_fdr=args.max_fdr,
            min_pct_in=args.min_pct_in,
            pseudocount=args.pseudocount,
        )
        de_parts.append(de)
        assignment_parts.append(assignment)
        weights, max_difference = extract_weights(
            case_dir,
            prediction_root=args.prediction_root,
            weights_dir=args.output_dir / "gene_weight_matrices",
        )
        weight_parts.append(weights)
        validation_rows.append(
            {
                "dataset": dataset,
                "tissue": tissue,
                "max_prediction_abs_difference_vs_saved_run": max_difference,
            }
        )

    de = pd.concat(de_parts, ignore_index=True)
    assignments = pd.concat(assignment_parts, ignore_index=True)
    weights = pd.concat(weight_parts, ignore_index=True)
    summary, target_genes, mixture_summary = analyze_targets(de, weights, args.data_root)

    de.to_csv(args.output_dir / "signature_gene_de_all_celltypes.csv.gz", index=False, compression="gzip")
    assignments.to_csv(args.output_dir / "signature_gene_primary_marker_assignment.csv", index=False)
    assignment_counts = assignments.copy()
    assignment_counts["primary_marker_cell_type"] = assignment_counts[
        "primary_marker_cell_type"
    ].fillna("Unassigned")
    assignment_counts = (
        assignment_counts.groupby(
            ["dataset", "tissue", "primary_marker_cell_type"], as_index=False
        )
        .agg(n_signature_genes=("gene", "size"))
    )
    assignment_counts["fraction_of_signature"] = assignment_counts["n_signature_genes"] / (
        assignment_counts.groupby(["dataset", "tissue"])["n_signature_genes"].transform("sum")
    )
    assignment_counts.to_csv(args.output_dir / "signature_primary_marker_counts.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(args.output_dir / "rerun_prediction_validation.csv", index=False)
    summary.to_csv(args.output_dir / "rare_celltype_marker_weight_summary.csv", index=False)
    target_genes.to_csv(args.output_dir / "rare_celltype_marker_gene_detail.csv", index=False)
    mixture_summary.to_csv(args.output_dir / "rare_celltype_marker_weight_by_mixture.csv", index=False)
    top_markers = (
        target_genes.loc[target_genes["is_marker"]]
        .sort_values(
            ["dataset", "tissue", "target_cell_type", "mean_meta_weight"],
            ascending=[True, True, True, False],
        )
        .groupby(["dataset", "tissue", "target_cell_type"], as_index=False)
        .head(30)
    )
    top_markers.to_csv(args.output_dir / "rare_celltype_top30_markers_by_mean_weight.csv", index=False)
    make_plots(summary, target_genes, mixture_summary, args.output_dir, args.dpi)

    config = {
        "data_root": str(args.data_root),
        "prediction_root": str(args.prediction_root),
        "targets": TARGETS,
        "de_test": "one-vs-rest Mann-Whitney U, greater",
        "min_log2fc": args.min_log2fc,
        "max_fdr": args.max_fdr,
        "min_pct_in": args.min_pct_in,
        "pseudocount": args.pseudocount,
        "metasort": {
            "use_sqrt_sphere_hessian": True,
            "convergence_tol": 0.01,
            "averaging_old_weight": 0,
            "normalize_meta_weight_mean": True,
            "meta_weight_baseline": 1.0,
            "meta_weight_floor": 0.01,
        },
    }
    (args.output_dir / "analysis_config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_report(summary, args.output_dir, config)
    print(summary.to_string(index=False), flush=True)
    print(f"Wrote analysis to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
