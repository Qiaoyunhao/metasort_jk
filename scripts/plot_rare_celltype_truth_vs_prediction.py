from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


METHODS = ("MetaSort_jk", "DWLS", "MuSic", "LinDeconSeq", "bayesprism", "NNLS", "HiDecon")
COLORS = {
    "MetaSort_jk": "#D55E00",
    "DWLS": "#0072B2",
    "MuSic": "#009E73",
    "LinDeconSeq": "#CC79A7",
    "bayesprism": "#E69F00",
    "NNLS": "#56B4E9",
    "HiDecon": "#666666",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def correlations(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float, float]:
    if len(truth) < 2 or np.std(truth) == 0.0 or np.std(prediction) == 0.0:
        return np.nan, np.nan, np.nan, np.nan
    pearson = pearsonr(truth, prediction)
    spearman = spearmanr(truth, prediction)
    return float(pearson.statistic), float(pearson.pvalue), float(spearman.statistic), float(spearman.pvalue)


def correlation_row(group: pd.DataFrame) -> dict[str, float | int]:
    truth = group["truth"].to_numpy(dtype=float)
    prediction = group["prediction"].to_numpy(dtype=float)
    pearson_r, pearson_p, spearman_r, spearman_p = correlations(truth, prediction)
    error = prediction - truth
    return {
        "n": len(group),
        "mean_truth": float(np.mean(truth)),
        "mean_prediction": float(np.mean(prediction)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def draw_panel(ax: plt.Axes, frame: pd.DataFrame, method: str, limit: float, show_labels: bool) -> None:
    method_frame = frame.loc[frame["method"] == method]
    truth = method_frame["truth"].to_numpy(dtype=float)
    prediction = method_frame["prediction"].to_numpy(dtype=float)
    pearson_r, _, spearman_r, _ = correlations(truth, prediction)
    ax.scatter(
        truth,
        prediction,
        s=23,
        alpha=0.72,
        color=COLORS[method],
        edgecolors="white",
        linewidths=0.3,
    )
    ax.plot([0, limit], [0, limit], linestyle="--", color="#222222", linewidth=1.0, alpha=0.75)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{method}\nPearson r={pearson_r:.3f}; Spearman ρ={spearman_r:.3f}", fontsize=9)
    ax.grid(True, linewidth=0.45, alpha=0.25)
    if show_labels:
        ax.set_xlabel("True proportion")
        ax.set_ylabel("Predicted proportion")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(args.analysis_dir / "celltype_mixture_method_detail.csv")
    winners = pd.read_csv(args.analysis_dir / "low_mean_proportion_celltype_winners.csv")
    winners = winners.loc[winners["metasort_best"].astype(bool)].copy()
    if winners.empty:
        raise ValueError("No MetaSort_jk low-mean-proportion winners were found")

    selected_parts: list[pd.DataFrame] = []
    correlation_rows: list[dict[str, object]] = []
    for winner in winners.itertuples(index=False):
        frame = detail.loc[
            (detail["dataset"] == winner.dataset)
            & (detail["tissue"] == winner.tissue)
            & (detail["cell_type"] == winner.cell_type)
        ].copy()
        selected_parts.append(frame)
        for method in METHODS:
            group = frame.loc[frame["method"] == method]
            correlation_rows.append(
                {
                    "dataset": winner.dataset,
                    "tissue": winner.tissue,
                    "cell_type": winner.cell_type,
                    "method": method,
                    **correlation_row(group),
                }
            )

        limit = max(0.05, float(frame[["truth", "prediction"]].to_numpy().max())) * 1.05
        fig, axes = plt.subplots(2, 4, figsize=(14, 7.3), constrained_layout=True)
        for ax, method in zip(axes.flat, METHODS):
            draw_panel(ax, frame, method, limit, show_labels=True)
        axes.flat[-1].axis("off")
        fig.suptitle(
            f"{winner.dataset} / {winner.tissue}\n{winner.cell_type}",
            fontsize=14,
            fontweight="bold",
        )
        output_name = f"{safe_name(winner.dataset)}__{safe_name(winner.tissue)}__{safe_name(winner.cell_type)}"
        fig.savefig(args.output_dir / f"{output_name}.png", dpi=args.dpi, bbox_inches="tight")
        fig.savefig(args.output_dir / f"{output_name}.pdf", bbox_inches="tight")
        plt.close(fig)

    n_rows = len(winners)
    fig, axes = plt.subplots(n_rows, len(METHODS), figsize=(24, 3.55 * n_rows), squeeze=False)
    for row_index, winner in enumerate(winners.itertuples(index=False)):
        frame = selected_parts[row_index]
        limit = max(0.05, float(frame[["truth", "prediction"]].to_numpy().max())) * 1.05
        for column_index, method in enumerate(METHODS):
            ax = axes[row_index, column_index]
            draw_panel(ax, frame, method, limit, show_labels=False)
            if row_index == n_rows - 1:
                ax.set_xlabel("True proportion")
            if column_index == 0:
                ax.set_ylabel(
                    textwrap.fill(f"{winner.dataset} / {winner.tissue}\n{winner.cell_type}", width=32)
                    + "\nPredicted proportion",
                    fontsize=9,
                )
    fig.suptitle(
        "Low-mean-proportion cell types won by MetaSort_jk: truth vs prediction",
        fontsize=16,
        fontweight="bold",
        y=1.002,
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "all_selected_celltypes_truth_vs_prediction.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.output_dir / "all_selected_celltypes_truth_vs_prediction.pdf", bbox_inches="tight")
    plt.close(fig)

    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(args.output_dir / "correlations_all_methods.csv", index=False)
    correlations.loc[correlations["method"] == "MetaSort_jk"].to_csv(
        args.output_dir / "correlations_metasort_jk.csv", index=False
    )
    print(correlations.loc[correlations["method"] == "MetaSort_jk"].to_string(index=False))
    print(f"Wrote plots and correlations to {args.output_dir}")


if __name__ == "__main__":
    main()
