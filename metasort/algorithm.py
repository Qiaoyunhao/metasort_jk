from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls


@dataclass
class DWLSConfig:
    convergence_tol: float = 0.005
    max_iter: int = 1000
    averaging_old_weight: int = 4
    cv_repeats: int = 100
    cv_fraction: float = 0.5
    min_weight_floor: float = 1e-12
    random_seed_start: int = 1
    final_weight_max: float = 10.0


@dataclass
class DWLSResult:
    proportions: list[float]
    raw_solution: list[float]
    ols_initial_proportions: list[float]
    iterations: int
    converged: bool
    selected_j: int
    selected_multiplier: float
    change_history: list[float]
    dampened_weight_min: float
    dampened_weight_mean: float
    dampened_weight_max: float
    cell_types: list[str]


def load_bulk_signature_inputs(
    data_root: str | Path,
    mixture_name: str = "Mixture1",
    signature_name: str = "signature.txt",
    bulk_name: str = "bulk.txt",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    root = Path(data_root)
    bulk_df = pd.read_csv(root / bulk_name, sep="\t", index_col=0)
    sig_df = pd.read_csv(root / signature_name, sep="\t", index_col=0)
    common_genes = bulk_df.index.intersection(sig_df.index)
    sc_ref_matrix = sig_df.loc[common_genes].to_numpy(dtype=float)
    bulk_vector = bulk_df.loc[common_genes, mixture_name].to_numpy(dtype=float)
    return sc_ref_matrix, bulk_vector, common_genes.to_list(), sig_df.columns.to_list()


class DWLSSolver:
    def __init__(self, config: DWLSConfig | None = None) -> None:
        self.config = config or DWLSConfig()

    @staticmethod
    def _normalize_solution(x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=float), 0.0, None)
        total = float(np.sum(x))
        if total <= 0:
            raise ValueError("Solution sum is zero after projection.")
        return x / total

    @staticmethod
    def solve_ols_internal(signature: np.ndarray, bulk: np.ndarray) -> np.ndarray:
        x_raw, _ = nnls(np.asarray(signature, dtype=float), np.asarray(bulk, dtype=float))
        return np.asarray(x_raw, dtype=float)

    def solve_ols(self, signature: np.ndarray, bulk: np.ndarray) -> np.ndarray:
        return self._normalize_solution(self.solve_ols_internal(signature, bulk))

    def _compute_scaled_weights(self, signature: np.ndarray, solution: np.ndarray, bulk: np.ndarray) -> np.ndarray:
        mu = np.asarray(signature, dtype=float) @ np.asarray(solution, dtype=float)
        mu = np.clip(mu, self.config.min_weight_floor, None)
        bulk_vec = np.clip(np.asarray(bulk, dtype=float), self.config.min_weight_floor, None)
        denom = 0.5 * (mu + bulk_vec)
        denom = np.clip(denom, self.config.min_weight_floor, None)
        ws = 1.0 / (denom ** 2)
        ws = ws / np.min(ws)
        return ws

    def _apply_dampening(self, ws_scaled: np.ndarray, j: int) -> tuple[np.ndarray, float]:
        multiplier = float(2 ** (j - 1))
        ws_dampened = np.minimum(np.asarray(ws_scaled, dtype=float), multiplier)
        max_weight = float(np.max(ws_dampened))
        if max_weight > 0:
            ws_dampened = ws_dampened * (self.config.final_weight_max / max_weight)
        return ws_dampened, multiplier

    def solve_dampened_wls_j(
        self,
        signature: np.ndarray,
        bulk: np.ndarray,
        gold_standard: np.ndarray,
        j: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        ws_scaled = self._compute_scaled_weights(signature, gold_standard, bulk)
        ws_dampened, multiplier = self._apply_dampening(ws_scaled, j)
        sqrt_w = np.sqrt(ws_dampened)
        a_weighted = np.asarray(signature, dtype=float) * sqrt_w[:, None]
        b_weighted = np.asarray(bulk, dtype=float) * sqrt_w
        solution, _ = nnls(a_weighted, b_weighted)
        return solution, ws_dampened, multiplier

    def find_dampening_constant(
        self,
        signature: np.ndarray,
        bulk: np.ndarray,
        gold_standard: np.ndarray,
    ) -> int:
        cfg = self.config
        ws_scaled = self._compute_scaled_weights(signature, gold_standard, bulk)
        finite_ws = ws_scaled[np.isfinite(ws_scaled)]
        max_weight = float(np.max(finite_ws))
        max_j = max(1, int(np.ceil(np.log2(max_weight))))

        score_by_j: list[float] = []
        for j in range(1, max_j + 1):
            ws_dampened, _ = self._apply_dampening(ws_scaled, j)
            repeat_solutions = []
            for seed in range(cfg.random_seed_start, cfg.random_seed_start + cfg.cv_repeats):
                rng = np.random.default_rng(seed)
                subset_size = max(1, int(len(ws_dampened) * cfg.cv_fraction))
                subset = rng.choice(len(ws_dampened), size=subset_size, replace=False)
                s_sub = np.asarray(signature, dtype=float)[subset, :]
                b_sub = np.asarray(bulk, dtype=float)[subset]
                w_sub = ws_dampened[subset]
                sqrt_w = np.sqrt(w_sub)
                # R code uses lm(...) without nonnegativity for CV scoring.
                coef, *_ = np.linalg.lstsq(s_sub * sqrt_w[:, None], b_sub * sqrt_w, rcond=None)
                coef = np.asarray(coef, dtype=float)
                coef_sum = float(np.sum(coef))
                if abs(coef_sum) < cfg.min_weight_floor:
                    coef = gold_standard.copy()
                else:
                    coef = coef * (float(np.sum(gold_standard)) / coef_sum)
                repeat_solutions.append(coef)

            repeat_matrix = np.vstack(repeat_solutions)
            std_per_celltype = np.std(repeat_matrix, axis=0, ddof=0)
            score_by_j.append(float(np.mean(std_per_celltype ** 2)))

        return int(np.argmin(score_by_j) + 1)

    def solve_dwls(
        self,
        signature: np.ndarray,
        bulk: np.ndarray,
        cell_types: list[str] | None = None,
    ) -> DWLSResult:
        cfg = self.config
        signature = np.asarray(signature, dtype=float)
        bulk = np.asarray(bulk, dtype=float)
        if signature.ndim != 2 or bulk.ndim != 1:
            raise ValueError("signature must be 2D and bulk must be 1D.")
        if signature.shape[0] != bulk.shape[0]:
            raise ValueError("signature and bulk must have the same gene dimension.")

        initial_solution = self.solve_ols_internal(signature, bulk)
        initial_proportions = self._normalize_solution(initial_solution)
        j = self.find_dampening_constant(signature, bulk, initial_solution)

        solution = initial_solution.copy()
        changes: list[float] = []
        converged = False
        final_ws = np.ones(signature.shape[0], dtype=float)
        final_multiplier = float(2 ** (j - 1))

        for iteration in range(cfg.max_iter):
            new_solution, ws_dampened, multiplier = self.solve_dampened_wls_j(signature, bulk, solution, j)
            # Match the R implementation: (new + 4 * old) / 5 by default.
            solution_average = (
                new_solution + cfg.averaging_old_weight * solution
            ) / float(cfg.averaging_old_weight + 1)
            change = float(np.linalg.norm(solution_average - solution))
            changes.append(change)
            solution = solution_average
            final_ws = ws_dampened
            final_multiplier = multiplier
            if change <= cfg.convergence_tol:
                converged = True
                break

        proportions = self._normalize_solution(solution)
        return DWLSResult(
            proportions=proportions.tolist(),
            raw_solution=solution.tolist(),
            ols_initial_proportions=initial_proportions.tolist(),
            iterations=len(changes),
            converged=converged,
            selected_j=j,
            selected_multiplier=final_multiplier,
            change_history=changes,
            dampened_weight_min=float(np.min(final_ws)),
            dampened_weight_mean=float(np.mean(final_ws)),
            dampened_weight_max=float(np.max(final_ws)),
            cell_types=[] if cell_types is None else list(cell_types),
        )

    @staticmethod
    def result_to_dict(result: DWLSResult) -> dict[str, Any]:
        return asdict(result)


def main() -> None:
    data_root = Path(__file__).resolve().parents[1] / "data" / "Fat"
    signature, bulk, _, cell_types = load_bulk_signature_inputs(data_root, mixture_name="Mixture1")
    solver = DWLSSolver()
    result = solver.solve_dwls(signature, bulk, cell_types=cell_types)
    print(result)


if __name__ == "__main__":
    main()
