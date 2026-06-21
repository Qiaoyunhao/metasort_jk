from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _normalize_columns_to_sum_one(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    column_sums = np.sum(matrix, axis=0)
    if np.any(~np.isfinite(column_sums)) or np.any(column_sums <= 0.0):
        raise ValueError(f"{name} columns must have positive finite sums for normalization.")
    return matrix / column_sums[None, :]


def _normalize_vector_to_sum_one(vector: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    total = float(np.sum(vector))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have a positive finite sum for normalization.")
    return vector / total


def _joint_gene_zscore(signature: np.ndarray, bulk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    combined = np.column_stack([np.asarray(signature, dtype=float), np.asarray(bulk, dtype=float)])
    row_means = np.mean(combined, axis=1, keepdims=True)
    row_stds = np.std(combined, axis=1, keepdims=True)
    row_stds = np.where(row_stds > 0.0, row_stds, 1.0)
    zscored = (combined - row_means) / row_stds
    return zscored[:, :-1], zscored[:, -1]


def solve_simplex_constrained_ls(
    signature: np.ndarray,
    bulk: np.ndarray,
    weights: np.ndarray | None = None,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    signature = np.asarray(signature, dtype=float)
    bulk = np.asarray(bulk, dtype=float)
    if signature.ndim != 2 or bulk.ndim != 1:
        raise ValueError("signature must be 2D and bulk must be 1D.")
    if signature.shape[0] != bulk.shape[0]:
        raise ValueError("signature and bulk must have the same gene dimension.")
    if signature.shape[1] == 0:
        raise ValueError("signature must have at least one cell type.")
    if signature.shape[1] == 1:
        return np.ones(1, dtype=float)

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or weights.shape[0] != signature.shape[0]:
            raise ValueError("weights must be 1D with the same gene dimension.")
        sqrt_weights = np.sqrt(np.clip(weights, 1e-12, None))
        signature = signature * sqrt_weights[:, None]
        bulk = bulk * sqrt_weights

    n_cell_types = signature.shape[1]
    hessian = signature.T @ signature
    hessian = 0.5 * (hessian + hessian.T)
    linear = signature.T @ bulk

    def objective(x: np.ndarray) -> float:
        return 0.5 * float(x @ hessian @ x) - float(linear @ x)

    def gradient(x: np.ndarray) -> np.ndarray:
        return hessian @ x - linear

    def normalize_start(x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=float), 0.0, None)
        total = float(np.sum(x))
        if total <= 0.0:
            return np.full(n_cell_types, 1.0 / n_cell_types, dtype=float)
        return x / total

    def project_to_simplex(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        sorted_x = np.sort(x)[::-1]
        cssv = np.cumsum(sorted_x) - 1.0
        indices = np.arange(1, len(x) + 1)
        valid = sorted_x - cssv / indices > 0.0
        if not np.any(valid):
            return np.full(len(x), 1.0 / len(x), dtype=float)
        rho = indices[valid][-1]
        theta = cssv[valid][-1] / rho
        projected = np.maximum(x - theta, 0.0)
        return projected / np.sum(projected)

    starts: list[np.ndarray] = []
    if initial is not None:
        starts.append(normalize_start(np.asarray(initial, dtype=float)))
    starts.append(np.full(n_cell_types, 1.0 / n_cell_types, dtype=float))
    best_vertex = np.zeros(n_cell_types, dtype=float)
    best_vertex[int(np.argmin(np.diag(hessian) * 0.5 - linear))] = 1.0
    starts.append(best_vertex)
    try:
        equality_solution = np.linalg.solve(
            np.block(
                [
                    [hessian, np.ones((n_cell_types, 1), dtype=float)],
                    [np.ones((1, n_cell_types), dtype=float), np.zeros((1, 1), dtype=float)],
                ]
            ),
            np.concatenate([linear, np.ones(1, dtype=float)]),
        )[:n_cell_types]
        starts.append(project_to_simplex(equality_solution))
    except np.linalg.LinAlgError:
        pass

    unique_starts = []
    for start in starts:
        if not any(np.allclose(start, seen, atol=1e-10, rtol=0.0) for seen in unique_starts):
            unique_starts.append(start)

    best_solution: np.ndarray | None = None
    best_objective = float("inf")
    last_message = ""
    for start in unique_starts:
        result = minimize(
            objective,
            start,
            jac=gradient,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_cell_types,
            constraints=[
                {
                    "type": "eq",
                    "fun": lambda x: float(np.sum(x) - 1.0),
                    "jac": lambda x: np.ones_like(x),
                }
            ],
            options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
        )
        last_message = str(result.message)
        if np.all(np.isfinite(result.x)):
            candidate = project_to_simplex(result.x)
            candidate_objective = objective(candidate)
            if candidate_objective < best_objective:
                best_solution = candidate
                best_objective = candidate_objective
        if result.success and best_solution is not None:
            return best_solution

    if best_solution is None:
        best_solution = unique_starts[0]

    eigvals = np.linalg.eigvalsh(hessian)
    step = 1.0 / max(float(np.max(eigvals)), 1e-12)
    solution = best_solution.copy()
    for _ in range(10000):
        next_solution = project_to_simplex(solution - step * gradient(solution))
        if np.linalg.norm(next_solution - solution) <= 1e-10:
            return next_solution
        solution = next_solution

    if np.all(np.isfinite(solution)):
        return project_to_simplex(solution)
    raise RuntimeError(f"Simplex-constrained least squares failed: {last_message}")


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
    sc_ref_matrix = _normalize_columns_to_sum_one(
        sig_df.loc[common_genes].to_numpy(dtype=float),
        name="signature",
    )
    bulk_vector = _normalize_vector_to_sum_one(
        bulk_df.loc[common_genes, mixture_name].to_numpy(dtype=float),
        name="bulk",
    )
    sc_ref_matrix, bulk_vector = _joint_gene_zscore(sc_ref_matrix, bulk_vector)
    return sc_ref_matrix, bulk_vector, common_genes.to_list(), sig_df.columns.to_list()
