from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .algorithm import DWLSConfig, DWLSResult, DWLSSolver, load_bulk_signature_inputs, solve_simplex_constrained_ls


@dataclass
class MetaSortConfig(DWLSConfig):
    lambda_hessian: float = 1.0
    lambda_avg_gradient: float = 0.0
    lambda_residual: float = 0.005
    lambda_gene_importance: float = 0.0
    lambda3: float = 0.01
    lambda4: float = 0.001
    kappa: float = 0.1
    hessian_epsilon: float = 1e-8
    use_sqrt_sphere_hessian: bool = False
    gene_importance_epsilon: float = 1e-12
    meta_weight_floor: float = 1e-2
    meta_lbfgs_lr: float = 1.0
    meta_lbfgs_max_iter: int = 100
    meta_lbfgs_rounds: int = 3
    meta_tol: float = 1e-8
    avg_gradient_n_perturbations: int = 20
    avg_gradient_perturb_scale: float = 0.02
    avg_gradient_seed: int = 0
    use_dwls_base_weight: bool = False
    use_log_base_weight: bool = True
    normalize_base_weight_mean: bool = True
    normalize_meta_weight_mean: bool = True
    print_info: bool = False


@dataclass
class MetaSortResult(DWLSResult):
    meta_weight_min: float = 1.0
    meta_weight_mean: float = 1.0
    meta_weight_max: float = 1.0
    total_weight_min: float = 1.0
    total_weight_mean: float = 1.0
    total_weight_max: float = 1.0
    meta_weight_metrics: dict[str, float] | None = None
    meta_config: dict[str, float | int | bool] | None = None


class MetaSortSolver(DWLSSolver):
    def __init__(self, config: MetaSortConfig | None = None) -> None:
        super().__init__(config=config or MetaSortConfig())
        self.config: MetaSortConfig

    @staticmethod
    def _build_simplex_basis(n_cell_types: int) -> torch.Tensor:
        basis = torch.zeros((n_cell_types, n_cell_types - 1), dtype=torch.double)
        basis[:-1, :] = torch.eye(n_cell_types - 1, dtype=torch.double)
        basis[-1, :] = -1.0
        q, _ = torch.linalg.qr(basis, mode="reduced")
        return q

    @staticmethod
    def _build_sphere_tangent_basis(point: torch.Tensor) -> torch.Tensor:
        point = point / torch.clamp(torch.linalg.norm(point), min=1e-12)
        n_cell_types = int(point.shape[0])
        if n_cell_types <= 1:
            return torch.empty(
                (n_cell_types, 0),
                dtype=point.dtype,
                device=point.device,
            )

        anchor = int(torch.argmax(torch.abs(point)).detach().cpu().item())
        tangent_indices = [idx for idx in range(n_cell_types) if idx != anchor]
        seed = torch.eye(n_cell_types, dtype=point.dtype, device=point.device)[:, tangent_indices]
        seed = seed - point[:, None] * (point @ seed)[None, :]
        basis, _ = torch.linalg.qr(seed, mode="reduced")
        return basis

    @staticmethod
    def _hessian_eigvals(projected_hessian: torch.Tensor, epsilon: float) -> torch.Tensor:
        if projected_hessian.shape[0] == 0:
            return torch.ones(1, dtype=projected_hessian.dtype, device=projected_hessian.device)
        projected_hessian = 0.5 * (projected_hessian + projected_hessian.T)
        eigvals = torch.linalg.eigvalsh(projected_hessian)
        return torch.clamp(eigvals, min=epsilon)

    @staticmethod
    def _normalize_meta_weights_np(
        weights: np.ndarray,
        floor: float,
        normalize_mean: bool,
    ) -> np.ndarray:
        if floor <= 0.0 or floor > 1.0:
            raise ValueError("meta_weight_floor must be in (0, 1].")
        weights = np.clip(np.asarray(weights, dtype=float), floor, None)
        if not normalize_mean:
            return weights

        low = 0.0
        high = 1.0 / max(float(np.mean(weights)), 1e-300)
        for _ in range(80):
            mid = 0.5 * (low + high)
            candidate = np.maximum(floor, mid * weights)
            if float(np.mean(candidate)) > 1.0:
                high = mid
            else:
                low = mid
        return np.maximum(floor, high * weights)

    @staticmethod
    def _normalize_meta_weights_torch(
        weights: torch.Tensor,
        floor: float,
        normalize_mean: bool,
    ) -> torch.Tensor:
        if floor <= 0.0 or floor > 1.0:
            raise ValueError("meta_weight_floor must be in (0, 1].")

        floor_tensor = torch.tensor(floor, dtype=weights.dtype, device=weights.device)
        weights = torch.clamp(weights, min=floor)
        if not normalize_mean:
            return weights

        low = torch.zeros((), dtype=weights.dtype, device=weights.device)
        high = 1.0 / torch.clamp(torch.mean(weights), min=1e-300)
        for _ in range(40):
            mid = 0.5 * (low + high)
            candidate = torch.maximum(floor_tensor, mid * weights)
            mean_too_high = torch.mean(candidate) > 1.0
            high = torch.where(mean_too_high, mid, high)
            low = torch.where(mean_too_high, low, mid)
        return torch.maximum(floor_tensor, high * weights)

    @staticmethod
    def _sample_local_perturbation_proportions(
        proportions: np.ndarray,
        n_samples: int,
        perturb_scale: float,
        seed: int,
    ) -> np.ndarray:
        center = np.clip(np.asarray(proportions, dtype=float), 1e-12, None)
        center = center / np.sum(center)
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(max(1, int(n_samples))):
            proposal = center + rng.normal(0.0, perturb_scale, size=center.shape[0])
            proposal = np.clip(proposal, 1e-12, None)
            proposal = proposal / np.sum(proposal)
            samples.append(proposal)
        return np.vstack(samples)

    @staticmethod
    def _compute_gene_importance_scores(
        signature: np.ndarray,
        epsilon: float,
    ) -> np.ndarray:
        signature = np.asarray(signature, dtype=float)
        gene_mean = np.mean(signature, axis=1)
        gene_var = np.var(signature, axis=1)
        raw_score = gene_var / np.clip(gene_mean, epsilon, None)
        raw_score = np.clip(raw_score, 0.0, None)
        max_score = float(np.max(raw_score))
        if max_score <= 0.0:
            return np.ones(signature.shape[0], dtype=float)
        return raw_score / max_score

    def optimize_meta_weights(
        self,
        signature: np.ndarray,
        bulk: np.ndarray,
        proportions: np.ndarray,
        base_weights: np.ndarray,
        prev_meta_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        cfg = self.config
        a_tensor = torch.tensor(np.asarray(signature, dtype=float), dtype=torch.double)
        bulk_tensor = torch.tensor(np.asarray(bulk, dtype=float), dtype=torch.double)
        p_tensor = torch.tensor(self._normalize_solution(proportions), dtype=torch.double)
        base_tensor = torch.tensor(
            np.clip(np.asarray(base_weights, dtype=float), cfg.min_weight_floor, None),
            dtype=torch.double,
        )
        prev_meta = (
            np.ones(signature.shape[0], dtype=float)
            if prev_meta_weights is None
            else np.asarray(prev_meta_weights, dtype=float)
        )
        prev_meta = self._normalize_meta_weights_np(
            prev_meta,
            floor=cfg.meta_weight_floor,
            normalize_mean=cfg.normalize_meta_weight_mean,
        )
        prev_meta_tensor = torch.tensor(prev_meta, dtype=torch.double)
        simplex_basis = self._build_simplex_basis(signature.shape[1])
        sqrt_sphere_basis = None
        sqrt_proportions = None
        if cfg.use_sqrt_sphere_hessian:
            sqrt_proportions = torch.sqrt(torch.clamp(p_tensor, min=cfg.hessian_epsilon))
            sqrt_proportions = sqrt_proportions / torch.clamp(
                torch.linalg.norm(sqrt_proportions),
                min=cfg.hessian_epsilon,
            )
            sqrt_sphere_basis = self._build_sphere_tangent_basis(sqrt_proportions)
        perturb_props = self._sample_local_perturbation_proportions(
            proportions=self._normalize_solution(proportions),
            n_samples=cfg.avg_gradient_n_perturbations,
            perturb_scale=cfg.avg_gradient_perturb_scale,
            seed=cfg.avg_gradient_seed,
        )
        perturb_bulks_tensor = torch.tensor(
            np.asarray(signature, dtype=float) @ perturb_props.T,
            dtype=torch.double,
        )
        gene_importance_tensor = torch.tensor(
            self._compute_gene_importance_scores(signature, cfg.gene_importance_epsilon),
            dtype=torch.double,
        )

        meta_tensor = prev_meta_tensor.clone().detach().requires_grad_(True)
        best_meta = prev_meta.copy()
        best_loss = float("inf")
        best_metrics: dict[str, float] = {}
        previous_round_loss: float | None = None
        previous_round_meta = prev_meta_tensor.clone()

        def loss_torch() -> tuple[torch.Tensor, dict[str, float]]:
            meta_clamped = self._normalize_meta_weights_torch(
                meta_tensor,
                floor=cfg.meta_weight_floor,
                normalize_mean=cfg.normalize_meta_weight_mean,
            )
            total_weights = base_tensor * meta_clamped
            h_p = a_tensor.T @ (total_weights[:, None] * a_tensor)
            if cfg.use_sqrt_sphere_hessian:
                if sqrt_proportions is None or sqrt_sphere_basis is None:
                    raise RuntimeError("sqrt-sphere Hessian basis was not initialized.")
                h_sphere = 4.0 * (
                    sqrt_proportions[:, None]
                    * h_p
                    * sqrt_proportions[None, :]
                )
                projected_hessian = sqrt_sphere_basis.T @ h_sphere @ sqrt_sphere_basis
            else:
                projected_hessian = simplex_basis.T @ h_p @ simplex_basis
            eigvals = self._hessian_eigvals(projected_hessian, cfg.hessian_epsilon)

            hessian_loss = -cfg.lambda_hessian * torch.sum(torch.log(eigvals + cfg.hessian_epsilon))
            pred_bulk = a_tensor @ p_tensor
            residual_vector = pred_bulk - bulk_tensor
            residuals = pred_bulk[:, None] - perturb_bulks_tensor
            weighted_residuals = total_weights[:, None] * residuals
            gradients = 2.0 * (a_tensor.T @ weighted_residuals)
            mean_gradient = torch.mean(gradients, dim=1)
            projected_mean_gradient = simplex_basis.T @ mean_gradient
            avg_gradient_loss = cfg.lambda_avg_gradient * torch.sum(projected_mean_gradient ** 2)
            residual_penalty = cfg.lambda_residual * torch.sum((residual_vector ** 2) * total_weights)
            meta_diff = torch.norm(meta_clamped - torch.ones_like(meta_clamped))
            reg_strength = cfg.lambda3 / (1.0 + torch.exp(-meta_diff / cfg.kappa)) + cfg.lambda4
            reg_loss = reg_strength * torch.sum((meta_clamped - 1.0) ** 2)
            gene_importance_loss = -cfg.lambda_gene_importance * torch.sum(meta_clamped * gene_importance_tensor)
            total_loss = hessian_loss + avg_gradient_loss + residual_penalty + reg_loss + gene_importance_loss
            metrics = {
                "total_loss": float(total_loss.detach().item()),
                "hessian_loss": float(hessian_loss.detach().item()),
                "avg_gradient_loss": float(avg_gradient_loss.detach().item()),
                "residual_penalty": float(residual_penalty.detach().item()),
                "gene_importance_loss": float(gene_importance_loss.detach().item()),
                "reg_loss": float(reg_loss.detach().item()),
                "min_projected_hessian_eig": float(torch.min(eigvals).detach().item()),
                "mean_projected_hessian_eig": float(torch.mean(eigvals).detach().item()),
                "max_projected_hessian_eig": float(torch.max(eigvals).detach().item()),
            }
            if cfg.use_sqrt_sphere_hessian:
                metrics.update(
                    {
                        "min_sqrt_sphere_hessian_eig": metrics["min_projected_hessian_eig"],
                        "mean_sqrt_sphere_hessian_eig": metrics["mean_projected_hessian_eig"],
                        "max_sqrt_sphere_hessian_eig": metrics["max_projected_hessian_eig"],
                    }
                )
            else:
                metrics.update(
                    {
                        "min_simplex_hessian_eig": metrics["min_projected_hessian_eig"],
                        "mean_simplex_hessian_eig": metrics["mean_projected_hessian_eig"],
                        "max_simplex_hessian_eig": metrics["max_projected_hessian_eig"],
                    }
                )
            return total_loss, metrics

        for lbfgs_round in range(cfg.meta_lbfgs_rounds):
            optimizer = torch.optim.LBFGS(
                [meta_tensor],
                lr=cfg.meta_lbfgs_lr,
                max_iter=cfg.meta_lbfgs_max_iter,
                line_search_fn="strong_wolfe",
                tolerance_grad=cfg.meta_tol,
                tolerance_change=cfg.meta_tol,
            )

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                total_loss, _ = loss_torch()
                total_loss.backward()
                return total_loss

            optimizer.step(closure)
            with torch.no_grad():
                normalized_meta = self._normalize_meta_weights_torch(
                    meta_tensor.data,
                    floor=cfg.meta_weight_floor,
                    normalize_mean=cfg.normalize_meta_weight_mean,
                )
                meta_tensor.data.copy_(normalized_meta)
                total_loss, metrics = loss_torch()
                round_loss = float(total_loss.detach().item())
                meta_now = meta_tensor.detach().cpu().numpy().copy()

                if round_loss < best_loss:
                    best_loss = round_loss
                    best_meta = meta_now.copy()
                    best_metrics = metrics

                rel_meta_change = float(
                    torch.linalg.norm(meta_tensor.detach() - previous_round_meta)
                    / max(torch.linalg.norm(previous_round_meta).item(), 1e-12)
                )
                rel_loss_change = float("inf") if previous_round_loss is None else abs(round_loss - previous_round_loss) / max(
                    abs(previous_round_loss), 1e-12
                )
                previous_round_meta = meta_tensor.detach().clone()
                previous_round_loss = round_loss

                if cfg.print_info:
                    print(
                        f"meta_round={lbfgs_round} total={round_loss:.6e} "
                        f"hessian={metrics['hessian_loss']:.6e} "
                        f"avg_grad={metrics['avg_gradient_loss']:.6e} "
                        f"resid={metrics['residual_penalty']:.6e} "
                        f"importance={metrics['gene_importance_loss']:.6e} "
                        f"reg={metrics['reg_loss']:.6e}"
                    )
                if rel_meta_change < cfg.meta_tol and rel_loss_change < cfg.meta_tol:
                    break

        return best_meta, best_metrics

    def solve_meta_dampened_wls_j(
        self,
        signature: np.ndarray,
        bulk: np.ndarray,
        gold_standard: np.ndarray,
        j: int,
        prev_meta_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, float]]:
        if self.config.use_dwls_base_weight:
            ws_scaled = self._compute_scaled_weights(signature, gold_standard, bulk)
            ws_dampened, multiplier = self._apply_dampening(ws_scaled, j)
            base_weights = np.log1p(ws_dampened) if self.config.use_log_base_weight else ws_dampened.copy()
            base_weights = np.clip(base_weights, self.config.min_weight_floor, None)
            if self.config.normalize_base_weight_mean:
                base_weights = base_weights / np.mean(base_weights)
        else:
            ws_dampened = np.ones(signature.shape[0], dtype=float)
            multiplier = 1.0
            base_weights = np.ones(signature.shape[0], dtype=float)
        meta_weights, meta_metrics = self.optimize_meta_weights(
            signature=signature,
            bulk=bulk,
            proportions=gold_standard,
            base_weights=base_weights,
            prev_meta_weights=prev_meta_weights,
        )
        total_weights = np.clip(base_weights * meta_weights, self.config.min_weight_floor, None)
        solution = solve_simplex_constrained_ls(
            signature,
            bulk,
            weights=total_weights,
            initial=gold_standard,
        )
        return solution, base_weights, meta_weights, multiplier, meta_metrics

    def solve_metasort(
        self,
        signature: np.ndarray,
        bulk: np.ndarray,
        cell_types: list[str] | None = None,
    ) -> MetaSortResult:
        cfg = self.config
        signature = np.asarray(signature, dtype=float)
        bulk = np.asarray(bulk, dtype=float)
        initial_solution = self.solve_ols_internal(signature, bulk)
        initial_proportions = self._normalize_solution(initial_solution)
        j = self.find_dampening_constant(signature, bulk, initial_solution) if cfg.use_dwls_base_weight else 1

        solution = initial_solution.copy()
        meta_weights = np.ones(signature.shape[0], dtype=float)
        changes: list[float] = []
        converged = False
        final_ws = np.ones(signature.shape[0], dtype=float)
        final_meta = np.ones(signature.shape[0], dtype=float)
        final_multiplier = float(2 ** (j - 1))
        final_meta_metrics: dict[str, float] = {}

        for _ in range(cfg.max_iter):
            new_solution, ws_dampened, meta_step, multiplier, meta_metrics = self.solve_meta_dampened_wls_j(
                signature=signature,
                bulk=bulk,
                gold_standard=solution,
                j=j,
                prev_meta_weights=meta_weights,
            )
            solution_average = (
                new_solution + cfg.averaging_old_weight * solution
            ) / float(cfg.averaging_old_weight + 1)
            change = float(np.linalg.norm(solution_average - solution))
            changes.append(change)
            solution = solution_average
            meta_weights = meta_step
            final_ws = ws_dampened
            final_meta = meta_step
            final_multiplier = multiplier
            final_meta_metrics = meta_metrics
            if change <= cfg.convergence_tol:
                converged = True
                break

        proportions = self._normalize_solution(solution)
        total_weights = np.clip(final_ws * final_meta, cfg.min_weight_floor, None)
        return MetaSortResult(
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
            meta_weight_min=float(np.min(final_meta)),
            meta_weight_mean=float(np.mean(final_meta)),
            meta_weight_max=float(np.max(final_meta)),
            total_weight_min=float(np.min(total_weights)),
            total_weight_mean=float(np.mean(total_weights)),
            total_weight_max=float(np.max(total_weights)),
            meta_weight_metrics=final_meta_metrics,
            meta_config=asdict(cfg),
        )

    @staticmethod
    def result_to_dict(result: MetaSortResult) -> dict[str, Any]:
        return asdict(result)

    def solve(
        self,
        signature: np.ndarray,
        bulk: np.ndarray,
        cell_types: list[str] | None = None,
    ) -> MetaSortResult:
        return self.solve_metasort(signature, bulk, cell_types=cell_types)


def main() -> None:
    data_root = Path(__file__).resolve().parents[1] / "data" / "Fat"
    signature, bulk, _, cell_types = load_bulk_signature_inputs(data_root, mixture_name="Mixture1")
    solver = MetaSortSolver()
    result = solver.solve(signature, bulk, cell_types=cell_types)
    print(result)


if __name__ == "__main__":
    main()
