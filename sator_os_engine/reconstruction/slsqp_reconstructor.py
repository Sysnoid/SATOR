from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize


def _sum_constraint_factory(n_ingredients: int, target_sum: float):
    def sum_constraint(x: np.ndarray) -> float:
        if n_ingredients <= 0:
            return 0.0
        return float(np.sum(x[:n_ingredients]) - target_sum)

    return sum_constraint


def _combine_bounds(
    ingredient_bounds: list[list[float]], parameter_bounds: list[list[float]]
) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    bounds.extend([(float(a), float(b)) for a, b in ingredient_bounds])
    bounds.extend([(float(a), float(b)) for a, b in parameter_bounds])
    return bounds


def reconstruct(
    target_encoded: np.ndarray,
    encoder_components: np.ndarray,
    encoder_mean: np.ndarray | None,
    ingredient_bounds: list[list[float]],
    parameter_bounds: list[list[float]],
    n_ingredients: int,
    target_precision: float = 1e-7,
    sum_target: float = 1.0,
    ratio_constraints: list[dict[str, float]] | None = None,
    ingredient_names: list[str] | None = None,
    parameter_names: list[str] | None = None,
) -> dict[str, Any]:
    dim_x = len(ingredient_bounds) + len(parameter_bounds)
    components = np.array(encoder_components, dtype=float)
    mean = np.array(encoder_mean, dtype=float) if encoder_mean is not None else np.zeros(dim_x)

    def encoder_func(x: np.ndarray) -> np.ndarray:
        x2d = np.atleast_2d(x)
        z = (x2d - mean) @ components.T
        return z.squeeze()

    bounds = _combine_bounds(ingredient_bounds, parameter_bounds)
    # Initial guess: mid of bounds
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    x0 = (lb + ub) / 2.0
    if n_ingredients > 0:
        s = np.sum(x0[:n_ingredients])
        if s > 0:
            x0[:n_ingredients] = x0[:n_ingredients] / s * float(sum_target)

    def objective(x: np.ndarray) -> float:
        enc = encoder_func(x)
        return float(np.linalg.norm(enc - target_encoded))

    constraints = []
    if n_ingredients > 0:
        sum_con = _sum_constraint_factory(n_ingredients, float(sum_target))
        constraints.append(NonlinearConstraint(sum_con, 0.0, 0.0))
    # Ratio constraints: x_i/x_j in [min_ratio, max_ratio] -> linear: x_i - max_ratio*x_j <= 0 and -x_i + min_ratio*x_j <= 0
    if ratio_constraints:
        for rc in ratio_constraints:
            i = int(rc.get("i", -1))
            j = int(rc.get("j", -1))
            if i < 0 or j < 0 or i == j:
                continue
            min_ratio = rc.get("min_ratio")
            max_ratio = rc.get("max_ratio")
            n = dim_x
            if max_ratio is not None:
                A = np.zeros((1, n))
                A[0, i] = 1.0
                A[0, j] = -float(max_ratio)
                constraints.append(LinearConstraint(A, -np.inf, 0.0))
            if min_ratio is not None:
                A = np.zeros((1, n))
                A[0, i] = -1.0
                A[0, j] = float(min_ratio)
                constraints.append(LinearConstraint(A, -np.inf, 0.0))

    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"ftol": target_precision}
    )

    # SLSQP is known to occasionally report success while the final x drifts
    # slightly outside the declared bounds when combined with equality +
    # inequality constraints. Enforce hard constraints post-hoc so callers
    # never see a "successful" solution that silently violates bounds, sum
    # or ratio. If the post-processed solution still violates a hard
    # constraint, we honestly flip ``success`` to False.
    final_solution = np.clip(result.x, lb, ub)
    if n_ingredients > 0 and float(sum_target) > 0.0:
        s = float(np.sum(final_solution[:n_ingredients]))
        if s > 0:
            scale = float(sum_target) / s
            final_solution[:n_ingredients] = np.clip(
                final_solution[:n_ingredients] * scale, lb[:n_ingredients], ub[:n_ingredients]
            )
            s2 = float(np.sum(final_solution[:n_ingredients]))
            if abs(s2 - float(sum_target)) > 1e-6:
                diff = float(sum_target) - s2
                headroom = ub[:n_ingredients] - final_solution[:n_ingredients] if diff > 0 else final_solution[:n_ingredients] - lb[:n_ingredients]
                headroom_sum = float(np.sum(headroom))
                if headroom_sum > 1e-12:
                    final_solution[:n_ingredients] = final_solution[:n_ingredients] + diff * headroom / headroom_sum

    feasible = bool(result.success)
    if np.any(final_solution < lb - 1e-6) or np.any(final_solution > ub + 1e-6):
        feasible = False
    if n_ingredients > 0:
        s_check = float(np.sum(final_solution[:n_ingredients]))
        if abs(s_check - float(sum_target)) > 1e-4:
            feasible = False
    if ratio_constraints:
        for rc in ratio_constraints:
            i = int(rc.get("i", -1))
            j = int(rc.get("j", -1))
            if i < 0 or j < 0 or i == j:
                continue
            denom = max(float(final_solution[j]), 1e-12)
            ratio = float(final_solution[i]) / denom
            mn = rc.get("min_ratio")
            mx = rc.get("max_ratio")
            if mn is not None and ratio < float(mn) - 1e-4:
                feasible = False
            if mx is not None and ratio > float(mx) + 1e-4:
                feasible = False

    final_error = objective(final_solution)

    out: dict[str, Any] = {
        "success": feasible,
        "solution": final_solution.tolist(),
        "ingredients": final_solution[:n_ingredients].tolist() if n_ingredients > 0 else [],
        "parameters": final_solution[n_ingredients:].tolist() if n_ingredients < len(final_solution) else [],
        "final_error": final_error,
        "iterations": int(getattr(result, "nit", 0)),
        "method": "SLSQP_Constrained",
    }
    if (
        ingredient_names is not None
        and parameter_names is not None
        and len(ingredient_names) == n_ingredients
        and len(parameter_names) == len(parameter_bounds)
    ):
        by_name: dict[str, float] = {}
        for name, val in zip(ingredient_names, final_solution[:n_ingredients], strict=True):
            by_name[str(name)] = float(val)
        for name, val in zip(parameter_names, final_solution[n_ingredients:], strict=True):
            by_name[str(name)] = float(val)
        out["solution_by_name"] = by_name
    return out
