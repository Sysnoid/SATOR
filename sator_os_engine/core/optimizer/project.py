"""Euclidean projection onto bounds, sum groups, and ratio inequalities."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, minimize

_FEAS_ATOL = 1e-6


def _ratio_constraints(ratio_constraints: list[dict[str, Any]] | None, dim: int) -> list[LinearConstraint]:
    cons: list[LinearConstraint] = []
    for rc in ratio_constraints or []:
        try:
            i = int(rc.get("i", -1))
            j = int(rc.get("j", -1))
        except (TypeError, ValueError):
            continue
        if i < 0 or j < 0 or i == j or i >= dim or j >= dim:
            continue
        if rc.get("max_ratio") is not None:
            a = np.zeros((1, dim))
            a[0, i] = 1.0
            a[0, j] = -float(rc["max_ratio"])
            cons.append(LinearConstraint(a, -np.inf, 0.0))
        if rc.get("min_ratio") is not None:
            a = np.zeros((1, dim))
            a[0, i] = -1.0
            a[0, j] = float(rc["min_ratio"])
            cons.append(LinearConstraint(a, -np.inf, 0.0))
    return cons


def _sum_constraints(sum_groups: list[dict[str, Any]] | None, dim: int) -> list[LinearConstraint]:
    cons: list[LinearConstraint] = []
    for group in sum_groups or []:
        idxs = [int(i) for i in group.get("indices", []) if 0 <= int(i) < dim]
        if not idxs:
            continue
        target = float(group.get("target", group.get("target_sum", 1.0)))
        a = np.zeros((1, dim))
        for i in idxs:
            a[0, i] = 1.0
        cons.append(LinearConstraint(a, target, target))
    return cons


def is_feasible(
    x: np.ndarray,
    bounds: list[tuple[float, float]],
    sum_groups: list[dict[str, Any]] | None = None,
    ratio_constraints: list[dict[str, Any]] | None = None,
    atol: float = _FEAS_ATOL,
) -> bool:
    v = np.asarray(x, dtype=float).reshape(-1)
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    if v.shape[0] != lb.shape[0]:
        return False
    if np.any(v < lb - atol) or np.any(v > ub + atol):
        return False
    for group in sum_groups or []:
        idxs = [int(i) for i in group.get("indices", []) if 0 <= int(i) < v.shape[0]]
        if not idxs:
            continue
        target = float(group.get("target", group.get("target_sum", 1.0)))
        if abs(float(np.sum(v[idxs])) - target) > atol:
            return False
    for rc in ratio_constraints or []:
        try:
            i = int(rc.get("i", -1))
            j = int(rc.get("j", -1))
        except (TypeError, ValueError):
            continue
        if i < 0 or j < 0 or i == j or i >= v.shape[0] or j >= v.shape[0]:
            continue
        denom = max(float(v[j]), 1e-12)
        ratio = float(v[i]) / denom
        if rc.get("min_denominator") is not None and float(v[j]) + atol < float(rc["min_denominator"]):
            return False
        if rc.get("min_ratio") is not None and ratio < float(rc["min_ratio"]) - atol:
            return False
        if rc.get("max_ratio") is not None and ratio > float(rc["max_ratio"]) + atol:
            return False
    return True


def project_to_feasible(
    x_hat: np.ndarray,
    bounds: list[tuple[float, float]],
    sum_groups: list[dict[str, Any]] | None = None,
    ratio_constraints: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, bool]:
    """Closest point in the feasible polytope to ``x_hat`` (squared Euclidean)."""
    target = np.asarray(x_hat, dtype=float).reshape(-1)
    lb = np.array([lo for lo, _ in bounds], dtype=float)
    ub = np.array([hi for _, hi in bounds], dtype=float)
    for rc in ratio_constraints or []:
        if rc.get("min_denominator") is None:
            continue
        try:
            j = int(rc.get("j", -1))
        except (TypeError, ValueError):
            continue
        if 0 <= j < lb.shape[0]:
            lb[j] = max(float(lb[j]), float(rc["min_denominator"]))
    dim = int(target.shape[0])
    cons = _sum_constraints(sum_groups, dim) + _ratio_constraints(ratio_constraints, dim)

    def objective(x: np.ndarray) -> float:
        d = np.asarray(x, dtype=float) - target
        return 0.5 * float(np.dot(d, d))

    x0 = np.clip(target, lb, ub)
    for group in sum_groups or []:
        idxs = [int(i) for i in group.get("indices", []) if 0 <= int(i) < dim]
        if not idxs:
            continue
        want = float(group.get("target", group.get("target_sum", 1.0)))
        total = float(np.sum(x0[idxs]))
        if total > 0.0:
            x0[idxs] = x0[idxs] * (want / total)
    x0 = np.clip(x0, lb, ub)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=list(zip(lb.tolist(), ub.tolist(), strict=True)),
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 250, "disp": False},
    )
    x = np.clip(np.asarray(result.x, dtype=float), lb, ub)
    ok = is_feasible(x, bounds, sum_groups, ratio_constraints)
    if not ok:
        # The repair can leave a sum slightly off after the bound clip. Try once more
        # from that point so a feasible polytope is not reported as infeasible.
        result2 = minimize(
            objective,
            x,
            method="SLSQP",
            bounds=list(zip(lb.tolist(), ub.tolist(), strict=True)),
            constraints=cons,
            options={"ftol": 1e-14, "maxiter": 250, "disp": False},
        )
        x = np.clip(np.asarray(result2.x, dtype=float), lb, ub)
        ok = is_feasible(x, bounds, sum_groups, ratio_constraints)
    return x, bool(ok)
