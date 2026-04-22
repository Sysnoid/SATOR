from __future__ import annotations

from typing import Any

import numpy as np


def sample_candidates(search_space: dict[str, Any], n: int, seed: int | None = None) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    params = search_space.get("parameters", [])
    cands: list[dict[str, float]] = []
    for _ in range(n):
        cand: dict[str, float] = {}
        for p in params:
            name = p["name"]
            ptype = p.get("type", "float")
            if ptype in ("float", "int"):
                lo, hi = float(p["min"]), float(p["max"])
                val = rng.uniform(lo, hi)
                cand[name] = float(int(val)) if ptype == "int" else float(val)
            elif ptype == "categorical":
                choices = p.get("choices", [])
                cand[name] = choices[int(rng.integers(0, len(choices)))] if choices else None
        cands.append(cand)
    return cands


def dummy_objective(cand: dict[str, float]) -> list[float]:
    vals = np.array([v for v in cand.values() if isinstance(v, (int, float))], dtype=float)
    if vals.size == 0:
        return [0.0, 0.0]
    return [float(np.sum(vals)), float(-np.var(vals))]


def pareto_front(
    points: list[list[float]],
    minimize_frame: np.ndarray | None = None,
) -> list[int]:
    """Non-dominated set in a consistent minimization frame.

    *minimize_frame*: optional length-n_obj array such that
    P_work[i,j] = P[i,j] * minimize_frame[j], where *lower* P_work is better.
    Use ``-signs`` from the optimization engine, where *signs* is +1 for *max* and -1 for *min* on original Y.
    """
    P = np.array(points, dtype=float)
    if minimize_frame is not None:
        m = np.asarray(minimize_frame, dtype=float)
        P = P * m
    n = P.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(P[j] <= P[i]) and np.any(P[j] < P[i]):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


def build_linear_constraints(
    req, params: list[dict[str, Any]]
) -> tuple[list[tuple[list[int], list[float], float]], list[tuple[list[int], list[float], float]]]:
    ineq: list[tuple[list[int], list[float], float]] = []
    eqpairs: list[tuple[list[int], list[float], float]] = []
    dim = len(params)
    sums = (
        (req.optimization_config.sum_constraints or []) if hasattr(req.optimization_config, "sum_constraints") else []
    )
    for sc in sums:
        idxs = [int(i) for i in sc.get("indices", []) if 0 <= int(i) < dim]
        # Default 1.0 for mixture-style sums when target_sum is omitted
        target = float(sc.get("target_sum", 1.0))
        if not idxs:
            continue
        coeff = [1.0] * len(idxs)
        coeff_neg = [-1.0] * len(idxs)
        ineq.append((idxs, coeff, target))
        ineq.append((idxs, coeff_neg, -target))
    ratios = (
        (req.optimization_config.ratio_constraints or [])
        if hasattr(req.optimization_config, "ratio_constraints")
        else []
    )
    for rc in ratios:
        i = int(rc.get("i", -1))
        j = int(rc.get("j", -1))
        if not (0 <= i < dim and 0 <= j < dim) or i == j:
            continue
        # BoTorch's optimize_acqf uses >= rhs semantics for inequality tuples
        # (see the sum encoding above). Ratio bounds must be rewritten in the
        # same direction:
        #   x_i / x_j <= max_ratio  <=>  max_ratio * x_j - x_i >= 0
        #   x_i / x_j >= min_ratio  <=>  x_i - min_ratio * x_j >= 0
        if rc.get("max_ratio") is not None:
            mr = float(rc["max_ratio"])
            ineq.append(([i, j], [-1.0, mr], 0.0))
        if rc.get("min_ratio") is not None:
            mrn = float(rc["min_ratio"])
            ineq.append(([i, j], [1.0, -mrn], 0.0))
    return ineq, eqpairs


def feasible_mask(points: list[list[float]], req, params: list[dict[str, Any]], tol: float = 1e-6) -> list[bool]:
    X = np.asarray(points, dtype=float)
    dim = X.shape[1]
    mask = np.ones(X.shape[0], dtype=bool)
    sums = (
        (req.optimization_config.sum_constraints or []) if hasattr(req.optimization_config, "sum_constraints") else []
    )
    for sc in sums:
        idxs = [int(i) for i in sc.get("indices", []) if 0 <= int(i) < dim]
        target = float(sc.get("target_sum", 1.0))
        if idxs:
            s = X[:, idxs].sum(axis=1)
            mask &= np.isclose(s, target, atol=tol)
    ratios = (
        (req.optimization_config.ratio_constraints or [])
        if hasattr(req.optimization_config, "ratio_constraints")
        else []
    )
    for rc in ratios:
        i = int(rc.get("i", -1))
        j = int(rc.get("j", -1))
        if not (0 <= i < dim and 0 <= j < dim) or i == j:
            continue
        xi = X[:, i]
        xj = X[:, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            r = xi / xj
        if rc.get("max_ratio") is not None:
            mask &= r <= float(rc["max_ratio"]) + tol
        if rc.get("min_ratio") is not None:
            mask &= r >= float(rc["min_ratio"]) - tol
    return mask.tolist()


def _project_sum_to_target_with_bounds(
    sub: np.ndarray,
    target: float,
    lo: np.ndarray,
    hi: np.ndarray,
    max_iter: int = 48,
    atol: float = 1e-6,
) -> np.ndarray:
    """Row-wise: scale toward *target* sum, clip to ``[lo,hi]``, repeat until stable.

    A single scale+clip round can leave the sum != *target* when bounds bite; iterating
    reduces that error for typical feasible boxes.
    """
    out = np.array(sub, dtype=float, copy=True)
    n_rows, k = out.shape
    if k == 0:
        return out
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    for r in range(n_rows):
        x = out[r].copy()
        for _ in range(max_iter):
            tot = float(np.sum(x))
            if abs(tot - target) < atol:
                break
            if tot > 1e-18:
                x = x * (target / tot)
            else:
                x[:] = target / k if k else 0.0
            x = np.clip(x, lo, hi)
        out[r] = x
    return out


def enforce_sum_constraints_np(cands: np.ndarray, params: list[dict[str, Any]], req) -> np.ndarray:
    if cands.size == 0:
        return cands
    sums = (
        (req.optimization_config.sum_constraints or []) if hasattr(req.optimization_config, "sum_constraints") else []
    )
    if not sums:
        return cands
    bounds: list[tuple[float | None, float | None]] = []
    for p in params:
        if p.get("type", "float") in ("float", "int"):
            bounds.append((float(p["min"]), float(p["max"])))
        else:
            bounds.append((None, None))
    X = np.array(cands, dtype=float, copy=True)
    for sc in sums:
        idxs = [int(i) for i in sc.get("indices", []) if 0 <= int(i) < X.shape[1]]
        if not idxs:
            continue
        target = float(sc.get("target_sum", 1.0))
        sub = X[:, idxs]
        lo = np.array(
            [bounds[idx][0] if bounds[idx][0] is not None else -np.inf for idx in idxs],
            dtype=float,
        )
        hi = np.array(
            [bounds[idx][1] if bounds[idx][1] is not None else np.inf for idx in idxs],
            dtype=float,
        )
        sub = _project_sum_to_target_with_bounds(sub, target, lo, hi)
        X[:, idxs] = sub
    return X


def enforce_ratio_constraints_np(
    cands: np.ndarray,
    params: list[dict[str, Any]],
    req,
) -> np.ndarray:
    """Row-wise repair ``x_i / x_j`` to lie inside each declared ratio window.

    For every declared ``ratio_constraints`` entry we snap ``x_i`` to
    ``clip(x_i, min_ratio * x_j, max_ratio * x_j)`` and then clip to the
    declared per-parameter bounds. The caller is expected to follow up with
    :func:`enforce_sum_constraints_np` so that any mass moved by this snap
    is redistributed back to a valid sum.

    Rows that cannot be repaired (e.g. ``x_j`` is zero or the target ratio
    window is incompatible with the declared bounds) are left untouched; the
    feasibility check downstream will surface them honestly instead of us
    silently fabricating a different recipe.
    """
    if cands.size == 0:
        return cands
    ratios = (
        (req.optimization_config.ratio_constraints or [])
        if hasattr(req.optimization_config, "ratio_constraints")
        else []
    )
    if not ratios:
        return cands
    X = np.array(cands, dtype=float, copy=True)
    dim = X.shape[1]
    for rc in ratios:
        try:
            i = int(rc.get("i", -1))
            j = int(rc.get("j", -1))
        except (TypeError, ValueError):
            continue
        if not (0 <= i < dim and 0 <= j < dim) or i == j:
            continue
        mn = rc.get("min_ratio")
        mx = rc.get("max_ratio")
        if mn is None and mx is None:
            continue
        i_lo = float(params[i].get("min", -np.inf)) if params[i].get("type", "float") in ("float", "int") else -np.inf
        i_hi = float(params[i].get("max", np.inf)) if params[i].get("type", "float") in ("float", "int") else np.inf
        xj = X[:, j]
        # Guard: we can't reason about a ratio when the denominator is zero.
        safe = xj > 1e-12
        if not np.any(safe):
            continue
        xi = X[:, i].copy()
        if mn is not None:
            target_lo = float(mn) * xj
            xi = np.where(safe, np.maximum(xi, target_lo), xi)
        if mx is not None:
            target_hi = float(mx) * xj
            xi = np.where(safe, np.minimum(xi, target_hi), xi)
        xi = np.clip(xi, i_lo, i_hi)
        X[:, i] = xi
    return X


def infer_ingredient_and_param_indices(params: list[dict[str, Any]], req) -> tuple[list[int], list[int]]:
    dim = len(params)
    ing_set = set()
    sums = (
        (req.optimization_config.sum_constraints or []) if hasattr(req.optimization_config, "sum_constraints") else []
    )
    for sc in sums:
        idxs = [int(i) for i in sc.get("indices", []) if 0 <= int(i) < dim]
        for i in idxs:
            ing_set.add(i)
    ing_idx = sorted(list(ing_set))
    other_idx = [i for i in range(dim) if i not in ing_set]
    return ing_idx, other_idx
