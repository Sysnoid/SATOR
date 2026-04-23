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


# ---------------------------------------------------------------------------
# Hard-constraint goal enforcement
# ---------------------------------------------------------------------------
#
# The "enforce_*" goal family turns a soft scoring term into a **hard
# feasibility gate** on the GP posterior. Unlike ``minimize_below`` or
# ``maximize_above`` (which only add a penalty to the acquisition score),
# ``enforce_below`` / ``enforce_above`` / ``enforce_within_range`` require the
# GP-predicted mean (or a confidence bound, if an uncertainty margin is
# configured) to satisfy the threshold or every candidate is flagged as
# infeasible in the response.
#
# Two helpers ship together:
#   * ``extract_enforced_goal_specs``: normalises the per-objective
#     configuration into a lightweight list of ``(obj_name, kind, bounds)``
#     tuples; returns ``[]`` when no enforce_* goal is present so callers
#     can skip the whole check with a single ``if specs:``.
#   * ``evaluate_enforced_goals``: given stacked posterior ``mu`` and
#     ``var`` arrays (one column per objective, in the same order as
#     ``req.objectives``) and the uncertainty margin, returns a boolean
#     mask of row-wise feasibility plus a list of per-row violation
#     labels suitable for the ``diagnostics.enforcement`` block.
#
# Goal payload shapes accepted by the parser:
#   * ``{"goal": "enforce_above", "threshold_value": 4.2}``
#   * ``{"goal": "enforce_above", "threshold": {"value": 4.2}}``  (back-compat)
#   * ``{"goal": "enforce_below", "threshold_value": 1.0}``
#   * ``{"goal": "enforce_within_range",
#          "range": {"min": 80, "max": 150}}``


ENFORCED_GOAL_NAMES: tuple[str, ...] = (
    "enforce_above",
    "enforce_below",
    "enforce_within_range",
)


def extract_enforced_goal_specs(req) -> list[dict[str, Any]]:
    """Return a normalised list of hard-enforce goal specs.

    Each spec is ``{"index": int, "name": str, "kind": str, "lo": float | None,
    "hi": float | None}``. ``kind`` is one of ``"above"``, ``"below"``,
    ``"within_range"``; ``lo`` is the lower threshold (above / within_range),
    ``hi`` the upper threshold (below / within_range). Returns an empty list
    if no enforce_* goal is present, so callers can bail cheaply.
    """
    if not hasattr(req, "objectives") or not isinstance(req.objectives, dict):
        return []
    specs: list[dict[str, Any]] = []
    for idx, (obj_name, cfg) in enumerate(req.objectives.items()):
        if not isinstance(cfg, dict):
            continue
        goal = str(cfg.get("goal", "")).lower()
        if goal not in ENFORCED_GOAL_NAMES:
            continue
        thr_val = cfg.get("threshold_value")
        if thr_val is None and isinstance(cfg.get("threshold"), dict):
            thr_val = cfg["threshold"].get("value")
        rng = cfg.get("range") if isinstance(cfg.get("range"), dict) else None
        lo: float | None = None
        hi: float | None = None
        if goal == "enforce_above" and thr_val is not None:
            lo = float(thr_val)
        elif goal == "enforce_below" and thr_val is not None:
            hi = float(thr_val)
        elif goal == "enforce_within_range" and rng is not None:
            if rng.get("min") is not None:
                lo = float(rng["min"])
            if rng.get("max") is not None:
                hi = float(rng["max"])
            if lo is not None and hi is not None and lo > hi:
                lo, hi = hi, lo
        # Skip specs we can't act on (missing bounds).
        if lo is None and hi is None:
            continue
        specs.append(
            {
                "index": idx,
                "name": str(obj_name),
                "kind": goal.removeprefix("enforce_"),
                "lo": lo,
                "hi": hi,
            }
        )
    return specs


def evaluate_enforced_goals(
    specs: list[dict[str, Any]],
    mu_matrix: np.ndarray,
    var_matrix: np.ndarray | None = None,
    margin: float = 0.0,
) -> tuple[np.ndarray, list[list[str]]]:
    """Evaluate every row of ``mu_matrix`` against the enforce_* specs.

    ``mu_matrix`` has shape ``(n_rows, n_objectives)`` where the column
    order matches ``req.objectives`` (and therefore each spec's ``index``).
    ``var_matrix`` is the matching variance matrix; only consulted when
    ``margin > 0``. Returns ``(mask, violations)`` where ``mask`` is shape
    ``(n_rows,)`` and True means every enforce_* constraint is satisfied,
    and ``violations[i]`` is a list of ``"<obj_name>:<reason>"`` labels for
    row ``i`` (empty list when the row is feasible).
    """
    n_rows = int(mu_matrix.shape[0]) if mu_matrix.ndim > 1 else mu_matrix.shape[0]
    mask = np.ones(n_rows, dtype=bool)
    violations: list[list[str]] = [[] for _ in range(n_rows)]
    if not specs:
        return mask, violations
    m = max(0.0, float(margin))
    for spec in specs:
        col = int(spec["index"])
        mu = np.asarray(mu_matrix[:, col], dtype=float)
        sigma = (
            np.sqrt(np.maximum(np.asarray(var_matrix[:, col], dtype=float), 0.0))
            if (m > 0.0 and var_matrix is not None)
            else np.zeros_like(mu)
        )
        lo = spec["lo"]
        hi = spec["hi"]
        name = spec["name"]
        # For enforce_above / lower edge of a range, the worst case is the
        # lower confidence bound ``mu - m*sigma`` (we want it above lo).
        # For enforce_below / upper edge, the worst case is the upper
        # confidence bound ``mu + m*sigma`` (we want it below hi).
        if lo is not None:
            lcb = mu - m * sigma
            ok_lo = lcb >= lo - 1e-12
            for i in np.where(~ok_lo)[0]:
                violations[int(i)].append(f"{name}<{lo:g}")
            mask &= ok_lo
        if hi is not None:
            ucb = mu + m * sigma
            ok_hi = ucb <= hi + 1e-12
            for i in np.where(~ok_hi)[0]:
                violations[int(i)].append(f"{name}>{hi:g}")
            mask &= ok_hi
    return mask, violations
