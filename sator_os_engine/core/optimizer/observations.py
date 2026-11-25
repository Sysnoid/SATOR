"""Collapse replicate recipes and align known observation noise."""

from __future__ import annotations

from typing import Any

import numpy as np

_VAR_FLOOR = 1e-10


def _row_key(row: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(row, dtype=float), decimals=8))


def prepare_observations(
    x: np.ndarray,
    y: np.ndarray,
    yvar: np.ndarray | None = None,
    failed: list[Any] | None = None,
) -> dict[str, Any]:
    """Group identical recipes.

    A group becomes one row: the mean of its finite outcomes, and a variance
    taken from ``yvar`` when that was supplied, otherwise the sample variance
    when the recipe was measured more than once. Unreplicated rows with no
    supplied variance inherit the median known variance of that objective so
    a column can use a fixed-noise GP. ``feasibility`` is 1 for a success and
    0 for a failure, and is omitted when every row has the same label.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n, n_obj = y.shape
    supplied = None if yvar is None else np.asarray(yvar, dtype=float).reshape(n, n_obj)
    labels = None
    if failed is not None:
        labels = np.asarray([0.0 if bool(flag) else 1.0 for flag in failed], dtype=float)

    groups: dict[tuple[float, ...], list[int]] = {}
    for i in range(n):
        groups.setdefault(_row_key(x[i]), []).append(i)

    x_out = np.zeros((len(groups), x.shape[1]), dtype=float)
    y_out = np.zeros((len(groups), n_obj), dtype=float)
    v_out = np.full((len(groups), n_obj), np.nan, dtype=float)
    f_out = np.ones(len(groups), dtype=float) if labels is not None else None
    saw_variance = False
    for row_i, idxs in enumerate(groups.values()):
        idx = np.asarray(idxs, dtype=int)
        x_out[row_i] = x[idx[0]]
        if f_out is not None and labels is not None:
            f_out[row_i] = float(np.min(labels[idx]))
        for j in range(n_obj):
            vals = y[idx, j]
            finite = np.isfinite(vals)
            if not np.any(finite):
                y_out[row_i, j] = np.nan
                continue
            y_out[row_i, j] = float(np.mean(vals[finite]))
            if supplied is not None:
                known = supplied[idx, j]
                known = known[np.isfinite(known) & finite]
                if known.size:
                    v_out[row_i, j] = float(np.mean(known))
                    saw_variance = True
                    continue
            if int(np.sum(finite)) >= 2:
                v_out[row_i, j] = float(np.var(vals[finite], ddof=1))
                saw_variance = True

    if saw_variance:
        for j in range(n_obj):
            known = v_out[:, j]
            finite = np.isfinite(known)
            if not np.any(finite):
                continue
            fill = float(np.median(known[finite]))
            known = np.where(finite, known, fill)
            v_out[:, j] = np.maximum(known, _VAR_FLOOR)
    else:
        v_out = None

    feasibility = None
    if f_out is not None and np.any(f_out <= 0.0) and np.any(f_out >= 1.0):
        feasibility = f_out

    return {
        "X": x_out,
        "Y": y_out,
        "Yvar": v_out,
        "feasibility": feasibility,
        "collapsed": int(n - len(groups)),
        "fixed_noise": v_out is not None,
    }
