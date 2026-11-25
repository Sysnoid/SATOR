"""Encode mixed search-space rows and snap optimizer output back to types."""

from __future__ import annotations

from typing import Any

import numpy as np


def _choices(param: dict[str, Any]) -> list[Any]:
    return list(param.get("choices") or [])


def encode_value(value: Any, param: dict[str, Any]) -> float:
    ptype = param.get("type", "float")
    if ptype == "categorical":
        choices = _choices(param)
        labels = [str(c) for c in choices]
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            idx = int(value)
        else:
            try:
                idx = labels.index(str(value))
            except ValueError as exc:
                raise ValueError(f"parameter {param.get('name')!r}: {value!r} is not one of {labels}") from exc
        if idx < 0 or idx >= len(choices):
            raise ValueError(f"parameter {param.get('name')!r}: index {idx} is outside 0..{len(choices) - 1}")
        return float(idx)
    return float(value)


def encode_dataset(rows: list[Any], params: list[dict[str, Any]]) -> np.ndarray:
    encoded: list[list[float]] = []
    n = len(params)
    for r, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != n:
            raise ValueError(f"dataset.X[{r}] must have length {n}")
        encoded.append([encode_value(row[j], params[j]) for j in range(n)])
    return np.asarray(encoded, dtype=float)


def snap_numeric(x: np.ndarray, params: list[dict[str, Any]]) -> np.ndarray:
    """Round ints, snap categoricals to a level, and clip floats to bounds."""
    y = np.asarray(x, dtype=float).reshape(-1).copy()
    for j, param in enumerate(params):
        ptype = param.get("type", "float")
        if ptype == "int":
            lo, hi = int(param["min"]), int(param["max"])
            y[j] = float(min(max(int(np.rint(y[j])), lo), hi))
        elif ptype == "categorical":
            n = max(len(_choices(param)), 1)
            y[j] = float(min(max(int(np.rint(y[j])), 0), n - 1))
        else:
            y[j] = float(np.clip(y[j], float(param["min"]), float(param["max"])))
            step = param.get("resolution")
            if step is not None and float(step) > 0:
                step_f = float(step)
                y[j] = float(np.clip(round(y[j] / step_f) * step_f, float(param["min"]), float(param["max"])))
    return y


def balance_resolved_sums(
    x: np.ndarray,
    params: list[dict[str, Any]],
    groups: list[dict[str, Any]] | None,
) -> np.ndarray:
    """Put the rounding remainder on the largest ingredient so each sum stays exact."""
    y = np.asarray(x, dtype=float).reshape(-1).copy()
    for group in groups or []:
        indices = [int(i) for i in group.get("indices", []) if 0 <= int(i) < y.shape[0]]
        if len(indices) < 2:
            continue
        if not any(params[i].get("resolution") not in (None, 0, 0.0) for i in indices):
            continue
        target = float(group.get("target", group.get("target_sum", 1.0)))
        carrier = max(indices, key=lambda i: y[i])
        others = [i for i in indices if i != carrier]
        remainder = target - float(np.sum(y[others]))
        lo, hi = float(params[carrier]["min"]), float(params[carrier]["max"])
        y[carrier] = float(min(max(remainder, lo), hi))
    return y


def candidate_by_name(x: np.ndarray, params: list[dict[str, Any]], *, snap: bool = True) -> dict[str, Any]:
    values = snap_numeric(x, params) if snap else np.asarray(x, dtype=float).reshape(-1)
    out: dict[str, Any] = {}
    for j, param in enumerate(params):
        name = str(param["name"])
        ptype = param.get("type", "float")
        if ptype == "categorical":
            choices = _choices(param)
            out[name] = choices[int(values[j])] if choices else None
        elif ptype == "int":
            out[name] = int(values[j])
        else:
            out[name] = float(values[j])
    return out


def clip_to_bounds(x: np.ndarray, params: list[dict[str, Any]]) -> tuple[np.ndarray, int]:
    """Clip float/int columns into their bounds. Returns ``(X, n_clipped_entries)``."""
    out = np.asarray(x, dtype=float).copy()
    n_clip = 0
    for j, param in enumerate(params):
        if param.get("type", "float") == "categorical":
            continue
        lo, hi = float(param["min"]), float(param["max"])
        col = out[:, j]
        outside = (col < lo) | (col > hi)
        n_clip += int(np.sum(outside))
        out[:, j] = np.clip(col, lo, hi)
    return out, n_clip
