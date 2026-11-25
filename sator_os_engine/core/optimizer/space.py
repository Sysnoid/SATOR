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
    return y


def candidate_by_name(x: np.ndarray, params: list[dict[str, Any]]) -> dict[str, Any]:
    snapped = snap_numeric(x, params)
    out: dict[str, Any] = {}
    for j, param in enumerate(params):
        name = str(param["name"])
        ptype = param.get("type", "float")
        if ptype == "categorical":
            choices = _choices(param)
            out[name] = choices[int(snapped[j])] if choices else None
        elif ptype == "int":
            out[name] = int(snapped[j])
        else:
            out[name] = float(snapped[j])
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
