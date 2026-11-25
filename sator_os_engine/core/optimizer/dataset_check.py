"""Report data problems before a job is started."""

from __future__ import annotations

from typing import Any

import numpy as np


def _issue(code: str, message: str, rows: list[int] | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "message": message}
    if rows:
        item["rows"] = rows
    return item


def validate_dataset(
    x: np.ndarray,
    y: np.ndarray,
    params: list[dict[str, Any]],
    sum_groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Duplicates, bounds, missing values, sums, and collinear columns."""
    rows = np.asarray(x, dtype=float)
    outcomes = np.atleast_2d(np.asarray(y, dtype=float))
    if outcomes.shape[0] == 1 and rows.ndim == 2 and outcomes.shape[1] == rows.shape[0] and rows.shape[0] != 1:
        outcomes = outcomes.T
    issues: list[dict[str, Any]] = []
    if rows.ndim != 2 or outcomes.shape[0] != rows.shape[0]:
        return {"ok": False, "issues": [_issue("shape", "dataset.X and dataset.Y must have the same number of rows")]}

    nan_rows = sorted(
        {int(i) for i in np.where(~np.isfinite(rows).all(axis=1) | ~np.isfinite(outcomes).all(axis=1))[0]}
    )
    if nan_rows:
        issues.append(_issue("nan", "a row contains NaN or infinity", nan_rows))

    seen: dict[tuple[float, ...], int] = {}
    duplicate_rows: list[int] = []
    for index, row in enumerate(np.round(rows, 8)):
        key = tuple(float(v) for v in row)
        if key in seen:
            duplicate_rows.append(index)
        else:
            seen[key] = index
    if duplicate_rows:
        issues.append(_issue("duplicate", "dataset.X has repeated rows", duplicate_rows))

    outside: list[int] = []
    for index, row in enumerate(rows):
        for j, param in enumerate(params):
            if param.get("type") == "categorical" or j >= row.shape[0]:
                continue
            lo, hi = float(param["min"]), float(param["max"])
            if np.isfinite(row[j]) and (row[j] < lo - 1e-8 or row[j] > hi + 1e-8):
                outside.append(index)
                break
    if outside:
        issues.append(_issue("out_of_bounds", "a row is outside the search bounds", outside))

    for group in sum_groups or []:
        indices = [int(i) for i in group.get("indices", []) if 0 <= int(i) < rows.shape[1]]
        if not indices:
            continue
        target = float(group.get("target", group.get("target_sum", 1.0)))
        totals = rows[:, indices].sum(axis=1)
        bad = [int(i) for i, total in enumerate(totals) if np.isfinite(total) and abs(float(total) - target) > 1e-3]
        if bad:
            issues.append(_issue("sum", f"sum of columns {indices} is off the target {target}", bad))

    finite_cols = [j for j in range(rows.shape[1]) if rows.shape[0] >= 3 and np.isfinite(rows[:, j]).all()]
    for j in finite_cols:
        if float(np.std(rows[:, j])) < 1e-8:
            name = str(params[j]["name"]) if j < len(params) else str(j)
            issues.append(_issue("constant", f"column {name} does not vary"))
    varying = [j for j in finite_cols if float(np.std(rows[:, j])) >= 1e-8]
    for a, left in enumerate(varying):
        for right in varying[a + 1 :]:
            corr = np.corrcoef(rows[:, left], rows[:, right])[0, 1]
            if np.isfinite(corr) and abs(float(corr)) >= 0.999:
                left_name = str(params[left]["name"]) if left < len(params) else str(left)
                right_name = str(params[right]["name"]) if right < len(params) else str(right)
                issues.append(_issue("collinear", f"columns {left_name} and {right_name} are collinear"))
    return {"ok": not issues, "issues": issues}
