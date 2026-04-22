"""Shared helpers for SATOR demo scripts.

Two groups of helpers:

* HTTP / job helpers — submit a request, poll the job endpoint, save
  request/result JSON to ``examples/responses/<name>_{request,result}.json``.
* Reporting helpers — print section banners, render wide tables (with
  console truncation + full text artifacts on disk), and run explicit
  constraint / metric checks with ``PASS`` / ``FAIL`` labels, so audit
  demos can demonstrate engine correctness end-to-end.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_API_KEY = "dev-key"
RESPONSES_DIR = Path("examples/responses")


def api_env() -> tuple[str, str]:
    """Return (base_url, api_key) from environment with dev defaults."""
    return (
        os.environ.get("SATOR_BASE_URL", DEFAULT_BASE_URL),
        os.environ.get("SATOR_API_KEY", DEFAULT_API_KEY),
    )


def save_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def post_optimize_and_wait(
    name: str,
    payload: dict[str, Any],
    *,
    timeout_s: float = 60.0,
    poll_interval_s: float = 0.25,
    max_poll_s: float = 120.0,
) -> dict[str, Any]:
    """POST /v1/optimize, then poll /v1/jobs/<id>/result until predictions land.

    Saves both the request and the final result to ``examples/responses/``.
    Raises ``RuntimeError`` if the job fails or times out.
    """
    base, api_key = api_env()
    headers = {"x-api-key": api_key}
    req_path = save_json(RESPONSES_DIR / f"{name}_request.json", payload)
    print(f"[{name}] request saved -> {req_path}")

    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(f"{base}/v1/optimize", json=payload, headers=headers)
        if r.status_code != 202:
            print(f"[{name}] /v1/optimize returned {r.status_code}: {r.text}")
        r.raise_for_status()
        job_id = r.json()["job_id"]
        print(f"[{name}] job accepted: {job_id}")

        result: dict[str, Any] | None = None
        deadline = time.time() + max_poll_s
        while time.time() < deadline:
            rr = client.get(f"{base}/v1/jobs/{job_id}/result", headers=headers)
            data = rr.json() if rr.headers.get("content-type", "").startswith("application/json") else {}
            if data.get("status") == "FAILED":
                raise RuntimeError(f"[{name}] job failed: {data.get('error')}")
            if rr.status_code == 200 and data.get("predictions"):
                result = data
                break
            time.sleep(poll_interval_s)
    if result is None:
        raise RuntimeError(f"[{name}] job did not complete within {max_poll_s:.0f}s")

    res_path = save_json(RESPONSES_DIR / f"{name}_result.json", result)
    print(f"[{name}] result saved -> {res_path}")
    return result


def section(title: str, ch: str = "=") -> None:
    """Print a clear banner for a demo stage."""
    bar = ch * max(8, min(80, len(title) + 4))
    print()
    print(bar)
    print(f" {title}")
    print(bar)


def _format_table(
    col_headers: list[str],
    row_headers: list[str],
    matrix: np.ndarray,
    fmt: str,
    row_label: str = "",
) -> str:
    """Render a fixed-width text table. ``matrix`` is ``rows x cols``."""
    rows = matrix.shape[0]
    cols = matrix.shape[1] if matrix.ndim > 1 else 0
    str_matrix: list[list[str]] = []
    for r in range(rows):
        row = []
        for c in range(cols):
            v = matrix[r, c]
            if isinstance(v, float) and (np.isnan(v)):
                row.append("...")
            else:
                row.append(fmt.format(v))
        str_matrix.append(row)

    col_widths = [len(h) for h in col_headers]
    for r in range(rows):
        for c in range(cols):
            col_widths[c] = max(col_widths[c], len(str_matrix[r][c]))
    label_w = max([len(row_label)] + [len(h) for h in row_headers])

    sep = "-" * (label_w + 3 + sum(w + 3 for w in col_widths))
    lines: list[str] = []
    header = f"{row_label:<{label_w}} | " + " | ".join(f"{h:>{col_widths[c]}}" for c, h in enumerate(col_headers))
    lines.append(header)
    lines.append(sep)
    for r in range(rows):
        body = f"{row_headers[r]:<{label_w}} | " + " | ".join(
            f"{str_matrix[r][c]:>{col_widths[c]}}" for c in range(cols)
        )
        lines.append(body)
    return "\n".join(lines)


def render_table(
    title: str,
    col_headers: list[str],
    row_headers: list[str],
    matrix: np.ndarray,
    fmt: str = "{:.4f}",
    row_label: str = "",
    max_cols_console: int = 10,
    save_path: Path | None = None,
) -> None:
    """Print a wide table with optional console truncation and full text artifact."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    full_text = _format_table(col_headers, row_headers, arr, fmt, row_label=row_label)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(f"{title}\n\n{full_text}\n", encoding="utf-8")
    print(f"\n{title}:")
    if arr.shape[1] > max_cols_console:
        truncated_cols = col_headers[:max_cols_console] + ["..."]
        trunc_mat = np.concatenate(
            [arr[:, :max_cols_console], np.full((arr.shape[0], 1), np.nan)], axis=1
        )
        truncated = _format_table(truncated_cols, row_headers, trunc_mat, fmt, row_label=row_label)
        print(truncated)
        if save_path is not None:
            print(f"  (full {arr.shape[1]} columns written to {save_path})")
    else:
        print(full_text)


def render_key_value_block(title: str, items: list[tuple[str, str]], save_path: Path | None = None) -> None:
    """Print a two-column aligned 'key: value' block (used for goals, config, etc)."""
    width = max(len(k) for k, _ in items) if items else 1
    lines = [f"  {k:<{width}}  {v}" for k, v in items]
    text = "\n".join(lines)
    print(f"\n{title}:")
    print(text)
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(f"{title}\n\n{text}\n", encoding="utf-8")


def check_result(label: str, ok: bool, detail: str = "") -> bool:
    """Uniform ``PASS`` / ``FAIL`` line, returns ``ok``."""
    tag = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"  [{tag}] {label}{suffix}")
    return ok


def check_sum_to_one(
    candidates: np.ndarray,
    ingredient_indices: list[int],
    target: float = 1.0,
    tol: float = 1e-4,
) -> tuple[bool, np.ndarray]:
    """Return (all_ok, per_row_error)."""
    sums = candidates[:, ingredient_indices].sum(axis=1)
    errors = sums - target
    ok = bool(np.all(np.abs(errors) <= tol))
    return ok, errors


def check_bounds(
    candidates: np.ndarray,
    bounds: list[tuple[float, float]],
    tol: float = 1e-6,
) -> tuple[bool, list[list[int]]]:
    """Return (all_ok, violations_per_row_as_index_lists)."""
    violations: list[list[int]] = []
    ok = True
    for r in range(candidates.shape[0]):
        row_viol: list[int] = []
        for c, (lo, hi) in enumerate(bounds):
            if candidates[r, c] < lo - tol or candidates[r, c] > hi + tol:
                row_viol.append(c)
        if row_viol:
            ok = False
        violations.append(row_viol)
    return ok, violations


def check_ratio(
    candidates: np.ndarray,
    i: int,
    j: int,
    min_ratio: float | None,
    max_ratio: float | None,
    tol: float = 1e-4,
) -> tuple[bool, np.ndarray]:
    """Return (all_ok, per_row_ratio) for the ``x_i / x_j`` ratio constraint."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = candidates[:, i] / np.maximum(candidates[:, j], 1e-18)
    ok = True
    if min_ratio is not None and np.any(r < min_ratio - tol):
        ok = False
    if max_ratio is not None and np.any(r > max_ratio + tol):
        ok = False
    return ok, r


def gp_calibration(
    gp_mean: np.ndarray, gp_std: np.ndarray, truth: np.ndarray
) -> dict[str, float]:
    """Return basic GP-vs-truth stats: MAE, RMSE, mean |z| where z = (y - mean)/std."""
    err = truth - gp_mean
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    z = err / np.maximum(gp_std, 1e-9)
    return {"mae": mae, "rmse": rmse, "mean_abs_z": float(np.mean(np.abs(z)))}


def mpl_setup():
    """Best-effort interactive matplotlib setup; falls back to Agg if no display."""
    try:
        import matplotlib  # noqa: F401

        try:
            matplotlib.use("TkAgg")
        except Exception:
            matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt  # noqa: F401
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    return plt
