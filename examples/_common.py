"""Shared helpers for SATOR demo scripts.

Kept small on purpose: submit a request, poll the job endpoint, save
request/result JSON to ``examples/responses/<name>_{request,result}.json``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

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
