"""Small HTTP client for submitting a job and waiting for its result."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx


class SatorJobError(RuntimeError):
    """The job finished as failed or cancelled."""

    def __init__(self, message: str, status: str = "FAILED") -> None:
        super().__init__(message)
        self.status = status


class SatorClient:
    """Calls `/v1/optimize`, `/v1/explain`, `/v1/design`, and `/v1/validate`.

    Pass an existing ``httpx.Client`` when a test needs a fake transport.
    That client must already send ``x-api-key``.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 30.0,
        http: httpx.Client | None = None,
    ) -> None:
        self._owns_http = http is None
        self._http = http or httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"x-api-key": api_key},
            timeout=timeout,
        )

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> SatorClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self._http.request(method, path, json=payload)
        response.raise_for_status()
        body = response.json()
        return body if isinstance(body, dict) else {"result": body}

    def optimize(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", "/v1/optimize", payload)

    def explain(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", "/v1/explain", payload)

    def design(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", "/v1/design", payload)

    def validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", "/v1/validate", payload)

    def job(self, job_id: str) -> dict[str, Any]:
        return self._json("GET", f"/v1/jobs/{job_id}")

    def result(self, job_id: str) -> dict[str, Any]:
        return self._json("GET", f"/v1/jobs/{job_id}/result")

    def wait(self, job_id: str, *, timeout: float = 120.0, interval: float = 0.5) -> dict[str, Any]:
        """Poll until the job completes, then return its result."""
        deadline = time.monotonic() + float(timeout)
        while True:
            status = self.job(job_id)
            state = str(status.get("status") or "")
            if state in {"FAILED", "CANCELLED"}:
                raise SatorJobError(str(status.get("error") or state), state)
            if state in {"COMPLETED", "SUCCEEDED"}:
                return self.result(job_id)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"job {job_id} is still {state or 'pending'}")
            time.sleep(interval)

    def suggest(self, payload: dict[str, Any], *, timeout: float = 120.0, interval: float = 0.5) -> dict[str, Any]:
        """Submit an optimization and return the finished result."""
        accepted = self.optimize(payload)
        return self.wait(str(accepted["job_id"]), timeout=timeout, interval=interval)


def dataset_from_frame(frame: Any, objective_columns: Sequence[str]) -> dict[str, list[Any]]:
    """Build ``dataset.X`` and ``dataset.Y`` from a pandas DataFrame."""
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("expected a pandas DataFrame")
    columns = list(objective_columns)
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise KeyError(f"missing objective columns: {missing}")
    outcomes = frame.loc[:, columns]
    inputs = frame.drop(columns=columns)
    return {"X": inputs.to_numpy(dtype=float).tolist(), "Y": outcomes.to_numpy(dtype=float).tolist()}


def predictions_frame(result: dict[str, Any]) -> Any:
    """One row per well, with the candidate, predicted outcomes, replicate, and control flag."""
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for pred in result.get("predictions") or []:
        row = dict(pred.get("candidate") or {})
        for index, value in enumerate(pred.get("objectives") or []):
            row[f"y{index}"] = value
        row["replicate"] = int(pred.get("replicate", 1))
        row["control"] = bool(pred.get("control", False))
        rows.append(row)
    return pd.DataFrame(rows)
