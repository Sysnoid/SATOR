"""Integration tests for async API flows.

These tests spin up the FastAPI app in-memory and exercise:
- /v1/optimize: submit an asynchronous optimization job and poll until results
  are available; verify predictions are returned.
- /v1/reconstruct: submit a reconstruction job and poll until success; verify
  a reconstructed formulation is returned.

Authentication via x-api-key is included to mirror real usage.
"""

from __future__ import annotations

import os
import time

from fastapi.testclient import TestClient

from sator_os_engine.api.app import create_app
from sator_os_engine.settings import Settings, get_settings


def _app():
    os.environ["SATOR_API_KEY"] = "test-key"
    get_settings.cache_clear()
    settings = Settings()
    return create_app(settings)


def test_optimize_async_flow():
    app = _app()
    payload = {
        "dataset": {
            "X": [[0.2, 0.1], [0.5, 0.3], [0.8, 0.2], [0.3, 0.6], [0.4, 0.4]],
            "Y": [[0.5, 0.6], [0.2, 0.4], [0.3, 0.5], [0.35, 0.45], [0.25, 0.55]],
        },
        "search_space": {
            "parameters": [
                {"name": "x1", "type": "float", "min": 0.0, "max": 1.0},
                {"name": "x2", "type": "float", "min": -1.0, "max": 1.0},
            ]
        },
        "objectives": {"o1": {"goal": "min"}, "o2": {"goal": "min"}},
        "optimization_config": {"algorithm": "qnehvi", "batch_size": 3, "max_evaluations": 10, "seed": 42},
    }
    with TestClient(app) as client:
        r = client.post("/v1/optimize", json=payload, headers={"x-api-key": "test-key"})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        for _ in range(120):
            rr = client.get(f"/v1/jobs/{job_id}/result", headers={"x-api-key": "test-key"})
            if rr.status_code == 200 and "predictions" in rr.json():
                data = rr.json()
                assert isinstance(data["predictions"], list)
                break
            time.sleep(0.2)
        else:
            raise AssertionError("optimize result not ready in time")


def test_reconstruct_async_flow():
    app = _app()
    # Simple 3-dim (2 ingredients + 1 parameter), identity components
    pca_info = {
        "pc_mins": [0.0, 0.0],
        "pc_maxs": [1.0, 1.0],
        "components": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        "mean": [0.0, 0.0, 0.0],
    }
    payload = {
        "coordinates": [0.5, 0.5],
        "pca_info": pca_info,
        "bounds": {
            "ingredients": [[0.0, 1.0], [0.0, 1.0]],
            "parameters": [[0.0, 1.0]],
        },
        "n_ingredients": 2,
        "target_precision": 1e-7,
    }
    with TestClient(app) as client:
        r = client.post("/v1/reconstruct", json=payload, headers={"x-api-key": "test-key"})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        for _ in range(50):
            rr = client.get(f"/v1/jobs/{job_id}/result", headers={"x-api-key": "test-key"})
            if rr.status_code == 200 and rr.json().get("success") is True:
                data = rr.json()
                assert "reconstructed_formulation" in data
                break
            time.sleep(0.1)
        else:
            raise AssertionError("reconstruct result not ready in time")
