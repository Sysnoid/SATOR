"""Explanations attached to proposals and to caller-supplied recipes."""

from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sator_os_engine.api.app import create_app
from sator_os_engine.core.models.optimize import OptimizationConfig, OptimizeRequest
from sator_os_engine.core.optimizer.explain import _pca_scores, nearest_tested
from sator_os_engine.core.optimizer.mobo_engine import run_optimization
from sator_os_engine.settings import Settings, get_settings


def _line_request(**config):
    rng = np.random.default_rng(0)
    x1 = rng.uniform(0.0, 1.0, size=36)
    x2 = rng.uniform(0.0, 1.0, size=36)
    x = np.column_stack([x1, x2])
    base = dict(batch_size=1, seed=0, advanced={"num_restarts": 2, "raw_samples": 8, "acq_maxiter": 8})
    base.update(config)
    return OptimizeRequest(
        dataset={"X": x.tolist(), "Y": x1.reshape(-1, 1).tolist()},
        search_space={
            "parameters": [
                {"name": "x1", "type": "float", "min": 0.0, "max": 1.0},
                {"name": "x2", "type": "float", "min": 0.0, "max": 1.0},
            ]
        },
        objectives={"f": {"goal": "min"}},
        optimization_config=OptimizationConfig(**base),
    )


def test_nearest_tested_is_zero_on_a_training_row():
    trained = np.array([[0.0, 0.0], [1.0, 1.0]])
    found = nearest_tested(np.array([1.0, 1.0]), trained, [(0.0, 1.0), (0.0, 1.0)])
    assert found["index"] == 1
    assert found["distance"] == pytest.approx(0.0)


def test_pca_sensitivity_follows_the_loadings():
    components = np.array([[1.0, 0.0], [0.0, 1.0]])
    scores = _pca_scores(np.array([0.2, 1.0]), components)
    assert scores[0] > scores[1]


def test_proposal_explains_the_sensitive_input():
    res = run_optimization(_line_request(), device="cpu")
    ranked = [item["parameter"] for item in res["diagnostics"]["sensitivities"]["parameters"]]
    assert ranked[0] == "x1"
    explanation = res["predictions"][0]["explanation"]
    assert explanation["sensitive_parameters"][0] == "x1"
    assert explanation["nearest_tested"]["distance"] >= 0.0
    assert "x1" in explanation["summary"]
    assert explanation["objectives"][0]["sd"] >= 0.0


def test_explain_endpoint_scores_the_supplied_recipe(monkeypatch):
    monkeypatch.setenv("SATOR_API_KEY", "test-key")
    get_settings.cache_clear()
    try:
        request = _line_request()
        payload = request.model_dump()
        payload["candidates"] = [{"x1": 0.25, "x2": 0.75}]
        app = create_app(Settings())
        with TestClient(app) as client:
            started = time.perf_counter()
            response = client.post("/v1/explain", json=payload, headers={"x-api-key": "test-key"})
            assert time.perf_counter() - started < 1.0
            assert response.status_code == 202, response.text
            job_id = response.json()["job_id"]
            body = None
            last = None
            for _ in range(60):
                live = client.get("/livez")
                assert live.status_code == 200
                result = client.get(f"/v1/jobs/{job_id}/result", headers={"x-api-key": "test-key"})
                last = result.json()
                if result.status_code == 200 and "predictions" in last:
                    body = last
                    break
                if last.get("status") == "FAILED":
                    break
                time.sleep(0.5)
        assert body is not None, last
        assert body["diagnostics"]["acquisition"] == "explain"
        candidate = body["predictions"][0]["candidate"]
        assert candidate["x1"] == pytest.approx(0.25, abs=1e-6)
        assert candidate["x2"] == pytest.approx(0.75, abs=1e-6)
        assert "summary" in body["predictions"][0]["explanation"]
    finally:
        get_settings.cache_clear()


def test_explain_rejects_a_candidate_missing_a_parameter(monkeypatch):
    monkeypatch.setenv("SATOR_API_KEY", "test-key")
    get_settings.cache_clear()
    try:
        payload = _line_request().model_dump()
        payload["candidates"] = [{"x1": 0.2}]
        app = create_app(Settings())
        with TestClient(app) as client:
            response = client.post("/v1/explain", json=payload, headers={"x-api-key": "test-key"})
        assert response.status_code == 422
        assert all("input" not in item for item in response.json()["detail"])
    finally:
        get_settings.cache_clear()
