"""Lab roadmap: noisy acquisition, known noise, outcome constraints, failures, design."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sator_os_engine.api.app import create_app
from sator_os_engine.core.models.optimize import OptimizationConfig, OptimizeRequest
from sator_os_engine.core.optimizer.mobo_engine import run_optimization
from sator_os_engine.core.optimizer.observations import prepare_observations
from sator_os_engine.settings import Settings, get_settings


def _line(goal: str = "min", **config):
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, size=(28, 1))
    base = dict(batch_size=1, seed=0, advanced={"num_restarts": 3, "raw_samples": 16, "acq_maxiter": 25})
    base.update(config)
    return OptimizeRequest(
        dataset={"X": x.tolist(), "Y": x.tolist()},
        search_space={"parameters": [{"name": "x", "type": "float", "min": 0.0, "max": 1.0}]},
        objectives={"f": {"goal": goal}},
        optimization_config=OptimizationConfig(**base),
    )


def test_auto_in_lab_mode_selects_noisy_ei():
    res = run_optimization(_line(acquisition="auto", mode="lab"), device="cpu")
    assert res["diagnostics"]["acquisition"] == "qlognei"


def test_auto_in_throughput_mode_selects_log_ei():
    res = run_optimization(_line(acquisition="auto", mode="throughput"), device="cpu")
    assert res["diagnostics"]["acquisition"] == "qlogei"


def test_replicate_rows_become_fixed_noise():
    x = np.array([[0.2], [0.2], [0.8], [0.8]], dtype=float)
    y = np.array([[0.1], [0.3], [0.0], [0.2]], dtype=float)
    prepared = prepare_observations(x, y, None, None)
    assert prepared["collapsed"] == 2
    assert prepared["fixed_noise"] is True
    # Sample variance of (0.1, 0.3) is 0.02.
    assert prepared["Yvar"][0, 0] == pytest.approx(0.02)

    req = OptimizeRequest(
        dataset={"X": x.tolist(), "Y": y.tolist()},
        search_space={"parameters": [{"name": "x", "type": "float", "min": 0.0, "max": 1.0}]},
        objectives={"f": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qei", batch_size=1, seed=1, advanced={"num_restarts": 2, "raw_samples": 8, "acq_maxiter": 15}
        ),
    )
    res = run_optimization(req, device="cpu")
    assert res["diagnostics"]["fixed_noise"] is True
    assert res["diagnostics"]["rows_collapsed"] == 2
    assert res["predictions"]


def test_outcome_constraint_upper_bound_holds():
    req = _line("max")
    req.objectives["f"]["outcome_constraint"] = {"upper": 0.4}
    res = run_optimization(req, device="cpu")
    assert float(res["predictions"][0]["candidate"]["x"]) <= 0.55


def test_failed_batches_keep_the_proposal_in_the_successful_region():
    rng = np.random.default_rng(2)
    x = rng.uniform(0.0, 1.0, size=(32, 1))
    y = x.copy()
    failed = (x[:, 0] > 0.6).tolist()
    req = OptimizeRequest(
        dataset={"X": x.tolist(), "Y": y.tolist(), "failed": failed},
        search_space={"parameters": [{"name": "x", "type": "float", "min": 0.0, "max": 1.0}]},
        objectives={"f": {"goal": "max"}},
        optimization_config=OptimizationConfig(
            acquisition="qei", batch_size=1, seed=2, advanced={"num_restarts": 4, "raw_samples": 24, "acq_maxiter": 30}
        ),
    )
    res = run_optimization(req, device="cpu")
    assert res["diagnostics"]["feasibility_model"] is True
    assert float(res["predictions"][0]["candidate"]["x"]) <= 0.75


def test_design_endpoint_respects_the_sum(monkeypatch):
    monkeypatch.setenv("SATOR_API_KEY", "test-key")
    get_settings.cache_clear()
    app = create_app(Settings())
    payload = {
        "n": 5,
        "seed": 3,
        "search_space": {
            "parameters": [
                {"name": "a", "type": "float", "min": 0.0, "max": 1.0},
                {"name": "b", "type": "float", "min": 0.0, "max": 1.0},
                {"name": "c", "type": "float", "min": 0.0, "max": 1.0},
            ]
        },
        "sum_constraints": [{"indices": [0, 1, 2], "target_sum": 1.0}],
        "max_nonzero": 2,
    }
    with TestClient(app) as client:
        response = client.post("/v1/design", json=payload, headers={"x-api-key": "test-key"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["candidates"]) == 5
    for row in body["candidates"]:
        assert abs(row["a"] + row["b"] + row["c"] - 1.0) < 1e-5
        assert sum(value > 1e-6 for value in row.values()) <= 2
