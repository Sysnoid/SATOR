"""Replicate wells, the best-so-far control, dataset checks, and the polling client."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sator_os_engine.api.app import create_app
from sator_os_engine.client import SatorClient, SatorJobError
from sator_os_engine.core.models.optimize import OptimizationConfig, OptimizeRequest
from sator_os_engine.core.optimizer.mobo_engine import run_optimization
from sator_os_engine.core.optimizer.replicates import best_observed_index, expand_replicates
from sator_os_engine.core.optimizer.trust_region import replay_trust_region
from sator_os_engine.settings import Settings, get_settings


def test_best_row_matches_the_trust_region_center():
    x = np.array([[0.2, 0.8], [0.5, 0.5], [0.1, 0.1]])
    y = np.array([[0.9, 0.9], [0.4, 0.5], [0.1, 0.2]])
    signs = np.array([-1.0, -1.0])
    index = best_observed_index(y, signs)
    state = replay_trust_region(x, y, [(0.0, 1.0), (0.0, 1.0)], signs)
    assert np.allclose(x[index], state["center"])


def test_expand_replicates_keeps_a_single_control_well():
    rows = expand_replicates(
        [{"candidate": {"a": 0.2}}, {"candidate": {"a": 0.9}, "control": True}],
        3,
    )
    assert [row["replicate"] for row in rows] == [1, 2, 3, 1]
    assert rows[-1]["control"] is True
    assert rows[0]["candidate"] == rows[2]["candidate"]


def test_batch_repeats_recipes_and_appends_the_best():
    x = np.linspace(0.1, 0.9, 8).reshape(-1, 1)
    res = run_optimization(
        OptimizeRequest(
            dataset={"X": x.tolist(), "Y": x.tolist()},
            search_space={"parameters": [{"name": "x", "type": "float", "min": 0.0, "max": 1.0}]},
            objectives={"f": {"goal": "min"}},
            optimization_config=OptimizationConfig(
                batch_size=2,
                n_replicates=2,
                repeat_best=True,
                seed=0,
                advanced={"num_restarts": 2, "raw_samples": 8, "acq_maxiter": 10},
            ),
        ),
        device="cpu",
    )
    preds = res["predictions"]
    controls = [pred for pred in preds if pred.get("control")]
    fresh = [pred for pred in preds if not pred.get("control")]
    assert len(controls) == 1
    assert abs(float(controls[0]["candidate"]["x"]) - 0.1) < 1e-8
    assert controls[0]["replicate"] == 1
    assert len(fresh) == 4
    assert fresh[0]["candidate"] == fresh[1]["candidate"]
    assert [pred["replicate"] for pred in fresh] == [1, 2, 1, 2]
    assert res["pareto"]["indices"]
    for index in res["pareto"]["indices"]:
        assert preds[index]["replicate"] == 1


def test_validate_reports_dataset_problems(monkeypatch):
    monkeypatch.setenv("SATOR_API_KEY", "test-key")
    get_settings.cache_clear()
    try:
        payload = {
            "dataset": {
                "X": [
                    [0.0, 0.0, 0.2],
                    [0.5, 1.0, 0.2],
                    [1.0, 2.0, 0.2],
                    [0.0, 0.0, 0.2],
                    [1.5, 3.0, 0.2],
                ],
                "Y": [[1.0], [1.0], [None], [1.0], [1.0]],
            },
            "search_space": {
                "parameters": [
                    {"name": "a", "type": "float", "min": 0.0, "max": 1.0},
                    {"name": "b", "type": "float", "min": 0.0, "max": 1.0},
                    {"name": "c", "type": "float", "min": 0.0, "max": 1.0},
                ]
            },
            "sum_constraints": [{"indices": [0, 1, 2], "target_sum": 1.0}],
        }
        app = create_app(Settings())
        with TestClient(app) as client:
            response = client.post("/v1/validate", json=payload, headers={"x-api-key": "test-key"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["ok"] is False
        codes = {issue["code"] for issue in body["issues"]}
        assert {"duplicate", "out_of_bounds", "nan", "sum", "constant", "collinear"} <= codes
    finally:
        get_settings.cache_clear()


def test_client_polls_until_the_result_is_ready():
    import httpx

    calls = {"status": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "lab-key"
        if request.url.path == "/v1/optimize":
            return httpx.Response(202, json={"job_id": "job_1", "status": "QUEUED"})
        if request.url.path == "/v1/jobs/job_1":
            calls["status"] += 1
            state = "RUNNING" if calls["status"] < 2 else "COMPLETED"
            return httpx.Response(200, json={"job_id": "job_1", "status": state})
        if request.url.path == "/v1/jobs/job_1/result":
            return httpx.Response(200, json={"predictions": [{"candidate": {"a": 0.4}, "objectives": [1.0]}]})
        return httpx.Response(404)

    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://sator.test",
        headers={"x-api-key": "lab-key"},
    )
    with SatorClient("http://sator.test", "lab-key", http=http) as client:
        result = client.suggest({"dataset": {}}, timeout=5, interval=0)
    assert result["predictions"][0]["candidate"]["a"] == 0.4
    assert calls["status"] == 2


def test_client_raises_when_the_job_fails():
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/jobs/job_9"):
            return httpx.Response(200, json={"job_id": "job_9", "status": "FAILED", "error": "bad data"})
        return httpx.Response(202, json={"job_id": "job_9", "status": "QUEUED"})

    http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://sator.test")
    client = SatorClient("http://sator.test", "lab-key", http=http)
    with pytest.raises(SatorJobError, match="bad data"):
        client.wait("job_9", timeout=1, interval=0)


def test_predictions_frame_round_trips_a_table():
    pandas = pytest.importorskip("pandas")
    from sator_os_engine.client import dataset_from_frame, predictions_frame

    frame = pandas.DataFrame({"a": [0.2, 0.8], "yield": [1.0, 3.0]})
    dataset = dataset_from_frame(frame, ["yield"])
    assert dataset["X"] == [[0.2], [0.8]]
    assert dataset["Y"] == [[1.0], [3.0]]
    table = predictions_frame(
        {"predictions": [{"candidate": {"a": 0.2}, "objectives": [1.5], "replicate": 2, "control": True}]}
    )
    assert table.loc[0, "a"] == 0.2
    assert bool(table.loc[0, "control"]) is True
    assert int(table.loc[0, "replicate"]) == 2
