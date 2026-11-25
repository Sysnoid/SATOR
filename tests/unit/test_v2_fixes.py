"""Checks from the 25 Nov verification of commit 959ef68."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from sator_os_engine.api.app import create_app
from sator_os_engine.core.models.optimize import OptimizationConfig, OptimizeRequest
from sator_os_engine.core.models.reconstruct import ReconstructionRequest
from sator_os_engine.core.optimizer import acquisition as acquisition_module
from sator_os_engine.core.optimizer.gp import _data_envelope
from sator_os_engine.core.optimizer.mobo_engine import run_optimization
from sator_os_engine.core.optimizer.preprocess import ScaledPCA, scaled_pca_inverse
from sator_os_engine.core.optimizer.project import project_to_feasible
from sator_os_engine.runtime.worker import run_reconstruct_payload
from sator_os_engine.settings import Settings, get_settings


def test_scaled_pca_inverse_matches_inverse_transform():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 4)) * np.array([0.2, 1.0, 5.0, 0.01])
    pca = ScaledPCA(n_components=2).fit(x)
    z = pca.transform(x[:3])
    for row in z:
        recovered = scaled_pca_inverse(row, pca.components_, pca.mean_, pca.scaler_mean_, pca.scaler_scale_)
        direct = pca.inverse_transform(row.reshape(1, -1)).reshape(-1)
        assert np.allclose(recovered, direct, atol=1e-8)


def test_pca_candidates_are_encoded_exactly():
    rng = np.random.default_rng(1)
    x = rng.dirichlet(np.ones(3), size=24)
    y = np.sum((x - np.array([0.2, 0.3, 0.5])) ** 2, axis=1, keepdims=True)
    params = [{"name": name, "type": "float", "min": 0.0, "max": 1.0} for name in ("a", "b", "c")]
    req = OptimizeRequest(
        dataset={"X": x.tolist(), "Y": y.tolist()},
        search_space={"parameters": params},
        objectives={"cost": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qei",
            batch_size=3,
            seed=1,
            use_pca=True,
            pca_dimension=2,
            sum_constraints=[{"indices": [0, 1, 2], "target_sum": 1.0}],
            advanced={"num_restarts": 2, "raw_samples": 12, "acq_maxiter": 20},
        ),
    )
    res = run_optimization(req, device="cpu")
    assert res["diagnostics"]["pca_search"] == "x"
    for pred in res["predictions"]:
        total = sum(pred["candidate"][name] for name in ("a", "b", "c"))
        assert abs(total - 1.0) < 1e-4
        assert np.allclose(pred["encoded"], pred["encoded_actual"], atol=1e-3)
        assert pred["reconstructed"]["on_target"] is True


def test_data_envelope_stays_inside_the_search_box_and_near_the_data():
    data = torch.tensor([[0.1, 0.2], [0.15, 0.25], [0.12, 0.22]], dtype=torch.double)
    box = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    env = _data_envelope(data, box, widen=0.1)
    assert torch.all(env[0] >= box[0] - 1e-12)
    assert torch.all(env[1] <= box[1] + 1e-12)
    assert float(env[1, 0] - env[0, 0]) < 0.5


def test_identical_pca_rows_are_rejected():
    row = [0.2, 0.3, 0.5]
    with pytest.raises(ValueError, match="degenerate"):
        OptimizeRequest(
            dataset={"X": [row, row, row], "Y": [[1.0], [2.0], [3.0]]},
            search_space={
                "parameters": [
                    {"name": "a", "type": "float", "min": 0.0, "max": 1.0},
                    {"name": "b", "type": "float", "min": 0.0, "max": 1.0},
                    {"name": "c", "type": "float", "min": 0.0, "max": 1.0},
                ]
            },
            objectives={"f": {"goal": "min"}},
            optimization_config=OptimizationConfig(use_pca=True, pca_dimension=1, batch_size=1),
        )


def test_reconstruct_endpoint_matches_the_scaled_inverse():
    assert ReconstructionRequest.model_fields["target_precision"].default == 1e-3
    rng = np.random.default_rng(2)
    x = rng.dirichlet(np.ones(3), size=16)
    pca = ScaledPCA(n_components=3).fit(x)
    scores = pca.transform(x)
    pc_mins = scores.min(axis=0)
    pc_maxs = scores.max(axis=0)
    span = np.maximum(pc_maxs - pc_mins, 1e-9)
    z_norm = (scores[0] - pc_mins) / span
    direct = pca.inverse_transform(scores[0].reshape(1, -1)).reshape(-1)
    payload = {
        "coordinates": z_norm.tolist(),
        "pca_info": {
            "pc_mins": pc_mins.tolist(),
            "pc_maxs": pc_maxs.tolist(),
            "components": pca.components_.tolist(),
            "mean": pca.mean_.tolist(),
            "scaler_mean": pca.scaler_mean_.tolist(),
            "scaler_scale": pca.scaler_scale_.tolist(),
        },
        "bounds": {"ingredients": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], "parameters": []},
        "n_ingredients": 3,
        "sum_groups": [{"indices": [0, 1, 2], "target": 1.0}],
        "variable_bounds": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        "column_names": ["a", "b", "c"],
    }
    out = run_reconstruct_payload(payload)
    assert out["success"] is True
    assert np.allclose(out["reconstructed_formulation"]["combined"], direct, atol=1e-6)


def test_three_objective_batches_use_cached_noisy_ehvi(monkeypatch):
    seen: dict[str, object] = {}
    real = acquisition_module.qLogNoisyExpectedHypervolumeImprovement

    class _Spy(real):
        def __init__(self, *args, **kwargs):
            seen["cache_root"] = kwargs.get("cache_root")
            seen["prune_baseline"] = kwargs.get("prune_baseline")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(acquisition_module, "qLogNoisyExpectedHypervolumeImprovement", _Spy)
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 1.0, size=(12, 2))
    y = np.column_stack([x.sum(axis=1), x[:, 0], x[:, 1]])
    req = OptimizeRequest(
        dataset={"X": x.tolist(), "Y": y.tolist()},
        search_space={
            "parameters": [
                {"name": "x1", "type": "float", "min": 0.0, "max": 1.0},
                {"name": "x2", "type": "float", "min": 0.0, "max": 1.0},
            ]
        },
        objectives={"a": {"goal": "min"}, "b": {"goal": "min"}, "c": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qnehvi",
            batch_size=2,
            seed=3,
            advanced={"num_restarts": 2, "raw_samples": 8, "acq_maxiter": 8},
        ),
    )
    run_optimization(req, device="cpu")
    assert seen["cache_root"] is True
    assert seen["prune_baseline"] is True


def test_validation_errors_omit_the_request_body(monkeypatch):
    monkeypatch.setenv("SATOR_API_KEY", "test-key")
    get_settings.cache_clear()
    try:
        app = create_app(Settings())
        with TestClient(app) as client:
            response = client.post(
                "/v1/optimize",
                json={"dataset": "not-a-dataset"},
                headers={"x-api-key": "test-key"},
            )
        assert response.status_code == 422, response.text
        body = response.json()
        assert isinstance(body["detail"], list)
        assert all("input" not in item for item in body["detail"])
    finally:
        get_settings.cache_clear()


def test_min_denominator_keeps_the_ratio_defined():
    x, ok = project_to_feasible(
        np.array([0.0, 0.0]),
        [(0.0, 1.0), (0.0, 1.0)],
        None,
        [{"i": 0, "j": 1, "min_ratio": 0.5, "max_ratio": 2.0, "min_denominator": 0.1}],
    )
    assert ok
    assert x[1] >= 0.1 - 1e-8
