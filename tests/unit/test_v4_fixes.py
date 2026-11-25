"""Checks from the 25 Nov verification of commit ea6e329."""

from __future__ import annotations

import json

import numpy as np

from sator_os_engine.core.models.optimize import OptimizationConfig, OptimizeRequest
from sator_os_engine.core.optimizer.mobo_engine import run_optimization
from sator_os_engine.core.optimizer.pca_model import off_subspace_residual, pull_toward_subspace
from sator_os_engine.core.optimizer.preprocess import ScaledPCA
from sator_os_engine.runtime.json_safe import sanitize_for_json


def _tiny(**config):
    base = dict(batch_size=1, seed=0, advanced={"num_restarts": 2, "raw_samples": 8, "acq_maxiter": 12})
    base.update(config)
    x = [[0.1], [0.2], [0.85]]
    return x, base


def test_nan_training_row_stays_json_serialisable():
    x, base = _tiny()
    for columns in (1, 2):
        y = [[0.1] * columns, [None] + [0.2] * (columns - 1), [0.9] * columns]
        names = [f"f{i}" for i in range(columns)]
        res = run_optimization(
            OptimizeRequest(
                dataset={"X": x, "Y": y},
                search_space={"parameters": [{"name": "a", "type": "float", "min": 0.0, "max": 1.0}]},
                objectives={name: {"goal": "min"} for name in names},
                optimization_config=OptimizationConfig(**base),
            ),
            device="cpu",
        )
        json.dumps(res, allow_nan=False)
        nearest = res["predictions"][0]["explanation"]["nearest_tested"]
        assert nearest["index"] != 1
        assert all(value is None or np.isfinite(value) for value in nearest["objectives"])


def test_job_result_encoder_turns_nan_into_null():
    cleaned = sanitize_for_json({"value": float("nan"), "nested": [float("inf"), 1.0]})
    assert cleaned == {"value": None, "nested": [None, 1.0]}
    json.dumps(cleaned, allow_nan=False)


def test_pull_toward_subspace_keeps_the_scores_and_meets_the_cap():
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(12, 1))
    basis = rng.normal(size=(1, 4))
    plane = latent @ basis
    pca = ScaledPCA(n_components=1).fit(plane)
    drifted = plane[0] + rng.normal(size=4)
    before = float(off_subspace_residual(drifted, pca.components_, pca.mean_, pca.scaler_mean_, pca.scaler_scale_)[0])
    cap = before * 0.25
    pulled = pull_toward_subspace(
        drifted, pca.components_, pca.mean_, pca.scaler_mean_, pca.scaler_scale_, cap
    ).reshape(-1)
    after = float(off_subspace_residual(pulled, pca.components_, pca.mean_, pca.scaler_mean_, pca.scaler_scale_)[0])
    assert after <= cap * (1.0 + 1e-6) + 1e-8
    assert after < before


def test_explain_keeps_an_infeasible_recipe():
    x = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]])
    y = x[:, :1]
    supplied = np.array([[1.3, 0.0]])
    res = run_optimization(
        OptimizeRequest(
            dataset={"X": x.tolist(), "Y": y.tolist()},
            search_space={
                "parameters": [
                    {"name": "a", "type": "float", "min": 0.0, "max": 1.0},
                    {"name": "b", "type": "float", "min": 0.0, "max": 1.0},
                ]
            },
            objectives={"f": {"goal": "min"}},
            optimization_config=OptimizationConfig(
                batch_size=1,
                seed=0,
                sum_constraints=[{"indices": [0, 1], "target_sum": 1.0}],
                advanced={"num_restarts": 2, "raw_samples": 8, "acq_maxiter": 12},
            ),
        ),
        device="cpu",
        explain_rows=supplied,
    )
    candidate = res["predictions"][0]["candidate"]
    assert candidate["a"] == 1.3
    assert candidate["b"] == 0.0
    assert res["predictions"][0]["supplied_feasible"] is False
    assert any("sum" in item or "bounds" in item for item in res["predictions"][0]["violations"])
