"""Verify ``acquisition`` / ``algorithm`` names map to distinct BoTorch acquisition classes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sator_os_engine.core.optimizer.acquisition import select_candidates_multiobjective
from sator_os_engine.core.optimizer.gp import bounds_input, build_models


def _two_obj_setup(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(40, 2))
    f1 = np.sum((X - 0.25) ** 2, axis=1)
    f2 = -np.sum((X - 0.75) ** 2, axis=1)
    Y = np.stack([f1, f2], axis=1)
    X_t = torch.tensor(X, dtype=torch.double)
    Y_t = torch.tensor(Y, dtype=torch.double)
    model = build_models(X_t, Y_t, SimpleNamespace(gp_config={"noise": 1e-5}))
    return model, X, Y


@pytest.mark.parametrize(
    "acq_name, expected_type",
    [
        ("qehvi", "qExpectedHypervolumeImprovement"),
        ("qnehvi", "qLogExpectedHypervolumeImprovement"),
        ("parego", "qLogExpectedImprovement"),
        ("qpi", "qProbabilityOfImprovement"),
        ("qucb", "qUpperConfidenceBound"),
        ("qnoisyehvi", "qLogNoisyExpectedHypervolumeImprovement"),
    ],
)
def test_multi_objective_acquisition_class_by_name(monkeypatch: pytest.MonkeyPatch, acq_name: str, expected_type: str):
    model, X, Y = _two_obj_setup(seed=5)
    params = [
        {"name": "x1", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "x2", "type": "float", "min": 0.0, "max": 1.0},
    ]
    tdtype = torch.double
    tdevice = torch.device("cpu")
    b_in = bounds_input(params, tdtype, tdevice)
    tX = torch.tensor(X, dtype=tdtype, device=tdevice)
    goals = ["min", "max"]
    req = SimpleNamespace(
        objectives={"o1": {"goal": "min"}, "o2": {"goal": "max"}},
        optimization_config=SimpleNamespace(
            acquisition=acq_name,
            sum_constraints=[],
            ratio_constraints=[],
        ),
    )
    captured: dict = {}

    def fake_optimize_acqf(acqf, bounds, q, num_restarts, raw_samples, inequality_constraints=None, options=None):
        captured["type"] = type(acqf).__name__
        d_model = bounds.shape[1]
        return torch.full((q, d_model), 0.5, dtype=tdtype), None

    monkeypatch.setattr("sator_os_engine.core.optimizer.acquisition.optimize_acqf", fake_optimize_acqf)

    select_candidates_multiobjective(
        model=model,
        params=params,
        bounds_input=b_in,
        bounds_model=b_in,
        use_pca_model=False,
        pca=None,
        pc_mins=None,
        pc_range=None,
        n=2,
        rng_seed=99,
        tdtype=tdtype,
        tdevice=tdevice,
        req=req,
        goals=goals,
        Y_np=Y,
        train_X=tX,
    )
    assert captured.get("type") == expected_type
