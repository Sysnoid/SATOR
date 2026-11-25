"""Checks from the 25 Nov verification of commit 99f5988."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.model import ModelList

from sator_os_engine.core.models.optimize import OptimizationConfig, OptimizeRequest
from sator_os_engine.core.optimizer.mobo_engine import run_optimization
from sator_os_engine.core.optimizer.pca_model import PcaObservationModel, off_subspace_residual
from sator_os_engine.core.optimizer.preprocess import ScaledPCA
from sator_os_engine.core.optimizer.project import project_to_feasible
from sator_os_engine.runtime.worker import run_reconstruct_payload


def _budget(**extra):
    base = {"num_restarts": 2, "raw_samples": 16, "acq_maxiter": 20}
    base.update(extra)
    return base


def test_pca_list_posterior_accepts_output_indices():
    torch.manual_seed(0)
    z = torch.rand(10, 2, dtype=torch.double)
    y1 = torch.rand(10, 1, dtype=torch.double)
    y2 = torch.rand(10, 1, dtype=torch.double)
    components = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.double)
    mean = torch.zeros(3, dtype=torch.double)
    listed = ModelList(
        PcaObservationModel(SingleTaskGP(z, y1), components, mean, torch.zeros(2), torch.ones(2)),
        PcaObservationModel(SingleTaskGP(z, y2), components.clone(), mean.clone(), torch.zeros(2), torch.ones(2)),
    )
    posterior = listed.posterior(torch.rand(3, 3, dtype=torch.double), output_indices=[0, 1])
    assert posterior.mean.shape[-1] == 2


@pytest.mark.parametrize("acquisition", ["qnehvi", "parego"])
def test_multiobjective_pca_returns_a_batch(acquisition: str):
    rng = np.random.default_rng(0)
    x = rng.random((16, 3))
    y = np.column_stack([x.sum(axis=1), x[:, 0]])
    req = OptimizeRequest(
        dataset={"X": x.tolist(), "Y": y.tolist()},
        search_space={
            "parameters": [{"name": name, "type": "float", "min": 0.0, "max": 1.0} for name in ("a", "b", "c")]
        },
        objectives={"f": {"goal": "min"}, "g": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition=acquisition,
            batch_size=2,
            seed=0,
            use_pca=True,
            pca_dimension=2,
            advanced=_budget(),
        ),
    )
    res = run_optimization(req, device="cpu")
    assert len(res["predictions"]) == 2
    assert res["diagnostics"]["pca_search"] == "x"


def test_pca_candidates_stay_inside_the_training_subspace():
    rng = np.random.default_rng(1)
    latent = rng.normal(size=(28, 2))
    basis = rng.normal(size=(2, 6))
    raw = latent @ basis
    span = np.maximum(raw.max(axis=0) - raw.min(axis=0), 1e-6)
    x = (raw - raw.min(axis=0)) / span
    y = ((x[:, 0] - 0.4) ** 2).reshape(-1, 1)
    req = OptimizeRequest(
        dataset={"X": x.tolist(), "Y": y.tolist()},
        search_space={"parameters": [{"name": f"x{i}", "type": "float", "min": 0.0, "max": 1.0} for i in range(6)]},
        objectives={"f": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qei",
            batch_size=3,
            seed=1,
            use_pca=True,
            pca_dimension=2,
            advanced=_budget(),
        ),
    )
    res = run_optimization(req, device="cpu")
    pca = ScaledPCA(n_components=2).fit(x)
    cap = float(np.max(off_subspace_residual(x, pca.components_, pca.mean_, pca.scaler_mean_, pca.scaler_scale_)))
    for pred in res["predictions"]:
        row = np.array([pred["candidate"][f"x{i}"] for i in range(6)])
        residual = float(off_subspace_residual(row, pca.components_, pca.mean_, pca.scaler_mean_, pca.scaler_scale_)[0])
        assert residual <= max(cap, 1e-8) * (1.0 + 1e-3) + 1e-6
        assert pred["off_subspace_residual"] == pytest.approx(residual, rel=1e-6, abs=1e-8)
        assert np.all(np.asarray(pred["encoded"]) >= -0.05)
        assert np.all(np.asarray(pred["encoded"]) <= 1.05)


def test_multiobjective_threshold_batch_stays_feasible():
    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 1.0, size=(30, 1))
    y = np.column_stack([x.ravel(), x.ravel()])
    req = OptimizeRequest(
        dataset={"X": x.tolist(), "Y": y.tolist()},
        search_space={"parameters": [{"name": "a", "type": "float", "min": 0.0, "max": 1.0}]},
        objectives={
            "low": {"goal": "maximize_below", "threshold": {"value": 0.5}},
            "high": {"goal": "max"},
        },
        optimization_config=OptimizationConfig(
            acquisition="qnehvi",
            batch_size=4,
            seed=42,
            advanced={"num_restarts": 3, "raw_samples": 32, "acq_maxiter": 30},
        ),
    )
    res = run_optimization(req, device="cpu")
    values = [float(pred["candidate"]["a"]) for pred in res["predictions"]]
    assert values
    assert max(values) <= 0.55, values


def test_declared_ratio_rejects_a_zero_denominator():
    repaired, ok = project_to_feasible(
        np.array([0.0, 0.0]),
        [(0.0, 1.0), (0.0, 1.0)],
        None,
        [{"i": 0, "j": 1, "min_ratio": 0.5, "max_ratio": 2.0}],
    )
    assert ok
    assert repaired[1] >= 1e-4 - 1e-8


def test_reconstruct_follows_x_reference():
    rng = np.random.default_rng(2)
    x = rng.dirichlet(np.ones(3), size=16)
    pca = ScaledPCA(n_components=3).fit(x)
    scores = pca.transform(x)
    pc_mins = scores.min(axis=0)
    pc_maxs = scores.max(axis=0)
    span = np.maximum(pc_maxs - pc_mins, 1e-9)
    target = x[0]
    z_norm = (scores[0] - pc_mins) / span
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
        "x_reference": target.tolist(),
    }
    out = run_reconstruct_payload(payload)
    assert out["success"] is True
    assert np.allclose(out["reconstructed_formulation"]["combined"], target, atol=1e-8)
