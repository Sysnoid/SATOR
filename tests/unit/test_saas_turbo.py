"""MAP-SAAS surrogate and the replayed TuRBO box."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from sator_os_engine.core.models.optimize import OptimizationConfig, OptimizeRequest
from sator_os_engine.core.optimizer.explain import parameter_sensitivity
from sator_os_engine.core.optimizer.gp import build_models
from sator_os_engine.core.optimizer.kpls import fit_kpls_weights
from sator_os_engine.core.optimizer.mobo_engine import run_optimization
from sator_os_engine.core.optimizer.trust_region import replay_trust_region, thompson_candidates


def test_thompson_candidates_lie_inside_the_box():
    torch.manual_seed(0)
    train_x = torch.rand(8, 2, dtype=torch.double)
    train_y = train_x.sum(dim=-1, keepdim=True)
    from botorch.models import SingleTaskGP

    model = SingleTaskGP(train_x, train_y)
    bounds = torch.tensor([[0.2, 0.2], [0.6, 0.6]], dtype=torch.double)
    picked = thompson_candidates(model, bounds, 3, seed=1)
    assert picked.shape == (3, 2)
    assert torch.all(picked >= bounds[0] - 1e-8)
    assert torch.all(picked <= bounds[1] + 1e-8)


def _request(**config):
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, size=(12, 1))
    return OptimizeRequest(
        dataset={"X": x.tolist(), "Y": x.reshape(-1, 1).tolist()},
        search_space={"parameters": [{"name": "x", "type": "float", "min": 0.0, "max": 1.0}]},
        objectives={"f": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            batch_size=1,
            seed=0,
            advanced={"num_restarts": 2, "raw_samples": 8, "acq_maxiter": 10},
            **config,
        ),
    )


def test_default_model_and_route_are_echoed():
    res = run_optimization(_request(), device="cpu")
    assert res["diagnostics"]["model"] == "gp"
    assert res["diagnostics"]["route"] == "direct"


def test_unknown_model_and_illegal_combinations_are_rejected():
    with pytest.raises(ValidationError, match="model"):
        OptimizationConfig(model="vae")
    with pytest.raises(ValidationError, match="route"):
        OptimizationConfig(route="turbo")
    with pytest.raises(ValidationError, match="use_pca"):
        OptimizationConfig(model="saas_map", use_pca=True, pca_dimension=1)
    with pytest.raises(ValidationError, match="ilr"):
        OptimizationConfig(model="saas", mixture_transform="ilr")
    with pytest.raises(ValidationError, match="trust_region"):
        OptimizationConfig(route="trust_region", use_pca=True, pca_dimension=1)
    with pytest.raises(ValidationError, match="categorical"):
        OptimizeRequest(
            dataset={"X": [[0.0, "a"]], "Y": [[0.0]]},
            search_space={
                "parameters": [
                    {"name": "x", "type": "float", "min": 0.0, "max": 1.0},
                    {"name": "c", "type": "categorical", "choices": ["a", "b"]},
                ]
            },
            objectives={"f": {"goal": "min"}},
            optimization_config=OptimizationConfig(model="saas_map", batch_size=1),
        )


def test_map_saas_keeps_the_relevant_input_short():
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1.0, size=(36, 6))
    y = x[:, :1]
    names = [f"x{i}" for i in range(6)]
    res = run_optimization(
        OptimizeRequest(
            dataset={"X": x.tolist(), "Y": y.tolist()},
            search_space={"parameters": [{"name": name, "type": "float", "min": 0.0, "max": 1.0} for name in names]},
            objectives={"f": {"goal": "min"}},
            optimization_config=OptimizationConfig(
                model="saas_map",
                batch_size=1,
                seed=1,
                advanced={"num_restarts": 2, "raw_samples": 8, "acq_maxiter": 12},
            ),
        ),
        device="cpu",
    )
    assert res["diagnostics"]["model"] == "saas_map"
    ranked = [item["parameter"] for item in res["diagnostics"]["sensitivities"]["parameters"]]
    assert ranked[0] == "x0"


def test_map_saas_respects_a_sum_constraint():
    rng = np.random.default_rng(2)
    x = rng.dirichlet(np.ones(3), size=24)
    y = ((x[:, 0] - 0.2) ** 2).reshape(-1, 1)
    res = run_optimization(
        OptimizeRequest(
            dataset={"X": x.tolist(), "Y": y.tolist()},
            search_space={
                "parameters": [{"name": name, "type": "float", "min": 0.0, "max": 1.0} for name in ("a", "b", "c")]
            },
            objectives={"f": {"goal": "min"}},
            optimization_config=OptimizationConfig(
                model="saas_map",
                batch_size=2,
                seed=2,
                sum_constraints=[{"indices": [0, 1, 2], "target_sum": 1.0}],
                advanced={"num_restarts": 2, "raw_samples": 12, "acq_maxiter": 20},
            ),
        ),
        device="cpu",
    )
    for pred in res["predictions"]:
        total = pred["candidate"]["a"] + pred["candidate"]["b"] + pred["candidate"]["c"]
        assert abs(total - 1.0) < 1e-4


def test_trust_region_grows_after_successes_and_shrinks_after_failures():
    bounds = [(0.0, 1.0)]
    improving = np.linspace(1.0, 0.0, 7).reshape(-1, 1)
    grown = replay_trust_region(improving, improving, bounds, np.array([-1.0]), success_tolerance=3)
    assert grown["length"] > 0.8
    assert grown["center"] == [0.0]
    noisy = np.array([[0.2], [0.9], [0.8], [0.7], [0.6], [0.5]])
    shrunk = replay_trust_region(
        noisy,
        noisy,
        bounds,
        np.array([-1.0]),
        failure_tolerance=2,
        length_min=0.01,
    )
    assert shrunk["length"] < 0.8
    assert shrunk["bounds"][0][0] >= 0.0
    assert shrunk["bounds"][0][1] <= 1.0
    again = replay_trust_region(noisy, noisy, bounds, np.array([-1.0]), failure_tolerance=2, length_min=0.01)
    assert again == shrunk


def test_trust_region_search_stays_inside_the_replayed_box():
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 1.0, size=(20, 4))
    # The first row is the best. The rest are worse, so the box shrinks around it.
    y = np.linalg.norm(x - x[0], axis=1)
    y[0] = 0.0
    res = run_optimization(
        OptimizeRequest(
            dataset={"X": x.tolist(), "Y": y.reshape(-1, 1).tolist()},
            search_space={"parameters": [{"name": f"x{i}", "type": "float", "min": 0.0, "max": 1.0} for i in range(4)]},
            objectives={"f": {"goal": "min"}},
            optimization_config=OptimizationConfig(
                route="trust_region",
                batch_size=2,
                seed=3,
                gp_config={"tr_failure_tolerance": 3, "tr_length_min": 0.05},
                advanced={"num_restarts": 2, "raw_samples": 16, "acq_maxiter": 20},
            ),
        ),
        device="cpu",
    )
    box = res["diagnostics"]["trust_region"]["bounds"]
    for pred in res["predictions"]:
        for i, (lo, hi) in enumerate(box):
            value = pred["candidate"][f"x{i}"]
            assert lo - 1e-5 <= value <= hi + 1e-5, (i, value, lo, hi)


def test_saas_nuts_returns_a_finite_relevance_vector():
    rng = np.random.default_rng(4)
    x = rng.uniform(0.0, 1.0, size=(12, 4))
    y = x[:, :1]
    bounds = torch.tensor([[0.0] * 4, [1.0] * 4], dtype=torch.double)
    cfg = OptimizationConfig(
        model="saas",
        batch_size=1,
        gp_config={"warmup_steps": 2, "num_samples": 2, "thinning": 1},
    )
    model = build_models(torch.tensor(x, dtype=torch.double), torch.tensor(y, dtype=torch.double), cfg, bounds=bounds)
    posterior = model.posterior(torch.tensor(x[:2], dtype=torch.double))
    assert torch.isfinite(posterior.mean).all()
    names = [f"x{i}" for i in range(4)]
    report = parameter_sensitivity(model, names, n_obj=1, space="input")
    assert len(report["parameters"]) == 4
    assert all(np.isfinite(item["sensitivity"]) for item in report["parameters"])


@pytest.mark.slow
def test_saas_production_budget_fits():
    rng = np.random.default_rng(5)
    x = rng.uniform(0.0, 1.0, size=(10, 3))
    bounds = torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.double)
    cfg = OptimizationConfig(model="saas", batch_size=1)
    model = build_models(
        torch.tensor(x, dtype=torch.double),
        torch.tensor(x[:, :1], dtype=torch.double),
        cfg,
        bounds=bounds,
    )
    posterior = model.posterior(torch.tensor(x[:1], dtype=torch.double))
    assert torch.isfinite(posterior.mean).all()


def test_kpls_direction_follows_the_outcome():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 6))
    y = x[:, 0] - x[:, 1]
    weights, rank = fit_kpls_weights(x, y, None)
    assert rank == 4
    largest = set(np.argsort(-np.abs(weights[:, 0]))[:2])
    assert largest == {0, 1}


def test_kpls_optimize_stays_feasible_and_reports_rank():
    rng = np.random.default_rng(6)
    raw = rng.dirichlet(np.ones(3), size=24)
    # Pad two noise columns so the PLS rank is meaningful, then drop them from the sum.
    noise = rng.normal(size=(24, 2))
    x = np.column_stack([raw, noise])
    y = (x[:, 0] - x[:, 1]).reshape(-1, 1)
    res = run_optimization(
        OptimizeRequest(
            dataset={"X": x.tolist(), "Y": y.tolist()},
            search_space={
                "parameters": [
                    {"name": name, "type": "float", "min": 0.0, "max": 1.0} for name in ("a", "b", "c", "n1", "n2")
                ]
            },
            objectives={"f": {"goal": "min"}},
            optimization_config=OptimizationConfig(
                model="kpls",
                batch_size=2,
                seed=6,
                gp_config={"kpls_rank": 2},
                sum_constraints=[{"indices": [0, 1, 2], "target_sum": 1.0}],
                advanced={"num_restarts": 2, "raw_samples": 12, "acq_maxiter": 20},
            ),
        ),
        device="cpu",
    )
    assert res["diagnostics"]["kpls"]["f"]["k"] == 2
    assert "a" in res["diagnostics"]["kpls"]["f"]["parameters"][:3]
    for pred in res["predictions"]:
        total = pred["candidate"]["a"] + pred["candidate"]["b"] + pred["candidate"]["c"]
        assert abs(total - 1.0) < 1e-4
        for name in ("a", "b", "c", "n1", "n2"):
            assert 0.0 <= pred["candidate"][name] <= 1.0
