"""Correctness fixes from the 15 Nov 2026 benchmark notes (F1–F4, F7, F12–F16, F21, F22)."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from sator_os_engine.core.models.optimize import OptimizationConfig, OptimizeRequest
from sator_os_engine.core.optimizer.acquisition import _reference_point
from sator_os_engine.core.optimizer.mobo_engine import run_optimization
from sator_os_engine.core.optimizer.utils import goal_sign
from sator_os_engine.reconstruction.slsqp_reconstructor import reconstruct as slsqp_reconstruct
from sator_os_engine.runtime.executor import Executor
from sator_os_engine.runtime.jobs import JobStatus, JobStore


def _linear_request(
    goal: str, *, threshold: float | None = None, weight: float = 1.0, seed: int | None = 0, n: int = 36
):
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, size=(n, 1))
    y = x.copy()
    obj: dict = {"goal": goal}
    if threshold is not None:
        obj["threshold"] = {"value": threshold, "weight": weight}
    return OptimizeRequest(
        dataset={"X": x.tolist(), "Y": y.tolist()},
        search_space={"parameters": [{"name": "x", "type": "float", "min": 0.0, "max": 1.0}]},
        objectives={"f": obj},
        optimization_config=OptimizationConfig(
            acquisition="qei",
            batch_size=1,
            seed=seed,
            advanced={"num_restarts": 4, "raw_samples": 32, "acq_maxiter": 40},
        ),
    )


@pytest.mark.parametrize(
    ("goal", "high"),
    [
        ("max", True),
        ("maximize", True),
        ("min", False),
        ("minimize", False),
    ],
)
def test_goal_spellings_seek_the_matching_corner(goal: str, high: bool):
    res = run_optimization(_linear_request(goal), device="cpu")
    x = float(res["predictions"][0]["candidate"]["x"])
    if high:
        assert x > 0.8, x
    else:
        assert x < 0.2, x


def test_goal_sign_matches_the_documented_sets():
    for name in ("max", "maximize", "maximize_above", "maximize_below"):
        assert goal_sign(name) == 1.0
    for name in ("min", "minimize", "minimize_below", "minimize_above"):
        assert goal_sign(name) == -1.0
    with pytest.raises(ValueError, match="unknown goal"):
        goal_sign("maximise")


def test_unknown_goal_and_acquisition_are_rejected():
    with pytest.raises(ValidationError, match="unknown goal"):
        _linear_request("maximise")
    with pytest.raises(ValidationError, match="unknown acquisition"):
        OptimizeRequest(
            dataset={"X": [[0.0], [1.0]], "Y": [[0.0], [1.0]]},
            search_space={"parameters": [{"name": "x", "type": "float", "min": 0.0, "max": 1.0}]},
            objectives={"f": {"goal": "min"}},
            optimization_config=OptimizationConfig(acquisition="qlogeiX", batch_size=1, seed=0),
        )


def test_batch_size_is_capped():
    with pytest.raises(ValidationError):
        OptimizationConfig(batch_size=32)


@pytest.mark.parametrize("weight", [0.25, 1.0, 5.0])
def test_maximize_below_stays_at_the_threshold(weight: float):
    res = run_optimization(_linear_request("maximize_below", threshold=0.5, weight=weight), device="cpu")
    x = float(res["predictions"][0]["candidate"]["x"])
    assert abs(x - 0.5) < 0.15, x


def test_minimize_above_stays_at_the_threshold():
    res = run_optimization(_linear_request("minimize_above", threshold=0.5), device="cpu")
    x = float(res["predictions"][0]["candidate"]["x"])
    assert abs(x - 0.5) < 0.15, x


def test_minimize_below_stays_in_the_feasible_region():
    res = run_optimization(_linear_request("minimize_below", threshold=0.5), device="cpu")
    x = float(res["predictions"][0]["candidate"]["x"])
    assert x <= 0.55, x


def test_reference_point_scales_with_the_objective():
    y = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=torch.double)
    y_small = y.clone()
    y_small[:, 1] *= 1e-4
    signs = np.array([1.0, 1.0])
    req = _linear_request("min")
    rp = _reference_point(y, req, signs)
    rp_small = _reference_point(y_small, req, signs)
    span = y.max(dim=0).values - y.min(dim=0).values
    span_small = y_small.max(dim=0).values - y_small.min(dim=0).values
    rel = (rp - y.min(dim=0).values) / span.clamp_min(1e-6)
    rel_small = (rp_small - y_small.min(dim=0).values) / span_small.clamp_min(1e-6)
    assert torch.allclose(rel, rel_small, atol=1e-6)
    assert torch.allclose(rel, torch.full_like(rel, -0.1), atol=1e-6)


def test_nan_row_is_dropped_and_reported():
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, 1.0, size=(24, 1))
    y = x.copy()
    y[3, 0] = np.nan
    req = OptimizeRequest(
        dataset={"X": x.tolist(), "Y": y.tolist()},
        search_space={"parameters": [{"name": "x", "type": "float", "min": 0.0, "max": 1.0}]},
        objectives={"f": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qei", batch_size=1, seed=1, advanced={"num_restarts": 3, "raw_samples": 16, "acq_maxiter": 30}
        ),
    )
    res = run_optimization(req, device="cpu")
    assert res["diagnostics"]["dropped_rows"] == 1
    assert res["predictions"]


def test_missing_seed_is_echoed_and_reproducible():
    first = run_optimization(_linear_request("min", seed=None), device="cpu")
    second = run_optimization(_linear_request("min", seed=None), device="cpu")
    seed = first["diagnostics"]["seed"]
    assert seed != second["diagnostics"]["seed"]
    again = run_optimization(_linear_request("min", seed=seed), device="cpu")
    assert again["predictions"][0]["candidate"] == first["predictions"][0]["candidate"]


def test_int_and_categorical_candidates_are_valid_levels():
    rng = np.random.default_rng(2)
    rows = []
    ys = []
    choices = ["A", "B", "C"]
    for _ in range(30):
        i = int(rng.integers(0, 6))
        c = str(rng.choice(choices))
        rows.append([i, c])
        ys.append([float(i) + choices.index(c)])
    req = OptimizeRequest(
        dataset={"X": rows, "Y": ys},
        search_space={
            "parameters": [
                {"name": "n", "type": "int", "min": 0, "max": 5},
                {"name": "cat", "type": "categorical", "choices": choices},
            ]
        },
        objectives={"f": {"goal": "max"}},
        optimization_config=OptimizationConfig(
            acquisition="qei", batch_size=2, seed=3, advanced={"num_restarts": 3, "raw_samples": 16, "acq_maxiter": 20}
        ),
    )
    res = run_optimization(req, device="cpu")
    for pred in res["predictions"]:
        assert isinstance(pred["candidate"]["n"], int)
        assert 0 <= pred["candidate"]["n"] <= 5
        assert pred["candidate"]["cat"] in choices


def test_two_sum_groups_and_parameter_before_ingredients():
    rng = np.random.default_rng(4)
    # Column 0 is a process parameter; 1-2 and 3-4 are two mixture groups.
    raw = rng.dirichlet(np.ones(4), size=28)
    x = np.column_stack([rng.uniform(10.0, 20.0, size=28), raw[:, :2] * 0.6, raw[:, 2:] * 0.4])
    y = np.sum((x[:, 1:] - 0.2) ** 2, axis=1, keepdims=True)
    names = ["temp", "a", "b", "c", "d"]
    params = [{"name": "temp", "type": "float", "min": 0.0, "max": 30.0}]
    params += [{"name": n, "type": "float", "min": 0.0, "max": 1.0} for n in names[1:]]
    req = OptimizeRequest(
        dataset={"X": x.tolist(), "Y": y.tolist()},
        search_space={"parameters": params},
        objectives={"cost": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qei",
            batch_size=2,
            seed=5,
            use_pca=True,
            pca_dimension=2,
            sum_constraints=[
                {"indices": [1, 2], "target_sum": 0.6},
                {"indices": [3, 4], "target_sum": 0.4},
            ],
            advanced={"num_restarts": 3, "raw_samples": 16, "acq_maxiter": 30},
        ),
    )
    res = run_optimization(req, device="cpu")
    assert res["encoding_info"]["k_effective"] >= 1
    for pred in res["predictions"]:
        cand = pred["candidate"]
        assert abs(cand["a"] + cand["b"] - 0.6) < 1e-5
        assert abs(cand["c"] + cand["d"] - 0.4) < 1e-5
        combined = pred["reconstructed"]["combined"]
        assert [combined[i] for i in range(5)] == [cand[n] for n in names]
        assert pred["candidate_feasible"] is True


def test_out_of_envelope_reconstruction_is_feasible_but_not_on_target():
    components = np.array([[1.0, 0.0]], dtype=float)
    res = slsqp_reconstruct(
        target_encoded=np.array([5.0]),
        encoder_components=components,
        encoder_mean=np.zeros(2),
        ingredient_bounds=[[0.0, 1.0], [0.0, 1.0]],
        parameter_bounds=[],
        n_ingredients=2,
        sum_target=1.0,
        target_precision=1e-3,
    )
    assert res["feasible"] is True
    assert res["on_target"] is False
    assert res["success"] is False
    assert abs(sum(res["ingredients"]) - 1.0) < 1e-6


def test_executor_timeout_stops_the_worker():
    store = JobStore()
    executor = Executor(store, max_workers=1, timeout_sec=1)
    job = store.create_job("tester")

    async def _go():
        await executor.submit(job.id, "sleep", (30,))

    asyncio.run(_go())
    finished = store.get_job(job.id)
    assert finished is not None
    assert finished.status == JobStatus.FAILED
    assert "timed out" in (finished.error or "").lower()
    executor.shutdown(wait=False)
