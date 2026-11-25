"""Cardinality, ILR coordinates, and mixed categorical models."""

from __future__ import annotations

import numpy as np

from sator_os_engine.core.models.optimize import OptimizationConfig, OptimizeRequest
from sator_os_engine.core.optimizer.mixture import ilr_forward, ilr_inverse
from sator_os_engine.core.optimizer.mobo_engine import run_optimization


def test_ilr_roundtrip_on_the_simplex():
    rng = np.random.default_rng(0)
    raw = rng.dirichlet(np.ones(4), size=12)
    restored = ilr_inverse(ilr_forward(raw))
    assert np.allclose(restored, raw, atol=1e-8)
    assert np.allclose(restored.sum(axis=1), 1.0)


def test_optimizer_respects_cardinality():
    rng = np.random.default_rng(1)
    x = rng.dirichlet(np.ones(4), size=30)
    y = np.sum((x - 0.25) ** 2, axis=1, keepdims=True)
    params = [{"name": f"w{i}", "type": "float", "min": 0.0, "max": 1.0} for i in range(4)]
    req = OptimizeRequest(
        dataset={"X": x.tolist(), "Y": y.tolist()},
        search_space={"parameters": params},
        objectives={"cost": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qei",
            batch_size=3,
            seed=1,
            sum_constraints=[{"indices": [0, 1, 2, 3], "target_sum": 1.0}],
            ingredient_rules={"max_nonzero": 2, "min_dose": 0.05},
            advanced={"num_restarts": 3, "raw_samples": 16, "acq_maxiter": 20},
        ),
    )
    res = run_optimization(req, device="cpu")
    for pred in res["predictions"]:
        vals = [pred["candidate"][f"w{i}"] for i in range(4)]
        assert abs(sum(vals) - 1.0) < 1e-5
        assert sum(v > 1e-6 for v in vals) <= 2
        assert all(v == 0.0 or v >= 0.05 - 1e-6 for v in vals)


def test_ilr_optimize_returns_a_feasible_mixture():
    rng = np.random.default_rng(2)
    x = rng.dirichlet(np.ones(3), size=24)
    target = np.array([0.2, 0.3, 0.5])
    y = np.sum((x - target) ** 2, axis=1, keepdims=True)
    params = [{"name": name, "type": "float", "min": 0.0, "max": 1.0} for name in ("a", "b", "c")]
    req = OptimizeRequest(
        dataset={"X": x.tolist(), "Y": y.tolist()},
        search_space={"parameters": params},
        objectives={"cost": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qei",
            batch_size=2,
            seed=2,
            sum_constraints=[{"indices": [0, 1, 2], "target_sum": 1.0}],
            mixture_transform="ilr",
            advanced={"num_restarts": 3, "raw_samples": 16, "acq_maxiter": 20},
        ),
    )
    res = run_optimization(req, device="cpu")
    assert res["diagnostics"]["mixture_transform"] == "ilr"
    for pred in res["predictions"]:
        total = pred["candidate"]["a"] + pred["candidate"]["b"] + pred["candidate"]["c"]
        assert abs(total - 1.0) < 1e-5


def test_categorical_uses_a_mixed_model_and_a_valid_level():
    rng = np.random.default_rng(3)
    rows = []
    ys = []
    for _ in range(24):
        x = float(rng.uniform(0.0, 1.0))
        level = str(rng.choice(["A", "B", "C"]))
        penalty = {"A": 0.0, "B": 3.0, "C": 6.0}[level]
        rows.append([x, level])
        ys.append([x + penalty])
    req = OptimizeRequest(
        dataset={"X": rows, "Y": ys},
        search_space={
            "parameters": [
                {"name": "x", "type": "float", "min": 0.0, "max": 1.0},
                {"name": "cat", "type": "categorical", "choices": ["A", "B", "C"]},
            ]
        },
        objectives={"f": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qei", batch_size=2, seed=3, advanced={"num_restarts": 2, "raw_samples": 8, "acq_maxiter": 15}
        ),
    )
    res = run_optimization(req, device="cpu")
    assert res["diagnostics"]["mixed_model"] is True
    for pred in res["predictions"]:
        assert pred["candidate"]["cat"] == "A"
