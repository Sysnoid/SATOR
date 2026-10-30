"""Per-objective ``weight`` / ``target_tolerance`` on ``goal=target``."""

from __future__ import annotations

import numpy as np

from sator_os_engine.core.models.optimize import OptimizationConfig, OptimizeRequest
from sator_os_engine.core.optimizer.mobo_engine import run_optimization


def test_higher_weight_on_one_target_pulls_that_channel():
    """Two targets: emphasize o2 so candidates track T2 more tightly than T1."""
    rng = np.random.default_rng(4)
    # x1 drives o1, x2 drives o2
    X = rng.uniform(0.0, 1.0, size=(60, 2))
    Y = np.stack([X[:, 0], X[:, 1]], axis=1)
    t1, t2 = 0.3, 0.8

    def _run(w2: float):
        req = OptimizeRequest(
            dataset={"X": X.tolist(), "Y": Y.tolist()},
            search_space={
                "parameters": [
                    {"name": "x1", "type": "float", "min": 0.0, "max": 1.0},
                    {"name": "x2", "type": "float", "min": 0.0, "max": 1.0},
                ]
            },
            objectives={
                "o1": {"goal": "target", "target_value": t1, "target_tolerance": 0.05, "weight": 1.0},
                "o2": {"goal": "target", "target_value": t2, "target_tolerance": 0.05, "weight": w2},
            },
            optimization_config=OptimizationConfig(
                acquisition="qei",
                batch_size=1,
                max_evaluations=48,
                seed=7,
                return_maps=False,
            ),
        )
        preds = run_optimization(req, device="cpu").get("predictions") or []
        assert preds
        return np.asarray(preds[0]["objectives"], float)

    bal = _run(1.0)
    heavy = _run(8.0)
    # With heavy weight on o2, |o2 − t2| should improve vs balanced (or stay ≤).
    assert abs(heavy[1] - t2) <= abs(bal[1] - t2) + 0.05
    assert abs(heavy[1] - t2) < 0.15
