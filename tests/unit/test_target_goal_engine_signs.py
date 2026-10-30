"""Regression: ``goal=target`` through ``run_optimization`` (signed Y path).

The engine fits GPs on ``Y * sign`` (``sign=-1`` for non-max goals). Soft
scoring must compare the *unsigned* posterior to ``target_value``. A prior
bug compared the signed mean to the unsigned target, so target search
collapsed toward minimizing |Y| instead of approaching T.
"""

from __future__ import annotations

import numpy as np

from sator_os_engine.core.models.optimize import OptimizationConfig, OptimizeRequest
from sator_os_engine.core.optimizer.mobo_engine import run_optimization


def test_run_optimization_target_goal_approaches_target_not_zero():
    rng = np.random.default_rng(0)
    # Bowl with minimum 0 at origin; values in ~[0, 10].
    X = rng.uniform(-5.0, 5.0, size=(50, 2))
    Y = (np.sum(X**2, axis=1) / 5.0)[:, None]
    target = 2.5

    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={
            "parameters": [
                {"name": "x1", "type": "float", "min": -5.0, "max": 5.0},
                {"name": "x2", "type": "float", "min": -5.0, "max": 5.0},
            ]
        },
        objectives={"f": {"goal": "target", "target_value": target}},
        optimization_config=OptimizationConfig(
            acquisition="qei",
            batch_size=3,
            max_evaluations=40,
            seed=1,
            target_tolerance=0.1,
            target_variance_penalty=0.05,
            return_maps=False,
        ),
    )
    result = run_optimization(req, device="cpu")
    preds = result.get("predictions") or []
    assert preds, "expected target-goal predictions"
    errs = [abs(float(p["objectives"][0]) - target) for p in preds]
    # Before the fix, candidates sat near f≈0 (|err|≈2.5). After the fix they
    # should approach the level set f≈2.5.
    assert min(errs) < 0.75, f"target-goal errs={errs} (signed/unsigned mismatch?)"


def test_run_optimization_multi_target_lab_like():
    """Three target channels (Lab-like): selection should beat chasing zero."""
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, size=(40, 2))
    # Synthetic outputs around (3, -1, 2)
    L = 3.0 + 0.5 * X[:, 0] + 0.2 * X[:, 1]
    a = -1.0 - 0.3 * X[:, 0] + 0.4 * X[:, 1]
    b = 2.0 + 0.4 * X[:, 0] - 0.2 * X[:, 1]
    Y = np.stack([L, a, b], axis=1)
    targets = np.array([3.2, -1.1, 2.1])

    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={
            "parameters": [
                {"name": "x1", "type": "float", "min": 0.0, "max": 1.0},
                {"name": "x2", "type": "float", "min": 0.0, "max": 1.0},
            ]
        },
        objectives={
            "L": {"goal": "target", "target_value": float(targets[0])},
            "a": {"goal": "target", "target_value": float(targets[1])},
            "b": {"goal": "target", "target_value": float(targets[2])},
        },
        optimization_config=OptimizationConfig(
            acquisition="qei",
            batch_size=3,
            max_evaluations=48,
            seed=2,
            return_maps=False,
        ),
    )
    result = run_optimization(req, device="cpu")
    preds = result.get("predictions") or []
    assert preds
    dE = [
        float(np.linalg.norm(np.asarray(p["objectives"], float) - targets))
        for p in preds
    ]
    # Training ΔE to target for comparison
    train_dE = np.linalg.norm(Y - targets, axis=1)
    assert min(dE) <= float(np.median(train_dE)) + 0.15, f"pred dE={dE}, train median={np.median(train_dE)}"
