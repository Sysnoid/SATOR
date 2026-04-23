"""Tests for the hard-constraint goal family (``enforce_*``).

These tests exercise the contract that makes enforce_above / enforce_below /
enforce_within_range different from the soft ``minimize_below`` /
``maximize_above`` / ``within_range`` goals:

1. The GP posterior is compared against the threshold during selection and
   used as a feasibility mask on the Sobol scoring grid.
2. Every returned prediction carries ``enforced_goals_satisfied`` and
   ``enforced_violations`` fields so a downstream caller can filter or
   trust the batch deterministically.
3. The top-level ``diagnostics.enforcement`` block summarises the result
   across the whole batch.

The helper ``evaluate_enforced_goals`` is unit-tested directly so that a
regression in the mask logic is caught without having to run a full GP fit.
"""

from __future__ import annotations

import numpy as np

from sator_os_engine.core.models.optimize import OptimizationConfig, OptimizeRequest
from sator_os_engine.core.optimizer.mobo_engine import run_optimization
from sator_os_engine.core.optimizer.utils import (
    evaluate_enforced_goals,
    extract_enforced_goal_specs,
)


class _ReqStub:
    """Minimal stand-in for ``OptimizeRequest`` for pure-helper tests."""

    def __init__(self, objectives, margin: float = 0.0):
        self.objectives = objectives

        class _Cfg:
            enforcement_uncertainty_margin = margin

        self.optimization_config = _Cfg()


def _params_2d():
    return [
        {"name": "x1", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "x2", "type": "float", "min": 0.0, "max": 1.0},
    ]


# ---------------------------------------------------------------------------
# Helper-level tests (no GP fit, fully deterministic)
# ---------------------------------------------------------------------------


def test_extract_specs_skips_soft_goals():
    req = _ReqStub(
        objectives={
            "a": {"goal": "max"},
            "b": {"goal": "maximize_above", "threshold": {"value": 1.0}},
            "c": {"goal": "enforce_above", "threshold_value": 2.0},
            "d": {"goal": "enforce_below", "threshold_value": 5.0},
            "e": {"goal": "enforce_within_range", "range": {"min": -1, "max": 1}},
        }
    )
    specs = extract_enforced_goal_specs(req)
    assert [s["name"] for s in specs] == ["c", "d", "e"]
    assert [s["kind"] for s in specs] == ["above", "below", "within_range"]
    assert [s["index"] for s in specs] == [2, 3, 4]
    assert specs[0]["lo"] == 2.0 and specs[0]["hi"] is None
    assert specs[1]["lo"] is None and specs[1]["hi"] == 5.0
    assert specs[2]["lo"] == -1.0 and specs[2]["hi"] == 1.0


def test_extract_specs_supports_threshold_dict_backcompat():
    req = _ReqStub(
        objectives={
            "only_val": {"goal": "enforce_above", "threshold_value": 1.0},
            "legacy_dict": {"goal": "enforce_below", "threshold": {"value": 7.0}},
        }
    )
    specs = extract_enforced_goal_specs(req)
    assert len(specs) == 2
    assert specs[0]["lo"] == 1.0
    assert specs[1]["hi"] == 7.0


def test_extract_specs_drops_incomplete_specs():
    # enforce_above without a threshold_value has nothing to enforce
    req = _ReqStub(
        objectives={
            "bad": {"goal": "enforce_above"},
            "good": {"goal": "enforce_below", "threshold_value": 0.0},
        }
    )
    specs = extract_enforced_goal_specs(req)
    assert len(specs) == 1
    assert specs[0]["name"] == "good"


def test_evaluate_enforced_goals_mask_logic_mean_only():
    specs = extract_enforced_goal_specs(
        _ReqStub(
            objectives={
                "f": {"goal": "enforce_above", "threshold_value": 4.2},
                "g": {"goal": "enforce_below", "threshold_value": 1.0},
                "h": {"goal": "enforce_within_range", "range": {"min": 80, "max": 150}},
            }
        )
    )
    # Row 0: all satisfied. Row 1: f below threshold. Row 2: g above ceiling.
    # Row 3: h above window.
    mu = np.array(
        [
            [4.5, 0.5, 120.0],
            [4.0, 0.5, 120.0],
            [4.5, 1.5, 120.0],
            [4.5, 0.5, 160.0],
        ]
    )
    var = np.zeros_like(mu)
    mask, violations = evaluate_enforced_goals(specs, mu, var, margin=0.0)
    assert list(mask) == [True, False, False, False]
    assert violations[0] == []
    assert violations[1] == ["f<4.2"]
    assert violations[2] == ["g>1"]
    assert violations[3] == ["h>150"]


def test_evaluate_enforced_goals_uncertainty_margin_tightens():
    specs = extract_enforced_goal_specs(
        _ReqStub(
            objectives={"f": {"goal": "enforce_above", "threshold_value": 4.2}},
            margin=2.0,
        )
    )
    # mu = 4.5, sigma = 0.5 -> LCB = 4.5 - 2*0.5 = 3.5 < 4.2 so margin=2 rejects.
    mu = np.array([[4.5]])
    var = np.array([[0.25]])  # sigma = 0.5
    mask0, _ = evaluate_enforced_goals(specs, mu, var, margin=0.0)
    mask2, viol2 = evaluate_enforced_goals(specs, mu, var, margin=2.0)
    assert bool(mask0[0]) is True  # mean passes
    assert bool(mask2[0]) is False  # LCB fails
    assert viol2[0] == ["f<4.2"]


# ---------------------------------------------------------------------------
# End-to-end tests (run a real GP fit and check response contract)
# ---------------------------------------------------------------------------


def test_orchestrator_enforce_above_tags_every_prediction():
    """Every prediction must carry an honest ``enforced_goals_satisfied``
    flag, regardless of whether the threshold was achievable."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0.0, 1.0, size=(60, 2))
    # y = 1 - ||x - 0.5||^2: peaks near 1.0 at (0.5, 0.5).
    Y = (1.0 - np.sum((X - 0.5) ** 2, axis=1))[:, None]

    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={"parameters": _params_2d()},
        objectives={"f": {"goal": "enforce_above", "threshold_value": 0.85}},
        optimization_config=OptimizationConfig(
            acquisition="qei",
            batch_size=3,
            max_evaluations=10,
            seed=7,
            return_maps=False,
            use_pca=False,
        ),
    )
    res = run_optimization(req, device="cpu")
    preds = res["predictions"]
    assert len(preds) == 3
    for p in preds:
        assert "enforced_goals_satisfied" in p
        assert "enforced_violations" in p
        if p["enforced_goals_satisfied"]:
            assert p["objectives"][0] >= 0.85 - 1e-6
            assert p["enforced_violations"] == []
        else:
            assert any("f<" in v for v in p["enforced_violations"])
    diag = res["diagnostics"]
    assert "enforcement" in diag
    enf = diag["enforcement"]
    assert enf["enabled"] is True
    assert enf["n_total"] == 3
    assert 0 <= enf["n_satisfied"] <= 3
    assert isinstance(enf["all_infeasible"], bool)
    assert enf["goals"][0]["kind"] == "above"
    assert enf["goals"][0]["lo"] == 0.85


def test_orchestrator_enforce_above_selects_feasible_when_possible():
    """With a threshold well within the achievable range, every returned
    candidate should satisfy it on the GP posterior mean."""
    rng = np.random.default_rng(43)
    X = rng.uniform(0.0, 1.0, size=(80, 2))
    Y = (1.0 - np.sum((X - 0.5) ** 2, axis=1))[:, None]  # peaks at ~1.0

    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={"parameters": _params_2d()},
        # Threshold at 0.6 is easy: most of the support near (0.5, 0.5) clears it.
        objectives={"f": {"goal": "enforce_above", "threshold_value": 0.6}},
        optimization_config=OptimizationConfig(
            acquisition="qei",
            batch_size=4,
            max_evaluations=12,
            seed=11,
            return_maps=False,
            use_pca=False,
        ),
    )
    res = run_optimization(req, device="cpu")
    preds = res["predictions"]
    assert len(preds) == 4
    for p in preds:
        assert p["enforced_goals_satisfied"] is True, p["enforced_violations"]
        assert p["objectives"][0] >= 0.6 - 1e-6


def test_orchestrator_enforce_within_range_multi_objective():
    """``enforce_within_range`` must hold alongside other soft objectives."""
    rng = np.random.default_rng(44)
    X = rng.uniform(0.0, 1.0, size=(80, 2))
    # f1: minimize (bowl at 0.3, 0.3). f2: keep inside [0.2, 0.8].
    f1 = np.sum((X - 0.3) ** 2, axis=1)
    f2 = X[:, 0] + X[:, 1]  # spans ~[0, 2]
    Y = np.stack([f1, f2], axis=1)

    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={"parameters": _params_2d()},
        objectives={
            "cost": {"goal": "min"},
            "band": {"goal": "enforce_within_range", "range": {"min": 0.2, "max": 0.8}},
        },
        optimization_config=OptimizationConfig(
            acquisition="qnehvi",
            batch_size=3,
            max_evaluations=12,
            seed=13,
            return_maps=False,
            use_pca=False,
        ),
    )
    res = run_optimization(req, device="cpu")
    preds = res["predictions"]
    assert len(preds) == 3
    # At least one candidate should land in the band; every feasible one must.
    feasible = [p for p in preds if p["enforced_goals_satisfied"]]
    assert feasible, "expected at least one enforce_within_range feasible prediction"
    for p in feasible:
        band_mu = p["objectives"][1]
        assert 0.2 - 1e-6 <= band_mu <= 0.8 + 1e-6


def test_orchestrator_no_enforce_goal_leaves_response_clean():
    """Plain min/max should NOT add the enforcement block or prediction flags."""
    rng = np.random.default_rng(45)
    X = rng.uniform(0.0, 1.0, size=(50, 2))
    Y = np.sum((X - 0.3) ** 2, axis=1)[:, None]
    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={"parameters": _params_2d()},
        objectives={"f": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qei", batch_size=2, max_evaluations=8, seed=17, return_maps=False, use_pca=False
        ),
    )
    res = run_optimization(req, device="cpu")
    assert "enforcement" not in res["diagnostics"]
    for p in res["predictions"]:
        assert "enforced_goals_satisfied" not in p
        assert "enforced_violations" not in p
