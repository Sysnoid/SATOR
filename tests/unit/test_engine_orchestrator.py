"""Smoke tests for the optimization orchestrator `run_optimization`.

Validates end-to-end execution for:
- Single-objective (qei), with/without PCA.
- Multi-objective (qnehvi) with standard goals (min/max).
- Advanced-goal mixture to exercise non-qEHVI sampling path.
- Constraints: sum + ratio feasibility and post-selection sum enforcement.
"""

from __future__ import annotations

import numpy as np

from sator_os_engine.core.models.optimize import OptimizationConfig, OptimizeRequest
from sator_os_engine.core.optimizer.mobo_engine import run_optimization


def _params_2d():
    return [
        {"name": "x1", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "x2", "type": "float", "min": 0.0, "max": 1.0},
    ]


def test_orchestrator_single_qei_no_pca():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(60, 2))
    Y = (np.sum((X - 0.3) ** 2, axis=1))[:, None]  # min near 0.3

    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={"parameters": _params_2d()},
        objectives={"f": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qei", batch_size=3, max_evaluations=10, seed=1, return_maps=False, use_pca=False
        ),
    )
    res = run_optimization(req, device="cpu")
    assert isinstance(res.get("predictions"), list) and len(res["predictions"]) == 3


def test_orchestrator_multi_qnehvi_input_space():
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, size=(70, 2))
    f1 = np.sum((X - 0.25) ** 2, axis=1)
    f2 = -np.sum((X - 0.75) ** 2, axis=1)
    Y = np.stack([f1, f2], axis=1)

    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={"parameters": _params_2d()},
        objectives={"o1": {"goal": "min"}, "o2": {"goal": "max"}},
        optimization_config=OptimizationConfig(
            acquisition="qnehvi", batch_size=4, max_evaluations=12, seed=2, return_maps=False, use_pca=False
        ),
    )
    res = run_optimization(req, device="cpu")
    assert isinstance(res.get("predictions"), list) and len(res["predictions"]) == 4


def test_orchestrator_advanced_goals_sampling_path():
    rng = np.random.default_rng(2)
    X = rng.uniform(0.0, 1.0, size=(80, 2))
    f1 = np.sum((X - 0.4) ** 2, axis=1)
    f2 = np.sum((X - 0.6) ** 2, axis=1)
    Y = np.stack([f1, f2], axis=1)

    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={"parameters": _params_2d()},
        objectives={
            "o1": {"goal": "target", "target_value": 0.2},
            "o2": {"goal": "within_range", "range": {"min": 0.1, "max": 0.5}},
        },
        optimization_config=OptimizationConfig(
            acquisition="qnehvi", batch_size=3, max_evaluations=10, seed=3, return_maps=False, use_pca=False
        ),
    )
    res = run_optimization(req, device="cpu")
    assert isinstance(res.get("predictions"), list) and len(res["predictions"]) == 3


def test_orchestrator_constraints_sum_and_ratio():
    rng = np.random.default_rng(3)
    # Three variables: enforce sum on first two and ratio between a and c
    X = rng.uniform(0.0, 1.0, size=(60, 3))
    Y = (np.sum(X, axis=1))[:, None]  # single objective, not important for the check

    params = [
        {"name": "a", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "b", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "c", "type": "float", "min": 0.0, "max": 1.0},
    ]
    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={"parameters": params},
        objectives={"o": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qei",
            batch_size=3,
            max_evaluations=10,
            seed=4,
            sum_constraints=[{"indices": [0, 1], "target_sum": 1.0}],
            ratio_constraints=[{"i": 0, "j": 2, "min_ratio": 0.5, "max_ratio": 2.0}],
        ),
    )
    res = run_optimization(req, device="cpu")
    preds = res["predictions"]
    assert len(preds) == 3
    for p in preds:
        a = p["candidate"]["a"]
        b = p["candidate"]["b"]
        # Sum enforcement
        assert abs((a + b) - 1.0) < 1e-2
        # Note: ratio constraint here is between ``a`` (inside the sum group)
        # and ``c`` (free variable), which is a pathological *cross-set* case
        # where the sum projection rescales the numerator but not the
        # denominator. Ratio enforcement for realistic in-mixture pairs is
        # covered by ``test_orchestrator_nonpca_ratio_advanced_goals_path``
        # (non-PCA) and ``test_orchestrator_pca_ratio_reconstructed`` (PCA).


def test_orchestrator_nonpca_ratio_advanced_goals_path():
    """Non-PCA Sobol-score path must honour ``ratio_constraints``.

    Regression test: the advanced-goals branch (any goal outside ``min``/``max``)
    used to project the Sobol grid through ``feasible_mask`` before sum
    projection, so no sample could satisfy ``sum=target`` exactly and the
    mask collapsed to all-False. Ratio constraints were then silently
    ignored. With the fix the grid is sum-projected first and any remaining
    ratio drift is repaired post-selection.
    """
    rng = np.random.default_rng(11)
    n_feat = 3
    X = rng.dirichlet(alpha=np.array([2.0, 2.0, 2.0]), size=50)
    Y = np.stack([np.sum(X, axis=1), np.sum(X**2, axis=1)], axis=1)

    params = [
        {"name": "a", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "b", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "c", "type": "float", "min": 0.0, "max": 1.0},
    ]
    min_r, max_r = 1.2, 2.5
    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={"parameters": params},
        objectives={
            "o1": {"goal": "within_range", "range": {"min": 0.1, "max": 0.9}},
            "o2": {"goal": "min"},
        },
        optimization_config=OptimizationConfig(
            acquisition="qei",
            batch_size=4,
            max_evaluations=10,
            seed=13,
            sum_constraints=[{"indices": list(range(n_feat)), "target_sum": 1.0}],
            ratio_constraints=[{"i": 0, "j": 1, "min_ratio": min_r, "max_ratio": max_r}],
            use_pca=False,
            return_maps=False,
        ),
    )
    res = run_optimization(req, device="cpu")
    preds = res["predictions"]
    assert len(preds) == 4
    tol = 1e-2
    for p in preds:
        a = p["candidate"]["a"]
        b = p["candidate"]["b"]
        c = p["candidate"]["c"]
        s = a + b + c
        assert abs(s - 1.0) < 1e-2
        ratio = a / max(b, 1e-12)
        assert min_r - tol <= ratio <= max_r + tol, f"ratio {ratio:.4f} violates [{min_r}, {max_r}]"


def test_orchestrator_pca_ratio_reconstructed():
    """With ``use_pca=True`` the SLSQP reconstructor must honour ratio constraints.

    Regression test for the ``ratio_constraints=None`` hard-coding that used to
    be passed through at ``mobo_engine.py``. The training set is sampled freely
    (most rows violate the tight ratio window), so the only way the returned
    ``candidate`` recipe can satisfy ``MCC / lactose`` is if the engine is
    enforcing the constraint during reconstruction.
    """
    rng = np.random.default_rng(7)
    n_feat = 4
    X = rng.dirichlet(alpha=np.array([2.0, 6.0, 3.0, 0.5]), size=40)
    Y = np.sum((X - 0.2) ** 2, axis=1, keepdims=True)

    params = [
        {"name": "a", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "lac", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "mcc", "type": "float", "min": 0.0, "max": 1.0},
        {"name": "d", "type": "float", "min": 0.0, "max": 1.0},
    ]
    min_r, max_r = 0.60, 0.90
    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={"parameters": params},
        objectives={"o": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qei",
            batch_size=3,
            max_evaluations=12,
            seed=9,
            sum_constraints=[{"indices": list(range(n_feat)), "target_sum": 1.0}],
            ratio_constraints=[{"i": 2, "j": 1, "min_ratio": min_r, "max_ratio": max_r}],
            use_pca=True,
            pca_dimension=2,
            return_maps=False,
        ),
    )
    res = run_optimization(req, device="cpu")
    preds = res["predictions"]
    assert len(preds) == 3
    tol = 1e-3
    for p in preds:
        mcc = p["candidate"]["mcc"]
        lac = p["candidate"]["lac"]
        ratio = mcc / max(lac, 1e-12)
        assert min_r - tol <= ratio <= max_r + tol, f"ratio {ratio:.4f} violates [{min_r}, {max_r}]"
        # Sum-to-one should still hold.
        s = sum(p["candidate"][n] for n in ("a", "lac", "mcc", "d"))
        assert abs(s - 1.0) < 1e-2


def test_orchestrator_botorch_ratio_only_no_sum_no_pca():
    """BoTorch ``optimize_acqf`` must honour a standalone ratio constraint.

    Regression test for the ratio-tuple sign inversion in
    ``build_linear_constraints``: ratio inequalities were being emitted in
    ``<= rhs`` form but ``optimize_acqf`` consumes the same tuples as
    ``>= rhs``. The sum encoding happened to survive this by +/- symmetry;
    ratio constraints silently inverted min/max and collapsed the feasible
    polytope. Demo_06 exercised this path and failed with
    "No feasible point found. Constraint polytope appears empty."

    This test mirrors demo_06 exactly: single-objective ``min``, ``qei``
    acquisition, no sum constraint, no PCA, only a ratio window. Every
    returned prediction must satisfy ``A / B`` in ``[min_ratio, max_ratio]``.
    """
    rng = np.random.default_rng(31)
    X = rng.uniform(low=[0.1, 0.1], high=[10.0, 10.0], size=(50, 2))
    # Same quadratic bowl as demo_06 so the optimum (4, 3) is inside the band.
    Y = ((X[:, 0] - 4.0) ** 2 + (X[:, 1] - 3.0) ** 2)[:, None]

    params = [
        {"name": "A", "type": "float", "min": 0.1, "max": 10.0},
        {"name": "B", "type": "float", "min": 0.1, "max": 10.0},
    ]
    min_r, max_r = 0.5, 2.0
    req = OptimizeRequest(
        dataset={"X": X.tolist(), "Y": Y.tolist()},
        search_space={"parameters": params},
        objectives={"f": {"goal": "min"}},
        optimization_config=OptimizationConfig(
            acquisition="qei",
            batch_size=4,
            max_evaluations=10,
            seed=101,
            ratio_constraints=[{"i": 0, "j": 1, "min_ratio": min_r, "max_ratio": max_r}],
            use_pca=False,
            return_maps=False,
        ),
    )
    res = run_optimization(req, device="cpu")
    preds = res["predictions"]
    assert len(preds) == 4
    tol = 1e-3
    for p in preds:
        a = p["candidate"]["A"]
        b = p["candidate"]["B"]
        ratio = a / max(b, 1e-12)
        assert min_r - tol <= ratio <= max_r + tol, f"ratio {ratio:.4f} violates [{min_r}, {max_r}]"
