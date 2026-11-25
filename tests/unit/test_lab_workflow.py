"""Route advisor, gram sheet, and weighable ingredient steps."""

from __future__ import annotations

import numpy as np

from sator_os_engine.core.models.optimize import OptimizationConfig, OptimizeRequest
from sator_os_engine.core.optimizer.advisor import advise_route
from sator_os_engine.core.optimizer.mobo_engine import run_optimization
from sator_os_engine.core.optimizer.space import balance_resolved_sums, snap_numeric


def test_advisor_selects_a_sparse_model_for_a_wide_short_history():
    choice = advise_route(40, 24, model="gp", use_pca=False, mode="lab", categorical=False, ilr=False)
    assert choice["model"] == "saas_map"
    assert choice["route"] == "direct"
    assert "sparse" in choice["reason"]


def test_advisor_selects_a_trust_region_for_a_long_history():
    choice = advise_route(400, 8, model="kpls", use_pca=False, mode="lab", categorical=False, ilr=False)
    assert choice["model"] == "kpls"
    assert choice["route"] == "trust_region"


def test_advisor_is_echoed_on_a_small_problem():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, size=(12, 2))
    res = run_optimization(
        OptimizeRequest(
            dataset={"X": x.tolist(), "Y": x[:, :1].tolist()},
            search_space={
                "parameters": [
                    {"name": "a", "type": "float", "min": 0.0, "max": 1.0},
                    {"name": "b", "type": "float", "min": 0.0, "max": 1.0},
                ]
            },
            objectives={"f": {"goal": "min"}},
            optimization_config=OptimizationConfig(
                route="auto",
                batch_size=1,
                seed=0,
                advanced={"num_restarts": 2, "raw_samples": 8, "acq_maxiter": 10},
            ),
        ),
        device="cpu",
    )
    assert res["diagnostics"]["advisor"]["model"] == "gp"
    assert res["diagnostics"]["advisor"]["route"] == "direct"
    assert res["diagnostics"]["model"] == "gp"


def test_resolution_restores_the_sum():
    params = [
        {"name": "a", "type": "float", "min": 0.0, "max": 1.0, "resolution": 0.1},
        {"name": "b", "type": "float", "min": 0.0, "max": 1.0, "resolution": 0.1},
        {"name": "c", "type": "float", "min": 0.0, "max": 1.0, "resolution": 0.1},
    ]
    snapped = snap_numeric(np.array([0.33, 0.33, 0.34]), params)
    balanced = balance_resolved_sums(snapped, params, [{"indices": [0, 1, 2], "target": 1.0}])
    assert abs(float(balanced.sum()) - 1.0) < 1e-9
    carrier = int(np.argmax(balanced))
    for index, value in enumerate(balanced):
        if index != carrier:
            assert abs(value / 0.1 - round(value / 0.1)) < 1e-8


def test_batch_sheet_weighs_each_well_to_the_target_mass():
    rng = np.random.default_rng(1)
    x = rng.dirichlet(np.ones(3), size=16)
    y = x[:, :1]
    res = run_optimization(
        OptimizeRequest(
            dataset={"X": x.tolist(), "Y": y.tolist()},
            search_space={
                "parameters": [
                    {"name": name, "type": "float", "min": 0.0, "max": 1.0, "resolution": 0.05}
                    for name in ("a", "b", "c")
                ]
            },
            objectives={"f": {"goal": "min"}},
            optimization_config=OptimizationConfig(
                batch_size=2,
                seed=1,
                batch_mass=100.0,
                sum_constraints=[{"indices": [0, 1, 2], "target_sum": 1.0}],
                advanced={"num_restarts": 2, "raw_samples": 12, "acq_maxiter": 15},
            ),
        ),
        device="cpu",
    )
    sheet = res["batch_sheet"]
    assert sheet["mass"] == 100.0
    assert len(sheet["rows"]) == 2
    for row in sheet["rows"]:
        assert abs(row["total"] - 100.0) < 1e-6
    for pred in res["predictions"]:
        total = pred["candidate"]["a"] + pred["candidate"]["b"] + pred["candidate"]["c"]
        assert abs(total - 1.0) < 1e-6
