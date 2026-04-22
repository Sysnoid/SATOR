"""Unit tests for SLSQP reconstructor (named solution mapping)."""

from __future__ import annotations

import numpy as np

from sator_os_engine.reconstruction.slsqp_reconstructor import reconstruct as slsqp_reconstruct


def test_reconstruct_solution_by_name_two_ingredients():
    # One PC over 2 ingredients: z = 0.5 * x0 + 0.5 * x1; with sum=1, target z=0.5 at x0=x1=0.5
    components = np.array([[0.5, 0.5]], dtype=float)
    mean = np.array([0.0, 0.0], dtype=float)
    target = np.array([0.5], dtype=float)
    res = slsqp_reconstruct(
        target_encoded=target,
        encoder_components=components,
        encoder_mean=mean,
        ingredient_bounds=[[0.0, 1.0], [0.0, 1.0]],
        parameter_bounds=[],
        n_ingredients=2,
        target_precision=1e-6,
        sum_target=1.0,
        ingredient_names=["ing_a", "ing_b"],
        parameter_names=[],
    )
    assert res.get("success", False) is True
    assert "solution_by_name" in res
    by = res["solution_by_name"]
    assert set(by) == {"ing_a", "ing_b"}
    assert abs(by["ing_a"] + by["ing_b"] - 1.0) < 1e-4
    assert abs(by["ing_a"] - 0.5) < 0.02 and abs(by["ing_b"] - 0.5) < 0.02


def test_reconstruct_ingredient_plus_parameter_by_name():
    # 3D: first two ingredients sum to 1, third is a free parameter. z1 = 0.5(x0+x1), z2 = x2
    components = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    mean = np.zeros(3, dtype=float)
    target = np.array([0.5, 2.0], dtype=float)
    res = slsqp_reconstruct(
        target_encoded=target,
        encoder_components=components,
        encoder_mean=mean,
        ingredient_bounds=[[0.0, 1.0], [0.0, 1.0]],
        parameter_bounds=[[0.0, 5.0]],
        n_ingredients=2,
        sum_target=1.0,
        ingredient_names=["A", "B"],
        parameter_names=["C"],
    )
    assert "solution_by_name" in res
    b = res["solution_by_name"]
    assert abs(b["A"] + b["B"] - 1.0) < 1e-4
    assert abs(b["C"] - 2.0) < 0.01
