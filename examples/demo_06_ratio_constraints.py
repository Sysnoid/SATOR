"""Demo 06 — 2D optimization under a ratio constraint.

Scenario:   two reagents (A, B), each in [0.1, 10]. Keep A/B ∈ [0.5, 2.0]
            (neither reagent dominates).
Truth:      f(A, B) = (A - 4)^2 + (B - 3)^2  — unique minimum at (4, 3),
            whose ratio 4/3 ≈ 1.33 lies inside the allowed band.
Objective:  minimize f.
Visual:     2D contour of f + the two ratio lines (A = 0.5*B and A = 2*B)
            bracketing the feasible wedge, plus training/predicted points.
"""

from __future__ import annotations

import numpy as np
from _common import mpl_setup, post_optimize_and_wait


def truth(x: np.ndarray) -> np.ndarray:
    return ((x[:, 0] - 4.0) ** 2 + (x[:, 1] - 3.0) ** 2).astype(float)


def main() -> None:
    rng = np.random.default_rng(31)
    X = rng.uniform(low=[0.1, 0.1], high=[10.0, 10.0], size=(60, 2))
    Y = truth(X)[:, None]

    payload = {
        "dataset": {"X": X.tolist(), "Y": Y.tolist()},
        "search_space": {
            "parameters": [
                {"name": "A", "type": "float", "min": 0.1, "max": 10.0},
                {"name": "B", "type": "float", "min": 0.1, "max": 10.0},
            ]
        },
        "objectives": {"f": {"goal": "min"}},
        "optimization_config": {
            "acquisition": "qei",
            "batch_size": 4,
            "max_evaluations": 20,
            "seed": 101,
            "ratio_constraints": [{"i": 0, "j": 1, "min_ratio": 0.5, "max_ratio": 2.0}],
            "return_maps": False,
        },
    }

    result = post_optimize_and_wait("demo_06_ratio", payload)
    preds = result.get("predictions", [])
    P = np.array([[p["candidate"]["A"], p["candidate"]["B"]] for p in preds])

    plt = mpl_setup()
    gx, gy = np.meshgrid(np.linspace(0.1, 10.0, 200), np.linspace(0.1, 10.0, 200), indexing="xy")
    gz = (gx - 4.0) ** 2 + (gy - 3.0) ** 2

    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    cf = ax.contourf(gx, gy, gz, levels=20, cmap="viridis")
    fig.colorbar(cf, label="f(A,B)")
    line_b = np.linspace(0.1, 10.0, 50)
    ax.plot(0.5 * line_b, line_b, "w--", lw=1.5, label="A = 0.5·B")
    ax.plot(2.0 * line_b, line_b, "w-", lw=1.5, label="A = 2·B")
    ax.fill_betweenx(line_b, 0.5 * line_b, 2.0 * line_b, color="white", alpha=0.08, label="feasible")
    ax.scatter(X[:, 0], X[:, 1], c="white", edgecolor="k", s=25, alpha=0.8, label="train")
    if P.size:
        ax.scatter(P[:, 0], P[:, 1], c="red", s=90, marker="X", label="pred")
    ax.scatter([4.0], [3.0], c="lime", s=140, marker="P", label="true min")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("A")
    ax.set_ylabel("B")
    ax.set_title("Demo 06 — Ratio constraint 0.5 ≤ A/B ≤ 2.0")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    if P.size:
        r = P[:, 0] / P[:, 1]
        print("A/B per prediction:", r.round(3).tolist())


if __name__ == "__main__":
    main()
