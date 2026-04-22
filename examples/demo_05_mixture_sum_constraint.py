"""Demo 05 — 3-ingredient mixture under a sum-to-one constraint.

Scenario:   blend three solvents w1+w2+w3 = 1. Each weight in [0, 1].
Truth:      f(w) = 2*w1 + 0.5*w2 + 3*w3 + 4*(w2 - 0.4)**2 — a linear cost with
            a bump penalizing w2 straying from 0.4. Global min on the simplex
            near (w1=0.6, w2=0.4, w3=0).
Objective:  minimize f.
Visual:     Ternary-style 2D scatter of (w1, w2) with w3 = 1 - w1 - w2,
            coloured by f-value; training white, prediction red.
"""

from __future__ import annotations

import numpy as np
from _common import mpl_setup, post_optimize_and_wait


def truth(W: np.ndarray) -> np.ndarray:
    return (2.0 * W[:, 0] + 0.5 * W[:, 1] + 3.0 * W[:, 2] + 4.0 * (W[:, 1] - 0.4) ** 2).astype(float)


def simplex_samples(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.dirichlet(alpha=np.ones(3), size=n)


def main() -> None:
    rng = np.random.default_rng(23)
    W = simplex_samples(rng, 50)
    Y = truth(W)[:, None]

    payload = {
        "dataset": {"X": W.tolist(), "Y": Y.tolist()},
        "search_space": {
            "parameters": [
                {"name": "w1", "type": "float", "min": 0.0, "max": 1.0},
                {"name": "w2", "type": "float", "min": 0.0, "max": 1.0},
                {"name": "w3", "type": "float", "min": 0.0, "max": 1.0},
            ]
        },
        "objectives": {"cost": {"goal": "min"}},
        "optimization_config": {
            "acquisition": "qei",
            "batch_size": 5,
            "max_evaluations": 24,
            "seed": 29,
            "sum_constraints": [{"indices": [0, 1, 2], "target_sum": 1.0}],
            "return_maps": False,
        },
    }

    result = post_optimize_and_wait("demo_05_mixture_sum", payload)
    preds = result.get("predictions", [])
    Wp = np.array([[p["candidate"]["w1"], p["candidate"]["w2"], p["candidate"]["w3"]] for p in preds])

    plt = mpl_setup()
    n = 120
    a = np.linspace(0.0, 1.0, n)
    A, B = np.meshgrid(a, a, indexing="xy")
    C = 1.0 - A - B
    mask = C >= 0.0
    Wg = np.stack([A[mask], B[mask], C[mask]], axis=1)
    Fg = truth(Wg)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    sc = ax.scatter(Wg[:, 0], Wg[:, 1], c=Fg, cmap="viridis", s=6, alpha=0.85)
    fig.colorbar(sc, label="cost")
    ax.scatter(W[:, 0], W[:, 1], c="white", edgecolor="k", s=30, alpha=0.9, label="train")
    if Wp.size:
        ax.scatter(Wp[:, 0], Wp[:, 1], c="red", s=100, marker="X", label="pred")
    ax.plot([0, 1], [1, 0], "k--", lw=1, label="w1+w2=1 (w3=0)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_title("Demo 05 — Sum-to-one mixture (cost surface on simplex)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    if Wp.size:
        print("sum(W_pred) per row:", Wp.sum(axis=1).round(4).tolist())


if __name__ == "__main__":
    main()
