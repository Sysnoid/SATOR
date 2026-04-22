"""Demo 03 — ``within_range`` goal on the Himmelblau surface.

Surface:    f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2 on [-5, 5]^2
            (four symmetric minima = 0).
Objective:  keep f(x,y) inside [10, 25] (favor "pleasantly mediocre" regions,
            not too close to a minimum, not too high on the ridge).
Visual:     3D surface + band-feasible ring + predictions.
"""

from __future__ import annotations

import numpy as np
from _common import mpl_setup, post_optimize_and_wait

LO, HI = 10.0, 25.0


def himmelblau(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[..., 0], x[..., 1]
    return ((x1 ** 2 + x2 - 11.0) ** 2 + (x1 + x2 ** 2 - 7.0) ** 2).astype(float)


def main() -> None:
    rng = np.random.default_rng(19)
    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    X = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * rng.uniform(size=(70, 2))
    Y = himmelblau(X)[:, None]

    payload = {
        "dataset": {"X": X.tolist(), "Y": Y.tolist()},
        "search_space": {
            "parameters": [
                {"name": "x1", "type": "float", "min": -5.0, "max": 5.0},
                {"name": "x2", "type": "float", "min": -5.0, "max": 5.0},
            ]
        },
        "objectives": {
            "f": {
                "goal": "within_range",
                "range": {"min": LO, "max": HI, "ideal": 0.5 * (LO + HI), "weight": 1.0, "ideal_weight": 0.5},
            }
        },
        "optimization_config": {
            "acquisition": "qei",
            "batch_size": 6,
            "max_evaluations": 24,
            "seed": 77,
            "return_maps": False,
        },
    }

    result = post_optimize_and_wait("demo_03_himmelblau_range", payload)
    preds = result.get("predictions", [])
    P = np.array([[p["candidate"]["x1"], p["candidate"]["x2"]] for p in preds])

    plt = mpl_setup()
    gx, gy = np.meshgrid(np.linspace(-5, 5, 180), np.linspace(-5, 5, 180), indexing="xy")
    gz = himmelblau(np.stack([gx.ravel(), gy.ravel()], axis=1)).reshape(gx.shape)
    gz_plot = np.log1p(gz)

    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(gx, gy, gz_plot, cmap="magma", alpha=0.8, linewidth=0)
    fig.colorbar(surf, shrink=0.7, pad=0.08, label="log(1 + f)")
    ax.contour(gx, gy, gz, levels=[LO, HI], colors="lime", offset=np.log1p(LO), linewidths=1.5)
    ax.scatter(X[:, 0], X[:, 1], np.log1p(himmelblau(X)), c="white", s=12, alpha=0.7, label="train")
    if P.size:
        ax.scatter(P[:, 0], P[:, 1], np.log1p(himmelblau(P)), c="red", s=80, marker="X", label="pred")
    ax.set_title(f"Demo 03 — Himmelblau, want f ∈ [{LO:.0f}, {HI:.0f}]")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("log(1+f)")
    ax.view_init(elev=30, azim=-60)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
