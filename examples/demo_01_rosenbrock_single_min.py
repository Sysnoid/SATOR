"""Demo 01 — single-objective minimization of the Rosenbrock "banana".

Surface:    f(x1, x2) = (1 - x1)^2 + 100 * (x2 - x1^2)^2 over [-2, 2] x [-1, 3]
Objective:  minimize f
Acquisition: qei (qLogExpectedImprovement single-objective path)
Visual:     3D surface with training samples (white) and prediction (red star).
"""

from __future__ import annotations

import numpy as np
from _common import mpl_setup, post_optimize_and_wait


def rosenbrock(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[..., 0], x[..., 1]
    return ((1.0 - x1) ** 2 + 100.0 * (x2 - x1 ** 2) ** 2).astype(float)


def main() -> None:
    rng = np.random.default_rng(7)
    bounds = np.array([[-2.0, 2.0], [-1.0, 3.0]])
    X = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * rng.uniform(size=(60, 2))
    Y = rosenbrock(X)[:, None]

    payload = {
        "dataset": {"X": X.tolist(), "Y": Y.tolist()},
        "search_space": {
            "parameters": [
                {"name": "x1", "type": "float", "min": -2.0, "max": 2.0},
                {"name": "x2", "type": "float", "min": -1.0, "max": 3.0},
            ]
        },
        "objectives": {"f": {"goal": "min"}},
        "optimization_config": {
            "acquisition": "qei",
            "batch_size": 4,
            "max_evaluations": 20,
            "seed": 13,
            "return_maps": False,
        },
    }

    result = post_optimize_and_wait("demo_01_rosenbrock", payload)
    preds = result.get("predictions", [])
    P = np.array([[p["candidate"]["x1"], p["candidate"]["x2"]] for p in preds])

    plt = mpl_setup()
    gx, gy = np.meshgrid(
        np.linspace(bounds[0, 0], bounds[0, 1], 140),
        np.linspace(bounds[1, 0], bounds[1, 1], 140),
        indexing="xy",
    )
    gz = rosenbrock(np.stack([gx.ravel(), gy.ravel()], axis=1)).reshape(gx.shape)

    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(gx, gy, np.log1p(gz), cmap="viridis", alpha=0.85, linewidth=0)
    fig.colorbar(surf, shrink=0.7, pad=0.08, label="log(1 + f)")
    ax.scatter(X[:, 0], X[:, 1], np.log1p(rosenbrock(X)), c="white", s=14, alpha=0.7, label="train")
    if P.size:
        ax.scatter(P[:, 0], P[:, 1], np.log1p(rosenbrock(P)), c="red", s=90, marker="*", label="pred")
    ax.scatter([1.0], [1.0], [np.log1p(0.0)], c="lime", s=120, marker="P", label="true min (1,1)")
    ax.set_title("Demo 01 — Rosenbrock (log scale) — minimize")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("log(1+f)")
    ax.view_init(elev=32, azim=-60)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
