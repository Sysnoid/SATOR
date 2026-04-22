"""Demo 02 — ``target`` goal on the Ackley surface.

Surface:    Ackley 2D on [-5, 5]^2 (global min 0 at origin).
Objective:  drive f(x) toward the user-set target value T=2.5 (a non-trivial
            level set). Showcases the Sobol+scoring advanced-goal path.
Visual:     3D surface + the target level-set ring (green) + predictions (red).
"""

from __future__ import annotations

import math

import numpy as np
from _common import mpl_setup, post_optimize_and_wait

TARGET_VALUE = 2.5


def ackley(x: np.ndarray) -> np.ndarray:
    x1, x2 = x[..., 0], x[..., 1]
    a, b, c = 20.0, 0.2, 2.0 * math.pi
    r = np.sqrt(0.5 * (x1 ** 2 + x2 ** 2))
    t = 0.5 * (np.cos(c * x1) + np.cos(c * x2))
    return (-a * np.exp(-b * r) - np.exp(t) + a + math.e).astype(float)


def main() -> None:
    rng = np.random.default_rng(11)
    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    X = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * rng.uniform(size=(80, 2))
    Y = ackley(X)[:, None]

    payload = {
        "dataset": {"X": X.tolist(), "Y": Y.tolist()},
        "search_space": {
            "parameters": [
                {"name": "x1", "type": "float", "min": -5.0, "max": 5.0},
                {"name": "x2", "type": "float", "min": -5.0, "max": 5.0},
            ]
        },
        "objectives": {
            "f": {"goal": "target", "target_value": TARGET_VALUE},
        },
        "optimization_config": {
            "acquisition": "qei",
            "batch_size": 6,
            "max_evaluations": 24,
            "seed": 42,
            "target_tolerance": 0.05,
            "target_variance_penalty": 0.25,
            "return_maps": False,
        },
    }

    result = post_optimize_and_wait("demo_02_ackley_target", payload)
    preds = result.get("predictions", [])
    P = np.array([[p["candidate"]["x1"], p["candidate"]["x2"]] for p in preds])

    plt = mpl_setup()
    gx, gy = np.meshgrid(np.linspace(-5, 5, 160), np.linspace(-5, 5, 160), indexing="xy")
    gz = ackley(np.stack([gx.ravel(), gy.ravel()], axis=1)).reshape(gx.shape)

    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(gx, gy, gz, cmap="viridis", alpha=0.8, linewidth=0)
    fig.colorbar(surf, shrink=0.7, pad=0.08, label="f(x)")
    ax.contour(gx, gy, gz, levels=[TARGET_VALUE], colors="lime", offset=TARGET_VALUE, linewidths=2)
    ax.scatter(X[:, 0], X[:, 1], ackley(X), c="white", s=12, alpha=0.7, label="train")
    if P.size:
        zP = ackley(P)
        ax.scatter(P[:, 0], P[:, 1], zP, c="red", s=80, marker="X", label="pred")
    ax.set_title(f"Demo 02 — Ackley, target f ≈ {TARGET_VALUE}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")
    ax.view_init(elev=30, azim=-55)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
