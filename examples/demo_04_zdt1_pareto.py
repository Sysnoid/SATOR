"""Demo 04 — multi-objective Pareto optimization (ZDT1).

ZDT1 problem, 4-dim:
    f1(x) = x1
    g(x)  = 1 + 9 * mean(x2..xd)
    f2(x) = g * (1 - sqrt(f1 / g))

Both f1 and f2 minimized; Pareto front is f2 = 1 - sqrt(f1), f1 in [0, 1].

Visual:
    Objective-space scatter — training points + predicted batch + analytic front.
"""

from __future__ import annotations

import numpy as np
from _common import mpl_setup, post_optimize_and_wait

D = 4


def zdt1(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    g = 1.0 + 9.0 * X[:, 1:].mean(axis=1)
    f2 = g * (1.0 - np.sqrt(x1 / g))
    return np.stack([x1, f2], axis=1)


def main() -> None:
    rng = np.random.default_rng(3)
    X = rng.uniform(size=(40, D))
    Y = zdt1(X)

    params = [{"name": f"x{i + 1}", "type": "float", "min": 0.0, "max": 1.0} for i in range(D)]
    payload = {
        "dataset": {"X": X.tolist(), "Y": Y.tolist()},
        "search_space": {"parameters": params},
        "objectives": {"f1": {"goal": "min"}, "f2": {"goal": "min"}},
        "optimization_config": {
            "acquisition": "qnehvi",
            "batch_size": 6,
            "max_evaluations": 32,
            "seed": 5,
            "return_maps": False,
        },
    }

    result = post_optimize_and_wait("demo_04_zdt1", payload)
    preds = result.get("predictions", [])
    if preds:
        cand = np.array(
            [[pred["candidate"][f"x{i + 1}"] for i in range(D)] for pred in preds]
        )
        Yp = zdt1(cand)
    else:
        Yp = np.zeros((0, 2))

    plt = mpl_setup()
    xs = np.linspace(0.0, 1.0, 200)
    front = np.stack([xs, 1.0 - np.sqrt(xs)], axis=1)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    ax.plot(front[:, 0], front[:, 1], "g-", lw=2, label="true Pareto front")
    ax.scatter(Y[:, 0], Y[:, 1], c="white", edgecolor="k", s=30, alpha=0.9, label="train")
    if Yp.size:
        ax.scatter(Yp[:, 0], Yp[:, 1], c="red", s=80, marker="X", label="predicted batch")
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_title("Demo 04 — ZDT1 Pareto front (qNEHVI)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
