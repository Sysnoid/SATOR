"""Demo 07 — realistic multi-objective paint formulation.

Formulation (6 mass fractions, must sum to 1):
    pigment, binder, solvent, extender, rheology_mod, defoamer

Process parameters (free within bounds):
    temp_C     ∈ [15, 45]   — mix temperature
    shear_rpm  ∈ [100, 4000] — disperser speed

Objectives (a made-up but plausible forward model — see ``forward`` below):
    cost        — minimize      ($/kg, weighted by ingredient prices)
    opacity     — maximize      (unitless, proxy for hiding power)
    drying_time — within_range  (minutes, want [25, 45])

Constraints:
    sum(mass_fractions) = 1.

Acquisition:  qei — target-type goal triggers the advanced-goal scoring path.

Visual:
    2×2 diagnostic figure —
      (a) objective-space scatter cost vs opacity, shaded by drying_time,
      (b) drying_time histogram (train) with acceptance band shaded,
      (c) ingredient-composition bar chart for the top-5 predicted candidates,
      (d) sum-to-one check for predictions (should all equal 1).
"""

from __future__ import annotations

import numpy as np
from _common import mpl_setup, post_optimize_and_wait

INGREDIENTS = ["pigment", "binder", "solvent", "extender", "rheology_mod", "defoamer"]
N_ING = len(INGREDIENTS)
PRICE_PER_KG = np.array([18.0, 6.0, 1.5, 0.8, 30.0, 55.0])

DRY_MIN, DRY_MAX, DRY_IDEAL = 25.0, 45.0, 35.0


def forward(X: np.ndarray) -> np.ndarray:
    """Return (cost, opacity, drying_time) for each row of X (N, 8)."""
    W = X[:, :N_ING]
    temp = X[:, N_ING]
    shear = X[:, N_ING + 1]
    cost = W @ PRICE_PER_KG
    opacity = (
        60.0 * W[:, 0]
        + 10.0 * W[:, 1]
        + 5.0 * W[:, 3]
        + 0.006 * shear
        - 40.0 * (W[:, 0] - 0.3) ** 2
    )
    drying_time = (
        60.0 * (1.0 - W[:, 2])
        - 0.4 * (temp - 20.0)
        + 4.0 * W[:, 4]
        - 1.5 * W[:, 5]
    )
    return np.stack([cost, opacity, drying_time], axis=1)


def sample_formulations(rng: np.random.Generator, n: int) -> np.ndarray:
    W = rng.dirichlet(alpha=np.array([3.0, 5.0, 4.0, 2.0, 0.6, 0.3]), size=n)
    temp = rng.uniform(15.0, 45.0, size=n)
    shear = rng.uniform(100.0, 4000.0, size=n)
    return np.concatenate([W, temp[:, None], shear[:, None]], axis=1)


def main() -> None:
    rng = np.random.default_rng(37)
    X = sample_formulations(rng, 40)
    Y = forward(X)

    params = [{"name": n, "type": "float", "min": 0.0, "max": 1.0} for n in INGREDIENTS]
    params += [
        {"name": "temp_C", "type": "float", "min": 15.0, "max": 45.0},
        {"name": "shear_rpm", "type": "float", "min": 100.0, "max": 4000.0},
    ]

    payload = {
        "dataset": {"X": X.tolist(), "Y": Y.tolist()},
        "search_space": {"parameters": params},
        "objectives": {
            "cost": {"goal": "min"},
            "opacity": {"goal": "max"},
            "drying_time": {
                "goal": "within_range",
                "range": {"min": DRY_MIN, "max": DRY_MAX, "ideal": DRY_IDEAL, "weight": 1.0, "ideal_weight": 0.5},
            },
        },
        "optimization_config": {
            "acquisition": "qei",
            "batch_size": 5,
            "max_evaluations": 28,
            "seed": 41,
            "sum_constraints": [{"indices": list(range(N_ING)), "target_sum": 1.0}],
            "parameter_scaling": "minmax",
            "return_maps": False,
        },
    }

    result = post_optimize_and_wait("demo_07_paint", payload)
    preds = result.get("predictions", [])
    if not preds:
        print("no predictions returned")
        return

    names = INGREDIENTS + ["temp_C", "shear_rpm"]
    P = np.array([[pred["candidate"].get(n, 0.0) for n in names] for pred in preds])
    Yp = forward(P)

    plt = mpl_setup()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=120)

    ax = axes[0, 0]
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=Y[:, 2], cmap="viridis", s=45, alpha=0.8, label="train")
    ax.scatter(Yp[:, 0], Yp[:, 1], c=Yp[:, 2], cmap="viridis", s=160, marker="X",
               edgecolor="red", linewidth=1.5, label="pred")
    fig.colorbar(sc, ax=ax, label="drying_time (train)")
    ax.set_xlabel("cost ($/kg)")
    ax.set_ylabel("opacity")
    ax.set_title("(a) Objective space cost vs opacity")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    ax.hist(Y[:, 2], bins=20, alpha=0.6, label="train drying_time")
    ax.axvspan(DRY_MIN, DRY_MAX, color="green", alpha=0.2, label=f"want [{DRY_MIN:.0f},{DRY_MAX:.0f}]")
    ax.axvline(DRY_IDEAL, color="green", linestyle="--", label=f"ideal={DRY_IDEAL:.0f}")
    for v in Yp[:, 2]:
        ax.axvline(v, color="red", alpha=0.7)
    ax.set_xlabel("drying_time (min)")
    ax.set_ylabel("count")
    ax.set_title("(b) Drying-time band vs predictions")
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    k = min(5, len(P))
    order = np.argsort(Yp[:, 0])[:k]
    bar_x = np.arange(k)
    bottoms = np.zeros(k)
    cmap = plt.get_cmap("tab10")
    for i, nm in enumerate(INGREDIENTS):
        vals = P[order, i]
        ax.bar(bar_x, vals, bottom=bottoms, label=nm, color=cmap(i % 10))
        bottoms += vals
    ax.set_xticks(bar_x)
    ax.set_xticklabels([f"#{o + 1}" for o in order])
    ax.set_ylabel("mass fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("(c) Composition — lowest-cost predictions")
    ax.legend(loc="upper right", ncol=2, fontsize=8)

    ax = axes[1, 1]
    sums = P[:, :N_ING].sum(axis=1)
    ax.bar(range(len(sums)), sums, color="steelblue")
    ax.axhline(1.0, color="black", linestyle="--")
    ax.set_xlabel("candidate index")
    ax.set_ylabel("Σ mass fractions")
    ax.set_ylim(0.95, 1.05)
    ax.set_title("(d) Sum-to-one sanity check")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
