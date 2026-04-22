"""Demo 08 — EV-battery electrolyte: target conductivity, bounded viscosity,
threshold-gated stability window.

Composition (5 solvent/salt mass fractions, sum to 1):
    EC, DMC, EMC, LiPF6, additive

Process:
    temp_C  ∈ [-10, 60]

Forward model (fictional but plausible):
    conductivity_mS  — higher with LiPF6 and DMC, peaks near 0.10 salt.
    viscosity_cP     — higher with EC and additive.
    stability_V      — higher with EC and low temperature; LiPF6 reduces it.

Goals:
    conductivity  — target  T = 10.5 mS/cm  (±0.5 treated as equivalent)
    viscosity     — within_range [2.0, 5.0] cP (ideal 3.0)
    stability     — maximize_above  threshold 4.2 V  (safety floor)

Constraints:
    sum(mass fractions) = 1.

Visual:
    2×2 figure —
      (a) conductivity histogram (train) with target band,
      (b) viscosity histogram with acceptance band,
      (c) stability histogram with 4.2 V threshold,
      (d) composition bar chart for the top predicted candidates.
"""

from __future__ import annotations

import numpy as np
from _common import mpl_setup, post_optimize_and_wait

INGREDIENTS = ["EC", "DMC", "EMC", "LiPF6", "additive"]
N_ING = len(INGREDIENTS)

COND_TARGET, COND_TOL = 10.5, 0.5
VISC_MIN, VISC_MAX, VISC_IDEAL = 2.0, 5.0, 3.0
STAB_MIN = 4.2


def forward(X: np.ndarray) -> np.ndarray:
    W = X[:, :N_ING]
    temp = X[:, N_ING]
    salt = W[:, 3]
    conductivity = (
        25.0 * W[:, 1]
        + 15.0 * W[:, 2]
        + 120.0 * salt * np.exp(-50.0 * (salt - 0.10) ** 2)
        + 0.02 * (temp - 25.0)
    )
    viscosity = 8.0 * W[:, 0] + 1.5 * W[:, 1] + 1.0 * W[:, 2] + 4.0 * W[:, 4] + 1.0
    stability = 4.0 + 3.0 * W[:, 0] - 2.5 * salt - 0.01 * temp
    return np.stack([conductivity, viscosity, stability], axis=1)


def sample_formulations(rng: np.random.Generator, n: int) -> np.ndarray:
    W = rng.dirichlet(alpha=np.array([5.0, 4.0, 3.0, 0.8, 0.3]), size=n)
    temp = rng.uniform(-10.0, 60.0, size=n)
    return np.concatenate([W, temp[:, None]], axis=1)


def main() -> None:
    rng = np.random.default_rng(53)
    X = sample_formulations(rng, 50)
    Y = forward(X)

    params = [{"name": n, "type": "float", "min": 0.0, "max": 1.0} for n in INGREDIENTS]
    params += [{"name": "temp_C", "type": "float", "min": -10.0, "max": 60.0}]

    payload = {
        "dataset": {"X": X.tolist(), "Y": Y.tolist()},
        "search_space": {"parameters": params},
        "objectives": {
            "conductivity": {"goal": "target", "target_value": COND_TARGET},
            "viscosity": {
                "goal": "within_range",
                "range": {"min": VISC_MIN, "max": VISC_MAX, "ideal": VISC_IDEAL, "weight": 1.0, "ideal_weight": 0.5},
            },
            "stability": {
                "goal": "maximize_above",
                "threshold": {"type": ">=", "value": STAB_MIN, "weight": 1.0},
            },
        },
        "optimization_config": {
            "acquisition": "qei",
            "batch_size": 5,
            "max_evaluations": 28,
            "seed": 59,
            "sum_constraints": [{"indices": list(range(N_ING)), "target_sum": 1.0}],
            "target_tolerance": COND_TOL,
            "target_variance_penalty": 0.25,
            "parameter_scaling": "minmax",
            "return_maps": False,
        },
    }

    result = post_optimize_and_wait("demo_08_ev_electrolyte", payload)
    preds = result.get("predictions", [])
    if not preds:
        print("no predictions returned")
        return

    names = INGREDIENTS + ["temp_C"]
    P = np.array([[pred["candidate"].get(n, 0.0) for n in names] for pred in preds])
    Yp = forward(P)

    plt = mpl_setup()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=120)

    ax = axes[0, 0]
    ax.hist(Y[:, 0], bins=20, alpha=0.6, label="train")
    ax.axvspan(COND_TARGET - COND_TOL, COND_TARGET + COND_TOL, color="green", alpha=0.2,
               label=f"target {COND_TARGET}±{COND_TOL} mS")
    for v in Yp[:, 0]:
        ax.axvline(v, color="red", alpha=0.7)
    ax.set_xlabel("conductivity (mS/cm)")
    ax.set_title("(a) conductivity — target")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    ax.hist(Y[:, 1], bins=20, alpha=0.6, label="train")
    ax.axvspan(VISC_MIN, VISC_MAX, color="green", alpha=0.2, label=f"want [{VISC_MIN},{VISC_MAX}] cP")
    ax.axvline(VISC_IDEAL, color="green", linestyle="--", label=f"ideal {VISC_IDEAL}")
    for v in Yp[:, 1]:
        ax.axvline(v, color="red", alpha=0.7)
    ax.set_xlabel("viscosity (cP)")
    ax.set_title("(b) viscosity — within range")
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    ax.hist(Y[:, 2], bins=20, alpha=0.6, label="train")
    ax.axvline(STAB_MIN, color="green", linestyle="--", label=f"floor {STAB_MIN} V")
    for v in Yp[:, 2]:
        ax.axvline(v, color="red", alpha=0.7)
    ax.set_xlabel("stability (V)")
    ax.set_title("(c) stability — maximize_above")
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    k = len(P)
    bar_x = np.arange(k)
    bottoms = np.zeros(k)
    cmap = plt.get_cmap("tab10")
    for i, nm in enumerate(INGREDIENTS):
        vals = P[:, i]
        ax.bar(bar_x, vals, bottom=bottoms, label=nm, color=cmap(i % 10))
        bottoms += vals
    ax.set_xticks(bar_x)
    ax.set_xticklabels([f"#{i + 1}" for i in range(k)])
    ax.set_ylabel("mass fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("(d) Predicted compositions")
    ax.legend(loc="upper right", ncol=2, fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
