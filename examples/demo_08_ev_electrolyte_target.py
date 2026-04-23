"""Demo 08 — EV-battery electrolyte: target conductivity, bounded viscosity,
**hard-enforced** stability floor. PCA-backed pipeline.

This is the flagship demo for the ``enforce_above`` hard-constraint goal.
The soft twin ``maximize_above`` would only *prefer* candidates clearing the
stability floor — useful when the threshold is a preference, disastrous
when it is a safety requirement. ``enforce_above`` filters the Sobol
acquisition grid on the GP posterior so every returned candidate that
reports ``enforced_goals_satisfied=true`` is provably above the floor on
the surrogate, and the response's ``diagnostics.enforcement`` block
summarises the batch-level feasibility.

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
    stability     — **enforce_above 4.2 V** (safety floor; hard constraint
                     on the GP posterior, not just a soft preference)

Pipeline:
    - PCA fit on the 5-ingredient + 1-process input space (k=3).
    - GP surrogates in normalized PCA coordinates.
    - Sobol-scoring acquisition path (any non-min/max goal routes here).
    - Per-prediction feasibility flag + SLSQP reconstruction back to named
      ingredients.

Constraints:
    sum(mass fractions) = 1.

Visual:
    2×2 figure —
      (a) conductivity histogram (train) with target band,
      (b) viscosity histogram with acceptance band,
      (c) stability histogram with 4.2 V floor; predicted points colored
          by the ``enforced_goals_satisfied`` flag returned by the engine,
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
        25.0 * W[:, 1] + 15.0 * W[:, 2] + 120.0 * salt * np.exp(-50.0 * (salt - 0.10) ** 2) + 0.02 * (temp - 25.0)
    )
    viscosity = 8.0 * W[:, 0] + 1.5 * W[:, 1] + 1.0 * W[:, 2] + 4.0 * W[:, 4] + 1.0
    stability = 4.0 + 3.0 * W[:, 0] - 2.5 * salt - 0.01 * temp
    return np.stack([conductivity, viscosity, stability], axis=1)


def sample_formulations(rng: np.random.Generator, n: int) -> np.ndarray:
    # Dirichlet priors biased toward EC-rich formulations so the training set
    # spans a useful range of stabilities on both sides of the 4.2 V floor.
    W = rng.dirichlet(alpha=np.array([5.0, 4.0, 3.0, 0.8, 0.3]), size=n)
    temp = rng.uniform(-10.0, 60.0, size=n)
    return np.concatenate([W, temp[:, None]], axis=1)


def main() -> None:
    rng = np.random.default_rng(53)
    X = sample_formulations(rng, 80)
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
                "range": {
                    "min": VISC_MIN,
                    "max": VISC_MAX,
                    "ideal": VISC_IDEAL,
                    "weight": 1.0,
                    "ideal_weight": 0.5,
                },
            },
            # HARD safety floor: the GP-predicted mean of stability MUST clear
            # 4.2 V for a candidate to count as feasible. Every returned
            # prediction carries an ``enforced_goals_satisfied`` flag and the
            # response's ``diagnostics.enforcement`` block summarises the batch.
            "stability": {"goal": "enforce_above", "threshold_value": STAB_MIN},
        },
        "optimization_config": {
            "acquisition": "qnehvi",
            "batch_size": 5,
            "max_evaluations": 32,
            "seed": 59,
            "sum_constraints": [{"indices": list(range(N_ING)), "target_sum": 1.0}],
            "target_tolerance": COND_TOL,
            "target_variance_penalty": 0.25,
            # Optional safety knob: require the GP lower-confidence bound to
            # satisfy the floor rather than the mean alone. Set > 0 for more
            # conservative enforcement when training is sparse near the
            # threshold; 0.0 (default) uses the mean and matches the text.
            "enforcement_uncertainty_margin": 0.0,
            # PCA is SATOR's preferred pipeline even for low-dimensional
            # ingredient problems: it gives us a small, dense latent space
            # for the GP and a single SLSQP reconstruction back to named
            # ingredients that provably satisfies the sum constraint.
            "use_pca": True,
            "pca_dimension": 3,
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
    ok_flags = np.array([bool(pred.get("enforced_goals_satisfied", False)) for pred in preds])
    enf_diag = (result.get("diagnostics") or {}).get("enforcement") or {}

    plt = mpl_setup()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=120)

    ax = axes[0, 0]
    ax.hist(Y[:, 0], bins=20, alpha=0.6, label="train")
    ax.axvspan(
        COND_TARGET - COND_TOL,
        COND_TARGET + COND_TOL,
        color="green",
        alpha=0.2,
        label=f"target {COND_TARGET}±{COND_TOL} mS",
    )
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
    ax.hist(Y[:, 2], bins=20, alpha=0.6, color="0.7", label="train")
    ax.axvspan(-np.inf, STAB_MIN, color="red", alpha=0.08)
    ax.axvline(STAB_MIN, color="green", linestyle="--", linewidth=2, label=f"floor {STAB_MIN} V")
    # Predicted candidates: green for enforce_above-feasible, red for not.
    feasible_vals = Yp[ok_flags, 2] if ok_flags.any() else np.array([])
    infeasible_vals = Yp[~ok_flags, 2] if (~ok_flags).any() else np.array([])
    for v in feasible_vals:
        ax.axvline(v, color="#1f9d55", linewidth=2, alpha=0.9)
    for v in infeasible_vals:
        ax.axvline(v, color="red", linewidth=2, alpha=0.9, linestyle=":")
    n_ok = int(enf_diag.get("n_satisfied", int(ok_flags.sum())))
    n_tot = int(enf_diag.get("n_total", int(ok_flags.size)))
    margin = float(enf_diag.get("uncertainty_margin", 0.0))
    ax.set_xlabel("stability (V)")
    ax.set_title(
        f"(c) stability — enforce_above {STAB_MIN} V  ({n_ok}/{n_tot} feasible"
        f"{'' if margin == 0.0 else f', margin={margin:g}σ'})"
    )
    # Custom legend combining histogram + feasibility colors.
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color="green", linestyle="--", linewidth=2, label=f"floor {STAB_MIN} V"),
        Line2D([0], [0], color="#1f9d55", linewidth=2, label="predicted — feasible"),
        Line2D([0], [0], color="red", linewidth=2, linestyle=":", label="predicted — violates"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

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
    ax.set_xticklabels([f"#{i + 1}{' ✓' if ok_flags[i] else ' ✗'}" for i in range(k)])
    ax.set_ylabel("mass fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title("(d) Predicted compositions (✓ = enforce_above-feasible)")
    ax.legend(loc="upper right", ncol=2, fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
