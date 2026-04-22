"""Demo 10 - Cosmetic O/W emulsion (audit style, no PCA).

Complementary to ``demo_09_pharma_tablet_pca.py``. This demo exercises the
engine on the **non-PCA** path: the GP is built directly in ingredient +
process space, so the acquisition, reconstruction and constraint logic are
tested without the PCA round-trip. The audit format mirrors demo_09 at a
smaller scale:

1. Training formulations (ingredient x sample, parameter x sample, objective x sample)
2. Optimization spec (objectives + HARD constraints: sum, bounds, ratio)
3. Predicted candidates (named x candidate)
4. GP posterior mean + std per objective per candidate
5. Hard-constraint checks (sum = 1, per-param bounds, cetearyl/PEG ratio)
6. GP calibration vs forward model, soft-goal compliance, best-vs-best
7. Visual summary (objectives placed against training distributions; phase
   composition bars of top-3 predictions)

Ingredients (sum to 1.0):
    water, glycerin, propanediol, squalane, jojoba_oil, cetearyl_alcohol,
    PEG100_stearate, niacinamide, hyaluronic_acid, preservative.

Process parameters: mix_temp_C, homog_speed_rpm.

Named HARD constraints under test:
    * ``sum(mass fractions) == 1.0``
    * per-parameter bounds (10 ingredient + 2 process variables)
    * ``cetearyl_alcohol / PEG100_stearate`` in ``[1.5, 3.0]`` -- a standard
      co-emulsifier:primary-emulsifier ratio. This is the ratio-constraint
      regression check for the non-PCA path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from _common import (
    check_bounds,
    check_ratio,
    check_result,
    check_sum_to_one,
    gp_calibration,
    mpl_setup,
    post_optimize_and_wait,
    render_key_value_block,
    render_table,
    section,
)

OUT_DIR = Path("examples/responses/demo_10")

INGREDIENTS = [
    "water", "glycerin", "propanediol", "squalane", "jojoba_oil",
    "cetearyl_alcohol", "PEG100_stearate", "niacinamide", "hyaluronic_acid", "preservative",
]
PROCESS = ["mix_temp_C", "homog_speed_rpm"]

ING_BOUNDS = {
    "water":            (0.55,  0.80),
    "glycerin":         (0.02,  0.10),
    "propanediol":      (0.00,  0.05),
    "squalane":         (0.01,  0.15),
    "jojoba_oil":       (0.00,  0.10),
    "cetearyl_alcohol": (0.02,  0.06),
    "PEG100_stearate":  (0.01,  0.04),
    "niacinamide":      (0.00,  0.05),
    "hyaluronic_acid":  (0.00,  0.02),
    "preservative":     (0.005, 0.015),
}
PROC_BOUNDS = {"mix_temp_C": (60.0, 85.0), "homog_speed_rpm": (2000.0, 12000.0)}

CETEARYL_IDX = INGREDIENTS.index("cetearyl_alcohol")
PEG_IDX = INGREDIENTS.index("PEG100_stearate")
RATIO_MIN, RATIO_MAX = 1.5, 3.0

OBJECTIVES = ["viscosity_cP", "spreadability", "stability_days", "cost_per_kg"]

PRICE_PER_KG = np.array([
    0.5, 5.0, 8.0, 40.0, 30.0, 12.0, 25.0, 60.0, 800.0, 20.0,
])

VISC_LO, VISC_HI, VISC_IDEAL = 15000.0, 30000.0, 22500.0

PHASE_GROUPS = {
    "aqueous":    ["water", "glycerin", "propanediol"],
    "oil":        ["squalane", "jojoba_oil"],
    "emulsifier": ["cetearyl_alcohol", "PEG100_stearate"],
    "active":     ["niacinamide", "hyaluronic_acid"],
    "functional": ["preservative"],
}


def forward(X: np.ndarray) -> np.ndarray:
    """Return (viscosity, spreadability, stability_days, cost_per_kg)."""
    W = X[:, : len(INGREDIENTS)]
    (
        water, _gly, _pdo, squalane, jojoba, cetearyl, peg, niacin, _ha, pres,
    ) = W.T
    mix_temp = X[:, len(INGREDIENTS)]
    homog = X[:, len(INGREDIENTS) + 1]

    viscosity = (
        5000.0
        + 400000.0 * cetearyl
        + 300000.0 * peg
        + 100000.0 * (squalane + jojoba)
        - 15000.0 * (water - 0.70)
        + 0.08 * homog
    )
    spreadability = (
        1.0 + 3.0 * squalane + 5.0 * jojoba - 2.0 * cetearyl + 0.0001 * homog - 0.00002 * viscosity
    )
    stability_days = (
        45.0 + 2000.0 * pres - 200.0 * np.abs(niacin - 0.03) - 0.3 * np.abs(mix_temp - 75.0)
    )
    cost = W @ PRICE_PER_KG
    return np.stack([viscosity, spreadability, stability_days, cost], axis=1)


def sample_formulations(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample a diverse training set inside the declared bounds and sum=1.

    Mid-point of the declared per-ingredient bounds happens to sum to 1.0 by
    design; we perturb with bounded noise and then iterate a
    *scale-and-clip* projection so every row sums to 1 and every column
    respects its own bound. The ``cetearyl / PEG`` ratio is deliberately
    *not* enforced here; the engine must still return ratio-feasible
    predictions.
    """
    lo = np.array([ING_BOUNDS[name][0] for name in INGREDIENTS])
    hi = np.array([ING_BOUNDS[name][1] for name in INGREDIENTS])
    base = 0.5 * (lo + hi)
    W = np.tile(base, (n, 1)) + rng.normal(0.0, 0.025, size=(n, len(INGREDIENTS)))
    W = np.clip(W, lo, hi)
    for r in range(n):
        x = W[r]
        for _ in range(64):
            s = float(x.sum())
            if abs(s - 1.0) < 1e-7:
                break
            if s > 1e-18:
                x = x * (1.0 / s)
            x = np.clip(x, lo, hi)
        W[r] = x
    mix_temp = rng.uniform(*PROC_BOUNDS["mix_temp_C"], size=n)
    homog = rng.uniform(*PROC_BOUNDS["homog_speed_rpm"], size=n)
    return np.concatenate([W, mix_temp[:, None], homog[:, None]], axis=1)


def _stage1_inputs(X: np.ndarray, Y: np.ndarray) -> None:
    section("Stage 1 - Training formulations (input)")
    sample_cols = [f"s{i + 1:02d}" for i in range(X.shape[0])]
    render_table(
        "Ingredient mass fractions (ingredient x sample)",
        col_headers=sample_cols, row_headers=INGREDIENTS,
        matrix=X[:, : len(INGREDIENTS)].T, fmt="{:.4f}", row_label="ingredient",
        save_path=OUT_DIR / "01_ingredients.txt",
    )
    render_table(
        "Process parameters (parameter x sample)",
        col_headers=sample_cols, row_headers=PROCESS,
        matrix=X[:, len(INGREDIENTS) :].T, fmt="{:.2f}", row_label="parameter",
        save_path=OUT_DIR / "02_parameters.txt",
    )
    render_table(
        "Measured objective values (objective x sample)",
        col_headers=sample_cols, row_headers=OBJECTIVES,
        matrix=Y.T, fmt="{:.2f}", row_label="objective",
        save_path=OUT_DIR / "03_objectives.txt",
    )

    train_ratios = X[:, CETEARYL_IDX] / np.maximum(X[:, PEG_IDX], 1e-12)
    in_band = float(np.mean((train_ratios >= RATIO_MIN) & (train_ratios <= RATIO_MAX)))
    print(
        f"\n  note: {in_band * 100:.0f}% of training samples satisfy the"
        f" cetearyl/PEG ratio in [{RATIO_MIN}, {RATIO_MAX}]; the engine"
        "\n  must still return predictions that respect this constraint."
    )


def _stage2_spec() -> None:
    section("Stage 2 - Optimization specification")
    render_key_value_block(
        "Objectives (goal, limits)",
        [
            ("viscosity_cP",    f"within_range [{VISC_LO:.0f}, {VISC_HI:.0f}] ideal={VISC_IDEAL:.0f}"),
            ("spreadability",   "max"),
            ("stability_days",  "max"),
            ("cost_per_kg",     "min"),
        ],
        save_path=OUT_DIR / "04_goals.txt",
    )
    constraints: list[tuple[str, str]] = [
        ("sum(ingredients) = 1.0", f"indices {list(range(len(INGREDIENTS)))}"),
        ("ratio cetearyl/PEG",     f"in [{RATIO_MIN:.2f}, {RATIO_MAX:.2f}]"),
    ]
    for name in INGREDIENTS:
        lo, hi = ING_BOUNDS[name]
        constraints.append((f"bound {name}", f"[{lo:g}, {hi:g}]"))
    for name in PROCESS:
        lo, hi = PROC_BOUNDS[name]
        constraints.append((f"bound {name}", f"[{lo:g}, {hi:g}]"))
    render_key_value_block(
        "Constraints", constraints,
        save_path=OUT_DIR / "05_constraints.txt",
    )
    render_key_value_block(
        "Engine config",
        [
            ("acquisition",     "qei (ParEGO scalarization)"),
            ("batch_size",      "5"),
            ("max_evaluations", "32"),
            ("seed",            "11"),
            ("use_pca",         "False  (ingredient-space GP)"),
        ],
        save_path=OUT_DIR / "06_config.txt",
    )


def _stage3_predictions(preds: list[dict]) -> np.ndarray:
    section("Stage 3 - Predicted candidates (named x candidate)")
    cand_cols = [f"c{i + 1}" for i in range(len(preds))]
    names = INGREDIENTS + PROCESS
    P = np.array([[p["candidate"][n] for n in names] for p in preds], dtype=float)

    render_table(
        "Ingredient mass fractions (ingredient x candidate)",
        col_headers=cand_cols, row_headers=INGREDIENTS,
        matrix=P[:, : len(INGREDIENTS)].T,
        fmt="{:.4f}", row_label="ingredient",
        save_path=OUT_DIR / "07_predictions_ingredients.txt",
    )
    render_table(
        "Process parameters (parameter x candidate)",
        col_headers=cand_cols, row_headers=PROCESS,
        matrix=P[:, len(INGREDIENTS) :].T,
        fmt="{:.2f}", row_label="parameter",
        save_path=OUT_DIR / "08_predictions_parameters.txt",
    )
    return P


def _stage4_gp_posteriors(preds: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    section("Stage 4 - GP posterior at predicted candidates")
    cand_cols = [f"c{i + 1}" for i in range(len(preds))]
    mus = np.array([p["objectives"] for p in preds], dtype=float)
    vars_ = np.array([p.get("variances", [0.0] * len(OBJECTIVES)) for p in preds], dtype=float)
    stds = np.sqrt(np.maximum(vars_, 0.0))

    render_table(
        "GP posterior mean per objective (objective x candidate)",
        col_headers=cand_cols, row_headers=OBJECTIVES,
        matrix=mus.T, fmt="{:.2f}", row_label="objective",
        save_path=OUT_DIR / "09_gp_mean.txt",
    )
    render_table(
        "GP posterior std per objective (objective x candidate)",
        col_headers=cand_cols, row_headers=OBJECTIVES,
        matrix=stds.T, fmt="{:.2f}", row_label="objective",
        save_path=OUT_DIR / "10_gp_std.txt",
    )
    return mus, stds


def _stage5_hard_constraints(P: np.ndarray) -> dict[str, bool]:
    section("Stage 5 - Hard-constraint checks on predictions")
    ing_idx = list(range(len(INGREDIENTS)))
    ok_sum, sum_err = check_sum_to_one(P, ing_idx, target=1.0, tol=1e-4)
    check_result("sum(ingredients) == 1.0", ok_sum, f"max |error| = {np.max(np.abs(sum_err)):.2e}")

    all_bounds = [ING_BOUNDS[n] for n in INGREDIENTS] + [PROC_BOUNDS[n] for n in PROCESS]
    ok_bounds, bviol = check_bounds(P, all_bounds)
    viol_count = sum(len(v) for v in bviol)
    check_result("all values within [min, max] bounds", ok_bounds, f"{viol_count} field(s) out of bounds")

    ok_ratio, ratios = check_ratio(P, CETEARYL_IDX, PEG_IDX, RATIO_MIN, RATIO_MAX)
    ratio_str = ", ".join(f"{r:.3f}" for r in ratios)
    check_result(
        f"ratio cetearyl/PEG in [{RATIO_MIN:.2f}, {RATIO_MAX:.2f}]",
        ok_ratio, f"ratios = [{ratio_str}]",
    )
    return {"sum": ok_sum, "bounds": ok_bounds, "ratio": ok_ratio}


def _stage6_metrics(
    P: np.ndarray, Y_train: np.ndarray, mus: np.ndarray, stds: np.ndarray,
) -> None:
    section("Stage 6 - GP calibration and goal compliance")
    Yp_true = forward(P)

    cal_rows = []
    for k in range(len(OBJECTIVES)):
        stats = gp_calibration(mus[:, k], stds[:, k], Yp_true[:, k])
        cal_rows.append([stats["mae"], stats["rmse"], stats["mean_abs_z"]])
    render_table(
        "GP posterior vs forward-model truth at predicted points",
        col_headers=["MAE", "RMSE", "mean |z|"],
        row_headers=OBJECTIVES,
        matrix=np.array(cal_rows), fmt="{:.4g}", row_label="objective",
        save_path=OUT_DIR / "11_gp_calibration.txt",
    )

    print(
        "\n  Soft-goal compliance on predictions (informational; not hard"
        "\n  constraints, so less than all in-band does not imply a bug):"
    )
    in_band = (Yp_true[:, 0] >= VISC_LO) & (Yp_true[:, 0] <= VISC_HI)
    n = Yp_true.shape[0]
    print(f"    viscosity in [{VISC_LO:.0f}, {VISC_HI:.0f}]: {int(in_band.sum())}/{n}")

    best_train = np.array([
        np.abs(Y_train[:, 0] - VISC_IDEAL).min() + VISC_IDEAL,
        Y_train[:, 1].max(), Y_train[:, 2].max(), Y_train[:, 3].min(),
    ])
    score = (
        np.abs(Yp_true[:, 0] - VISC_IDEAL) / VISC_IDEAL
        - 0.3 * Yp_true[:, 1] - 0.02 * Yp_true[:, 2] + 0.005 * Yp_true[:, 3]
    )
    best_pred = Yp_true[int(np.argmin(score))]
    render_table(
        "Best-training vs best-prediction (forward model)",
        col_headers=["best_train", "best_pred", "delta"],
        row_headers=OBJECTIVES,
        matrix=np.stack([best_train, best_pred, best_pred - best_train], axis=1),
        fmt="{:.3f}", row_label="objective",
        save_path=OUT_DIR / "12_best_vs_best.txt",
    )


def _stage7_visual(P: np.ndarray, Y_train: np.ndarray) -> None:
    section("Stage 7 - Visual summary")
    plt = mpl_setup()
    Yp_true = forward(P)

    fig = plt.figure(figsize=(13, 6.5), dpi=110)
    gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.35)

    obj_info = [
        ("viscosity_cP",   "band",  VISC_LO, VISC_HI, VISC_IDEAL, f"ideal {VISC_IDEAL:.0f}"),
        ("spreadability",  "max",   None,    None,    None,       "higher is better"),
        ("stability_days", "max",   None,    None,    None,       "higher is better"),
        ("cost_per_kg",    "min",   None,    None,    None,       "lower is better"),
    ]
    score = (
        np.abs(Yp_true[:, 0] - VISC_IDEAL) / VISC_IDEAL
        - 0.3 * Yp_true[:, 1] - 0.02 * Yp_true[:, 2] + 0.005 * Yp_true[:, 3]
    )
    best_pred_idx = int(np.argmin(score))
    for k, (name, kind, lo, hi, ideal, subtitle) in enumerate(obj_info):
        ax = fig.add_subplot(gs[0, k])
        y = Y_train[:, k]
        ax.hist(y, bins=12, color="lightsteelblue", edgecolor="gray", alpha=0.9)
        if kind == "band":
            ax.axvspan(lo, hi, color="green", alpha=0.12, label="target band")
            ax.axvline(ideal, color="green", lw=1.8, ls="--", label="ideal")
        for y_pred in Yp_true[:, k]:
            ax.axvline(y_pred, color="red", alpha=0.35, lw=1.0)
        ax.axvline(Yp_true[best_pred_idx, k], color="red", lw=2.4, label="best pred")
        ax.set_title(f"{name}\n({subtitle})", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_yticks([])
        if k == 0:
            ax.legend(loc="upper right", fontsize=7)

    ax_bottom = fig.add_subplot(gs[1, :])
    order = np.argsort(score)[: min(3, len(P))]
    bar_x = np.arange(len(order))
    bottoms = np.zeros(len(order))
    phase_colors = {
        "aqueous": "#4c9ed9", "oil": "#f2b134", "emulsifier": "#b480d8",
        "active": "#78c66a", "functional": "#e96d6d",
    }
    for phase, ing_names in PHASE_GROUPS.items():
        phase_total = np.zeros(len(order))
        for n in ing_names:
            idx = INGREDIENTS.index(n)
            phase_total += P[order, idx]
        ax_bottom.bar(bar_x, phase_total, bottom=bottoms, label=phase, color=phase_colors[phase])
        for j, v in enumerate(phase_total):
            if v > 0.04:
                ax_bottom.text(bar_x[j], bottoms[j] + v / 2, f"{v * 100:.1f}%",
                               ha="center", va="center", fontsize=9, color="black")
        bottoms += phase_total
    ax_bottom.set_xticks(bar_x)
    ax_bottom.set_xticklabels([f"top #{k + 1}" for k in range(len(order))])
    ax_bottom.set_ylabel("mass fraction")
    ax_bottom.set_ylim(0, 1.05)
    ax_bottom.set_title("Top predictions - phase composition")
    ax_bottom.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)

    fig.suptitle(
        "Demo 10 - Cosmetic O/W emulsion: predictions against training distributions",
        fontsize=12,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.show()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(113)
    X = sample_formulations(rng, n=40)
    Y = forward(X)

    _stage1_inputs(X, Y)
    _stage2_spec()

    params = [
        {"name": n, "type": "float", "min": ING_BOUNDS[n][0], "max": ING_BOUNDS[n][1]}
        for n in INGREDIENTS
    ] + [
        {"name": n, "type": "float", "min": PROC_BOUNDS[n][0], "max": PROC_BOUNDS[n][1]}
        for n in PROCESS
    ]

    payload = {
        "dataset": {"X": X.tolist(), "Y": Y.tolist()},
        "search_space": {"parameters": params},
        "objectives": {
            "viscosity_cP": {
                "goal": "within_range",
                "range": {
                    "min": VISC_LO, "max": VISC_HI, "ideal": VISC_IDEAL,
                    "weight": 1.0, "ideal_weight": 0.5,
                },
            },
            "spreadability": {"goal": "max"},
            "stability_days": {"goal": "max"},
            "cost_per_kg": {"goal": "min"},
        },
        "optimization_config": {
            "acquisition": "qei",
            "batch_size": 5,
            "max_evaluations": 32,
            "seed": 11,
            "sum_constraints": [{"indices": list(range(len(INGREDIENTS))), "target_sum": 1.0}],
            "ratio_constraints": [
                {"i": CETEARYL_IDX, "j": PEG_IDX, "min_ratio": RATIO_MIN, "max_ratio": RATIO_MAX},
            ],
            "parameter_scaling": "minmax",
            "value_normalization": "standardize",
            "use_pca": False,
        },
    }

    result = post_optimize_and_wait("demo_10_cosmetic_emulsion", payload, max_poll_s=180.0)
    preds = result.get("predictions", [])
    if not preds:
        print("No predictions returned")
        return

    P = _stage3_predictions(preds)
    mus, stds = _stage4_gp_posteriors(preds)
    checks = _stage5_hard_constraints(P)
    _stage6_metrics(P, Y, mus, stds)
    _stage7_visual(P, Y)

    section("Final verdict", ch="*")
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        print(f"  {len(failed)} HARD-constraint check(s) FAILED: {failed}")
        print("  This indicates an engine correctness problem in the non-PCA path.")
    else:
        print(f"  All hard-constraint checks PASSED on {len(preds)} predictions")
        print("  (sum, bounds, ratio). Non-PCA path is correct.")


if __name__ == "__main__":
    main()
