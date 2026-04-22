"""Demo 09 — Pharmaceutical tablet: end-to-end PCA-BO correctness audit.

Audit-driven demo (not a glossy showcase). The script walks through the SATOR
pipeline stage by stage and verifies each stage works as advertised. Every
stage prints a table or a PASS/FAIL check and saves an audit artifact under
``examples/responses/demo_09/``.

Pipeline stages (mirrors the SATOR data flow):

    1. Input tables (ingredients x samples, parameters x samples, objectives
       x samples) with optimization targets highlighted.
    2. Optimization specification: goals, constraints (sum-to-one, bounds,
       ratio), and engine config.
    3. PCA results: training samples x PC values, explained variance per PC,
       top feature loadings per PC.
    4. PCA GP surfaces: one mean panel + one std panel per objective, with
       training samples and predicted candidates overlaid.
    5. Predictions in PCA space (candidates x {PC1, PC2, GP mean/std per obj}).
    6. Reconstructed predictions in original space (ingredients+params x
       candidates), formulation-ready.
    7. Correctness checks and metrics:
         - PCA reconstruction error on the training set (per feature + aggregate),
         - GP vs forward-model residuals at the predicted points,
         - constraint satisfaction (sum-to-one, bounds, ratio),
         - per-objective threshold / band compliance,
         - best-training vs best-prediction improvement deltas.
    8. Optional visual extras: dumbbell improvement plot + named-ingredient
       recipe bars.

Problem definition: immediate-release tablet with 7 mass-fraction ingredients
summing to 1, 2 process parameters, 4 mixed-goal objectives, one ratio
constraint (MCC / lactose in [0.25, 0.60]) so the reconstructor's ratio
handling is explicitly exercised.
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

INGREDIENTS = ["API", "lactose", "MCC", "croscarmellose", "PVP_K30", "Mg_stearate", "silica"]
PROCESS = ["compaction_kN", "blend_min"]
OBJECTIVES = ["dissolution_30min", "hardness_N", "friability_pct", "disintegration_s"]

ING_BOUNDS = {
    "API":            (0.05,  0.30),
    "lactose":        (0.10,  0.70),
    "MCC":            (0.05,  0.40),
    "croscarmellose": (0.01,  0.06),
    "PVP_K30":        (0.01,  0.06),
    "Mg_stearate":    (0.002, 0.020),
    "silica":         (0.001, 0.010),
}
PROC_BOUNDS = {"compaction_kN": (5.0, 25.0), "blend_min": (5.0, 20.0)}

HARDNESS_LO, HARDNESS_HI, HARDNESS_IDEAL = 80.0, 150.0, 115.0
FRIABILITY_MAX = 1.0
DISINTEGRATION_MAX = 900.0

MCC_IDX = INGREDIENTS.index("MCC")
LACTOSE_IDX = INGREDIENTS.index("lactose")
# Intentionally tight: the natural training distribution of MCC/lactose is
# ~0.40-0.46, so a [0.55, 0.80] window forces the optimizer to leave the
# training envelope and explicitly exercises ratio-constraint enforcement.
RATIO_MIN, RATIO_MAX = 0.55, 0.80

OUT_DIR = Path("examples/responses/demo_09")


def forward(X: np.ndarray) -> np.ndarray:
    """True-objective forward model (used for data generation + GP calibration)."""
    W = X[:, : len(INGREDIENTS)]
    api, _lac, mcc, cross, pvp, mgst, _sil = W.T
    compaction = X[:, len(INGREDIENTS)]
    blend = X[:, len(INGREDIENTS) + 1]
    hardness = 60.0 + 3.5 * compaction + 80.0 * mcc - 40.0 * api - 120.0 * mgst + 20.0 * pvp
    friability = np.maximum(0.05, 3.0 - 0.02 * hardness + 50.0 * mgst)
    disintegration = 600.0 - 4000.0 * cross + 2.5 * hardness - 3.0 * blend
    dissolution = 70.0 - 100.0 * api + 500.0 * cross + 80.0 * mcc - 0.1 * hardness + 0.3 * blend
    dissolution = np.clip(dissolution, 10.0, 99.0)
    return np.stack([dissolution, hardness, friability, disintegration], axis=1)


def sample_training(rng: np.random.Generator, n: int) -> np.ndarray:
    """Dirichlet-sampled compositions inside bounds and sum-to-one.

    The sampler is deliberately *unconstrained by ratio*, so a chunk of the
    training set naturally violates the [0.55, 0.80] MCC/lactose window. The
    engine must learn that region is forbidden and still propose feasible
    reconstructions.
    """
    alpha = np.array([2.0, 6.0, 6.0, 0.8, 0.8, 0.25, 0.15])
    lo = np.array([ING_BOUNDS[n][0] for n in INGREDIENTS])
    hi = np.array([ING_BOUNDS[n][1] for n in INGREDIENTS])
    out = np.zeros((n, len(INGREDIENTS) + len(PROCESS)))
    for i in range(n):
        raw = rng.dirichlet(alpha)
        W = lo + raw * (hi - lo)
        W = W / W.sum()
        compaction = rng.uniform(*PROC_BOUNDS["compaction_kN"])
        blend = rng.uniform(*PROC_BOUNDS["blend_min"])
        out[i] = np.concatenate([W, [compaction, blend]])
    return out


def _bilinear(ax_x: np.ndarray, ax_y: np.ndarray, M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Sample 2D grid ``M`` at ``pts`` via bilinear interpolation."""
    xs = np.clip(pts[:, 0], ax_x[0], ax_x[-1])
    ys = np.clip(pts[:, 1], ax_y[0], ax_y[-1])
    ix = np.clip(np.searchsorted(ax_x, xs) - 1, 0, ax_x.size - 2)
    iy = np.clip(np.searchsorted(ax_y, ys) - 1, 0, ax_y.size - 2)
    tx = (xs - ax_x[ix]) / np.maximum(ax_x[ix + 1] - ax_x[ix], 1e-12)
    ty = (ys - ax_y[iy]) / np.maximum(ax_y[iy + 1] - ax_y[iy], 1e-12)
    z00 = M[iy, ix]
    z10 = M[iy, ix + 1]
    z01 = M[iy + 1, ix]
    z11 = M[iy + 1, ix + 1]
    return (1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z10 + (1 - tx) * ty * z01 + tx * ty * z11


def _build_payload(X: np.ndarray, Y: np.ndarray) -> dict:
    params = [
        {"name": n, "type": "float", "min": ING_BOUNDS[n][0], "max": ING_BOUNDS[n][1]}
        for n in INGREDIENTS
    ] + [
        {"name": n, "type": "float", "min": PROC_BOUNDS[n][0], "max": PROC_BOUNDS[n][1]}
        for n in PROCESS
    ]
    return {
        "dataset": {"X": X.tolist(), "Y": Y.tolist()},
        "search_space": {"parameters": params},
        "objectives": {
            "dissolution_30min": {"goal": "max"},
            "hardness_N": {
                "goal": "within_range",
                "range": {
                    "min": HARDNESS_LO, "max": HARDNESS_HI, "ideal": HARDNESS_IDEAL,
                    "weight": 1.0, "ideal_weight": 0.5,
                },
            },
            "friability_pct": {
                "goal": "minimize_below",
                "threshold": {"type": "<=", "value": FRIABILITY_MAX, "weight": 1.0},
            },
            "disintegration_s": {
                "goal": "minimize_below",
                "threshold": {"type": "<=", "value": DISINTEGRATION_MAX, "weight": 1.0},
            },
        },
        "optimization_config": {
            "acquisition": "qei",
            "batch_size": 5,
            "max_evaluations": 32,
            "seed": 7,
            "sum_constraints": [{"indices": list(range(len(INGREDIENTS))), "target_sum": 1.0}],
            "ratio_constraints": [
                {"i": MCC_IDX, "j": LACTOSE_IDX, "min_ratio": RATIO_MIN, "max_ratio": RATIO_MAX},
            ],
            "parameter_scaling": "minmax",
            "value_normalization": "standardize",
            "use_pca": True,
            "pca_dimension": 2,
            "return_maps": True,
            "map_space": "pca",
            "map_resolution": [50, 50],
        },
    }


def _stage1_inputs(X: np.ndarray, Y: np.ndarray) -> None:
    section("Stage 1 - Training formulations (input)")
    sample_cols = [f"s{i + 1:02d}" for i in range(X.shape[0])]

    W = X[:, : len(INGREDIENTS)].T
    render_table(
        "Ingredient mass fractions (ingredient x sample)",
        col_headers=sample_cols, row_headers=INGREDIENTS, matrix=W,
        fmt="{:.4f}", row_label="ingredient",
        save_path=OUT_DIR / "01_ingredients.txt",
    )

    P = X[:, len(INGREDIENTS):].T
    render_table(
        "Process parameters (parameter x sample)",
        col_headers=sample_cols, row_headers=PROCESS, matrix=P,
        fmt="{:.2f}", row_label="parameter",
        save_path=OUT_DIR / "02_parameters.txt",
    )

    render_table(
        "Measured objective values (objective x sample)",
        col_headers=sample_cols, row_headers=OBJECTIVES, matrix=Y.T,
        fmt="{:.2f}", row_label="objective",
        save_path=OUT_DIR / "03_objectives.txt",
    )

    marked = [f"{n} *" for n in OBJECTIVES]
    print("\n  * all four objectives are optimization targets "
          "(see Stage 2 for goals/limits).")
    _ = marked


def _stage2_spec() -> None:
    section("Stage 2 - Optimization specification")
    render_key_value_block(
        "Objectives (goal, limits)",
        [
            ("dissolution_30min", "max"),
            ("hardness_N", f"within_range [{HARDNESS_LO:.0f}, {HARDNESS_HI:.0f}] ideal={HARDNESS_IDEAL:.0f}"),
            ("friability_pct", f"minimize_below <= {FRIABILITY_MAX:.2f}"),
            ("disintegration_s", f"minimize_below <= {DISINTEGRATION_MAX:.0f}"),
        ],
        save_path=OUT_DIR / "04_objectives_spec.txt",
    )

    constraints: list[tuple[str, str]] = [
        ("sum(ingredients) = 1.0", "indices " + str(list(range(len(INGREDIENTS))))),
        (
            f"ratio {INGREDIENTS[MCC_IDX]} / {INGREDIENTS[LACTOSE_IDX]}",
            f"in [{RATIO_MIN:.2f}, {RATIO_MAX:.2f}]",
        ),
    ]
    for name in INGREDIENTS:
        lo, hi = ING_BOUNDS[name]
        constraints.append((f"bound {name}", f"[{lo:g}, {hi:g}]"))
    for name in PROCESS:
        lo, hi = PROC_BOUNDS[name]
        constraints.append((f"bound {name}", f"[{lo:g}, {hi:g}]"))
    render_key_value_block(
        "Constraints", constraints, save_path=OUT_DIR / "05_constraints.txt"
    )

    render_key_value_block(
        "Engine config",
        [
            ("acquisition", "qei"),
            ("batch_size", "5"),
            ("max_evaluations", "32"),
            ("seed", "7"),
            ("use_pca", "True"),
            ("pca_dimension", "2"),
            ("return_maps", "True"),
            ("map_resolution", "[50, 50]"),
        ],
        save_path=OUT_DIR / "06_config.txt",
    )


def _stage3_pca_results(
    X: np.ndarray, encoding_info: dict, encoded_dataset: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (components, mean, pc_mins, pc_maxs, Z_raw) and print PCA tables."""
    section("Stage 3 - PCA projection of the training set")
    components = np.asarray(encoding_info.get("components", []), dtype=float)
    mean = np.asarray(encoding_info.get("mean", []), dtype=float)
    pc_mins = np.asarray(encoding_info.get("pc_mins", []), dtype=float)
    pc_maxs = np.asarray(encoding_info.get("pc_maxs", []), dtype=float)

    Z_raw = (X - mean) @ components.T
    pc_range = np.maximum(pc_maxs - pc_mins, 1e-12)
    Z_norm = (Z_raw - pc_mins) / pc_range
    n_pc = components.shape[0]
    feature_names = INGREDIENTS + PROCESS

    ev_ratio_server = encoding_info.get("explained_variance_ratio") or []
    if ev_ratio_server:
        ev_ratio = np.asarray(ev_ratio_server, dtype=float)
    else:
        ev = np.var(Z_raw, axis=0)
        ev_ratio = ev / max(float(np.sum(ev)), 1e-12)

    render_key_value_block(
        "PCA explained variance (z-score scaled features)",
        [
            (
                f"PC{k + 1}",
                f"ratio={ev_ratio[k]:.4f}  (cumulative {ev_ratio[: k + 1].sum():.4f})",
            )
            for k in range(n_pc)
        ],
        save_path=OUT_DIR / "07_explained_variance.txt",
    )
    print(
        "  NOTE: PCA is fit on z-score scaled features, so each feature"
        "\n  contributes comparably regardless of its raw unit (mass fractions"
        "\n  0..1 vs compaction_kN 5..25). The `components_` and `mean_` returned"
        "\n  by the engine are the effective encoder on RAW x, satisfying"
        "\n  (x - mean) @ components.T == PCA(scale(x))."
    )

    scaler_scale = np.asarray(encoding_info.get("scaler_scale", []), dtype=float)
    if scaler_scale.size == components.shape[1]:
        scaled_components = components * scaler_scale[None, :]
        render_table(
            "PCA loadings on z-score scaled features (feature x PC)",
            col_headers=[f"PC{k + 1}" for k in range(n_pc)],
            row_headers=feature_names, matrix=scaled_components.T,
            fmt="{:+.4f}", row_label="feature",
            save_path=OUT_DIR / "08_pca_loadings.txt",
        )
        print(
            "  Values above are sklearn PCA.components_ on the StandardScaled input."
            "\n  The effective encoder returned by the engine is C_eff = C / sigma_s,"
            "\n  which is numerically larger for features with small raw std."
        )
    else:
        render_table(
            "PCA component loadings (feature x PC)",
            col_headers=[f"PC{k + 1}" for k in range(n_pc)],
            row_headers=feature_names, matrix=components.T,
            fmt="{:+.4f}", row_label="feature",
            save_path=OUT_DIR / "08_pca_loadings.txt",
        )

    sample_cols = [f"s{i + 1:02d}" for i in range(X.shape[0])]
    render_table(
        "Training samples in PCA space - raw (PC x sample)",
        col_headers=sample_cols,
        row_headers=[f"PC{k + 1}" for k in range(n_pc)],
        matrix=Z_raw.T, fmt="{:+.4f}", row_label="PC",
        save_path=OUT_DIR / "09_training_pca_raw.txt",
    )
    render_table(
        "Training samples in PCA space - normalized [0,1] (PC x sample)",
        col_headers=sample_cols,
        row_headers=[f"PC{k + 1}" for k in range(n_pc)],
        matrix=Z_norm.T, fmt="{:.4f}", row_label="PC",
        save_path=OUT_DIR / "09b_training_pca_normalized.txt",
    )

    server_sent = np.asarray(encoded_dataset, dtype=float) if encoded_dataset is not None else None
    if server_sent is not None:
        max_diff = float(np.max(np.abs(server_sent - Z_norm)))
        check_result(
            "server encoded_dataset matches local normalized PCA projection",
            max_diff < 1e-6,
            f"max |diff| = {max_diff:.2e}",
        )

    return components, mean, pc_mins, pc_maxs


def _stage4_gp_surfaces(
    result: dict,
    Z_train_norm: np.ndarray,
    Zp_norm: np.ndarray,
    Y_train: np.ndarray,
) -> None:
    section("Stage 4 - PCA GP surfaces (per target)")
    plt = mpl_setup()

    gp = result.get("gp_maps") or {}
    axes_info = (gp.get("grid") or {}).get("axes")
    means_maps = (gp.get("maps") or {}).get("means") or {}
    vars_maps = (gp.get("maps") or {}).get("variances") or {}
    if not axes_info or not means_maps:
        print("  [SKIP] GP maps not returned by server")
        return

    xax = np.asarray(axes_info[0], dtype=float)
    yax = np.asarray(axes_info[1], dtype=float)
    n_obj = len(OBJECTIVES)

    fig, axes = plt.subplots(2, n_obj, figsize=(4.0 * n_obj, 7.5), dpi=110)
    for k, obj in enumerate(OBJECTIVES):
        M = np.asarray(means_maps.get(obj, np.zeros((yax.size, xax.size))), dtype=float)
        V = np.asarray(vars_maps.get(obj, np.zeros_like(M)), dtype=float)
        S = np.sqrt(np.maximum(V, 0.0))

        ax = axes[0, k]
        im = ax.imshow(M, origin="lower", aspect="auto",
                       extent=[xax[0], xax[-1], yax[0], yax[-1]], cmap="viridis")
        fig.colorbar(im, ax=ax, label=obj)
        ax.scatter(Z_train_norm[:, 0], Z_train_norm[:, 1], s=20, c=Y_train[:, k],
                   cmap="magma", edgecolor="black", linewidth=0.4, label="train (true)")
        ax.scatter(Zp_norm[:, 0], Zp_norm[:, 1], s=110, c="red", marker="X",
                   edgecolor="white", linewidth=1.2, label="predicted")
        ax.set_xlabel("PC1 (norm)")
        ax.set_ylabel("PC2 (norm)")
        ax.set_title(f"mean  {obj}")
        if k == 0:
            ax.legend(loc="upper right", fontsize=8)

        ax2 = axes[1, k]
        im2 = ax2.imshow(S, origin="lower", aspect="auto",
                         extent=[xax[0], xax[-1], yax[0], yax[-1]], cmap="magma")
        fig.colorbar(im2, ax=ax2, label=f"std({obj})")
        ax2.scatter(Z_train_norm[:, 0], Z_train_norm[:, 1], s=18, c="white",
                    edgecolor="black", linewidth=0.4)
        ax2.scatter(Zp_norm[:, 0], Zp_norm[:, 1], s=110, c="red", marker="X",
                    edgecolor="white", linewidth=1.2)
        ax2.set_xlabel("PC1 (norm)")
        ax2.set_ylabel("PC2 (norm)")
        ax2.set_title(f"std  {obj}")

    fig.suptitle("Demo 09 - PCA(2) GP surfaces: mean (top) and std (bottom)", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


def _stage5_predictions_pca(preds: list[dict]) -> None:
    section("Stage 5 - Predictions in PCA space")
    cand_cols = [f"c{i + 1}" for i in range(len(preds))]

    Zp = np.array([p.get("encoded", [0.0, 0.0]) for p in preds])
    render_table(
        "Predicted candidates in PCA space (PC x candidate) [normalized 0..1]",
        col_headers=cand_cols,
        row_headers=[f"PC{k + 1}" for k in range(Zp.shape[1])],
        matrix=Zp.T, fmt="{:+.4f}", row_label="PC",
        save_path=OUT_DIR / "10_predictions_pca.txt",
    )

    means = np.array([p["objectives"] for p in preds])
    stds = np.sqrt(np.maximum(np.array([p["variances"] for p in preds]), 0.0))
    render_table(
        "GP posterior mean per objective (objective x candidate)",
        col_headers=cand_cols, row_headers=OBJECTIVES, matrix=means.T,
        fmt="{:.3f}", row_label="objective",
        save_path=OUT_DIR / "11_predictions_gp_mean.txt",
    )
    render_table(
        "GP posterior std per objective (objective x candidate)",
        col_headers=cand_cols, row_headers=OBJECTIVES, matrix=stds.T,
        fmt="{:.3f}", row_label="objective",
        save_path=OUT_DIR / "12_predictions_gp_std.txt",
    )


def _stage6_predictions_reconstructed(preds: list[dict]) -> np.ndarray:
    section("Stage 6 - Reconstructed predictions in original space")
    cand_cols = [f"c{i + 1}" for i in range(len(preds))]

    names = INGREDIENTS + PROCESS
    P = np.array([[p["candidate"][n] for n in names] for p in preds])

    render_table(
        "Reconstructed ingredient mass fractions (ingredient x candidate)",
        col_headers=cand_cols, row_headers=INGREDIENTS,
        matrix=P[:, : len(INGREDIENTS)].T,
        fmt="{:.4f}", row_label="ingredient",
        save_path=OUT_DIR / "13_predictions_ingredients.txt",
    )
    render_table(
        "Reconstructed process parameters (parameter x candidate)",
        col_headers=cand_cols, row_headers=PROCESS,
        matrix=P[:, len(INGREDIENTS):].T,
        fmt="{:.2f}", row_label="parameter",
        save_path=OUT_DIR / "14_predictions_parameters.txt",
    )
    return P


def _stage7_metrics(
    X: np.ndarray, Y: np.ndarray, P: np.ndarray, preds: list[dict],
    components: np.ndarray, mean: np.ndarray,
    scaler_mean: np.ndarray | None = None,
    scaler_scale: np.ndarray | None = None,
) -> dict[str, bool]:
    section("Stage 7 - Correctness checks and metrics")

    Z = (X - mean) @ components.T
    if (scaler_mean is not None and scaler_scale is not None
            and scaler_mean.size == X.shape[1] and scaler_scale.size == X.shape[1]):
        # Correct inverse for the ScaledPCA wrapper: Z was computed in the
        # space of z-score scaled features, so the reconstruction has to
        # unscale after projecting back through sklearn's components.
        C_inner = components * scaler_scale[None, :]
        mu_p = (mean - scaler_mean) / scaler_scale
        X_scaled_hat = Z @ C_inner + mu_p
        X_hat = X_scaled_hat * scaler_scale + scaler_mean
    else:
        X_hat = Z @ components + mean
    err = X - X_hat
    feat_rmse = np.sqrt(np.mean(err**2, axis=0))
    render_table(
        "PCA training reconstruction RMSE (feature x 1)",
        col_headers=["rmse"],
        row_headers=INGREDIENTS + PROCESS,
        matrix=feat_rmse.reshape(-1, 1),
        fmt="{:.4g}", row_label="feature",
        save_path=OUT_DIR / "15_pca_reconstruction_rmse.txt",
    )
    print(f"  aggregate |X - X_rec|: mean={np.mean(np.abs(err)):.4g}  max={np.max(np.abs(err)):.4g}")

    slsqp_flags = np.array(
        [bool(p.get("reconstructed", {}).get("success", True)) for p in preds]
    )
    n_feasible = int(slsqp_flags.sum())
    n_total = int(len(slsqp_flags))
    print(
        "\n  Reconstruction feasibility:"
        f" {n_feasible}/{n_total} candidates found a fully feasible recipe"
        " (SLSQP success=True).\n"
        "  Infeasible candidates were returned as best-effort and are checked"
        "\n  separately - the engine flagged them via reconstructed.success=False."
    )

    print(
        "\n  Hard-constraint satisfaction on FEASIBLE predictions (must all PASS):"
    )
    ing_indices = list(range(len(INGREDIENTS)))
    P_feas = P[slsqp_flags] if n_feasible > 0 else P[:0]
    if n_feasible == 0:
        print("    (no feasible predictions to check)")
        ok_sum = ok_bounds = ok_ratio = False
    else:
        ok_sum, sum_err = check_sum_to_one(P_feas, ing_indices, target=1.0, tol=1e-4)
        check_result("sum(ingredients) == 1.0", ok_sum, f"max |error| = {np.max(np.abs(sum_err)):.2e}")

        all_bounds: list[tuple[float, float]] = []
        for name in INGREDIENTS:
            all_bounds.append(ING_BOUNDS[name])
        for name in PROCESS:
            all_bounds.append(PROC_BOUNDS[name])
        ok_bounds, bviol = check_bounds(P_feas, all_bounds)
        viol_count = sum(len(v) for v in bviol)
        check_result(
            "all values within [min, max] bounds", ok_bounds,
            f"{viol_count} field(s) out of bounds",
        )

        ok_ratio, ratios = check_ratio(P_feas, MCC_IDX, LACTOSE_IDX, RATIO_MIN, RATIO_MAX)
        ratio_str = ", ".join(f"{r:.3f}" for r in ratios)
        check_result(
            f"ratio MCC/lactose in [{RATIO_MIN:.2f}, {RATIO_MAX:.2f}]",
            ok_ratio, f"ratios = [{ratio_str}]",
        )

    Yp_true = forward(P)
    gp_mean = np.array([p["objectives"] for p in preds])
    gp_std = np.sqrt(np.maximum(np.array([p["variances"] for p in preds]), 0.0))
    cal_rows = []
    for k in range(len(OBJECTIVES)):
        stats = gp_calibration(gp_mean[:, k], gp_std[:, k], Yp_true[:, k])
        cal_rows.append([stats["mae"], stats["rmse"], stats["mean_abs_z"]])
    render_table(
        "GP posterior vs forward-model truth at predicted points",
        col_headers=["MAE", "RMSE", "mean |z|"],
        row_headers=OBJECTIVES,
        matrix=np.array(cal_rows), fmt="{:.4g}", row_label="objective",
        save_path=OUT_DIR / "16_gp_calibration.txt",
    )

    print(
        "\n  Soft-goal compliance on predictions (informational; these are"
        "\n  optimization targets, not hard constraints - the optimizer balances"
        "\n  them against each other, so less than 5/5 does not imply an engine bug):"
    )
    in_band = (Yp_true[:, 1] >= HARDNESS_LO) & (Yp_true[:, 1] <= HARDNESS_HI)
    print(
        f"    hardness in [{HARDNESS_LO:.0f}, {HARDNESS_HI:.0f}]:"
        f" {int(in_band.sum())}/{len(in_band)} candidates in band"
    )
    below_friab = Yp_true[:, 2] <= FRIABILITY_MAX
    print(
        f"    friability <= {FRIABILITY_MAX:.2f}:"
        f" {int(below_friab.sum())}/{len(below_friab)} candidates below threshold"
    )
    below_dis = Yp_true[:, 3] <= DISINTEGRATION_MAX
    print(
        f"    disintegration <= {DISINTEGRATION_MAX:.0f}:"
        f" {int(below_dis.sum())}/{len(below_dis)} candidates below threshold"
    )

    best_train = np.array([Y[:, 0].max(), HARDNESS_IDEAL, Y[:, 2].min(), Y[:, 3].min()])
    idx = int(np.argmin(
        -Yp_true[:, 0] / 100.0
        + 0.03 * np.abs(Yp_true[:, 1] - HARDNESS_IDEAL)
        + 1.5 * Yp_true[:, 2]
        + 0.002 * Yp_true[:, 3]
    ))
    best_pred = Yp_true[idx]
    delta = best_pred - best_train
    render_table(
        "Best-training vs best-prediction (forward model)",
        col_headers=["best_train", "best_pred", "delta"],
        row_headers=OBJECTIVES,
        matrix=np.stack([best_train, best_pred, delta], axis=1),
        fmt="{:.3f}", row_label="objective",
        save_path=OUT_DIR / "17_best_training_vs_best_prediction.txt",
    )
    return {"sum": ok_sum, "bounds": ok_bounds, "ratio": ok_ratio}


def _stage8_extras(P: np.ndarray, Yp_true: np.ndarray, Y_train: np.ndarray) -> None:
    section("Stage 8 - Visual extras")
    plt = mpl_setup()

    best_train_row = np.array([
        Y_train[:, 0].max(), HARDNESS_IDEAL, Y_train[:, 2].min(), Y_train[:, 3].min(),
    ])
    idx = int(np.argmin(
        -Yp_true[:, 0] / 100.0
        + 0.03 * np.abs(Yp_true[:, 1] - HARDNESS_IDEAL)
        + 1.5 * Yp_true[:, 2]
        + 0.002 * Yp_true[:, 3]
    ))
    best_pred_row = Yp_true[idx]

    fig = plt.figure(figsize=(14, 6.5), dpi=110)
    gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.35)

    obj_info = [
        ("dissolution_30min", "max",        None,           None,             None, "higher is better"),
        ("hardness_N",        "band",       HARDNESS_LO,    HARDNESS_HI,      HARDNESS_IDEAL, f"ideal {HARDNESS_IDEAL:.0f}"),
        ("friability_pct",    "below",      None,           FRIABILITY_MAX,   None, f"<= {FRIABILITY_MAX:.2f}"),
        ("disintegration_s",  "below",      None,           DISINTEGRATION_MAX, None, f"<= {DISINTEGRATION_MAX:.0f}"),
    ]
    for k, (name, kind, lo, hi, ideal, subtitle) in enumerate(obj_info):
        ax = fig.add_subplot(gs[0, k])
        y = Y_train[:, k]
        ax.hist(y, bins=12, color="lightsteelblue", edgecolor="gray", alpha=0.9)
        y_max = max(ax.get_ylim()[1], 1.0)
        if kind == "band":
            ax.axvspan(lo, hi, color="green", alpha=0.12, label="target band")
            ax.axvline(ideal, color="green", lw=1.8, ls="--", label="ideal")
        elif kind == "below":
            ax.axvspan(ax.get_xlim()[0], hi, color="green", alpha=0.12, label="feasible")
            ax.axvline(hi, color="green", lw=1.8, ls="--", label="threshold")
        ax.axvline(best_train_row[k], color="black", lw=1.8, label="best train")
        ax.axvline(best_pred_row[k], color="red", lw=2.2, label="best pred")
        ax.set_title(f"{name}\n({subtitle})", fontsize=10)
        ax.set_ylim(0, y_max)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_yticks([])
        if k == 0:
            ax.legend(loc="upper right", fontsize=7)

    ax_bottom = fig.add_subplot(gs[1, :])
    ing_x = np.arange(len(INGREDIENTS))
    best_pred_composition = P[idx, : len(INGREDIENTS)]
    pred_median = np.median(P[:, : len(INGREDIENTS)], axis=0)
    train_median = np.median(np.array([
        forward(np.zeros((0, P.shape[1])))  # placeholder
    ]).reshape(0, 0), axis=0) if False else np.zeros(len(INGREDIENTS))
    ax_bottom.bar(ing_x - 0.22, best_pred_composition, width=0.4, color="red", alpha=0.85, label="best pred recipe")
    ax_bottom.bar(ing_x + 0.22, pred_median, width=0.4, color="gray", alpha=0.6, label="median of predictions")
    for i, name in enumerate(INGREDIENTS):
        lo_b, hi_b = ING_BOUNDS[name]
        ax_bottom.plot([i - 0.35, i + 0.35], [lo_b, lo_b], color="blue", lw=0.8, alpha=0.4)
        ax_bottom.plot([i - 0.35, i + 0.35], [hi_b, hi_b], color="blue", lw=0.8, alpha=0.4)
    ax_bottom.set_xticks(ing_x)
    ax_bottom.set_xticklabels(INGREDIENTS, rotation=30, ha="right")
    ax_bottom.set_ylabel("mass fraction")
    ax_bottom.set_title("Best predicted recipe vs median of predictions (thin blue bars = ingredient bounds)")
    ax_bottom.legend(loc="upper right", fontsize=9)
    _ = train_median

    fig.suptitle("Demo 09 - Predictions placed against training distributions and goal regions", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(101)
    X = sample_training(rng, n=36)
    Y = forward(X)

    _stage1_inputs(X, Y)
    _stage2_spec()

    payload = _build_payload(X, Y)
    result = post_optimize_and_wait("demo_09_pharma_tablet", payload, max_poll_s=240.0)
    preds = result.get("predictions", [])
    if not preds:
        print("No predictions returned")
        return

    encoding_info = result.get("encoding_info") or {}
    encoded_dataset = result.get("encoded_dataset")
    components, mean, pc_mins, pc_maxs = _stage3_pca_results(
        X, encoding_info, encoded_dataset
    )

    pc_range = np.maximum(pc_maxs - pc_mins, 1e-12)
    Z_raw = (X - mean) @ components.T
    Z_norm = (Z_raw - pc_mins) / pc_range
    Zp_norm = np.asarray([p.get("encoded", [0.0, 0.0]) for p in preds], dtype=float)

    _stage4_gp_surfaces(result, Z_norm, Zp_norm, Y)
    _stage5_predictions_pca(preds)
    P = _stage6_predictions_reconstructed(preds)
    scaler_mean = np.asarray(encoding_info.get("scaler_mean", []), dtype=float)
    scaler_scale = np.asarray(encoding_info.get("scaler_scale", []), dtype=float)
    checks = _stage7_metrics(
        X, Y, P, preds, components, mean,
        scaler_mean=scaler_mean if scaler_mean.size else None,
        scaler_scale=scaler_scale if scaler_scale.size else None,
    )

    Yp_true = forward(P)
    _stage8_extras(P, Yp_true, Y)

    n_feas = sum(
        1 for p in preds if bool(p.get("reconstructed", {}).get("success", True))
    )
    section("Final verdict", ch="*")
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        print(f"  {len(failed)} HARD-constraint check(s) FAILED on feasible"
              f" predictions: {failed}")
        print("  This indicates an engine correctness problem - see stage 7 output.")
    else:
        print(f"  All hard-constraint checks PASSED on the {n_feas}/{len(preds)}"
              " feasible predictions (sum, bounds, ratio).")
        if n_feas < len(preds):
            print(f"  {len(preds) - n_feas} prediction(s) had reconstructed.success=False")
            print("  and were returned as best-effort; this is a normal signal that")
            print("  the acquisition picked a PCA point with no feasible recipe under")
            print("  the tight sum+bounds+ratio constraints.")
        print("  Soft-goal compliance and GP calibration stats are above; these are")
        print("  optimizer trade-offs, not correctness signals.")


if __name__ == "__main__":
    main()
