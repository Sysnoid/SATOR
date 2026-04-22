---
title: Examples
sidebar_position: 13
slug: /examples
---

# 13. Examples

The `examples/` directory ships ten runnable Python demos that exercise
every major feature of SATOR: single- and multi-objective optimization,
mixed goal types (`min`, `max`, `target`, `within_range`,
`minimize_below`, `maximize_above`), sum and ratio constraints, PCA
dimensionality reduction, SLSQP reconstruction, and GP surface maps.

Two of them — **demo 09** (pharmaceutical tablet, PCA path) and
**demo 10** (cosmetic O/W emulsion, non-PCA path) — are
**audit-style** scripts: their stdout and saved artifacts are designed
to be verified by a human reading this page. Real rendered figures
from their latest run are embedded below so you can see what a clean,
successful run looks like before you even install the repo.

For the short "how to launch the server and run a script" recipe, see
[`examples/how_to_run_examples.md`](../examples/how_to_run_examples.md).

---

## 13.1 Demo index

| # | File | Path | What it demonstrates |
|---|---|---|---|
| 01 | `demo_01_rosenbrock_single_min.py` | [link](../examples/demo_01_rosenbrock_single_min.py) | Single-objective `min` on Rosenbrock (qEI). |
| 02 | `demo_02_ackley_target.py`         | [link](../examples/demo_02_ackley_target.py)         | `target` goal on Ackley — hit a specific f-value. |
| 03 | `demo_03_himmelblau_within_range.py` | [link](../examples/demo_03_himmelblau_within_range.py) | `within_range` goal on Himmelblau. |
| 04 | `demo_04_zdt1_pareto.py`           | [link](../examples/demo_04_zdt1_pareto.py)           | Multi-objective Pareto front (qNEHVI) vs. analytic ZDT1. |
| 05 | `demo_05_mixture_sum_constraint.py`| [link](../examples/demo_05_mixture_sum_constraint.py)| 3-ingredient mixture with sum-to-one. |
| 06 | `demo_06_ratio_constraints.py`     | [link](../examples/demo_06_ratio_constraints.py)     | Ratio constraint `0.5 ≤ A/B ≤ 2.0`. |
| 07 | `demo_07_paint_formulation.py`     | [link](../examples/demo_07_paint_formulation.py)     | Realistic paint blend — `min` / `max` / `within_range` + sum-to-one. |
| 08 | `demo_08_ev_electrolyte_target.py` | [link](../examples/demo_08_ev_electrolyte_target.py) | EV electrolyte — `target` + `within_range` + `maximize_above` + sum-to-one. |
| 09 | `demo_09_pharma_tablet_pca.py`     | [link](../examples/demo_09_pharma_tablet_pca.py)     | **PCA flagship.** 7 excipients + 2 process params, 4 mixed-goal objectives, sum-to-one, `MCC/lactose` ratio, PCA(2) + GP surfaces + SLSQP reconstruction. 8-stage audit. |
| 10 | `demo_10_cosmetic_emulsion.py`     | [link](../examples/demo_10_cosmetic_emulsion.py)     | **Non-PCA flagship.** 10 ingredients + 2 process params, `within_range` / `max` / `max` / `min` objectives, sum-to-one, `cetearyl/PEG` ratio. 7-stage audit. |

Run them in any order. Demos 09 and 10 are the recommended reference
examples for new users because they print a self-verifying audit trail
and exercise the most important engine features
(PCA round-trip, reconstruction, multi-objective Bayesian
optimization, mixed goal types, sum / bound / ratio constraints).

---

## 13.2 Demo 09 — Pharmaceutical tablet (PCA flagship)

### 13.2.1 What it does

A pharmaceutical immediate-release (IR) tablet is formulated from 7
excipients (`API`, `lactose`, `MCC`, `croscarmellose`, `PVP_K30`,
`Mg_stearate`, `silica`) with two process parameters (compaction
force in kN, blend time in minutes). Four mixed-goal objectives:

- `dissolution_30min` — **maximize** (higher is better)
- `hardness_N`         — **within_range** `[80, 150]`, ideal `115`
- `friability_pct`     — **minimize_below** threshold `1.0`
- `disintegration_s`   — **minimize_below** threshold `900`

Hard constraints under test:

- `sum(mass fractions) = 1.0`
- per-parameter bounds (7 ingredient + 2 process variables)
- `MCC / lactose` in `[0.55, 0.80]`  (ratio regression check)

The training set deliberately **violates** the tight `MCC/lactose`
window on several rows so the audit can demonstrate the engine
enforcing the ratio during SLSQP reconstruction.

### 13.2.2 Pipeline

1. Fit a `ScaledPCA(2)` (`StandardScaler` followed by `sklearn.PCA`)
   on the training inputs. See [§7 Optimization pipeline](07-optimization-pipeline.md)
   and [§9 Reconstruction](09-reconstruction.md) for the full story.
2. Fit a `SingleTaskGP` per objective in normalized PCA coordinates.
3. Run the acquisition (`qei` with ParEGO scalarization) in PCA space.
4. Inverse-transform candidates back through `ScaledPCA`.
5. Repair the result with the SLSQP reconstructor so sum-to-one,
   bounds, and the `MCC/lactose` ratio all hold.

### 13.2.3 8-stage audit output

The script prints — and also saves to
`examples/responses/demo_09/NN_*.txt` — the following stages:

1. Training ingredient, parameter, and objective tables
2. Objectives spec + hard-constraint spec + engine config
3. PCA explained variance, loadings (on scaled features), training
   projections (raw and normalized)
4. Per-objective GP surface on the PCA(2) plane (mean + std)
5. Predicted candidates in PCA space
6. Reconstructed candidates in the original ingredient + process space
7. Metrics: reconstruction RMSE per feature, GP calibration
   (MAE / RMSE / mean `|z|`), best-train vs best-pred
8. Visual extras: objective histograms and a best-recipe bar chart

### 13.2.4 Figure 1 — PCA(2) GP surfaces (mean on top, std on bottom)

![demo_09 GP surfaces](assets/examples/demo_09_gp_surfaces.png)

The top row shows the GP posterior **mean** per objective on the
PCA(2) plane. The dots are training formulations (encoded into PCA
space) and the red crosses are the predicted candidates. Because each
objective has its own GP, the same PCA coordinates map to different
heights on different surfaces — that is the multi-objective trade-off
made visible.

The bottom row shows the GP **standard deviation**. Dark regions are
well-explored (low uncertainty); bright regions are where the GP
extrapolates. Predictions that land in bright regions are inherently
more speculative.

### 13.2.5 Figure 2 — Objective distributions + best recipe

![demo_09 histograms and best recipe](assets/examples/demo_09_histograms_recipe.png)

Top: one histogram per objective showing the training distribution,
the goal band or threshold (green shading), the best training value
(black line) and the best predicted value (red line). These four
subplots let you check at a glance whether the optimizer is pushing
*toward* the target for each objective.

Bottom: the best-predicted recipe in red against the median of all
predictions in grey, with thin blue bars marking each ingredient's
declared bounds. This is the concrete answer to "what should we
mix?".

### 13.2.6 Expected verdict

At the bottom of the audit stdout you should see:

```
  All hard-constraint checks PASSED on the 5/5 feasible predictions (sum, bounds, ratio).
```

Soft-goal compliance (how many predictions fall inside the `hardness`
band, below the `friability` threshold, etc.) is reported as an
informational metric — it is **not** a hard constraint and the
optimizer is free to trade one against another.

### 13.2.7 Files saved

- `examples/responses/demo_09_pharma_tablet_request.json`
- `examples/responses/demo_09_pharma_tablet_result.json`
- `examples/responses/demo_09/NN_*.txt` (stage-by-stage tables)
- `examples/responses/demo_09_pharma_tablet_pca_fig_*.png`

---

## 13.3 Demo 10 — Cosmetic O/W emulsion (non-PCA flagship)

### 13.3.1 What it does

A cosmetic oil-in-water face cream is formulated from 10 ingredients
(`water`, `glycerin`, `propanediol`, `squalane`, `jojoba_oil`,
`cetearyl_alcohol`, `PEG100_stearate`, `niacinamide`,
`hyaluronic_acid`, `preservative`) plus two process parameters
(`mix_temp_C`, `homog_speed_rpm`). Four mixed-goal objectives:

- `viscosity_cP`    — **within_range** `[15000, 30000]`, ideal `22500`
- `spreadability`   — **maximize**
- `stability_days`  — **maximize**
- `cost_per_kg`     — **minimize**

Hard constraints under test:

- `sum(mass fractions) = 1.0`
- per-parameter bounds (10 ingredient + 2 process variables)
- `cetearyl_alcohol / PEG100_stearate` in `[1.5, 3.0]`

Unlike demo 09 this demo runs **without PCA**: the GP is built
directly in the 12-dimensional ingredient + process space, so the
acquisition, the feasibility filter, and the ratio-enforcement logic
are all tested on the native input space.

### 13.3.2 Pipeline

1. Fit `SingleTaskGP` per objective directly on the 12-D training
   inputs (min/max `Normalize` input transform, `Standardize` outcome
   transform).
2. Because at least one objective is `within_range`, the acquisition
   takes the Sobol-plus-scoring branch — see
   [§7.5 Acquisition](07-optimization-pipeline.md) for details.
3. Project the Sobol grid onto sum-to-one, filter by the feasibility
   mask (sum, bounds, ratio), and rank by score.
4. On the top-k selection apply `sum → ratio-repair → sum` so every
   returned candidate strictly satisfies every hard constraint.

### 13.3.3 7-stage audit output

1. Training ingredient, parameter, and objective tables
2. Objectives spec + hard-constraint spec + engine config
3. Predicted candidate recipes (named ingredient + process columns)
4. GP posterior mean and std per objective at each candidate
5. Hard-constraint checks: sum, bounds, ratio (`[PASS]` lines)
6. GP calibration: MAE / RMSE / mean `|z|` vs a known forward model,
   plus soft-goal compliance and best-train vs best-pred
7. Visual summary

### 13.3.4 Figure 1 — Objectives + phase composition

![demo_10 summary figure](assets/examples/demo_10_summary.png)

Top row: one histogram per objective showing the training
distribution, the target band (green for `within_range`, thin red
lines for `max`/`min`) and the best predicted value (red line).

Bottom: the top-3 predicted recipes rolled up into **phase groups**
(`aqueous`, `oil`, `emulsifier`, `active`, `functional`) so you can
see at a glance what kind of emulsion was proposed.

### 13.3.5 Expected verdict

```
  All hard-constraint checks PASSED on 5 predictions
  (sum, bounds, ratio). Non-PCA path is correct.
```

Typical GP-calibration numbers on this demo (with the min/max input
transform enabled): `viscosity_cP` RMSE ≈ 250 on a ~50 000 range,
`mean |z| < 1` across all four objectives — a good sanity check that
the GP is neither collapsed nor wildly overconfident.

### 13.3.6 Files saved

- `examples/responses/demo_10_cosmetic_emulsion_request.json`
- `examples/responses/demo_10_cosmetic_emulsion_result.json`
- `examples/responses/demo_10/NN_*.txt` (stage-by-stage tables)
- `examples/responses/demo_10_cosmetic_emulsion_fig_1_1.png`

---

## 13.4 Short tour of demos 01–08

The simpler demos each exercise one feature at a time and make a good
testbed for a new acquisition function, goal type, or constraint
kind.

- **demo_01 — Rosenbrock.** Classic 2-D minimization; confirms the
  single-objective `qei` path converges toward the global minimum at
  `(1, 1)`.
- **demo_02 — Ackley `target`.** Single objective with `goal=target`
  and `target_value`; confirms the engine steers toward a specific
  f-value rather than the global min or max.
- **demo_03 — Himmelblau `within_range`.** Exercises the
  `within_range` goal on a function with four symmetric minima.
- **demo_04 — ZDT1 Pareto.** Multi-objective benchmark with an
  analytic Pareto front; overlays the predicted candidates on the
  true front so you can see their spread.
- **demo_05 — Mixture sum.** 3-ingredient mixture; the simplest case
  of a `sum_constraints` entry with `target_sum = 1.0`.
- **demo_06 — Ratio.** 2-ingredient mixture with
  `ratio_constraints` (`0.5 ≤ A/B ≤ 2.0`); the minimal ratio test.
- **demo_07 — Paint.** Realistic paint blend combining
  `min` / `max` / `within_range` goals with sum-to-one.
- **demo_08 — EV electrolyte.** `target` + `within_range` +
  `maximize_above` with sum-to-one — the most goal-varied of the
  mid-size demos.

Each of these writes a small `*_request.json` / `*_result.json` pair
under `examples/responses/` the first time it runs.

---

## 13.5 Notes on interpretation

- GP surfaces are models learned from your data; they are not ground
  truth. Peaks and valleys indicate the model's belief about where
  the objective is high or low.
- In multi-objective mode the same encoded candidate is shown on all
  objective surfaces. Its z-value differs per surface because each
  objective has its own GP.
- Reconstruction (demo 09) ensures suggested formulations are valid
  in the original space — ingredients sum exactly to one and any
  declared ratio windows hold.
- "Soft-goal compliance" in demos 09 and 10 is **not** a correctness
  signal: a candidate that lands outside a `within_range` band is a
  trade-off, not a bug. The hard-constraint checks (sum, bounds,
  ratio) are the actual correctness gates.

## 13.6 Related chapters

- [§5 API reference](05-api-reference.md) — the JSON payload shapes
  the demos submit.
- [§6 Objectives & constraints](06-objectives-and-constraints.md) —
  the goal types and constraint kinds demonstrated by the demos.
- [§7 Optimization pipeline](07-optimization-pipeline.md) — the
  preprocessing → GP → acquisition → feasibility → maps →
  reconstruction flow that every demo drives.
- [§9 Reconstruction](09-reconstruction.md) — the SLSQP inverse
  mapping used by demo 09.
