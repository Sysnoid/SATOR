---
title: Objectives & Constraints
sidebar_position: 6
slug: /objectives-and-constraints
---

# 6. Objectives & Constraints

This chapter defines the goal types and constraint types that SATOR
understands, and explains how each one shapes the optimization.

---

## 6.1 Goal types

Every entry in `objectives` must declare a `goal`. The optimizer interprets
the goal when scoring candidates, either via a BoTorch acquisition function
(for `min` / `max`) or via Sobol-grid scoring (for all other goals).

SATOR splits goal types into two families:

- **Soft goals** shape the acquisition score. The optimizer *prefers*
  candidates that satisfy your intent but does **not** guarantee it.
- **Hard goals** (the `enforce_*` family) are evaluated against the GP
  posterior and used as a feasibility mask during candidate selection.
  Every returned prediction carries an `enforced_goals_satisfied` flag so
  the caller can tell at a glance whether the threshold held on the
  surrogate. Read §6.1.11 before using a threshold in production R&D.

| Goal | Family | Meaning | Extra fields |
|---|---|---|---|
| [`min`](#611-min)                             | soft | Minimize `f(x)`. | — |
| [`max`](#612-max)                             | soft | Maximize `f(x)`. | — |
| [`target`](#613-target)                       | soft | Drive `f(x)` toward a specific value `T`. | `target_value`, optional `target_tolerance` |
| [`minimize_below`](#614-minimize_below)       | soft | Prefer `f(x) ≤ T`, softly penalize above. | `threshold` |
| [`maximize_above`](#615-maximize_above)       | soft | Prefer `f(x) ≥ T`, softly penalize below. | `threshold` |
| [`maximize_below`](#616-maximize_below)       | soft | Maximize `f(x)` but stay below `T`. | `threshold` |
| [`minimize_above`](#617-minimize_above)       | soft | Minimize `f(x)` but stay above `T`. | `threshold` |
| [`within_range`](#618-within_range)           | soft | Keep `f(x) ∈ [A, B]`, preferably near `ideal`. | `range` |
| [`explore`](#619-explore) *(alias `probe`)*   | soft | Prefer high-variance regions. | — |
| [`improve`](#6110-improve)                    | soft | Prefer the largest expected improvement over the current best. | — |
| [`enforce_above`](#6111-enforce)              | **hard** | GP posterior **must satisfy** `f(x) ≥ T`. | `threshold_value` |
| [`enforce_below`](#6111-enforce)              | **hard** | GP posterior **must satisfy** `f(x) ≤ T`. | `threshold_value` |
| [`enforce_within_range`](#6111-enforce)       | **hard** | GP posterior **must satisfy** `f(x) ∈ [A, B]`. | `range` |

### 6.1.1 `min`

```json
"cost": { "goal": "min" }
```

Standard minimization. Participates in multi-objective hypervolume when every
other objective is also `min` or `max`.

### 6.1.2 `max`

```json
"yield": { "goal": "max" }
```

Standard maximization.

### 6.1.3 `target`

Drive `f(x)` as close as possible to a desired value `T`:

```json
"ph": {
  "goal": "target",
  "target_value": 7.0
}
```

Optional knobs (set on `optimization_config`):

- `target_tolerance` — values within `±target_tolerance` of `T` are treated
  as equally good; wider bands yield more robust candidates.
- `target_variance_penalty` — weight of the posterior-variance penalty when
  choosing among equally accurate candidates.

### 6.1.4 `minimize_below`

**Soft.** Prefer `f(x) ≤ T`; anything above is *softly penalized* in the
acquisition score. The optimizer may still return candidates whose GP mean
sits above `T` when nothing better is available. Use
[`enforce_below`](#6111-enforce) if the threshold is a requirement rather
than a preference.

```json
"impurity_ppm": {
  "goal": "minimize_below",
  "threshold": { "type": "<=", "value": 50, "weight": 1.0 }
}
```

### 6.1.5 `maximize_above`

**Soft.** Prefer `f(x) ≥ T`; anything below is *softly penalized*. Same
caveat as `minimize_below`: this is a preference, not a guarantee. Use
[`enforce_above`](#6111-enforce) if the threshold must hold on every
returned candidate.

```json
"tensile_strength": {
  "goal": "maximize_above",
  "threshold": { "type": ">=", "value": 500, "weight": 1.0 }
}
```

### 6.1.6 `maximize_below`

Maximize `f(x)` but never cross above `T`.

```json
"speed": {
  "goal": "maximize_below",
  "threshold": { "type": "<=", "value": 100 }
}
```

### 6.1.7 `minimize_above`

Minimize `f(x)` but never drop below `T`.

```json
"temperature": {
  "goal": "minimize_above",
  "threshold": { "type": ">=", "value": 25 }
}
```

### 6.1.8 `within_range`

Keep `f(x)` inside the inclusive interval `[min, max]`. Optionally prefer a
sweet-spot `ideal`.

```json
"viscosity": {
  "goal": "within_range",
  "range": {
    "min": 1000, "max": 1500,
    "ideal": 1250,
    "weight": 1.0, "ideal_weight": 0.5
  }
}
```

### 6.1.9 `explore`

Prefer candidates with **high posterior variance**. Useful in early rounds
to learn the landscape before exploiting it.

### 6.1.10 `improve`

Prefer candidates with the **largest expected improvement** over the current
best. A greedy, exploitation-heavy choice.

### 6.1.11 `enforce_above` / `enforce_below` / `enforce_within_range`

**Hard.** These goal types turn a soft scoring term into a real feasibility
constraint on the GP posterior. They should be your default whenever a
threshold represents a *requirement* (safety floor, spec ceiling, in-spec
band) rather than a preference.

**How they work:**

1. On the Sobol scoring grid, the GP posterior mean `μ` is compared to the
   configured threshold. Rows that violate the threshold are masked out of
   the candidate pool before the acquisition score is ranked. If *no* row is
   feasible the mask is relaxed so a batch can still be returned — the
   violating candidates are then flagged downstream so the caller can see it.
2. After selection, every returned prediction carries:

   ```json
   {
     "enforced_goals_satisfied": true,
     "enforced_violations": []
   }
   ```

   Violating rows list the offending objectives, e.g.
   `["stability<4.2", "friability>1"]`.
3. The top-level `diagnostics.enforcement` block summarises the result:

   ```json
   "enforcement": {
     "enabled": true,
     "uncertainty_margin": 0.0,
     "n_total": 4,
     "n_satisfied": 3,
     "all_infeasible": false,
     "per_objective_violations": { "stability": 1 },
     "goals": [
       { "objective": "stability", "kind": "above", "lo": 4.2, "hi": null }
     ]
   }
   ```

**Schema:**

```json
"stability":  { "goal": "enforce_above",        "threshold_value": 4.2 },
"friability": { "goal": "enforce_below",        "threshold_value": 1.0 },
"hardness":   { "goal": "enforce_within_range", "range": { "min": 80, "max": 150 } }
```

**Uncertainty margin (optional, safer for R&D):**

By default, enforcement runs on the posterior **mean**. To demand that a
confidence bound satisfy the threshold instead, set

```json
"optimization_config": {
  "enforcement_uncertainty_margin": 2.0
}
```

With `margin = k`, `enforce_above` demands `μ − k·σ ≥ T` (the LCB must
clear the floor), `enforce_below` demands `μ + k·σ ≤ T`, and
`enforce_within_range` demands both bounds lie inside the window. `k = 2`
corresponds to roughly 95% confidence; use `0.0` (the default) when the
training data is already densely sampled near the threshold.

**When to pick soft vs. hard:**

| You want… | Use |
|---|---|
| A gentle nudge toward a target band, but candidates outside it are still OK. | `within_range`, `minimize_below`, `maximize_above` (soft). |
| A hard threshold that the surrogate must satisfy before the candidate even counts. | `enforce_*` (hard). |
| A hard *input-space* constraint (sum, ratio, bounds). | `sum_constraints` / `ratio_constraints` / parameter bounds — not an objective. |

---

## 6.2 Combining multiple objectives

SATOR handles trade-offs in two ways:

| Strategy | When to use | How to request it |
|---|---|---|
| **Pareto (default)** | You want the full set of non-dominated trade-offs. | `acquisition: "qnehvi"` or `"qehvi"` with `min` / `max` goals only. |
| **Scalarization (ParEGO)** | You want a single well-balanced compromise quickly. | `acquisition: "parego"`, optionally supply per-objective `weights`. |
| **Goal-shaped scoring** | Any objective uses an advanced goal. | Automatic; `acquisition` is ignored and the Sobol scoring path runs. |

When you mix a target-type objective with a `min` / `max` objective, SATOR
automatically falls back to the goal-shaped scoring path.

---

## 6.3 Input constraints

Input constraints are specified in `optimization_config`. They are enforced
during candidate selection and again post-hoc to guarantee feasibility.

### 6.3.1 Bounds

Every parameter already has `min` / `max` bounds defined in
`search_space.parameters`. No extra constraint is needed.

### 6.3.2 Sum constraints

Force a subset of variables to sum to a target:

```json
"sum_constraints": [
  { "indices": [0, 1, 2], "target_sum": 1.0 }
]
```

- `indices` — 0-based column positions into the input vector.
- `target_sum` — usually `1.0` for mixtures.

Multiple sum groups are allowed (one per disjoint set of indices).

### 6.3.3 Ratio constraints

Bound the ratio `x_i / x_j` between two variables:

```json
"ratio_constraints": [
  { "i": 0, "j": 1, "min_ratio": 0.5, "max_ratio": 2.0 }
]
```

Either `min_ratio` or `max_ratio` may be omitted if only one side is needed.

### 6.3.4 Enforcement details

- **In input space**, sum and ratio constraints become linear inequality
  constraints passed directly to BoTorch’s `optimize_acqf`.
- **In PCA space**, the linear constraints no longer align with model indices,
  so they are applied post-hoc via a projection-to-feasibility routine
  (`enforce_sum_constraints_np`) and a `feasible_mask` filter.
- Reconstruction (§9) applies the same constraints one more time via SLSQP
  to guarantee the returned point is *exactly* feasible.

---

## 6.4 Objectives versus constraints

If a quantity is **hard**, model it with the strictest mechanism available:

| Requirement | Model as |
|---|---|
| “pH must land near 7.0, but within `[6.8, 7.2]` is also fine.” | Soft `within_range` (or `target`) objective. |
| “pH must end up in `[6.8, 7.2]` on the predicted surface.” | **Hard** `enforce_within_range` objective. |
| “Impurity must not exceed 50 ppm.” | **Hard** `enforce_below` objective. |
| “Total moles must equal 1.” | `sum_constraints`, not an objective. |
| “Ratio of `A` to `B` between `0.5` and `2`.” | `ratio_constraints`, not an objective. |

Soft goals are things the optimizer should **trade off**; hard goals
(`enforce_*`) and input constraints are things it **must not violate** on
the surrogate. Pick the right one up front — changing your mind half-way
through a study usually means throwing away samples.

---

## 6.5 Normalization & scaling

- `parameter_scaling` — input preprocessing (`none` / `standardize` / `minmax`).
  Improves GP numerical conditioning, especially when parameters span very
  different magnitudes.
- `value_normalization` — output preprocessing of `Y`. Standardized `Y` makes
  targets and thresholds easier to reason about; SATOR automatically converts
  user-supplied `target_value` and `threshold.value` into the normalized
  space before scoring.

---

## 6.6 PCA

When `use_pca=true`, SATOR fits PCA on the training inputs, optimizes in the
normalized PCA space, and uses SLSQP reconstruction (§9) to map proposals back
to the original variables. Maps (`return_maps`) can be generated directly in
PCA space for 2-D / 3-D visualizations, regardless of the original input
dimensionality.
