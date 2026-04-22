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

| Goal | Meaning | Extra fields |
|---|---|---|
| [`min`](#611-min)                       | Minimize `f(x)`. | — |
| [`max`](#612-max)                       | Maximize `f(x)`. | — |
| [`target`](#613-target)                 | Drive `f(x)` toward a specific value `T`. | `target_value`, optional `target_tolerance` |
| [`minimize_below`](#614-minimize_below) | Prefer `f(x) ≤ T`, softly penalize above. | `threshold` |
| [`maximize_above`](#615-maximize_above) | Prefer `f(x) ≥ T`, softly penalize below. | `threshold` |
| [`maximize_below`](#616-maximize_below) | Maximize `f(x)` but stay below `T`. | `threshold` |
| [`minimize_above`](#617-minimize_above) | Minimize `f(x)` but stay above `T`. | `threshold` |
| [`within_range`](#618-within_range)     | Keep `f(x) ∈ [A, B]`, preferably near `ideal`. | `range` |
| [`explore`](#619-explore) *(alias `probe`)* | Prefer high-variance regions. | — |
| [`improve`](#6110-improve)              | Prefer the largest expected improvement over the current best. | — |

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

Prefer `f(x) ≤ T`; anything above is softly penalized.

```json
"impurity_ppm": {
  "goal": "minimize_below",
  "threshold": { "type": "<=", "value": 50, "weight": 1.0 }
}
```

### 6.1.5 `maximize_above`

Prefer `f(x) ≥ T`; anything below is softly penalized.

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

If a quantity is **hard**, model it as a constraint, not an objective:

| Requirement | Model as |
|---|---|
| “pH must be in `[6.8, 7.2]`.” | `within_range` objective **or** pre/post filter, depending on how expensive violations are. |
| “Total moles must equal 1.” | `sum_constraints`, not an objective. |
| “Ratio of `A` to `B` between `0.5` and `2`.” | `ratio_constraints`, not an objective. |

Objectives are things the optimizer should **trade off**; constraints are
things it **must not violate**.

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
