---
title: Reconstruction
sidebar_position: 9
slug: /reconstruction
---

# 9. Reconstruction

When you optimize in PCA-encoded space, candidates come back as PCA
coordinates. **Reconstruction** maps those coordinates back to a concrete,
feasible point in the original variables — ingredients (which must sum to
a target) and free parameters — while honoring bounds and ratio constraints.

- **Algorithm:** Sequential Least Squares Programming (SLSQP)
- **Implementation:** `sator_os_engine/reconstruction/slsqp_reconstructor.py`
- **Endpoint:** `POST /v1/reconstruct` (see [§5.4](05-api-reference.md#54-post-v1reconstruct))

---

## 9.1 When reconstruction runs

### Inline (during optimization)

When `use_pca=true`, every prediction in the response body includes a
`reconstructed` block alongside the raw PCA coordinates:

```jsonc
"predictions": [
  {
    "candidate":     { /* mapped back to original names */ },
    "encoded":       [0.47, 0.62],
    "reconstructed": {
      "ingredients": [0.6, 0.4],
      "parameters":  [0.2],
      "combined":    [0.6, 0.4, 0.2],
      "by_name":     { "water": 0.6, "oil": 0.4, "temperature": 0.2 }
    }
  }
]
```

### Standalone

You can also reconstruct an arbitrary point on demand via the endpoint. This
is useful when your client stores PCA coordinates and wants to produce a
formulation later.

---

## 9.2 Request shape

See the API reference [§5.4](05-api-reference.md#541-request-body) for the
full schema. At minimum you need:

| Block | Purpose |
|---|---|
| `coordinates` | The PCA-space point to reconstruct. |
| `pca_info`    | `pc_mins`, `pc_maxs`, `components`, `mean` — enough to denormalize and inverse-project. |
| `bounds`      | Per-ingredient and per-parameter `[min, max]` pairs. |
| `n_ingredients` | How many of the leading variables are sum-constrained. |
| `sum_target`    | Target sum for the ingredient columns (default `1.0`). |

Optional fields that control the problem shape:

| Block | Purpose |
|---|---|
| `ratio_constraints` | Bound `x_i / x_j` pairs during the SLSQP search. |
| `target_precision` | SLSQP convergence tolerance. |
| `ingredient_names` / `parameter_names` | When supplied, the response includes a `by_name` map. |

---

## 9.3 What SLSQP is solving

Given a candidate PCA point `z`, denormalize it with `pca_info` to obtain a
target location `x̂` in original-variable space (the **reconstruction target**).
The reconstructor then solves:

```
minimize      ½ ‖ x − x̂ ‖²
subject to    ingredient_min[i] ≤ x[i] ≤ ingredient_max[i]    (i < n_ingredients)
              parameter_min[j]  ≤ x[n_ing + j] ≤ parameter_max[j]
              Σᵢ x[i] = sum_target                            (i < n_ingredients)
              min_ratio ≤ x[i] / x[j] ≤ max_ratio             (per ratio rule)
```

`x̂` is not guaranteed to be feasible (PCA cannot know about sum or ratio
constraints by itself), so SLSQP drives the solution back onto the feasible
set while staying as close to `x̂` as possible.

---

## 9.4 Response

```json
{
  "success": true,
  "reconstructed_formulation": {
    "ingredients": [0.6, 0.4],
    "parameters":  [0.2],
    "combined":    [0.6, 0.4, 0.2],
    "by_name":     { "water": 0.6, "oil": 0.4, "temperature": 0.2 }
  },
  "reconstruction_metrics": {
    "final_error": 1e-7,
    "iterations":  6,
    "method":      "SLSQP_Constrained"
  }
}
```

| Field | Meaning |
|---|---|
| `success` | SLSQP terminated within tolerance. |
| `ingredients` | Sum-to-target feasible fractions. |
| `parameters` | Remaining variables inside their bounds. |
| `combined` | `[*ingredients, *parameters]`. |
| `by_name` | Keyed dictionary (requires `ingredient_names` + `parameter_names`). |
| `final_error` | Constraint residual at termination. |
| `iterations` | SLSQP inner iterations. |
| `method` | Always `"SLSQP_Constrained"` for this implementation. |

---

## 9.5 Accuracy & limitations

- **Precision** is controlled by `target_precision`. `1e-7` is a safe default;
  for very tight formulations, `1e-9` yields sub-milligram precision on
  typical mixtures but may occasionally fail to converge on pathological
  cases.
- **Degenerate problems** (e.g. a sum group of size 1 with a hard sum target
  that conflicts with the single bound) return `success=false` with a
  diagnostic error rather than an arbitrary point.
- **Ratio constraints** must be consistent with bounds and sum constraints.
  Conflicting constraints raise a validation error at request time.
- **Non-ingredient variables** are not sum-constrained, so you can freely mix
  compositional ingredients with discrete or continuous process parameters
  in the same request.
