---
title: API Reference
sidebar_position: 5
slug: /api-reference
---

# 5. API Reference

SATOR exposes a small JSON/HTTP API. The machine-readable source of truth is
[`openapi.yaml`](openapi.yaml) (OpenAPI 3.1). This page is the human-friendly
companion.

---

## 5.1 Base URL and headers

| | |
|---|---|
| **Base URL (HTTP)**  | `http://localhost:8080` |
| **Base URL (HTTPS)** | `https://localhost:8443` |
| **Auth header**      | `x-api-key: <SATOR_API_KEY>` |
| **Content type**     | `application/json` |
| **Idempotency**      | `idempotency-key: <uuid>` *(optional)* |

Every call except `/livez`, `/readyz`, and `/metrics` requires `x-api-key`.

## 5.2 Endpoint map

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/livez`                         | Liveness probe, always returns `{"status":"ok"}` |
| `GET`  | `/readyz`                        | Readiness probe |
| `GET`  | `/metrics`                       | Prometheus metrics *(if `SATOR_ENABLE_METRICS=true`)* |
| `POST` | `/v1/optimize`                   | Submit an optimization job *(async)* |
| `POST` | `/v1/reconstruct`                | Submit a reconstruction job *(async)* |
| `GET`  | `/v1/jobs/{job_id}`              | Job status |
| `GET`  | `/v1/jobs/{job_id}/result`       | Final job result *(once `SUCCEEDED`)* |

All optimization and reconstruction endpoints are **asynchronous**: they
return `202 Accepted` with a `job_id`, and you poll for the result.

---

## 5.3 `POST /v1/optimize`

Submit an optimization request.

### 5.3.1 Request body

```jsonc
{
  "dataset":          { /* see 5.3.2 */ },
  "search_space":     { /* see 5.3.3 */ },
  "objectives":       { /* see 5.3.4 */ },
  "optimization_config": { /* see 5.3.5 */ }
}
```

#### 5.3.2 `dataset`

| Field | Type | Description |
|---|---|---|
| `X` | `number[][]` | Prior input points, one row per observation. Categorical columns may be choice labels. |
| `Y` | `number[][]` | Prior objective values. Column order must match `objectives`. Non-finite entries are dropped per objective. |
| `pending_X` | `number[][]` | Recipes already running but not yet measured. Also accepted as `X_pending`. |
| `Yvar` | `number[][]` | Known outcome variance, same shape as `Y`. Selects a fixed-noise GP. |
| `failed` | `bool[]` | One flag per row. Failures train a feasibility model and down-weight infeasible proposals. |

`X` and `Y` are required, must have the same number of rows (at least one), and
`Y` must have one column per objective. A column that is entirely NaN or inf is
rejected. `batch_size` above 16, an unknown goal or acquisition, or a
`pca_dimension` outside `1 .. min(n-1, d)` returns 422. Unknown
`optimization_config` keys are rejected.

#### 5.3.3 `search_space`

```jsonc
{
  "parameters": [
    { "name": "x1", "type": "float", "min": 0.0, "max": 1.0 },
    { "name": "x2", "type": "int",   "min": 1,   "max": 10  },
    { "name": "x3", "type": "categorical", "choices": ["A", "B"] }
  ]
}
```

| `type`         | Required fields         |
|---------------|--------------------------|
| `float`       | `min`, `max`             |
| `int`         | `min`, `max`             |
| `categorical` | `choices` *(strings or numbers)* |

#### 5.3.4 `objectives`

A map of objective name → configuration. Order of iteration is preserved and
must match the column order of `dataset.Y`.

```jsonc
{
  "viscosity": { "goal": "target", "target_value": 1200.0 },
  "cost":      { "goal": "min" }
}
```

Supported `goal` values and their extra fields are detailed in
[§6 Objectives & constraints](06-objectives-and-constraints.md).

#### 5.3.5 `optimization_config`

| Field | Type | Default | Notes |
|---|---|---|---|
| `acquisition` | string | `qnehvi` | `qnehvi`, `qehvi`, `qnoisyehvi`, `qlognei`, `parego`, `qei`, `qpi`, `qucb`, `sobol`, or `auto`. `auto` with `mode: lab` uses qLogNEI / qLogNEHVI. |
| `mode` | `lab` \| `throughput` | `lab` | `lab` treats `acquisition: auto` as noisy. `throughput` keeps qLogEI / qLogEHVI unless `Yvar`, replicates, or failures are present. |
| `batch_size` | int 1–16 | `4` | Number of candidates returned per request. |
| `max_evaluations` | int ≥ 1 | `100` | Accepted for compatibility. The acquisition loop uses `advanced.acq_maxiter` (default 200). |
| `seed` | int | — | RNG seed. Omitted seeds are drawn at random and returned in `diagnostics.seed`. |
| `use_pca` | bool | `false` | Fit PCA on `X` for modeling and/or visualization. |
| `pca_dimension` | int or `"auto"` | — | Required when `use_pca=true`. `1 <= k <= min(n-1, d)`, or `"auto"` for 95% variance capped at 10. |
| `pca_margin` | float | `0` | Kept for compatibility. PCA acquisition searches in the original variables, so the latent margin is not the search box. |
| `parameter_scaling` | `none` \| `standardize` \| `minmax` | `minmax` | Input transform. Default min-max uses the search bounds. |
| `value_normalization` | `none` \| `standardize` \| `minmax` | `standardize` | Outcome transform inside the GP. |
| `target_tolerance` | float | — | Tolerance band around target-attainment goals. |
| `target_variance_penalty` | float | — | Variance weighting for `goal="target"`. |
| `sum_constraints` | array | — | `[{"indices":[0,1],"target_sum":1.0}, ...]` |
| `ratio_constraints` | array | — | `[{"i":0,"j":1,"min_ratio":0.5,"max_ratio":2.0}, ...]` |
| `ingredient_rules` | object | — | `{"max_nonzero": 4, "min_dose": 0.02}` inside each sum group. |
| `mixture_transform` | `none` \| `ilr` | `none` | ILR coordinates for sum groups. Not combined with PCA or categoricals. |
| `return_maps` | bool | `false` | Return GP posterior maps in the response. |
| `map_space` | `input` \| `pca` | `input` | Coordinate system for returned maps. |
| `map_resolution` | `[nx, ny, (nz)]` | — | Grid resolution for maps. |
| `gp_config` | object | — | GP hyperparameter hints, see [§8.1](08-advanced-tuning.md#81-gaussian-process). |
| `acquisition_params` | object | — | Acquisition knobs, see [§8.2](08-advanced-tuning.md#82-acquisition). |
| `advanced` | object | — | Escape hatch for experimental settings, see [§8](08-advanced-tuning.md). |

#### 5.3.6 Acquisition functions

| Name            | Family                   | Best for |
|-----------------|--------------------------|----------|
| `qnehvi`        | qLogExpectedHypervolumeImprovement | **Default** for multi-objective; tolerant of noise |
| `qehvi`         | qExpectedHypervolumeImprovement    | Faster multi-objective when noise is low |
| `qnoisyehvi`    | qNoisyExpectedHypervolumeImprovement | Multi-objective with explicit noise modeling |
| `parego`        | qLogEI on a Chebyshev-scalarized objective | Quick multi-objective compromises |
| `qei`           | qLogExpectedImprovement   | Single-objective, well-behaved noise |
| `qpi`           | qProbabilityOfImprovement | Single-objective, risk-averse |
| `qucb`          | qUpperConfidenceBound     | Single-objective, exploration-heavy |
| `sobol`         | Sobol grid + posterior scoring | Advanced goals (`target`, thresholds, …) |

Problems with *advanced* goal types (anything other than `min`/`max`) always
use the `sobol` grid path regardless of the `acquisition` string.

#### 5.3.7 Example request

```json
{
  "dataset": {
    "X": [[0.1, 0.2], [0.7, -0.1]],
    "Y": [[1.2, 0.5], [0.7, 0.9]]
  },
  "search_space": {
    "parameters": [
      { "name": "x1", "type": "float", "min": 0.0, "max": 1.0  },
      { "name": "x2", "type": "float", "min": -1.0, "max": 1.0 }
    ]
  },
  "objectives": {
    "o1": { "goal": "min" },
    "o2": { "goal": "target", "target_value": 0.8 }
  },
  "optimization_config": {
    "acquisition": "qnehvi",
    "batch_size": 4,
    "max_evaluations": 50,
    "seed": 42,
    "parameter_scaling": "standardize",
    "value_normalization": "standardize",
    "sum_constraints":   [{ "indices": [0], "target_sum": 1.0 }],
    "ratio_constraints": [{ "i": 0, "j": 1, "min_ratio": 0.5, "max_ratio": 2.0 }],
    "return_maps": true,
    "map_space": "input",
    "map_resolution": [60, 60]
  }
}
```

### 5.3.8 Responses

#### 202 Accepted

```json
{ "job_id": "job_abc123456789", "status": "QUEUED" }
```

#### 200 OK — job result

```jsonc
{
  "predictions": [
    {
      "candidate":     { "x1": 0.33, "x2": -0.1 },
      "objectives":    [0.91, 0.42],
      "variances":     [0.12, 0.05],
      "encoded":       [0.47, 0.62],            // only when use_pca=true
      "reconstructed": {                        // only when PCA + reconstruction applies
        "ingredients": [0.6, 0.4],
        "parameters":  [0.2],
        "combined":    [0.6, 0.4, 0.2],
        "by_name":     { "water": 0.6, "oil": 0.4, "temp": 0.2 }
      }
    }
  ],
  "pareto": { "indices": [0], "points": [[0.91, 0.42]] },
  "encoding_info": null,
  "diagnostics":   { "device": "cpu" },
  "gp_maps": {
    "space": "input",
    "dimension": 2,
    "grid": { "axes": [[0.0, 0.02, /* … */ 1.0], [-1.0, /* … */ 1.0]], "resolution": [60, 60] },
    "maps": {
      "means":     { "o1": [[/* … */]], "o2": [[/* … */]] },
      "variances": { "o1": [[/* … */]], "o2": [[/* … */]] }
    }
  }
}
```

---

## 5.4 `POST /v1/reconstruct`

Given a point in PCA-encoded space (plus enough PCA metadata), return a
feasible point in the original variable space that satisfies sum and ratio
constraints. Powered by SLSQP.

### 5.4.1 Request body

```jsonc
{
  "coordinates":       [0.5, 0.7],
  "pca_info": {
    "pc_mins":    [0.0, 0.0],
    "pc_maxs":    [1.0, 1.0],
    "components": [[1, 0, 0], [0, 1, 0]],
    "mean":       [0, 0, 0]
  },
  "bounds": {
    "ingredients": [[0, 1], [0, 1]],
    "parameters":  [[0, 1]]
  },
  "n_ingredients":      2,
  "sum_target":         1.0,
  "ratio_constraints":  [{ "i": 0, "j": 1, "min_ratio": 0.5, "max_ratio": 2.0 }],
  "target_precision":   1e-7,
  "ingredient_names":   ["water", "oil"],
  "parameter_names":    ["temperature"]
}
```

| Field | Description |
|---|---|
| `coordinates` | Point in PCA space, either normalized `[0,1]^D` or raw (controlled by `pca_info`). |
| `pca_info` | PCA components, mean, and per-component min/max for denormalization. |
| `bounds.ingredients` | Per-ingredient `[min, max]` pairs. |
| `bounds.parameters` | Per-parameter `[min, max]` pairs for the non-ingredient variables. |
| `n_ingredients` | Number of ingredient columns at the front of the combined vector. |
| `sum_target` | Target sum for the ingredient columns (default `1.0`). |
| `ratio_constraints` | Pairwise ratio bounds on `x_i / x_j`. |
| `target_precision` | SLSQP convergence tolerance. |
| `ingredient_names` / `parameter_names` | Optional; when supplied, the response includes `reconstructed_formulation.by_name`. |

### 5.4.2 Responses

#### 202 Accepted

```json
{ "job_id": "job_abc123", "status": "QUEUED" }
```

#### 200 OK — job result

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

`by_name` is included only when `ingredient_names` and `parameter_names` are
supplied in the request.

---

## 5.5 Job polling

### 5.5.1 `GET /v1/jobs/{job_id}`

```json
{ "job_id": "job_abc123", "status": "RUNNING" }
```

`status` is one of `QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`.

### 5.5.2 `GET /v1/jobs/{job_id}/result`

Returns the terminal payload. When `status=FAILED`, the body is:

```json
{ "status": "FAILED", "error": "human-readable reason" }
```

Results are retained for `SATOR_RESULT_TTL_SEC` seconds after completion.

---

## 5.6 Error format

All error responses share the shape:

```json
{ "detail": "message describing what went wrong" }
```

| Status | Meaning |
|---|---|
| `400` | Request validation failed (malformed body, wrong types, contradictory options). |
| `401` | Missing or invalid `x-api-key`. |
| `403` | Source IP not on the whitelist or explicitly blacklisted. |
| `404` | Unknown `job_id`. |
| `409` | Idempotency conflict (same key, different body). |
| `429` | Rate limit exceeded. |
| `500` | Internal error. Exception text is included only when `SATOR_EXPOSE_ERROR_DETAILS=true`. |
