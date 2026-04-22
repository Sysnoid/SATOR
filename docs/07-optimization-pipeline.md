---
title: Optimization Pipeline
sidebar_position: 7
slug: /optimization-pipeline
---

# 7. Optimization Pipeline

This chapter walks the end-to-end path an optimization request takes through
the engine. Each step lists the **primary source file** so you can drill down
quickly.

---

## 7.1 Pipeline overview

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                          POST /v1/optimize                                │
 └──────────────────────────────────────────────────────────────────────────┘
                                  │
      1. Request ingestion        ▼            sator_os_engine/api/routes/optimize.py
      ─────────────────── OptimizeRequest ──── sator_os_engine/core/optimizer/mobo_engine.py
                                  │
      2. Preprocessing            ▼            preprocess.py
                                  │                (sum-to-target scaling · PCA · normalization)
                                  ▼
      3. GP model building                    gp.py
                                  │                (SingleTaskGP per objective · ModelListGP)
                                  ▼
      4. Candidate selection                  acquisition.py
                                  │                (qEHVI · qNEHVI · ParEGO · qEI · qPI · qUCB · Sobol+score)
                                  ▼
      5. Feasibility & constraints            utils.py
                                  │                (sum · ratio · bounds)
                                  ▼
      6. GP maps (optional)                   maps.py
                                  │                (2-D / 3-D posterior mean & variance)
                                  ▼
      7. Response assembly                    mobo_engine.py
                                  │                (predictions · pareto · diagnostics · maps)
                                  ▼
      8. (separate) Reconstruction            reconstruction/slsqp_reconstructor.py
                                              (PCA → original variables via SLSQP)
```

---

## 7.2 Step 1 — Request ingestion

- **File:** `sator_os_engine/api/routes/optimize.py`
- **Handoff:** `sator_os_engine/core/optimizer/mobo_engine.py` → `run_optimization(...)`

The route validates the Pydantic model, converts numerical arrays to NumPy /
PyTorch tensors, and dispatches the job to the async executor. The job id is
returned to the client immediately.

`run_optimization` owns the rest of the pipeline:

1. Parse & validate the structured config.
2. Preprocess inputs.
3. Fit GP models.
4. Select candidates.
5. Enforce feasibility.
6. Optionally build GP maps.
7. Assemble the response.

## 7.3 Step 2 — Preprocessing & normalization

- **File:** `sator_os_engine/core/optimizer/preprocess.py`

| Function | Purpose |
|---|---|
| `enforce_sum_to_target_training(X, sums_cfg)` | Scales the columns in a sum group so each training row meets its target sum, ensuring the GP sees feasible exemplars. |
| `fit_pca_normalize(X, k)` | Returns `(pca, pc_mins, pc_maxs, pc_range, Z_norm)`. Produces the `encoding_info` echoed back in the response. |
| `input_to_z_norm(pca, pc_mins, pc_range, X)` | Forward projection into the normalized PCA space `[0, 1]^k`. |
| `z_norm_to_input(pca, pc_mins, pc_range, z_norm)` | Inverse projection back to original variables. |

Downstream modeling then runs in **one** of two spaces:

- **Input space** — `X`, optionally standardized by the GP outcome transform.
- **PCA space** — `Z_norm ∈ [0, 1]^k`.

## 7.4 Step 3 — Gaussian-process model building

- **File:** `sator_os_engine/core/optimizer/gp.py`

| Function | Purpose |
|---|---|
| `build_models(tX, tY, cfg)` | Builds a `ModelListGP` of one `SingleTaskGP` per objective, with a `Standardize` outcome transform. |
| `bounds_input(params, tdtype, tdevice)` | Tensor bounds in input space. |
| `bounds_model_pca(k, tdtype, tdevice)` | `[0, 1]` bounds in PCA space. |

GP hyperparameters are fit via `fit_gpytorch_mll`. Optional hints from
`gp_config` are applied *before* the MLL fit; fields prefixed with `fix_` are
re-applied *after* the fit and the parameter is frozen so optimization never
overwrites it. See [§8.1](08-advanced-tuning.md#81-gaussian-process) for
every supported knob.

## 7.5 Step 4 — Candidate selection (acquisition)

- **File:** `sator_os_engine/core/optimizer/acquisition.py`
- **Entry points:**
  - `select_candidates_single_objective(...)`
  - `select_candidates_multiobjective(...)`

### 7.5.1 Multi-objective path (all goals are `min` / `max`)

The `acquisition` field is dispatched to the corresponding BoTorch class:

| Request token | BoTorch class |
|---|---|
| `qnehvi` *(default)* | `qLogExpectedHypervolumeImprovement` |
| `qehvi` | `qExpectedHypervolumeImprovement` |
| `qnoisyehvi` | `qNoisyExpectedHypervolumeImprovement` |
| `parego` | `qLogExpectedImprovement` with Chebyshev scalarization |
| `qei` | `qLogExpectedImprovement` |
| `qpi` | `qProbabilityOfImprovement` |
| `qucb` | `qUpperConfidenceBound` |

Reference-point partitioning for hypervolume acquisitions is done by
`NondominatedPartitioning`. The Monte-Carlo sampler is `SobolQMCNormalSampler`.

### 7.5.2 Single-objective path

The same dispatch applies for supported single-objective tokens (`qei`,
`qpi`, `qucb`, or `parego` if you want a scalarized single-objective run).
For all *other* cases, `select_candidates_single_objective` falls back to a
Sobol grid + posterior-scoring strategy.

### 7.5.3 Advanced-goal path

If **any** objective uses an advanced goal (`target`, `within_range`,
threshold types, `explore`, `improve`), the engine ignores `acquisition` and
uses Sobol sampling plus goal-shaped posterior scoring. The score penalizes
distance to target, threshold violations, range-membership failure, or
rewards high variance, depending on the goal.

### 7.5.4 Constraint handling during acquisition

- **Input space** — linear inequalities (sum / ratio) are passed as
  `inequality_constraints` to `optimize_acqf`.
- **PCA space** — indices do not align with the model dimension, so
  constraints are enforced post-selection via `feasible_mask` and
  `enforce_sum_constraints_np` (§7.6).

## 7.6 Step 5 — Feasibility & constraint enforcement

- **File:** `sator_os_engine/core/optimizer/utils.py`

| Function | Purpose |
|---|---|
| `build_linear_constraints(req, params)` | Translates sum/ratio constraints into linear (in)equalities on input-space indices. |
| `feasible_mask(points, req, params, tol)` | Per-point feasibility check for grid-scoring paths. |
| `enforce_sum_constraints_np(cands, params, req)` | Iterative projection that adjusts a chosen candidate to hit target sums while respecting bounds. |

`enforce_sum_constraints_np` runs at most a small, bounded number of
iterations; if convergence fails, the offending candidate is discarded and
the next best is tried.

## 7.7 Step 6 — GP maps (optional)

- **File:** `sator_os_engine/core/optimizer/maps.py`
- **Function:** `compute_gp_maps(...)`

When `optimization_config.return_maps=true`, SATOR constructs a 1-D, 2-D, or
3-D grid over either input space or PCA-normalized space, evaluates the GP
posterior mean and variance per objective, and returns a compact
JSON-serializable structure:

```jsonc
"gp_maps": {
  "space": "input",           // or "pca"
  "dimension": 2,             // 1 | 2 | 3
  "grid": {
    "axes": [[/* x */], [/* y */]],
    "resolution": [nx, ny]
  },
  "maps": {
    "means":     { "<obj>": [[/* … */]] },
    "variances": { "<obj>": [[/* … */]] }
  }
}
```

Maps are only produced when they can be rendered: PCA maps require
`pca_dimension ∈ {2, 3}`; input maps require at least two continuous
parameters.

## 7.8 Step 7 — Response assembly

The orchestrator assembles:

- `predictions` — each selected candidate with:
  - `candidate` — mapped back to original variable names.
  - `objectives`, `variances` — GP posterior at the candidate.
  - `encoded` — PCA coordinates, when PCA was used.
  - `reconstructed` — formulation (ingredients, parameters, combined, optional
    `by_name`), when reconstruction applied.
- `pareto` — indices and points of non-dominated predictions.
- `encoding_info` — PCA metadata for client-side reconstruction.
- `diagnostics` — compute device, CUDA index, GP fit details.
- `gp_maps` — if requested and available.

## 7.9 Step 8 — Reconstruction

- **File:** `sator_os_engine/reconstruction/slsqp_reconstructor.py`
- **Route:** `sator_os_engine/api/routes/reconstruct.py`
- **Details:** see [§9 Reconstruction](09-reconstruction.md).

Reconstruction is available both **inline** (returned in each prediction when
PCA is used) and via a **standalone** endpoint that takes a raw PCA point plus
the necessary metadata and returns a feasible formulation.

---

## 7.10 Order-of-operations summary

1. `POST /v1/optimize` → `run_optimization`.
2. Sum-group scaling on training `X` (if configured).
3. Fit PCA and normalize to `Z_norm`, **or** keep working in input space.
4. Build one `SingleTaskGP` per objective (`ModelListGP`).
5. Compute tensor bounds for the active space.
6. Select candidates:
   - Multi-objective `min` / `max` only → BoTorch acquisition class.
   - Any advanced goal, or PCA space → Sobol grid + scoring.
7. Enforce feasibility on the selected batch.
8. Optionally compute GP maps.
9. Assemble and return the response.
10. Reconstruct PCA coordinates on demand via `POST /v1/reconstruct`.

---

## 7.11 File / function index

| Area | File | Key symbols |
|---|---|---|
| Orchestration | `sator_os_engine/core/optimizer/mobo_engine.py` | `run_optimization` |
| Preprocessing | `sator_os_engine/core/optimizer/preprocess.py` | `enforce_sum_to_target_training`, `fit_pca_normalize`, `input_to_z_norm`, `z_norm_to_input` |
| GP models | `sator_os_engine/core/optimizer/gp.py` | `build_models`, `bounds_input`, `bounds_model_pca` |
| Acquisition | `sator_os_engine/core/optimizer/acquisition.py` | `select_candidates_single_objective`, `select_candidates_multiobjective` |
| Feasibility | `sator_os_engine/core/optimizer/utils.py` | `build_linear_constraints`, `feasible_mask`, `enforce_sum_constraints_np` |
| GP maps | `sator_os_engine/core/optimizer/maps.py` | `compute_gp_maps` |
| Device resolution | `sator_os_engine/core/optimizer/device.py` | `resolve_torch_device` |
| Reconstruction | `sator_os_engine/reconstruction/slsqp_reconstructor.py` | `reconstruct` |
| HTTP routes | `sator_os_engine/api/routes/optimize.py`, `reconstruct.py`, `jobs.py` | — |
