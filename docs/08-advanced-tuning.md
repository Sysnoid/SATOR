---
title: Advanced Tuning
sidebar_position: 8
slug: /advanced-tuning
---

# 8. Advanced Tuning

All fields documented here live inside the optimization request under
`optimization_config`. Every field is **optional** — defaults are calibrated
for typical industrial problems.

Three related config blocks exist:

| Block | Purpose |
|---|---|
| `gp_config` | Hyperparameter hints and freezes for the Gaussian-process surrogate. |
| `acquisition_params` | Knobs for the acquisition function (e.g. UCB exploration weight). |
| `advanced` | Escape hatch for numerical/solver settings (MC samples, restarts, etc.). |

---

## 8.1 Gaussian process

### 8.1.1 `gp_config`

Hints applied *before* the MLL fit, and optional **freezes** applied *after*:

| Field | Type | Meaning |
|---|---|---|
| `lengthscale` | float or list | Initial lengthscale(s); per-dim when ARD is on. |
| `outputscale` | float | Kernel variance prior / initial value. |
| `noise` | float or `"auto"` | Observation-noise variance σ². `"auto"` is the default. |
| `ard` | bool | Enable Automatic Relevance Determination (per-dimension lengthscales). |
| `fix_lengthscale` | bool | Re-apply `lengthscale` after fit and freeze it. |
| `fix_outputscale` | bool | Re-apply `outputscale` after fit and freeze it. |
| `fix_noise` | bool | Re-apply `noise` after fit and freeze it. |

Rules of thumb:

- **Standardized `Y`** — `noise` in `[1e-4, 1e-2]` works for most problems.
- **Raw `Y`** — start at `noise ≈ 0.5 %–5 %` of `Var(Y)`.
- Freezing hyperparameters is powerful for short or noisy data sets where
  the MLL fit overfits; consider freezing `lengthscale` first.

### 8.1.2 `advanced` (GP fit)

| Field | Type | Meaning |
|---|---|---|
| `kernel` | `matern52` \| `rbf` | Covariance kernel. `matern52` (default) is best for most physical systems; `rbf` only when the response is very smooth. |
| `jitter` | float | Diagonal added to the kernel for Cholesky stability. Default ≈ `1e-8` in float64. Bump to `1e-7` / `1e-6` if you see `cholesky` errors. |
| `fit_maxiter` | int | Max iterations for MLL optimization. Typical: `100–500`. |
| `fit_lr` | float | Learning rate for hyperparameter optimization. Typical: `0.05–0.2`. |
| `float64` | bool | Use double precision. **Default `true`**; improves numerical stability. |

---

## 8.2 Acquisition

### 8.2.1 `acquisition_params`

Knobs specific to the acquisition function chosen in `acquisition`:

| Field | Used by | Meaning | Typical |
|---|---|---|---|
| `ucb_beta` | `qucb` | Exploration weight in `qUpperConfidenceBound`. Larger = more exploration. | `0.1–10` |
| `pi_tau` | `qpi` | Improvement threshold τ for `qProbabilityOfImprovement`. | `0.0–1.0` |
| `qmc_samples` | all MC acquisitions | Sobol QMC samples per posterior evaluation. | `64–512` |

### 8.2.2 `advanced` (acquisition optimization)

| Field | Meaning | Typical |
|---|---|---|
| `mc_samples` | Monte-Carlo samples for `q*` acquisitions. | `128–512` |
| `num_restarts` | Multi-starts for `optimize_acqf`. More starts = better global optimum at proportional cost. | `5–20` |
| `raw_samples` | Sobol raw samples used to initialize each restart. Rule of thumb: `20–50 × dim`. | `128–1024` |
| `batch_limit` | Per-iteration batch size for the internal line search. | `5–20` |
| `acq_maxiter` | Acquisition-optimizer iterations. Diminishing returns past ~`200`. | `50–300` |
| `sequential` | When `true`, propose the `q` candidates one at a time. Slower but often more diverse. | `false` |
| `ei_gamma` / `pi_gamma` | Baseline offsets for `qEI` / `qPI`. Higher = prefer larger improvements over tiny ones. | `0–1` |

### 8.2.3 Speed vs. stability cheat-sheet

| Priority | Recipe |
|---|---|
| **Fastest** | `acquisition: qehvi`, low `mc_samples` (64–128), low `num_restarts` (3–5). |
| **Balanced (default)** | `acquisition: qnehvi`, `mc_samples: 128`, `num_restarts: 8`, `raw_samples: 256`. |
| **Most stable** | `acquisition: qnehvi`, `mc_samples: 256–512`, `num_restarts: 16–20`, `raw_samples: 512`. |

---

## 8.3 Constraints handling

| Field | Meaning |
|---|---|
| `method` | `feasibility_weight` \| `penalty` *(planned)* — how constraints influence scoring. |
| `penalty_coef` | Scale of penalty when `method=penalty` (e.g. `1.0 – 100.0`). |

The current release enforces sum and ratio constraints via `optimize_acqf`
linear inequalities (in input space) and by post-selection projection
(`enforce_sum_constraints_np`) in PCA space.

---

## 8.4 Reference point & fantasization (multi-objective)

| Field | Meaning |
|---|---|
| `ref_point` | Explicit hypervolume reference point. For minimization, choose slightly worse than the worst observed `Y` per objective. |
| `ref_point_strategy` | `observed_min_margin` (default) derives the point from data with a safety margin. |
| `pending_as_fantasies` | When `true` (default for async use), model pending candidates as fantasy observations to reduce duplicate suggestions. |

---

## 8.5 Device & precision

| Field | Meaning |
|---|---|
| `float64` | Use double precision for GP solves (default `true`). |
| `device` | `cpu` \| `cuda`. Also controlled by the `SATOR_DEVICE` environment variable. CUDA accelerates large workloads, but double-precision performance varies by GPU. |

---

## 8.6 Example

```json
{
  "optimization_config": {
    "acquisition": "qnehvi",
    "batch_size": 4,

    "gp_config": {
      "ard": true,
      "noise": "auto",
      "fix_noise": false
    },

    "acquisition_params": {
      "qmc_samples": 256
    },

    "advanced": {
      "mc_samples": 256,
      "num_restarts": 8,
      "raw_samples": 256,
      "batch_limit": 8,
      "acq_maxiter": 200,
      "kernel": "matern52",
      "fit_maxiter": 200
    }
  }
}
```

Unrecognized keys in `advanced` are ignored, so it is safe to leave forward-
looking settings in a request that targets multiple engine versions.
