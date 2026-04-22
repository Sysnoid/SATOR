---
title: Overview
sidebar_position: 1
slug: /overview
---

# 1. Overview

## 1.1 What is SATOR?

**SATOR** (*Statistical Adaptive Tuning and Optimization Runtime*) is an
open-source optimization engine for **black-box, multi-objective problems**.
It was originally built for chemical formulation and material composition
work, and it generalizes to any continuous-parameter optimization task where
each experiment is expensive and you want to choose the next ones carefully.

SATOR runs as a stateless HTTP server. There is no database, no UI, no queue
broker. You send a JSON request, you poll a job, you get results.

## 1.2 Capabilities

### 1.2.1 Optimization
- **Multi-objective Bayesian optimization** via [BoTorch](https://botorch.org/)
  with Gaussian-process surrogates.
- Acquisition functions selectable per request:
  - `qnehvi` — noisy expected hypervolume improvement *(default, multi-objective)*
  - `qehvi` — expected hypervolume improvement
  - `parego` — Chebyshev-scalarized qLogEI *(good for quick compromises)*
  - `qei`, `qpi`, `qucb` — single-objective variants
- Advanced goal types (`target`, `within_range`, thresholds, `explore`, `improve`)
  fall back to a **Sobol-grid + posterior-scoring** path that shapes the score
  according to the goal semantics.

### 1.2.2 Inputs and constraints
- Free continuous parameters with bounds.
- **Mixture / ingredient** parameters that must sum to a target
  (e.g. sum-to-one for formulations).
- **Ratio constraints** between any two parameters.
- Optional **PCA** encoding for high-dimensional inputs, with automatic
  reverse-mapping back to the original variables.

### 1.2.3 Diagnostics and visualization
- GP posterior **means** and **variances** returned per candidate.
- Optional **GP maps**: 1-D curves, 2-D surfaces, 3-D volumes, either in
  original input space or in PCA-normalized space.
- Pareto-front indices and points in the response.

### 1.2.4 Operational
- Single-API-key authentication, per-key and per-IP rate limiting, IP
  allow/deny lists, and idempotency keys.
- Async job model with `/v1/jobs/{id}` and `/v1/jobs/{id}/result`.
- Health (`/livez`, `/readyz`) and Prometheus (`/metrics`) endpoints.
- Runs on CPU or NVIDIA GPU, directly or in Docker.

## 1.3 Typical use cases

- **Chemical formulations** — ingredient fractions that must sum to one, with
  physical targets such as viscosity, pH, cost.
- **Materials science** — compositional mixtures plus process parameters.
- **Process tuning** — industrial recipes combining continuous knobs.
- **Hyperparameter search** — any continuous-parameter black-box objective
  with an expensive evaluation cost, including algorithmic-trading strategies.

## 1.4 Architecture at a glance

```
+-------------------+        HTTP/JSON         +----------------------------+
|    Client / SDK   | -----------------------> |  SATOR server              |
|                   | <-----------------------  |  (FastAPI + Uvicorn)      |
+-------------------+      job_id / result      +--------------+-------------+
                                                               |
                                                               v
                                                   +-----------+-----------+
                                                   |   Optimization core   |
                                                   |  (NumPy / PyTorch /   |
                                                   |   BoTorch / GPyTorch) |
                                                   +-----------+-----------+
                                                               |
                                     (optional) GP maps, PCA, SLSQP reconstruction
```

- **Transport:** HTTP only.
- **State:** in-process; job results expire after `SATOR_RESULT_TTL_SEC`.
- **Concurrency:** bounded by `SATOR_CONCURRENCY`; each job has a hard
  timeout (`SATOR_JOB_TIMEOUT_SEC`).
- **Scaling:** horizontally by putting a reverse proxy in front of multiple
  instances. See [§10 Operations](10-operations.md).

## 1.5 Request lifecycle

1. `POST /v1/optimize` → server validates, returns `202 Accepted` with a `job_id`.
2. Job runs asynchronously on a worker coroutine.
3. `GET /v1/jobs/{job_id}` → status (`QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`).
4. `GET /v1/jobs/{job_id}/result` → full result payload once `SUCCEEDED`.

The same pattern applies to `POST /v1/reconstruct` for inverse-mapping
PCA coordinates back to original variables.

## 1.6 Where to go next

- You want to see it run: [§2 Quickstart](02-quickstart.md).
- You want to design a real request: [§5 API reference](05-api-reference.md).
- You want to understand the math: [§7 Optimization pipeline](07-optimization-pipeline.md).
