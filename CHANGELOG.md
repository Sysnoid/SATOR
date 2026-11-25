# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- **Explainability:** each prediction includes `explanation` with the predicted mean ± standard deviation, the distance to the nearest tested recipe, and a short reason for the proposal. `diagnostics.sensitivities` ranks inputs by inverse GP lengthscale. `POST /v1/explain` scores recipes the caller already has, without proposing a new batch.
- **Verification follow-up (v3):** multi-objective PCA no longer crashes when BoTorch asks for a subset of outputs. Constrained batches are optimized one point at a time, and a candidate whose posterior mean violates a threshold is replaced. PCA search keeps normalized scores inside the training box (widened by `pca_margin`) and reports `off_subspace_residual`, shrinking any candidate that leaves the subspace farther than the training data. A declared ratio keeps its denominator at least 1e-4 unless `min_denominator` is set. `/v1/reconstruct` accepts `x_reference` and otherwise returns a feasible recipe near the scaled-PCA inverse, which need not be the optimizer's candidate.
- **Verification follow-up:** PCA acquisition searches in the original variables and evaluates the GP on the encoding of that point, so `encoded` matches `encoded_actual`. `pca_margin` now defaults to 0. `/v1/reconstruct` uses the scaled-PCA inverse when scaler fields are present, and its default tolerance is 1e-3. GP min-max normalization uses the data envelope widened by 10% of the search box. Three-objective batches use qLogNEHVI with caching instead of forced sequential EHVI. Identical PCA rows are a 422. Ratio constraints accept an optional `min_denominator`. Validation errors no longer echo the request body.
- **Lab mode:** `acquisition: "auto"` selects qLogNEI / qLogNEHVI when `mode` is `lab`, and qLogEI / qLogEHVI in `throughput` unless the data carries known noise. `dataset.Yvar` and replicate rows train a fixed-noise GP. `dataset.failed` trains a feasibility model used as an acquisition constraint. Objectives accept `outcome_constraint: {lower, upper}`. `POST /v1/design` returns a Sobol design projected onto sums, ratios, and an optional cardinality limit. `ingredient_rules` applies that cardinality during optimization. `mixture_transform: "ilr"` fits sum groups in isometric log-ratio coordinates. Small categorical spaces use a mixed GP.
- **Correctness:** one sign convention for `max` / `maximize` / `min` / `minimize`
  and the threshold spellings. Threshold goals (`minimize_below`,
  `maximize_above`, `maximize_below`, `minimize_above`) are BoTorch outcome
  constraints, so a threshold weight can no longer flip the feasible side.
  Unknown goals and acquisitions return 422.
- **Requests:** `X`/`Y` shape, bounds, non-finite columns, `batch_size` ≤ 16,
  and `pca_dimension` are validated. `num_restarts`, `raw_samples`, `ref_point`,
  and `sequential` are read from `advanced` / `acquisition_params`. The
  hypervolume reference point scales with each objective. Omitted seeds are
  random and echoed in `diagnostics.seed`.
- **Search space:** `int` values are rounded and `categorical` values snap to a
  declared level. `dataset.pending_X` is passed through as pending experiments.
- **PCA:** reconstruction uses the original column order and every sum group,
  projects onto the feasible set, and reports `success` only when the recipe
  is feasible and on the latent target. Degenerate components are dropped
  (`k_effective`). `pca_dimension: "auto"` selects the latent size. Predictions are scored at the encoding of the returned recipe.
- **Runtime:** optimize and reconstruct jobs run in a child process. A timeout
  terminates that process, and the result queue is drained while the child
  is alive so large results cannot deadlock the worker.
- **Fix:** soft advanced-goal scoring (`target`, `within_range`, thresholds,
  `improve`) now converts GP posterior means from the engine's signed training
  frame back to original `Y` units before comparing to user `target_value` /
  ranges. Previously `run_optimization` fit on `Y * sign` but scored against
  unsigned targets, so `goal=target` collapsed toward minimizing `|Y|` instead
  of approaching `T`. Hard `enforce_*` masking was already correct. Regression
  tests in `tests/unit/test_target_goal_engine_signs.py`.
- **Objectives:** per-objective `target_tolerance`, `target_variance_penalty`,
  and `weight` on `goal: target` (fallback to the global optimization_config
  knobs). Multi-target soft scores are a weighted sum — document that color
  matching often prefers a single `min` on ΔE instead.

## [0.2.1] - 2026-04-23

- **Objectives:** hard-constraint goal family `enforce_above`, `enforce_below`,
  and `enforce_within_range` — GP-posterior feasibility mask during candidate
  selection, per-prediction `enforced_goals_satisfied` / `enforced_violations`,
  and `diagnostics.enforcement`; optional `enforcement_uncertainty_margin` in
  `OptimizationConfig` for LCB/UCB-style enforcement.
- **Docs:** `docs/06-objectives-and-constraints.md` and `docs/13-examples.md`
  updated; demo figures refreshed for Himmelblau, ZDT1, and EV electrolyte
  (`enforce_above` + PCA).
- **Examples:** `demo_03` z-axis / contour fix; `demo_04` denser training;
  `demo_08` uses `enforce_above` for the stability floor and the PCA path.
- **Tests:** `tests/unit/test_enforced_goals.py`.

## [0.2.0] - 2026-04-22
- **Transport:** removed the NATS transport and `nats-py` dependency — SATOR
  now exposes the HTTP API exclusively. Related settings (`SATOR_ENABLE_NATS`,
  `SATOR_NATS_URL`), the `sator-nats` console script, and the corresponding
  VS Code launch configuration have been dropped.
- **Packaging:** full `pyproject.toml` dependencies and `pip install -e ".[dev]"`; `requirements*.txt` kept in sync.
- **Docker:** `Dockerfile` (CPU), `Dockerfile.cuda` (NVIDIA), `docker-compose.yml` (CPU default, GPU via `--profile gpu`); see `docs/03-installation.md`.
- **Server:** single API key (`SATOR_API_KEY`); health endpoints `/livez`, `/readyz`; Prometheus `/metrics`.
- **Acquisition dispatch:** per-name dispatch to BoTorch classes
  (`qnehvi`, `qehvi`, `qnoisyehvi`, `parego`, `qei`, `qpi`, `qucb`) for
  multi-objective `min`/`max`; Sobol + posterior scoring for advanced goals.
- **Refactor:** modular optimizer (`preprocess.py`, `gp.py`, `acquisition.py`, `maps.py`, `utils.py`).
- **Failure semantics:** fallbacks removed — failures are explicit and surfaced via job errors.
- **PCA workflow:** automatic encoded dataset return; predictions include encoded and reconstructed values (with optional `by_name` map when ingredient and parameter names are supplied).
- **GP maps:** correct PCA-space surface generation for PCA(2); plotted via examples.
- **Constraints:** sum-to-one (ingredients) and ratio constraints enforced in acquisition and reconstruction.
- **Objectives:** threshold/range goals (`minimize_below`, `maximize_above`, `maximize_below`, `minimize_above`, `within_range`), plus `target`, `explore`/`probe`, `improve`.
- **Examples:** a curated demo set under `examples/` — single-objective (Rosenbrock), advanced goals (`target` on Ackley, `within_range` on Himmelblau), multi-objective Pareto (ZDT1), and realistic chemical formulations (3-ingredient mixture, ratio constraints, paint blend, EV electrolyte). Plus the original HTTP Branin and Chemical PCA demos. Outputs saved under `examples/responses/`.
- **Docs:** YAML frontmatter (`title`, `sidebar_position`, `slug`) added to every docset file for static-site ingestion.
- **Docs:** complete restructure into a numbered, indexed docset under `docs/` — overview, quickstart, installation, configuration, API reference, objectives, pipeline, advanced tuning, reconstruction, operations, local HTTPS, troubleshooting.

## [0.1.0]
- Initial open-source codebase scaffolding
- HTTP server with health endpoints
- Single API key auth (`SATOR_API_KEY`)
- Basic optimization and reconstruction scaffolding

