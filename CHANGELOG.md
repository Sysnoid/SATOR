# Changelog

All notable changes to this project will be documented in this file.

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

