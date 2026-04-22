<h1 align="center">SATOR Engine</h1>
<p align="center">
  Statistical Adaptive Tuning and Optimization Runtime </br>
</p>
<p align="center">
  <a href="https://github.com/WytchDocQ/SatorOptimizer/actions/workflows/ci.yml">
    <img src="https://github.com/WytchDocQ/SatorOptimizer/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0.txt">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0">
  </a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
</p>

SATOR is an open-source, **stateless multi-objective Bayesian optimization
server** with a simple HTTP API. It ships with sensible defaults, no database,
no UI, and no message broker — just a single process you can run locally or
in Docker on CPU or NVIDIA GPU.

SATOR was originally built for **chemical formulations** and **material
compositions**, but it generalizes to any continuous-parameter black-box
optimization problem. It supports two input classes simultaneously:

1. **Compositional “ingredients”** that must sum to a target (mixtures).
2. **Free parameters** normalized independently.

This makes SATOR suitable for industrial formulation work, process tuning,
hyperparameter search, or any domain where sample-efficient optimization
matters.

## What you get

- Multi-objective Bayesian optimization via [BoTorch](https://botorch.org/)
  (`qnehvi`, `qehvi`, `qnoisyehvi`, `parego`) and matching single-objective
  variants (`qei`, `qpi`, `qucb`).
- Advanced goal types: `target`, `within_range`, `minimize_below`,
  `maximize_above`, `maximize_below`, `minimize_above`, `explore`, `improve`.
- Sum-to-one and ratio constraints as first-class inputs.
- Optional PCA encoding with automatic SLSQP reconstruction back to
  original variables.
- GP posterior maps (1-D / 2-D / 3-D) for visualization.
- Production essentials: API-key auth, rate limiting, IP allow/deny lists,
  idempotency, health, and Prometheus metrics.

## Quick look

![Branin 3D surface — short BO run (minimize)](tests/artifacts/visual_branin_3d.png)

## Run it

Easiest:

- Press `F5` in VS Code or Cursor and pick *“SATOR: Run Server”* — a
  `.vscode/launch.json` is shipped.
- Or, from a shell (virtualenv active):

  ```bash
  sator-server
  ```

Configure a single API key in `.env`:

```dotenv
SATOR_API_KEY=dev-key
```

Health probes on `http://localhost:8080`: `/livez`, `/readyz`.

Full installation options (CPU, GPU, Docker) are covered in
[§3 Installation](docs/03-installation.md).

## Try an optimization

```bash
curl -s \
  -H "x-api-key: dev-key" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": { "X": [[0.1],[0.9]], "Y": [[0.5],[0.2]] },
    "search_space": { "parameters": [ { "name": "x1", "type": "float", "min": 0, "max": 1 } ] },
    "objectives": { "o1": { "goal": "min" } },
    "optimization_config": { "acquisition": "qei", "batch_size": 2, "max_evaluations": 10 }
  }' \
  http://localhost:8080/v1/optimize
```

Then poll `/v1/jobs/<job_id>/result` for the result.

## Documentation

The full, numbered documentation lives in [`docs/`](docs/README.md):

| § | Topic |
|---|---|
| 1 | [Overview](docs/01-overview.md) |
| 2 | [Quickstart](docs/02-quickstart.md) |
| 3 | [Installation](docs/03-installation.md) *(local + Docker, CPU + GPU)* |
| 4 | [Configuration](docs/04-configuration.md) *(environment-variable reference)* |
| 5 | [API Reference](docs/05-api-reference.md) — also [`openapi.yaml`](docs/openapi.yaml) |
| 6 | [Objectives & Constraints](docs/06-objectives-and-constraints.md) |
| 7 | [Optimization Pipeline](docs/07-optimization-pipeline.md) |
| 8 | [Advanced Tuning](docs/08-advanced-tuning.md) |
| 9 | [Reconstruction](docs/09-reconstruction.md) |
| 10 | [Operations](docs/10-operations.md) |
| 11 | [Local HTTPS Setup](docs/11-local-https-setup.md) |
| 12 | [Troubleshooting](docs/12-troubleshooting.md) |

Release notes live in [`CHANGELOG.md`](CHANGELOG.md).

## Stewardship & governance

This project is created and maintained by **Sysnoid Technologies Oy**. We
welcome community pull requests. By contributing, you agree that your
changes are licensed under Apache-2.0, the same license as this repository.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines and
[`SECURITY.md`](SECURITY.md) for vulnerability reporting. Contributors are
credited in the git history and release notes.

## License

Apache-2.0. See [`LICENSE`](LICENSE) — full text:
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt).
