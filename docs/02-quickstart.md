---
title: Quickstart
sidebar_position: 2
slug: /quickstart
---

# 2. Quickstart

This guide brings you from zero to a working optimization call in under five
minutes. For a fuller install with GPU or Docker, see
[§3 Installation](03-installation.md).

## 2.1 Prerequisites

- **Python** 3.10 or newer
- **pip** and **venv** (shipped with Python)
- Roughly 2 GB free disk space for PyTorch wheels

## 2.2 Install

From a clean checkout of the repository:

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"
```

This installs the SATOR package in editable mode together with the development
extras (`pytest`, `ruff`, `vulture`). Plain users can drop `[dev]`.

## 2.3 Configure an API key

Copy `.env.example` to `.env` and set a key of your choosing:

```bash
cp .env.example .env
# then edit .env so it contains:
# SATOR_API_KEY=dev-key
```

The full environment-variable reference lives in
[§4 Configuration](04-configuration.md).

## 2.4 Start the server

Any of the following work:

- **In an IDE** — press `F5` in VS Code or Cursor and pick *“SATOR: Run Server”*
  (a `.vscode/launch.json` is shipped with the repo).
- **From a terminal:**
  ```bash
  sator-server
  ```
- **As a module:**
  ```bash
  python -m sator_os_engine.server.main
  ```

The server starts on the host/port configured by `SATOR_HTTP_HOST` /
`SATOR_HTTP_PORT` (default `0.0.0.0:8080`, plain HTTP). For HTTPS locally,
see [§11 Local HTTPS setup](11-local-https-setup.md).

Sanity check:

```bash
curl http://localhost:8080/livez
# → {"status":"ok"}
```

## 2.5 Run your first optimization

The request below proposes two new candidates for a single-objective
minimization problem in a one-dimensional search space:

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

You will get back a `job_id`. Poll for the result:

```bash
curl -s -H "x-api-key: dev-key" \
  http://localhost:8080/v1/jobs/<job_id>/result
```

When the job is `SUCCEEDED`, the response contains a list of **predictions**
(next candidate points) plus their expected objective values and variances.

## 2.6 Run a worked example

The repository ships two end-to-end examples that talk to a running server
and produce plots:

```bash
python .\examples\http_visualize_branin.py
python .\examples\http_chem_pca_optimize_visualize.py
```

The first draws a 3-D surface of the two-input Branin function; the second
runs an optimization in PCA(2) space, overlays training points (blue) with
predictions (red), and saves the figures under `examples/responses/`.

## 2.7 Next steps

- [§5 API reference](05-api-reference.md) — full request and response schemas.
- [§6 Objectives & constraints](06-objectives-and-constraints.md) — goal types beyond `min`/`max`.
- [§3 Installation](03-installation.md) — GPU installs and Docker.
