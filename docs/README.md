---
title: SATOR Documentation
sidebar_position: 0
slug: /
---

# SATOR OS Engine — Documentation

Welcome to the official documentation for the **SATOR OS Engine**: a stateless,
production-ready optimization server for black-box, multi-objective problems.
SATOR exposes a single HTTP API and runs either locally or in Docker, on CPU
or NVIDIA GPU.

> **Version:** `0.2.0` &nbsp;·&nbsp; **API:** HTTP/JSON &nbsp;·&nbsp; **License:** Apache-2.0

---

## Table of contents

### Getting started
1. [Overview](01-overview.md) — what SATOR is, core concepts, and architecture at a glance
2. [Quickstart](02-quickstart.md) — install, run, and issue your first request in under five minutes
3. [Installation](03-installation.md) — local and Docker installs, CPU and NVIDIA GPU

### Configuration & API
4. [Configuration](04-configuration.md) — the complete environment-variable reference
5. [API reference](05-api-reference.md) — endpoints, payload shapes, responses
   - See also: [`openapi.yaml`](openapi.yaml) — OpenAPI 3.1 machine-readable spec

### Concepts
6. [Objectives & constraints](06-objectives-and-constraints.md) — goal types, thresholds, ranges, sum and ratio constraints
7. [Optimization pipeline](07-optimization-pipeline.md) — end-to-end flow: preprocessing → GP → acquisition → feasibility → maps → reconstruction
8. [Advanced tuning](08-advanced-tuning.md) — GP kernels, noise, acquisition tuning, Monte-Carlo sampling
9. [Reconstruction](09-reconstruction.md) — SLSQP inverse mapping from PCA to the original variables

### Running in production
10. [Operations](10-operations.md) — TLS, reverse proxy, health endpoints, metrics, job runtime
11. [Local HTTPS setup](11-local-https-setup.md) — generating trusted dev certificates with `mkcert`
12. [Troubleshooting](12-troubleshooting.md) — known limitations and common errors

---

## Reading paths

| If you want to… | Start at |
|---|---|
| Understand what SATOR does | [01 Overview](01-overview.md) |
| Try it as fast as possible | [02 Quickstart](02-quickstart.md) |
| Install with GPU support | [03 Installation](03-installation.md) § GPU |
| Design an optimization request | [05 API reference](05-api-reference.md) + [06 Objectives & constraints](06-objectives-and-constraints.md) |
| Tune solver performance | [08 Advanced tuning](08-advanced-tuning.md) |
| Deploy behind a reverse proxy | [10 Operations](10-operations.md) |
| Diagnose an error | [12 Troubleshooting](12-troubleshooting.md) |

## Conventions used in these docs

- `code` is a literal (filename, environment variable, CLI token, JSON key).
- **Bold** marks an important term the first time it appears.
- JSON examples use the same field names as the API; unspecified fields fall back to defaults.
- All HTTP examples assume a base URL of `http://localhost:8080` (or `https://localhost:8443` when TLS is enabled) and an API key passed via the `x-api-key` header.

## Support

- Release notes live in [`CHANGELOG.md`](../CHANGELOG.md).
- Contribution guidelines live in [`CONTRIBUTING.md`](../CONTRIBUTING.md).
- Security contact is in [`SECURITY.md`](../SECURITY.md).
