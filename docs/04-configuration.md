---
title: Configuration
sidebar_position: 4
slug: /configuration
---

# 4. Configuration

SATOR reads all configuration from **environment variables**, optionally loaded
from a `.env` file at the server’s working directory. Every variable has a
sensible default; the only variable you *must* set in production is
`SATOR_API_KEY`.

A ready-to-edit template lives in [`.env.example`](../.env.example).

---

## 4.1 Quick reference

| Variable | Default | Purpose |
|---|---|---|
| `SATOR_API_KEY` | *(unset)* | Single API key required in the `x-api-key` header |
| `SATOR_ENABLE_HTTP` | `true` | Enable the HTTP transport (always true for now) |
| `SATOR_HTTP_HOST` | `0.0.0.0` | Bind address |
| `SATOR_HTTP_PORT` | `8080` | Bind port (use `8443` when TLS is on) |
| `SATOR_ENABLE_TLS` | `false` | Terminate TLS inside Uvicorn |
| `SATOR_TLS_CERT_FILE` | — | PEM certificate (required when TLS is on) |
| `SATOR_TLS_KEY_FILE` | — | Private key (required when TLS is on) |
| `SATOR_TLS_KEY_PASSWORD` | — | Optional key password |
| `SATOR_TLS_CA_CERTS` | — | Optional CA bundle for client-cert verification |
| `SATOR_DEVICE` | `cpu` | `cpu` or `cuda` |
| `SATOR_CUDA_DEVICE` | `0` | CUDA device index when `SATOR_DEVICE=cuda` |
| `SATOR_RATE_LIMIT_PER_MIN` | `300` | Per-(api-key, IP) request-per-minute cap |
| `SATOR_IP_WHITELIST` | *(empty)* | Comma-separated allow list of CIDRs/IPs |
| `SATOR_IP_BLACKLIST` | *(empty)* | Comma-separated deny list of CIDRs/IPs |
| `SATOR_TRUSTED_PROXY_CIDRS` | *(empty)* | CIDRs whose `X-Forwarded-For` is trusted |
| `SATOR_RESULT_TTL_SEC` | `600` | Seconds a terminal job result is retained |
| `SATOR_JOB_TIMEOUT_SEC` | `300` | Maximum wall time for a single job |
| `SATOR_CONCURRENCY` | `4` | Max concurrent running jobs |
| `SATOR_ENABLE_METRICS` | `true` | Expose `GET /metrics` in Prometheus format |
| `SATOR_STORE_SWEEP_INTERVAL_SEC` | `60` | Interval for in-process store hygiene sweeps |
| `SATOR_RATE_LIMIT_MAX_KEYS` | `50000` | Hard cap on rate-limiter key table size |
| `SATOR_EXPOSE_ERROR_DETAILS` | `false` | Include exception text in 500 responses (dev only) |
| `LOG_LEVEL` | `info` | `debug`\|`info`\|`warning`\|`error` |
| `LOG_FORMAT` | `human` | `human` or `json` |
| `LOG_TO_FILE` | `false` | Also write logs to a file |
| `LOG_FILE_PATH` | *(empty)* | Log file path when `LOG_TO_FILE=true` |

---

## 4.2 Authentication

### `SATOR_API_KEY`

A single API key used for **all** clients. Required on every request via the
`x-api-key` header. Unset or empty means every request is rejected with
`401 Unauthorized`.

For multi-tenant scenarios, run multiple instances with different keys behind
a reverse proxy and route by path, subdomain, or header.

## 4.3 Networking

### HTTP bind
- `SATOR_ENABLE_HTTP` must remain `true`.
- `SATOR_HTTP_HOST` / `SATOR_HTTP_PORT` — use `0.0.0.0:8080` when fronted by a
  reverse proxy on the same host; use `127.0.0.1:8080` if the proxy is
  co-located and you want to prevent public binding.

### TLS (HTTPS)

Uvicorn terminates TLS directly when `SATOR_ENABLE_TLS=true` *and* both
certificate and key files are provided. See
[§11 Local HTTPS setup](11-local-https-setup.md) for the full procedure
with `mkcert`, and [§10 Operations](10-operations.md) for the production
recommendation of terminating TLS at a reverse proxy.

### Port conventions
- `SATOR_HTTP_PORT=8080` → plain HTTP (`SATOR_ENABLE_TLS=false`).
- `SATOR_HTTP_PORT=8443` → HTTPS (`SATOR_ENABLE_TLS=true`).

### IP policy
- `SATOR_IP_WHITELIST` — if non-empty, only sources matching any CIDR/IP are
  accepted.
- `SATOR_IP_BLACKLIST` — sources matching any entry are rejected.
- `SATOR_TRUSTED_PROXY_CIDRS` — when the direct peer matches, the first
  `X-Forwarded-For` entry is used as the client IP for rate limiting and
  allow/deny checks. Leave empty when not behind a proxy.

## 4.4 Compute device

- `SATOR_DEVICE=cpu` — PyTorch runs on CPU. Always available.
- `SATOR_DEVICE=cuda` — PyTorch runs on the GPU identified by
  `SATOR_CUDA_DEVICE`. Requires a CUDA-enabled PyTorch build and a visible
  driver (see [§3.2](03-installation.md#32-local-gpu-nvidia-cuda) or
  [§3.4](03-installation.md#34-docker-gpu-nvidia)).

## 4.5 Job runtime

- `SATOR_CONCURRENCY` — an `asyncio.Semaphore` bound to this value gates the
  number of jobs executing simultaneously. The rest queue with status
  `QUEUED`.
- `SATOR_JOB_TIMEOUT_SEC` — jobs exceeding this limit are cancelled and marked
  `FAILED` with a timeout error. No fallbacks are attempted.
- `SATOR_RESULT_TTL_SEC` — terminal (`SUCCEEDED` / `FAILED`) jobs are pruned
  this many seconds after completion so results do not accumulate in memory.

## 4.6 Rate limiting

- `SATOR_RATE_LIMIT_PER_MIN` — sliding-window cap on requests per
  (API key, client IP) tuple. Returns `429 Too Many Requests` when exceeded.
- `SATOR_RATE_LIMIT_MAX_KEYS` — hard upper bound on the in-memory table; when
  exceeded, least-recently-used entries are dropped.

## 4.7 Observability

- `LOG_LEVEL`, `LOG_FORMAT`, `LOG_TO_FILE`, `LOG_FILE_PATH` — standard logging
  knobs. JSON format is recommended for log aggregators.
- `SATOR_ENABLE_METRICS=true` exposes `GET /metrics` in
  [Prometheus text format](https://prometheus.io/docs/concepts/data_model/).
  The endpoint is **unauthenticated**; protect it at the network layer or
  disable it on untrusted networks. See [§10 Operations](10-operations.md).

## 4.8 Store hygiene

- `SATOR_STORE_SWEEP_INTERVAL_SEC` — background task period for pruning
  idempotency records, expired rate-limit keys, and timed-out jobs.

## 4.9 Precedence

Values are resolved in this order (first wins):
1. Real environment variables (set in the shell, container, or systemd unit).
2. Variables loaded from `.env` in the current working directory.
3. Built-in defaults listed in [§4.1 Quick reference](#41-quick-reference).

Empty strings are treated as *unset* so you can leave fields blank in
`.env` without triggering JSON parsing errors on list-typed variables.
