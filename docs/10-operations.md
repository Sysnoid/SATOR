---
title: Operations
sidebar_position: 10
slug: /operations
---

# 10. Operations

This chapter is written for whoever deploys and runs SATOR in production or
on a shared development host. For one-machine installs, skim §10.1, §10.3,
and §10.4; the rest covers TLS, reverse-proxy setup, scaling, and metrics.

---

## 10.1 Process lifecycle

SATOR runs as a single Uvicorn process. Nothing is written to disk besides
logs (if enabled); all state is in-process and expires automatically.

| Responsibility | Mechanism |
|---|---|
| HTTP serving | Uvicorn, `asyncio` |
| Concurrency cap | `asyncio.Semaphore` with `SATOR_CONCURRENCY` slots |
| Per-job timeout | `asyncio.wait_for`, `SATOR_JOB_TIMEOUT_SEC` |
| Job TTL / cleanup | Background sweep every `SATOR_STORE_SWEEP_INTERVAL_SEC` |
| Rate limiting | In-memory sliding window, capped at `SATOR_RATE_LIMIT_MAX_KEYS` entries |
| Idempotency | In-memory map keyed on `idempotency-key` header |

Recommended systemd unit outline (Linux):

```ini
[Unit]
Description=SATOR OS Engine
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/sator
EnvironmentFile=/opt/sator/.env
ExecStart=/opt/sator/.venv/bin/sator-server
Restart=always
RestartSec=5
User=sator
Group=sator

[Install]
WantedBy=multi-user.target
```

## 10.2 Running the server

Three equivalent ways to start the process:

```bash
# entry point
sator-server

# module form
python -m sator_os_engine.server.main

# Docker
docker compose up                  # CPU
docker compose --profile gpu up    # GPU
```

All three read `.env` from the working directory by default.

## 10.3 Health endpoints

| Endpoint | Purpose | Expected on success |
|---|---|---|
| `GET /livez` | Liveness — is the process responding? | `200 {"status":"ok"}` |
| `GET /readyz` | Readiness — can the process accept work? | `200 {"status":"ok"}` |

Both endpoints are unauthenticated so they can be probed by load balancers
and orchestrators without distributing the API key. They are safe to expose
inside a private network; do **not** expose them directly to the public
internet without at least a reverse proxy in front.

## 10.4 Metrics

When `SATOR_ENABLE_METRICS=true` (the default), the server exposes:

```
GET /metrics
```

in Prometheus text format. The payload includes:

- HTTP-request counters & histograms (by route and status).
- Job lifecycle gauges (queued, running, succeeded, failed).
- Store sizes (idempotency map, rate-limit map).
- Process metrics (CPU, RSS, open file descriptors).

**The endpoint is unauthenticated.** Expose it inside your cluster only, or
disable it on untrusted networks:

```dotenv
SATOR_ENABLE_METRICS=false
```

If exposure is necessary, put it behind a reverse proxy that rejects
requests not originating from Prometheus.

## 10.5 TLS

SATOR can either terminate TLS itself or sit behind a reverse proxy that
does. **The recommended production layout is the proxy.**

### 10.5.1 Direct TLS (Uvicorn)

Enable with:

```dotenv
SATOR_ENABLE_TLS=true
SATOR_HTTP_PORT=8443
SATOR_TLS_CERT_FILE=/path/to/cert.pem
SATOR_TLS_KEY_FILE=/path/to/key.pem
# optional:
# SATOR_TLS_KEY_PASSWORD=...
# SATOR_TLS_CA_CERTS=/path/to/ca-bundle.pem
```

Use this for development and for simple single-host deployments. For local
certificates see [§11 Local HTTPS setup](11-local-https-setup.md).

### 10.5.2 Reverse-proxy TLS (recommended)

Terminate TLS at the proxy and forward to the engine over plain HTTP on
`127.0.0.1`.

**Caddy:**

```caddy
your.domain.com {
    reverse_proxy 127.0.0.1:8080
}
```

**NGINX:**

```nginx
server {
    listen 443 ssl http2;
    server_name your.domain.com;

    ssl_certificate     /etc/letsencrypt/live/your.domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your.domain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host              $host;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
    }
}
```

Bind the engine to `127.0.0.1` via `SATOR_HTTP_HOST=127.0.0.1` so it is not
reachable from outside the host directly.

### 10.5.3 Trusting `X-Forwarded-For`

Only honor `X-Forwarded-For` when the direct peer is a known proxy. Set:

```dotenv
SATOR_TRUSTED_PROXY_CIDRS=127.0.0.1,10.0.0.0/8
```

Leave empty when no proxy is in place; otherwise clients could spoof their
source IP for rate-limiting or allow/deny purposes.

## 10.6 Security posture

| Control | Configuration |
|---|---|
| API key auth | `SATOR_API_KEY` |
| Per-(key, IP) rate limit | `SATOR_RATE_LIMIT_PER_MIN` |
| IP allow / deny | `SATOR_IP_WHITELIST`, `SATOR_IP_BLACKLIST` |
| Transport security | `SATOR_ENABLE_TLS` or reverse-proxy TLS |
| Safe error messages | `SATOR_EXPOSE_ERROR_DETAILS=false` in prod |
| Idempotency | `idempotency-key` header (client-provided) |

Rotate `SATOR_API_KEY` by redeploying the process with a new value; there
is no built-in key-list mechanism.

## 10.7 Scaling

SATOR is stateless across process restarts (results live only in memory). To
scale horizontally:

1. Run `N` instances behind a reverse proxy or load balancer.
2. Configure **sticky sessions by job id** (or ensure the client polls the
   same instance that submitted the job, e.g. via a `job_id → instance` map
   returned to the client).
3. Size `SATOR_CONCURRENCY` so that peak simultaneous jobs × average per-job
   CPU/GPU usage stays well below your host’s capacity.

Because the job store is in-process, sticky routing is required for correct
polling behavior.

## 10.8 Logging

| Knob | Values |
|---|---|
| `LOG_LEVEL`  | `debug` \| `info` \| `warning` \| `error` |
| `LOG_FORMAT` | `human` \| `json` |
| `LOG_TO_FILE`| `true` to also write to file |
| `LOG_FILE_PATH` | Path when `LOG_TO_FILE=true` |

Use `LOG_FORMAT=json` with any structured-log aggregator (Loki, ELK, Splunk,
Cloud Logging, etc.). Every log line carries a `request_id` when it
originated from an HTTP handler.

## 10.9 Backups and persistence

There is nothing to back up. SATOR is intentionally stateless; the only
persistence expectations are:

- The `.env` file (or your secret manager).
- Cert/key files when using direct TLS.

Client-side systems are responsible for persisting optimization histories.
