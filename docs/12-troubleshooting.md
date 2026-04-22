---
title: Troubleshooting
sidebar_position: 12
slug: /troubleshooting
---

# 12. Troubleshooting

This chapter catalogs known limitations and the errors you are most likely
to hit in practice, together with their root cause and recommended fix.
When in doubt, enable `LOG_LEVEL=debug` and re-run the failing request — the
logs often contain the specific NumPy / PyTorch / BoTorch message that
pinpoints the issue.

---

## 12.1 Known limitations

- **PCA maps** are returned only when `use_pca=true` and
  `pca_dimension ∈ {2, 3}`.
- **Input-space maps** require at least two continuous parameters;
  categorical-only search spaces cannot be plotted.
- **Linear constraints in PCA space** — because PCA indices do not align with
  the original columns, sum/ratio inequalities are not passed to
  `optimize_acqf` when optimizing in PCA space. They are instead enforced
  post-hoc during reconstruction (§9).
- **Long-running jobs** are bounded by `SATOR_JOB_TIMEOUT_SEC`. Timed-out
  jobs fail explicitly; there is no fallback acquisition path.
- **Horizontal scaling** requires sticky routing. Job state lives in the
  submitting process; see [§10.7 Scaling](10-operations.md#107-scaling).

## 12.2 Common HTTP responses

| Status | Likely cause | Fix |
|---|---|---|
| `401 Unauthorized` | Missing or wrong `x-api-key` header. | Ensure the header matches `SATOR_API_KEY`. |
| `403 Forbidden` | Source IP not on `SATOR_IP_WHITELIST` or is on `SATOR_IP_BLACKLIST`. | Adjust allow/deny lists, or verify `SATOR_TRUSTED_PROXY_CIDRS` when behind a proxy. |
| `409 Conflict` | `idempotency-key` replayed with a different body. | Pick a fresh UUID per logical request. |
| `422 Unprocessable Entity` | Request failed Pydantic validation. | Read `detail` for the offending field and fix the payload. |
| `429 Too Many Requests` | Per-(key, IP) rate limit hit. | Raise `SATOR_RATE_LIMIT_PER_MIN` or throttle the client. |
| `500 Internal Server Error` | Unhandled exception. | Check server logs. Set `SATOR_EXPOSE_ERROR_DETAILS=true` *in dev only* to see the exception text in the body. |

## 12.3 Job-level errors

### `FAILED` with `"timeout"`

Job wall time exceeded `SATOR_JOB_TIMEOUT_SEC`. Options:

- Raise the timeout.
- Reduce `batch_size` or `max_evaluations`.
- Lower `mc_samples`, `num_restarts`, or `raw_samples` in `advanced`
  (see [§8.2](08-advanced-tuning.md#82-acquisition)).
- Switch `acquisition` from `qnehvi` to `qehvi` when noise is small.

### `FAILED` with a `cholesky` / `not positive definite` message

The GP Cholesky factorization failed. Causes and fixes:

| Cause | Fix |
|---|---|
| Near-duplicate rows in `X`. | Add small jitter to duplicates or drop them. |
| Tiny input ranges. | Enable `parameter_scaling=standardize` or `minmax`. |
| Extremely small noise. | Let `noise="auto"` run, or raise `jitter` to `1e-6`. |
| Too-long lengthscale prior. | Let ARD run; do not freeze `lengthscale` prematurely. |

### `FAILED` with a SLSQP message from reconstruction

The sum/ratio constraints are infeasible for the given bounds, or the PCA
target lies far outside the feasible set. Double-check:

- `sum_target` is reachable given per-ingredient bounds.
- No ratio constraint contradicts bounds (e.g. `min_ratio=2` with `x_i ∈ [0,1]` and `x_j ∈ [0.5, 1]`).
- `target_precision` is not unreasonably tight (`1e-10` may be too strict
  for your data).

## 12.4 Installation errors

- **`ModuleNotFoundError: sator_os_engine`** — you installed runtime
  dependencies but forgot `pip install -e . --no-deps`. See
  [§3](03-installation.md).
- **`torch.cuda.is_available()` returns `False`** — driver, CUDA runtime,
  and PyTorch build are out of sync. Align them with
  `requirements-cuda.txt`’s `--extra-index-url` and pinned `torch` version.
- **Docker GPU cannot see the GPU** — install the NVIDIA Container Toolkit,
  and run with `--gpus all` (or Compose `gpus: all`). On Windows this also
  requires WSL2 with NVIDIA support.

## 12.5 TLS errors

- **`FileNotFoundError` on startup** — `SATOR_TLS_CERT_FILE` /
  `SATOR_TLS_KEY_FILE` point to missing files. Use absolute paths.
- **Browser says “not secure”** after `mkcert` setup — run `mkcert -install`
  again as administrator / root.
- **Reverse-proxy shows the wrong client IP** in logs — populate
  `SATOR_TRUSTED_PROXY_CIDRS` with your proxy’s address.

## 12.6 Where to look next

- Server logs (`LOG_LEVEL=debug` if needed).
- `GET /readyz` — if it reports not-ready, the process knows it is unhealthy.
- `GET /metrics` — surface-level counters often reveal a spike of failed
  jobs or throttled requests.
- [§10 Operations](10-operations.md) for runtime/environmental issues.
- [§8 Advanced tuning](08-advanced-tuning.md) for optimizer-performance issues.

## 12.7 Filing a bug

Please open an issue in the repository with:

1. A minimal JSON payload that reproduces the problem.
2. The server log excerpt at `LOG_LEVEL=debug`.
3. Environment details (`SATOR_DEVICE`, OS, Python version, Docker image tag).

Release notes live in [`CHANGELOG.md`](../CHANGELOG.md); check the latest
entries before filing.
