from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=".env",
        env_nested_delimiter=",",
        # Blank lines in .env (e.g. SATOR_IP_WHITELIST=) must not be JSON-decoded as List fields.
        env_ignore_empty=True,
    )
    # Auth & security (single server API key)
    api_key: str | None = Field(default=None, alias="SATOR_API_KEY")
    rate_limit_per_min: int = Field(300, alias="SATOR_RATE_LIMIT_PER_MIN")
    ip_whitelist: list[str] = Field(default_factory=list, alias="SATOR_IP_WHITELIST")
    ip_blacklist: list[str] = Field(default_factory=list, alias="SATOR_IP_BLACKLIST")
    # Comma-separated list of IPs or CIDRs. When the direct peer matches, X-Forwarded-For is trusted.
    trusted_proxy_cidrs: str = Field(default="", alias="SATOR_TRUSTED_PROXY_CIDRS")

    # Transports
    enable_http: bool = Field(True, alias="SATOR_ENABLE_HTTP")
    http_host: str = Field("0.0.0.0", alias="SATOR_HTTP_HOST")
    http_port: int = Field(8080, alias="SATOR_HTTP_PORT")
    # TLS (HTTPS)
    enable_tls: bool = Field(False, alias="SATOR_ENABLE_TLS")
    tls_cert_file: str | None = Field(default=None, alias="SATOR_TLS_CERT_FILE")
    tls_key_file: str | None = Field(default=None, alias="SATOR_TLS_KEY_FILE")
    tls_key_password: str | None = Field(default=None, alias="SATOR_TLS_KEY_PASSWORD")
    tls_ca_certs: str | None = Field(default=None, alias="SATOR_TLS_CA_CERTS")

    # Logging
    log_level: str = Field("info", alias="LOG_LEVEL")
    log_format: str = Field("human", alias="LOG_FORMAT")  # json|human
    log_to_file: bool = Field(False, alias="LOG_TO_FILE")
    log_file_path: str | None = Field(None, alias="LOG_FILE_PATH")

    # Device
    device: str = Field("cpu", alias="SATOR_DEVICE")  # cpu|cuda
    cuda_device: int = Field(0, alias="SATOR_CUDA_DEVICE")

    # Runtime
    job_ttl_sec: int = Field(600, alias="SATOR_RESULT_TTL_SEC")
    job_timeout_sec: int = Field(300, alias="SATOR_JOB_TIMEOUT_SEC")
    concurrency: int = Field(4, alias="SATOR_CONCURRENCY")
    # When true, 500 JSON responses include exception text (use only in dev; avoid leaking stack traces in prod)
    expose_error_details: bool = Field(False, alias="SATOR_EXPOSE_ERROR_DETAILS")

    # Metrics (Prometheus)
    enable_metrics: bool = Field(True, alias="SATOR_ENABLE_METRICS")
    # How often in-process sweeps run for idempotency / rate limiter / job store hygiene
    store_sweep_interval_sec: int = Field(60, ge=5, alias="SATOR_STORE_SWEEP_INTERVAL_SEC")
    # Hard cap on rate-limiter (api_key, ip) map entries (after per-minute window trim, extras dropped)
    rate_limit_max_keys: int = Field(50_000, ge=1000, alias="SATOR_RATE_LIMIT_MAX_KEYS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
