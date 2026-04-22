# SATOR — CPU (PyTorch CPU wheels; smaller and no NVIDIA driver on host)
# Build:  docker build -t sator:cpu .
# Run:     docker run --rm -p 8080:8080 -e SATOR_API_KEY=dev-key sator:cpu
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY sator_os_engine ./sator_os_engine

# CPU-only PyTorch wheels (smaller than default PyPI torch on many platforms)
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt \
    && pip install -e . --no-deps

ENV SATOR_HTTP_HOST=0.0.0.0 \
    SATOR_HTTP_PORT=8080 \
    SATOR_DEVICE=cpu

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/livez', timeout=3)"

CMD ["sator-server"]
