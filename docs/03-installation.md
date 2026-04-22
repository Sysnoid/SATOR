---
title: Installation
sidebar_position: 3
slug: /installation
---

# 3. Installation

SATOR supports two installation modes and two hardware targets:

|           | CPU                                  | NVIDIA GPU                                  |
|-----------|--------------------------------------|---------------------------------------------|
| **Local** | [§3.1 Local CPU](#31-local-cpu)      | [§3.2 Local GPU](#32-local-gpu-nvidia-cuda) |
| **Docker**| [§3.3 Docker CPU](#33-docker-cpu)    | [§3.4 Docker GPU](#34-docker-gpu-nvidia)    |

Runtime dependencies are declared in [`pyproject.toml`](../pyproject.toml) and
mirrored in [`requirements.txt`](../requirements.txt). **CUDA-enabled PyTorch**
is not a standard PyPI extra: its wheels live on the
[PyTorch download index](https://download.pytorch.org/whl/) and are pulled in
through [`requirements-cuda.txt`](../requirements-cuda.txt) or the CUDA Docker
image.

---

## 3.1 Local CPU

From the repository root:

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev]"
```

Optional: to use the smaller PyTorch CPU-only wheel explicitly,

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

Set `SATOR_DEVICE=cpu` in `.env` (this is the default).

## 3.2 Local GPU (NVIDIA CUDA)

1. Install an NVIDIA driver and CUDA runtime that match the PyTorch build you
   intend to install. The shipped [`requirements-cuda.txt`](../requirements-cuda.txt)
   currently pins `torch==2.11.0+cu128`; update in lock-step if you change it.
2. Create a fresh virtualenv and install:
   ```bash
   pip install --upgrade pip
   pip install -r requirements-cuda.txt
   pip install -e . --no-deps
   ```
3. In `.env`, set:
   ```dotenv
   SATOR_DEVICE=cuda
   SATOR_CUDA_DEVICE=0
   ```

Verify the GPU is visible to PyTorch:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## 3.3 Docker CPU

Build and run the CPU image from the repository root:

```bash
docker build -t sator:cpu -f Dockerfile .
docker run --rm -p 8080:8080 -e SATOR_API_KEY=your-key sator:cpu
```

Or use Docker Compose (CPU service is the default, no profile needed):

```bash
docker compose up --build
```

[`Dockerfile`](../Dockerfile) installs `requirements.txt` with the
**PyTorch CPU wheel index** and then adds the SATOR package via
`pip install -e . --no-deps`.

## 3.4 Docker GPU (NVIDIA)

**Host prerequisite:** the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
so `docker run --gpus all` (or Compose `gpus: all`) works.

Direct docker:

```bash
docker build -t sator:gpu -f Dockerfile.cuda .
docker run --rm --gpus all -p 8080:8080 \
  -e SATOR_API_KEY=your-key \
  -e SATOR_DEVICE=cuda \
  sator:gpu
```

Compose (GPU service is behind the `gpu` profile and maps host port **8081**
to the container’s `8080` by default, so both CPU and GPU services can run
together on one host):

```bash
docker compose --profile gpu up --build
```

[`Dockerfile.cuda`](../Dockerfile.cuda) uses
`nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` and installs
`requirements-cuda.txt` so package versions stay aligned with the repository’s
CUDA pin.

## 3.5 Environment file for Compose

Docker Compose expects a `.env` at the repository root. Copy the example:

```bash
cp .env.example .env
# edit SATOR_API_KEY and any other values
```

All variables documented in [§4 Configuration](04-configuration.md) apply
identically inside and outside the container.

## 3.6 Host ports (Compose)

| Service    | Host port variable       | Default |
|------------|--------------------------|---------|
| `sator-cpu`| `SATOR_HOST_PORT`        | `8080`  |
| `sator-gpu`| `SATOR_GPU_HOST_PORT`    | `8081`  |

Both map to the container’s internal `8080`.

## 3.7 Verifying the install

With the server running on `http://localhost:8080`:

```bash
curl http://localhost:8080/livez         # liveness
curl http://localhost:8080/readyz        # readiness
curl -H "x-api-key: dev-key" http://localhost:8080/metrics | head   # Prometheus (if enabled)
```

If `/readyz` returns non-OK, consult [§12 Troubleshooting](12-troubleshooting.md).

## 3.8 Troubleshooting installs

- **GPU container does not see the GPU** — the host lacks the NVIDIA driver or
  Container Toolkit; or the container was not launched with `--gpus all` /
  Compose `gpus: all`. On Windows, GPU containers require WSL2 with NVIDIA
  support.
- **Torch / CUDA version mismatch** — bump `requirements-cuda.txt` *and* the
  base image in `Dockerfile.cuda` together, then rebuild with `--no-cache` when
  changing the CUDA major version.
- **`ModuleNotFoundError: sator_os_engine`** — you installed the runtime
  dependencies but forgot `pip install -e . --no-deps` after them.
