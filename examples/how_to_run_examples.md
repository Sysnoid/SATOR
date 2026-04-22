# How to run the examples

This file is the short "setup and run" recipe that lives next to the
demo scripts. For the full, illustrated walkthrough of what each demo
does and the expected figures, see
[`docs/13-examples.md`](../docs/13-examples.md).

---

## 1. Start the server (in a separate terminal)

- Windows PowerShell
  - `./venv/Scripts/Activate.ps1`
  - `$env:SATOR_API_KEY = "dev-key"`
  - `python -m sator_os_engine.server.main`
- Linux / macOS
  - `source ./venv/bin/activate`
  - `export SATOR_API_KEY=dev-key`
  - `python -m sator_os_engine.server.main`

If the server runs elsewhere, set `SATOR_BASE_URL` (default is
`http://localhost:8080`). For the HTTPS option see
[`docs/11-local-https-setup.md`](../docs/11-local-https-setup.md) and
[`docs/10-operations.md`](../docs/10-operations.md).

## 2. Install plotting dependencies (first time only)

```powershell
pip install matplotlib scikit-learn httpx
```

## 3. Run a demo

The examples share a tiny helper (`examples/_common.py`) that submits
the request, polls the job, and saves both the request and result
JSON under `examples/responses/`. Because the helper sits next to the
demos, run them from the repo root so Python can resolve
`import _common`:

```powershell
python .\examples\demo_09_pharma_tablet_pca.py
```

Each demo that opens matplotlib windows can be rendered headlessly
(saving PNGs instead of popping windows) with the smoke wrapper:

```powershell
python .\examples\_render_smoke.py demo_09_pharma_tablet_pca
```

## 4. Configuration

- API key: `SATOR_API_KEY` (defaults to `dev-key`)
- Server address: `SATOR_BASE_URL` (defaults to `http://localhost:8080`;
  use `https://localhost:8443` for TLS)

If you see HTTP 401/403, check the API key. If you see 422, print the
server error by temporarily adding `print(r.text)` before raising in
the script to view validation details.

## 5. What each demo does

See [§13 Examples](../docs/13-examples.md) for the full index, the
two flagship audit demos (`demo_09`, `demo_10`) with embedded
figures, and a short tour of the simpler demos `demo_01`–`demo_08`.
