# Contributing

Thanks for your interest in contributing! Please:

1. Discuss big changes in an issue first.
2. Fork the repo and create a feature branch.
3. Add tests for new behavior; keep coverage for core logic.
4. Run unit tests locally: `pytest -q tests/unit` (or `pytest` for the full suite).
5. Run **Ruff** before pushing: `ruff check sator_os_engine tests` and `ruff format sator_os_engine tests` (install via `pip install -e ".[dev]"`). Optionally run **Vulture** for dead code: `vulture sator_os_engine --min-confidence 100`.
6. Follow the code style in this repo.
7. Update docs and CHANGELOG when user-facing behavior changes.

## Development quickstart
- Create and activate a virtualenv, then from the repo root: `pip install -e ".[dev]"` (or `pip install -r requirements.txt` and `pip install -e .`). See [`docs/03-installation.md`](docs/03-installation.md) for CPU vs GPU installs and containers.
- Run the server locally: `sator-server`.
- Examples live in `examples/`; tests in `tests/`.

## Licensing of contributions
- By submitting a contribution, you agree to license your work under the
  repository’s license (Apache‑2.0). This ensures a uniform, permissive
  license for the whole project.

## DCO (Developer Certificate of Origin) sign‑off (optional)
- If your company requires it, add a `Signed-off-by: Your Name <email>` line to
  your commit messages (`git commit -s`). We will accept contributions with or
  without DCO, but enabling it may help your internal compliance.

## Reporting vulnerabilities
Please see `SECURITY.md` for how to report security issues.

