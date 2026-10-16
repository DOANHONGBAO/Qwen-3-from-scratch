# Progress log

Results are reported as **PASSED / FAILED / SKIPPED / NOT RUN**, exactly as
observed. Nothing is listed as passed unless it was executed.

## 2026-10-16 — M0: project foundation

### What was done
- Analysed references R1–R7 (see `docs/references.md`), including 12 recorded
  discrepancies between the educational and official Qwen3 implementations.
- Created the repository layout, packaging (`pyproject.toml`, `environment.yml`),
  `Qwen3Config` with presets and analytic parameter count, environment check
  script, three-tier pytest setup, Ruff config, C++20 CMake/CTest scaffold
  mirroring the config, GitHub Actions CI, Kaggle templates, and documentation.
- No model component implemented (by design). No weights downloaded.

### Where validation ran
The checks below ran in an **isolated Linux sandbox used by the coding assistant**
(Ubuntu 24.04, x86_64, 2 vCPU, 7.8 GiB RAM, no GPU), **not** on the maintainer's
Fedora machine and **not** inside the `qwen3-scratch` conda environment (conda
was not available there). Python 3.11.17 in a `venv`; torch `2.14.1+cu130` from
PyPI (the CUDA build, running on CPU, because the PyTorch CPU index was not
reachable from that sandbox); numpy 2.4.6, pytest 9.1.1, ruff 0.17.1; CMake
3.28.3, GCC 13.3.0, Ninja.

| Check | Command | Result |
|---|---|---|
| Ruff lint | `ruff check .` | PASSED (all checks passed) |
| Ruff format | `ruff format --check .` | PASSED (20 files formatted) |
| Pytest, CPU tier | `pytest` | PASSED — 33 passed, 2 skipped |
| GPU tier | `pytest -m gpu` | SKIPPED (no CUDA device) |
| Weights tier, real checkpoint | `QWEN3_WEIGHTS_DIR=... pytest -m weights` | NOT RUN (no weights downloaded, by design) |
| Weights tier wiring | same, with a directory containing only the official `config.json` | PASSED (1 passed) — verifies skip/run logic only |
| Environment script | `python scripts/check_env.py` | PASSED (exit 0; conda env reported as INFO: not active) |
| C++ build + CTest | `bash scripts/build_cpp.sh` | PASSED — 1/1 tests (`config.presets`), no compiler warnings with `-Wall -Wextra -Wpedantic -Wconversion` |
| Kaggle smoke script, CPU dry run | `python kaggle/smoke_test/kaggle_smoke_test.py` | PASSED (CPU only) |
| Kaggle kernel rendering | `KAGGLE_USERNAME=demo-user bash kaggle/prepare_kernel.sh smoke_test` | PASSED (rendered, not pushed) |
| Kaggle auth check | `bash scripts/kaggle_check_auth.sh` | NOT RUN meaningfully — Kaggle CLI not installed (exit 2), no credentials, by design |
| Kaggle GPU kernel | `kaggle kernels push ...` | NOT RUN (no explicit experimental objective) |
| Notebook `00_environment_and_config` | code cells executed as a script | PASSED |
| Shell scripts syntax | `bash -n` | PASSED (4 scripts) |
| CI workflow / environment.yml syntax | YAML parse | PASSED |
| `conda env create -f environment.yml` on Fedora | — | NOT RUN (must be run by the maintainer) |
| GitHub Actions on `main` | — | NOT RUN (pending first push) |

### Blockers / open items
- Conda environment must be created on the Fedora machine: `bash scripts/setup_env.sh`.
- Push to GitHub requires the maintainer's credentials (see the M0 report).
- Verify Kaggle `kernel-metadata.json` fields with `kaggle kernels init` once the CLI is installed.

### Next
M1.1 — RMSNorm (awaiting approval).
