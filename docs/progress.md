# Progress log

Results are reported as **PASSED / FAILED / SKIPPED / NOT RUN**, exactly as
observed. Nothing is listed as passed unless it was executed. Newest first.

## 2026-10-16 — M1.1: RMSNorm

### What was done
- `src/qwen3_scratch/layers/rmsnorm.py`: `RMSNorm` (`nn.Module`, `weight` of shape
  `(d,)` initialised to ones), the Qwen3/HF-compatible functional `rms_norm`, and
  the educational `rms_norm_textbook`.
- Tests: `tests/test_rmsnorm.py` (101 cases), `tests/parity/test_rmsnorm_hf.py`
  (13 cases, needs `transformers`), `tests/test_notebooks.py` (executes notebooks).
- `notebooks/01_rmsnorm.ipynb`, `docs/components/rmsnorm.md` (derivation, worked
  example, gradients, mixed precision, Qwen3 usage, C++ port requirements),
  `scripts/rmsnorm_report.py` (error report).
- Decisions D-013 to D-016. CI now installs `.[dev,hf]` so parity tests run there.
- Verified in the HF source: `Qwen3RMSNorm` initialises `weight` to ones and has no
  `1 + w` offset. It computes statistics in float32 and casts back before scaling.

### Where validation ran
Same assistant sandbox as M0 (Ubuntu 24.04, Python 3.11.17 venv, torch
`2.14.1+cu130` running on CPU), plus `transformers 5.20.0` for the parity tests.
**Not** run on the maintainer's Fedora machine.

| Check | Result |
|---|---|
| `python scripts/check_env.py` | PASSED |
| `python -m ruff check .` / `python -m ruff format --check .` | PASSED / PASSED (29 files) |
| `python -m pytest` with `transformers` | PASSED: 149 passed, 2 skipped (gpu, weights) |
| `python -m pytest` without `transformers` (import simulated as missing) | PASSED: 136 passed, 3 skipped (+ HF parity module) |
| `tests/test_rmsnorm.py` | PASSED: 101 passed |
| `tests/parity/test_rmsnorm_hf.py` | PASSED: 13 passed; forward and backward **bit-identical** to HF in fp32/bf16/fp16 |
| `gradcheck` + `gradgradcheck` (float64) | PASSED |
| Autograd vs closed-form gradient (float64) | PASSED (`rtol=atol=1e-12`); notebook: max diff `4.4e-16` (dx), `1.8e-15` (dγ) |
| Notebook `01_rmsnorm.ipynb` executed with `nbclient` (Jupyter kernel) | PASSED, all cells ran; committed without outputs |
| Notebook code cells via `tests/test_notebooks.py` | PASSED |
| C++ build + CTest (unchanged) | PASSED |
| Same checks on Fedora (`qwen3-scratch` env) | NOT RUN by the assistant |

### Numerical comparison (`python scripts/rmsnorm_report.py`)
Inputs `2·N(0,1)`, weights `1 + 0.2·N(0,1)`, seed 0, ε = 1e-6.
max_rel uses elements with `|ref| ≥ 1e-3`.

| tensor | dtype | ours vs | max_abs | max_rel | elements differing |
|---|---|---|---|---|---|
| hidden (4,128,1024) | float32 | textbook float64 | 6.916e-07 | 2.234e-07 | 100.0% |
| hidden (4,128,1024) | float32 | torch.nn.RMSNorm | 0 | 0 | 0.0% |
| hidden (4,128,1024) | float32 | HF Qwen3RMSNorm | 0 | 0 | 0.0% |
| hidden (4,128,1024) | bfloat16 | textbook float64 | 2.788e-02 | 7.359e-03 | 100.0% |
| hidden (4,128,1024) | bfloat16 | torch.nn.RMSNorm | 3.125e-02 | 7.812e-03 | 25.5% |
| hidden (4,128,1024) | bfloat16 | HF Qwen3RMSNorm | 0 | 0 | 0.0% |
| hidden (4,128,1024) | bfloat16 | scale-then-cast order | 3.125e-02 | 7.812e-03 | 25.5% |
| QK head (4,16,128,128) | float32 | textbook float64 | 6.854e-07 | 2.377e-07 | 100.0% |
| QK head (4,16,128,128) | float32 | torch.nn.RMSNorm | 0 | 0 | 0.0% |
| QK head (4,16,128,128) | float32 | HF Qwen3RMSNorm | 0 | 0 | 0.0% |
| QK head (4,16,128,128) | bfloat16 | textbook float64 | 2.615e-02 | 7.475e-03 | 100.0% |
| QK head (4,16,128,128) | bfloat16 | torch.nn.RMSNorm | 3.125e-02 | 7.812e-03 | 25.9% |
| QK head (4,16,128,128) | bfloat16 | HF Qwen3RMSNorm | 0 | 0 | 0.0% |
| QK head (4,16,128,128) | bfloat16 | scale-then-cast order | 3.125e-02 | 7.812e-03 | 25.9% |

`7.8e-3 = 2⁻⁷` is one bfloat16 ulp. In this torch version, `torch.nn.RMSNorm` in
bf16 is bit-identical to the scale-then-cast order.

### Open items
- Run on Fedora: `python -m pytest`. Optionally install `pip install -e ".[hf]"`
  first so the 13 HF parity tests run instead of being skipped.
- Pending from M0: confirm GitHub Actions is green after pushing.

### Next
M1.2 — SwiGLU FeedForward (awaiting approval).

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
