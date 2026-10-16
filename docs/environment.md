# Environment and reproducibility

## Target machine

- Fedora Linux, x86_64, **CPU only** (no CUDA).
- Conda (Miniforge recommended), environment **`qwen3-scratch`**, Python **3.11**.
- Optional for M3: `cmake >= 3.20`, a C++20 compiler (GCC ≥ 11 or Clang ≥ 14), Ninja.

## Automated setup

```bash
bash scripts/setup_env.sh
```

Creates (or updates) the env from `environment.yml`, installs the package in
editable mode, then runs `check_env.py`, Ruff and pytest. No sudo, no model download.

## Manual setup

```bash
conda env create -f environment.yml        # python 3.11 + CPU torch + numpy + pytest + ruff + -e .
conda activate qwen3-scratch
python scripts/check_env.py
pytest
ruff check . && ruff format --check .
```

Without the yml (equivalent):

```bash
conda create -n qwen3-scratch -c conda-forge python=3.11 pip
conda activate qwen3-scratch
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

Why the PyTorch CPU index: the default PyPI wheel of torch for Linux x86_64
depends on the CUDA runtime packages (several GB of `nvidia-*` wheels). The CPU
wheel from `download.pytorch.org/whl/cpu` is far smaller and is all a CPU-only
machine needs.

## Dependency groups

| Group | Packages | Purpose | When |
|---|---|---|---|
| core | `torch`, `numpy` | Tensors, autograd, reference ops; NumPy interop (silences torch warning, fixture export later) | M0+ |
| `dev` | `pytest`, `ruff` | Tests, lint, format | M0+ |
| `notebooks` | `jupyterlab`, `matplotlib` | Explanatory notebooks, plots | M1 (optional) |
| `hf` | `transformers`, `huggingface-hub`, `safetensors`, `tokenizers` | Reference parity, checkpoint and tokenizer loading | M2 |
| `kaggle` | `kaggle` | Remote GPU kernels | M4 (optional) |
| C++ (system) | `cmake`, `gcc-c++`, `ninja-build` | C++20 build and CTest | M3 (scaffold builds in M0) |

Install extras with `pip install -e ".[notebooks]"`, `".[hf]"`, `".[kaggle]"`.
Fedora C++ tools: `sudo dnf install cmake gcc-c++ ninja-build` (needs sudo; only
when you start M3 — or `conda install -c conda-forge cmake ninja` without sudo).

## Pinned versions

`environment.yml` pins the versions validated in M0 (see `docs/progress.md`):
`torch==2.14.1`, `numpy==2.4.6`, `pytest==9.1.1`, `ruff==0.17.1`, Python 3.11. `pyproject.toml`
keeps compatible ranges so the package also installs elsewhere (CI, Kaggle).

## Reproducibility rules

- Fixed seeds in tests (`torch.Generator().manual_seed(...)`).
- No network access in tests; no checkpoint downloads outside explicit M2 steps.
- Model weights and caches go to directories ignored by Git (`/weights/`,
  `/checkpoints/`, `/hf_cache/`) or outside the repo; point tests at them with
  `QWEN3_WEIGHTS_DIR`.
- `torch.set_num_threads` may change float summation order; tests use tolerances.

## Hardware notes

- Tiny-config tests need < 1 GB RAM and run in seconds.
- Qwen3-0.6B in bf16 needs ~1.2 GB for weights (~2.4 GB in fp32) plus
  activations; CPU inference is feasible but slow.
- GPU availability is checked, never assumed (`check_env.py` reports it as INFO).
