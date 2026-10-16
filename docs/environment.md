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
M2 runs used, on the maintainer's Fedora machine: `transformers 5.20.0`,
`safetensors 0.8.0`, `tokenizers 0.23.3`, `huggingface_hub 2.2.0` (recorded in every
`results/m2/*.json` together with torch and Python versions). The `hf` extra is needed
at runtime only for checkpoint loading (`safetensors`), the tokenizer (`tokenizers`)
and downloading (`huggingface_hub`); `transformers` is used only as the reference.
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
  `QWEN3_WEIGHTS_DIR`. M2 downloads the pinned revision into the standard HF cache
  (`~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb…`).
- `torch.set_num_threads` may change float summation order; tests use tolerances.

## Hardware notes

- Tiny-config tests need < 1 GB RAM and run in seconds.
- Qwen3-0.6B: weights 1.11 GiB in bf16, 2.22 GiB in fp32. Measured peak RSS of our
  fp32 inference process: 2.7–3.4 GiB (see [hardware_assessment.md](hardware_assessment.md)).
  Close the browser before the fp32 parity check, which loads HF and our model one
  after the other.
- GPU availability is checked, never assumed (`check_env.py` reports it as INFO).

## Troubleshooting

**`No module named 'qwen3_scratch'` / `No module named 'torch'` / `ruff: command not found`,
and tracebacks show `/usr/lib64/python3.x`.** The shell is using Fedora's system
Python, not the conda env. Typical causes: the active env is not `qwen3-scratch`
(check the prompt — hyphen, not underscore), or the env was created without
Python (`conda create -n NAME` with no packages), so `python` falls through to
`/usr/bin/python3`. Fix:

```bash
conda deactivate
conda env create -f environment.yml      # creates qwen3-scratch with Python 3.11
conda activate qwen3-scratch
which python                             # must be .../envs/qwen3-scratch/bin/python
python scripts/check_env.py
```

`check_env.py` prints the interpreter path, Python version and active env first,
and warns about each of these mistakes before importing anything.
