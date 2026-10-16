# Qwen3 From Scratch

[![CI](https://github.com/DOANHONGBAO/Qwen-3-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/DOANHONGBAO/Qwen-3-from-scratch/actions/workflows/ci.yml)

An educational, test-driven implementation of the **Qwen3 dense Transformer**,
built from first principles in **PyTorch**, verified numerically against the
official checkpoint, and then re-implemented in **modern C++20**.

The goal is understanding, not a wrapper: every component (RMSNorm, SwiGLU, RoPE,
grouped-query attention with per-head QK-Norm, KV cache, ...) is written by hand,
explained mathematically, and unit-tested. Hugging Face `transformers` is used only
as a *reference* to verify numbers, never as the implementation.

> **Status: M1 in progress.** Implemented and verified bit-identical to the
> Hugging Face Qwen3 modules: RMSNorm (M1.1) and SwiGLU (M1.2). See
> [docs/roadmap.md](docs/roadmap.md).

## Target model

| | Tiny Qwen3 (tests) | Qwen3-0.6B-Base (target checkpoint) |
|---|---|---|
| layers | 2 | 28 |
| hidden size | 64 | 1024 |
| attention heads / KV heads | 4 / 2 | 16 / 8 |
| head_dim | 32 | 128 (≠ 1024/16) |
| MLP intermediate size | 192 | 3072 |
| vocab | 512 | 151,936 |
| QK-Norm / tied embeddings | yes / yes | yes / yes |
| parameters (analytic) | 156,096 | 596,049,920 (0.44B non-embedding) |

All dimensions come from `Qwen3Config` (`src/qwen3_scratch/config.py`); nothing
is hardcoded. Presets live in [`configs/`](configs/).

## Strategy: Python first, then C++

1. **PyTorch reference (M1–M2).** Small, readable modules, each tested in
   isolation, then assembled and compared tensor-by-tensor against the official
   Hugging Face implementation and checkpoint.
2. **C++20 port (M3).** Each verified Python module maps to one C++ module
   (`cpp/`), starting with plain CPU kernels, checked against tensors exported
   from the Python implementation.
3. **GPU and performance (M4).** Profiling, numerical parity analysis and
   benchmarks, with GPU runs on Kaggle when needed.

The module map is documented in [docs/architecture.md](docs/architecture.md).

## Quick start (Fedora, CPU only)

```bash
git clone https://github.com/DOANHONGBAO/Qwen-3-from-scratch.git
cd Qwen-3-from-scratch
bash scripts/setup_env.sh          # creates conda env "qwen3-scratch" and validates it
conda activate qwen3-scratch

python scripts/check_env.py        # environment report
pytest                             # CPU test tier
ruff check . && ruff format --check .
bash scripts/build_cpp.sh          # optional: C++20 scaffold (needs cmake + g++)
```

Manual alternative and troubleshooting: [docs/environment.md](docs/environment.md).

### CPU limitations

Development and all foundational tests run on a CPU-only machine. Unit tests use
the tiny configuration and finish in seconds. The full 0.6B model (~1.2 GB in
bfloat16) fits in RAM for inference on CPU in M2 but will be slow; training is out
of scope. GPU work is optional and runs on Kaggle ([docs/kaggle.md](docs/kaggle.md)).

## Tests are split into three tiers

| Tier | Marker | Runs where | Needs |
|---|---|---|---|
| Local CPU | *(none)* | laptop, CI | nothing |
| GPU | `@pytest.mark.gpu` | Kaggle | CUDA device (auto-skipped otherwise) |
| Pretrained weights | `@pytest.mark.weights` | laptop/Kaggle | `QWEN3_WEIGHTS_DIR` set (auto-skipped otherwise) |

## Repository layout

```
src/qwen3_scratch/   Python package (one module per model component)
tests/               pytest suite (CPU / GPU / weights tiers)
configs/             model presets + official HF config copy
cpp/                 C++20 port (CMake + CTest)
scripts/             environment setup/validation, C++ build, Kaggle auth check
kaggle/              Kaggle kernel templates (never auto-pushed)
notebooks/           explanatory notebooks
docs/                roadmap, architecture, environment, references, decisions, progress
```

## Documentation

- [Roadmap](docs/roadmap.md) — milestones, checklists, acceptance criteria
- [Architecture](docs/architecture.md) — Qwen3 math, shapes, module map, test conventions
- Components: [RMSNorm](docs/components/rmsnorm.md) · [SwiGLU](docs/components/swiglu.md)
- [Environment](docs/environment.md) — reproducible setup
- [References](docs/references.md) — sources, licenses, reference discrepancies
- [Decisions](docs/decisions.md) — technical decision log
- [Progress](docs/progress.md) — development log and validation results
- [Kaggle](docs/kaggle.md) — optional remote GPU workflow

## Acknowledgements and license

This project learns from Sebastian Raschka's
[*Build a Reasoning Model (From Scratch)*](https://github.com/rasbt/reasoning-from-scratch)
(Apache-2.0) and the Qwen3 implementation in Hugging Face
[`transformers`](https://github.com/huggingface/transformers) (Apache-2.0). The
Qwen3 model weights and configuration are released by the Qwen team under
Apache-2.0. See [docs/references.md](docs/references.md).

Code in this repository is licensed under the [Apache License 2.0](LICENSE).
