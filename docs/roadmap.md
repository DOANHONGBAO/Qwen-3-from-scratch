# Roadmap

Milestones are done strictly in order. A milestone starts only after the previous
one meets its acceptance criteria and the maintainer approves.

Legend: `[x]` done and verified · `[ ]` open · `[~]` prepared, not yet exercised.

## M0 — Foundation

- [x] Repository layout (`src/`, `tests/`, `cpp/`, `scripts/`, `configs/`, `docs/`, `kaggle/`, `notebooks/`, CI)
- [x] `pyproject.toml` with separated extras (`dev`, `notebooks`, `hf`, `kaggle`)
- [x] `environment.yml` (conda, Python 3.11, CPU torch)
- [x] `Qwen3Config` (HF field names, validation, presets tiny + 0.6B, analytic parameter count)
- [x] Environment validation script (`scripts/check_env.py`)
- [x] pytest tiers (CPU / `gpu` / `weights`) with automatic skipping
- [x] Ruff lint + format configuration
- [x] C++20 CMake + CTest scaffold mirroring `Qwen3Config`
- [x] GitHub Actions: Python lint/tests on CPU, C++ build/test
- [~] Kaggle templates and auth check (not run: needs credentials, by design)
- [x] Documentation: README, architecture, environment, references, decisions, progress, CLAUDE.md
- [ ] Conda environment created and validated on the maintainer's Fedora machine
- [ ] First commit pushed to GitHub and CI green

**Acceptance:** `scripts/setup_env.sh` succeeds on Fedora; `pytest`, `ruff check`,
`ruff format --check`, `scripts/build_cpp.sh` pass; CI green on `main`; no
secrets or weights in Git.

## M1 — Qwen3 in PyTorch from scratch (tiny config, CPU)

Each item: math + shapes doc, implementation, synthetic example, tests (shape,
values, determinism, invalid input), "common mistakes" notes.

- [x] M1.1 RMSNorm — `layers/rmsnorm.py`, 101 unit tests + 13 HF parity tests, gradcheck, notebook `01_rmsnorm.ipynb`, doc `docs/components/rmsnorm.md` (bit-identical to HF `Qwen3RMSNorm` in fp32/bf16/fp16)
- [x] M1.2 SwiGLU FeedForward — `layers/swiglu.py`, unit + HF parity tests (bit-identical to `Qwen3MLP` fwd/bwd in fp32/bf16), gradcheck, notebook `02_swiglu.ipynb`, doc `docs/components/swiglu.md`, C++ fixture `tests/fixtures/swiglu_small.json`
- [ ] M1.3 RoPE (tables, `rotate_half`, offset; checks: norm preservation, relative-position property)
- [ ] M1.4 QK-Norm (per-head RMSNorm before RoPE)
- [ ] M1.5 Causal mask (with KV offset)
- [ ] M1.6 Grouped Query Attention (vs. naive MHA with repeated KV; vs. `scaled_dot_product_attention`)
- [ ] M1.7 Decoder block (pre-norm residuals)
- [ ] M1.8 Qwen3 model (tied embeddings, parameter count == `estimate_num_parameters`)
- [ ] M1.9 Notebook walkthrough of the tiny model

**Acceptance:** all modules tested on CPU; full tiny-model forward is
deterministic; parameter count matches the analytic count; CI green.

## M2 — Validation against the official checkpoint

- [ ] HF → native weight-name mapping with a completeness test (no missing/unexpected keys)
- [ ] Parity with a randomly initialised tiny `transformers` Qwen3 (no download): every intermediate tensor
- [ ] Download Qwen3-0.6B-Base (explicit step, outside Git), compare logits (fp32 and bf16, documented tolerances)
- [ ] Tokenizer (official `tokenizer.json`), round-trip and ID parity with HF
- [ ] KV cache: cached vs. uncached logits identical within tolerance
- [ ] Greedy generation: identical token IDs to HF for fixed prompts

**Acceptance:** logits within tolerance and identical greedy token IDs for a
fixed prompt set, executed and recorded in `docs/progress.md`.

## M3 — C++20 CPU implementation

- [ ] Tensor storage + minimal ops (row-major, fp32)
- [ ] Ports of RMSNorm, SwiGLU, RoPE, QK-Norm, mask, GQA, block, model
- [ ] Weight loading (safetensors reader or exported raw arrays)
- [ ] Parity fixtures exported from Python, checked in CTest
- [ ] KV cache and greedy generation, identical token IDs to Python

**Acceptance:** module-level and full-model parity with Python within fp32
tolerance; identical greedy tokens.

## M4 — GPU, profiling, benchmarks

- [ ] Kaggle GPU smoke test executed (explicit objective approved)
- [ ] Numerical parity analysis CPU vs GPU, fp32 vs bf16
- [ ] Profiling (PyTorch profiler; `perf` for C++)
- [ ] Optimisations (blocking, threading, SIMD; CUDA experiments) each with before/after measurements
- [ ] Reproducible benchmark reports (hardware, versions, commands, raw numbers)

**Acceptance:** every reported number comes from a recorded, reproducible run.
