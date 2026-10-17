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
- [x] M1.3 RoPE — `layers/rope.py` (split-half/`rotate_half`, explicit `position_ids`, offsets, optional table cache), unit + HF parity tests (bit-identical tables and rotations in fp32/bf16), relative-position and KV-decoding checks, notebook `03_rope.ipynb`, doc `docs/components/rope.md`, C++ fixture `tests/fixtures/rope_small.json`
- [x] M1.4 QK-Norm + grouped-query attention with causal masking (renumbered 2026-10-16, see D-022)
  - [x] QK-Norm — `layers/qk_norm.py` (reuses RMSNorm), unit + HF parity tests (bit-identical), doc `docs/components/qk_norm.md`
  - [x] GQA attention — `layers/attention.py` (projections, QK-Norm, RoPE, contiguous KV grouping, position-based causal mask, float32 softmax, `o_proj`); every intermediate and all gradients bit-identical to HF eager `Qwen3Attention` for MHA/GQA/MQA; leakage, float64 loop and SDPA references; notebook `04_qk_norm_gqa.ipynb`; doc `docs/components/attention.md`; fixture `tests/fixtures/attention_small.json`
- [x] M1.5 Decoder block — `layers/decoder.py` (`Qwen3DecoderLayer`: pre-norm attention + SwiGLU residual sublayers, HF parameter names), 29 unit tests (ordering, residual/identity ablations, norm placement, leakage, gradcheck), HF parity (every intermediate and all fp32 gradients bit-identical), notebook `05_decoder_block.ipynb`, doc `docs/components/decoder.md`
- [x] M1.6 Tiny Qwen3 causal LM — `model.py` (`Qwen3Model`, `Qwen3ForCausalLM`, real weight tying, HF-style init, padding mask + position ids), `loss.py` (shift-once causal LM loss, ignore_index, explicit no-target error), `generation.py` (greedy, context limit), `configs/qwen3_tiny.json` (426,752 params = estimate); embeddings, every layer, final norm, logits, loss and **all gradients bit-identical** to HF `Qwen3ForCausalLM`; mandatory prefix-leakage test; CLI demo; fixture `tests/fixtures/model_tiny.json`; doc `docs/components/model.md`
- [x] M1.7 Notebook walkthrough of the tiny model — delivered as `notebooks/06_tiny_qwen3.ipynb` together with M1.6

**Acceptance:** all modules tested on CPU; full tiny-model forward is
deterministic; parameter count matches the analytic count; CI green.

## M2 — Official checkpoint, tokenizer, KV cache **(done, pending review)**

### M2.1 — Official Qwen3 pretrained weights
- [x] Hardware audit on the real machine; memory model; local vs Kaggle decision (`docs/hardware_assessment.md`)
- [x] HF → native weight-name mapping generated from the config, strict, with a completeness test (no missing/unexpected keys, every parameter written once)
- [x] Parity with randomly initialised tiny `transformers` checkpoints written by `save_pretrained` (no download)
- [x] Download Qwen3-0.6B-Base at a pinned revision (explicit step, HF cache, outside Git)
- [x] Compare embeddings, all 28 layers, final norm, logits with HF in fp32 **and** bf16 — bit-identical
- [x] Tokenizer adapter on the official files; round trips; ID parity with HF (incl. config-only added tokens)
- [x] Greedy generation: identical token IDs to HF for 5 fixed prompts
- [x] Notebook 07, `scripts/run_pretrained_qwen3.py`, `scripts/verify_pretrained.py`

### M2.2 — KV cache and generation
- [x] Caller-owned per-layer KV cache (post-RoPE keys, native KV heads), absolute-position masks
- [x] Full forward == prefill + token-by-token / chunked cached decoding (Tiny: float64 ≤ 1e-12)
- [x] Cached decoding equal to HF `DynamicCache`; greedy ids equal to `transformers.generate`
- [x] Greedy, temperature, top-k, top-p, seeded sampling, EOS, context limits
- [x] Cached vs uncached on the real model (logits ≤ 4.2e-5 in fp32, identical tokens)
- [x] Benchmark (prefill, decode tok/s, peak RSS, KV size) on the target laptop; notebook 08
- [x] C++ fixtures: checkpoint name map, KV cache prefill/decode, sampling filters
- [~] Kaggle fallback kernel `kaggle/qwen3_inference/` (prepared; not submitted because the local run sufficed)
- [ ] Optional: padded batches with a cache, pre-allocated cache (deferred)

**Acceptance (met 2026-10-16):** logits within tolerance (observed: exactly equal) and
identical greedy token IDs for a fixed prompt set, executed on the real checkpoint
and recorded in `docs/progress.md`.

## M3 — C++20 CPU implementation

- [x] Tensor storage + minimal ops (row-major, fp32)
- [x] Ports of RMSNorm, SwiGLU, RoPE, QK-Norm, mask, GQA, block, model
- [x] Weight loading (own JSON + safetensors reader: F32 / BF16 / F16, strict name map)
- [x] Parity fixtures exported from Python, checked in CTest (`scripts/export_cpp_*_reference.py`)
- [x] KV cache and greedy generation, identical token IDs to Python (tiny model, CTest)
- [x] Real Qwen3-0.6B-Base: C++ vs saved HF float32 run on the maintainer's laptop
  (`scripts/verify_cpp_pretrained.py`)

**Acceptance:** module-level and full-model parity with Python within fp32
tolerance; identical greedy tokens.

## M4 — GPU, profiling, benchmarks

- [ ] Kaggle GPU smoke test executed (explicit objective approved)
- [ ] Numerical parity analysis CPU vs GPU, fp32 vs bf16
- [x] Profiling of the C++ model (built-in per-part timers in `qwen3_bench`)
- [x] C++ CPU optimisations with before/after measurements: weight-row reuse, SIMD-friendly
  dot product, OpenMP threads, `-march=native` (M4.1)
- [ ] Same benchmark on the maintainer's laptop (`scripts/benchmark_cpp.py --download`)
- [ ] bfloat16 weight storage in C++ (half the memory traffic per decoded token)
- [ ] CUDA experiments
- [ ] Reproducible benchmark reports (hardware, versions, commands, raw numbers)

**Acceptance:** every reported number comes from a recorded, reproducible run.
