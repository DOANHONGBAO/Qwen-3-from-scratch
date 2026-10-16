# Progress log

Results are reported as **PASSED / FAILED / SKIPPED / NOT RUN**, exactly as
observed. Nothing is listed as passed unless it was executed. Newest first.

## 2026-10-16 — M1.5: Qwen3 decoder block

### What was done
- Baseline before the change: 452 passed, 2 skipped (repository identical to the pushed `8260394`).
- `src/qwen3_scratch/layers/decoder.py`: `Qwen3DecoderLayer` (`input_layernorm`,
  `self_attn`, `post_attention_layernorm`, `mlp`) and `DecoderTrace`. It only wires
  M1.1–M1.4 components; no earlier module needed changes.
- Tests: `tests/test_decoder.py` (29), `tests/parity/test_decoder_hf.py` (7).
- `notebooks/05_decoder_block.ipynb`, `docs/components/decoder.md`,
  `scripts/decoder_report.py`; decision D-026.

### Validation (assistant sandbox: Python 3.11.17, torch 2.14.1 on CPU, transformers 5.20.0)

| Check | Result |
|---|---|
| `python scripts/check_env.py` | PASSED |
| ruff check / format | PASSED / PASSED (59 files) |
| `python -m pytest` (with transformers + matplotlib) | PASSED: 489 passed, 2 skipped |
| `python -m pytest` (transformers and matplotlib simulated missing) | PASSED: 426 passed, 8 skipped |
| `tests/test_decoder.py` | PASSED: 29 passed (shapes for GQA/MHA/MQA/D>H/N, manual composition, wrong orderings differ, norm placement, exact identity ablation incl. identity gradient, attention-only / MLP-only, pre-norm scale property, leakage, mask forwarding, gradients, `gradcheck`, determinism, invalid inputs, parameter count = config estimate) |
| `tests/parity/test_decoder_hf.py` | PASSED: 7 passed; input norm, attention output, first residual, post-attention norm, MLP output and output **bit-identical** to HF `Qwen3DecoderLayer` (eager) for GQA/MHA/MQA in fp32 and bf16; fp32 gradients for the input and all parameters bit-identical |
| `notebooks/05_decoder_block.ipynb` executed with `nbclient` (1 plot inspected, 0 errors) | PASSED |
| Same on Fedora | NOT RUN by the assistant |

`python scripts/decoder_report.py` (0.6B layer sizes, `x: (1, 64, 1024)`):

| dtype | ours vs | tensor | max_abs | mean_abs | rel_l2 |
|---|---|---|---|---|---|
| float32 | float64 layer | attn_output | 9.869e-07 | 1.006e-07 | 6.311e-07 |
| float32 | float64 layer | mlp_output | 6.804e-07 | 1.202e-07 | 5.938e-07 |
| float32 | float64 layer | output | 1.043e-06 | 1.625e-07 | 1.958e-07 |
| float32 | float64 layer | dL/dx | 2.820e-06 | 2.578e-07 | 3.001e-07 |
| float32 | HF Qwen3DecoderLayer | output / dL/dx | 0 | 0 | 0 |
| bfloat16 | float64 layer | output | 2.631e-02 | 2.993e-03 | 3.758e-03 |
| bfloat16 | float64 layer | dL/dx | 3.741e-02 | 4.180e-03 | 4.915e-03 |
| bfloat16 | HF Qwen3DecoderLayer | output / dL/dx | 0 | 0 | 0 |

## 2026-10-16 — M1.4 (part B): grouped-query attention

### What was done
- `src/qwen3_scratch/layers/attention.py`: `GroupedQueryAttention` (q/k/v/o
  projections, `q_norm`/`k_norm` via `apply_qk_norm`, RoPE via `apply_rope_to_qk`,
  contiguous KV grouping, scaled scores, position-based causal mask before a
  float32 softmax, `o_proj`, optional `AttentionTrace`), plus `repeat_kv`,
  `query_to_kv_head`, `causal_mask`, `bool_to_additive_mask`, `attention_softmax`.
- Tests: `tests/test_attention.py` (59), `tests/parity/test_attention_hf.py` (13),
  attention fixture in `tests/test_fixtures.py`, notebook 04 in `tests/test_notebooks.py`.
- `notebooks/04_qk_norm_gqa.ipynb`, `docs/components/attention.md`,
  `scripts/attention_report.py`, `tests/fixtures/attention_small.json`.
  Decisions D-005 (accepted), D-023 to D-025; discrepancies X7/X8 updated.
- **Bug found and fixed during development** (never committed): a float32-only softmax
  disagreed with the float64 loop reference at ~7e-8. Fixed with
  `promote_types(dtype, float32)` (D-023); covered by the float64 loop tests and `gradcheck`.
- No changes were needed to RMSNorm or RoPE.

### Validation (assistant sandbox: Python 3.11.17, torch 2.14.1 on CPU, transformers 5.20.0)

| Check | Result |
|---|---|
| `python scripts/check_env.py` | PASSED |
| `python -m ruff check .` / `python -m ruff format --check .` | PASSED / PASSED (53 files) |
| `python -m pytest` (with transformers + matplotlib) | PASSED: 452 passed, 2 skipped |
| `python -m pytest` (transformers and matplotlib simulated missing) | PASSED: 396 passed, 7 skipped |
| `tests/test_attention.py` | PASSED: 59 passed (shapes, QK-Norm/RoPE placement, grouping, modulo-mapping counterexample, MHA/MQA/GQA vs float64 loop at 1e-12, vs `F.scaled_dot_product_attention(enable_gqa=True)`, cache sizes, invalid configs/inputs, causal zeros, prefix leakage, gradient leakage, padding/fully-masked rows, extreme scales, bf16, gradients, `gradcheck`, T = 1, batch independence, relative-position invariance, determinism) |
| `tests/parity/test_attention_hf.py` | PASSED: 13 passed; projections, normed q/k, RoPE q/k, grouped k/v, probabilities, context and output **bit-identical** to HF eager `Qwen3Attention` (fp32 and bf16; GQA 4/2, MHA 4/4, MQA 4/1, 16/8); fp32 gradients for the input and all six parameters bit-identical; HF without a mask shown to be non-causal |
| `notebooks/04_qk_norm_gqa.ipynb` executed with `nbclient` (11 code cells, 4 plots, 0 errors) | PASSED |
| `tests/test_fixtures.py` | PASSED: 12 passed |
| C++ build + CTest (unchanged) | PASSED |
| Same on Fedora | NOT RUN by the assistant |

`python scripts/attention_report.py` (0.6B attention sizes, `x: (1, 128, 1024)`, positions 1000–1127):

| dtype | ours vs | tensor | max_abs | max_rel (÷ max abs ref) | mean_abs |
|---|---|---|---|---|---|
| float32 | float64 module | probs | 5.548e-06 | 5.548e-06 | 3.150e-08 |
| float32 | float64 module | context | 1.184e-05 | 5.298e-06 | 5.385e-07 |
| float32 | float64 module | output | 4.883e-06 | 2.096e-06 | 5.286e-07 |
| float32 | float64 module | dL/dx | 1.323e-05 | 3.149e-06 | 1.066e-06 |
| float32 | HF Qwen3Attention | probs / output / dL/dx | 0 | 0 | 0 |
| bfloat16 | float64 module | probs | 6.021e-03 | 6.021e-03 | 4.557e-05 |
| bfloat16 | float64 module | context | 1.389e-02 | 6.217e-03 | 9.146e-04 |
| bfloat16 | float64 module | output | 1.135e-02 | 4.871e-03 | 9.016e-04 |
| bfloat16 | float64 module | dL/dx | 2.043e-02 | 4.860e-03 | 1.648e-03 |
| bfloat16 | HF Qwen3Attention | probs / output / dL/dx | 0 | 0 | 0 |

### Open items
- Run on Fedora (optionally with `.[hf]` for the parity tests) and push.
- Pending from M0: confirm GitHub Actions is green.

### Next
M1.5 — Qwen3 decoder block (done, see above).

## 2026-10-16 — M1.4 (part A): QK-Norm

### What was done
- `src/qwen3_scratch/layers/qk_norm.py`: `apply_qk_norm` and `QKNorm`, reusing the
  M1.1 `RMSNorm` (no new kernel, no RMSNorm changes needed).
- Tests: `tests/test_qk_norm.py` (44), `tests/parity/test_qk_norm_hf.py` (9).
- `docs/components/qk_norm.md`; decisions D-021 (design) and D-022 (M1 renumbering).
- Baseline before the change: 322 passed, 2 skipped.

### Validation (assistant sandbox: Python 3.11.17, torch 2.14.1 on CPU, transformers 5.20.0)

| Check | Result |
|---|---|
| `python scripts/check_env.py` | PASSED |
| ruff check / format | PASSED / PASSED |
| `python -m pytest` (with transformers + matplotlib) | PASSED: 375 passed, 2 skipped |
| `python -m pytest` (transformers and matplotlib simulated missing) | PASSED: 332 passed, 6 skipped |
| `tests/test_qk_norm.py` | PASSED: 44 passed (shapes for MHA/GQA/MQA head counts, manual example, per-head independence, layout independence, logit bound, separate gradients, gradcheck, eps, near-zero, fp32 vs fp64, invalid shapes) |
| `tests/parity/test_qk_norm_hf.py` | PASSED: 9 passed, **bit-identical** to `Qwen3Attention.q_norm/k_norm` applied in HF's `[B, T, N, D]` order (fp32, bf16) |
| Same on Fedora | NOT RUN by the assistant |

## 2026-10-16 — M1.3: Rotary position embedding

### What was done
- `src/qwen3_scratch/layers/rope.py`: `compute_inv_freq`, `make_position_ids`,
  `rope_cos_sin`, `rotate_half`, `apply_rope`, `apply_rope_to_qk`,
  `RotaryEmbedding` (optional table cache), plus educational `rotate_2d`,
  `apply_rope_interleaved`, `half_layout_permutation`.
- Tests: `tests/test_rope.py` (71), `tests/parity/test_rope_hf.py` (12),
  `tests/fixtures/rope_small.json` added to the fixture checks.
- `notebooks/03_rope.ipynb`, `docs/components/rope.md`, `scripts/rope_report.py`.
  Decisions D-019, D-020; discrepancies X6/X12 updated with measurements.
- Verified in the HF source: `inv_freq = 1/θ^(arange(0, D, 2)/D)` (float32),
  `freqs = position_ids.float() * inv_freq`, `emb = cat(freqs, freqs)`, cos/sin cast
  to `x.dtype`, `q*cos + rotate_half(q)*sin` with `unsqueeze_dim=1`; default
  `position_ids = arange(past_seen, past_seen + T)`.

### Validation (assistant sandbox: Python 3.11.17, torch 2.14.1 on CPU, transformers 5.20.0)

| Check | Result |
|---|---|
| `python scripts/check_env.py` | PASSED |
| `python -m ruff check .` / `python -m ruff format --check .` | PASSED / PASSED (43 files) |
| `python -m pytest` (with transformers + matplotlib) | PASSED: 322 passed, 2 skipped |
| `python -m pytest` (transformers and matplotlib simulated missing) | PASSED: 288 passed, 5 skipped |
| `tests/test_rope.py` | PASSED: 71 passed |
| `tests/parity/test_rope_hf.py` | PASSED: 12 passed; `inv_freq`, cos/sin, rotated q/k and gradients **bit-identical** to HF (fp32, bf16; prefill, decode offset 1234, per-sequence offsets, positions 32760–32762) |
| Relative-position property (float64) | PASSED, `1e-10` |
| Incremental vs full rotation; simulated cached decoding scores | PASSED (bit-identical rotation; scores within float64 tolerance) |
| `gradcheck` (x, cos, sin; float64); gradient = inverse rotation | PASSED |
| `notebooks/03_rope.ipynb` executed with `nbclient` (3 plots rendered and inspected) | PASSED |
| `tests/test_fixtures.py` (swiglu + rope fixtures) | PASSED: 8 passed |
| Same on Fedora | NOT RUN by the assistant |

`python scripts/rope_report.py` (head_dim 128, θ = 1e6):

| quantity | float32 vs float64, max_abs (all 32,768 positions) | positions ≤ 4095 |
|---|---|---|
| inv_freq | 4.3e-08 | 4.3e-08 |
| angle (rad) | 2.4e-03 | 2.9e-04 |
| cos / sin | 2.3e-03 / 2.3e-03 | 2.6e-04 / 2.8e-04 |

| offset | dtype | ours vs | max_abs | rel_l2 | elements differing |
|---|---|---|---|---|---|
| 0 | float32 | float64 truth | 1.229e-04 | 2.537e-06 | 99.8% |
| 0 | float32 / bfloat16 | HF apply_rotary_pos_emb | 0 | 0 | 0.0% |
| 0 | bfloat16 | rotate-in-fp32 order (R2) | 3.125e-02 | 2.508e-03 | 24.5% |
| 8192 | float32 | float64 truth | 2.306e-03 | 7.220e-05 | 100.0% |
| 8192 | float32 / bfloat16 | HF | 0 | 0 | 0.0% |
| 32256 | float32 | float64 truth | 8.642e-03 | 2.789e-04 | 100.0% |
| 32256 | bfloat16 | float64 truth | 3.088e-02 | 2.729e-03 | 100.0% |
| 32256 | float32 / bfloat16 | HF | 0 | 0 | 0.0% |
| 32256 | bfloat16 | rotate-in-fp32 order (R2) | 3.125e-02 | 3.118e-03 | 38.0% |

Relative-position identity with float32 tables, max over 200 random `(q, k)`:
shift 1 → 2.9e-6, 100 → 4.5e-5, 1000 → 3.5e-4, 8000 → 1.4e-3, 32000 → 6.4e-3.

### Open items
- Run on Fedora and push (commits created locally by the assistant).
- Pending from M0: confirm GitHub Actions is green.

### Next
M1.4 — QK-Norm and grouped-query attention (awaiting approval).

## 2026-10-16 — M1.2: SwiGLU feed-forward

### What was done
- `src/qwen3_scratch/layers/swiglu.py`: `SwiGLUFeedForward` (`gate_proj`, `up_proj`,
  `down_proj`, no biases, SiLU on the gate), `from_config`, plus the educational
  `swiglu_textbook` / `silu_textbook`.
- Tests: `tests/test_swiglu.py` (71), `tests/parity/test_swiglu_hf.py` (9),
  `tests/test_fixtures.py` (4, C++ fixtures), notebooks auto-collected by
  `tests/test_notebooks.py`.
- `notebooks/02_swiglu.ipynb`, `docs/components/swiglu.md`, `scripts/swiglu_report.py`,
  `scripts/export_fixtures.py` + `tests/fixtures/swiglu_small.json`. Decisions D-017, D-018.
- Verified in the HF source: `Qwen3MLP` uses `bias=False` for all three
  projections and `ACT2FN[hidden_act]` = SiLU on `gate_proj`.

### Validation (assistant sandbox: Python 3.11.17, torch 2.14.1 on CPU, transformers 5.20.0)

| Check | Result |
|---|---|
| `python scripts/check_env.py` | PASSED |
| `python -m ruff check .` / `python -m ruff format --check .` | PASSED / PASSED |
| `python -m pytest` (with transformers + matplotlib) | PASSED: 234 passed, 2 skipped |
| `python -m pytest` (transformers and matplotlib simulated missing) | PASSED: 212 passed, 4 skipped |
| `tests/test_swiglu.py` | PASSED: 71 passed |
| `tests/parity/test_swiglu_hf.py` | PASSED: 9 passed, forward **and** backward bit-identical to `Qwen3MLP` (fp32, bf16) |
| Closed-form gradients vs autograd; `gradcheck`/`gradgradcheck` (float64) | PASSED |
| `notebooks/02_swiglu.ipynb` executed with `nbclient` (plot rendered and inspected) | PASSED |
| Same on Fedora | NOT RUN by the assistant |

`python scripts/swiglu_report.py` (0.6B sizes, input (2, 64, 1024), weights 0.02·N(0,1)):

| dtype | ours vs | max_abs | max_abs / max abs(ref) | rel_l2 | elements differing |
|---|---|---|---|---|---|
| float32 | textbook float64 | 6.434e-07 | 5.051e-07 | 5.497e-07 | 100.0% |
| float32 | HF Qwen3MLP | 0 | 0 | 0 | 0.0% |
| bfloat16 | textbook float64 | 5.686e-03 | 4.462e-03 | 3.910e-03 | 100.0% |
| bfloat16 | HF Qwen3MLP | 0 | 0 | 0 | 0.0% |

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
M1.2 — SwiGLU FeedForward (done, see above).

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
