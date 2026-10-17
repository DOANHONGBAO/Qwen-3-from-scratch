# Progress log

Results are reported as **PASSED / FAILED / SKIPPED / NOT RUN**, exactly as
observed. Nothing is listed as passed unless it was executed. Newest first.

## 2026-10-17 — M4.1: C++ CPU profiling and optimisation

### What was done
- `cpp/include/qwen3/profiler.hpp` (named timers, off by default) with timers around
  norms, attention (projections and core), MLP and LM head.
- `cpp/tools/qwen3_bench.cpp` and `scripts/benchmark_cpp.py` (D-045); `--threads` for
  `qwen3_run`.
- `linear`: weight-row-outer loop, `dot_product` with 8 partial sums (SIMD), OpenMP
  threads; attention uses `dot_product`; CMake options `QWEN3_OPENMP` (on) and
  `QWEN3_NATIVE` (off). Test `test_threads` (bit-identical results for 1–4 threads).
  Decision D-044.

### Step-by-step measurements
Sandbox only (Intel Xeon @ 2.1 GHz, 2 vCPU, GCC 13), **random synthetic** checkpoint with
the exact Qwen3-0.6B shapes, 16 new tokens, median of 2. Tokens/s for prompt 16 / 128:

| Step | Threads | Prefill tok/s | Decode tok/s |
|---|---|---|---|
| M3 baseline (naive loops) | 1 | 3.0 / 3.3 | 1.91 / 1.82 |
| + weight-row-outer loop | 1 | 3.2 / 3.3 | 1.88 / 1.81 |
| + `dot_product` (8 partial sums) | 1 | 13.9 / 16.6 | 2.95 / 2.86 |
| + OpenMP | 2 | 20.7 / 27.9 | 5.12 / 5.11 |
| + `-march=native` | 1 | 15.7 / 18.6 | 3.49 / 3.37 |
| + `-march=native` | 2 | 26.1 / 32.0 | 6.32 / 6.60 |

Time shares at the last step (2 threads, prompt 128): MLP 46 %, attention projections
37 %, LM head 9 %, attention core 4 %, norms < 1 %. Decode reads all 2.4 GB of float32
weights per token, so it is now close to memory-bandwidth bound.

### Validation
| Check | Where | Result |
|---|---|---|
| CTest (11 tests, incl. `threads.determinism`), 0 warnings; also with ASan/UBSan and with `QWEN3_OPENMP=OFF` | sandbox | PASSED |
| Synthetic 0.6B parity after optimisation (native build), C++ vs our PyTorch | sandbox | PASSED — max \|Δlogit\| 5.8e-6, greedy identical |
| Real Qwen3-0.6B-Base parity rerun after optimisation | maintainer's laptop | NOT RUN |
| Benchmark on the maintainer's laptop | maintainer's laptop | NOT RUN |

### Open items
Laptop benchmark and parity rerun; bfloat16 weight storage; parallel attention core for
long contexts.

## 2026-10-17 — M3: C++20 port of the full model

### What was done
- Written by the maintainer in guided lessons: `Tensor`, ops (elementwise, SiLU,
  stable softmax, matmul, linear), `RMSNorm`, `SwiGLU`, `RoPE` (split-half), each with
  a PyTorch reference header and CTest.
- Added: `Json` (minimal parser), `SafetensorsFile` (F32/BF16/F16, strict offsets) and
  `save_safetensors`, `KVCache` (time-major, D-043), `Attention` (GQA + QK-Norm + RoPE
  + causal softmax, cached or not), `DecoderLayer`, `Qwen3Model` (embedding, layers,
  final norm, tied/untied LM head, `last_only`), `load_pretrained` (single file or
  sharded index), `greedy_generate`, CLI `cpp/tools/qwen3_run.cpp`.
- Reference exporters `scripts/export_cpp_{attention,model}_reference.py`; parity
  script `scripts/verify_cpp_pretrained.py` (C++ vs the saved M2 Hugging Face float32
  logits and greedy ids). Decisions D-041–D-043.

### Validation
| Check | Where | Result |
|---|---|---|
| CTest (10 tests: config, tensor, ops, rmsnorm, swiglu, rope, kv_cache, attention, safetensors, model), 0 warnings with `-Wall -Wextra -Wpedantic -Wconversion` | sandbox | PASSED |
| Same with AddressSanitizer + UBSan (Debug) | sandbox | PASSED |
| Attention vs PyTorch (GQA 4/2, D=8 ≠ H/Nq), uncached and cached 3+1+1 | sandbox | PASSED — max \|Δ\| 1.2e-7 |
| Tiny random model vs PyTorch: each layer, logits, cached 4+1+1+1, greedy 10 tokens | sandbox | PASSED — logits max \|Δ\| 1.0e-5 (\|logit\| ≤ 14.5), tokens identical |
| Checkpoint loading round trip (safetensors F32), missing / wrong shape / extra tensor errors | sandbox | PASSED |
| Name map vs the real Qwen3-0.6B-Base tensor list (M2 fixture, 310 tensors) | sandbox | PASSED |
| Full 0.6B-shape **random synthetic** checkpoint (BF16), C++ vs our PyTorch, 2 prompts × 8 tokens | sandbox (2 vCPU) | PASSED — max \|Δlogit\| 8.6e-6, greedy identical; load 1.9 s, peak RSS 2.23 GiB, decode 1.8 tok/s (1 thread) |
| Real Qwen3-0.6B-Base (da87bfb), C++ vs saved HF float32 run (`scripts/verify_cpp_pretrained.py`), 5 prompts × 16 greedy tokens | maintainer's laptop (Ryzen 3 3250U, Fedora 44) | PASSED — max \|Δlogit\| 2.1–4.1e-4 (rel L2 4.5e-6–1.3e-5), argmax equal at every position, top-5 equal, greedy ids identical in 5/5; load 4.5 s, peak RSS 2.26 GiB, decode 1.87–1.89 tok/s (1 thread, naive loops); `results/m3/parity_cpp_float32.json` |

### Open items
Tokenizer in C++ (not planned); BLAS / threads / SIMD (M4): the PyTorch cached decode
runs at 4.1–4.6 tok/s on the same laptop vs 1.9 tok/s for the naive C++.

## 2026-10-16 — M2.2: KV cache and efficient generation

### What was done
- `src/qwen3_scratch/kv_cache.py`: `KVCache` (per-layer `[B, Nkv, T, D]` post-RoPE keys
  and values, `update`/`commit`/`rollback`/`reset`, capacity, batch, head, dtype and
  device validation) and `kv_cache_bytes`.
- Attention / decoder / model: `kv_cache`, `past_key_values`, `use_cache`; positions
  continue from the cache length; masks from absolute indices (`T_q != T_k`); padding
  masks with a cache rejected; `logits_to_keep`; `model.new_cache()`.
- `src/qwen3_scratch/generation.py`: `generate()` with prefill + cached decode,
  greedy / temperature / top-k / top-p, seeded sampling, EOS, streaming callback,
  per-step logits; `greedy_generate` kept (uncached by default).
- `scripts/benchmark_generation.py` (+ plots), `scripts/m2_local_run.sh`,
  `notebooks/08_kv_cache_generation.ipynb`, `docs/components/kv_cache.md`,
  fixtures `kv_cache_tiny.json`, `sampling_small.json`, Kaggle fallback package
  `kaggle/qwen3_inference/` (not submitted), decisions D-038–D-040, X13.
- Tests: `tests/test_kv_cache.py` (37, incl. the mandatory full-vs-cached test),
  `tests/test_sampling.py` (34), cache/generate parity with HF in
  `tests/parity/test_pretrained_hf.py`.

### Validation
| Check | Where | Result |
|---|---|---|
| Full forward vs prefill + token-by-token, float64, 2 tiny configs | sandbox | PASSED (≤ 1e-12) |
| Chunked decoding (4 chunkings), future leakage inside a chunk | sandbox | PASSED |
| Our cache vs HF `DynamicCache` (step logits, layer-0 keys), tiny | sandbox | PASSED (≤ 1e-5, fp32) |
| Greedy ids vs `transformers.generate`, tiny, cached and uncached | sandbox | PASSED (identical) |
| Real Qwen3-0.6B-Base, cached vs uncached, parity script (5 prompts × 8 steps) | maintainer's laptop | PASSED — fp32 max \|Δlogit\| 2.4–3.0e-5, ids identical; bf16 Δ = 0 |
| Real model, benchmark (9 cases with both modes) | maintainer's laptop | PASSED — max \|Δlogit\| 1.8–4.2e-5, tokens identical in all 9 |
| Benchmark | maintainer's laptop, fp32, 2 threads | RUN — cached decode 4.1–4.6 tok/s (P ≤ 128), 3.6 tok/s (P = 256); uncached 0.32–0.96 tok/s; decode speed-up 4.8–12.8×; peak RSS 2.56–2.81 GiB; KV cache 6.8–69.8 MiB (= 224 KiB/token analytic). Uncached P = 256: SKIPPED (time budget) |
| Kaggle kernel | — | NOT RUN (not needed) |
| Weights-tier `slow` test (fp32 cached vs uncached in pytest) | — | NOT RUN on real weights (covered by the parity script); PASSED on a random 0.6B-shape checkpoint in the sandbox |

### Open items
Padded batches with the KV cache; pre-allocated/paged cache; bf16 speed on CPUs with
native bf16 (this laptop has none).

## 2026-10-16 — M2.1: official Qwen3-0.6B-Base weights, tokenizer, HF parity

### What was done
- `src/qwen3_scratch/weight_loader.py` (config-derived name map and shapes,
  `SafetensorsCheckpoint`, strict `load_into_model`, `LoadReport`),
  `src/qwen3_scratch/pretrained.py` (pinned revision, `download_checkpoint`,
  `load_pretrained` via meta + `to_empty`), `src/qwen3_scratch/tokenizer.py`
  (`Qwen3Tokenizer`), `sysinfo.py`, `synthetic.py`.
- `Qwen3Config.from_hf_config` accepts the transformers 5 `rope_parameters` format.
- Scripts: `hardware_report.py`, `verify_pretrained.py`, `run_pretrained_qwen3.py`,
  `make_synthetic_checkpoint.py`. Notebook 07. Docs: `components/pretrained.md`,
  `hardware_assessment.md`, decisions D-032–D-037, discrepancies X14/X15.
- Fixture `checkpoint_name_map.json` for the C++ loader.
- **Found on the real files:** `tokenizer.json` has 22 added tokens but
  `tokenizer_config.json` declares 26; four (`<tool_response>`, `</tool_response>`,
  `<think>`, `</think>`, ids 151665–151668) exist only in the config. The first
  weights-tier run FAILED on this (vocabulary 151,665 instead of 151,669); the adapter
  now registers them like transformers and checks their ids (D-034); rerun PASSED.
- Tests: `tests/test_weight_loader.py` (19), `tests/test_tokenizer.py` (26),
  `tests/parity/test_pretrained_hf.py` (loader part), `tests/test_pretrained_weights.py`
  (weights tier, 11), `tests/test_config.py` (+5).

### Where validation ran
- **Maintainer's Fedora 44 laptop** (Ryzen 3 3250U, 5.66 GiB RAM, CPU only; Python
  3.11.17, torch 2.14.1+cpu, transformers 5.20.0, safetensors 0.8.0, tokenizers 0.23.3,
  huggingface_hub 2.2.0): everything that needs the real checkpoint. Raw results in
  `results/m2/`.
- **Assistant sandbox** (Xeon 2 vCPU, 7.8 GiB, Python 3.13, torch 2.14.1+cu130 on CPU,
  transformers 5.20.1): unit/parity tests and a rehearsal with a random-weight checkpoint
  of the exact 0.6B shapes. huggingface.co is not reachable from the sandbox.

### Validation
| Check | Result |
|---|---|
| Checkpoint `Qwen/Qwen3-0.6B-Base` @ `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, `model.safetensors` sha256 `cd2a5120…9223eba` | downloaded (165 s), recorded in `results/m2/checkpoint.json` |
| Mapping: 310/310 tensors, 0 missing / 0 unexpected / 0 shape errors, 596,049,920 params = HF, no `lm_head.weight` in the file, head tied | PASSED |
| `pytest -m "weights and not slow"` (real files) | first run FAILED (tokenizer, see above) → after fix PASSED (11 passed) |
| Parity fp32: embeddings, 28 layers, final norm, logits, 5 prompts | PASSED — max \|Δ\| = 0 for every tensor (bit-identical) |
| Parity bf16 | PASSED — max \|Δ\| = 0 |
| Top-1 / top-5 agreement, all positions | 1.000 / 1.000 (smallest HF top-2 margin 0.0625) |
| Tokenizer ids vs `AutoTokenizer`, 5 prompts | identical |
| Greedy ids (16 tokens) vs `transformers.generate`, 5 prompts | identical (fp32 and bf16) |
| Real text completion with our model (`run_pretrained_qwen3.py`) | RUN — e.g. "Hà Nội là thủ đô của Việt Nam, là một trong những thành phố…" |
| Peak RSS fp32: ours 3.60 GiB, HF 3.52 GiB (separate processes) | measured |
| Free-RAM guard | refused fp32 3× at 2.98–3.07 GiB available; fp32 run started with `--force` at 3.55 GiB (estimate 2.97 GiB) |
| CPU test suite with transformers (sandbox) | PASSED — 731 passed, 13 skipped |
| CPU test suite without transformers/safetensors/tokenizers/matplotlib (simulated) | PASSED — 604 passed, 24 skipped |
| Ruff lint / format; fixtures `--check`; `check_env.py`; C++ build + CTest | PASSED |
| Notebooks 07 and 08 executed (synthetic / tiny mode) | PASSED |

## 2026-10-16 — M1.6: Tiny Qwen3 causal language model (full forward pass)

### What was done
- `src/qwen3_scratch/model.py`: `Qwen3Model` (embedding, decoder stack, final
  RMSNorm, one shared RoPE table per forward, padding mask + position ids, strict
  input validation) and `Qwen3ForCausalLM` (LM head without bias, real weight
  tying, HF-style init, unique parameter count, optional loss).
- `src/qwen3_scratch/loss.py` (shift-once causal LM loss, `ignore_index=-100`,
  `NoValidTargetsError`), `src/qwen3_scratch/generation.py` (greedy, EOS, context limit).
- `Qwen3Config`: `attention_dropout`, `initializer_range`, bool validation; new preset
  `QWEN3_TINY` + `configs/qwen3_tiny.json`; existing config JSONs regenerated.
- **Integration bug found and fixed (D-027):** `model.to(torch.bfloat16)` rounded
  RoPE `inv_freq` to bf16 and the angles were computed in bf16. Now frequencies stay
  float32 and angles are computed in ≥ float32; two regression tests fail on the old
  code and pass on the new.
- The float64 loss path was also downcast to float32; it now keeps float64 (D-029).
- Tests: `tests/test_model.py` (42), `tests/test_causal_lm.py` (18),
  `tests/test_generation.py` (12), `tests/test_config.py` (33),
  `tests/parity/test_model_hf.py` (10), `tests/test_scripts.py` (1), fixture checks (16).
- `notebooks/06_tiny_qwen3.ipynb`, `scripts/demo_tiny_qwen3.py`, `scripts/model_report.py`,
  `tests/fixtures/model_tiny.json`, `docs/components/model.md`; decisions D-009 (implemented),
  D-027 to D-031; discrepancies X1/X4 updated.

### Validation (assistant sandbox: Python 3.11.17, torch 2.14.1 on CPU, transformers 5.20.0)

| Check | Result |
|---|---|
| `python scripts/check_env.py` | PASSED |
| `python -m ruff check .` / `python -m ruff format --check .` | PASSED / PASSED (71 files) |
| `python -m pytest` (with transformers + matplotlib) | PASSED: 588 passed, 2 skipped |
| `python -m pytest` (transformers and matplotlib simulated missing) | PASSED: 515 passed, 9 skipped |
| `tests/parity/test_model_hf.py` | PASSED: 10 passed; parameter coverage/shapes/values/counts/tying identical (Tiny, `tiny` D≠H/N, untied); embeddings, each layer output, final norm, logits and loss **bit-identical**; gradients of **all** parameters bit-identical; padded batch identical on real tokens; bf16 forward bit-identical |
| Mandatory prefix-leakage test (full model) | PASSED: shared-prefix logits equal, future logits differ; zero input-gradient from the future |
| `python scripts/demo_tiny_qwen3.py` | PASSED (exit 0, ~3 s; also run by `tests/test_scripts.py`) |
| `notebooks/05_decoder_block.ipynb`, `notebooks/06_tiny_qwen3.ipynb` executed with `nbclient` | PASSED (0 errors; plots inspected) |
| README code example executed | PASSED |
| Same on Fedora | NOT RUN by the assistant |

`python scripts/model_report.py` (Tiny Qwen3, 426,752 parameters, ids `(4, 64)`, float32):

| ours vs | tensor | max_abs | mean_abs | rel_l2 |
|---|---|---|---|---|
| HF | embeddings, layer 0, layer 1, final norm, logits, loss | 0 | 0 | 0 |
| HF | grads: embed_tokens, layer0 q_proj, layer0 q_norm, layer1 down_proj, model.norm | 0 | 0 | 0 |
| float64 model | logits | 7.271e-07 | 8.684e-08 | 4.374e-07 |
| float64 model | loss | 5.691e-07 | 5.691e-07 | 1.017e-07 |

Loss of the random-weight model: 5.5940800 (HF identical) vs ln 256 = 5.5451775.

### Open items
- Run on Fedora and push. Confirm GitHub Actions is green.
- Not implemented by design: KV cache, checkpoint loading, tokenizer, sampling, training.

### Next
M2 — load the official Qwen3-0.6B-Base weights (explicit download step) and tokenizer,
compare logits/generation with HF, add the KV cache (awaiting approval).

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
