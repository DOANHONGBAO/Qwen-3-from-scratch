# Decision log

Format: ID · date · decision · rationale · status.

## D-001 · 2026-10-16 · PyTorch reference first, C++20 second
Correctness is established in Python where tensors are easy to inspect and the
official HF implementation is available for parity; the C++ port then has a
verified oracle for every module. **Accepted.**

## D-002 · 2026-10-16 · Config field names = Hugging Face `config.json` keys
The official config is the source of truth for checkpoint compatibility, so
`Qwen3Config` uses `hidden_size`, `num_key_value_heads`, `head_dim`, ... rather than
the book's `emb_dim`, `n_kv_groups`, ... A mapping to Raschka's names is kept in
`RASCHKA_KEY_MAP` for cross-reading. **Accepted.**

## D-003 · 2026-10-16 · Config is framework-independent
`qwen3_scratch.config` does not import torch; dtype is a string. The same struct
is mirrored in C++ (`cpp/include/qwen3/config.hpp`) and both sides test the same
analytic parameter counts. **Accepted.**

## D-004 · 2026-10-16 · `max_position_embeddings` defaults to 32,768
R5 says 32,768; the educational reference uses 40,960. The value only sizes the
RoPE table and the maximum input, so the official value is the default and it
stays configurable. (Discrepancy X1.) **Accepted.**

## D-005 · 2026-10-16 · Module/parameter names follow the HF checkpoint
Using `q_proj`, `k_proj`, `gate_proj`, `input_layernorm`, ... makes weight loading
a near-identity mapping and parity debugging easier. (Discrepancy X10.)
**Accepted** — RMSNorm, SwiGLU and attention state dicts load HF modules with
`strict=True` (M1.1–M1.4).

## D-006 · 2026-10-16 · Tooling: setuptools + pip/conda, Ruff, pytest
Conda for the interpreter (project requirement), pip for Python packages
(official CPU torch index), setuptools build backend (same as R1). Ruff rule set
`E,F,W,I,B,UP,SIM,NPY` (broader than R1's `E4,E7,E9,F`) and Ruff formatter,
line length 100. **Accepted.**

## D-007 · 2026-10-16 · Three test tiers with automatic skipping
CPU (unmarked), `gpu`, `weights`, plus `slow`. Skipping is decided in
`tests/conftest.py` from `torch.cuda.is_available()` and `QWEN3_WEIGHTS_DIR`, so
CI and the laptop never need a GPU or weights. **Accepted.**

## D-008 · 2026-10-16 · Numerical precision order follows HF for parity
Where the educational and official code differ only in cast order (RMSNorm, RoPE,
softmax — discrepancies X5–X7), we follow HF so bf16 parity tests can be tight.
**Proposed — confirm per component in M1.** RMSNorm: confirmed by D-013. RoPE: D-019.

## D-009 · 2026-10-16 · Real weight tying
When `tie_word_embeddings` is true the LM head will share the embedding
parameter (one tensor), not hold a copy. (Discrepancy X4.) **Accepted and implemented in M1.6** (identity, storage and shared
gradient tested; matches HF).

## D-010 · 2026-10-16 · License Apache-2.0
Matches the licenses of R1, R6 and the Qwen3 weights, and keeps attribution
simple. **Accepted, maintainer may change before first public release.**

## D-011 · 2026-10-16 · Do not commit local upstream reference copies
`documents/` (local copies of `modeling_qwen3.py` and `chC_main.ipynb`) is
git-ignored; we link to upstream instead of redistributing third-party code. The
official `config.json` (a small data file) is committed under `configs/hf/` with
attribution because tests depend on it. **Accepted.**

## D-012 · 2026-10-16 · C++ tests without a framework in M0
CTest runs a dependency-free executable. GoogleTest/Catch2 may be introduced in
M3 if test volume justifies a dependency (would need network at configure time).
**Accepted for M0.**

## D-013 · 2026-10-16 · RMSNorm follows the HF `Qwen3RMSNorm` operation order
Statistics in float32, cast the normalised activations back to the input dtype,
then multiply by `weight` (initialised to ones, no `1 + w` offset). This makes our
output bit-identical to HF in fp32/bf16/fp16 (tested), whereas the
"scale-then-cast" order of R2 and `torch.nn.RMSNorm` differs by 1 bf16 ulp on
~25 % of elements. The plain formula is kept separately as `rms_norm_textbook`
for teaching and as a float64 reference. **Accepted.**

## D-014 · 2026-10-16 · float64 inputs are not downcast in RMSNorm
Compute dtype is `promote_types(x.dtype, float32)`: fp16/bf16/fp32 → fp32,
float64 → float64. HF always uses float32 and would downcast float64. The
difference only affects float64 inputs (tests, `gradcheck`), never the model;
it is pinned by `tests/parity/test_rmsnorm_hf.py::test_documented_float64_difference`.
**Accepted.**

## D-015 · 2026-10-16 · HF parity tests are optional locally, required in CI
Parity tests import `transformers` via `pytest.importorskip` and construct
modules directly with random weights (no downloads). Locally they are SKIPPED
unless the `hf` extra is installed; CI installs `.[dev,hf]` so they always run
there. **Accepted.**

## D-016 · 2026-10-16 · Notebooks are executed by the test suite
`tests/test_notebooks.py` runs every notebook's code cells with `exec` (no Jupyter
dependency) and fails if a notebook is committed with outputs. Keeps notebooks in
sync with the package API. **Accepted.**

## D-017 · 2026-10-16 · SwiGLU module naming and bias policy
`SwiGLUFeedForward` uses HF parameter names (`gate_proj`, `up_proj`, `down_proj`)
so `Qwen3MLP` state dicts load with `strict=True`. `from_config` always builds it
without biases, matching HF (hardcoded `bias=False`, independent of
`attention_bias`). An explicit `bias` flag remains for teaching and tests.
**Accepted.**

## D-018 · 2026-10-16 · C++ parity fixtures are small exact JSON files
`scripts/export_fixtures.py` writes tiny fixtures (a few KB) to `tests/fixtures/`
as JSON whose numbers are exact float32 values. They are reviewable, need no
binary format, and are not blocked by the `*.npy`/`*.bin` ignore rules. They are
committed. `tests/test_fixtures.py` checks inputs and parameters bit-exactly and
outputs within the stored tolerance. Large fixtures for M3 will be generated at
test time instead. Supersedes the "generate as CTest step" note in
docs/components/rmsnorm.md for small fixtures. **Accepted.**

## D-019 · 2026-10-16 · RoPE follows HF: split-half pairing, float32 tables, cast before rotating
`rotate_half` pairing `(x_i, x_{i+D/2})`, `inv_freq` and angles in float32, cos/sin
cast to the activation dtype before the rotation. This is bit-identical to HF
`Qwen3RotaryEmbedding` + `apply_rotary_pos_emb` (tested in fp32/bf16, up to
position 32,767). The interleaved RoFormer layout is implemented only as
`apply_rope_interleaved` to demonstrate that it needs a weight permutation; it is
never used by the model. `apply_rope` refuses tables whose dtype differs from the
activations, so the order cannot be mixed up silently. **Accepted.**

## D-020 · 2026-10-16 · RoPE API takes explicit position ids
`RotaryEmbedding.forward(position_ids, dtype)` returns `(cos, sin)` for
arbitrary `[T]` / `[B, T]` positions. Offsets for KV-cache decoding are expressed
as positions (`make_position_ids(T, offset=cache_len)`), not as a hidden counter in
the model (unlike R2's `current_pos`). This keeps the module stateless and maps 1:1
to the C++ port. An optional precomputed table (`max_position_embeddings`) is a
pure optimisation, bit-identical to direct computation. **Accepted.**

## D-021 · 2026-10-16 · QK-Norm reuses RMSNorm; attention owns `q_norm`/`k_norm` directly
No new kernel: `apply_qk_norm` validates `[B, N, T, D]` shapes and calls the two
`RMSNorm(head_dim)` modules. `GroupedQueryAttention` stores them as `q_norm` /
`k_norm` attributes, so state-dict keys equal HF's
(`self_attn.q_norm.weight`). `QKNorm` is a thin standalone wrapper around the same
function. We normalise after the head transpose; HF normalises before it. The
results are bit-identical (tested), so the layout choice is free. **Accepted.**

## D-022 · 2026-10-16 · Roadmap renumbering of M1
The maintainer's task plan merges QK-Norm, causal masking and GQA into **M1.4**.
The decoder block becomes M1.5, the model M1.6 and the walkthrough notebook
M1.7. Earlier references to "M1.5 causal mask" / "M1.6 GQA" refer to M1.4.
**Accepted.**

## D-023 · 2026-10-16 · Attention numerics follow HF eager attention
Scores are `matmul(q, kᵀ) * D**-0.5` (multiply, not divide). The causal mask is
applied before a float32 softmax, and the result is cast back to the activation
dtype. Grouping uses contiguous `repeat_kv`. This is bit-identical to HF
`Qwen3Attention` (eager) in fp32/bf16 for every intermediate, and for fp32
gradients. As with RMSNorm (D-014), float64 inputs keep a float64 softmax
(`promote_types(dtype, float32)`); HF would downcast. This was found when the
float64 loop reference disagreed at 1e-7. **Accepted.**

## D-024 · 2026-10-16 · Boolean masks built from absolute positions
The attention API takes a boolean "may attend" mask; `causal_mask(q_positions,
k_positions)` derives it from absolute positions, so the same code is correct for
non-square cached decoding (M2). The default is causal over the call's own tokens.
Rows with no allowed key give zero probabilities (finite gradients) instead of NaN
(HF: uniform). `bool_to_additive_mask` produces HF's additive form for parity tests.
**Accepted.**

## D-025 · 2026-10-16 · Materialise repeated KV heads in the educational version
`repeat_kv` copies each KV head `g` times for readability and exact HF parity.
The KV cache will store only `Nkv` heads, and the C++ port will index head `h // g`
without copying (docs/components/attention.md §9). **Accepted for M1.**

## D-026 · 2026-10-16 · Decoder layer only wires verified components
`Qwen3DecoderLayer` composes RMSNorm, `GroupedQueryAttention` and
`SwiGLUFeedForward` with HF names (`input_layernorm`, `self_attn`,
`post_attention_layernorm`, `mlp`). It takes precomputed `(cos, sin)` and an
optional boolean mask instead of computing positions itself, so one RoPE table
per model forward is shared by all layers (as HF passes `position_embeddings`).
**Accepted.**

## D-027 · 2026-10-16 · RoPE frequencies stay float32 under module dtype casts (bug fix)
Found while testing the full model in bf16: `model.to(torch.bfloat16)` cast
`RotaryEmbedding.inv_freq` to bf16, and `rope_cos_sin` then multiplied positions
and frequencies in bf16 (positions > 256 not representable). Fix:
`RotaryEmbedding._apply` keeps `inv_freq` (and cached tables) in float32 (float64
allowed), and `rope_cos_sin` computes angles in `promote_types(inv_freq.dtype,
float32)`. Regression tests fail on the old code and pass now. HF shows the same
`.to(bf16)` rounding; HF models created directly in bf16 keep float32 frequencies,
which is the behaviour we now always have. **Accepted.**

## D-028 · 2026-10-16 · Model API: padding masks only, explicit positions, hard limits
`Qwen3Model.forward(input_ids | inputs_embeds, attention_mask, position_ids)`.
`attention_mask` is a `[B, T]` 0/1 padding mask combined with causality by token
index; additive float masks are rejected. Pad ids are never masked implicitly.
Default positions are `0..T-1`; positions or lengths ≥ `max_position_embeddings`
raise. The RoPE tables are computed once per forward and shared by all layers.
Outputs are small dataclasses (`Qwen3ModelOutput`, `CausalLMOutput`). **Accepted.**

## D-029 · 2026-10-16 · Causal LM loss: HF label convention, explicit failure on no targets
Labels have the same shape as `input_ids` and are shifted once inside
`causal_lm_loss` (padded with −100 at the end, exactly like HF, so the reduction
is bit-identical). The mean is over non-ignored targets in float32 (float64 for
float64 logits). With zero valid targets we raise `NoValidTargetsError` instead
of returning NaN. **Accepted.**

## D-030 · 2026-10-16 · Two tiny configurations
`configs/qwen3_tiny.json` (`QWEN3_TINY`, the maintainer's spec) is the
educational Tiny LM. Its `head_dim` equals `H / N` (32 = 128 / 4), so the model
and parity tests also run `TINY_QWEN3`, where they differ, plus an untied
variant. Routine tests never instantiate the 0.6B model; its parameter count is
checked on the `meta` device. **Accepted.**

## D-031 · 2026-10-16 · Minimal greedy generation without KV cache
`greedy_generate` re-runs the full prefix every step, refuses to exceed
`max_position_embeddings`, and supports an optional EOS. It does not support
padded batches or sampling. A byte-level toy tokenizer (V = 256) is used only in
the demo and notebook; the real tokenizer and the KV cache are M2. **Accepted.**


## D-032 · 2026-10-16 · Pinned official checkpoint, explicit download only
M2 uses `Qwen/Qwen3-0.6B-Base` at revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`
(`QWEN3_0_6B_BASE_REVISION`), never the instruct model. `download_checkpoint()` is the
only network call (`snapshot_download` with an allow-list: config, generation config,
safetensors, tokenizer files, license) and is never called by tests. Files stay in the
HF cache; `*.safetensors` is git-ignored. The weights tier uses `QWEN3_WEIGHTS_DIR`.
**Accepted.**

## D-033 · 2026-10-16 · Weight loader: config-derived name map, strict, no random weights
The checkpoint → module map and every expected shape are generated from
`Qwen3Config`, not taken from a `state_dict`. Names and shapes are checked before any
write; missing tensors and shape mismatches always fail (also with `strict=False`,
which only tolerates extra tensors); every parameter must be written exactly once.
Only safetensors is accepted (no pickle). The model is built on `meta`, materialised
with `to_empty`, RoPE buffers rebuilt and the head re-tied; tensors are copied one by
one with `param.copy_` (dtype conversion without a temporary). A tied checkpoint may
omit `lm_head.weight`; if present it must equal the embedding. **Accepted.**

## D-034 · 2026-10-16 · Tokenizer adapter over `tokenizers`; config-only added tokens
We do not re-implement byte-level BPE in M2: `Qwen3Tokenizer` loads `tokenizer.json`
with the Rust `tokenizers` library and applies `tokenizer_config.json`. Discrepancy
found on the real files: `tokenizer.json` contains 22 added tokens (151643–151664),
`tokenizer_config.json` declares 26; `<tool_response>`, `</tool_response>`, `<think>`,
`</think>` (151665–151668) exist only in the config. transformers registers them, so
the adapter does too and checks that each receives its declared id (vocabulary
151,669). No BOS; EOS = PAD = 151643; NFC normalisation. **Accepted.**

## D-035 · 2026-10-16 · Pretrained parity methodology and tolerances
`scripts/verify_pretrained.py` runs HF (reference) and our model in two separate
processes on identical token ids, comparing embeddings, all 28 layer outputs, the
final norm and the logits (forward hooks), top-1/top-5 agreement (near-ties listed
with their margin), greedy ids against `transformers.generate`, and cached vs.
uncached step logits. Tolerance: relative L2 ≤ 1e-5 in float32, ≤ 2e-2 in bf16, chosen
from the expected rounding (identical operation order) and never relaxed to pass.
A free-RAM guard refuses to start when one fp32 model would not fit with a 15 %
margin (`--force` overrides it and is recorded in the run log). Result on the real
checkpoint: bit-identical in fp32 and bf16 (see `docs/progress.md`). **Accepted.**

## D-036 · 2026-10-16 · Accept transformers ≥ 5 config format
`Qwen3Config.from_hf_config` accepts both the official top-level `rope_theta` and the
transformers 5 `rope_parameters`/`layer_types` layout (as written by `save_pretrained`
in transformers 5.20), and rejects non-default RoPE types and non-full-attention
layers. Found by the new HF round-trip test. **Accepted.**

## D-037 · 2026-10-16 · M2 execution plan: local CPU, sequential models, no Kaggle
Measured: 5.66 GiB RAM (2.9–3.6 GiB available with the desktop running), 2-core
Ryzen 3 3250U without bf16 instructions. One fp32 model (2.22 GiB weights, ~2.4–2.6 GiB
anonymous memory in use) fits; two do not. Therefore HF and our model run in
separate processes, fp32 is the reference dtype, bf16 is checked as well, threads = 2,
and the uncached benchmark is limited to prompts ≤ 128 tokens. Everything ran locally
on the maintainer's machine; the Kaggle fallback kernel is prepared but was not needed
and was not submitted. Details: `docs/hardware_assessment.md`. **Accepted.**

## D-038 · 2026-10-16 · KV cache: caller-owned, per-layer, post-RoPE, native KV heads
`KVCache` stores `[B, Nkv, T, D]` keys (after QK-Norm and RoPE) and values per layer
and grows by `torch.cat`. The caller creates and owns it; the model appends in place
when `use_cache=True` and never keeps it. Each layer may append once per forward
(`commit` after the last layer; `rollback` on exceptions); capacity, batch, heads,
dtype and device are validated. Positions continue from `seq_len`; masks are built
from absolute indices. Padding masks with a cache are rejected (`NotImplementedError`).
Pre-allocated/paged caches are deferred to optimisation work. **Accepted.**

## D-039 · 2026-10-16 · Generation semantics
`generate()` does prefill + one-token decode steps with the cache (or full
recomputation without it) under `inference_mode`, using `logits_to_keep=1`. Token
selection on float32 logits: greedy if `do_sample=False` or `temperature == 0`; else
temperature → top-k (ties kept) → top-p (minimal nucleus, top-1 kept) → softmax →
`multinomial` with an explicit generator (`seed=`). top-k/top-p without sampling is a
configuration error. Finished rows keep emitting EOS; the loop stops when all rows are
finished. `greedy_generate` (M1.6) remains as an uncached-by-default wrapper. **Accepted.**

## D-040 · 2026-10-16 · Benchmark methodology
`scripts/benchmark_generation.py` times prefill (first forward over the prompt) and
every decode step separately, reports decode tokens/s = (N − 1) / decode time (never
end-to-end throughput), resets peak RSS (`/proc/self/clear_refs`) per case, uses one
warm-up, 2 repeats (median), fixed threads, synthetic prompt ids, greedy decoding with
EOS ignored, and checkpoint download excluded. Uncached cases beyond the time budget
are marked SKIPPED, never extrapolated. Cached vs uncached logits and tokens are
compared for every case that ran both. **Accepted.**

## D-041 · 2026-10-17 · C++ port: simple, dependency-free, float32
The C++20 port uses one owning, always-contiguous row-major float32 `Tensor`
(`reshape`/`transpose` copy), naive loops (`linear` = row-dot-row on the `[out, in]`
checkpoint layout, no transpose) and plain classes with public weights named like the
checkpoint. No BLAS, threads or SIMD until correctness is established (M4 measures
optimisations). Weights are converted to float32 at load time (~2.4 GiB for 0.6B).
Reference values come from the verified PyTorch modules through generated headers
(`cpp/tests/*_reference.hpp`, 9 significant digits = exact float32 round trip), so
CTest needs no Python. **Accepted.**

## D-042 · 2026-10-17 · C++ checkpoint loading: own JSON + safetensors reader
A minimal JSON parser reads `config.json`, the safetensors header and the index of
sharded checkpoints. Tensors are read one at a time straight into the model's
pre-allocated weights (BF16/F16 converted in 1M-element chunks), so peak memory is the
float32 model plus one chunk. Loading is strict like D-033: every expected tensor
exactly once with the right shape, unexpected names are errors, `lm_head.weight` is
ignored when tied. The expected list is derived from the config and checked against
the M2 fixture of the real checkpoint (310 tensors). The tokenizer stays in Python
(`tokenizers`); `qwen3_run` takes token ids. **Accepted.**

## D-043 · 2026-10-17 · C++ KV cache layout: time-major
The C++ cache stores each layer as one `vector<float>` in `[S, B, Nkv, D]` order, so a
decode step appends at the end (amortised O(1)) instead of copying the whole cache as
`torch.cat` on `[B, Nkv, S, D]` does. Attention reads keys/values through
`key(layer, j, b, h)`; without a cache it uses a temporary cache of the same layout, so
there is a single attention code path. Rules match D-038 (append once per layer per
forward, `commit`, `rollback`, `max_length`). **Accepted.**

## D-044 · 2026-10-17 · C++ CPU optimisations (M4.1): order, SIMD, threads
Each change was measured on its own (`qwen3_bench`, synthetic 0.6B-shape checkpoint)
and kept only if it helped: (1) `linear` iterates over weight rows outside and input
rows inside, so a prefill reads each weight row once (no gain alone: the single
accumulator chain was the limit); (2) `dot_product` with 8 independent partial sums,
which the compiler turns into SIMD (prefill ~5×, decode ~1.6×); (3) OpenMP over the
output features of `linear`: every output element is still computed by the same
sequential code, so results are bit-identical for any thread count (`test_threads`);
(4) optional `-march=native` (`QWEN3_NATIVE`, off by default for portable CI builds).
The new summation order changes results in the last bits only (CTest tolerances
unchanged; synthetic 0.6B parity max |Δlogit| 5.8e-6). Attention core (4 % of the time
at 128 tokens) is not parallelised yet. No BLAS dependency. **Accepted.**

## D-045 · 2026-10-17 · C++ benchmark methodology
`qwen3_bench` uses synthetic prompt ids, greedy decoding without EOS, one warm-up and
the median of `--repeats` runs; prefill tokens/s = P / prefill time, decode tokens/s =
(N − 1) / decode time; load time and peak RSS reported separately. The built-in
profiler (off unless benchmarking) gives exclusive time shares per part.
`scripts/benchmark_cpp.py` builds an `-march=native` copy in `build/cpp-native` and
writes `results/m4/benchmark_cpp.{json,csv}` with CPU, compiler and commit. **Accepted.**
