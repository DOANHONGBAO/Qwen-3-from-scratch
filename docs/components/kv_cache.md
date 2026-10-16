# KV cache and generation (M2.2)

Code: [`kv_cache.py`](../../src/qwen3_scratch/kv_cache.py),
[`layers/attention.py`](../../src/qwen3_scratch/layers/attention.py),
[`model.py`](../../src/qwen3_scratch/model.py),
[`generation.py`](../../src/qwen3_scratch/generation.py).
Tests: `tests/test_kv_cache.py`, `tests/test_sampling.py`, `tests/parity/test_pretrained_hf.py`.
Notebook: [`08_kv_cache_generation.ipynb`](../../notebooks/08_kv_cache_generation.ipynb).
Benchmark: `scripts/benchmark_generation.py`.

## 1. Autoregressive generation, prefill and decode

A causal LM outputs at position $t$ the distribution $p(x_{t+1}\mid x_{0..t})$.
Generation picks a token from the last position, appends it, and repeats.

| phase | input | work per layer | output used |
|---|---|---|---|
| prefill | prompt `[B, T0]` | Q, K, V for all `T0` tokens; causal `T0 × T0` attention | logits of the last position |
| decode (cached) | new token `[B, 1]` | Q, K, V for 1 token; attention `1 × (T_past + 1)` | its logits |
| decode (uncached, M1.6) | whole sequence `[B, T0 + t]` | everything again | only the last row |

**Why K and V can be reused.** In layer $\ell$,
$k_j = \mathrm{RoPE}_j(\mathrm{RMSNorm}_k(W_k h_j))$ and $v_j = W_v h_j$. With a causal
mask, $h_j$ depends only on tokens $0..j$, so $k_j, v_j$ are final once computed.
**Why queries are not cached:** $q_j$ only produces position $j$'s own output, which
was already emitted; later positions need old *keys and values*, never old queries.

**Cost.** Uncached step $t$: projections and MLP over $T_0+t$ tokens plus
$O((T_0+t)^2)$ attention, so $O(T^3)$ attention work over a whole generation.
Cached step: projections/MLP for one token plus $O(T_0+t)$ attention. Caching
removes the recomputation; it does **not** remove attending over the past, so the
decode step still grows (slowly) with the context, and memory grows linearly.

## 2. Cache layout and memory

One entry per decoder layer, native KV heads, keys **after QK-Norm and RoPE**:

```
keys[layer]    [B, Nkv, T_cached, D]
values[layer]  [B, Nkv, T_cached, D]
```

$$\text{bytes} = 2 \cdot L \cdot B \cdot N_{kv} \cdot T \cdot D \cdot s$$

Qwen3-0.6B ($L=28$, $N_{kv}=8$, $D=128$): 114,688 B = **112 KiB per token** in bf16,
224 KiB in fp32. bf16: 14 MiB at 128 tokens, 224 MiB at 2,048, 3.5 GiB at the full
32,768 context (batch 1). GQA halves this compared with storing 16 heads.

## 3. Attention with a cache

```
x_new [B, T, H] -> q [B, Nq, T, D], k_new, v_new [B, Nkv, T, D]
q, k_new <- QK-Norm, then RoPE at ABSOLUTE positions T_past .. T_past+T-1
k_all, v_all <- cache.update(layer, k_new, v_new)        [B, Nkv, T_past+T, D]
k_grp, v_grp <- repeat_kv(k_all/v_all, Nq/Nkv)            (compute only, not stored)
scores [B, Nq, T, T_past+T] masked by allowed[i, j] = (j <= T_past + i)
```

The mask is built from absolute indices (`causal_mask(q_idx, k_idx)`), never with
`tril(ones(T, T_k))`, which would be aligned top-left and hide the past when
$T \ne T_k$. Single-token decode: one row of ones (the new token sees everything,
itself included). Chunked decode: the past is fully visible, the chunk is causal.

## 4. Positions

Default positions continue from `cache.seq_len`: the model computes
`position_ids = T_past + arange(T)`, so RoPE rotates the new token by its true
position. Explicit `position_ids` are still accepted (and validated against
`max_position_embeddings`). Cached keys are never rotated again.

## 5. Ownership, lifecycle, validation (`KVCache`)

* The caller owns the cache (`model.new_cache()` / `KVCache.from_config(cfg)`) and passes
  it with `use_cache=True`; the model appends in place and returns it in
  `out.past_key_values`. The model never stores a cache, so sessions cannot leak into
  each other. `use_cache=False` never touches an external cache (passing one is an error).
* `update(layer, k, v)` checks rank, shapes, dtype/device, batch, `Nkv`, `D`, and that the
  layer is appended exactly once per forward. `commit(T)` advances `seq_len` after the
  last layer; if a forward raises half-way, `rollback()` removes the partial appends.
* `max_length` (= `max_position_embeddings`) is checked **before** any layer is touched.
* `reset()` starts a new session (batch/dtype may change).
* Padded batches with a cache raise `NotImplementedError`; all-ones masks are accepted.
* Growth is by `torch.cat` (O(T) copy per step): simple and correct; a pre-allocated or
  paged cache is a later optimisation (M4 / C++).

## 6. Generation API

```python
from qwen3_scratch.generation import GenerationConfig, generate

out = generate(model, input_ids, max_new_tokens=32, eos_token_id=151643)  # greedy + cache
out = generate(
    model,
    input_ids,
    max_new_tokens=32,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    seed=0,
)
out.sequences, out.new_tokens, out.finished, out.cache, out.step_logits
```

Token selection on float32 logits, in HF's order:

1. greedy when `do_sample=False` **or** `temperature == 0` (defined, not an error);
2. `logits / temperature`;
3. top-k: values below the k-th largest become −inf (ties at the k-th value kept);
4. top-p: a token is removed when the probability mass of strictly better tokens is
   already ≥ p (the top-1 token always survives);
5. `softmax` (max-subtracted) and `torch.multinomial(generator=...)`; `seed=` builds a CPU
   generator for reproducible runs.

Validation: `max_new_tokens` ≥ 0, `temperature` finite ≥ 0, `top_k` ≥ 1, `0 < top_p ≤ 1`,
top-k/top-p without `do_sample` is an error, EOS ids in range, prompt length +
`max_new_tokens` ≤ `max_position_embeddings`. Rows that emitted an EOS keep emitting the
first EOS id; the loop stops when all rows are finished. Everything runs under
`torch.inference_mode()`; the model's train/eval mode is restored afterwards.
`greedy_generate` (M1.6 API) is a thin wrapper, uncached by default.

## 7. Verification

| check | result (CPU) |
|---|---|
| full forward vs prefill + token-by-token, float64 (Tiny, two configs) | ≤ 1e-12 (observed ~1e-15) |
| same, float32 | atol = rtol = 1e-5 |
| chunked decoding (4 chunkings) | float64 ≤ 1e-12 |
| future leakage inside a chunk | earlier logits bit-identical |
| our cache vs HF `DynamicCache` (step logits, layer-0 keys) | float32 ≤ 1e-5 |
| greedy ids vs `transformers.generate` (tiny random checkpoints) | identical |
| cached vs uncached on a random 0.6B-shape model (sandbox, fp32) | max \|Δlogit\| ≈ 2–4e-6, ids identical |
| real Qwen3-0.6B-Base, fp32: parity script (5 prompts, 8 steps) | max \|Δlogit\| 2.4–3.0e-5, ids identical; greedy ids = `transformers.generate` |
| real Qwen3-0.6B-Base, fp32: benchmark (9 cases, up to 64 steps) | max \|Δlogit\| 1.8–4.2e-5, ids identical in every case |
| real Qwen3-0.6B-Base, bf16: parity script | identical logits (Δ = 0) |

The tolerances are documented in the test modules: identical math, different matmul
shapes, so float32 differences are rounding-level (~1e-6 relative; the real model's
logits are O(10), hence ~3e-5 absolute); bugs give 1e-2 … 1. No near-tie flipped a
greedy token in any run.

## 7b. Measured speed (real model, fp32, 2-core laptop)

Cached decode 4.1–4.6 tok/s for prompts ≤ 128 tokens and 3.6 tok/s at 256, versus
0.3–1.0 tok/s without the cache; decode speed-up 4.8× (P = 16) to 12.8× (P = 128),
growing with the context because the uncached step recomputes the whole sequence.
The cached step itself slows from 216 to 278 ms/token between P = 16 and P = 256: the
new query still attends over every cached key. Prefill (0.9–4.9 s) is identical in both
modes and is excluded from decode throughput. Full table:
[`hardware_assessment.md` §3b](../hardware_assessment.md), raw data `results/m2/benchmark_local_fp32.*`.

## 8. Common bugs (each has a test)

new token at position 0 · cached keys rotated twice · `tril` mask with $T_q \ne T_k$ ·
storing `Nq` instead of `Nkv` heads · appending the prompt twice · sharing a cache
across sessions · half-updated cache after an exception · padding with a cache.

## 9. C++ contract

`tests/fixtures/kv_cache_tiny.json`: prefill logits and per-layer K/V after the prompt,
three teacher-forced decode steps (positions 5, 6, 7), final K/V, and the uncached
logits of the whole sequence. Layout `[B, Nkv, T, D]` row-major float32; keys stored
post-RoPE; tolerance rtol = atol = 1e-5. `sampling_small.json` fixes the deterministic
part of sampling (filtered probabilities, greedy ids); the random draw is not part of
the contract.
