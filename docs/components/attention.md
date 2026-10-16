# Grouped-query attention (M1.4, part B)

Implementation: [`src/qwen3_scratch/layers/attention.py`](../../src/qwen3_scratch/layers/attention.py) ·
Tests: [`tests/test_attention.py`](../../tests/test_attention.py),
[`tests/parity/test_attention_hf.py`](../../tests/parity/test_attention_hf.py) ·
Notebook: [`notebooks/04_qk_norm_gqa.ipynb`](../../notebooks/04_qk_norm_gqa.ipynb) ·
Error report: `python scripts/attention_report.py` ·
C++ fixture: [`tests/fixtures/attention_small.json`](../../tests/fixtures/attention_small.json) ·
Builds on: [RMSNorm](rmsnorm.md), [QK-Norm](qk_norm.md), [RoPE](rope.md)

## 1. Scaled dot-product attention

```
Attention(Q, K, V) = softmax( Q Kᵀ / √D + M ) V
```

Per head, with `Tq` query rows and `Tk` key rows:

| operation | shape | meaning |
|---|---|---|
| `Q` | `[Tq, D]` | what each position looks for |
| `K` | `[Tk, D]` | what each position offers |
| `V` | `[Tk, D]` | content to copy |
| `S = Q Kᵀ · D^{-1/2}` | `[Tq, Tk]` | similarity logits, scaled to unit variance |
| `M` | `[Tq, Tk]` | 0 where allowed, −∞ where masked (causal: `j > i`) |
| `P = softmax_row(S + M)` | `[Tq, Tk]` | each row a probability distribution |
| `O = P V` | `[Tq, D]` | weighted average of values |

**Why `1/√D`.** With independent unit-variance entries, `q·k` has variance `D`.
Without the scaling, softmax saturates as `D` grows. QK-Norm (part A) adds a bound
on vector norms; it does not replace this scaling.

**Stable softmax.** `softmax(s)_j = exp(s_j − max s) / Σ exp(s − max s)`.
Subtracting the row maximum avoids overflow and gives the same result.
`torch.softmax` does this internally. We compute it in float32 for
half-precision inputs (as HF does) and cast back.

## 2. MHA, MQA, GQA

| | KV heads `Nkv` | query → KV | K/V params (H=1024, D=128) | KV cache / token / layer (bf16) |
|---|---|---|---|---|
| Multi-head (MHA) | `Nq` = 16 | 1:1 | 4,194,304 | 8 KiB |
| **Grouped-query (GQA)**, Qwen3-0.6B | 8 | groups of `g = Nq/Nkv = 2` | 2,097,152 | 4 KiB |
| Multi-query (MQA) | 1 | all → one | 262,144 | 0.5 KiB |

- **Query heads stay separate.** Every query head has its own rows of `W_q`, so the
  16 heads still compute 16 different attention patterns. Only the keys and values
  they read are shared within a group.
- **Memory.** The KV cache stores `2 · Nkv · D` numbers per token per layer. For
  Qwen3-0.6B in bf16 over 28 layers that is 112 KiB per token: 3.5 GiB for a
  32,768-token context, versus 7 GiB with MHA and 0.44 GiB with MQA (tested).
- **Decoding speed.** Generating one token reads the entire cache. Decoding is
  usually memory-bandwidth bound, so halving the cache roughly halves that
  traffic. Larger batches and contexts fit in the same memory.
- **Quality.** MQA saves the most but loses some quality. GQA (Ainslie et al. 2023)
  recovers almost all of MHA's quality with a few KV heads. Qwen3 uses GQA for
  this trade-off; the 0.6B model uses 16 query / 8 KV heads.
- **Parameters.** `k_proj` and `v_proj` shrink by `g`; `q_proj` and `o_proj` are
  unchanged.

**Mapping.** Query head `h` reads KV head `⌊h / g⌋`. These are contiguous groups:
for Qwen3-0.6B, query heads 0–1 → KV 0, 2–3 → KV 1, …, 14–15 → KV 7.
`repeat_kv` (identical to `repeat_interleave(dim=1)` and to HF's `repeat_kv`)
produces exactly this order. A modulo mapping `h mod Nkv` (0→0, 1→1, …, 8→0)
also "shares" heads, but it pairs a different set of query heads with each KV
head. With trained weights that is simply wrong. A test shows it changes the
outputs.

**Numerical example** (`D = 1`, 2 KV heads, `g = 2`): KV heads `[[0, 1, 2], [3, 4, 5]]`
become query-head inputs `[[0,1,2], [0,1,2], [3,4,5], [3,4,5]]` (notebook §6–8).

**Logical sharing vs. physical copies.** Our first implementation materialises
`repeat_kv(k, g)` (a `[B, Nq, T, D]` copy) for clarity. Autograd sums the gradient
of each copy back into the shared head (tested). Logically, though, only `Nkv`
heads exist: the cache stores `Nkv`, and an optimised kernel indexes KV head
`h // g` instead of copying. That optimisation belongs to M2 (cache) and M3/M4
(C++).

## 3. Qwen3 attention forward pass

`B` batch, `T` tokens, `H = 1024`, `Nq = 16`, `Nkv = 8`, `D = 128` for 0.6B.

| step | operation | shape |
|---|---|---|
| input | hidden states (already `input_layernorm`-ed in the block) | `[B, T, H]` |
| 1–3 | `q_proj`, `k_proj`, `v_proj` (no bias) | `[B, T, Nq·D]`, `[B, T, Nkv·D]` ×2 |
| split heads | `view(B, T, N, D).transpose(1, 2)` | `[B, Nq, T, D]`, `[B, Nkv, T, D]` |
| 4–5 | `q_norm`, `k_norm`: RMSNorm over `D`, per head | same |
| 6 | RoPE on q and k (split-half, positions from caller) | same |
| 7 | `repeat_kv(k, g)`, `repeat_kv(v, g)` | `[B, Nq, T, D]` |
| 8 | `scores = (q @ kᵀ) · D^{-1/2}` | `[B, Nq, T, T]` |
| 9 | causal mask: `−∞` where `k_pos > q_pos`, **before** softmax | same |
| 10 | softmax over keys (float32, cast back) | same |
| 11 | `ctx = probs @ v` | `[B, Nq, T, D]` |
| 12 | merge heads `transpose(1, 2).reshape(B, T, Nq·D)`, then `o_proj` | `[B, T, H]` |

`Nq·D = 2048 ≠ H = 1024`. All sizes come from `Qwen3Config` and are tested on the
tiny config (`H = 64`, `4/2` heads, `D = 32`) and with the 0.6B shapes on the
`meta` device.

**How the pieces fit.** QK-Norm bounds the *norms* of `q` and `k`. RoPE, an
orthogonal rotation, then encodes *positions* without changing those norms. GQA
decides *which* keys/values each query head reads. The `1/√D` scaling and the
softmax turn the bounded, position-aware dot products into weights. QK-Norm must
come before RoPE: applying the elementwise `γ` scale after the rotation would
break RoPE's dependence on relative position only (see [qk_norm.md](qk_norm.md) §2).

## 4. Masking

- **Representation.** Our API takes a **boolean "may attend" mask**
  (`True` = allowed). Inside, masked logits are set to `−∞`. HF uses an
  **additive** float mask (0 / `finfo(dtype).min`). `bool_to_additive_mask`
  converts between the two. For every row with at least one allowed key, the two
  representations give bit-identical probabilities: `exp` underflows to exactly 0
  either way (tested).
- **Default.** Causal over the `T` tokens of the call (`k_pos ≤ q_pos`, a lower
  triangle). Absolute offsets do not matter for this case because query and key
  positions are the same tokens.
- **Shapes.** The mask may be `[T, T]`, `[B, T, T]` or `[B, 1|Nq, T, T]`. It is
  broadcast over heads and validated against the score shape.
- **From positions.** `causal_mask(q_positions, k_positions)` builds the mask from
  absolute positions. For cached decoding (`Tq ≠ Tk`, e.g. queries at 7–8 and keys
  at 0–8), a plain upper-triangular `[Tq, Tk]` matrix would be wrong; the
  position-based mask is right (tested). Padding is combined explicitly:
  `causal_mask(...) & key_is_real[:, None, :]`.
- **Fully masked rows** (e.g. queries at padding positions): `−∞` everywhere would
  give NaN. We return an all-zero probability row and finite gradients. HF's
  finite `finfo.min` gives a *uniform* row instead. This only affects positions
  whose outputs are discarded anyway.
- **HF caveat.** `Qwen3Attention` on its own is **not causal**. Called with
  `attention_mask=None`, it attends to future tokens (tested). `Qwen3Model` builds
  the mask. Our module is causal by default.

**Leakage test.** Two sequences share a 5-token prefix and differ only in later
tokens (scaled ×100). The outputs at the shared positions are identical, while the
later positions differ. A backward pass from position 2 gives exactly zero
gradient at positions 3 and later.

## 5. Verification

**Against Hugging Face** (`tests/parity/test_attention_hf.py`, `transformers 5.20.0`).
Setup: `attn_implementation="eager"`, `eval()`, and an explicit additive causal
mask `[B, 1, T, T]`. `position_embeddings` come from HF `Qwen3RotaryEmbedding`
with `position_ids` starting at offset 3. Weights are mapped 1:1 with
`load_state_dict(strict=True)`, using the same key names. Head configurations
tested: GQA 4/2, MHA 4/4, MQA 4/1, and 16/8.

Intermediates come from forward hooks (`q_proj`, `k_proj`, `v_proj`, `q_norm`,
`k_norm`, and the input of `o_proj`) and from HF's returned attention weights.
RoPE outputs, grouped K/V and logits are not exposed by HF, so they are recomputed
from the hooked tensors with HF's own `apply_rotary_pos_emb`, `repeat_kv` and
`scaling`.

| result (fp32 and bf16) | |
|---|---|
| projections, normed q/k, RoPE q/k, grouped k/v, probabilities, context, output | **bit-identical** |
| logits at allowed positions | bit-identical (masked: `−∞` vs ≈`finfo.min`, as designed) |
| gradients w.r.t. input and all six parameters (fp32) | **bit-identical** |

**Against an independent float64 loop** (explicit `h // g` indexing, no
`repeat_kv`): agreement to `1e-12` for MHA, MQA and three GQA shapes. Against
`torch.nn.functional.scaled_dot_product_attention(..., is_causal=True,
enable_gqa=True)`: agreement to `1e-5` (fp32).

`python scripts/attention_report.py` (0.6B attention sizes, `x: (1, 128, 1024)`,
positions 1000–1127). `max_rel = max|a−b| / max|ref|`:

| dtype | vs | tensor | max_abs | max_rel | mean_abs |
|---|---|---|---|---|---|
| float32 | float64 module | probs | 5.5e-06 | 5.5e-06 | 3.2e-08 |
| float32 | float64 module | output | 4.9e-06 | 2.1e-06 | 5.3e-07 |
| float32 | float64 module | dL/dx | 1.3e-05 | 3.1e-06 | 1.1e-06 |
| float32 / bf16 | HF `Qwen3Attention` | probs, output, dL/dx | **0** | **0** | **0** |
| bfloat16 | float64 module | output | 1.1e-02 | 4.9e-03 | 9.0e-04 |
| bfloat16 | float64 module | dL/dx | 2.0e-02 | 4.9e-03 | 1.6e-03 |

## 6. Discrepancies with the educational reference (Raschka, Appendix C)

| topic | R2 | HF / ours |
|---|---|---|
| names | `W_query`, `W_key`, `W_value`, `out_proj` | `q_proj`, `k_proj`, `v_proj`, `o_proj` (X10) |
| KV grouping | `repeat_interleave(group_size, dim=1)` | `repeat_kv` (same contiguous order) |
| mask | boolean `triu` + `masked_fill(-inf)`; sliced for the cache | additive `finfo.min` (HF) / boolean from positions (ours) (X8) |
| scaling | `softmax(scores / √D)` after masking | `scores · D^{-1/2}` before masking (X7) |
| softmax dtype | activation dtype | float32, then cast back (X7) |
| QK-Norm ε | hard-coded `1e-6` | from config (X3) |

In float32, all of these are mathematically equivalent. In bf16 they can differ
by rounding. We follow HF.

## 7. Limitations of the educational implementation

- Repeated K/V heads are materialised, which costs `g×` memory for K/V during
  attention.
- `[B, Nq, T, T]` scores and probabilities are materialised: O(T²) memory per head.
  There is no FlashAttention-style tiling.
- Full-sequence self-attention only (`Tq == Tk`). The KV cache (M2) will pass
  cached keys and values plus a position-based mask.
- No attention dropout (Qwen3's `attention_dropout = 0.0`), no sliding window.
- Float64 inputs keep a float64 softmax (HF would downcast); this only matters for
  tests.

## 8. Future KV-cache requirements (M2)

- Store **rotated** keys (after QK-Norm and RoPE) and raw values per layer, with
  `Nkv` heads, shape `[B, Nkv, S, D]`. Never store repeated heads.
- New tokens get `position_ids = cache_len + arange(T_new)`. Their RoPE tables use
  those positions.
- The mask must be `causal_mask(new_positions, all_key_positions)`, which is
  non-square, combined with any padding mask.
- The attention core must accept `Tq ≠ Tk`. The current `scores`/`probs` code
  already supports it; only the forward signature (past K/V input) is missing.

## 9. Requirements for the C++20 port (M3)

- **Q/K/V projections.** Three GEMMs `[N, H] × [H, Nq·D | Nkv·D]` with
  `N = B·T`, weights row-major `(out, in)`, float32 accumulation. They can be fused
  into one GEMM with concatenated weights `[(Nq + 2·Nkv)·D, H]`.
- **Layout.** Keep the projection output `[N, Nq·D]` row-major. Head `h` of token
  `n` is the contiguous slice `[n, h·D : (h+1)·D]`. The `[B, N, T, D]` transpose
  does not need to be physical: kernels index `(b, t, h)` directly. Materialise a
  head-major copy only if a GEMM-based score kernel needs it.
- **Per-head RMSNorm.** Loop over `D`-length slices with the M1.1 kernel and
  weight vector `q_norm[D]` / `k_norm[D]`. Apply in place.
- **RoPE.** The M1.3 in-place pair rotation on each head slice, with the table row
  chosen by position `t` (never by head).
- **KV sharing.** No copy: query head `h` reads K/V head `h / g` (integer division).
- **Scores, mask, softmax.** For each `(b, h, i)`: compute
  `s_j = dot(q_i, k_j) · D^{-1/2}` for allowed `j ≤ pos_i` only. Causality means
  the loop simply stops at `j = i`, with no `−∞` arithmetic. Track the running max
  (online softmax: `m, l` update), so the score row never needs to be stored. That
  gives O(T) extra memory instead of O(T²). Accumulate `Σ p_j v_j` in float32.
- **Merge and `o_proj`.** The context is written directly into `[N, Nq·D]` and
  multiplied by `W_o` `(H, Nq·D)`.
- **Complexity.** Projections cost `2·N·H·(Nq + 2·Nkv)·D` FLOPs (+`2·N·Nq·D·H` for
  `o_proj`). Scores and `PV` cost `≈ 2·B·Nq·T²·D` (half of that with causal
  skipping). Memory: `O(B·T·(Nq + 2·Nkv)·D)` for activations, plus the KV cache
  `2·L·B·S·Nkv·D`.
- **Fixture** (`tests/fixtures/attention_small.json`): `H = 8`, `Nq = 4`,
  `Nkv = 2`, `D = 4`, `T = 3`, positions 2–4, θ = 10⁴, ε = 1e-6. It includes all
  six weight tensors, the input `x`, and expected `q_rope`, `k_rope`, `probs`,
  `context` and `output`, with tolerance `rtol = 1e-5, atol = 1e-6`.
