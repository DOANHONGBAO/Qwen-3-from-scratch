# Tiny Qwen3: the complete causal language model (M1.6)

Implementation: [`src/qwen3_scratch/model.py`](../../src/qwen3_scratch/model.py)
(`Qwen3Model`, `Qwen3ForCausalLM`), [`loss.py`](../../src/qwen3_scratch/loss.py),
[`generation.py`](../../src/qwen3_scratch/generation.py) ·
Configs: [`configs/qwen3_tiny.json`](../../configs/qwen3_tiny.json),
[`configs/qwen3_0.6b_base.json`](../../configs/qwen3_0.6b_base.json) ·
Tests: `tests/test_model.py`, `tests/test_causal_lm.py`, `tests/test_generation.py`,
`tests/test_config.py`, `tests/parity/test_model_hf.py`, `tests/test_scripts.py` ·
Notebook: [`notebooks/06_tiny_qwen3.ipynb`](../../notebooks/06_tiny_qwen3.ipynb) ·
Demo: `python scripts/demo_tiny_qwen3.py` · Report: `python scripts/model_report.py`

## 0. What this is, and what it is not

| | status |
|---|---|
| **Educational Tiny Qwen3 from scratch** | done: random weights, `configs/qwen3_tiny.json`, runs on CPU in milliseconds |
| **Official Qwen3 architecture compatibility** | verified: same parameter names and shapes; logits, loss and gradients **bit-identical** to HF `Qwen3ForCausalLM` built from the same config (fp32, and bf16 forward) |
| **Official pretrained checkpoint inference** | NOT yet (M2): no safetensors loading, no tokenizer, no KV cache |
| **Training** | NOT implemented: no optimizer, data pipeline or dropout. The loss and gradients exist and are tested. |
| **C++20 implementation** | NOT started (M3): this document fixes the contract and the fixtures |

The Tiny model is **not a language model in any useful sense**. Its weights are
random, its loss is ≈ ln V, and its "generations" are accidents of initialisation.

## 1. Forward pass

```
input_ids [B, T] (int64, 0 ≤ id < V)
  │ embed_tokens: E ∈ ℝ^{V×H}                       h₀ = E[input_ids]                      [B, T, H]
  │ positions: position_ids (default 0..T−1) → cos, sin = RoPE(positions)  [1|B, T, D]  (once, shared)
  │ mask: causal by token index  (∧ padding mask [B, T] if given)          [B, T, T] bool
  │ for ℓ = 0..L−1:  h_{ℓ+1} = DecoderLayer_ℓ(h_ℓ, cos, sin, mask)                          [B, T, H]
  │ final norm:     z = RMSNorm(h_L; w_final)                                              [B, T, H]
  │ LM head:        logits = z Wᵀ,  W = E when tied                                       [B, T, V]
  └ loss (optional): mean_{valid t} −log softmax(logits[:, t])[labels[:, t+1]]              scalar
```

With `y = DecoderLayer(x)` defined as `h = x + Attn(RMSNorm₁(x)); y = h + SwiGLU(RMSNorm₂(h))`,
and attention as in [attention.md](attention.md).

**Shapes, Tiny config** (`B = 2`, `T = 5`): ids `[2, 5]` → embeddings `[2, 5, 128]` →
per layer: q `[2, 4, 5, 32]`, k/v `[2, 2, 5, 32]`, scores `[2, 4, 5, 5]`, MLP hidden
`[2, 5, 384]` → block output `[2, 5, 128]` → final norm `[2, 5, 128]` → logits
`[2, 5, 256]` → loss `[]`.

## 2. Configuration

`Qwen3Config` (frozen dataclass, HF field names, validated) now also carries
`attention_dropout` (must be 0.0 to build a model) and `initializer_range`.
`head_dim` is **independent** of `hidden_size / num_attention_heads`.

| | `qwen3-tiny` (configs/qwen3_tiny.json) | `tiny` (TINY_QWEN3, component tests) | Qwen3-0.6B-Base |
|---|---|---|---|
| vocab / hidden / layers | 256 / 128 / 2 | 512 / 64 / 2 | 151,936 / 1,024 / 28 |
| heads / KV heads / head_dim | 4 / 2 / 32 | 4 / 2 / 32 | 16 / 8 / 128 |
| intermediate / max positions | 384 / 128 | 192 / 128 | 3,072 / 32,768 |
| rope_theta / tied | 1e6 / yes | 1e4 / yes | 1e6 / yes |
| parameters | **426,752** | 156,096 | 596,049,920 |

In `qwen3-tiny`, `head_dim` happens to equal `128 / 4`. The model tests and
the HF parity tests therefore also run `tiny`, where `head_dim ≠ H / N`, and an
untied variant.

## 3. Parameters and weight tying

Tiny Qwen3: embedding 32,768 (7.7 %) · attention incl. QK-Norm 98,432 (23.1 %) ·
SwiGLU 294,912 (69.1 %) · block norms 512 · final norm 128 → **426,752 total,
all trainable**.

With `tie_word_embeddings = true`, `lm_head.weight` **is**
`model.embed_tokens.weight`: `self.lm_head.weight = self.model.embed_tokens.weight`
registers the same `nn.Parameter` object under both names. That means one
storage (`data_ptr` equal), one gradient, and `named_parameters()` lists it once.
Untied adds `V·H` parameters (+32,768 for Tiny; +155,582,464 for 0.6B). Tests
verify identity, shared storage, shared writes and the shared `.grad` object, and
independence when untied. There is no LM-head bias (`bias=False`, as in HF).

**Initialisation** follows HF `_init_weights`: Linear and Embedding weights
~ `N(0, initializer_range²)`, RMSNorm weights = 1. The tied matrix is initialised
once. Equal seeds do **not** give equal weights in two implementations, so parity
tests copy HF's `state_dict` into ours with `strict=True`.

## 4. Masks and positions

- `attention_mask` is a **padding mask** `[B, T]` (1/True = real token, 0/False =
  padding), combined with the causal mask by token index:
  `allowed[b, i, j] = (j ≤ i) ∧ mask[b, j]`. Float masks (additive biases such as
  0/−inf) are **rejected** with an explanatory `TypeError`. Pad token *ids* are
  never masked automatically; you must pass the mask.
- `position_ids` default to `0..T−1`. For left-padded batches, pass e.g.
  `cumsum(mask) − 1` (tested: real tokens then give the same logits as the
  unpadded sequence). Positions ≥ `max_position_embeddings`, and sequences longer
  than it, are rejected.
- Query rows of padding tokens have no allowed key. They get zero attention (finite
  outputs); HF gives a uniform row. Only padded positions differ, and they are
  never used.
- RoPE tables are computed **once per forward** and the same tensor object is
  passed to every layer (tested).

## 5. Causal LM loss

Teacher forcing feeds the true sequence. Position `t` (having read tokens `≤ t`,
enforced by the causal mask) predicts token `t+1`. Labels have the same shape as
`input_ids`, and the shift happens **once**, inside `causal_lm_loss`:
`targets = labels[:, 1:]`, with the last position padded with −100, exactly like
HF. That keeps the same row order, so the loss is bit-identical to HF.

- **Logits, not probabilities.** `F.cross_entropy` computes
  `−z_y + logsumexp(z)` with the max subtracted. `log(softmax(z))` done separately
  can underflow to `log 0`. The gradient w.r.t. the logits is
  `softmax(z) − onehot(y)` (tested).
- **Ignored targets.** `ignore_index = −100` (padding, prompt tokens). The mean is
  over valid targets only.
- **No valid targets** (`T = 1`, or everything ignored): we raise
  `NoValidTargetsError` instead of returning HF's `NaN` (0/0).
- The loss is computed in float32 (float64 for float64 logits); labels are
  validated (shape, integer dtype, range).
- At random init the loss is ≈ `ln 256 = 5.545` (tested within 0.2).

## 6. Greedy generation

`greedy_generate(model, ids, max_new_tokens, eos_token_id=None)` runs the full
prefix each step (no KV cache: O(T²) attention per step), takes `argmax` of the
last position's logits, and appends. It uses `eval()` + `torch.inference_mode()`
and restores the previous mode afterwards. Batches must be unpadded. It raises if
`T + max_new_tokens > max_position_embeddings`; there is no silent truncation.
Finished sequences keep emitting EOS, and generation stops when all have finished.
Each step is checked against the argmax of a fresh full forward.

## 7. Verification summary (CPU, torch 2.14.1, transformers 5.20.0)

| check | result |
|---|---|
| parameter names, shapes, values after copy, counts (426,752 = HF = config estimate), tying | identical, for Tiny, `tiny` (D ≠ H/N) and untied variants |
| embeddings, every layer output, final norm, logits, loss (with ignored labels) | **bit-identical** to HF `Qwen3ForCausalLM` (eager, eval) |
| gradients of **all** parameters after `loss.backward()` | **bit-identical** |
| left-padded batch with `attention_mask` + `position_ids` | bit-identical on all real tokens |
| bfloat16 forward | bit-identical (after accounting for HF's `.to(bf16)` RoPE buffer quirk, D-027) |
| float64 model vs float32 (Tiny, `(4, 64)`) | logits max abs 7.3e-07, loss 5.7e-07 |
| prefix leakage (identical prefix, different future) | shared-prefix logits identical; later ones differ |
| changing token 3 | logits 0–2 bitwise unchanged, 3–7 all change |

## 8. C++20 port contract (M3)

All tensors are row-major and contiguous, float32 (bf16 storage optional later),
with `N = B·T` rows.

| component | input | output | parameters | operations | notes / C++ interface |
|---|---|---|---|---|---|
| embedding | ids `[N]` int32 | `[N, H]` | `E [V, H]` | row gather | bounds-check ids; `embed(span<const int32_t>, const Tensor& E, span<float>)` |
| RoPE tables | positions `[T]` | cos, sin `[T, D/2]` | — (θ, D) | `pos · inv_freq` in fp32, `cos`/`sin` | compute once per forward; see [rope.md §12](rope.md) |
| mask | positions, padding `[B, T]` | implicit | — | `j ≤ i ∧ valid[j]` | never materialise `[T, T]`; loop bounds plus a padding check |
| decoder layer ×L | `[N, H]` | `[N, H]` | per layer: norms `[H]`×2; q `[Nq·D, H]`, k/v `[Nkv·D, H]`, o `[H, Nq·D]`, q/k norm `[D]`; gate/up `[I, H]`, down `[H, I]` | 2× RMSNorm, 7 GEMMs, per-head RMSNorm, RoPE, causal online softmax, SiLU·mul, 2 residual adds | in place on the residual buffer; see [decoder.md](decoder.md) and [attention.md §9](attention.md) |
| final norm | `[N, H]` | `[N, H]` | `w [H]` | RMSNorm (fp32 statistics) | [rmsnorm.md §9](rmsnorm.md) |
| LM head | `[N, H]` | `[N, V]` | `E` (tied) or `W [V, H]` | GEMM `z Eᵀ` | for generation compute only the last row (`[B, V]`) |
| loss | logits `[N, V]`, targets `[N]` | scalar | — | stable `logsumexp − z_y`, mean over valid | float64 accumulation acceptable |
| greedy step | logits `[B, V]` | ids `[B]` | — | argmax (lowest index on ties) | context limit check before decoding |

Memory: weights 4·params bytes (1.7 MB Tiny, 2.4 GB 0.6B in fp32). Activations
per layer are `O(N·(H + I + (Nq + 2·Nkv)·D))`. The online softmax avoids `O(T²)`.
Compute ≈ `2 · params · N` FLOPs for the matmuls, plus attention
`≈ 2·L·B·Nq·T²·D`.

**Fixture** (`tests/fixtures/model_tiny.json`, 43 KB, generated by
`python scripts/export_fixtures.py`, kept current by `tests/test_fixtures.py`):

- config: V = 16, H = 8, L = 2, 2/1 heads, D = 4, I = 12, tied;
- input ids `[[3, 14, 1, 5, 9]]`, position ids, an all-ones attention mask, and labels;
- all unique parameters (the tied matrix stored once, as `model.embed_tokens.weight`);
- expected embeddings, both layer outputs, final norm, logits `[1, 5, 16]` and loss;
- tolerance `rtol = atol = 1e-5`.

## 9. Not supported yet

KV cache · loading official safetensors checkpoints · Qwen3 BPE tokenizer and chat
template · sampling (temperature, top-k/p) · padded batched generation · dropout,
optimizer, training loop · sliding-window attention, RoPE scaling · GPU-specific
kernels.
