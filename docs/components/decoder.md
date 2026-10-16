# Qwen3 decoder block (M1.5)

Implementation: [`src/qwen3_scratch/layers/decoder.py`](../../src/qwen3_scratch/layers/decoder.py) ·
Tests: [`tests/test_decoder.py`](../../tests/test_decoder.py),
[`tests/parity/test_decoder_hf.py`](../../tests/parity/test_decoder_hf.py) ·
Notebook: [`notebooks/05_decoder_block.ipynb`](../../notebooks/05_decoder_block.ipynb) ·
Error report: `python scripts/decoder_report.py`

## 1. From the original Transformer to Qwen3

| architecture | sublayers per block | normalisation |
|---|---|---|
| Transformer decoder (Vaswani 2017) | masked self-attention → **cross-attention to the encoder** → FFN | **post-norm**: `LN(x + f(x))` |
| decoder-only LM (GPT family) | masked self-attention → FFN | pre-norm (GPT-2 onward): `x + f(LN(x))` |
| **Qwen3** | causal GQA self-attention (QK-Norm, RoPE) → SwiGLU MLP | pre-norm with **RMSNorm** |

A decoder-only language model reads and predicts one token stream. There is no
encoder output to attend to, so cross-attention does not exist. Causality comes
from the attention mask.

## 2. The two equations

```
h1 = x  + SelfAttention( RMSNorm_1(x)  )     # input_layernorm, self_attn
y  = h1 + SwiGLU       ( RMSNorm_2(h1) )     # post_attention_layernorm, mlp
```

```
x [B,T,H] ──┬──────────────────────────────────────────────(+)── h1 [B,T,H] ──┬─────────────────────────────(+)── y [B,T,H]
            └─ input_layernorm ─ self_attn (QK-Norm, RoPE, GQA) ─┘              └─ post_attention_layernorm ─ mlp ─┘
```

- **Self-attention** mixes information *across positions*. It is the only place
  where token `t` reads tokens `< t`.
- **SwiGLU** transforms *each position independently* (no mixing), using most of
  the parameters (9.44M of the 15.73M per 0.6B block).
- **Residual connections.** Each sublayer computes an update that is *added* to
  the residual stream: `∂y/∂x = I + ∂(sublayers)/∂x`. The identity term carries
  gradients through any number of layers unchanged. That makes very deep networks
  optimisable: no vanishing through repeated products, and a zero-initialised
  sublayer starts as the identity.
- **Pre-norm.** Each sublayer reads a normalised copy (`RMSNorm(x)`), so its input
  scale is controlled no matter how large the stream has grown. The stream itself
  is never normalised inside the block, which keeps the identity path clean. The
  stream grows with depth; hence the final `model.norm` before the LM head (M1.6).
  Post-norm (`norm(x + f(x))`) puts a normalisation on the identity path at every
  step. Xiong et al. (2020) show that it needs learning-rate warm-up to train
  stably. Notebook 05 shows a random 24-block stack in which the post-norm input
  gradient varies about 2,000× across depths, versus about 13× for pre-norm.
- **Positions** enter only through RoPE inside attention. Rotations of `q`/`k`
  make the scores depend on relative offsets without adding anything to the
  stream. One `(cos, sin)` pair per forward is shared by all layers.

## 3. Implementation

`Qwen3DecoderLayer` holds `input_layernorm`, `self_attn` (the M1.4
`GroupedQueryAttention`), `post_attention_layernorm` and `mlp` (the M1.2
`SwiGLUFeedForward`). Parameter names equal HF's, so
`load_state_dict(hf_layer.state_dict(), strict=True)` works. The forward follows
the nine steps literally: save residual → norm → attention → add → save → norm →
MLP → add → return. `return_trace=True` also returns a `DecoderTrace` with every
intermediate, including the attention trace. No component is re-implemented.

Parameters (0.6B): `input_layernorm` 1,024 · `self_attn` 6,291,712 (incl. QK-Norm 256) ·
`post_attention_layernorm` 1,024 · `mlp` 9,437,184 → **15,730,944 per block**
(equal to `Qwen3Config.estimate_num_parameters` per layer, tested).

## 4. Verification

| check | result |
|---|---|
| intermediates vs. a manual composition of the submodules | bit-identical |
| post-norm, swapped sublayers, swapped norms | all differ (tested) |
| `o_proj = 0` and `down_proj = 0` → identity | output **exactly** `x`; gradient exactly the upstream |
| only attention / only MLP active | exactly `x + attn(norm1(x))` / `x + mlp(norm2(x))` |
| scaling `x` by 10 (pre-norm property) | update `y − x` unchanged to 1e-5 |
| each norm's weight affects only its own sublayer | tested |
| prefix leakage through the block | none |
| `gradcheck` (float64) | passed |
| HF `Qwen3DecoderLayer` (eager, additive causal mask, explicit `position_embeddings`) | input norm, attention output, `h1`, post-attention norm, MLP output and final output **bit-identical** for GQA/MHA/MQA in fp32 and bf16; fp32 gradients for the input and all 11 parameters bit-identical |

`python scripts/decoder_report.py` (0.6B sizes, `x: (1, 64, 1024)`):

| dtype | vs | tensor | max_abs | mean_abs | rel_l2 |
|---|---|---|---|---|---|
| float32 | float64 layer | output | 1.0e-06 | 1.6e-07 | 2.0e-07 |
| float32 | float64 layer | dL/dx | 2.8e-06 | 2.6e-07 | 3.0e-07 |
| float32 / bf16 | HF `Qwen3DecoderLayer` | output, dL/dx | **0** | **0** | **0** |
| bfloat16 | float64 layer | output | 2.6e-02 | 3.0e-03 | 3.8e-03 |

## 5. Common mistakes

- Post-norm wiring, or normalising the stream itself.
- Adding the *normalised* tensor to the residual (`h1 = norm(x) + attn`).
- Swapping the two norms. Both are `(H,)`, so loading succeeds silently; only
  parity tests catch it.
- MLP before attention; an extra cross-attention.
- Absolute position embeddings added to `x`; different RoPE tables per layer.

## 6. C++ port notes (M3)

`y = x + attn(rmsnorm(x, w1)); y += mlp(rmsnorm(y, w2))` on a contiguous
`[N, H]` buffer. Two residual adds are in-place `axpy`s. Two scratch buffers of
size `[N, H]` (normalised input, sublayer output) plus the attention and MLP
workspaces suffice. Interface:
`void decoder_layer(const LayerWeights&, std::span<float> x /*[N,H], in-place*/, const RopeTables&, const Mask&, Workspace&)`.
