# SwiGLU feed-forward (M1.2)

Implementation: [`src/qwen3_scratch/layers/swiglu.py`](../../src/qwen3_scratch/layers/swiglu.py) ·
Tests: [`tests/test_swiglu.py`](../../tests/test_swiglu.py),
[`tests/parity/test_swiglu_hf.py`](../../tests/parity/test_swiglu_hf.py) ·
Notebook: [`notebooks/02_swiglu.ipynb`](../../notebooks/02_swiglu.ipynb) ·
Error report: `python scripts/swiglu_report.py` ·
C++ fixture: [`tests/fixtures/swiglu_small.json`](../../tests/fixtures/swiglu_small.json)

## 1. What the feed-forward network does

In each decoder block, attention moves information **between** token positions.
The feed-forward network (FFN, "MLP") then transforms every token vector
**independently** with the same weights. It is where most of the parameters live
(Qwen3-0.6B: 9.44M per layer, versus 6.29M for the attention projections).

## 2. From ReLU FFN to SwiGLU

| variant | formula (per token, row vector `x`) | weights | used in |
|---|---|---|---|
| ReLU FFN | `ReLU(x W₁) W₂` | 2 | original Transformer (2017) |
| GELU FFN | `GELU(x W₁) W₂` | 2 | BERT, GPT-2 |
| GLU | `(σ(x W_g) ⊙ x W_u) W_d` | 3 | Dauphin et al. (2017) |
| **SwiGLU** | `(SiLU(x W_g) ⊙ x W_u) W_d` | 3 | PaLM, Llama, **Qwen3** |

with `SiLU(z) = z · σ(z) = z / (1 + e^{−z})` (also called *swish*), and
`GELU(z) = z · Φ(z)`.

A **gated linear unit** computes two projections of the same input:

- `up = x W_upᵀ`: the candidate *values*;
- `gate = x W_gateᵀ`: per hidden unit and per token, how much of the value
  passes through. `SiLU(gate)` is ≈ 0 for very negative gates (closed), ≈ `gate`
  for large positive gates (open), and smooth in between.

The elementwise product `SiLU(gate) ⊙ up` makes the layer **multiplicative**.
For open gates, the output grows roughly quadratically in `x`
(`f(2x) ≈ 4 f(x)` per open unit, tested). An ordinary MLP applies a fixed
nonlinearity to a single projection; it cannot express "unit *k* is useful for
this token only if feature *j* is present" as directly.

Shazeer (2020, *GLU Variants Improve Transformer*) showed that gated variants,
SwiGLU among the best, reach lower perplexity at matched parameter count and
compute. To keep the parameter count comparable with a 2-matrix FFN of width
`4H`, gated FFNs use a smaller hidden width. Llama uses about `8H/3`; Qwen3-0.6B
uses `I = 3H = 3072`. SiLU is preferred to ReLU because it is smooth: it has a
non-zero gradient for negative inputs, so there are no "dead" units. It is also
self-gated and cheap to evaluate.

## 3. Shapes and worked example

```
x      [B, T, H] ──gate_proj (W: [I, H])──► gate   [B, T, I] ──SiLU──┐
   └──────────────up_proj   (W: [I, H])──► up     [B, T, I] ────────⊙──► hidden [B, T, I]
hidden ──down_proj (W: [H, I])──► out [B, T, H]
```

| tensor | shape | Qwen3-0.6B |
|---|---|---|
| `x`, `out` | `[B, T, H]` | H = 1024 |
| `gate_proj.weight`, `up_proj.weight` | `[I, H]` (`nn.Linear` stores `(out, in)`) | 3072 × 1024 |
| `gate`, `up`, `hidden` | `[B, T, I]` | I = 3072 |
| `down_proj.weight` | `[H, I]` | 1024 × 3072 |

The hidden dimension changes `H → I → H`. The block expands into a wider space
where the gate selects features, then projects back so the residual addition
`x + MLP(·)` works.

**Manual example** (`H = 2`, `I = 3`, `x = [1, −1]`):

| step | value |
|---|---|
| `W_gate = [[1,0],[1,1],[0,−2]]` → `gate` | `[1, 0, 2]` |
| `W_up = [[2,0],[0,1],[1,0]]` → `up` | `[2, −1, 1]` |
| `SiLU(gate)` | `[σ(1), 0, 2σ(2)] = [0.731059, 0, 1.761594]` |
| `hidden = SiLU(gate) ⊙ up` | `[1.462117, 0, 1.761594]` |
| `W_down = [[1,1,1],[0.5,−1,0]]` → `out` | `[3.223711, 0.731059]` |

Unit 2 shows the gate at work: its value `up = −1` is discarded because
`SiLU(0) = 0`. The test suite checks this example to `1e-15` in float64.

## 4. Gradients through both branches

With upstream `u = ∂L/∂out` and `δ = u W_down` (shape `[.., I]`):

```
∂L/∂W_down = uᵀ · hidden
∂L/∂up     = δ ⊙ SiLU(gate)                      # value branch, scaled by the gate
∂L/∂gate   = δ ⊙ up ⊙ SiLU'(gate)                # gate branch, scaled by the value
SiLU'(z)   = σ(z) · (1 + z (1 − σ(z)))
∂L/∂W_up   = (∂L/∂up)ᵀ · x,   ∂L/∂W_gate = (∂L/∂gate)ᵀ · x
∂L/∂x      = (∂L/∂gate) W_gate + (∂L/∂up) W_up    # both branches add up
```

Each branch's gradient is multiplied by the *other* branch's activation. A closed
gate blocks gradient to the value path, and a zero value gives the gate nothing to
learn. Autograd matches these formulas to `1e-16` (float64). `gradcheck` and
`gradgradcheck` pass for the module and for the functional form with respect to
all three weight matrices.

## 5. Qwen3-specific details (verified against HF `Qwen3MLP`)

- Names `gate_proj`, `up_proj`, `down_proj`: our `state_dict` loads HF's with
  `strict=True`.
- **No biases**: HF hardcodes `bias=False` for all three projections. This is
  independent of `config.attention_bias`. `SwiGLUFeedForward.from_config` always
  uses `bias=False`; the `bias` flag exists only to make the convention explicit.
- Activation `hidden_act = "silu"`, applied to `gate_proj` only.
- Weight initialisation does not matter for parity because weights are copied or
  loaded. Our module keeps PyTorch's `nn.Linear` default; HF uses
  `normal(0, initializer_range = 0.02)`.
- The educational reference (Raschka, Appendix C) names the same matrices `fc1`
  (gate, SiLU applied), `fc2` (up) and `fc3` (down). The math is identical; only
  the names differ (discrepancy X10).

## 6. Cost

- Parameters: `3·H·I` = **9,437,184** per layer for Qwen3-0.6B (264M across 28
  layers, 44 % of the model, 60 % of the non-embedding parameters).
- Forward FLOPs per token ≈ `2·3·H·I` ≈ **18.9 MFLOP** per layer, plus `O(I)` for
  SiLU and the product. That is ≈ 528 MFLOP per token for the 28 MLPs.
- Activation memory: `gate`, `up` and `hidden` are each `[B, T, I]`, which is 3×
  the hidden-state size.

## 7. Verified results (CPU)

`python scripts/swiglu_report.py`: 0.6B sizes, input `(2, 64, 1024)`, weights
`0.02·N(0,1)`. `rel_l2 = ‖a−b‖₂ / ‖b‖₂`. Elementwise relative error is not
reported because matmul outputs contain near-zero values.

| dtype | ours vs | max_abs | max_abs / max abs(ref) | rel_l2 |
|---|---|---|---|---|
| float32 | textbook float64 | 6.4e-7 | 5.1e-7 | 5.5e-7 |
| float32 | HF `Qwen3MLP` | **0** (bit-identical) | 0 | 0 |
| bfloat16 | textbook float64 | 5.7e-3 | 4.5e-3 | 3.9e-3 |
| bfloat16 | HF `Qwen3MLP` | **0** (bit-identical) | 0 | 0 |

Forward and backward are bit-identical to HF in fp32 and bf16, because the same
`nn.Linear` / `F.silu` kernels run in the same order.

## 8. Common mistakes

- Swapping `gate_proj` and `up_proj`: shapes still match but the outputs are
  wrong. Only a reference comparison catches it.
- Applying SiLU to `up`, to both branches, or to the product.
- `sigmoid(gate) * up` is GLU, not SwiGLU.
- Adding biases, or tying their presence to `attention_bias`.
- Forgetting that `nn.Linear` computes `x @ W.T` with `W` stored as `(out, in)`.
- Assuming `intermediate_size = 4·hidden_size`; always read it from the config.

## 9. Use in the Qwen3 decoder block (M1.7)

```
h = x + Attention(input_layernorm(x))
y = h + SwiGLUFeedForward(post_attention_layernorm(h))
```

The MLP is the second sublayer, preceded by RMSNorm and wrapped in a residual
connection.

## 10. Requirements for the C++20 port (M3)

Not implemented yet.

- **Matmuls.** Three GEMMs per token batch: `G = X·W_gateᵀ`, `U = X·W_upᵀ`
  (`[N, H] × [H, I]`) and `Y = Hid·W_downᵀ` (`[N, I] × [I, H]`), with
  `N = B·T` rows in row-major order. Weights keep the checkpoint layout
  `(out, in)` row-major, so each output element is a dot product of an input row
  with a weight row (contiguous for both, cache friendly). Accumulate in float32.
- **Bias.** None. The API should not even accept a bias for Qwen3; a generic
  `Linear` may support one, defaulting to off.
- **SiLU.** `z / (1 + std::exp(-z))` in float32. Use the form
  `z * sigmoid(z)` with a numerically stable sigmoid for large `|z|`
  (`std::exp(-z)` overflows to `inf` for `z < −88`, which gives the correct limit
  0 but should be tested). Elementwise, fused with the product.
- **Fusion and memory.** Compute `hidden[n, k] = silu(G[n, k]) * U[n, k]` in place
  into `G`'s buffer, so only one `[N, I]` intermediate plus `U` is live. A later
  optimisation can compute `gate` and `up` row-by-row to avoid materialising both
  (M4).
- **bf16.** Store bf16, convert to fp32 for the dot products, and round
  results to bf16 at the same points as PyTorch (after each `Linear` and after
  the product) for bit-level parity attempts. Otherwise use a tolerance.
- **Tolerance vs. fixture** (`tests/fixtures/swiglu_small.json`, float32,
  `H = 4`, `I = 6`, `x: (2, 3, 4)`): `rtol = 1e-5, atol = 1e-6`. Different GEMM
  summation orders change the last bits.
- **Fixture format.** JSON with `config`, `inputs.x`, `parameters.{gate,up,down}_proj.weight`
  and `outputs.y`, each with `shape`, `dtype` and nested `data`. Values are exact
  float32 numbers. Regenerate with `python scripts/export_fixtures.py`;
  `tests/test_fixtures.py` keeps them in sync with the Python module.
