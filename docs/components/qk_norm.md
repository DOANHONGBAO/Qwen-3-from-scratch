# QK-Norm (M1.4, part A)

Implementation: [`src/qwen3_scratch/layers/qk_norm.py`](../../src/qwen3_scratch/layers/qk_norm.py)
(reuses [`RMSNorm`](rmsnorm.md)) ·
Tests: [`tests/test_qk_norm.py`](../../tests/test_qk_norm.py),
[`tests/parity/test_qk_norm_hf.py`](../../tests/parity/test_qk_norm_hf.py) ·
Used by: [grouped-query attention](attention.md)

## 1. Q, K, V and attention logits

For a token with hidden state `x ∈ ℝ^H`, attention computes three projections per head:

- **query** `q = x W_qᵀ`: what this token is looking for;
- **key** `k = x W_kᵀ`: what this token offers to be found by;
- **value** `v = x W_vᵀ`: the content that is copied when the token is attended.

The logit of query position `i` for key position `j` is `s_ij = q_i · k_j / √D`.
Softmax over `j` turns logits into weights, and the output is `Σ_j p_ij v_j`.

**Why magnitudes matter.** `|q·k| ≤ ‖q‖ ‖k‖`. If `q` and `k` have independent
unit-variance entries, `q·k` has variance `D`. Dividing by `√D` restores unit
variance; that is the standard scaling. During training, however, the *norms*
of `q` and `k` can grow without bound, because nothing constrains `W_q`, `W_k` or
the scale of the residual stream. Logits then grow quadratically with that scale,
softmax saturates to one-hot, gradients vanish for all other keys, and training
can diverge ("attention logit growth"). `√D` fixes the *dimension* effect, not
the *norm* effect.

## 2. What QK-Norm does

Each head vector of `q` and `k` is RMS-normalised over its own `D` features,
with learnable scales `γ_q, γ_k ∈ ℝ^D`:

```
q̂_{b,h,t} = RMSNorm_q(q_{b,h,t}) = q_{b,h,t} / rms(q_{b,h,t}) ⊙ γ_q
k̂_{b,j,t} = RMSNorm_k(k_{b,j,t}) = k_{b,j,t} / rms(k_{b,j,t}) ⊙ γ_k
```

With unit `γ`, `‖q̂‖ = ‖k̂‖ ≈ √D`, so `|q̂·k̂| / √D ≤ √D` (≈ 11.3 for D = 128),
whatever the input scale. The tests check this bound for inputs scaled by
`10⁻²…10³`. Without QK-Norm, the same inputs scaled by 10³ give logits above 10⁵.
`γ_q ⊙ γ_k` lets the model learn a per-feature "temperature", so attention can
still become sharp when it needs to.

| | hidden-state RMSNorm (`input_layernorm`) | QK-Norm (`q_norm`, `k_norm`) |
|---|---|---|
| normalises | the residual stream entering the block, `[B, T, H]` | each projected head vector, `[B, N, T, D]` |
| reduction axis | `H` (1024) | `D` (128), per head |
| weight shape | `(H,)` | `(D,)`, one for Q and one for K |
| purpose | well-scaled input to every sublayer | bounded attention logits |

**Why per head, over `D`.** Each head computes its own dot products in its own
`D`-dimensional subspace, so the quantity to control is each head vector's norm.
Normalising over the full `N·D` width would let one head's large values shrink
the others. HF's source marks this choice: "unlike olmo, only on the head dim".

**Why separate γ for Q and K, shared over tokens and heads.** Queries and keys
play different roles and come from different projections, so each gets its own
scale. The scale is a property of the feature dimension inside a head, the same
for every token (position independence) and every head of that projection. That
gives `2·D = 256` parameters per layer for Qwen3-0.6B. The gradient of `γ_q` is
summed over all batch, head and token positions (tested).

**Why not V.** Values are not compared with each other; they are averaged. Their
scale does not affect the softmax and is already handled by `o_proj` and the
next RMSNorm. The HF module has no `v_norm` (tested).

**Why after the projections and before RoPE.** The projection creates the
vectors whose norm we want to control. RoPE is an orthogonal rotation that
preserves norms, so normalising before it gives the same bound. Normalising
*after* RoPE would multiply the rotated vectors elementwise by `γ`. That breaks
the property that the score depends only on relative position, because `γ` does
not commute with the rotation of each feature pair (`γ_i ≠ γ_{i+D/2}`).
Normalising first keeps RoPE a pure rotation of the already normalised vectors.

**Why it does not replace `1/√D`.** Qwen3 keeps the `D^{-1/2}` scaling. The two
are complementary: the scaling sets the expected logit size for normalised
vectors, and QK-Norm prevents norm growth. Removing the scaling would make the
logits `√D` times larger at initialisation.

## 3. Shapes and numerical example

| tensor | shape (0.6B) |
|---|---|
| `q_proj(x)` → split heads | `[B, T, 16·128]` → `[B, 16, T, 128]` |
| `k_proj(x)` → split heads | `[B, T, 8·128]` → `[B, 8, T, 128]` |
| `q_norm.weight`, `k_norm.weight` | `(128,)` each |
| output | same shapes; `v` untouched |

Example (`D = 2`, `γ_q = [2, 0.5]`): `q = [3, 4]` has mean square 12.5 and
`rms = √(12.5 + 10⁻⁶) ≈ 3.5355`, giving `q̂ = [2·3/3.5355, 0.5·4/3.5355] =
[1.6971, 0.5657]`. `k = [−1, 1]` with `γ_k = 1` gives `k̂ ≈ [−1, 1]`.

## 4. Implementation choice

The existing `RMSNorm` (M1.1) already reduces over the last dimension of any
`(..., D)` tensor in HF precision order, so **no change to RMSNorm was needed**.
`apply_qk_norm(q, k, q_norm, k_norm)` validates shapes (`[B, N, T, D]`, matching
`B`/`T`, `D == head_dim`) and calls the two norms. There is no reshape, and
different head counts for Q and K are supported.

- `QKNorm` is a small module holding `q_norm` and `k_norm`, for teaching and tests.
- `GroupedQueryAttention` holds `q_norm` / `k_norm` **directly**, so its parameter
  names match the checkpoint (`self_attn.q_norm.weight`), and calls the same
  `apply_qk_norm` function (decision D-021).

HF normalises in the `[B, T, N, D]` layout before the transpose; we normalise
after it. The vectors are the same, and the results are **bit-identical** in fp32
and bf16 (tested against `Qwen3Attention.q_norm/k_norm` for MHA, MQA and GQA head
counts).

## 5. Discrepancies with the educational reference

Raschka's Appendix C applies the same per-head RMSNorm, after `view` +
`transpose`, but hard-codes `eps = 1e-6` and uses the "scale-then-cast" precision
order (X3, X5). We take ε from the config and follow HF's order.

## 6. Common mistakes

- Normalising over `N·D` (the whole projection) instead of per head.
- One shared weight for Q and K, or a separate weight per head.
- Normalising `v`, or applying the norm after RoPE.
- Dropping the `1/√D` scaling because "QK-Norm already normalises".
- Missing the norm weights when loading a checkpoint (`strict=True` catches this).
