# Rotary Position Embedding (M1.3)

Implementation: [`src/qwen3_scratch/layers/rope.py`](../../src/qwen3_scratch/layers/rope.py) ·
Tests: [`tests/test_rope.py`](../../tests/test_rope.py),
[`tests/parity/test_rope_hf.py`](../../tests/parity/test_rope_hf.py) ·
Notebook: [`notebooks/03_rope.ipynb`](../../notebooks/03_rope.ipynb) ·
Error report: `python scripts/rope_report.py` ·
C++ fixture: [`tests/fixtures/rope_small.json`](../../tests/fixtures/rope_small.json)

## 1. Why positions are needed

Scaled dot-product attention is **permutation-equivariant**: shuffle the tokens
and the outputs are shuffled the same way. The score `q_m · k_n` does not know
that token `n` came 3 positions before token `m`. The causal mask leaks a little
order information, but language needs explicit positions.

| approach | mechanism | properties |
|---|---|---|
| learned absolute (GPT-2) | add trained `p_m` to the token embedding | simple; nothing learned beyond the trained length |
| sinusoidal absolute (Vaswani 2017) | add fixed `sin/cos(m·ω_i)` | parameter-free; relative distance only implicit |
| relative (Shaw 2018, T5) | bias or embedding of `n − m` inside attention | explicitly relative; extra parameters or compute |
| **RoPE** (Su et al. 2021) | **rotate** `q` and `k` by angles proportional to position | relative scores, parameter-free, attention formula unchanged |

RoPE suits causal LMs. It does not change the residual stream (no added vectors).
It works with any attention kernel. And because keys are rotated once at their own
position, they can be cached and reused unchanged as decoding proceeds (§7).

## 2. From a 2D rotation to a head vector

```
R(θ) = [ cos θ  −sin θ ]        R(α)ᵀ R(β) = R(β − α)
       [ sin θ   cos θ ]
```

For a 2-vector rotated by `m·ω` (query) and `n·ω` (key):

```
⟨R(mω) q, R(nω) k⟩ = qᵀ R(mω)ᵀ R(nω) k = qᵀ R((n − m) ω) k
```

The score depends only on the **relative offset** `n − m`. A head vector of size
`D` (even) is split into `D/2` independent pairs, and pair `i` gets its own
frequency `ω_i`. The full rotation is block-diagonal, so the score is a sum of
`D/2` such 2D terms, each a function of `(n − m)·ω_i`.

## 3. Frequencies, theta, positions, tables

```
ω_i = θ^(−2i/D),  i = 0 … D/2 − 1            (inverse frequencies, float32)
angle[p, i] = p · ω_i                         (p = token position)
cos/sin tables: [T, D] = cos/sin of cat(angle, angle)   (each angle used twice)
```

- `ω_0 = 1` rad/token: the fastest pair, with a wavelength of 2π ≈ 6.3 tokens. It
  resolves local order.
- `ω_{D/2−1} = θ^(−(D−2)/D) ≈ 1/θ`: the slowest pair. Qwen3-0.6B (`D = 128`,
  `θ = 10⁶`) has `ω_63 = 1.24·10⁻⁶`, a wavelength of 5.1 million tokens. It
  separates distant positions.
- **θ** (`rope_theta`) sets how far the geometric sequence stretches. Qwen3's
  `10⁶` (versus the original `10⁴`) keeps slow pairs distinguishable over 32K-token
  contexts.
- **Positions** are integers `0, 1, …`. With a KV cache, a chunk of `T` new tokens
  uses `offset … offset+T−1`, where `offset` is the number of cached tokens.
  `position_ids` may be `[T]` (shared) or `[B, T]` (per sequence, e.g. padded batches).

## 4. Rotation convention: split-half (`rotate_half`)

Qwen3 / Hugging Face pair feature `i` with feature `i + D/2`:

```
y = x ⊙ cos + rotate_half(x) ⊙ sin,     rotate_half([x₁, x₂]) = [−x₂, x₁]
```

That is, for each `i < D/2`:
`y_i = x_i cos − x_{i+D/2} sin` and `y_{i+D/2} = x_{i+D/2} cos + x_i sin`.

**Worked example** (`D = 4`, `θ = 10⁴` → `ω = [1, 0.01]`, position 1, `x = [1, 2, 3, 4]`):

| pair | features | angle | result |
|---|---|---|---|
| 0 | `(x₀, x₂) = (1, 3)` | 1 rad | `y₀ = cos1 − 3 sin1 = −1.984111`, `y₂ = 3 cos1 + sin1 = 2.462378` |
| 1 | `(x₁, x₃) = (2, 4)` | 0.01 rad | `y₁ = 2 cos0.01 − 4 sin0.01 = 1.959901`, `y₃ = 4 cos0.01 + 2 sin0.01 = 4.019800` |

Tests check this to `1e-15` (float64).

**Mathematical equivalence vs. checkpoint compatibility.** The original RoFormer
paper pairs *adjacent* features `(x_{2i}, x_{2i+1})` ("interleaved"). The two
conventions are the same mathematics up to the fixed permutation
`perm = [0, 2, 4, …, D−2, 1, 3, …, D−1]`
(`half(x[perm]) = interleaved(x)[perm]`, tested). Attention scores are identical
**if** the `q_proj`/`k_proj` output rows (and QK-Norm weights) are permuted the
same way. They are **not** interchangeable for a given checkpoint. Running Qwen3
weights through interleaved RoPE without permuting the weights changes the scores
by order 1–10 in the notebook example, with no error raised. We implement split-half
only. `apply_rope_interleaved` exists purely for that demonstration.

## 5. Layouts and broadcasting

| tensor | shape | note |
|---|---|---|
| `position_ids` | `[T]` or `[B, T]` | integer, non-negative |
| `cos`, `sin` | `[T, D]` or `[B, T, D]` | no head axis |
| inside `apply_rope` | `[1 or B, 1, T, D]` | head axis inserted, broadcast over heads |
| `q` | `[B, Nq, T, D]` | Qwen3-0.6B: `Nq = 16`, `D = 128` |
| `k` | `[B, Nkv, T, D]` | `Nkv = 8`; same tables as `q` |
| `v` | not rotated | |

`apply_rope_to_qk(q, k, cos, sin)` requires the same `B`, `T` and `D` for `q` and
`k`; only the head count may differ (GQA). The tables must already have the
activation dtype, otherwise an error is raised (see §8).

## 6. Effect on attention, Q/K only, extrapolation

- **Attention scores.** `score(m, n) = Σ_i ⟨q⁽ⁱ⁾, R((n − m) ω_i) k⁽ⁱ⁾⟩`. Fast pairs
  make the score sensitive to small offsets; slow pairs carry long-range position.
  For identical `q` and `k`, the score is maximal at distance 0 and tends to shrink
  and oscillate with distance (notebook §6), a soft locality bias the content can
  override.
- **Why not V.** Positions should influence *which* tokens are attended (the
  weights), not *what* is copied. The relative property comes from the
  `q`/`k` dot product; rotating `v` would make outputs depend on absolute
  positions.
- **Norms are preserved** (orthogonal map), so RoPE does not change logit scale.
  With float32 tables, `cos² + sin² = 1` holds only to float32 accuracy (≈ 1e-7
  relative).
- **Extrapolation.** Beyond the training length (Qwen3-0.6B: 32,768), slow pairs
  reach angles never seen in training and quality degrades. Methods such as
  YaRN/NTK scaling change `ω_i`. They are configured through `rope_scaling`, which
  our config rejects (`NotImplementedError`) until needed.

## 7. RoPE and the KV cache

Keys are rotated **once**, at their own absolute position, then stored. A new
token at position `t = cache_length` is rotated with `position_ids = [t]`.
Because RoPE is elementwise per position:

- rotating chunk by chunk with offsets gives bit-identical tensors to rotating the
  full sequence (tested);
- the scores of the new query against the cached keys equal row `t` of the
  full-sequence score matrix (tested with a simulated cache, without a cache class);
- restarting positions at 0 for each step is a classic bug: it destroys relative
  positions (notebook §8).

The real `KVCache` comes in M2.

## 8. Qwen3-specific details and precision (verified against HF)

- `head_dim = 128` is rotated **in full** (no partial rotary). `rope_theta = 1e6`,
  `rope_scaling = null`, `attention_scaling = 1.0` (default RoPE type).
- `inv_freq = 1 / θ^(arange(0, D, 2) / D)` in float32. The same expression as HF
  gives **bit-identical** values.
- Angles `position · inv_freq` and their cos/sin are computed in float32, then
  **cast to the activation dtype before rotating** (HF `Qwen3RotaryEmbedding.forward`).
  The educational reference keeps fp32 tables and rotates in fp32, casting once at
  the end (discrepancy X6). In bf16, that order differs from HF by 1 ulp on
  24–38 % of elements (`scripts/rope_report.py`). We follow HF (D-019).
- HF computes the tables per forward call from `position_ids`. The educational
  reference precomputes the full table and slices it (X12). Our `RotaryEmbedding`
  does either: lookup with `max_position_embeddings`, or direct computation. Both
  are bit-identical (tested).
- The tables are buffers, not parameters, and are not part of the checkpoint
  (`state_dict` is empty).
- Order inside attention (M1.4): `q_proj` → reshape → `q_norm` (RMSNorm over
  `D`) → transpose → RoPE.

**float32 limits** (inherited from the HF implementation):
angle error grows with position. Max angle error is `2.9e-4` rad for positions
< 4096 and `2.4e-3` rad up to 32,767. As a result, the relative-position identity
`score(m+s, n+s) = score(m, n)` holds to `3e-6` for `s = 1` but only to `6e-3`
for `s = 32000` (200 random unit-variance vectors, `D = 128`).

## 9. Verified results (CPU)

`python scripts/rope_report.py`, `q: (1, 16, 512, 128)`:

| offset | dtype | ours vs | max_abs | rel_l2 |
|---|---|---|---|---|
| 0 | float32 | float64 truth | 1.2e-4 | 2.5e-6 |
| 32256 | float32 | float64 truth | 8.6e-3 | 2.8e-4 |
| 0 / 8192 / 32256 | float32, bf16 | **HF** `apply_rotary_pos_emb` | **0** | **0** |
| 32256 | bfloat16 | float64 truth | 3.1e-2 | 2.7e-3 |

Gradients: `∂L/∂x = R(−angle) ∂L/∂y` (orthogonality) to `1e-14`. `gradcheck`
passes with respect to `x`, `cos` and `sin`. Backward is bit-identical to HF.

## 10. Common mistakes

- Interleaved vs split-half pairing (§4), or forgetting `cat([angle, angle])`.
- Rotating `v`; rotating before QK-Norm.
- Positions restarting at 0 during decoding; ignoring per-sequence offsets.
- Broadcasting the tables over the wrong axis (they need `[B, 1, T, D]`).
- `head_dim = hidden_size / num_heads` (64) instead of 128; theta `1e4` instead of `1e6`.
- Building tables in bf16: `30000 · ω` cannot be represented accurately.

## 11. Use in the Qwen3 attention block (M1.4–M1.6)

```
q = q_norm(q_proj(x).view(B, T, Nq, D)).transpose(1, 2)    # [B, Nq, T, D]
k = k_norm(k_proj(x).view(B, T, Nkv, D)).transpose(1, 2)   # [B, Nkv, T, D]
cos, sin = rotary(position_ids, dtype=x.dtype)             # once per forward, shared by layers
q, k = apply_rope_to_qk(q, k, cos, sin)
# -> KV cache append (M2), repeat k/v for GQA, causal softmax(q kᵀ / √D) v
```

## 12. Requirements for the C++20 port (M3)

Not implemented yet.

- **Frequencies.** `inv_freq[i] = 1.0f / std::pow(theta, float(2*i) / D)`.
  `std::pow` in float can differ from PyTorch's vectorised `pow` in the last bit.
  Either accept a tolerance or load `inv_freq` from the fixture or exported
  buffer for bit-exact tests.
- **Tables.** `angle = float(pos) * inv_freq[i]` (float32 multiply, like HF);
  `cos/sin` via `std::cos`/`std::sin` on float. Different libm implementations
  differ by ~1 ulp. Precompute `[max_pos, D/2]` tables once (storing only half,
  because the second half duplicates the first), or compute per step.
- **Rotation** (split-half, per head row of length `D`, contiguous):
  ```
  for i in 0..D/2-1:  a = x[i], b = x[i + D/2], c = cos[p][i], s = sin[p][i]
                      y[i] = a*c - b*s;   y[i + D/2] = b*c + a*s
  ```
  This is equivalent to `x*cos + rotate_half(x)*sin` without materialising
  `rotate_half`. It can be done in place.
- **Position indexing.** The caller passes absolute positions (`cache_len + t`).
  No implicit reset per call.
- **Broadcasting.** Loop over `B × N × T` rows. The table row depends on
  `(b, t)` only, never on the head, so Q (`Nq` heads) and K (`Nkv` heads) share it.
- **bf16.** Round the tables to bf16 before the rotation and round the result
  (HF order) to match PyTorch. Otherwise use a tolerance.
- **Tolerance vs. fixture** (`tests/fixtures/rope_small.json`: `D = 8`,
  `θ = 10⁴`, `q: (1, 2, 3, 8)`, `k: (1, 1, 3, 8)`, positions 5–7): `rtol = atol = 1e-6`
  for `cos`, `sin`, `q_rot` and `k_rot`. `inv_freq` is included for exact reuse.
