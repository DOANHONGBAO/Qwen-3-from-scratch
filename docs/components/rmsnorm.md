# RMSNorm (M1.1)

Implementation: [`src/qwen3_scratch/layers/rmsnorm.py`](../../src/qwen3_scratch/layers/rmsnorm.py) ·
Tests: [`tests/test_rmsnorm.py`](../../tests/test_rmsnorm.py),
[`tests/parity/test_rmsnorm_hf.py`](../../tests/parity/test_rmsnorm_hf.py) ·
Notebook: [`notebooks/01_rmsnorm.ipynb`](../../notebooks/01_rmsnorm.ipynb) ·
Error report: `python scripts/rmsnorm_report.py`

## 1. Why normalise, and why RMSNorm

A Transformer layer updates a residual stream: `x ← x + Attention(norm(x))`,
then `x ← x + MLP(norm(x))`. The magnitude of `x` changes from layer to layer and
from token to token. Without normalisation, each sublayer sees inputs of
unpredictable scale, which makes deep networks hard to train. Qwen3, like Llama,
uses **pre-norm**: it normalises the input of every sublayer and leaves the residual
stream itself unnormalised.

LayerNorm computes `(x − μ) / sqrt(σ² + ε) · γ + β`. That is two reductions per
token (mean and variance) plus a bias. **RMSNorm** (Zhang & Sennrich, 2019,
*Root Mean Square Layer Normalization*) keeps only the re-scaling:

- **No mean subtraction.** The authors found that LayerNorm's benefit comes from
  re-scaling invariance, not re-centering. Dropping μ saves one reduction per
  token and, in their experiments, cost no quality. The vector keeps its mean:
  RMSNorm is **not** invariant to adding a constant (tested).
- **No bias β.** The following linear layers can absorb any offset. Qwen3 uses no
  biases anywhere in its dense blocks.

The normalisation runs over the **hidden (last) dimension** because each token's
feature vector is the unit that sublayers consume. Normalising across tokens or
across the batch would mix information between positions, break causality and
make the result depend on the batch.

## 2. Definition, shapes, parameters

For a vector `x ∈ ℝ^d` (the last tensor axis):

```
RMS(x) = sqrt( (1/d) · Σ_i x_i² + ε )
y_i    = x_i / RMS(x) · γ_i
```

| tensor | shape | notes |
|---|---|---|
| input `x` | `(..., d)` | any leading dims: `(B, T, d)` for hidden states, `(B, T, H, d_h)` for QK-Norm |
| mean of squares `m` | `(..., 1)` | `keepdim=True` so it broadcasts back |
| `1/sqrt(m + ε)` | `(..., 1)` | `torch.rsqrt` |
| normalised `x̂` | `(..., d)` | each row has mean square `m / (m + ε) ≈ 1` |
| weight `γ` | `(d,)` | learnable, **initialised to ones**, HF name `weight` |
| output `y` | `(..., d)` | same shape as `x` |

The **learnable scale** `γ` lets the model restore a per-feature magnitude after
normalisation. For example, it can amplify features that the next projection needs
large. Initialising it to ones makes a fresh layer a pure normalisation. Qwen3
stores `γ` directly; some other models (e.g. Gemma) store `w` and compute
`(1 + w)` with `w` initialised to zero. **There is no such offset in Qwen3.** We
checked this against `Qwen3RMSNorm.__init__` (`torch.ones(hidden_size)`) and its
`forward`.

**ε (epsilon)** has two jobs:

1. It avoids division by zero for an all-zero vector. With ε, the output is 0;
   without it, the output is NaN.
2. It limits the gain applied to near-zero vectors. For `‖x‖_rms ≪ sqrt(ε) = 1e-3`,
   `y ≈ x / sqrt(ε)`, so noise is not blown up to unit size. For typical
   activations (`m ≈ 1`), ε changes the output by about `ε / (2m) ≈ 5·10⁻⁷`
   relative.

Qwen3 uses `ε = 1e-6` (`rms_norm_eps`) for every norm. Caution:
`torch.nn.RMSNorm(eps=None)` defaults to `torch.finfo(dtype).eps` (`1.19e-7` for
fp32, `7.8e-3` for bf16), so ε must be passed explicitly in comparisons.

## 3. Worked example (d = 4)

`x = [1, 2, 3, 4]`, `γ = [0.5, 1, −1, 2]`, `ε = 1e-6`

| step | value |
|---|---|
| squares | `[1, 4, 9, 16]`, sum `30` |
| mean of squares `m` | `30 / 4 = 7.5` |
| `m + ε` | `7.500001` |
| `RMS(x)` | `sqrt(7.500001) = 2.738612970100010…` |
| `x̂ = x / RMS` | `[0.365148, 0.730297, 1.095445, 1.460593]` |
| check: mean(`x̂²`) | `7.5 / 7.500001 = 0.99999987` |
| `y = γ ⊙ x̂` | `[0.182574, 0.730297, −1.095445, 2.921187]` |

The 30-digit values are used as constants in `tests/test_rmsnorm.py`. The float64
implementation matches them to `< 1e-14`.

## 4. Gradients

Let `u = ∂L/∂y`, `g = u ⊙ γ`, `r = sqrt(m + ε)`. Since `∂r/∂x_j = x_j / (d·r)`:

```
∂y_i/∂x_j = γ_i · ( δ_ij / r − x_i x_j / (d r³) )

∂L/∂x = g / r − x · (g · x) / (d r³)          # per row
∂L/∂γ = Σ_positions u ⊙ x̂                      # summed over all leading dims
```

The second term of `∂L/∂x` projects out the component along `x`:
`x · ∂L/∂x = ε (g·x) / r³ ≈ 0`. RMSNorm is (almost) scale invariant, so pushing
`x` along itself barely changes the output.

PyTorch autograd records the primitive ops (`to`, `pow`, `mean`, `add`, `rsqrt`,
`mul`) and applies each op's backward rule in reverse (the chain rule). No custom
`backward` is needed. The tests check autograd against the closed form above
(agreement `< 1e-12` in float64). They also run `torch.autograd.gradcheck` and
`gradgradcheck`, which compare against finite differences in float64.

## 5. Mixed precision

bfloat16 has an 8-bit significand: unit roundoff `2⁻⁸ ≈ 0.0039`, and one ulp is at
most `2⁻⁷ ≈ 0.0078` relative. Three precision decisions follow Hugging Face
`Qwen3RMSNorm` exactly:

1. **Statistics in float32.** `x` is upcast before squaring and averaging.
   Accumulating 1024 squares in bf16 loses precision, and in fp16, `|x| > 256`
   already overflows (`300² = inf`). The notebook demonstrates the fp16 overflow.
2. **Cast back before scaling.** `x̂` is rounded to the input dtype, *then*
   multiplied by `γ`. That is two roundings in bf16.
3. **Output dtype** = `promote_types(weight.dtype, x.dtype)`. In the model, both
   are bf16 (or both fp32).

The educational reference (Raschka, Appendix C) and `torch.nn.RMSNorm` instead
multiply by `γ` in float32 and cast once at the end ("scale-then-cast"). Both
orders are numerically fine, but they are not bit-identical. On random inputs,
**≈25 % of bf16 outputs differ by one ulp** (max abs `3.1e-2` for values near 4).
We follow HF so that M2 parity tests can be exact (decision D-013, discrepancy X5).

One deliberate deviation: a **float64** input stays float64 (`compute dtype =
promote_types(x.dtype, float32)`), whereas HF downcasts to float32. This keeps
`gradcheck` meaningful and never affects the model (D-014).

Known limitation, shared with HF: in float32, `x²` overflows for `|x| > ~1.8e19`.
Real activations are many orders of magnitude smaller.

## 6. Where RMSNorm appears in Qwen3

| module (HF name) | normalised tensor | `d` (0.6B) | count (0.6B) |
|---|---|---|---|
| `layers.i.input_layernorm` | hidden state before attention | 1024 | 28 |
| `layers.i.post_attention_layernorm` | hidden state before MLP | 1024 | 28 |
| `layers.i.self_attn.q_norm` | each query head after `q_proj` + reshape, **before RoPE** | 128 | 28 |
| `layers.i.self_attn.k_norm` | each key head after `k_proj` + reshape, **before RoPE** | 128 | 28 |
| `model.norm` | final hidden state before the LM head | 1024 | 1 |

That is 113 RMSNorm modules holding `28·(2·1024 + 2·128) + 1024 = 65,536`
parameters, about 0.01 % of the model. QK-Norm (new in Qwen3 versus Qwen2) uses one
`(head_dim,)` weight shared by all heads of that layer. It bounds the size of query
and key vectors and so stabilises attention logits. Qwen3 also drops Qwen2's QKV
biases (`attention_bias = false`).
QK-Norm itself is wired into attention in M1.4.

## 7. Verified results (CPU)

Measured with `python scripts/rmsnorm_report.py` (seed 0, `x ~ 2·N(0,1)`,
`γ ~ 1 + 0.2·N(0,1)`). "max_rel" is computed over elements with `|ref| ≥ 1e-3`.
Results are recorded in `docs/progress.md`. In summary:

| comparison | float32 max_abs / max_rel | bfloat16 max_abs / max_rel |
|---|---|---|
| vs float64 textbook (same rounded inputs) | `6.9e-7 / 2.4e-7` | `2.8e-2 / 7.5e-3` |
| vs `torch.nn.RMSNorm` (explicit ε) | `0 / 0` (bit-identical) | `3.1e-2 / 7.8e-3` (≈25 % elements, 1 ulp) |
| vs HF `Qwen3RMSNorm` | `0 / 0` (bit-identical) | `0 / 0` (bit-identical) |

Test tolerances: fp32 `rtol=1e-6, atol=1e-6` against references, bf16
`rtol=2·2⁻⁸` against the float64 textbook, and **exact equality** with HF.

## 8. Common mistakes

- Reducing over the wrong axis, or forgetting `keepdim=True`.
- A `(d, T)` layout, where the hidden dim is not last. Our module raises an error.
- Subtracting the mean or adding a bias: that turns RMSNorm into LayerNorm.
- `sqrt(m) + ε` instead of `sqrt(m + ε)`.
- Relying on default ε values of other libraries.
- Using a `1 + w` offset (Gemma style) with Qwen3 weights.
- Computing statistics in bf16/fp16.
- Casting after scaling when bit-exact HF parity is the goal.
- Applying QK-Norm after RoPE, or over the full `H·d_h` width instead of per head.

## 9. Requirements for the C++20 port (M3)

Not implemented yet. This is the contract the C++ module must satisfy.

**Memory layout.** Row-major, contiguous, last dimension stride 1. Any input
`(..., d)` is viewed as `N × d` with `N = Π leading dims`, and each row is
processed independently. Use `std::span<const T>` for input rows and
`std::span<T>` for output. Non-contiguous inputs are out of scope; callers
materialise contiguous tensors.

**Algorithm per row** (identical order to Python):
1. `sum = Σ float(x_i)²` accumulated in **float** (fp32). A `double` accumulator
   is allowed for a reference build but changes rounding.
2. `inv = 1.0f / std::sqrt(sum / d + eps)`. Do not use fast approximate rsqrt.
3. `h_i = float(x_i) * inv`, rounded to the storage type `T` (bf16:
   round-to-nearest-even via `std::bit_cast<uint32_t>`).
4. `y_i = T(float(h_i) * float(gamma_i))`, rounded to `T` again.

**Parameters.** `gamma` is a contiguous `d`-vector, loaded from the checkpoint key
`*.weight`. There is no offset; for construction without a checkpoint it defaults
to ones.

**C++20 operations needed:** elementwise square, row reduction (sum), scalar
`sqrt`/reciprocal, broadcast multiply, fp32↔bf16 conversion (`std::bit_cast`,
`<bit>`), `std::span`, and optionally `<execution>` or OpenMP parallelism over rows
in M4. The C++ port is inference only, so no backward pass is needed.

**Expected tolerances vs. Python fixtures.** PyTorch's CPU `mean` uses
vectorised/blocked summation, so a sequential C++ sum may differ in the last bits:

| dtype | tolerance |
|---|---|
| float32 | `rtol = 1e-5, atol = 1e-6` (observed float32-vs-float64 error is ~`2e-7` relative) |
| bfloat16 | equal, or differing by at most **1 ulp** (`rtol = 2⁻⁷`), on ≤ a few % of elements |

**Fixtures (to build in M3).** A Python script will export, for several shapes
(`d ∈ {1, 4, 128, 1024}`, `N ∈ {1, 7, 512}`) and dtypes (fp32, bf16): `x`, `gamma`,
`eps`, and the expected `y` from `qwen3_scratch.layers.RMSNorm`. Files will be raw
little-endian arrays (bf16 stored as `uint16` bit patterns) plus a JSON manifest
with shapes, dtypes, seeds and the generating commit. Because `*.bin`/`*.npy` are
git-ignored, fixtures will be generated as a CTest setup step rather than committed
(decision for M3). The manual example from §3 will be hard-coded in the C++ test as
a dependency-free sanity check.
