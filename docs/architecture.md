# Architecture

This document describes the Qwen3 dense architecture as implemented (or planned)
in this repository, the module boundaries shared by Python and C++, and the
testing conventions. **During M0 nothing below is implemented except
`Qwen3Config`**; each section lists the milestone that delivers it.

Notation: `B` batch, `T` new tokens in this forward pass, `S` total key length
(`S = past + T` with a KV cache), `d` = `hidden_size`, `H` = `num_attention_heads`,
`H_kv` = `num_key_value_heads`, `g = H / H_kv` (GQA group size), `d_h` =
`head_dim`, `d_ff` = `intermediate_size`, `V` = `vocab_size`.

## 1. Forward pass overview

```
token ids (B, T)
  └─ Embedding                    -> x (B, T, d)
  └─ N × DecoderBlock
       h = x + Attention(RMSNorm(x))          # pre-norm residual
       x = h + SwiGLU(RMSNorm(h))
  └─ final RMSNorm                -> (B, T, d)
  └─ LM head (weight tied to Embedding when tie_word_embeddings)
                                   -> logits (B, T, V)
```

Attention inside a block:

```
q = x Wq -> (B, T, H,    d_h) -> RMSNorm over d_h (q_norm) -> (B, H,    T, d_h) -> RoPE
k = x Wk -> (B, T, H_kv, d_h) -> RMSNorm over d_h (k_norm) -> (B, H_kv, T, d_h) -> RoPE
v = x Wv -> (B, T, H_kv, d_h)                               -> (B, H_kv, T, d_h)
[KV cache: append k, v along the sequence axis -> (B, H_kv, S, d_h)]
repeat k, v g times along the head axis       -> (B, H, S, d_h)
scores = q kᵀ / sqrt(d_h) + causal mask       -> (B, H, T, S)
out    = softmax(scores) v                    -> (B, H, T, d_h) -> (B, T, H·d_h) -> Wo -> (B, T, d)
```

## 2. Components (planned)

| # | Component | Math (summary) | In → out shape | Milestone |
|---|---|---|---|---|
| 1 | RMSNorm | `y = x / sqrt(mean(x², -1) + ε) ⊙ w` (computed in fp32) | `(..., n) → (..., n)` | M1.1 |
| 2 | SwiGLU FeedForward | `down( silu(gate(x)) ⊙ up(x) )` | `(B,T,d) → (B,T,d)` via `d_ff` | M1.2 |
| 3 | RoPE | rotate pairs `(x_i, x_{i+d_h/2})` by `p·θ^{-2i/d_h}` ("rotate_half" layout) | `(B,H,T,d_h) → same` | M1.3 |
| 4 | QK-Norm | RMSNorm over `d_h` of each Q and K head, **before** RoPE | `(B,T,H,d_h) → same` | M1.4 |
| 5 | Causal mask | position `i` may attend to keys `j ≤ past + i` | `(T, S)` bool | M1.5 |
| 6 | Grouped Query Attention | see §1; `H_kv` KV heads shared by `g` query heads | `(B,T,d) → (B,T,d)` | M1.6 |
| 7 | Decoder block | two pre-norm residual sublayers | `(B,T,d) → (B,T,d)` | M1.7 |
| 8 | Qwen3 model | embedding → N blocks → norm → tied LM head | `(B,T) → (B,T,V)` | M1.8 |
| 9 | Tokenizer | byte-level BPE from official `tokenizer.json` + special tokens | `str ↔ list[int]` | M2 |
| 10 | KV cache | per-layer `(k, v)` of shape `(B,H_kv,S,d_h)`; RoPE offset = past length | — | M2 |
| 11 | Text generation | greedy (then sampling) decoding loop, EOS stop | `(B,T) → (B,T+n)` | M2 |

Qwen3-specific details to get right:

- **Independent `head_dim`.** For 0.6B, `d/H = 64` but `d_h = 128`, so
  `Wq: 1024 → 2048` and `Wo: 2048 → 1024`. Code that derives `d_h = d / H` is wrong.
- **QK-Norm** is per head (weight shape `(d_h,)`), applied after the projection
  and *before* RoPE. It replaces Qwen2's QKV bias (`attention_bias = false`).
- **GQA** with `H_kv = 8` for 16 query heads (`g = 2`).
- **Tied embeddings** (`tie_word_embeddings = true`): the LM head reuses the
  embedding matrix; the checkpoint may contain no separate `lm_head.weight`.
- **RoPE** base `θ = 1,000,000`, "rotate_half" (non-interleaved) layout.
- **RMSNorm ε = 1e-6** everywhere (block norms, final norm, q_norm, k_norm).
- No dropout, no biases, no sliding window for the dense models.

## 3. Module map: Python ↔ C++

Each Python module has exactly one planned C++ counterpart so that parity tests
can be written module-by-module.

| Python (`src/qwen3_scratch/`) | C++ (`cpp/`) | Status |
|---|---|---|
| `config.py` — `Qwen3Config`, presets | `include/qwen3/config.hpp`, `src/config.cpp` | **M0: done** |
| `env.py` — environment checks | — (Python tooling only) | **M0: done** |
| `layers/rmsnorm.py` | `include/qwen3/rmsnorm.hpp` | M1.1 / M3 |
| `layers/feedforward.py` | `include/qwen3/feedforward.hpp` | M1.2 / M3 |
| `layers/rope.py` | `include/qwen3/rope.hpp` | M1.3 / M3 |
| `layers/attention.py` (QK-Norm, mask, GQA) | `include/qwen3/attention.hpp` | M1.4–1.6 / M3 |
| `layers/block.py` | `include/qwen3/block.hpp` | M1.7 / M3 |
| `model.py` | `include/qwen3/model.hpp` | M1.8 / M3 |
| `weights.py` (HF → native name mapping) | `include/qwen3/weights.hpp` | M2 / M3 |
| `tokenizer.py` | (reuse Python or a C++ BPE later) | M2 / M3+ |
| `kv_cache.py` | `include/qwen3/kv_cache.hpp` | M2 / M3 |
| `generate.py` | `include/qwen3/generate.hpp` | M2 / M3 |

Boundaries:

- Components receive explicit dimensions or a `Qwen3Config`; none reads global state.
- Pure functions where possible (`apply_rope(x, cos, sin, offset)`,
  `build_causal_mask(T, past)`), so the C++ port can mirror signatures.
- Parameter names follow the HF checkpoint where practical (`q_proj`, `k_proj`,
  `v_proj`, `o_proj`, `q_norm`, `k_norm`, `gate_proj`, `up_proj`, `down_proj`,
  `input_layernorm`, `post_attention_layernorm`, `embed_tokens`, `norm`) to make
  weight loading a near-identity mapping. (Decision D-005.)
- Tensors cross the Python→C++ boundary only as files of raw arrays plus a JSON
  manifest (format to be decided in M3); no pybind dependency is assumed.

## 4. Configuration

`Qwen3Config` (`src/qwen3_scratch/config.py`) is a frozen dataclass whose field
names equal the official HF `config.json` keys. It validates itself
(`H % H_kv == 0`, even `d_h`, positive sizes, supported dtype) and refuses HF
features we do not support (sliding window, RoPE scaling, non-SiLU) with
`NotImplementedError` instead of ignoring them. It does not import torch.
Presets: `TINY_QWEN3` (unit tests) and `QWEN3_0_6B_BASE` (target).

## 5. Testing conventions

- One test file per module: `tests/test_<module>.py`.
- Every module is tested for: output **shapes**; **numerical values** against a
  hand-computed or independent reference; **determinism** with fixed seeds;
  **invalid inputs** (`pytest.raises`); dtype behaviour (fp32 and bf16 where relevant).
- Default tolerances: fp32 `rtol=1e-5, atol=1e-6`; bf16 `rtol=1.6e-2, atol=1e-5`
  (torch's own defaults for `assert_close`; full-model bf16 comparisons will need
  looser, documented `atol`), tightened or justified per test.
- Use `torch.testing.assert_close`, never bare `==` on floats.
- Tiers: unmarked = CPU (seconds, CI); `gpu` = CUDA only; `weights` = needs
  `QWEN3_WEIGHTS_DIR`; `slow` = opt-out with `-m "not slow"`.
- Reference comparisons against `transformers` (M2) are marked `weights` or use
  randomly initialised tiny HF models, and live in `tests/parity/`.
- C++ tests are CTest executables; parity fixtures are exported from Python.
