# Architecture

This document describes the Qwen3 dense architecture as implemented in this
repository, the module boundaries shared by Python and C++, and the testing
conventions. The PyTorch implementation is complete through M2 (official weights,
tokenizer, KV cache, generation); the C++20 port (M3) runs the official checkpoint
with greedy generation, and M4.1 optimised it for the CPU. Each section lists the
milestone that delivers it.

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
scores = q kᵀ · d_h^(-1/2), −∞ where masked  -> (B, H, T, S)   (mask BEFORE softmax)
out    = softmax_fp32(scores) v               -> (B, H, T, d_h) -> (B, T, H·d_h) -> Wo -> (B, T, d)
(query head h reads kv head h // g; QK-Norm before RoPE; v is neither normed nor rotated)
```

## 2. Components

| # | Component | Math (summary) | In → out shape | Milestone |
|---|---|---|---|---|
| 1 | RMSNorm | `y = x / sqrt(mean(x², -1) + ε) ⊙ w` (computed in fp32) — [details](components/rmsnorm.md) | `(..., n) → (..., n)` | M1.1 **done** |
| 2 | SwiGLU FeedForward | `down( silu(gate(x)) ⊙ up(x) )`, no biases — [details](components/swiglu.md) | `(B,T,d) → (B,T,d)` via `d_ff` | M1.2 **done** |
| 3 | RoPE | rotate pairs `(x_i, x_{i+d_h/2})` by `p·θ^{-2i/d_h}` ("rotate_half" layout) — [details](components/rope.md) | `(B,H,T,d_h) → same` | M1.3 **done** |
| 4 | QK-Norm | RMSNorm over `d_h` of each Q and K head, **before** RoPE — [details](components/qk_norm.md) | `(B,N,T,d_h) → same` | M1.4 **done** |
| 5 | Causal mask | `causal_mask(q_pos, k_pos)`: key position ≤ query position, applied before softmax | `(T, S)` bool | M1.4 **done** |
| 6 | Grouped Query Attention | see §1; query head `h` reads KV head `h // g` — [details](components/attention.md) | `(B,T,d) → (B,T,d)` | M1.4 **done** |
| 7 | Decoder block | two pre-norm residual sublayers — [details](components/decoder.md) | `(B,T,d) → (B,T,d)` | M1.5 **done** |
| 8 | Qwen3 model | embedding → N blocks → norm → tied LM head (+ causal LM loss, greedy generation) — [details](components/model.md) | `(B,T) → (B,T,V)` | M1.6 **done** |
| 9 | Weight loading | explicit checkpoint name map, strict shape/dtype checks, tied head — [details](components/pretrained.md) | safetensors → model | M2.1 **done** |
| 10 | Tokenizer | adapter over the official `tokenizer.json` (+ config-only added tokens), no BOS — [details](components/pretrained.md#4-tokenizer-adapter) | `str ↔ list[int]` | M2.1 **done** |
| 11 | KV cache | per-layer post-RoPE `(k, v)` `(B,H_kv,S,d_h)`; positions continue from the cache length — [details](components/kv_cache.md) | — | M2.2 **done** |
| 12 | Text generation | prefill + cached decode; greedy, temperature, top-k, top-p; EOS stop — [details](components/kv_cache.md#6-generation-api) | `(B,T) → (B,T+n)` | M2.2 **done** |

Qwen3-specific details to get right:

- **Independent `head_dim`.** For 0.6B, `d/H = 64` but `d_h = 128`, so
  `Wq: 1024 → 2048` and `Wo: 2048 → 1024`. Code that derives `d_h = d / H` is wrong.
- **QK-Norm** is per head (weight shape `(d_h,)`), applied after the projection
  and *before* RoPE. Qwen3 also drops Qwen2's QKV bias (`attention_bias = false`).
- **GQA** with `H_kv = 8` for 16 query heads (`g = 2`).
- **Tied embeddings** (`tie_word_embeddings = true`): the LM head reuses the
  embedding matrix; the checkpoint may contain no separate `lm_head.weight`.
- **RoPE** base `θ = 1,000,000`, "rotate_half" (non-interleaved) layout.
- **RMSNorm ε = 1e-6** everywhere (block norms, final norm, q_norm, k_norm).
- No dropout, no biases, no sliding window for the dense models.

## 3. Module map: Python ↔ C++

Each Python module has one C++ counterpart (or none, for Python-only tooling) so that
parity tests are written module by module.

| Python (`src/qwen3_scratch/`) | C++ (`cpp/`) | Status |
|---|---|---|
| `config.py` — `Qwen3Config`, presets | `config.hpp/.cpp` (`qwen3::Config`), `config_from_json` in `model.cpp` | **done** (M0 / M3) |
| `env.py` — environment checks | — (Python tooling only) | **done** (M0) |
| — (torch tensors) | `tensor.hpp/.cpp` (owning, contiguous, row-major fp32), `ops.hpp/.cpp` | **done** (M3, M4.1) |
| `layers/rmsnorm.py` — `RMSNorm` | `rmsnorm.hpp/.cpp` (`RMSNorm`) | **done** (M1.1 / M3) |
| `layers/swiglu.py` — `SwiGLUFeedForward` | `swiglu.hpp/.cpp` (`SwiGLU`) | **done** (M1.2 / M3) |
| `layers/rope.py` — `RotaryEmbedding`, `apply_rope` | `rope.hpp/.cpp` (`RoPE`, split-half) | **done** (M1.3 / M3) |
| `layers/qk_norm.py` — `QKNorm` | inside `Attention` (`q_norm`, `k_norm` are `RMSNorm(head_dim)`) | **done** (M1.4 / M3) |
| `layers/attention.py` — `GroupedQueryAttention` | `attention.hpp/.cpp` (`Attention`) | **done** (M1.4 / M3) |
| `layers/decoder.py` — `Qwen3DecoderLayer` | `decoder.hpp/.cpp` (`DecoderLayer`) | **done** (M1.5 / M3) |
| `model.py` — `Qwen3Model`, `Qwen3ForCausalLM` | `model.hpp/.cpp` (`Qwen3Model`, tied/untied head, `last_only`) | **done** (M1.6 / M3) |
| `weight_loader.py`, `pretrained.py` | `json.hpp/.cpp`, `safetensors.hpp/.cpp`, `load_pretrained` in `model.cpp` (contract: `tests/fixtures/checkpoint_name_map.json`) | **done** (M2.1 / M3) |
| `tokenizer.py` — `Qwen3Tokenizer` | — : the C++ tools take token ids; Python tokenises (`scripts/verify_cpp_pretrained.py`) | **done** (M2.1); C++ BPE not planned |
| `kv_cache.py` — `KVCache` | `kv_cache.hpp/.cpp` (`KVCache`, time-major `[S, B, Nkv, D]`, D-043) | **done** (M2.2 / M3) |
| `generation.py` — `generate`, sampling | `generate.hpp/.cpp` (`greedy_generate`; sampling not ported) | **done** (M2.2 / M3, greedy only) |
| `sysinfo.py`, `synthetic.py` | `profiler.hpp/.cpp`; tools `cpp/tools/qwen3_run.cpp`, `qwen3_bench.cpp` | **done** (M2 / M4.1) |

Boundaries:

- Components receive explicit dimensions or a `Qwen3Config`; none reads global state.
- Pure functions where possible (`apply_rope(x, cos, sin)` with tables built from
  explicit `position_ids`,
  `build_causal_mask(T, past)`), so the C++ port can mirror signatures.
- Parameter names follow the HF checkpoint where practical (`q_proj`, `k_proj`,
  `v_proj`, `o_proj`, `q_norm`, `k_norm`, `gate_proj`, `up_proj`, `down_proj`,
  `input_layernorm`, `post_attention_layernorm`, `embed_tokens`, `norm`) to make
  weight loading a near-identity mapping. (Decision D-005.)
- Tensors cross the Python→C++ boundary only as files: checkpoints as safetensors,
  reference values as generated C++ headers (`cpp/tests/*_reference.hpp`), C++ logits
  as raw float32 files read back by Python. No pybind dependency (D-041, D-042).

## 4. Configuration

`Qwen3Config` (`src/qwen3_scratch/config.py`) is a frozen dataclass whose field
names equal the official HF `config.json` keys. It validates itself
(`H % H_kv == 0`, even `d_h`, positive sizes, supported dtype) and refuses HF
features we do not support (sliding window, RoPE scaling, non-SiLU) with
`NotImplementedError` instead of ignoring them. It does not import torch.
Presets: `TINY_QWEN3` (component unit tests, `head_dim ≠ H/N`), `QWEN3_TINY`
(`configs/qwen3_tiny.json`, the educational Tiny LM of M1.6) and `QWEN3_0_6B_BASE`
(target checkpoint architecture). M1.6 added `attention_dropout` (must be 0.0) and
`initializer_range`.

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
- C++ tests are CTest executables without a test framework (shared helpers in
  `cpp/tests/test_utils.hpp`). Reference values come from the verified PyTorch
  modules through generated headers (`scripts/export_cpp_*_reference.py` →
  `cpp/tests/*_reference.hpp`). The JSON fixtures of `scripts/export_fixtures.py`
  (`tests/fixtures/`, kept current by `tests/test_fixtures.py`) are a
  language-neutral record of the Python modules; the C++ tests read only
  `checkpoint_name_map.json` from them.
