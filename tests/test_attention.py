"""Unit tests for grouped-query causal self-attention (M1.4, part B). CPU tier."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from qwen3_scratch.config import QWEN3_0_6B_BASE, TINY_QWEN3
from qwen3_scratch.layers import (
    GroupedQueryAttention,
    RotaryEmbedding,
    apply_rope,
    attention_softmax,
    bool_to_additive_mask,
    causal_mask,
    compute_inv_freq,
    make_position_ids,
    query_to_kv_head,
    repeat_kv,
    rope_cos_sin,
)


def _randn(*shape: int, seed: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64).to(dtype)


def _attn(
    h: int = 32,
    nq: int = 4,
    nkv: int = 2,
    d: int = 16,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> GroupedQueryAttention:
    torch.manual_seed(seed)
    attn = GroupedQueryAttention(h, nq, nkv, d, dtype=dtype, **kwargs)
    if attn.q_norm is not None:  # non-trivial norm weights so placement matters
        with torch.no_grad():
            attn.q_norm.weight.copy_(1 + 0.3 * torch.randn(d))
            attn.k_norm.weight.copy_(1 - 0.3 * torch.randn(d))
    return attn


def _tables(t: int, d: int, offset: int = 0, dtype=torch.float32, theta: float = 10_000.0):
    if dtype == torch.float64:  # exact float64 tables for mathematical checks
        return rope_cos_sin(
            make_position_ids(t, offset=offset), compute_inv_freq(d, theta, dtype=torch.float64)
        )
    return RotaryEmbedding(d, theta)(make_position_ids(t, offset=offset), dtype=dtype)


def _reference_attention(attn: GroupedQueryAttention, x, cos, sin) -> dict[str, torch.Tensor]:
    """Independent loop-based reference: explicit head mapping h -> h // g, no repeat_kv."""
    b, t, _ = x.shape
    nq, nkv, d = attn.num_attention_heads, attn.num_key_value_heads, attn.head_dim
    g = nq // nkv
    q = (x @ attn.q_proj.weight.T).view(b, t, nq, d)
    k = (x @ attn.k_proj.weight.T).view(b, t, nkv, d)
    v = (x @ attn.v_proj.weight.T).view(b, t, nkv, d)
    if attn.q_norm is not None:
        q = attn.q_norm(q)
        k = attn.k_norm(k)
    q = apply_rope(q.transpose(1, 2), cos, sin)
    k = apply_rope(k.transpose(1, 2), cos, sin)
    v = v.transpose(1, 2)
    ctx = torch.zeros(b, nq, t, d, dtype=x.dtype)
    probs = torch.zeros(b, nq, t, t, dtype=x.dtype)
    future = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    for bi in range(b):
        for h in range(nq):
            kv = h // g
            s = q[bi, h] @ k[bi, kv].T / math.sqrt(d)
            s = s.masked_fill(future, float("-inf"))
            p = torch.softmax(s, dim=-1)
            probs[bi, h] = p
            ctx[bi, h] = p @ v[bi, kv]
    merged = ctx.transpose(1, 2).reshape(b, t, nq * d)
    return {"probs": probs, "context": merged, "output": merged @ attn.o_proj.weight.T}


# ------------------------------------------------- 1, 2. shapes / dims
@pytest.mark.parametrize("cfg", [TINY_QWEN3, QWEN3_0_6B_BASE], ids=["tiny", "0.6b"])
def test_projection_dimensions_from_config(cfg) -> None:
    attn = GroupedQueryAttention.from_config(cfg, device="meta")
    shapes = {n: tuple(p.shape) for n, p in attn.named_parameters()}
    q_dim, kv_dim = cfg.num_attention_heads * cfg.head_dim, cfg.num_key_value_heads * cfg.head_dim
    assert shapes == {
        "q_proj.weight": (q_dim, cfg.hidden_size),
        "k_proj.weight": (kv_dim, cfg.hidden_size),
        "v_proj.weight": (kv_dim, cfg.hidden_size),
        "o_proj.weight": (cfg.hidden_size, q_dim),
        "q_norm.weight": (cfg.head_dim,),
        "k_norm.weight": (cfg.head_dim,),
    }
    if cfg is QWEN3_0_6B_BASE:
        assert q_dim == 2048 != cfg.hidden_size  # 19: head_dim independent of H / Nq
        assert sum(p.numel() for p in attn.parameters()) == 6_291_456 + 256


@pytest.mark.parametrize(("b", "t"), [(1, 1), (1, 9), (3, 4), (2, 33)])
def test_output_and_trace_shapes(b, t) -> None:
    attn = _attn()
    x = _randn(b, t, 32, seed=1)
    y, tr = attn(x, *_tables(t, 16), return_trace=True)
    assert y.shape == (b, t, 32) and y.dtype == torch.float32
    assert tr.q_proj.shape == (b, t, 64) and tr.k_proj.shape == tr.v_proj.shape == (b, t, 32)
    assert tr.q_normed.shape == tr.q_rope.shape == (b, 4, t, 16)
    assert tr.k_normed.shape == tr.k_rope.shape == (b, 2, t, 16)
    assert tr.k_grouped.shape == tr.v_grouped.shape == (b, 4, t, 16)
    assert tr.scores.shape == tr.probs.shape == (b, 4, t, t)
    assert tr.context.shape == (b, t, 64)
    assert torch.equal(tr.output, y)


# --------------------------------- 3, 4. QK-Norm and RoPE placement
def test_qk_norm_applied_per_head_to_projections() -> None:
    attn = _attn()
    x = _randn(2, 5, 32, seed=2)
    _, tr = attn(x, *_tables(5, 16), return_trace=True)
    q_heads = tr.q_proj.view(2, 5, 4, 16).transpose(1, 2)
    k_heads = tr.k_proj.view(2, 5, 2, 16).transpose(1, 2)
    assert torch.equal(tr.q_normed, attn.q_norm(q_heads))
    assert torch.equal(tr.k_normed, attn.k_norm(k_heads))


def test_rope_applied_after_qk_norm_not_before() -> None:
    attn = _attn()
    x = _randn(1, 6, 32, seed=3)
    cos, sin = _tables(6, 16, offset=4)
    _, tr = attn(x, cos, sin, return_trace=True)
    assert torch.equal(tr.q_rope, apply_rope(tr.q_normed, cos, sin))
    assert torch.equal(tr.k_rope, apply_rope(tr.k_normed, cos, sin))
    # The opposite order (RoPE then norm) gives different tensors with non-unit weights.
    q_heads = tr.q_proj.view(1, 6, 4, 16).transpose(1, 2)
    wrong = attn.q_norm(apply_rope(q_heads, cos, sin))
    assert not torch.allclose(wrong, tr.q_rope, atol=1e-4)


def test_values_are_neither_normalised_nor_rotated() -> None:
    attn = _attn(nq=4, nkv=4)
    x = _randn(1, 5, 32, seed=4)
    _, tr = attn(x, *_tables(5, 16, offset=10), return_trace=True)
    assert torch.equal(tr.v_grouped, tr.v_proj.view(1, 5, 4, 16).transpose(1, 2))


def test_qk_norm_can_be_disabled() -> None:
    attn = _attn(qk_norm=False)
    assert attn.q_norm is None and "q_norm.weight" not in attn.state_dict()
    x = _randn(1, 4, 32, seed=5)
    _, tr = attn(x, *_tables(4, 16), return_trace=True)
    assert torch.equal(tr.q_normed, tr.q_proj.view(1, 4, 4, 16).transpose(1, 2))


# --------------------------------------------------- 5. head grouping
def test_query_to_kv_mapping_is_contiguous() -> None:
    assert query_to_kv_head(16, 8) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
    assert query_to_kv_head(4, 4) == [0, 1, 2, 3]
    assert query_to_kv_head(4, 1) == [0, 0, 0, 0]
    assert query_to_kv_head(6, 2) == [0, 0, 0, 1, 1, 1]
    modulo = [h % 8 for h in range(16)]
    assert modulo != query_to_kv_head(16, 8)  # a modulo mapping pairs the wrong heads


def test_repeat_kv_matches_repeat_interleave_and_mapping() -> None:
    k = _randn(2, 3, 5, 4, seed=6)
    rep = repeat_kv(k, 4)
    assert torch.equal(rep, torch.repeat_interleave(k, 4, dim=1))
    for h, kv in enumerate(query_to_kv_head(12, 3)):
        assert torch.equal(rep[:, h], k[:, kv])
    assert repeat_kv(k, 1) is k


def test_grouped_tensors_follow_mapping() -> None:
    attn = _attn(nq=6, nkv=2)
    _, tr = attn(_randn(2, 4, 32, seed=7), *_tables(4, 16), return_trace=True)
    for h in range(6):
        assert torch.equal(tr.k_grouped[:, h], tr.k_rope[:, h // 3])


def test_repeat_kv_gradient_sums_over_group() -> None:
    k = _randn(1, 2, 3, 4, seed=8).requires_grad_()
    w = _randn(1, 6, 3, 4, seed=9)
    (repeat_kv(k, 3) * w).sum().backward()
    assert_close(k.grad, w.view(1, 2, 3, 3, 4).sum(dim=2))


def test_modulo_grouping_changes_output() -> None:
    attn = _attn(nq=4, nkv=2)
    x = _randn(1, 5, 32, seed=10)
    cos, sin = _tables(5, 16)
    _, tr = attn(x, cos, sin, return_trace=True)
    k_mod = tr.k_rope[:, [h % 2 for h in range(4)]]
    v_mod = tr.v_grouped[:, [0, 2, 0, 2]]  # heads 0,2 of grouped == kv 0,1
    scores = tr.q_rope @ k_mod.transpose(-1, -2) / 4.0
    p = attention_softmax(scores, causal_mask(torch.arange(5), torch.arange(5)))
    ctx_mod = (p @ v_mod).transpose(1, 2).reshape(1, 5, 64)
    assert not torch.allclose(ctx_mod, tr.context, atol=1e-4)


# ------------------------------------------ 6, 7, 8. MHA / MQA / GQA
@pytest.mark.parametrize(
    ("nq", "nkv"),
    [(4, 4), (4, 1), (4, 2), (8, 2), (6, 3)],
    ids=["MHA", "MQA", "GQA-2", "GQA-4", "GQA-3x2"],
)
def test_matches_independent_loop_reference(nq, nkv) -> None:
    attn = _attn(h=24, nq=nq, nkv=nkv, d=8, dtype=torch.float64)
    x = _randn(2, 6, 24, seed=11, dtype=torch.float64)
    cos, sin = _tables(6, 8, offset=2, dtype=torch.float64)
    y, tr = attn(x, cos, sin, return_trace=True)
    ref = _reference_attention(attn, x, cos, sin)
    assert_close(tr.probs, ref["probs"], rtol=1e-12, atol=1e-12)
    assert_close(tr.context, ref["context"], rtol=1e-12, atol=1e-12)
    assert_close(y, ref["output"], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("nq", "nkv"), [(4, 4), (4, 1), (8, 2)])
def test_matches_torch_sdpa_with_gqa(nq, nkv) -> None:
    attn = _attn(nq=nq, nkv=nkv)
    _, tr = attn(_randn(2, 7, 32, seed=12), *_tables(7, 16), return_trace=True)
    v = tr.v_proj.view(2, 7, nkv, 16).transpose(1, 2)
    ref = F.scaled_dot_product_attention(
        tr.q_rope, tr.k_rope, v, is_causal=True, scale=16**-0.5, enable_gqa=True
    )
    assert_close(tr.context, ref.transpose(1, 2).reshape(2, 7, nq * 16), rtol=1e-5, atol=1e-6)


def test_mha_mqa_gqa_parameter_and_cache_sizes() -> None:
    h, nq, d = 1024, 16, 128
    kv_params = {}
    for name, nkv in {"MHA": 16, "GQA": 8, "MQA": 1}.items():
        a = GroupedQueryAttention(h, nq, nkv, d, device="meta")
        kv_params[name] = a.k_proj.weight.numel() + a.v_proj.weight.numel()
        assert a.kv_cache_bytes_per_token(2) == 2 * nkv * d * 2
    assert kv_params == {"MHA": 4_194_304, "GQA": 2_097_152, "MQA": 262_144}
    gqa = GroupedQueryAttention.from_config(QWEN3_0_6B_BASE, device="meta")
    # 28 layers, bf16: 112 KiB per token; 3.5 GiB for a full 32,768-token context.
    per_token = 28 * gqa.kv_cache_bytes_per_token(2)
    assert per_token == 114_688 and per_token * 32_768 == 3.5 * 2**30


# ---------------------------------------------------- 9. invalid config
@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_attention_heads": 4, "num_key_value_heads": 3},
        {"num_attention_heads": 2, "num_key_value_heads": 4},
        {"num_attention_heads": 0, "num_key_value_heads": 1},
        {"num_attention_heads": 4, "num_key_value_heads": 0},
        {"head_dim": 15},
        {"head_dim": 0},
        {"hidden_size": -1},
        {"num_attention_heads": 4.0},
    ],
)
def test_invalid_configurations(kwargs) -> None:
    base = {"hidden_size": 32, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16}
    with pytest.raises(ValueError):
        GroupedQueryAttention(**(base | kwargs))


def test_invalid_inputs() -> None:
    attn = _attn()
    cos, sin = _tables(5, 16)
    with pytest.raises(ValueError, match=r"\[B, T, H\]"):
        attn(torch.randn(5, 32), cos, sin)
    with pytest.raises(ValueError, match="hidden size"):
        attn(torch.randn(1, 5, 31), cos, sin)
    with pytest.raises(ValueError):  # tables for 4 tokens, 5 tokens given
        attn(torch.randn(1, 5, 32), *_tables(4, 16))
    with pytest.raises(ValueError, match="dtype"):
        attn(torch.randn(1, 5, 32), *_tables(5, 16, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="boolean"):
        attn(torch.randn(1, 5, 32), cos, sin, attention_mask=torch.zeros(5, 5))
    with pytest.raises(ValueError, match="covers"):
        attn(torch.randn(1, 5, 32), cos, sin, attention_mask=torch.ones(4, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="broadcast"):
        attn(torch.randn(2, 5, 32), cos, sin, attention_mask=torch.ones(3, 5, 5, dtype=torch.bool))
    with pytest.raises(ValueError):
        repeat_kv(torch.randn(2, 3, 4), 2)
    with pytest.raises(ValueError):
        repeat_kv(torch.randn(1, 2, 3, 4), 0)


# ----------------------------------- 10, 11, 12. causality and softmax
def test_probabilities_are_causal_and_normalised() -> None:
    attn = _attn()
    _, tr = attn(_randn(2, 8, 32, seed=13), *_tables(8, 16), return_trace=True)
    future = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    assert torch.equal(tr.probs[..., future], torch.zeros_like(tr.probs[..., future]))
    assert torch.isinf(tr.scores[..., future]).all() and (tr.scores[..., future] < 0).all()
    assert_close(tr.probs.sum(-1), torch.ones(2, 4, 8), rtol=1e-6, atol=1e-6)
    assert (tr.probs >= 0).all()
    assert torch.equal(tr.probs[..., 0, 0], torch.ones(2, 4))  # first token sees only itself


def test_no_future_leakage_shared_prefix() -> None:
    """Two sequences share a 5-token prefix; only later tokens differ."""
    attn = _attn()
    prefix = _randn(1, 5, 32, seed=14)
    seq_a = torch.cat([prefix, _randn(1, 4, 32, seed=15)], dim=1)
    seq_b = torch.cat([prefix, 100 * _randn(1, 4, 32, seed=16)], dim=1)
    cos, sin = _tables(9, 16)
    out = attn(torch.cat([seq_a, seq_b]), cos, sin)
    assert_close(out[0, :5], out[1, :5], rtol=1e-6, atol=1e-6)  # shared positions unchanged
    assert not torch.allclose(out[0, 5:], out[1, 5:])  # future positions do differ


def test_no_future_leakage_gradient() -> None:
    attn = _attn()
    x = _randn(1, 6, 32, seed=17).requires_grad_()
    out = attn(x, *_tables(6, 16))
    out[0, 2].sum().backward()  # loss at position 2
    assert x.grad[0, 3:].abs().max() == 0  # no gradient from future tokens
    assert x.grad[0, :3].abs().sum() > 0


def test_causal_mask_from_positions_supports_cached_decoding() -> None:
    q_pos = torch.tensor([7, 8])  # two new tokens after 7 cached ones
    k_pos = torch.arange(9)
    allowed = causal_mask(q_pos, k_pos)
    assert allowed.shape == (2, 9)
    assert allowed[0].tolist() == [True] * 8 + [False]
    assert allowed[1].tolist() == [True] * 9
    # A plain upper-triangular [2, 9] mask would be wrong here:
    naive = ~torch.triu(torch.ones(2, 9, dtype=torch.bool), diagonal=1)
    assert not torch.equal(naive, allowed)
    batched = causal_mask(torch.tensor([[0, 1], [5, 6]]), torch.tensor([[0, 1], [5, 6]]))
    assert batched.shape == (2, 2, 2)
    with pytest.raises(ValueError):
        causal_mask(torch.tensor([0.0, 1.0]), torch.arange(2))


def test_additive_mask_conversion() -> None:
    allowed = causal_mask(torch.arange(3), torch.arange(3))
    add = bool_to_additive_mask(allowed, torch.float32)
    assert add[0, 0] == 0 and add[0, 1] == torch.finfo(torch.float32).min


def test_explicit_mask_with_padding() -> None:
    # Batch item 1 has 2 left-padding tokens that nobody may attend to.
    attn = _attn()
    t = 5
    pad = torch.tensor([[True] * t, [False, False, True, True, True]])
    allowed = causal_mask(torch.arange(t), torch.arange(t))[None] & pad[:, None, :]
    _, tr = attn(_randn(2, t, 32, seed=18), *_tables(t, 16), allowed, return_trace=True)
    assert torch.equal(tr.probs[1, :, :, :2], torch.zeros(4, t, 2))
    # Rows 0 and 1 of item 1 have no allowed key: zeros, not NaN.
    assert torch.equal(tr.probs[1, :, :2], torch.zeros(4, 2, t))
    assert torch.isfinite(tr.output).all()


def test_fully_masked_rows_backward_is_finite() -> None:
    scores = _randn(1, 1, 3, 3, seed=19).requires_grad_()
    allowed = torch.tensor([[False, False, False], [True, False, False], [True, True, True]])
    p = attention_softmax(scores, allowed)
    assert torch.equal(p[0, 0, 0], torch.zeros(3))
    (p * _randn(1, 1, 3, 3, seed=20)).sum().backward()
    assert torch.isfinite(scores.grad).all()


# ------------------------------------------------------ 13. stability
@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e4])
def test_no_nan_or_inf_for_extreme_inputs(scale) -> None:
    attn = _attn()
    y, tr = attn(scale * _randn(2, 7, 32, seed=21), *_tables(7, 16), return_trace=True)
    assert torch.isfinite(y).all() and torch.isfinite(tr.probs).all()
    # QK-Norm keeps the scaled logits bounded by sqrt(D) * max|g_q * g_k| for any input scale.
    gq, gk = attn.q_norm.weight.abs().max(), attn.k_norm.weight.abs().max()
    finite = tr.scores[torch.isfinite(tr.scores)]
    assert finite.abs().max() <= 16**0.5 * gq * gk * (1 + 1e-4)


def test_bfloat16_forward_is_finite_and_close() -> None:
    a32 = _attn()
    a16 = GroupedQueryAttention(32, 4, 2, 16, dtype=torch.bfloat16)
    a16.load_state_dict({k: v.to(torch.bfloat16) for k, v in a32.state_dict().items()})
    x = _randn(2, 6, 32, seed=22)
    y16 = a16(x.bfloat16(), *_tables(6, 16, dtype=torch.bfloat16))
    y32 = a32(x.bfloat16().float(), *_tables(6, 16))
    assert y16.dtype == torch.bfloat16 and torch.isfinite(y16).all()
    rel = (y16.float() - y32).norm() / y32.norm()
    assert rel < 3e-2


# ------------------------------------------------- 14, 15. gradients
def test_gradients_reach_every_parameter_and_input() -> None:
    attn = _attn()
    x = _randn(2, 5, 32, seed=23).requires_grad_()
    attn(x, *_tables(5, 16)).pow(2).sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for name, p in attn.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_gradcheck_through_qk_norm_rope_and_attention() -> None:
    attn = _attn(h=8, nq=4, nkv=2, d=4, dtype=torch.float64)
    x = _randn(1, 3, 8, seed=24, dtype=torch.float64).requires_grad_()
    cos, sin = _tables(3, 4, offset=1, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda t: attn(t, cos, sin), (x,))
    names = ["q_norm.weight", "k_norm.weight", "q_proj.weight", "v_proj.weight"]
    params = tuple(attn.get_parameter(n).detach().clone().requires_grad_() for n in names)

    def with_params(*ps):
        return torch.func.functional_call(
            attn, dict(zip(names, ps, strict=True)), (x.detach(), cos, sin)
        )

    assert torch.autograd.gradcheck(with_params, params)


# ----------------------------------------- 16, 17, 18. sizes and T = 1
def test_single_token_output_is_projected_value() -> None:
    attn = _attn(nq=4, nkv=2)
    x = _randn(3, 1, 32, seed=25)
    y, tr = attn(x, *_tables(1, 16, offset=42), return_trace=True)
    assert torch.equal(tr.probs, torch.ones(3, 4, 1, 1))
    v = tr.v_proj.view(3, 1, 2, 16)
    merged = v[:, :, [0, 0, 1, 1]].reshape(3, 1, 64)  # each query head copies its kv head
    assert_close(y, attn.o_proj(merged))


@pytest.mark.parametrize("b", [1, 2, 5])
@pytest.mark.parametrize("t", [1, 3, 16])
def test_batch_items_independent(b, t) -> None:
    attn = _attn()
    x = _randn(b, t, 32, seed=b * 10 + t)
    cos, sin = _tables(t, 16)
    full = attn(x, cos, sin)
    for i in range(b):
        assert_close(full[i : i + 1], attn(x[i : i + 1], cos, sin), rtol=1e-5, atol=1e-6)


def test_relative_positions_only() -> None:
    # Shifting every position by the same offset leaves the output unchanged (float64 tables).
    attn = _attn(dtype=torch.float64)
    x = _randn(2, 6, 32, seed=26, dtype=torch.float64)
    a = attn(x, *_tables(6, 16, offset=0, dtype=torch.float64))
    b = attn(x, *_tables(6, 16, offset=1000, dtype=torch.float64))
    assert_close(a, b, rtol=1e-9, atol=1e-9)


def test_batched_position_tables() -> None:
    attn = _attn()
    x = _randn(2, 4, 32, seed=27)
    rope = RotaryEmbedding(16)
    pos = torch.tensor([[0, 1, 2, 3], [7, 8, 9, 10]])
    y = attn(x, *rope(pos))
    for i in range(2):
        assert_close(y[i : i + 1], attn(x[i : i + 1], *rope(pos[i])), rtol=1e-5, atol=1e-6)


# ------------------------------------------ 21, 22. determinism / CPU
def test_deterministic_and_cpu() -> None:
    x = _randn(2, 5, 32, seed=28)
    cos, sin = _tables(5, 16)
    a, b = _attn(seed=3), _attn(seed=3)
    assert torch.equal(a(x, cos, sin), a(x, cos, sin))
    assert torch.equal(a(x, cos, sin), b(x, cos, sin))
    assert all(p.device.type == "cpu" for p in a.parameters())
