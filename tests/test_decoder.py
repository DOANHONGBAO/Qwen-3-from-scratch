"""Unit tests for the Qwen3 decoder layer (M1.5). CPU tier."""

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

from qwen3_scratch.config import QWEN3_0_6B_BASE, TINY_QWEN3
from qwen3_scratch.layers import (
    GroupedQueryAttention,
    Qwen3DecoderLayer,
    RotaryEmbedding,
    SwiGLUFeedForward,
    causal_mask,
    make_position_ids,
)


def _randn(*shape: int, seed: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64).to(dtype)


def _layer(h=32, i=80, nq=4, nkv=2, d=16, seed=0, dtype=torch.float32, **kw) -> Qwen3DecoderLayer:
    torch.manual_seed(seed)
    layer = Qwen3DecoderLayer(h, i, nq, nkv, d, dtype=dtype, **kw)
    with torch.no_grad():  # non-trivial norm weights so that every norm matters
        for name, p in layer.named_parameters():
            if "norm" in name:
                p.copy_(1 + 0.3 * torch.randn(p.shape))
    return layer


def _tables(t, d, offset=0, dtype=torch.float32):
    return RotaryEmbedding(d, 1_000_000.0)(make_position_ids(t, offset=offset), dtype=dtype)


# ------------------------------------------------------ structure / names
def test_submodules_and_parameter_names() -> None:
    layer = Qwen3DecoderLayer.from_config(TINY_QWEN3, layer_idx=3)
    assert layer.layer_idx == 3
    assert isinstance(layer.self_attn, GroupedQueryAttention)
    assert isinstance(layer.mlp, SwiGLUFeedForward)
    names = {n for n, _ in layer.named_parameters()}
    assert names == {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }


@pytest.mark.parametrize("cfg", [TINY_QWEN3, QWEN3_0_6B_BASE], ids=["tiny", "0.6b"])
def test_parameter_count_matches_config_estimate(cfg) -> None:
    layer = Qwen3DecoderLayer.from_config(cfg, device="meta")
    n = sum(p.numel() for p in layer.parameters())
    per_layer = (cfg.estimate_num_parameters(include_embeddings=False) - cfg.hidden_size) // (
        cfg.num_hidden_layers
    )
    assert n == per_layer
    if cfg is QWEN3_0_6B_BASE:
        assert n == 15_730_944


# ------------------------------------------------- 1, 10, 11, 12, 13. shapes
@pytest.mark.parametrize(
    ("h", "i", "nq", "nkv", "d"),
    [(32, 80, 4, 2, 16), (32, 64, 4, 4, 8), (48, 96, 6, 1, 8), (16, 40, 2, 2, 32)],
    ids=["GQA", "MHA", "MQA", "d>H/N"],
)
@pytest.mark.parametrize(("b", "t"), [(1, 1), (2, 5), (3, 12)])
def test_shapes(h, i, nq, nkv, d, b, t) -> None:
    layer = _layer(h, i, nq, nkv, d)
    x = _randn(b, t, h, seed=1)
    y = layer(x, *_tables(t, d))
    assert y.shape == (b, t, h) and y.dtype == x.dtype and torch.isfinite(y).all()


# -------------------------------------------- 2, 3, 4, 8. ordering & residuals
def test_matches_manual_composition_of_submodules() -> None:
    layer = _layer()
    x = _randn(2, 6, 32, seed=2)
    cos, sin = _tables(6, 16, offset=4)
    y, tr = layer(x, cos, sin, return_trace=True)
    n1 = layer.input_layernorm(x)
    a = layer.self_attn(n1, cos, sin)
    h1 = x + a
    n2 = layer.post_attention_layernorm(h1)
    m = layer.mlp(n2)
    assert torch.equal(tr.input_norm, n1)
    assert torch.equal(tr.attn_output, a)
    assert torch.equal(tr.residual_1, h1)
    assert torch.equal(tr.post_attn_norm, n2)
    assert torch.equal(tr.mlp_output, m)
    assert torch.equal(y, h1 + m) and torch.equal(tr.output, y)
    assert tr.attention is not None and torch.equal(tr.attention.output, a)


def test_wrong_orderings_give_different_results() -> None:
    layer = _layer()
    x = _randn(1, 5, 32, seed=3)
    cos, sin = _tables(5, 16)
    y = layer(x, cos, sin)
    # Post-norm (original Transformer): normalise AFTER adding the residual.
    h1_post = layer.input_layernorm(x + layer.self_attn(x, cos, sin))
    post = layer.post_attention_layernorm(h1_post + layer.mlp(h1_post))
    # MLP before attention.
    swapped_h = x + layer.mlp(layer.input_layernorm(x))
    swapped = swapped_h + layer.self_attn(layer.post_attention_layernorm(swapped_h), cos, sin)
    # Norms swapped between the sublayers.
    h1_sw = x + layer.self_attn(layer.post_attention_layernorm(x), cos, sin)
    norms_swapped = h1_sw + layer.mlp(layer.input_layernorm(h1_sw))
    for wrong in (post, swapped, norms_swapped):
        assert not torch.allclose(wrong, y, atol=1e-4)


def test_each_norm_is_used_where_expected() -> None:
    layer = _layer()
    x = _randn(1, 4, 32, seed=4)
    cos, sin = _tables(4, 16)
    _, base = layer(x, cos, sin, return_trace=True)
    with torch.no_grad():
        layer.post_attention_layernorm.weight.mul_(2.0)
    _, tr = layer(x, cos, sin, return_trace=True)
    assert torch.equal(tr.attn_output, base.attn_output)  # attention path untouched
    assert torch.equal(tr.post_attn_norm, 2.0 * base.post_attn_norm)
    assert not torch.allclose(tr.mlp_output, base.mlp_output)
    with torch.no_grad():
        layer.input_layernorm.weight.mul_(0.5)
    _, tr2 = layer(x, cos, sin, return_trace=True)
    assert_close(tr2.input_norm, 0.5 * base.input_norm)
    assert not torch.allclose(tr2.attn_output, base.attn_output)


# -------------------------------------------------------- 5, 6, 7. ablations
def test_identity_when_both_sublayers_output_zero() -> None:
    layer = _layer()
    with torch.no_grad():
        layer.self_attn.o_proj.weight.zero_()
        layer.mlp.down_proj.weight.zero_()
    x = _randn(2, 7, 32, seed=5)
    y = layer(x, *_tables(7, 16))
    assert torch.equal(y, x)  # x + 0 + 0 is exact in floating point


def test_only_attention_active() -> None:
    layer = _layer()
    with torch.no_grad():
        layer.mlp.down_proj.weight.zero_()
    x = _randn(2, 5, 32, seed=6)
    cos, sin = _tables(5, 16)
    expected = x + layer.self_attn(layer.input_layernorm(x), cos, sin)
    assert torch.equal(layer(x, cos, sin), expected)


def test_only_mlp_active() -> None:
    layer = _layer()
    with torch.no_grad():
        layer.self_attn.o_proj.weight.zero_()
    x = _randn(2, 5, 32, seed=7)
    expected = x + layer.mlp(layer.post_attention_layernorm(x))
    assert torch.equal(layer(x, *_tables(5, 16)), expected)


def test_prenorm_sublayers_ignore_residual_scale() -> None:
    # Pre-norm: sublayer inputs are normalised, so scaling x by c changes the
    # attention branch only through eps; the update y - x is nearly unchanged.
    layer = _layer(dtype=torch.float64)
    with torch.no_grad():
        layer.mlp.down_proj.weight.zero_()
    x = _randn(1, 6, 32, seed=8, dtype=torch.float64)
    cos, sin = _tables(6, 16, dtype=torch.float64)
    delta_1 = layer(x, cos, sin) - x
    delta_10 = layer(10 * x, cos, sin) - 10 * x
    assert_close(delta_10, delta_1, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------- causality and masks
def test_no_future_leakage_through_the_block() -> None:
    layer = _layer()
    prefix = _randn(1, 4, 32, seed=9)
    a = torch.cat([prefix, _randn(1, 3, 32, seed=10)], dim=1)
    b = torch.cat([prefix, 50 * _randn(1, 3, 32, seed=11)], dim=1)
    out = layer(torch.cat([a, b]), *_tables(7, 16))
    assert_close(out[0, :4], out[1, :4], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(out[0, 4:], out[1, 4:])


def test_explicit_mask_is_forwarded_to_attention() -> None:
    layer = _layer()
    x = _randn(1, 5, 32, seed=12)
    cos, sin = _tables(5, 16)
    full = causal_mask(torch.arange(5), torch.arange(5))
    assert torch.equal(layer(x, cos, sin, full), layer(x, cos, sin))
    diag_only = torch.eye(5, dtype=torch.bool)  # each token attends to itself only
    _, tr = layer(x, cos, sin, diag_only, return_trace=True)
    assert torch.equal(tr.attention.probs[0, :, range(5), range(5)], torch.ones(4, 5))


# ----------------------------------------------------------- 9. gradients
def test_gradients_reach_every_parameter() -> None:
    layer = _layer()
    x = _randn(2, 5, 32, seed=13).requires_grad_()
    layer(x, *_tables(5, 16)).pow(2).sum().backward()
    assert torch.isfinite(x.grad).all()
    for name, p in layer.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_residual_path_gives_identity_gradient() -> None:
    # With both sublayers zeroed, d(out)/d(x) is exactly the identity.
    layer = _layer()
    with torch.no_grad():
        layer.self_attn.o_proj.weight.zero_()
        layer.mlp.down_proj.weight.zero_()
    x = _randn(1, 3, 32, seed=14).requires_grad_()
    u = _randn(1, 3, 32, seed=15)
    layer(x, *_tables(3, 16)).backward(u)
    assert torch.equal(x.grad, u)


def test_gradcheck_float64() -> None:
    layer = _layer(h=8, i=12, nq=2, nkv=1, d=4, dtype=torch.float64)
    x = _randn(1, 3, 8, seed=16, dtype=torch.float64).requires_grad_()
    cos, sin = _tables(3, 4, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda t: layer(t, cos, sin), (x,))


# ------------------------------------------------ 14. determinism / errors
def test_deterministic() -> None:
    x = _randn(2, 6, 32, seed=17)
    cos, sin = _tables(6, 16)
    a, b = _layer(seed=5), _layer(seed=5)
    assert torch.equal(a(x, cos, sin), a(x, cos, sin))
    assert torch.equal(a(x, cos, sin), b(x, cos, sin))


def test_invalid_inputs_are_rejected() -> None:
    layer = _layer()
    cos, sin = _tables(5, 16)
    with pytest.raises(ValueError):
        layer(torch.randn(1, 5, 31), cos, sin)
    with pytest.raises(ValueError):
        layer(torch.randn(5, 32), cos, sin)
    with pytest.raises(ValueError):
        Qwen3DecoderLayer(32, 80, 4, 3, 16)
    with pytest.raises(ValueError):
        Qwen3DecoderLayer(32, 0, 4, 2, 16)
