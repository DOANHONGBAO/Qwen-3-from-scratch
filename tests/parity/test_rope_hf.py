"""Parity of our RoPE with Hugging Face ``Qwen3RotaryEmbedding`` / ``apply_rotary_pos_emb``.

Requires the optional ``hf`` extra; skipped otherwise. No downloads: the HF
rotary module is built from a small ``Qwen3Config``.
"""

from __future__ import annotations

import pytest
import torch

modeling = pytest.importorskip(
    "transformers.models.qwen3.modeling_qwen3", reason="transformers not installed (hf extra)"
)
transformers = pytest.importorskip("transformers")

from qwen3_scratch.config import QWEN3_0_6B_BASE  # noqa: E402
from qwen3_scratch.layers import (  # noqa: E402
    RotaryEmbedding,
    apply_rope_to_qk,
    make_position_ids,
    rotate_half,
)


def _hf_rotary(head_dim: int, theta: float, max_pos: int = 4096):
    cfg = transformers.Qwen3Config(
        hidden_size=4 * head_dim,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        rope_theta=theta,
        max_position_embeddings=max_pos,
        num_hidden_layers=1,
        vocab_size=32,
    )
    return modeling.Qwen3RotaryEmbedding(cfg)


@pytest.mark.parametrize(("head_dim", "theta"), [(16, 10_000.0), (128, 1_000_000.0)])
def test_inv_freq_bit_identical(head_dim, theta) -> None:
    ours = RotaryEmbedding(head_dim, theta)
    hf = _hf_rotary(head_dim, theta)
    assert hf.inv_freq.dtype == ours.inv_freq.dtype == torch.float32
    assert torch.equal(ours.inv_freq, hf.inv_freq)


def test_rotate_half_identical() -> None:
    x = torch.randn(2, 3, 5, 8)
    assert torch.equal(rotate_half(x), modeling.rotate_half(x))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "position_ids",
    [
        make_position_ids(7, batch_size=2),  # prefill
        make_position_ids(1, offset=1234, batch_size=2),  # one decode step with a cache
        torch.tensor([[0, 1, 2, 3], [5, 6, 7, 8]]),  # per-sequence offsets
        make_position_ids(3, offset=32_760, batch_size=1),  # near the context limit
    ],
    ids=["prefill", "decode", "per-seq-offset", "long-context"],
)
def test_cos_sin_and_rotation_bit_identical(dtype, position_ids) -> None:
    d, theta = QWEN3_0_6B_BASE.head_dim, QWEN3_0_6B_BASE.rope_theta
    b, t = position_ids.shape
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(b, 16, t, d, generator=gen).to(dtype)  # Nq = 16
    k = torch.randn(b, 8, t, d, generator=gen).to(dtype)  # Nkv = 8

    hf_cos, hf_sin = _hf_rotary(d, theta)(q, position_ids)  # x is only used for dtype/device
    cos, sin = RotaryEmbedding(d, theta)(position_ids, dtype=dtype)
    assert cos.shape == hf_cos.shape == (b, t, d)
    assert torch.equal(cos, hf_cos) and torch.equal(sin, hf_sin)

    hf_q, hf_k = modeling.apply_rotary_pos_emb(q, k, hf_cos, hf_sin)
    q_rot, k_rot = apply_rope_to_qk(q, k, cos, sin)
    assert torch.equal(q_rot, hf_q) and torch.equal(k_rot, hf_k)


def test_backward_identical() -> None:
    d = 64
    pos = make_position_ids(5, offset=3, batch_size=2)
    cos, sin = RotaryEmbedding(d, 1e6)(pos)
    gen = torch.Generator().manual_seed(1)
    q = torch.randn(2, 4, 5, d, generator=gen)
    k = torch.randn(2, 2, 5, d, generator=gen)
    qa, ka = q.clone().requires_grad_(), k.clone().requires_grad_()
    qb, kb = q.clone().requires_grad_(), k.clone().requires_grad_()
    oq, ok = apply_rope_to_qk(qa, ka, cos, sin)
    rq, rk = modeling.apply_rotary_pos_emb(qb, kb, cos, sin)
    (oq.sum() + 2 * ok.sum()).backward()
    (rq.sum() + 2 * rk.sum()).backward()
    assert torch.equal(qa.grad, qb.grad) and torch.equal(ka.grad, kb.grad)
