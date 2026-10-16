"""Unit tests for QK-Norm (M1.4, part A). CPU tier."""

from __future__ import annotations

import math

import pytest
import torch
from torch.testing import assert_close

from qwen3_scratch.config import QWEN3_0_6B_BASE, TINY_QWEN3
from qwen3_scratch.layers import QKNorm, RMSNorm, apply_qk_norm, rms_norm_textbook

EPS = 1e-6


def _randn(*shape: int, seed: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64).to(dtype)


def _qknorm(d: int, seed: int = 1, dtype: torch.dtype = torch.float32, eps: float = EPS) -> QKNorm:
    m = QKNorm(d, eps, dtype=dtype)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.q_norm.weight.copy_(1 + 0.3 * torch.randn(d, generator=gen))
        m.k_norm.weight.copy_(1 - 0.3 * torch.randn(d, generator=gen))
    return m


# ---------------------------------------------------- structure / params
def test_two_independent_parameter_vectors() -> None:
    m = QKNorm(16)
    params = dict(m.named_parameters())
    assert set(params) == {"q_norm.weight", "k_norm.weight"}
    assert params["q_norm.weight"].shape == params["k_norm.weight"].shape == (16,)
    assert params["q_norm.weight"].data_ptr() != params["k_norm.weight"].data_ptr()
    assert "head_dim=16, eps=1e-06" in repr(m)


def test_from_qwen3_sizes() -> None:
    m = QKNorm(QWEN3_0_6B_BASE.head_dim, QWEN3_0_6B_BASE.rms_norm_eps)
    assert m.head_dim == 128 and sum(p.numel() for p in m.parameters()) == 256


# ------------------------------------------------- 1, 2, 3. shapes
@pytest.mark.parametrize(("nq", "nkv"), [(4, 4), (4, 2), (4, 1), (16, 8)])
@pytest.mark.parametrize(("b", "t"), [(1, 1), (2, 5), (3, 17)])
def test_shapes_with_different_head_counts(nq, nkv, b, t) -> None:
    d = 32
    q, k = _randn(b, nq, t, d, seed=2), _randn(b, nkv, t, d, seed=3)
    qn, kn = _qknorm(d)(q, k)
    assert qn.shape == q.shape and kn.shape == k.shape
    assert qn.dtype == kn.dtype == torch.float32


# ---------------------------------------------- 4. manual calculation
def test_manual_example() -> None:
    # One head vector q = [3, 4]: mean square 12.5 -> rms sqrt(12.5 + eps).
    m = QKNorm(2, dtype=torch.float64)
    with torch.no_grad():
        m.q_norm.weight.copy_(torch.tensor([2.0, 0.5]))
    q = torch.tensor([3.0, 4.0], dtype=torch.float64).view(1, 1, 1, 2)
    k = torch.tensor([-1.0, 1.0], dtype=torch.float64).view(1, 1, 1, 2)
    qn, kn = m(q, k)
    rms_q = math.sqrt(12.5 + EPS)
    assert_close(qn.flatten(), torch.tensor([2 * 3 / rms_q, 0.5 * 4 / rms_q], dtype=torch.float64))
    rms_k = math.sqrt(1.0 + EPS)
    assert_close(kn.flatten(), torch.tensor([-1 / rms_k, 1 / rms_k], dtype=torch.float64))


def test_each_head_vector_normalised_independently() -> None:
    d = 8
    m = QKNorm(d, dtype=torch.float64)  # unit weights
    q = _randn(2, 4, 3, d, seed=4, dtype=torch.float64)
    q[0, 1] *= 1000.0  # one head very large
    q[1, 3, 2] *= 1e-3  # one token of one head small
    qn, _ = m(q, _randn(2, 2, 3, d, seed=5, dtype=torch.float64))
    ms = qn.pow(2).mean(-1)  # mean square per (b, h, t)
    expected = 1 / (1 + EPS / q.pow(2).mean(-1))
    assert_close(ms, expected, rtol=1e-12, atol=0)


# --------------------------------------------- 5. reference equivalence
def test_matches_rms_norm_per_head_reference() -> None:
    d = 16
    m = _qknorm(d, dtype=torch.float64)
    q, k = (
        _randn(2, 4, 5, d, seed=6, dtype=torch.float64),
        _randn(2, 2, 5, d, seed=7, dtype=torch.float64),
    )
    qn, kn = m(q, k)
    for h in range(4):
        assert_close(qn[:, h], rms_norm_textbook(q[:, h], m.q_norm.weight, EPS))
    for h in range(2):
        assert_close(kn[:, h], rms_norm_textbook(k[:, h], m.k_norm.weight, EPS))


def test_layout_independent_bth_vs_bht() -> None:
    # HF normalises in (B, T, N, D) before transposing; we normalise (B, N, T, D).
    d = 32
    m = _qknorm(d)
    q_btnd = _randn(2, 6, 4, d, seed=8)
    k_btnd = _randn(2, 6, 2, d, seed=9)
    qa, ka = m(q_btnd.transpose(1, 2), k_btnd.transpose(1, 2))
    qb, kb = m.q_norm(q_btnd).transpose(1, 2), m.k_norm(k_btnd).transpose(1, 2)
    assert torch.equal(qa, qb) and torch.equal(ka, kb)


def test_functional_and_module_paths_identical() -> None:
    m = _qknorm(16)
    q, k = _randn(1, 4, 3, 16, seed=10), _randn(1, 2, 3, 16, seed=11)
    a = m(q, k)
    b = apply_qk_norm(q, k, m.q_norm, m.k_norm)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# ---------------------------------------- QK-Norm bounds attention logits
@pytest.mark.parametrize("scale", [1e-2, 1.0, 1e3])
def test_logits_bounded_independent_of_input_scale(scale) -> None:
    d = 64
    m = QKNorm(d)  # unit weights
    q = scale * _randn(1, 2, 8, d, seed=12)
    k = scale * _randn(1, 2, 8, d, seed=13)
    qn, kn = m(q, k)
    logits = qn @ kn.transpose(-1, -2) / math.sqrt(d)
    # |q_n|, |k_n| <= sqrt(D)  =>  |q_n . k_n| / sqrt(D) <= sqrt(D)
    assert logits.abs().max() <= math.sqrt(d) * (1 + 1e-5)
    raw = q @ k.transpose(-1, -2) / math.sqrt(d)
    if scale == 1e3:
        assert raw.abs().max() > 1e5  # without QK-Norm logits explode with scale


# ----------------------------------------------------- 6, 7. gradients
def test_gradients_flow_to_inputs_and_separate_weights() -> None:
    m = _qknorm(16)
    q = _randn(2, 4, 3, 16, seed=14).requires_grad_()
    k = _randn(2, 2, 3, 16, seed=15).requires_grad_()
    qn, kn = m(q, k)
    (qn.pow(2).sum() + 3 * kn.sum()).backward()
    for t in (q.grad, k.grad, m.q_norm.weight.grad, m.k_norm.weight.grad):
        assert t is not None and torch.isfinite(t).all() and t.abs().sum() > 0


def test_q_loss_does_not_touch_k_norm() -> None:
    m = _qknorm(16)
    qn, kn = m(_randn(1, 2, 3, 16, seed=16), _randn(1, 1, 3, 16, seed=17))
    qn.sum().backward()
    assert m.q_norm.weight.grad is not None
    assert m.k_norm.weight.grad is None  # independent parameters


def test_weight_shared_across_heads_and_tokens() -> None:
    # The weight gradient is the sum over all (batch, head, token) positions.
    d = 8
    m = QKNorm(d, dtype=torch.float64)
    q = _randn(2, 3, 4, d, seed=18, dtype=torch.float64)
    u = _randn(2, 3, 4, d, seed=19, dtype=torch.float64)
    qn, _ = m(q, _randn(2, 1, 4, d, seed=20, dtype=torch.float64))
    qn.backward(u)
    x_hat = q / torch.sqrt(q.pow(2).mean(-1, keepdim=True) + EPS)
    assert_close(m.q_norm.weight.grad, (u * x_hat).sum(dim=(0, 1, 2)))


def test_gradcheck() -> None:
    m = _qknorm(6, dtype=torch.float64)
    q = _randn(1, 2, 3, 6, seed=21, dtype=torch.float64).requires_grad_()
    k = _randn(1, 1, 3, 6, seed=22, dtype=torch.float64).requires_grad_()
    assert torch.autograd.gradcheck(lambda a, b: m(a, b), (q, k))


# -------------------------------------------- 8, 9. small values and eps
def test_zero_and_tiny_inputs() -> None:
    m = QKNorm(8)
    qn, kn = m(torch.zeros(1, 2, 3, 8), torch.full((1, 1, 3, 8), 1e-20))
    assert torch.equal(qn, torch.zeros_like(qn))
    assert torch.isfinite(kn).all() and kn.abs().max() < 1e-10  # eps damps tiny keys


@pytest.mark.parametrize("eps", [1e-8, 1e-6, 1e-5, 1e-2])
def test_eps_is_used(eps) -> None:
    d = 4
    q = torch.full((1, 1, 1, d), 0.01, dtype=torch.float64)  # mean square 1e-4
    qn, _ = QKNorm(d, eps, dtype=torch.float64)(q, q.clone())
    expected = 0.01 / math.sqrt(1e-4 + eps)
    assert_close(qn, torch.full_like(qn, expected))


# ---------------------------------------- 10, 11. sizes and fp32 CPU
@pytest.mark.parametrize("cfg", [TINY_QWEN3, QWEN3_0_6B_BASE], ids=["tiny", "0.6b"])
def test_fp32_cpu_matches_float64(cfg) -> None:
    d = cfg.head_dim
    m32 = _qknorm(d)
    m64 = QKNorm(d, dtype=torch.float64)
    m64.load_state_dict({k: v.double() for k, v in m32.state_dict().items()})
    q = _randn(2, cfg.num_attention_heads, 7, d, seed=23)
    k = _randn(2, cfg.num_key_value_heads, 7, d, seed=24)
    out32, out64 = m32(q, k), m64(q.double(), k.double())
    for a, b in zip(out32, out64, strict=True):
        assert a.device.type == "cpu" and a.dtype == torch.float32
        assert_close(a.double(), b, rtol=1e-6, atol=1e-6)


def test_bfloat16_keeps_dtype() -> None:
    m = QKNorm(16, dtype=torch.bfloat16)
    qn, kn = m(_randn(1, 4, 2, 16, dtype=torch.bfloat16), _randn(1, 2, 2, 16, dtype=torch.bfloat16))
    assert qn.dtype == kn.dtype == torch.bfloat16


def test_deterministic() -> None:
    m = _qknorm(16)
    q, k = _randn(2, 4, 3, 16, seed=25), _randn(2, 2, 3, 16, seed=26)
    a, b = m(q, k), m(q, k)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# --------------------------------------------------- 12. invalid input
def test_invalid_head_dim_and_eps() -> None:
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            QKNorm(bad)
    with pytest.raises(ValueError):
        QKNorm(8, eps=0.0)


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "match"),
    [
        ((1, 2, 3, 8), (1, 2, 3, 4), "head_dim"),  # k wrong D
        ((1, 2, 3, 4), (1, 2, 3, 8), "head_dim"),  # q wrong D
        ((2, 3, 8), (1, 2, 3, 8), r"\(B, N, T, D\)"),  # q missing an axis
        ((1, 2, 3, 8), (1, 2, 3, 2, 8), r"\(B, N, T, D\)"),
        ((1, 2, 3, 8), (2, 2, 3, 8), "batch"),
        ((1, 2, 3, 8), (1, 2, 4, 8), "sequence"),
    ],
)
def test_invalid_shapes(q_shape, k_shape, match) -> None:
    with pytest.raises(ValueError, match=match):
        QKNorm(8)(torch.randn(q_shape), torch.randn(k_shape))


def test_mismatched_norm_sizes_rejected() -> None:
    with pytest.raises(ValueError, match="same size"):
        apply_qk_norm(torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8), RMSNorm(8), RMSNorm(4))


def test_non_tensor_rejected() -> None:
    with pytest.raises(TypeError):
        QKNorm(2)([[1.0, 2.0]], torch.randn(1, 1, 1, 2))
