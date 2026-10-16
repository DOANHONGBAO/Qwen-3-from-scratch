"""Unit tests for rotary position embedding (M1.3). CPU tier, no attention module."""

from __future__ import annotations

import math

import pytest
import torch
from torch.testing import assert_close

from qwen3_scratch.config import QWEN3_0_6B_BASE, TINY_QWEN3
from qwen3_scratch.layers import (
    RotaryEmbedding,
    apply_rope,
    apply_rope_to_qk,
    compute_inv_freq,
    make_position_ids,
    rope_cos_sin,
    rotate_half,
)
from qwen3_scratch.layers.rope import (
    apply_rope_interleaved,
    half_layout_permutation,
    rotate_2d,
)


def _randn(*shape: int, seed: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64).to(dtype)


def _rope64(x: torch.Tensor, positions: torch.Tensor, theta: float = 10_000.0) -> torch.Tensor:
    """High-precision rotation: float64 frequencies, angles and tables."""
    inv = compute_inv_freq(x.shape[-1], theta, dtype=torch.float64)
    cos, sin = rope_cos_sin(positions, inv)
    return apply_rope(x, cos, sin)


# ------------------------------------------------------------ frequencies
def test_inv_freq_formula_and_range() -> None:
    inv = compute_inv_freq(8, 10_000.0, dtype=torch.float64)
    expected = torch.tensor([10_000.0 ** (-2 * i / 8) for i in range(4)], dtype=torch.float64)
    assert_close(inv, expected, rtol=1e-15, atol=0.0)
    assert inv[0] == 1.0 and (inv[1:] < inv[:-1]).all()  # strictly decreasing


def test_inv_freq_qwen3_0_6b() -> None:
    inv = compute_inv_freq(QWEN3_0_6B_BASE.head_dim, QWEN3_0_6B_BASE.rope_theta)
    assert inv.shape == (64,) and inv.dtype == torch.float32
    # Slowest pair: theta^(-126/128) ~ 1.24e-6 rad/token -> wavelength ~ 5.1M tokens.
    assert_close(inv[-1].double(), torch.tensor(1e6 ** (-126 / 128), dtype=torch.float64))
    assert 2 * math.pi / inv[-1].item() > 5e6


@pytest.mark.parametrize("theta", [10.0, 10_000.0, 1_000_000.0])
def test_theta_controls_slowest_frequency(theta) -> None:
    inv = compute_inv_freq(16, theta, dtype=torch.float64)
    assert_close(inv[-1], torch.tensor(theta ** (-14 / 16), dtype=torch.float64))


def test_cos_sin_shapes_and_duplication() -> None:
    inv = compute_inv_freq(8, 10_000.0)
    cos, sin = rope_cos_sin(make_position_ids(5), inv)
    assert cos.shape == sin.shape == (5, 8)
    assert torch.equal(cos[:, :4], cos[:, 4:]) and torch.equal(sin[:, :4], sin[:, 4:])
    cos_b, _ = rope_cos_sin(make_position_ids(5, batch_size=3), inv)
    assert cos_b.shape == (3, 5, 8)


def test_cos_sin_dtype_cast_happens_last() -> None:
    inv = compute_inv_freq(8, 10_000.0)
    pos = make_position_ids(7)
    cos32, sin32 = rope_cos_sin(pos, inv)
    cos16, sin16 = rope_cos_sin(pos, inv, dtype=torch.bfloat16)
    assert cos16.dtype == torch.bfloat16
    assert torch.equal(cos16, cos32.to(torch.bfloat16)) and torch.equal(
        sin16, sin32.to(torch.bfloat16)
    )


def test_position_ids_helper() -> None:
    assert make_position_ids(4).tolist() == [0, 1, 2, 3]
    assert make_position_ids(3, offset=5).tolist() == [5, 6, 7]
    assert make_position_ids(2, offset=1, batch_size=2).tolist() == [[1, 2], [1, 2]]
    assert make_position_ids(0).numel() == 0


# --------------------------------------------------- 1. shape preservation
@pytest.mark.parametrize(("b", "n", "t", "d"), [(1, 1, 1, 2), (2, 4, 7, 32), (3, 16, 5, 128)])
def test_shape_and_dtype_preserved(b, n, t, d) -> None:
    rope = RotaryEmbedding(d)
    x = _randn(b, n, t, d)
    cos, sin = rope(make_position_ids(t))
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape and y.dtype == x.dtype


# ----------------------------------------------- 2. manual / 2D rotation
def test_manual_example_head_dim_4() -> None:
    # docs/components/rope.md §4: D=4, theta=10000 -> w = [1, 0.01]; position 1.
    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64).view(1, 1, 1, 4)
    y = _rope64(x, torch.tensor([1])).flatten()
    c1, s1, c2, s2 = math.cos(1), math.sin(1), math.cos(0.01), math.sin(0.01)
    expected = [1 * c1 - 3 * s1, 2 * c2 - 4 * s2, 3 * c1 + 1 * s1, 4 * c2 + 2 * s2]
    assert_close(y, torch.tensor(expected, dtype=torch.float64), rtol=1e-15, atol=1e-15)


def test_each_pair_is_a_2d_rotation() -> None:
    d, pos = 8, 3
    x = _randn(1, 1, 1, d, dtype=torch.float64)
    y = _rope64(x, torch.tensor([pos])).flatten()
    inv = compute_inv_freq(d, 10_000.0, dtype=torch.float64)
    xf = x.flatten()
    for i in range(d // 2):
        pair = torch.stack((xf[i], xf[i + d // 2]))  # split-half pairing
        rotated = rotate_2d(pair, pos * inv[i])
        assert_close(torch.stack((y[i], y[i + d // 2])), rotated, rtol=1e-14, atol=1e-14)


def test_rotate_2d_matrix_convention() -> None:
    v = torch.tensor([1.0, 0.0], dtype=torch.float64)
    assert_close(rotate_2d(v, math.pi / 2), torch.tensor([0.0, 1.0], dtype=torch.float64))
    assert_close(rotate_2d(v, math.pi), torch.tensor([-1.0, 0.0], dtype=torch.float64))


def test_rotate_half_is_90_degree_turn() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert rotate_half(x).tolist() == [-3.0, -4.0, 1.0, 2.0]
    assert torch.equal(rotate_half(rotate_half(x)), -x)  # two 90-degree turns


# ------------------------------------------------- 3. norm preservation
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_norm_preserved_per_pair_and_per_head(dtype) -> None:
    d = 64
    x = _randn(2, 3, 50, d, seed=1, dtype=dtype)
    pos = make_position_ids(50, offset=1000)
    if dtype == torch.float64:  # float64 tables: cos^2 + sin^2 = 1 to ~1e-16
        y = _rope64(x, pos, theta=1_000_000.0)
        tol = {"rtol": 1e-12, "atol": 0.0}
    else:  # HF-style float32 tables: cos^2 + sin^2 = 1 only to float32 accuracy
        y = apply_rope(x, *RotaryEmbedding(d, 1_000_000.0)(pos, dtype=dtype))
        tol = {"rtol": 1e-6, "atol": 1e-6}
    assert_close(y.norm(dim=-1), x.norm(dim=-1), **tol)
    pair_norm = lambda t: t[..., : d // 2] ** 2 + t[..., d // 2 :] ** 2  # noqa: E731
    assert_close(pair_norm(y), pair_norm(x), **tol)


# --------------------------------------------------- 4. position zero
def test_position_zero_is_identity() -> None:
    x = _randn(2, 4, 1, 32, seed=2)
    cos, sin = RotaryEmbedding(32)(torch.zeros(1, dtype=torch.long))
    assert torch.equal(cos, torch.ones_like(cos)) and torch.equal(sin, torch.zeros_like(sin))
    assert torch.equal(apply_rope(x, cos, sin), x)


# ------------------------------------- 5, 6. positions and sequence lengths
def test_different_positions_rotate_differently() -> None:
    x = _randn(1, 1, 1, 16, seed=3).expand(1, 1, 4, 16).contiguous()  # same vector, 4 tokens
    y = apply_rope(x, *RotaryEmbedding(16)(make_position_ids(4)))
    for a in range(4):
        for b in range(a + 1, 4):
            assert not torch.allclose(y[..., a, :], y[..., b, :])


@pytest.mark.parametrize("t", [1, 2, 17, 128])
def test_sequence_lengths_match_float64_reference(t) -> None:
    x = _randn(2, 2, t, 32, seed=t)
    pos = make_position_ids(t)
    y = apply_rope(x, *RotaryEmbedding(32)(pos))
    assert_close(y.double(), _rope64(x.double(), pos), rtol=1e-5, atol=1e-5)


def test_batched_position_ids_with_different_offsets() -> None:
    # Two sequences at different positions (e.g. left padding, or different cache lengths).
    pos = torch.tensor([[0, 1, 2], [10, 11, 12]])
    x = _randn(2, 3, 3, 8, seed=4)
    rope = RotaryEmbedding(8)
    y = apply_rope(x, *rope(pos))
    for b in range(2):
        yb = apply_rope(x[b : b + 1], *rope(pos[b]))
        assert torch.equal(y[b : b + 1], yb)


# ------------------------------------------------ 7. different head counts
def test_q_and_k_with_different_head_counts() -> None:
    cfg = TINY_QWEN3
    t = 6
    q = _randn(2, cfg.num_attention_heads, t, cfg.head_dim, seed=5)
    k = _randn(2, cfg.num_key_value_heads, t, cfg.head_dim, seed=6)
    cos, sin = RotaryEmbedding.from_config(cfg)(make_position_ids(t))
    q_rot, k_rot = apply_rope_to_qk(q, k, cos, sin)
    assert q_rot.shape == q.shape and k_rot.shape == k.shape
    # Every head is rotated by the same per-position rotation.
    for h in range(cfg.num_key_value_heads):
        assert torch.equal(k_rot[:, h : h + 1], apply_rope(k[:, h : h + 1], cos, sin))


# --------------------------------------------------- 8. theta configuration
def test_from_config_uses_head_dim_and_theta() -> None:
    rope = RotaryEmbedding.from_config(QWEN3_0_6B_BASE)
    assert rope.head_dim == 128 and rope.rope_theta == 1_000_000.0
    assert_close(rope.inv_freq, compute_inv_freq(128, 1_000_000.0), rtol=0, atol=0)
    assert list(rope.state_dict()) == []  # tables are recomputed, never stored in checkpoints


def test_larger_theta_rotates_slower() -> None:
    pos = torch.tensor([100])
    _, sin_small = RotaryEmbedding(64, 10_000.0)(pos)
    _, sin_large = RotaryEmbedding(64, 1_000_000.0)(pos)
    angle_small = torch.asin(sin_small[0, 40].double())
    angle_large = torch.asin(sin_large[0, 40].double())
    assert angle_large.abs() < angle_small.abs()


# -------------------------------------------- 10. gradient propagation
def test_gradients_are_inverse_rotation() -> None:
    # y = R x with R orthogonal => dL/dx = R^T dL/dy, i.e. rotation by the negative angle.
    d, pos = 16, 7
    x = _randn(1, 2, 1, d, seed=7, dtype=torch.float64).requires_grad_()
    inv = compute_inv_freq(d, 10_000.0, dtype=torch.float64)
    cos, sin = rope_cos_sin(torch.tensor([pos]), inv)
    u = _randn(1, 2, 1, d, seed=8, dtype=torch.float64)
    apply_rope(x, cos, sin).backward(u)
    expected = apply_rope(u, cos, -sin)  # R(-angle)
    assert_close(x.grad, expected, rtol=1e-14, atol=1e-14)
    assert_close(x.grad.norm(), u.norm())


def test_gradcheck_float64() -> None:
    inv = compute_inv_freq(8, 10_000.0, dtype=torch.float64)
    cos, sin = rope_cos_sin(make_position_ids(3, offset=2), inv)
    x = _randn(2, 2, 3, 8, seed=9, dtype=torch.float64).requires_grad_()
    assert torch.autograd.gradcheck(lambda t: apply_rope(t, cos, sin), (x,))
    cos_r = cos.clone().requires_grad_()
    sin_r = sin.clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda a, b: apply_rope(x.detach(), a, b), (cos_r, sin_r))


# ------------------- 11, 12, 13. offsets, incremental and causal decoding
def test_offset_equals_slice_of_full_sequence() -> None:
    rope = RotaryEmbedding(32, 1_000_000.0)
    full_cos, full_sin = rope(make_position_ids(20))
    for offset, length in [(0, 5), (5, 1), (7, 13), (19, 1)]:
        cos, sin = rope(make_position_ids(length, offset=offset))
        assert torch.equal(cos, full_cos[offset : offset + length])
        assert torch.equal(sin, full_sin[offset : offset + length])


def test_incremental_rotation_equals_full_sequence() -> None:
    rope = RotaryEmbedding(32, 1_000_000.0)
    x = _randn(2, 4, 12, 32, seed=10)
    full = apply_rope(x, *rope(make_position_ids(12)))
    chunks, start = [], 0
    for length in (5, 1, 1, 3, 2):  # prefill 5 tokens, then decode in steps
        sl = slice(start, start + length)
        chunks.append(apply_rope(x[:, :, sl], *rope(make_position_ids(length, offset=start))))
        start += length
    assert torch.equal(torch.cat(chunks, dim=2), full)


def test_causal_decoding_scores_match_full_sequence() -> None:
    # Simulate a KV cache with plain tensors: keys are rotated once at their own
    # position and stored; each new query is rotated at its position.
    rope = RotaryEmbedding(16)
    t = 8
    q = _randn(1, 2, t, 16, seed=11, dtype=torch.float64)
    k = _randn(1, 2, t, 16, seed=12, dtype=torch.float64)
    cos, sin = rope(make_position_ids(t), dtype=torch.float64)
    q_full, k_full = apply_rope_to_qk(q, k, cos, sin)
    full_scores = q_full @ k_full.transpose(-1, -2)  # (1, 2, T, T)

    cached_k = []
    for pos in range(t):
        c, s = rope(make_position_ids(1, offset=pos), dtype=torch.float64)
        q_new, k_new = apply_rope_to_qk(q[:, :, pos : pos + 1], k[:, :, pos : pos + 1], c, s)
        cached_k.append(k_new)
        keys = torch.cat(cached_k, dim=2)  # positions 0..pos
        step_scores = q_new @ keys.transpose(-1, -2)  # (1, 2, 1, pos+1)
        assert_close(step_scores[..., 0, :], full_scores[..., pos, : pos + 1])


def test_cached_tables_are_bit_identical_to_direct() -> None:
    cached = RotaryEmbedding(64, 1_000_000.0, max_position_embeddings=256)
    direct = RotaryEmbedding(64, 1_000_000.0)
    pos = torch.tensor([[0, 5, 255], [100, 101, 102]])
    for a, b in zip(cached(pos, torch.bfloat16), direct(pos, torch.bfloat16), strict=True):
        assert torch.equal(a, b)
    beyond = torch.tensor([300, 301])  # past the cache -> computed directly
    for a, b in zip(cached(beyond), direct(beyond), strict=True):
        assert torch.equal(a, b)


# ------------------------------------- relative-position property (math)
@pytest.mark.parametrize(("m", "n", "shift"), [(0, 3, 5), (2, 9, 100), (7, 1, 1234), (40, 40, 7)])
def test_scores_depend_only_on_relative_position(m, n, shift) -> None:
    d = 32
    q = _randn(1, 1, 1, d, seed=13, dtype=torch.float64)
    k = _randn(1, 1, 1, d, seed=14, dtype=torch.float64)

    def score(pq: int, pk: int) -> torch.Tensor:
        return (_rope64(q, torch.tensor([pq])) * _rope64(k, torch.tensor([pk]))).sum()

    assert_close(score(m + shift, n + shift), score(m, n), rtol=1e-10, atol=1e-10)
    # <R_m q, R_n k> = <q, R_{n-m} k>
    if n >= m:
        assert_close(score(m, n), (q * _rope64(k, torch.tensor([n - m]))).sum())


def test_relative_property_with_float32_tables_degrades_slowly() -> None:
    # HF-compatible float32 angles: the property holds to float32 accuracy.
    d = 128
    rope = RotaryEmbedding(d, 1_000_000.0)
    q = _randn(1, 1, 1, d, seed=15, dtype=torch.float64)
    k = _randn(1, 1, 1, d, seed=16, dtype=torch.float64)

    def score(pq: int, pk: int) -> float:
        cq, sq = rope(torch.tensor([pq]), dtype=torch.float64)
        ck, sk = rope(torch.tensor([pk]), dtype=torch.float64)
        return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum().item()

    base = score(3, 10)
    # Error grows with absolute position because float32 angles lose precision
    # (see scripts/rope_report.py); bounds are loose to be platform independent.
    assert abs(score(1003, 1010) - base) < 1e-3
    assert abs(score(30003, 30010) - base) < 2e-2


# ------------------------ split-half vs interleaved: math vs compatibility
def test_interleaved_equals_split_half_up_to_permutation() -> None:
    d, t = 16, 5
    inv = compute_inv_freq(d, 10_000.0, dtype=torch.float64)
    pos = make_position_ids(t, offset=3)
    x_inter = _randn(2, 3, t, d, seed=17, dtype=torch.float64)  # features in interleaved order
    perm = half_layout_permutation(d)
    y_inter = apply_rope_interleaved(x_inter, pos, inv)
    y_half = apply_rope(x_inter[..., perm], *rope_cos_sin(pos, inv))
    assert_close(y_half, y_inter[..., perm], rtol=1e-14, atol=1e-14)


def test_permuted_layouts_give_identical_attention_scores() -> None:
    d = 16
    inv = compute_inv_freq(d, 10_000.0, dtype=torch.float64)
    q = _randn(1, 1, 4, d, seed=18, dtype=torch.float64)
    k = _randn(1, 1, 4, d, seed=19, dtype=torch.float64)
    pos = make_position_ids(4)
    perm = half_layout_permutation(d)
    scores_inter = apply_rope_interleaved(q, pos, inv) @ apply_rope_interleaved(k, pos, inv).mT
    cos, sin = rope_cos_sin(pos, inv)
    scores_half = apply_rope(q[..., perm], cos, sin) @ apply_rope(k[..., perm], cos, sin).mT
    assert_close(scores_half, scores_inter, rtol=1e-12, atol=1e-12)


def test_wrong_pairing_without_permutation_breaks_scores() -> None:
    # Same weights/activations, wrong convention -> different attention scores.
    d = 16
    inv = compute_inv_freq(d, 10_000.0, dtype=torch.float64)
    q = _randn(1, 1, 4, d, seed=20, dtype=torch.float64)
    k = _randn(1, 1, 4, d, seed=21, dtype=torch.float64)
    pos = make_position_ids(4)
    cos, sin = rope_cos_sin(pos, inv)
    right = apply_rope(q, cos, sin) @ apply_rope(k, cos, sin).mT
    wrong = apply_rope_interleaved(q, pos, inv) @ apply_rope_interleaved(k, pos, inv).mT
    assert (right - wrong).abs().max() > 1e-2


def test_permutation_layout() -> None:
    assert half_layout_permutation(8).tolist() == [0, 2, 4, 6, 1, 3, 5, 7]


# ------------------------------------------------------ determinism
def test_deterministic() -> None:
    x = _randn(2, 4, 9, 32, seed=22)
    pos = make_position_ids(9, offset=3)
    a = apply_rope(x, *RotaryEmbedding(32, 1e6)(pos))
    b = apply_rope(x, *RotaryEmbedding(32, 1e6)(pos))
    assert torch.equal(a, b)


# ---------------------------------------------------- 14. invalid input
@pytest.mark.parametrize("head_dim", [0, -2, 3, 7, 2.0, True, "8", None])
def test_invalid_head_dim(head_dim) -> None:
    with pytest.raises(ValueError):
        RotaryEmbedding(head_dim)
    with pytest.raises(ValueError):
        compute_inv_freq(head_dim, 10_000.0)


@pytest.mark.parametrize("theta", [0.0, -1.0, float("inf"), float("nan"), True, "1e4", None])
def test_invalid_theta(theta) -> None:
    with pytest.raises(ValueError):
        RotaryEmbedding(8, theta)


@pytest.mark.parametrize("max_pos", [0, -1, 1.5, True])
def test_invalid_max_positions(max_pos) -> None:
    with pytest.raises(ValueError):
        RotaryEmbedding(8, max_position_embeddings=max_pos)


@pytest.mark.parametrize(
    ("pos", "exc"),
    [
        (torch.tensor([0.0, 1.0]), TypeError),
        (torch.tensor([True, False]), TypeError),
        (torch.tensor(3), ValueError),
        (torch.zeros(1, 2, 3, dtype=torch.long), ValueError),
        (torch.tensor([0, -1]), ValueError),
        ([0, 1], TypeError),
    ],
)
def test_invalid_position_ids(pos, exc) -> None:
    with pytest.raises(exc):
        RotaryEmbedding(8)(pos)


def test_invalid_make_position_ids() -> None:
    for kwargs in ({"seq_len": -1}, {"seq_len": 2, "offset": -1}, {"seq_len": 2, "batch_size": 0}):
        with pytest.raises(ValueError):
            make_position_ids(**kwargs)


def test_apply_rope_shape_checks() -> None:
    rope = RotaryEmbedding(8)
    cos, sin = rope(make_position_ids(4))
    with pytest.raises(ValueError, match=r"\(B, N, T, D\)"):
        apply_rope(torch.randn(4, 8), cos, sin)  # missing batch/head axes
    with pytest.raises(ValueError, match="do not match"):
        apply_rope(torch.randn(1, 2, 5, 8), cos, sin)  # T mismatch
    with pytest.raises(ValueError, match="do not match"):
        apply_rope(torch.randn(1, 2, 4, 16), cos, sin)  # D mismatch
    cos_b, sin_b = rope(make_position_ids(4, batch_size=3))
    with pytest.raises(ValueError, match="do not match"):
        apply_rope(torch.randn(2, 2, 4, 8), cos_b, sin_b)  # batch 3 vs 2
    with pytest.raises(ValueError, match="dtype"):
        apply_rope(torch.randn(1, 2, 4, 8, dtype=torch.bfloat16), cos, sin)
    with pytest.raises(ValueError, match="differ"):
        apply_rope(torch.randn(1, 2, 4, 8), cos, sin[:3])
    with pytest.raises(ValueError, match="even"):
        apply_rope(torch.randn(1, 1, 4, 7), torch.ones(4, 7), torch.zeros(4, 7))
    with pytest.raises(TypeError):
        apply_rope(torch.ones(1, 1, 4, 8, dtype=torch.long), cos, sin)


def test_q_k_mismatch_rejected() -> None:
    cos, sin = RotaryEmbedding(8)(make_position_ids(4))
    with pytest.raises(ValueError, match="must agree"):
        apply_rope_to_qk(torch.randn(1, 4, 4, 8), torch.randn(1, 2, 3, 8), cos, sin)
