"""Unit tests for RMSNorm (M1.1). CPU tier: no GPU, no weights, no network."""

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

from qwen3_scratch.config import QWEN3_0_6B_BASE, TINY_QWEN3
from qwen3_scratch.layers import RMSNorm, rms_norm, rms_norm_textbook

EPS = 1e-6
# Hand-computed with 30-digit decimal arithmetic (see docs/components/rmsnorm.md §3).
MANUAL_X = [1.0, 2.0, 3.0, 4.0]
MANUAL_GAMMA = [0.5, 1.0, -1.0, 2.0]
MANUAL_RMS = 2.73861297010001031653443128486
MANUAL_Y_UNIT_GAMMA = [
    0.365148347326888398619471022052,
    0.730296694653776797238942044104,
    1.09544504198066519585841306616,
    1.46059338930755359447788408821,
]
MANUAL_Y = [
    0.182574173663444199309735511026,
    0.730296694653776797238942044104,
    -1.09544504198066519585841306616,
    2.92118677861510718895576817642,
]


def _randn(*shape: int, seed: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64).to(dtype)


def _random_norm(d: int, seed: int = 1, dtype: torch.dtype = torch.float32) -> RMSNorm:
    norm = RMSNorm(d, eps=EPS, dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.5 * _randn(d, seed=seed, dtype=torch.float64))
    return norm


# --------------------------------------------------------------- construction
def test_parameters_initialised_to_ones_with_hf_name() -> None:
    norm = RMSNorm(8)
    assert list(dict(norm.named_parameters())) == ["weight"]
    assert norm.weight.shape == (8,)
    assert norm.weight.requires_grad
    assert torch.equal(norm.weight, torch.ones(8))
    assert norm.eps == 1e-6
    assert "8, eps=1e-06" in repr(norm)


def test_dtype_argument_controls_weight_dtype() -> None:
    assert RMSNorm(4, dtype=torch.bfloat16).weight.dtype == torch.bfloat16


@pytest.mark.parametrize("cfg", [TINY_QWEN3, QWEN3_0_6B_BASE], ids=["tiny", "0.6b"])
def test_sizes_from_config(cfg) -> None:
    block_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    qk_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
    assert block_norm.weight.numel() == cfg.hidden_size
    assert qk_norm.weight.numel() == cfg.head_dim


# ------------------------------------------------------------- 1. shapes
@pytest.mark.parametrize(
    "shape",
    [(16,), (3, 16), (2, 5, 16), (2, 4, 5, 16), (0, 16), (2, 0, 16)],
    ids=["1d", "2d", "BTd", "BHTd", "empty-batch", "empty-seq"],
)
def test_output_shape_and_dtype_preserved(shape) -> None:
    x = _randn(*shape)
    y = RMSNorm(16)(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


# ------------------------------------------ 2. normalisation over last dim
def test_unit_weight_gives_unit_mean_square_on_last_dim() -> None:
    x = _randn(4, 7, 32, dtype=torch.float64) * 3.0 + 1.0
    y = RMSNorm(32, dtype=torch.float64)(x)
    mean_square = y.pow(2).mean(dim=-1)
    # Exactly 1 / (1 + eps / mean_square(x)); far from 1 only for tiny inputs.
    expected = 1.0 / (1.0 + EPS / x.pow(2).mean(dim=-1))
    assert_close(mean_square, expected, rtol=1e-12, atol=0.0)


def test_rows_are_normalised_independently() -> None:
    x = _randn(3, 5, 16)
    norm = _random_norm(16)
    y = norm(x)
    x2 = x.clone()
    x2[1, 2] *= 1000.0  # perturb one position only
    y2 = norm(x2)
    mask = torch.ones(3, 5, dtype=torch.bool)
    mask[1, 2] = False
    assert torch.equal(y[mask], y2[mask])


def test_reduction_is_not_over_other_dims() -> None:
    # Normalising over the wrong axis is a classic bug: transposing must NOT commute.
    x = _randn(6, 6, seed=3)
    y = RMSNorm(6)(x)
    wrong = RMSNorm(6)(x.T).T
    assert not torch.allclose(y, wrong)


# ------------------------------------------------- 3. manual example
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_manual_example_unit_gamma(dtype) -> None:
    x = torch.tensor(MANUAL_X, dtype=dtype)
    y = RMSNorm(4, dtype=dtype)(x)
    tol = {"rtol": 1e-6, "atol": 1e-7} if dtype == torch.float32 else {"rtol": 1e-14, "atol": 0.0}
    assert_close(y, torch.tensor(MANUAL_Y_UNIT_GAMMA, dtype=dtype), **tol)


def test_manual_example_with_gamma_float64() -> None:
    x = torch.tensor(MANUAL_X, dtype=torch.float64)
    gamma = torch.tensor(MANUAL_GAMMA, dtype=torch.float64)
    assert_close(rms_norm(x, gamma, EPS), torch.tensor(MANUAL_Y, dtype=torch.float64))
    rms = torch.sqrt(x.pow(2).mean() + EPS)
    assert abs(rms.item() - MANUAL_RMS) < 1e-14


# --------------------------------------- 4. trusted references (float)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matches_textbook_formula(dtype) -> None:
    x = _randn(2, 5, 64, seed=4, dtype=dtype)
    norm = _random_norm(64, dtype=dtype)
    expected = rms_norm_textbook(x, norm.weight, EPS)
    tol = {} if dtype == torch.float64 else {"rtol": 1e-6, "atol": 1e-6}
    assert_close(norm(x), expected, **tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matches_torch_nn_rmsnorm_with_explicit_eps(dtype) -> None:
    # torch.nn.RMSNorm's default eps is None -> finfo(dtype).eps, NOT 1e-6,
    # so eps must be passed explicitly for a like-for-like comparison.
    x = _randn(2, 5, 64, seed=5, dtype=dtype)
    ours = _random_norm(64, dtype=dtype)
    ref = torch.nn.RMSNorm(64, eps=EPS, dtype=dtype)
    with torch.no_grad():
        ref.weight.copy_(ours.weight)
    tol = {} if dtype == torch.float64 else {"rtol": 1e-6, "atol": 1e-6}
    assert_close(ours(x), ref(x), **tol)


def test_bfloat16_matches_float64_reference_within_bf16_tolerance() -> None:
    x64 = _randn(4, 8, 128, seed=6, dtype=torch.float64)
    norm64 = _random_norm(128, dtype=torch.float64)
    norm_bf16 = RMSNorm(128, dtype=torch.bfloat16)
    with torch.no_grad():
        norm_bf16.weight.copy_(norm64.weight.to(torch.bfloat16))
    x_bf16 = x64.to(torch.bfloat16)
    # Reference sees the SAME (already rounded) inputs and weights, in float64.
    expected = rms_norm_textbook(x_bf16.double(), norm_bf16.weight.double(), EPS)
    y = norm_bf16(x_bf16)
    assert y.dtype == torch.bfloat16
    # Two bf16 roundings (cast-back, then product): <= ~2 * 2**-8 relative.
    assert_close(y.double(), expected, rtol=2 * 2**-8, atol=1e-6)


def test_half_precision_statistics_are_computed_in_float32() -> None:
    # Values whose squares overflow float16 (max 65504) but not float32.
    x = torch.full((4,), 300.0, dtype=torch.float16)
    assert torch.isinf(x * x).all()  # naive fp16 math overflows ...
    y = RMSNorm(4, dtype=torch.float16)(x)
    assert torch.isfinite(y).all()  # ... ours upcasts to fp32 and stays finite
    assert_close(y, torch.ones(4, dtype=torch.float16))


def test_mixed_dtype_follows_type_promotion() -> None:
    norm = RMSNorm(8)  # float32 weight
    y = norm(_randn(2, 8, dtype=torch.bfloat16))
    assert y.dtype == torch.float32  # same rule as HF: weight * x.to(input_dtype)


# ------------------------------------------ 5 + 6. batch, seq, hidden sizes
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("seq_len", [1, 7, 33])
@pytest.mark.parametrize("hidden", [1, 2, 3, 32, 128, 1024])
def test_many_shapes_against_float64_reference(batch, seq_len, hidden) -> None:
    x = _randn(batch, seq_len, hidden, seed=batch * 100 + seq_len)
    norm = _random_norm(hidden, seed=hidden)
    expected = rms_norm_textbook(x.double(), norm.weight.double(), EPS)
    assert_close(norm(x).double(), expected, rtol=1e-5, atol=1e-6)


def test_hidden_size_one_is_sign_times_weight() -> None:
    x = torch.tensor([[-3.0], [0.5], [2.0]])
    y = RMSNorm(1)(x)
    expected = torch.sign(x) * (x.abs() / torch.sqrt(x * x + EPS))
    assert_close(y, expected)


# ------------------------------------------------- 7. scaling behaviour
@pytest.mark.parametrize("scale", [1e-2, 0.5, 3.0, 1e4])
def test_positive_scale_invariance(scale) -> None:
    x = _randn(2, 3, 64, seed=7, dtype=torch.float64)
    norm = _random_norm(64, dtype=torch.float64)
    ms = x.pow(2).mean(dim=-1, keepdim=True)
    # Exact relation: only eps breaks scale invariance.
    exact = norm(x) * torch.sqrt(ms + EPS) / torch.sqrt(ms + EPS / scale**2)
    assert_close(norm(scale * x), exact, rtol=1e-12, atol=1e-12)
    # Practical consequence: relative change is about eps / (2 * mean_square).
    bound = EPS * max(1.0, 1.0 / scale**2) / ms.min().item()
    assert_close(norm(scale * x), norm(x), rtol=bound, atol=1e-12)


def test_negative_scale_flips_sign() -> None:
    x = _randn(2, 64, seed=8, dtype=torch.float64)
    norm = _random_norm(64, dtype=torch.float64)
    assert torch.equal(norm(-2.0 * x), -norm(2.0 * x))  # odd function


def test_not_invariant_to_shift() -> None:
    # Unlike LayerNorm, RMSNorm does NOT subtract the mean.
    x = _randn(64, seed=9, dtype=torch.float64)
    norm = RMSNorm(64, dtype=torch.float64)
    assert not torch.allclose(norm(x + 5.0), norm(x))
    layer_norm = torch.nn.LayerNorm(64, eps=EPS, elementwise_affine=False, dtype=torch.float64)
    assert_close(layer_norm(x + 5.0), layer_norm(x))


def test_weight_scales_output_per_feature() -> None:
    x = _randn(5, 16, seed=10)
    norm = RMSNorm(16)
    base = norm(x)
    with torch.no_grad():
        norm.weight.mul_(torch.arange(1, 17, dtype=torch.float32))
    assert_close(norm(x), base * torch.arange(1, 17, dtype=torch.float32))


# -------------------------------------------- 8. stability near zero
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_zero_input_gives_zero_not_nan(dtype) -> None:
    y = RMSNorm(16, dtype=dtype)(torch.zeros(2, 16, dtype=dtype))
    assert torch.equal(y, torch.zeros_like(y))


def test_tiny_inputs_are_damped_by_eps() -> None:
    # For |x| << sqrt(eps)=1e-3, y ~ x / sqrt(eps): eps prevents blow-up of noise.
    x = torch.full((8,), 1e-8, dtype=torch.float64)
    y = RMSNorm(8, dtype=torch.float64)(x)
    assert_close(y, x / torch.sqrt(x * x + EPS))
    assert y.abs().max() < 1e-4


def test_eps_matters_only_for_small_mean_square() -> None:
    x = _randn(64, seed=11, dtype=torch.float64)
    a = RMSNorm(64, eps=1e-6, dtype=torch.float64)(x)
    b = RMSNorm(64, eps=1e-5, dtype=torch.float64)(x)
    assert (a - b).abs().max() < 1e-4  # ~ |y| * 9e-6 / (2 * mean_square)
    tiny = 1e-4 * x
    a = RMSNorm(64, eps=1e-6, dtype=torch.float64)(tiny)
    b = RMSNorm(64, eps=1e-5, dtype=torch.float64)(tiny)
    assert (a - b).abs().max() > 0.1


def test_gradients_finite_at_zero_input() -> None:
    x = torch.zeros(3, 8, requires_grad=True)
    RMSNorm(8)(x).sum().backward()
    assert torch.isfinite(x.grad).all()


# ------------------------------------------------ 9. input gradients
def _analytic_input_grad(x, gamma, upstream, eps):
    """dL/dx for y = gamma * x / r, r = sqrt(mean(x^2) + eps)."""
    d = x.shape[-1]
    r = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    g = upstream * gamma
    return g / r - x * (g * x).sum(dim=-1, keepdim=True) / (d * r**3)


def test_input_gradient_matches_closed_form() -> None:
    x = _randn(2, 3, 16, seed=12, dtype=torch.float64).requires_grad_()
    norm = _random_norm(16, dtype=torch.float64)
    upstream = _randn(2, 3, 16, seed=13, dtype=torch.float64)
    norm(x).backward(upstream)
    expected = _analytic_input_grad(x.detach(), norm.weight.detach(), upstream, EPS)
    assert_close(x.grad, expected, rtol=1e-12, atol=1e-12)


def test_gradient_is_orthogonal_to_input_when_eps_negligible() -> None:
    # Scale invariance => sum_i x_i * dL/dx_i = (eps-sized) ~ 0 for every row.
    x = _randn(4, 32, seed=14, dtype=torch.float64).requires_grad_()
    norm = _random_norm(32, dtype=torch.float64)
    norm(x).backward(_randn(4, 32, seed=15, dtype=torch.float64))
    assert (x.detach() * x.grad).sum(dim=-1).abs().max() < 1e-5


def test_gradcheck_float64() -> None:
    x = _randn(2, 3, 5, seed=16, dtype=torch.float64).requires_grad_()
    w = (1.0 + 0.3 * _randn(5, seed=17, dtype=torch.float64)).requires_grad_()
    assert torch.autograd.gradcheck(lambda a, b: rms_norm(a, b, EPS), (x, w))
    assert torch.autograd.gradgradcheck(lambda a, b: rms_norm(a, b, EPS), (x, w))


# ----------------------------------------------- 10. weight gradients
def test_weight_gradient_is_sum_of_upstream_times_normalised_x() -> None:
    x = _randn(2, 3, 16, seed=18, dtype=torch.float64)
    norm = _random_norm(16, dtype=torch.float64)
    upstream = _randn(2, 3, 16, seed=19, dtype=torch.float64)
    norm(x).backward(upstream)
    x_hat = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    assert norm.weight.grad.shape == (16,)
    assert_close(norm.weight.grad, (upstream * x_hat).sum(dim=(0, 1)), rtol=1e-12, atol=1e-12)


def test_weight_receives_gradient_in_bfloat16() -> None:
    norm = RMSNorm(16, dtype=torch.bfloat16)
    norm(_randn(4, 16, dtype=torch.bfloat16)).float().pow(2).sum().backward()
    assert norm.weight.grad is not None
    assert norm.weight.grad.dtype == torch.bfloat16
    assert torch.isfinite(norm.weight.grad.float()).all()


# ---------------------------------------------------- 11. determinism
def test_forward_is_deterministic() -> None:
    x = _randn(3, 9, 64, seed=20)
    norm = _random_norm(64)
    assert torch.equal(norm(x), norm(x))
    assert torch.equal(norm(x), _random_norm(64)(x))  # same seed -> identical layer


def test_backward_is_deterministic() -> None:
    grads = []
    for _ in range(2):
        x = _randn(3, 9, 64, seed=21).requires_grad_()
        norm = _random_norm(64)
        norm(x).sum().backward()
        grads.append((x.grad.clone(), norm.weight.grad.clone()))
    assert torch.equal(grads[0][0], grads[1][0])
    assert torch.equal(grads[0][1], grads[1][1])


# -------------------------------------------------- 12. invalid inputs
@pytest.mark.parametrize("hidden_size", [0, -4, 2.5, True, "8", None])
def test_invalid_hidden_size(hidden_size) -> None:
    with pytest.raises(ValueError):
        RMSNorm(hidden_size)


@pytest.mark.parametrize("eps", [0.0, -1e-6, float("nan"), float("inf"), True, "1e-6", None])
def test_invalid_eps(eps) -> None:
    with pytest.raises(ValueError):
        RMSNorm(8, eps=eps)


def test_integer_weight_dtype_rejected() -> None:
    with pytest.raises(ValueError):
        RMSNorm(8, dtype=torch.int64)


def test_wrong_last_dimension_rejected() -> None:
    with pytest.raises(ValueError, match="last dimension"):
        RMSNorm(8)(torch.randn(2, 7))


def test_hidden_dim_not_last_rejected() -> None:
    with pytest.raises(ValueError):
        RMSNorm(8)(torch.randn(8, 3))  # (d, T) layout is a common mistake


def test_scalar_input_rejected() -> None:
    with pytest.raises(ValueError, match="at least one dimension"):
        RMSNorm(1)(torch.tensor(1.0))


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
def test_non_float_input_rejected(dtype) -> None:
    with pytest.raises(TypeError):
        RMSNorm(4)(torch.ones(2, 4, dtype=dtype))


def test_non_tensor_input_rejected() -> None:
    with pytest.raises(TypeError):
        RMSNorm(4)([1.0, 2.0, 3.0, 4.0])


def test_functional_rejects_2d_weight() -> None:
    with pytest.raises(ValueError, match="1-D"):
        rms_norm(torch.randn(2, 4), torch.ones(1, 4), EPS)
