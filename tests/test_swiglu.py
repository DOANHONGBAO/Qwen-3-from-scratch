"""Unit tests for the SwiGLU feed-forward block (M1.2). CPU tier."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from qwen3_scratch.config import QWEN3_0_6B_BASE, TINY_QWEN3
from qwen3_scratch.layers import SwiGLUFeedForward, silu_textbook, swiglu_textbook

# Manual example (docs/components/swiglu.md §3): H = 2, I = 3.
X = [1.0, -1.0]
W_GATE = [[1.0, 0.0], [1.0, 1.0], [0.0, -2.0]]  # gate = [1, 0, 2]
W_UP = [[2.0, 0.0], [0.0, 1.0], [1.0, 0.0]]  # up   = [2, -1, 1]
W_DOWN = [[1.0, 1.0, 1.0], [0.5, -1.0, 0.0]]
SIG1 = 1.0 / (1.0 + math.exp(-1.0))
SILU2 = 2.0 / (1.0 + math.exp(-2.0))
EXPECTED = [2.0 * SIG1 + SILU2, 0.5 * 2.0 * SIG1]  # [3.22371131..., 0.73105857...]


def _randn(*shape: int, seed: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float64).to(dtype)


def _mlp(h: int, i: int, seed: int = 1, dtype: torch.dtype = torch.float32) -> SwiGLUFeedForward:
    torch.manual_seed(seed)
    return SwiGLUFeedForward(h, i, dtype=dtype)


def _manual_module(dtype: torch.dtype = torch.float64) -> SwiGLUFeedForward:
    mlp = SwiGLUFeedForward(2, 3, dtype=dtype)
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(torch.tensor(W_GATE))
        mlp.up_proj.weight.copy_(torch.tensor(W_UP))
        mlp.down_proj.weight.copy_(torch.tensor(W_DOWN))
    return mlp


# ------------------------------------------------------------ structure
def test_parameter_names_shapes_and_no_bias() -> None:
    mlp = SwiGLUFeedForward(8, 24)
    shapes = {name: tuple(p.shape) for name, p in mlp.named_parameters()}
    assert shapes == {
        "gate_proj.weight": (24, 8),
        "up_proj.weight": (24, 8),
        "down_proj.weight": (8, 24),
    }
    assert "hidden_size=8, intermediate_size=24" in repr(mlp)


def test_bias_flag_adds_three_biases() -> None:
    mlp = SwiGLUFeedForward(8, 24, bias=True)
    biases = {n: tuple(p.shape) for n, p in mlp.named_parameters() if n.endswith("bias")}
    assert biases == {"gate_proj.bias": (24,), "up_proj.bias": (24,), "down_proj.bias": (8,)}


@pytest.mark.parametrize("cfg", [TINY_QWEN3, QWEN3_0_6B_BASE], ids=["tiny", "0.6b"])
def test_from_config_parameter_count(cfg) -> None:
    mlp = SwiGLUFeedForward.from_config(cfg, device="meta")  # meta: no memory allocated
    n_params = sum(p.numel() for p in mlp.parameters())
    assert n_params == 3 * cfg.hidden_size * cfg.intermediate_size
    if cfg is QWEN3_0_6B_BASE:
        assert n_params == 9_437_184  # per decoder layer


# -------------------------------------------------------- 1, 2, 3. shapes
@pytest.mark.parametrize("lead", [(), (1,), (3,), (2, 5), (4, 1), (2, 3, 7), (0, 4)])
@pytest.mark.parametrize(("h", "i"), [(1, 1), (4, 6), (16, 48), (64, 192), (24, 7)])
def test_output_shape(lead, h, i) -> None:
    x = _randn(*lead, h)
    y = _mlp(h, i)(x)
    assert y.shape == x.shape
    assert y.dtype == torch.float32


def test_intermediate_shapes() -> None:
    mlp = _mlp(16, 48)
    x = _randn(2, 5, 16)
    assert mlp.gate_proj(x).shape == (2, 5, 48)
    assert mlp.up_proj(x).shape == (2, 5, 48)
    assert mlp.down_proj(torch.zeros(2, 5, 48)).shape == (2, 5, 16)


# ---------------------------------------------------- 4. manual example
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_manual_example(dtype) -> None:
    y = _manual_module(dtype)(torch.tensor(X, dtype=dtype))
    tol = {"rtol": 1e-6, "atol": 1e-7} if dtype == torch.float32 else {"rtol": 1e-15, "atol": 0.0}
    assert_close(y, torch.tensor(EXPECTED, dtype=dtype), **tol)


def test_manual_example_intermediates() -> None:
    mlp = _manual_module()
    x = torch.tensor(X, dtype=torch.float64)
    gate, up = mlp.gate_proj(x), mlp.up_proj(x)
    assert gate.tolist() == [1.0, 0.0, 2.0]
    assert up.tolist() == [2.0, -1.0, 1.0]
    hidden = F.silu(gate) * up
    assert_close(hidden, torch.tensor([2 * SIG1, 0.0, SILU2], dtype=torch.float64))


# --------------------------------------------- 5. reference equivalence
def test_silu_matches_definition() -> None:
    z = torch.linspace(-20, 20, 2001, dtype=torch.float64)
    assert_close(F.silu(z), z / (1 + torch.exp(-z)), rtol=1e-15, atol=1e-15)
    assert_close(F.silu(z), silu_textbook(z), rtol=1e-15, atol=1e-15)
    assert F.silu(torch.tensor(0.0)).item() == 0.0
    # SiLU is not monotone: its minimum is about -0.2785 at z ~ -1.2785.
    assert abs(F.silu(z).min().item() + 0.278464542761074) < 1e-4


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_matches_textbook_formula(dtype) -> None:
    mlp = _mlp(32, 96, dtype=dtype)
    x = _randn(3, 7, 32, seed=2, dtype=dtype)
    ref = swiglu_textbook(x, mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight)
    tol = {} if dtype == torch.float64 else {"rtol": 1e-5, "atol": 1e-6}
    assert_close(mlp(x), ref, **tol)


def test_float32_matches_float64_reference() -> None:
    mlp = _mlp(64, 192)
    x = _randn(2, 9, 64, seed=3)
    ref = swiglu_textbook(
        x.double(),
        mlp.gate_proj.weight.double(),
        mlp.up_proj.weight.double(),
        mlp.down_proj.weight.double(),
    )
    assert_close(mlp(x).double(), ref, rtol=1e-5, atol=1e-6)


def test_bias_is_applied_like_linear_layers() -> None:
    torch.manual_seed(4)
    mlp = SwiGLUFeedForward(8, 12, bias=True, dtype=torch.float64)
    x = _randn(5, 8, seed=5, dtype=torch.float64)
    gate = x @ mlp.gate_proj.weight.T + mlp.gate_proj.bias
    up = x @ mlp.up_proj.weight.T + mlp.up_proj.bias
    ref = (F.silu(gate) * up) @ mlp.down_proj.weight.T + mlp.down_proj.bias
    assert_close(mlp(x), ref)


def test_tokens_are_processed_independently() -> None:
    mlp = _mlp(16, 48)
    x = _randn(2, 6, 16, seed=6)
    full = mlp(x)
    # Batched and per-token matmuls may use different kernels -> tolerance, not equality.
    for b in range(2):
        for t in range(6):
            assert_close(mlp(x[b, t]), full[b, t], rtol=1e-5, atol=1e-6)


def test_gate_closed_means_zero_output_without_bias() -> None:
    mlp = _mlp(8, 16)
    with torch.no_grad():
        mlp.gate_proj.weight.zero_()  # gate = 0 -> SiLU(0) = 0 -> nothing passes
    assert torch.equal(mlp(_randn(3, 8)), torch.zeros(3, 8))


def test_not_an_ordinary_mlp() -> None:
    # SwiGLU is quadratic in x for large inputs: f(2x) != 2 f(x) and != f(x).
    mlp = _mlp(8, 16, dtype=torch.float64)
    x = 10 * _randn(8, seed=7, dtype=torch.float64)
    assert not torch.allclose(mlp(2 * x), 2 * mlp(x))
    # Positively-gated units grow ~4x when x doubles (SiLU(z) ~ z for large z).
    gate = mlp.gate_proj(x)
    open_units = gate > 5
    if open_units.any():
        h1 = (F.silu(gate) * mlp.up_proj(x))[open_units]
        h2 = (F.silu(2 * gate) * mlp.up_proj(2 * x))[open_units]
        assert_close(h2, 4 * h1, rtol=1e-3, atol=1e-6)


# ------------------------------------------------ 6, 7. gradients
def test_gradients_reach_all_parameters_and_input() -> None:
    mlp = _mlp(16, 48)
    x = _randn(2, 3, 16, seed=8).requires_grad_()
    mlp(x).pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for name, p in mlp.named_parameters():
        assert p.grad is not None, name
        assert p.grad.shape == p.shape
        assert p.grad.abs().sum() > 0, name


def _silu_grad(z: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(z)
    return s * (1 + z * (1 - s))


def test_gradients_match_closed_form() -> None:
    mlp = _mlp(8, 20, dtype=torch.float64)
    x = _randn(4, 8, seed=9, dtype=torch.float64).requires_grad_()
    u = _randn(4, 8, seed=10, dtype=torch.float64)  # upstream dL/dout
    mlp(x).backward(u)

    xd = x.detach()
    wg, wu, wd = (mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight)
    gate, up = xd @ wg.T, xd @ wu.T
    hidden = F.silu(gate) * up
    d_hidden = u @ wd  # (4, I)
    d_gate = d_hidden * up * _silu_grad(gate)  # gate branch
    d_up = d_hidden * F.silu(gate)  # value branch
    with torch.no_grad():
        assert_close(mlp.down_proj.weight.grad, u.T @ hidden)
        assert_close(mlp.gate_proj.weight.grad, d_gate.T @ xd)
        assert_close(mlp.up_proj.weight.grad, d_up.T @ xd)
        assert_close(x.grad, d_gate @ wg + d_up @ wu)


def test_gradcheck_float64() -> None:
    mlp = _mlp(4, 6, dtype=torch.float64)
    x = _randn(2, 3, 4, seed=11, dtype=torch.float64).requires_grad_()
    assert torch.autograd.gradcheck(mlp, (x,))
    assert torch.autograd.gradgradcheck(mlp, (x,))
    params = tuple(p.detach().clone().requires_grad_() for p in mlp.parameters())

    def as_function(wg, wu, wd):
        return swiglu_textbook(x.detach(), wg, wu, wd)

    assert torch.autograd.gradcheck(as_function, params)


# ------------------------------------------------------ 8. invalid input
@pytest.mark.parametrize(
    ("h", "i"), [(0, 4), (4, 0), (-1, 4), (4, -3), (2.0, 4), (4, "8"), (True, 4), (4, None)]
)
def test_invalid_sizes(h, i) -> None:
    with pytest.raises(ValueError):
        SwiGLUFeedForward(h, i)


def test_invalid_bias_and_dtype() -> None:
    with pytest.raises(ValueError):
        SwiGLUFeedForward(4, 8, bias=1)
    with pytest.raises(ValueError):
        SwiGLUFeedForward(4, 8, dtype=torch.int32)


@pytest.mark.parametrize("shape", [(2, 7), (8, 3), (3, 4, 5)])
def test_wrong_input_width_rejected(shape) -> None:
    with pytest.raises(ValueError, match="expected input"):
        SwiGLUFeedForward(8, 16)(torch.randn(*shape))


def test_scalar_and_non_tensor_rejected() -> None:
    mlp = SwiGLUFeedForward(1, 4)
    with pytest.raises(ValueError):
        mlp(torch.tensor(1.0))
    with pytest.raises(TypeError):
        mlp([1.0])


# ------------------------------------------------------ 9. determinism
def test_deterministic_forward_and_init() -> None:
    x = _randn(3, 5, 32, seed=12)
    a, b = _mlp(32, 64, seed=13), _mlp(32, 64, seed=13)
    assert torch.equal(a(x), a(x))
    assert torch.equal(a(x), b(x))
    assert not torch.equal(a(x), _mlp(32, 64, seed=14)(x))


def test_deterministic_backward() -> None:
    grads = []
    for _ in range(2):
        mlp = _mlp(16, 32, seed=15)
        x = _randn(4, 16, seed=16).requires_grad_()
        mlp(x).sum().backward()
        grads.append([x.grad.clone()] + [p.grad.clone() for p in mlp.parameters()])
    for g0, g1 in zip(*grads, strict=True):
        assert torch.equal(g0, g1)


# ------------------------------------------------- 10. dtypes on CPU
def test_float32_cpu_execution() -> None:
    mlp = SwiGLUFeedForward.from_config(TINY_QWEN3)
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in mlp.parameters())
    y = mlp(_randn(2, 4, TINY_QWEN3.hidden_size))
    assert y.device.type == "cpu" and y.dtype == torch.float32 and torch.isfinite(y).all()


def test_bfloat16_close_to_float64() -> None:
    mlp64 = _mlp(64, 192, dtype=torch.float64)
    mlp16 = SwiGLUFeedForward(64, 192, dtype=torch.bfloat16)
    mlp16.load_state_dict({k: v.to(torch.bfloat16) for k, v in mlp64.state_dict().items()})
    x = _randn(4, 64, seed=17, dtype=torch.bfloat16)
    ref = swiglu_textbook(
        x.double(),
        *(
            mlp16.get_parameter(n).double()
            for n in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")
        ),
    )
    y = mlp16(x)
    assert y.dtype == torch.bfloat16
    # Several bf16 roundings (two matmuls + elementwise): loose, documented bound.
    assert (y.double() - ref).abs().max() < 0.05 * ref.abs().max()
