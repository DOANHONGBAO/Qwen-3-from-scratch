"""Parity of our RMSNorm with the official Hugging Face ``Qwen3RMSNorm``.

Requires the optional ``hf`` extra (``pip install -e ".[hf]"``); skipped otherwise.
No checkpoint is downloaded: ``Qwen3RMSNorm`` is constructed directly with
random weights, which is enough to compare the operation itself.
"""

from __future__ import annotations

import pytest
import torch

transformers_qwen3 = pytest.importorskip(
    "transformers.models.qwen3.modeling_qwen3", reason="transformers not installed (hf extra)"
)
Qwen3RMSNorm = transformers_qwen3.Qwen3RMSNorm

from qwen3_scratch.layers import RMSNorm  # noqa: E402

EPS = 1e-6


def _pair(d: int, dtype: torch.dtype, seed: int = 0) -> tuple[RMSNorm, torch.nn.Module]:
    ours = RMSNorm(d, eps=EPS, dtype=dtype)
    ref = Qwen3RMSNorm(d, eps=EPS).to(dtype)
    gen = torch.Generator().manual_seed(seed)
    w = (1.0 + 0.5 * torch.randn(d, generator=gen)).to(dtype)
    with torch.no_grad():
        ours.weight.copy_(w)
        ref.weight.copy_(w)
    return ours, ref


def test_default_initialisation_and_state_dict_keys_match() -> None:
    ours, ref = RMSNorm(16), Qwen3RMSNorm(16)
    assert ours.state_dict().keys() == ref.state_dict().keys() == {"weight"}
    assert torch.equal(ours.weight, ref.weight)  # both ones, no "1 + w" offset
    assert ref.variance_epsilon == ours.eps == EPS


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(2, 7, 64), (2, 4, 7, 128), (1, 1024)])
def test_forward_bitwise_identical_to_hf(dtype, shape) -> None:
    ours, ref = _pair(shape[-1], dtype)
    gen = torch.Generator().manual_seed(1)
    x = (3.0 * torch.randn(*shape, generator=gen)).to(dtype)
    y_ours, y_ref = ours(x), ref(x)
    assert y_ours.dtype == y_ref.dtype == dtype
    # Same operations in the same order on the same backend -> identical bits.
    assert torch.equal(y_ours, y_ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backward_identical_to_hf(dtype) -> None:
    ours, ref = _pair(64, dtype)
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(3, 5, 64, generator=gen).to(dtype)
    upstream = torch.randn(3, 5, 64, generator=gen).to(dtype)
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    ours(xa).backward(upstream)
    ref(xb).backward(upstream)
    assert torch.equal(xa.grad, xb.grad)
    assert torch.equal(ours.weight.grad, ref.weight.grad)


def test_documented_float64_difference() -> None:
    # HF always computes statistics in float32 (downcasting float64 inputs);
    # ours keeps float64. Outputs agree to float32 precision only.
    ours, ref = _pair(64, torch.float64)
    x = torch.randn(4, 64, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    torch.testing.assert_close(ours(x), ref(x), rtol=1e-6, atol=1e-6)
    assert not torch.equal(ours(x), ref(x))
