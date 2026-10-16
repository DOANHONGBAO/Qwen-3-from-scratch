"""Parity of SwiGLUFeedForward with the official Hugging Face ``Qwen3MLP``.

Requires the optional ``hf`` extra; skipped otherwise. No checkpoint download:
``Qwen3MLP`` is built from a small ``Qwen3Config`` and its random weights are
copied into our module through ``load_state_dict`` (identical key names).
"""

from __future__ import annotations

import pytest
import torch

modeling = pytest.importorskip(
    "transformers.models.qwen3.modeling_qwen3", reason="transformers not installed (hf extra)"
)
transformers = pytest.importorskip("transformers")

from qwen3_scratch.config import TINY_QWEN3  # noqa: E402
from qwen3_scratch.layers import SwiGLUFeedForward  # noqa: E402


def _hf_config(hidden: int, inter: int):
    return transformers.Qwen3Config(
        hidden_size=hidden,
        intermediate_size=inter,
        hidden_act="silu",
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=1,
        vocab_size=32,
    )


def _pair(hidden: int, inter: int, dtype: torch.dtype, seed: int = 0):
    torch.manual_seed(seed)
    ref = modeling.Qwen3MLP(_hf_config(hidden, inter)).to(dtype)
    ours = SwiGLUFeedForward(hidden, inter, dtype=dtype)
    missing, unexpected = ours.load_state_dict(ref.state_dict(), strict=True)
    assert not missing and not unexpected
    return ours, ref


def test_state_dict_keys_and_shapes_identical() -> None:
    ours, ref = _pair(TINY_QWEN3.hidden_size, TINY_QWEN3.intermediate_size, torch.float32)
    assert {k: v.shape for k, v in ours.state_dict().items()} == {
        k: v.shape for k, v in ref.state_dict().items()
    }
    assert all(not hasattr(m, "bias") or m.bias is None for m in ref.children())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("hidden", "inter", "shape"), [(64, 192, (2, 7)), (32, 80, (1, 1)), (128, 384, (3, 5))]
)
def test_forward_bitwise_identical(dtype, hidden, inter, shape) -> None:
    ours, ref = _pair(hidden, inter, dtype)
    x = torch.randn(*shape, hidden, generator=torch.Generator().manual_seed(1)).to(dtype)
    with torch.no_grad():
        assert torch.equal(ours(x), ref(x))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backward_bitwise_identical(dtype) -> None:
    ours, ref = _pair(64, 192, dtype)
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(2, 5, 64, generator=gen).to(dtype)
    upstream = torch.randn(2, 5, 64, generator=gen).to(dtype)
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    ours(xa).backward(upstream)
    ref(xb).backward(upstream)
    assert torch.equal(xa.grad, xb.grad)
    for (name, p_ours), (_, p_ref) in zip(
        ours.named_parameters(), ref.named_parameters(), strict=True
    ):
        assert torch.equal(p_ours.grad, p_ref.grad), name
