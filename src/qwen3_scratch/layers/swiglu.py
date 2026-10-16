"""SwiGLU feed-forward network (the MLP of every Qwen3 decoder block).

Math, for one token vector ``x`` of size ``H`` (``hidden_size``) and
``I = intermediate_size``::

    gate   = x W_gate^T          (I,)   gate_proj: H -> I, no bias
    up     = x W_up^T            (I,)   up_proj:   H -> I, no bias
    hidden = SiLU(gate) * up     (I,)   elementwise ("gated linear unit")
    out    = hidden W_down^T     (H,)   down_proj: I -> H, no bias

    SiLU(z) = z * sigmoid(z)     (also called "swish")

Shapes for a batch: ``x: (B, T, H) -> gate, up, hidden: (B, T, I) -> out: (B, T, H)``.
Any number of leading dimensions is accepted; only the last one is ``H``.

Qwen3 specifics (verified against HF ``Qwen3MLP``): the three projections are named
``gate_proj``, ``up_proj``, ``down_proj``, have **no bias** (hardcoded ``bias=False``
in HF, independent of ``attention_bias``), and the activation is SiLU
(``hidden_act = "silu"``). For Qwen3-0.6B, ``H = 1024`` and ``I = 3072``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from qwen3_scratch.config import Qwen3Config

__all__ = ["SwiGLUFeedForward", "silu_textbook", "swiglu_textbook"]


def silu_textbook(z: torch.Tensor) -> torch.Tensor:
    """``SiLU(z) = z * sigmoid(z) = z / (1 + exp(-z))``, written out for teaching."""
    return z * torch.sigmoid(z)


def swiglu_textbook(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU with explicit weight matrices in ``nn.Linear`` layout ``(out, in)``.

    ``w_gate, w_up: (I, H)``, ``w_down: (H, I)``. Educational/reference version
    with no ``nn.Module`` machinery.
    """
    gate = x @ w_gate.T  # (..., I)
    up = x @ w_up.T  # (..., I)
    return (silu_textbook(gate) * up) @ w_down.T  # (..., H)


def _check_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


class SwiGLUFeedForward(nn.Module):
    """Qwen3-style gated feed-forward block.

    Args:
        hidden_size: model width ``H`` (input and output size).
        intermediate_size: width ``I`` of the gated hidden layer.
        bias: add biases to the three projections. Qwen3 uses ``False``; the flag
            exists only to make the convention explicit and testable.
        device, dtype: forwarded to the ``nn.Linear`` layers.

    Parameters (names match the HF checkpoint ``model.layers.N.mlp.*``):
        gate_proj.weight: ``(I, H)``
        up_proj.weight:   ``(I, H)``
        down_proj.weight: ``(H, I)``

    Weights use PyTorch's default ``nn.Linear`` initialisation; pretrained
    weights are loaded in M2, so the initial values do not matter for parity.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        bias: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = _check_positive_int(hidden_size, "hidden_size")
        self.intermediate_size = _check_positive_int(intermediate_size, "intermediate_size")
        if not isinstance(bias, bool):
            raise ValueError(f"bias must be a bool, got {bias!r}")
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError(f"dtype must be a floating-point dtype, got {dtype}")
        factory = {"device": device, "dtype": dtype}
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias, **factory)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias, **factory)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias, **factory)

    @classmethod
    def from_config(cls, config: Qwen3Config, **kwargs) -> SwiGLUFeedForward:
        """Build with Qwen3 sizes. Qwen3's MLP never has biases."""
        return cls(config.hidden_size, config.intermediate_size, bias=False, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if x.dim() < 1 or x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected input (..., {self.hidden_size}), got shape {tuple(x.shape)}"
            )
        gate = self.gate_proj(x)  # (..., I)  how much of each hidden unit to let through
        up = self.up_proj(x)  # (..., I)  the candidate values
        hidden = F.silu(gate) * up  # (..., I)  smooth gate times value
        return self.down_proj(hidden)  # (..., H)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}"
