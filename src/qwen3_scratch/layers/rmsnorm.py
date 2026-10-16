"""Root Mean Square Layer Normalization (RMSNorm) as used by Qwen3.

Math (for one vector ``x`` of length ``d`` and learnable scale ``gamma``)::

    rms(x) = sqrt( (1/d) * sum_i x_i**2 + eps )
    y_i    = x_i / rms(x) * gamma_i

Shapes:
    input  x      : (..., d)   any number of leading (batch / sequence / head) dims
    weight gamma  : (d,)       initialised to ones
    output y      : (..., d)   same shape as x

The reduction is over the LAST dimension only; every leading position is
normalised independently.

Two functions are provided on purpose:

* :func:`rms_norm_textbook` - the formula above, evaluated in the input dtype with
  no casts. Educational; used as a mathematical reference in tests.
* :func:`rms_norm` - the Qwen3 / Hugging Face ``Qwen3RMSNorm`` compatible
  version. It differs only in *precision handling* (see docs/components/rmsnorm.md):

  1. statistics are computed in float32 even for bfloat16/float16 inputs,
  2. the normalised activations are cast back to the input dtype,
  3. only then multiplied by ``weight`` (no ``1 + weight`` offset in Qwen3).

:class:`RMSNorm` is the ``nn.Module`` used by the model; it calls :func:`rms_norm`.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["RMSNorm", "rms_norm", "rms_norm_textbook"]


def _check_input(x: torch.Tensor, weight: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if not torch.is_floating_point(x):
        raise TypeError(f"x must be a floating-point tensor, got dtype {x.dtype}")
    if x.dim() < 1:
        raise ValueError("x must have at least one dimension (the hidden dimension)")
    if weight.dim() != 1:
        raise ValueError(f"weight must be 1-D, got shape {tuple(weight.shape)}")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"last dimension of x ({x.shape[-1]}) must equal hidden_size ({weight.shape[0]})"
        )


def rms_norm_textbook(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm exactly as written in the formula, in ``x``'s own dtype.

    Use with float64 inputs as a high-precision mathematical reference.
    """
    _check_input(x, weight)
    mean_square = (x * x).mean(dim=-1, keepdim=True)  # (..., 1)
    rms = torch.sqrt(mean_square + eps)  # (..., 1)
    return x / rms * weight  # (..., d)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3-compatible RMSNorm (same operation order as HF ``Qwen3RMSNorm``).

    Statistics are computed in ``float32`` for half-precision inputs. Unlike HF,
    a ``float64`` input stays in ``float64`` (HF would silently downcast it to
    float32); this only matters for tests such as ``gradcheck`` and never for the
    model, which runs in float32 or bfloat16.

    The output dtype is ``torch.promote_types(weight.dtype, x.dtype)`` - identical
    to ``x.dtype`` when weight and input share a dtype, as they do in the model.
    """
    _check_input(x, weight)
    input_dtype = x.dtype
    compute_dtype = torch.promote_types(input_dtype, torch.float32)  # fp16/bf16 -> fp32
    h = x.to(compute_dtype)
    variance = h.pow(2).mean(dim=-1, keepdim=True)  # mean of squares, (..., 1)
    h = h * torch.rsqrt(variance + eps)  # x / rms(x)
    return weight * h.to(input_dtype)  # cast back BEFORE scaling, like HF


class RMSNorm(nn.Module):
    """RMSNorm over the last dimension with a learnable per-feature scale.

    Args:
        hidden_size: size ``d`` of the normalised (last) dimension.
        eps: small constant added to the mean square for numerical stability
            (Qwen3: ``rms_norm_eps = 1e-6``).
        device, dtype: where/how to allocate ``weight``.

    Attributes:
        weight: ``nn.Parameter`` of shape ``(hidden_size,)``, initialised to ones,
            so a freshly created layer is a pure normalisation. The name matches
            the HF checkpoint keys (e.g. ``model.norm.weight``).
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(f"hidden_size must be a positive int, got {hidden_size!r}")
        if isinstance(eps, bool) or not isinstance(eps, int | float):
            raise ValueError(f"eps must be a positive finite float, got {eps!r}")
        if not (math.isfinite(eps) and eps > 0):
            raise ValueError(f"eps must be a positive finite float, got {eps!r}")
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError(f"dtype must be a floating-point dtype, got {dtype}")
        self.hidden_size = hidden_size
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise the scale to ones (Qwen3/HF convention; no ``1 + w`` offset)."""
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}"
