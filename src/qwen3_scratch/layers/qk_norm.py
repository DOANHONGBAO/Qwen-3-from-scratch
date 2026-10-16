"""Query-Key normalisation (QK-Norm) as used in Qwen3 attention.

After the Q and K projections are split into heads, every *head vector* (length
``head_dim``) is RMS-normalised on its own, with one learnable scale vector for
queries (``q_norm.weight``) and another for keys (``k_norm.weight``)::

    q[b, h, t, :] <- RMSNorm_q(q[b, h, t, :])     weight shape (head_dim,)
    k[b, j, t, :] <- RMSNorm_k(k[b, j, t, :])     weight shape (head_dim,)

* Each scale vector is shared by all tokens and all heads of its projection.
* Values are NOT normalised.
* Order inside Qwen3 attention: projection -> split heads -> QK-Norm -> RoPE.

No new normalisation kernel is introduced: this module reuses the verified
:class:`~qwen3_scratch.layers.rmsnorm.RMSNorm` (M1.1), which already reduces over
the last dimension of any ``(..., head_dim)`` tensor in HF precision order.

Design note: :class:`~qwen3_scratch.layers.attention.GroupedQueryAttention` stores
``q_norm`` / ``k_norm`` directly as attributes (the HF checkpoint names
``self_attn.q_norm.weight`` / ``self_attn.k_norm.weight``) and calls
:func:`apply_qk_norm`. :class:`QKNorm` packages the same pair as a standalone
module for teaching and unit tests; both paths run the same function.
"""

from __future__ import annotations

import torch
from torch import nn

from qwen3_scratch.layers.rmsnorm import RMSNorm

__all__ = ["QKNorm", "apply_qk_norm"]


def _check_heads(name: str, x: torch.Tensor, head_dim: int) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if x.dim() != 4:
        raise ValueError(f"{name} must have shape (B, N, T, D), got {tuple(x.shape)}")
    if x.shape[-1] != head_dim:
        raise ValueError(f"{name} head_dim is {x.shape[-1]}, expected {head_dim}")


def apply_qk_norm(
    q: torch.Tensor, k: torch.Tensor, q_norm: RMSNorm, k_norm: RMSNorm
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalise every query head vector with ``q_norm`` and every key head with ``k_norm``.

    ``q``: ``(B, Nq, T, D)``, ``k``: ``(B, Nkv, T, D)``; ``Nq`` and ``Nkv`` may differ.
    Shapes are unchanged; no reshape is needed because RMSNorm reduces over ``D`` only.
    """
    if q_norm.hidden_size != k_norm.hidden_size:
        raise ValueError("q_norm and k_norm must have the same size (head_dim)")
    head_dim = q_norm.hidden_size
    _check_heads("q", q, head_dim)
    _check_heads("k", k, head_dim)
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(
            f"q {tuple(q.shape)} and k {tuple(k.shape)} must agree on batch and sequence length"
        )
    return q_norm(q), k_norm(k)


class QKNorm(nn.Module):
    """A pair of RMSNorms over ``head_dim``: one for queries, one for keys.

    Args:
        head_dim: size ``D`` of each attention head (Qwen3-0.6B: 128).
        eps: RMSNorm epsilon (Qwen3: ``rms_norm_eps = 1e-6``).
        device, dtype: allocation of the two ``(D,)`` weight vectors (initialised to ones).
    """

    def __init__(
        self,
        head_dim: int,
        eps: float = 1e-6,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.q_norm = RMSNorm(head_dim, eps, device=device, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, eps, device=device, dtype=dtype)

    @property
    def head_dim(self) -> int:
        return self.q_norm.hidden_size

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return apply_qk_norm(q, k, self.q_norm, self.k_norm)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, eps={self.q_norm.eps}"
