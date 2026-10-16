"""Rotary Position Embedding (RoPE) as used by Qwen3.

RoPE encodes the position ``p`` of a token by *rotating* its query and key
vectors. A head vector of size ``D`` (``head_dim``, must be even) is split into
``D/2`` two-dimensional pairs; pair ``i`` is rotated by the angle ``p * w_i``::

    w_i = theta ** (-2i / D),   i = 0 .. D/2 - 1          (inverse frequencies)
    [y_a]   [cos(p w_i)  -sin(p w_i)] [x_a]
    [y_b] = [sin(p w_i)   cos(p w_i)] [x_b]

Pairing convention (IMPORTANT for checkpoint compatibility): Qwen3 / Hugging Face
use the **split-half** layout - pair ``i`` is ``(x[i], x[i + D/2])`` - implemented
with ``rotate_half``::

    y = x * cos + rotate_half(x) * sin,   rotate_half([x1, x2]) = [-x2, x1]
    cos = [cos(p w_0..p w_{D/2-1}), cos(p w_0..p w_{D/2-1})]   (angles repeated)

The original RoFormer paper pairs *adjacent* features ``(x[2i], x[2i+1])``
("interleaved"). Both are the same mathematics up to a fixed permutation of the
D features, but they are NOT interchangeable with given weights: using the
interleaved form with Qwen3's ``q_proj``/``k_proj`` weights gives wrong attention
scores unless those weight rows are permuted too. ``apply_rope_interleaved`` is
provided only to demonstrate this.

Layouts and broadcasting (B batch, N heads, T tokens, D head_dim):

* ``position_ids``: integer ``(T,)`` (shared by the batch) or ``(B, T)``.
* ``cos``, ``sin``: ``(T, D)`` or ``(B, T, D)`` - one row per token, no head axis.
* ``q``: ``(B, Nq, T, D)``, ``k``: ``(B, Nkv, T, D)``; ``Nq`` and ``Nkv`` may differ
  (grouped-query attention) because ``cos``/``sin`` broadcast over the head axis
  as ``(1 or B, 1, T, D)``.

Precision (follows HF ``Qwen3RotaryEmbedding``): inverse frequencies and angles in
float32, cos/sin computed in float32, then cast to the activation dtype *before*
the rotation. RoPE is applied to queries and keys only, never to values.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from qwen3_scratch.config import Qwen3Config

__all__ = [
    "RotaryEmbedding",
    "apply_rope",
    "apply_rope_interleaved",
    "apply_rope_to_qk",
    "compute_inv_freq",
    "half_layout_permutation",
    "make_position_ids",
    "rope_cos_sin",
    "rotate_2d",
    "rotate_half",
]


# ------------------------------------------------------------------ validation
def _check_head_dim(head_dim: object) -> int:
    if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
        raise ValueError(f"head_dim must be a positive int, got {head_dim!r}")
    if head_dim % 2:
        raise ValueError(f"head_dim must be even for RoPE (pairs of features), got {head_dim}")
    return head_dim


def _check_theta(rope_theta: object) -> float:
    if isinstance(rope_theta, bool) or not isinstance(rope_theta, int | float):
        raise ValueError(f"rope_theta must be a positive finite number, got {rope_theta!r}")
    if not (math.isfinite(rope_theta) and rope_theta > 0):
        raise ValueError(f"rope_theta must be a positive finite number, got {rope_theta!r}")
    return float(rope_theta)


def _check_position_ids(position_ids: torch.Tensor) -> None:
    if not isinstance(position_ids, torch.Tensor):
        raise TypeError(f"position_ids must be a tensor, got {type(position_ids).__name__}")
    if position_ids.dtype.is_floating_point or position_ids.dtype == torch.bool:
        raise TypeError(f"position_ids must be an integer tensor, got {position_ids.dtype}")
    if position_ids.dim() not in (1, 2):
        raise ValueError(
            f"position_ids must have shape (T,) or (B, T), got {tuple(position_ids.shape)}"
        )
    if position_ids.numel() and position_ids.min() < 0:
        raise ValueError("position_ids must be non-negative")


# ------------------------------------------------------------ frequency tables
def compute_inv_freq(
    head_dim: int,
    rope_theta: float,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Inverse frequencies ``w_i = theta^(-2i/D)``, shape ``(D/2,)``.

    ``w_0 = 1`` (fastest rotation: one radian per token); ``w_{D/2-1}`` is close to
    ``1/theta`` (slowest). Same expression as HF, so float32 results are bit-identical.
    Pass ``dtype=torch.float64`` for high-precision mathematical checks.
    """
    head_dim = _check_head_dim(head_dim)
    theta = _check_theta(rope_theta)
    exponents = torch.arange(0, head_dim, 2, dtype=dtype, device=device) / head_dim
    return 1.0 / (theta**exponents)


def make_position_ids(
    seq_len: int,
    *,
    offset: int = 0,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Positions ``offset, offset+1, ..., offset+seq_len-1``.

    ``offset`` is the number of tokens already processed (the KV-cache length
    during decoding). Returns ``(T,)`` or, with ``batch_size``, ``(B, T)``.
    """
    for name, value in (("seq_len", seq_len), ("offset", offset)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    ids = torch.arange(offset, offset + seq_len, dtype=torch.long, device=device)
    if batch_size is not None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
        ids = ids.unsqueeze(0).expand(batch_size, -1)
    return ids


def rope_cos_sin(
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables for the given positions, shape ``position_ids.shape + (D,)``.

    Angles are ``position * inv_freq`` computed in ``inv_freq.dtype`` (float32 for
    Qwen3), duplicated to length ``D`` for the split-half layout, then cos/sin are
    taken and finally cast to ``dtype`` (default: ``inv_freq.dtype``).
    """
    _check_position_ids(position_ids)
    if inv_freq.dim() != 1:
        raise ValueError(f"inv_freq must be 1-D, got shape {tuple(inv_freq.shape)}")
    # Angles in at least float32: a bfloat16 position cannot even represent 257.
    angle_dtype = torch.promote_types(inv_freq.dtype, torch.float32)
    angles = position_ids[..., None].to(angle_dtype) * inv_freq.to(angle_dtype)  # (..., T, D/2)
    angles = torch.cat((angles, angles), dim=-1)  # (..., T, D): pair i uses angle i twice
    out_dtype = dtype or inv_freq.dtype
    return angles.cos().to(out_dtype), angles.sin().to(out_dtype)


# ----------------------------------------------------------------- rotations
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[x1, x2] -> [-x2, x1]`` on the last dim (the 90-degree turn of every pair)."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape ``(B, N, T, D)`` with tables ``(T, D)`` or ``(B, T, D)``.

    ``y = x * cos + rotate_half(x) * sin``. The tables get a head axis inserted
    (``(1|B, 1, T, D)``) so the same rotation applies to every head.
    """
    if not isinstance(x, torch.Tensor) or not torch.is_floating_point(x):
        raise TypeError("x must be a floating-point tensor")
    if x.dim() != 4:
        raise ValueError(f"x must have shape (B, N, T, D), got {tuple(x.shape)}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin shapes differ: {tuple(cos.shape)} vs {tuple(sin.shape)}")
    if cos.dtype != x.dtype or sin.dtype != x.dtype:
        raise ValueError(
            f"cos/sin dtype ({cos.dtype}) must equal x dtype ({x.dtype}); "
            "build the tables with dtype=x.dtype"
        )
    batch, _, seq_len, head_dim = x.shape
    if head_dim % 2:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    if cos.dim() == 2:
        expected_ok = cos.shape == (seq_len, head_dim)
        cos, sin = cos[None, None], sin[None, None]  # (1, 1, T, D)
    elif cos.dim() == 3:
        expected_ok = cos.shape[1:] == (seq_len, head_dim) and cos.shape[0] in (1, batch)
        cos, sin = cos[:, None], sin[:, None]  # (B, 1, T, D)
    else:
        expected_ok = False
    if not expected_ok:
        raise ValueError(
            f"cos/sin of shape {tuple(cos.shape)} do not match x of shape {tuple(x.shape)}; "
            "expected (T, D) or (B or 1, T, D)"
        )
    return x * cos + rotate_half(x) * sin


def apply_rope_to_qk(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate queries ``(B, Nq, T, D)`` and keys ``(B, Nkv, T, D)`` with the same tables."""
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError(
            f"q {tuple(q.shape)} and k {tuple(k.shape)} must agree on B, T and D "
            "(only the number of heads may differ)"
        )
    return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


# ------------------------------------------------------ educational helpers
def rotate_2d(v: torch.Tensor, angle: torch.Tensor | float) -> torch.Tensor:
    """Multiply 2-vectors ``(..., 2)`` by ``R(angle) = [[cos, -sin], [sin, cos]]``."""
    angle = torch.as_tensor(angle, dtype=v.dtype)
    c, s = torch.cos(angle), torch.sin(angle)
    return torch.stack((c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]), dim=-1)


def apply_rope_interleaved(
    x: torch.Tensor, position_ids: torch.Tensor, inv_freq: torch.Tensor
) -> torch.Tensor:
    """RoFormer-style RoPE on ADJACENT pairs ``(x[2i], x[2i+1])``. Not Qwen3's layout.

    ``x``: ``(..., T, D)``; ``position_ids``: ``(T,)``. Kept only to demonstrate that
    the two pairings agree up to the permutation from :func:`half_layout_permutation`.
    """
    _check_position_ids(position_ids)
    angles = position_ids[..., None].to(inv_freq.dtype) * inv_freq  # (T, D/2)
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]  # (..., T, D/2) each
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


def half_layout_permutation(head_dim: int) -> torch.Tensor:
    """Index ``perm`` such that ``x_half = x_interleaved[..., perm]``.

    Interleaved pair ``i`` lives at features ``(2i, 2i+1)``; split-half pair ``i``
    lives at ``(i, i + D/2)``. So ``perm = [0, 2, 4, ..., D-2, 1, 3, ..., D-1]``.
    """
    head_dim = _check_head_dim(head_dim)
    return torch.cat((torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)))


# --------------------------------------------------------------------- module
class RotaryEmbedding(nn.Module):
    """Produces RoPE cos/sin tables for given token positions.

    Args:
        head_dim: rotated size ``D`` (Qwen3: ``config.head_dim``; the whole head is rotated).
        rope_theta: base ``theta`` (Qwen3-0.6B: 1,000,000).
        max_position_embeddings: if given, cos/sin are precomputed (float32) for
            positions ``0 .. max-1`` and looked up; positions beyond fall back to
            direct computation. Lookup and direct computation are bit-identical.

    Buffers (not saved in the state dict, like HF): ``inv_freq`` ``(D/2,)`` float32.
    """

    def __init__(
        self,
        head_dim: int,
        rope_theta: float = 10_000.0,
        *,
        max_position_embeddings: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = _check_head_dim(head_dim)
        self.rope_theta = _check_theta(rope_theta)
        self.register_buffer(
            "inv_freq", compute_inv_freq(head_dim, rope_theta, device=device), persistent=False
        )
        max_pos = max_position_embeddings
        if max_pos is not None and (
            isinstance(max_pos, bool) or not isinstance(max_pos, int) or max_pos <= 0
        ):
            raise ValueError(f"max_position_embeddings must be a positive int, got {max_pos!r}")
        self.max_position_embeddings = max_pos
        cos = sin = None
        if max_pos is not None:  # float32 tables for positions 0 .. max_pos - 1
            cos, sin = rope_cos_sin(make_position_ids(max_pos, device=device), self.inv_freq)
        self.register_buffer("_cos_cached", cos, persistent=False)
        self.register_buffer("_sin_cached", sin, persistent=False)

    @classmethod
    def from_config(cls, config: Qwen3Config, *, cache: bool = False, **kwargs) -> RotaryEmbedding:
        return cls(
            config.head_dim,
            config.rope_theta,
            max_position_embeddings=config.max_position_embeddings if cache else None,
            **kwargs,
        )

    def forward(
        self, position_ids: torch.Tensor, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` of shape ``position_ids.shape + (D,)`` in ``dtype``."""
        _check_position_ids(position_ids)
        if self._cos_cached is not None and (
            position_ids.numel() == 0 or int(position_ids.max()) < self._cos_cached.shape[0]
        ):
            return self._cos_cached[position_ids].to(dtype), self._sin_cached[position_ids].to(
                dtype
            )
        return rope_cos_sin(position_ids, self.inv_freq, dtype=dtype)

    def _apply(self, fn, recurse=True):
        # ``module.to(torch.bfloat16)`` would round inv_freq (and the cached tables) to
        # bfloat16. Keep them float32 (recomputed on the new device), like HF models
        # created directly in bfloat16; device moves and float64 are honoured.
        super()._apply(fn, recurse)
        if self.inv_freq.dtype not in (torch.float32, torch.float64):
            device = self.inv_freq.device
            self._buffers["inv_freq"] = compute_inv_freq(
                self.head_dim, self.rope_theta, device=device
            )
            if self._cos_cached is not None:
                pos = make_position_ids(self._cos_cached.shape[0], device=device)
                cos, sin = rope_cos_sin(pos, self._buffers["inv_freq"])
                self._buffers["_cos_cached"], self._buffers["_sin_cached"] = cos, sin
        return self

    def extra_repr(self) -> str:
        cached = self.max_position_embeddings or 0
        return (
            f"head_dim={self.head_dim}, rope_theta={self.rope_theta:g}, cached_positions={cached}"
        )
