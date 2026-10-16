"""Qwen3 self-attention: projections, QK-Norm, RoPE, grouped KV heads, causal softmax.

Shapes (B batch, T tokens, H hidden, Nq query heads, Nkv key/value heads,
D head_dim, g = Nq / Nkv)::

    x                          [B, T, H]
    q_proj(x) -> view/transpose [B, Nq,  T, D]     q_proj: H -> Nq*D
    k_proj(x) -> view/transpose [B, Nkv, T, D]     k_proj: H -> Nkv*D
    v_proj(x) -> view/transpose [B, Nkv, T, D]     v_proj: H -> Nkv*D
    q, k <- q_norm(q), k_norm(k)                   per-head RMSNorm over D (M1.4 part A)
    q, k <- RoPE(q), RoPE(k)                       split-half rotation (M1.3)
    k, v <- repeat_kv(k, g), repeat_kv(v, g)       [B, Nq, T, D]; query head h uses kv head h // g
    scores = (q @ k^T) * D**-0.5                   [B, Nq, T, T]
    scores[not allowed] = -inf                     causal mask, BEFORE softmax
    probs  = softmax(scores, dim=-1) (float32)     [B, Nq, T, T], rows sum to 1
    ctx    = probs @ v                             [B, Nq, T, D]
    out    = o_proj(ctx.transpose(1, 2).reshape(B, T, Nq*D))   [B, T, H]

``Nq * D`` need not equal ``H`` (Qwen3-0.6B: 16 * 128 = 2048 vs H = 1024).
Numerics follow Hugging Face eager attention (``Qwen3Attention`` +
``eager_attention_forward``): scores scaled by multiplying with ``D**-0.5``,
softmax computed in float32 then cast back to the activation dtype, no dropout
(``attention_dropout = 0.0``).

This first version materialises the repeated K/V heads for clarity; the KV cache
(M2) and the C++ port will index the shared heads instead (see docs/components/attention.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from qwen3_scratch.config import Qwen3Config
from qwen3_scratch.layers.qk_norm import apply_qk_norm
from qwen3_scratch.layers.rmsnorm import RMSNorm
from qwen3_scratch.layers.rope import apply_rope_to_qk

__all__ = [
    "AttentionTrace",
    "GroupedQueryAttention",
    "attention_softmax",
    "bool_to_additive_mask",
    "causal_mask",
    "query_to_kv_head",
    "repeat_kv",
]


# ----------------------------------------------------------- KV head grouping
def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def query_to_kv_head(num_attention_heads: int, num_key_value_heads: int) -> list[int]:
    """KV head used by each query head: contiguous groups, ``h -> h // g``.

    Qwen3-0.6B (16 query, 8 KV heads, g = 2): ``[0, 0, 1, 1, 2, 2, ..., 7, 7]``.
    A modulo mapping (``h % Nkv``: ``[0, 1, ..., 7, 0, 1, ..., 7]``) would pair the
    wrong heads with trained weights.
    """
    nq = _positive_int(num_attention_heads, "num_attention_heads")
    nkv = _positive_int(num_key_value_heads, "num_key_value_heads")
    if nq % nkv:
        raise ValueError(
            f"num_attention_heads ({nq}) must be divisible by num_key_value_heads ({nkv})"
        )
    group = nq // nkv
    return [h // group for h in range(nq)]


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """``[B, Nkv, T, D] -> [B, Nkv * n_rep, T, D]``, each KV head repeated ``n_rep`` times in place.

    Output head ``j * n_rep + r`` is a copy of input head ``j`` (same as
    ``torch.repeat_interleave(x, n_rep, dim=1)``). ``n_rep = 1`` returns ``x`` itself.
    The copy is materialised for readability; gradients of the copies are summed
    back into the shared head by autograd.
    """
    n_rep = _positive_int(n_rep, "n_rep")
    if not isinstance(x, torch.Tensor) or x.dim() != 4:
        raise ValueError(f"expected a [B, Nkv, T, D] tensor, got {getattr(x, 'shape', type(x))}")
    if n_rep == 1:
        return x
    b, nkv, t, d = x.shape
    return x[:, :, None, :, :].expand(b, nkv, n_rep, t, d).reshape(b, nkv * n_rep, t, d)


# --------------------------------------------------------------------- masks
def causal_mask(q_positions: torch.Tensor, k_positions: torch.Tensor) -> torch.Tensor:
    """Boolean "may attend" mask from ABSOLUTE positions: ``allowed[i, j] = k_pos[j] <= q_pos[i]``.

    ``q_positions``: ``[Tq]`` or ``[B, Tq]``; ``k_positions``: ``[Tk]`` or ``[B, Tk]``.
    Returns ``[Tq, Tk]`` or ``[B, Tq, Tk]``. Using positions (not a plain
    upper-triangular matrix) keeps the mask correct when ``Tq != Tk`` during
    cached decoding: a new token at position 7 may see cached keys 0..7.
    """
    for name, pos in (("q_positions", q_positions), ("k_positions", k_positions)):
        if (
            not isinstance(pos, torch.Tensor)
            or pos.dtype.is_floating_point
            or pos.dim() not in (1, 2)
        ):
            raise ValueError(f"{name} must be an integer tensor of shape [T] or [B, T]")
    return k_positions[..., None, :] <= q_positions[..., :, None]


def bool_to_additive_mask(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """HF-style additive mask: 0 where allowed, ``finfo(dtype).min`` where masked."""
    zero = torch.zeros((), dtype=dtype, device=allowed.device)
    return torch.where(allowed, zero, torch.finfo(dtype).min)


def _broadcast_mask(allowed: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    if allowed.dtype != torch.bool:
        raise ValueError(f"attention_mask must be boolean (True = may attend), got {allowed.dtype}")
    if allowed.dim() == 2:
        allowed = allowed[None, None]  # [1, 1, Tq, Tk]
    elif allowed.dim() == 3:
        allowed = allowed[:, None]  # [B, 1, Tq, Tk]
    elif allowed.dim() != 4:
        raise ValueError(f"attention_mask must have 2, 3 or 4 dims, got {tuple(allowed.shape)}")
    if allowed.shape[-2:] != scores.shape[-2:]:
        raise ValueError(
            f"attention_mask covers {tuple(allowed.shape[-2:])} "
            f"but scores are {tuple(scores.shape[-2:])}"
        )
    try:
        torch.broadcast_shapes(allowed.shape, scores.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"attention_mask {tuple(allowed.shape)} does not broadcast to "
            f"scores {tuple(scores.shape)}"
        ) from exc
    return allowed


def attention_softmax(scores: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Masked, numerically stable softmax over the key axis (float32 or wider).

    Masked logits are set to ``-inf`` *before* the softmax, so they get exactly zero
    probability. ``torch.softmax`` subtracts the row maximum internally (stable).
    A row with no allowed key (fully masked) returns all zeros instead of NaN.
    The result is cast back to ``scores.dtype``, as in HF eager attention.
    """
    allowed = _broadcast_mask(allowed, scores)
    masked = scores.masked_fill(~allowed, float("-inf"))
    has_key = allowed.any(dim=-1, keepdim=True)
    masked = torch.where(has_key, masked, torch.zeros((), dtype=scores.dtype))  # avoid NaN rows
    # float32 for half precision (as HF); float64 stays float64 (HF would downcast it).
    softmax_dtype = torch.promote_types(scores.dtype, torch.float32)
    probs = torch.softmax(masked, dim=-1, dtype=softmax_dtype).to(scores.dtype)
    return probs.masked_fill(~has_key, 0.0)


# ---------------------------------------------------------------- the module
@dataclass
class AttentionTrace:
    """Intermediate tensors of one forward pass (for tests, parity checks, teaching)."""

    q_proj: torch.Tensor  # [B, T, Nq*D]
    k_proj: torch.Tensor  # [B, T, Nkv*D]
    v_proj: torch.Tensor  # [B, T, Nkv*D]
    q_normed: torch.Tensor  # [B, Nq, T, D]
    k_normed: torch.Tensor  # [B, Nkv, T, D]
    q_rope: torch.Tensor  # [B, Nq, T, D]
    k_rope: torch.Tensor  # [B, Nkv, T, D]
    k_grouped: torch.Tensor  # [B, Nq, T, D]
    v_grouped: torch.Tensor  # [B, Nq, T, D]
    scores: torch.Tensor  # [B, Nq, T, T], scaled, -inf where masked
    probs: torch.Tensor  # [B, Nq, T, T]
    context: torch.Tensor  # [B, T, Nq*D], merged heads before o_proj
    output: torch.Tensor  # [B, T, H]


class GroupedQueryAttention(nn.Module):
    """Qwen3 causal self-attention with grouped KV heads.

    Covers MHA (``Nkv == Nq``), GQA (``1 < Nkv < Nq``) and MQA (``Nkv == 1``).

    Args:
        hidden_size: ``H``.
        num_attention_heads: ``Nq``.
        num_key_value_heads: ``Nkv`` (must divide ``Nq``).
        head_dim: ``D`` (even, for RoPE); independent of ``H / Nq``.
        rms_norm_eps: epsilon of ``q_norm`` / ``k_norm``.
        attention_bias: biases on the four projections (Qwen3: ``False``).
        qk_norm: apply QK-Norm (Qwen3: ``True``; ``False`` gives plain Llama-style GQA).

    Parameters use the HF checkpoint names: ``q_proj``, ``k_proj``, ``v_proj``,
    ``o_proj``, ``q_norm``, ``k_norm``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        *,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        qk_norm: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = _positive_int(hidden_size, "hidden_size")
        self.head_dim = _positive_int(head_dim, "head_dim")
        if head_dim % 2:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.kv_head_of_query = query_to_kv_head(num_attention_heads, num_key_value_heads)
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.scaling = head_dim**-0.5
        factory = {"device": device, "dtype": dtype}
        q_dim, kv_dim = num_attention_heads * head_dim, num_key_value_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, q_dim, bias=attention_bias, **factory)
        self.k_proj = nn.Linear(hidden_size, kv_dim, bias=attention_bias, **factory)
        self.v_proj = nn.Linear(hidden_size, kv_dim, bias=attention_bias, **factory)
        self.o_proj = nn.Linear(q_dim, hidden_size, bias=attention_bias, **factory)
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, rms_norm_eps, **factory)
            self.k_norm = RMSNorm(head_dim, rms_norm_eps, **factory)
        else:
            self.q_norm = self.k_norm = None

    @classmethod
    def from_config(cls, config: Qwen3Config, **kwargs) -> GroupedQueryAttention:
        return cls(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            qk_norm=config.qk_norm,
            **kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        return_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, AttentionTrace]:
        """Full-sequence self-attention.

        Args:
            hidden_states: ``[B, T, H]``.
            cos, sin: RoPE tables for the ``T`` positions, ``[T, D]`` or ``[B, T, D]``,
                in the activation dtype (``RotaryEmbedding(position_ids, dtype=x.dtype)``).
                Positions may start at any offset.
            attention_mask: optional boolean "may attend" mask ``[T, T]``, ``[B, T, T]``
                or ``[B, 1 | Nq, T, T]``. Default: causal over the ``T`` tokens of this
                call (query and key positions are the same tokens). Combine padding
                masks with :func:`causal_mask` explicitly.
            return_trace: also return an :class:`AttentionTrace`.
        """
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.dim() != 3:
            raise ValueError(
                "hidden_states must have shape [B, T, H], "
                f"got {getattr(hidden_states, 'shape', None)}"
            )
        b, t, h = hidden_states.shape
        if h != self.hidden_size:
            raise ValueError(f"hidden size is {h}, expected {self.hidden_size}")
        nq, nkv, d = self.num_attention_heads, self.num_key_value_heads, self.head_dim

        # 1-3. projections, then split heads: [B, T, N*D] -> [B, T, N, D] -> [B, N, T, D]
        q_lin = self.q_proj(hidden_states)
        k_lin = self.k_proj(hidden_states)
        v_lin = self.v_proj(hidden_states)
        q = q_lin.view(b, t, nq, d).transpose(1, 2)
        k = k_lin.view(b, t, nkv, d).transpose(1, 2)
        v = v_lin.view(b, t, nkv, d).transpose(1, 2)

        # 4-5. QK-Norm (per head, over D), values untouched
        if self.q_norm is not None:
            q, k = apply_qk_norm(q, k, self.q_norm, self.k_norm)
        q_normed, k_normed = q, k

        # 6. RoPE on queries and keys (same tables for all heads)
        q, k = apply_rope_to_qk(q, k, cos, sin)

        # 7. grouped KV heads: query head h reads kv head h // g
        k_grp = repeat_kv(k, self.num_kv_groups)
        v_grp = repeat_kv(v, self.num_kv_groups)

        # 8-10. scaled scores, causal mask before softmax, float32 softmax
        scores = torch.matmul(q, k_grp.transpose(2, 3)) * self.scaling  # [B, Nq, T, T]
        if attention_mask is None:
            pos = torch.arange(t, device=hidden_states.device)
            attention_mask = causal_mask(pos, pos)
        allowed = _broadcast_mask(attention_mask, scores)
        probs = attention_softmax(scores, allowed)

        # 11-12. weighted sum of values, merge heads, output projection
        ctx = torch.matmul(probs, v_grp)  # [B, Nq, T, D]
        ctx = ctx.transpose(1, 2).reshape(b, t, nq * d)
        out = self.o_proj(ctx)
        if not return_trace:
            return out
        trace = AttentionTrace(
            q_proj=q_lin,
            k_proj=k_lin,
            v_proj=v_lin,
            q_normed=q_normed,
            k_normed=k_normed,
            q_rope=q,
            k_rope=k,
            k_grouped=k_grp,
            v_grouped=v_grp,
            scores=scores.masked_fill(~allowed, float("-inf")),
            probs=probs,
            context=ctx,
            output=out,
        )
        return out, trace

    def kv_cache_bytes_per_token(self, bytes_per_element: int = 2) -> int:
        """K and V cache size for one token in this layer: ``2 * Nkv * D * bytes``."""
        return 2 * self.num_key_value_heads * self.head_dim * bytes_per_element

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_attention_heads={self.num_attention_heads}, "
            f"num_key_value_heads={self.num_key_value_heads}, head_dim={self.head_dim}, "
            f"qk_norm={self.q_norm is not None}"
        )
