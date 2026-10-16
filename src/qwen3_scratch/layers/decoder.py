"""Qwen3 decoder layer: two pre-norm residual sublayers (attention, then SwiGLU).

    h1  = x  + SelfAttention( input_layernorm(x) )              [B, T, H]
    out = h1 + SwiGLU( post_attention_layernorm(h1) )          [B, T, H]

* Pre-norm: each sublayer sees a normalised copy of the residual stream; the
  stream itself (``x``, ``h1``) is never normalised inside the block.
* Decoder-only: self-attention is causal; there is no cross-attention.
* Positions enter only through RoPE inside attention (``cos``/``sin`` are computed
  once per model forward and shared by all layers); nothing is added to ``x``.

All submodules are the verified components from M1.1-M1.4; this module only wires
them. Parameter names equal the HF checkpoint (``model.layers.{i}.input_layernorm``,
``.self_attn``, ``.post_attention_layernorm``, ``.mlp``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from qwen3_scratch.config import Qwen3Config
from qwen3_scratch.kv_cache import KVCache
from qwen3_scratch.layers.attention import AttentionTrace, GroupedQueryAttention
from qwen3_scratch.layers.rmsnorm import RMSNorm
from qwen3_scratch.layers.swiglu import SwiGLUFeedForward

__all__ = ["DecoderTrace", "Qwen3DecoderLayer"]


@dataclass
class DecoderTrace:
    """Intermediate tensors of one decoder layer forward pass (all ``[B, T, H]``)."""

    input_norm: torch.Tensor  # input_layernorm(x)
    attn_output: torch.Tensor  # self_attn(...)
    residual_1: torch.Tensor  # h1 = x + attn_output
    post_attn_norm: torch.Tensor  # post_attention_layernorm(h1)
    mlp_output: torch.Tensor  # mlp(...)
    output: torch.Tensor  # h1 + mlp_output
    attention: AttentionTrace | None = None


class Qwen3DecoderLayer(nn.Module):
    """One Qwen3 transformer block.

    Args mirror :class:`~qwen3_scratch.config.Qwen3Config`; use :meth:`from_config`.
    ``layer_idx`` selects this layer's slot in a :class:`~qwen3_scratch.kv_cache.KVCache`.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        *,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        qk_norm: bool = True,
        layer_idx: int = 0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps, **factory)
        self.self_attn = GroupedQueryAttention(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            qk_norm=qk_norm,
            **factory,
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps, **factory)
        self.mlp = SwiGLUFeedForward(hidden_size, intermediate_size, bias=False, **factory)

    @classmethod
    def from_config(cls, config: Qwen3Config, layer_idx: int = 0, **kwargs) -> Qwen3DecoderLayer:
        return cls(
            config.hidden_size,
            config.intermediate_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            qk_norm=config.qk_norm,
            layer_idx=layer_idx,
            **kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        kv_cache: KVCache | None = None,
        return_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, DecoderTrace]:
        """``[B, T, H] -> [B, T, H]``.

        ``cos``/``sin`` and ``attention_mask`` are passed unchanged to
        :class:`GroupedQueryAttention` (boolean "may attend" mask; default causal).
        With ``kv_cache`` the attention appends this layer's new K/V to cache layer
        ``self.layer_idx`` and attends over all cached tokens; the MLP and norms act
        on the new tokens only (they are position-wise).
        """
        residual = hidden_states  # 1. save the residual stream
        normed = self.input_layernorm(hidden_states)  # 2. pre-norm
        attn_out = self.self_attn(  # 3. causal self-attention
            normed,
            cos,
            sin,
            attention_mask,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
            return_trace=return_trace,
        )
        attn_trace = None
        if return_trace:
            attn_out, attn_trace = attn_out
        h1 = residual + attn_out  # 4. first residual connection

        residual = h1  # 5. save again
        normed_2 = self.post_attention_layernorm(h1)  # 6. pre-norm
        mlp_out = self.mlp(normed_2)  # 7. SwiGLU
        out = residual + mlp_out  # 8. second residual connection
        if not return_trace:
            return out
        return out, DecoderTrace(normed, attn_out, h1, normed_2, mlp_out, out, attn_trace)
