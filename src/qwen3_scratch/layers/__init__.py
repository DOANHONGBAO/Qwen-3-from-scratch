"""Neural-network building blocks of Qwen3, one module per component.

Implemented so far (see docs/roadmap.md):

* :mod:`qwen3_scratch.layers.rmsnorm` - RMSNorm (M1.1)
* :mod:`qwen3_scratch.layers.swiglu` - SwiGLU feed-forward (M1.2)
* :mod:`qwen3_scratch.layers.rope` - rotary position embedding (M1.3)
* :mod:`qwen3_scratch.layers.qk_norm` - query/key normalisation (M1.4)
* :mod:`qwen3_scratch.layers.attention` - grouped-query causal self-attention (M1.4)
* :mod:`qwen3_scratch.layers.decoder` - pre-norm decoder layer (M1.5)
"""

from qwen3_scratch.layers.attention import (
    AttentionTrace,
    GroupedQueryAttention,
    attention_softmax,
    bool_to_additive_mask,
    causal_mask,
    query_to_kv_head,
    repeat_kv,
)
from qwen3_scratch.layers.decoder import DecoderTrace, Qwen3DecoderLayer
from qwen3_scratch.layers.qk_norm import QKNorm, apply_qk_norm
from qwen3_scratch.layers.rmsnorm import RMSNorm, rms_norm, rms_norm_textbook
from qwen3_scratch.layers.rope import (
    RotaryEmbedding,
    apply_rope,
    apply_rope_to_qk,
    compute_inv_freq,
    make_position_ids,
    rope_cos_sin,
    rotate_half,
)
from qwen3_scratch.layers.swiglu import SwiGLUFeedForward, silu_textbook, swiglu_textbook

__all__ = [
    "AttentionTrace",
    "GroupedQueryAttention",
    "attention_softmax",
    "bool_to_additive_mask",
    "causal_mask",
    "query_to_kv_head",
    "repeat_kv",
    "DecoderTrace",
    "Qwen3DecoderLayer",
    "QKNorm",
    "RMSNorm",
    "RotaryEmbedding",
    "SwiGLUFeedForward",
    "apply_qk_norm",
    "apply_rope",
    "apply_rope_to_qk",
    "compute_inv_freq",
    "make_position_ids",
    "rope_cos_sin",
    "rotate_half",
    "rms_norm",
    "rms_norm_textbook",
    "silu_textbook",
    "swiglu_textbook",
]
