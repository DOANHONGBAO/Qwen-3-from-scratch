"""Neural-network building blocks of Qwen3, one module per component.

Implemented so far (see docs/roadmap.md):

* :mod:`qwen3_scratch.layers.rmsnorm` - RMSNorm (M1.1)
* :mod:`qwen3_scratch.layers.swiglu` - SwiGLU feed-forward (M1.2)
"""

from qwen3_scratch.layers.rmsnorm import RMSNorm, rms_norm, rms_norm_textbook
from qwen3_scratch.layers.swiglu import SwiGLUFeedForward, silu_textbook, swiglu_textbook

__all__ = [
    "RMSNorm",
    "SwiGLUFeedForward",
    "rms_norm",
    "rms_norm_textbook",
    "silu_textbook",
    "swiglu_textbook",
]
