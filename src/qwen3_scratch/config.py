"""Model configuration for Qwen3 dense models.

This module intentionally does **not** import torch: the configuration is a plain,
validated data structure that the PyTorch model (M1) and the C++20 port (M3) will
both consume. Field names follow the official Hugging Face ``config.json`` of
``Qwen/Qwen3-0.6B-Base`` because that file is the source of truth for checkpoint
compatibility. Names used by the educational reference implementation
(Raschka, *Build a Reasoning Model (From Scratch)*, Appendix C) are listed in
``RASCHKA_KEY_MAP`` for cross-reading.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

SUPPORTED_DTYPES = ("float32", "bfloat16", "float16")

# Hugging Face config key -> key used in rasbt/reasoning-from-scratch Appendix C.
RASCHKA_KEY_MAP: dict[str, str] = {
    "vocab_size": "vocab_size",
    "max_position_embeddings": "context_length",
    "hidden_size": "emb_dim",
    "num_attention_heads": "n_heads",
    "num_hidden_layers": "n_layers",
    "intermediate_size": "hidden_dim",
    "head_dim": "head_dim",
    "qk_norm": "qk_norm",
    "num_key_value_heads": "n_kv_groups",
    "rope_theta": "rope_base",
    "dtype": "dtype",
}


@dataclass(frozen=True)
class Qwen3Config:
    """Hyperparameters of a Qwen3 dense decoder-only Transformer.

    Shapes implied by the config (B = batch, T = sequence length):

    * token embedding: ``(vocab_size, hidden_size)``
    * q_proj: ``hidden_size -> num_attention_heads * head_dim``
    * k_proj / v_proj: ``hidden_size -> num_key_value_heads * head_dim``
    * o_proj: ``num_attention_heads * head_dim -> hidden_size``
    * MLP (SwiGLU): ``hidden_size -> intermediate_size -> hidden_size``

    Note that ``head_dim`` is independent of ``hidden_size / num_attention_heads``
    (for Qwen3-0.6B: 1024 / 16 = 64, but ``head_dim`` = 128).
    """

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool
    attention_bias: bool = False
    qk_norm: bool = True
    dtype: str = "float32"
    attention_dropout: float = 0.0
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        """Raise ``ValueError`` if the configuration is inconsistent."""
        positive_ints = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads "
                f"(got {self.num_attention_heads} and {self.num_key_value_heads})"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")
        if not self.rope_theta > 0:
            raise ValueError(f"rope_theta must be > 0, got {self.rope_theta}")
        if not self.rms_norm_eps > 0:
            raise ValueError(f"rms_norm_eps must be > 0, got {self.rms_norm_eps}")
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"dtype must be one of {SUPPORTED_DTYPES}, got {self.dtype!r}")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError(f"attention_dropout must be in [0, 1), got {self.attention_dropout}")
        if not self.initializer_range > 0:
            raise ValueError(f"initializer_range must be > 0, got {self.initializer_range}")
        for name in ("attention_bias", "qk_norm", "tie_word_embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}")

    # ------------------------------------------------------- derived quantities
    @property
    def num_queries_per_kv(self) -> int:
        """GQA group size: how many query heads share one key/value head."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def q_proj_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_proj_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def estimate_num_parameters(self, include_embeddings: bool = True) -> int:
        """Analytic parameter count (no torch needed).

        Counts embeddings, attention projections, QK-Norm weights, the SwiGLU MLP,
        both RMSNorms per block, and the final RMSNorm. The LM head is counted
        separately only when embeddings are *not* tied.
        """
        h, layers = self.hidden_size, self.num_hidden_layers
        bias = self.attention_bias
        attn = h * self.q_proj_dim + 2 * h * self.kv_proj_dim + self.q_proj_dim * h
        if bias:
            attn += self.q_proj_dim + 2 * self.kv_proj_dim + h
        qk_norm = 2 * self.head_dim if self.qk_norm else 0
        mlp = 3 * h * self.intermediate_size
        block_norms = 2 * h
        per_layer = attn + qk_norm + mlp + block_norms
        total = layers * per_layer + h  # + final norm
        if include_embeddings:
            total += self.vocab_size * h
            if not self.tie_word_embeddings:
                total += self.vocab_size * h
        return total

    # --------------------------------------------------------------- (de)serial
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Qwen3Config:
        """Build from a dict using this class's own field names (strict)."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_hf_config(cls, hf: dict[str, Any], dtype: str = "float32") -> Qwen3Config:
        """Map an official Hugging Face Qwen3 ``config.json`` dict onto this schema.

        Features this project does not (yet) support raise ``NotImplementedError``
        instead of being silently ignored.
        """
        if hf.get("model_type") != "qwen3":
            raise ValueError(f"Expected model_type 'qwen3', got {hf.get('model_type')!r}")
        if hf.get("hidden_act", "silu") != "silu":
            raise NotImplementedError(f"Only SiLU is supported, got {hf['hidden_act']!r}")
        if hf.get("use_sliding_window", False):
            raise NotImplementedError("Sliding-window attention is not supported")
        if hf.get("rope_scaling") is not None:
            raise NotImplementedError("RoPE scaling variants are not supported yet")
        head_dim = hf.get("head_dim") or hf["hidden_size"] // hf["num_attention_heads"]
        return cls(
            vocab_size=hf["vocab_size"],
            hidden_size=hf["hidden_size"],
            intermediate_size=hf["intermediate_size"],
            num_hidden_layers=hf["num_hidden_layers"],
            num_attention_heads=hf["num_attention_heads"],
            num_key_value_heads=hf["num_key_value_heads"],
            head_dim=head_dim,
            max_position_embeddings=hf["max_position_embeddings"],
            rope_theta=float(hf["rope_theta"]),
            rms_norm_eps=float(hf["rms_norm_eps"]),
            tie_word_embeddings=bool(hf.get("tie_word_embeddings", False)),
            attention_bias=bool(hf.get("attention_bias", False)),
            qk_norm=True,  # Qwen3 always applies per-head RMSNorm to Q and K
            dtype=dtype,
            attention_dropout=float(hf.get("attention_dropout", 0.0)),
            initializer_range=float(hf.get("initializer_range", 0.02)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> Qwen3Config:
        """Load either a native config JSON or an official HF ``config.json``."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "model_type" in data:
            return cls.from_hf_config(data)
        return cls.from_dict(data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


# Official Qwen/Qwen3-0.6B-Base hyperparameters (config.json, transformers 4.51.0).
QWEN3_0_6B_BASE = Qwen3Config(
    vocab_size=151_936,
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    max_position_embeddings=32_768,
    rope_theta=1_000_000.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
    attention_bias=False,
    qk_norm=True,
    dtype="bfloat16",
)

# Tiny configuration for fast CPU unit tests of individual components. It deliberately keeps the
# Qwen3-specific structure: GQA (4 query heads, 2 KV heads), head_dim
# independent of hidden_size / num_heads (64 / 4 = 16 != 32), QK-Norm and tied
# embeddings.
TINY_QWEN3 = Qwen3Config(
    vocab_size=512,
    hidden_size=64,
    intermediate_size=192,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    max_position_embeddings=128,
    rope_theta=10_000.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
    attention_bias=False,
    qk_norm=True,
    dtype="float32",
)

# Tiny Qwen3 *language model* used by the M1.6 model tests, notebook and CLI demo
# (configs/qwen3_tiny.json). Educational random-weight model, NOT a checkpoint.
# Note: here head_dim == hidden_size / num_attention_heads (128 / 4 = 32); tests also
# run the full model with TINY_QWEN3, where the two differ.
QWEN3_TINY = Qwen3Config(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=384,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    max_position_embeddings=128,
    rope_theta=1_000_000.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=True,
    attention_bias=False,
    qk_norm=True,
    dtype="float32",
    attention_dropout=0.0,
    initializer_range=0.02,
)

PRESETS: dict[str, Qwen3Config] = {
    "qwen3-0.6b-base": QWEN3_0_6B_BASE,
    "qwen3-tiny": QWEN3_TINY,
    "tiny": TINY_QWEN3,
}
