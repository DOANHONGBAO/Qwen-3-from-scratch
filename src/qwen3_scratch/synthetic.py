"""Synthetic, randomly initialised checkpoints in the official Hugging Face layout.

Used for CI tests and for dry runs of the M2 scripts without downloading anything:
``config.json`` (official key names, top-level ``rope_theta`` like the Qwen3 Hub
file), ``model.safetensors`` (bfloat16, tied checkpoints omit ``lm_head.weight``),
``tokenizer.json`` / ``tokenizer_config.json`` (a small byte-level BPE with
Qwen-like settings: NFC, no BOS, ``<|endoftext|>`` as EOS and PAD).

The weights are RANDOM. Nothing produced here is a pretrained model.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from qwen3_scratch.config import Qwen3Config

__all__ = [
    "TOY_CONFIG_ONLY_TOKENS",
    "TOY_CORPUS",
    "TOY_SPECIAL_TOKENS",
    "build_toy_tokenizer",
    "write_synthetic_checkpoint",
]

TOY_SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
TOY_CONFIG_ONLY_TOKENS = ["<think>", "</think>"]
TOY_CORPUS = [
    "Artificial intelligence is the study of intelligent agents.",
    "Trí tuệ nhân tạo là một lĩnh vực của khoa học máy tính.",
    "Xin chào thế giới! Hôm nay trời đẹp quá. Hà Nội là thủ đô của Việt Nam.",
    "The year 2026 has 365 days; 3.14159 is pi. The capital of France is Paris.",
    "Unicode: café, naïve, 東京, Ελληνικά, emoji 🙂.",
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)\t# code",
    "1, 1, 2, 3, 5, 8, 13, 21, 34",
]


def build_toy_tokenizer(out_dir: str | Path, vocab_size: int = 400) -> Path:
    """Train a tiny byte-level BPE on ``TOY_CORPUS`` and write Qwen-like tokenizer files."""
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=TOY_SPECIAL_TOKENS,
        show_progress=False,
    )
    tok.train_from_iterator(TOY_CORPUS * 5, trainer=trainer)
    tok.save(str(out / "tokenizer.json"))
    added = {
        str(tok.token_to_id(t)): {
            "content": t,
            "special": True,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "single_word": False,
        }
        for t in TOY_SPECIAL_TOKENS
    }
    # Like Qwen3-0.6B-Base: two added tokens declared ONLY in tokenizer_config.json
    # (not in tokenizer.json); loaders must register them with exactly these ids.
    next_id = tok.get_vocab_size(with_added_tokens=True)
    for offset, content in enumerate(TOY_CONFIG_ONLY_TOKENS):
        added[str(next_id + offset)] = {
            "content": content,
            "special": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "single_word": False,
        }
    cfg = {
        "add_bos_token": False,
        "add_prefix_space": False,
        "bos_token": None,
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "unk_token": None,
        "errors": "replace",
        "clean_up_tokenization_spaces": False,
        "split_special_tokens": False,
        "model_max_length": 131072,
        "added_tokens_decoder": added,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    (out / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return out


def hf_config_dict(config: Qwen3Config) -> dict:
    """``config.json`` content in the official Qwen3 (transformers 4.51) format."""
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": config.attention_bias,
        "attention_dropout": config.attention_dropout,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "head_dim": config.head_dim,
        "hidden_act": "silu",
        "hidden_size": config.hidden_size,
        "initializer_range": config.initializer_range,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
        "max_window_layers": config.num_hidden_layers,
        "model_type": "qwen3",
        "num_attention_heads": config.num_attention_heads,
        "num_hidden_layers": config.num_hidden_layers,
        "num_key_value_heads": config.num_key_value_heads,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_scaling": None,
        "rope_theta": config.rope_theta,
        "sliding_window": None,
        "tie_word_embeddings": config.tie_word_embeddings,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": config.vocab_size,
    }


@torch.no_grad()
def write_synthetic_checkpoint(
    config: Qwen3Config,
    out_dir: str | Path,
    *,
    seed: int = 0,
    weight_std: float | None = None,
    with_tokenizer: bool = True,
) -> Path:
    """Write a random bfloat16 checkpoint for ``config`` (+ toy tokenizer) to ``out_dir``.

    ``weight_std`` overrides the init std of matrices (a larger std gives peaked,
    input-dependent next-token distributions, useful for greedy-generation tests).
    Norm weights get small random offsets so that they are really exercised.
    Tensors are created one at a time (peak memory ~ one tensor + the file).
    """
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    std = config.initializer_range if weight_std is None else weight_std
    from qwen3_scratch.weight_loader import expected_shapes

    tensors = {}
    for name, shape in expected_shapes(config).items():
        if name.endswith("norm.weight"):
            t = 1.0 + 0.1 * torch.randn(shape, generator=g)
        else:
            t = std * torch.randn(shape, generator=g)
        tensors[name] = t.to(torch.bfloat16)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(hf_config_dict(config), indent=2), encoding="utf-8")
    (out / "SYNTHETIC_RANDOM_WEIGHTS.txt").write_text(
        "Random weights generated by qwen3_scratch.synthetic - NOT a pretrained model.\n",
        encoding="utf-8",
    )
    if with_tokenizer:
        build_toy_tokenizer(out, vocab_size=min(400, config.vocab_size))
    return out
