"""Build OUR Qwen3 model from an official Hugging Face checkpoint directory.

Hugging Face is used only as a file provider here (``huggingface_hub`` downloads
``config.json``, ``model.safetensors`` and tokenizer files); the network is never
touched unless :func:`download_checkpoint` is called explicitly. Inference runs on
:class:`qwen3_scratch.model.Qwen3ForCausalLM`, never on ``transformers``.

Loading strategy (no random init, no full extra copy)::

    config.json --Qwen3Config.from_hf_config--> config
    Qwen3ForCausalLM(config, device="meta")     # shapes only, no memory
    .to_empty(device)                           # uninitialised storage, target dtype
    rebuild RoPE buffers, re-tie lm_head        # buffers do not survive to_empty
    load_into_model(model, safetensors)         # tensor by tensor, bf16 -> dtype

Peak memory ~= model in the target dtype + one checkpoint tensor (+ the page cache
of the memory-mapped safetensors file, which the OS can reclaim).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from qwen3_scratch.config import Qwen3Config
from qwen3_scratch.layers.rope import RotaryEmbedding
from qwen3_scratch.model import Qwen3ForCausalLM
from qwen3_scratch.weight_loader import LoadReport, SafetensorsCheckpoint, load_into_model

__all__ = [
    "CHECKPOINT_FILES",
    "QWEN3_0_6B_BASE_REPO",
    "QWEN3_0_6B_BASE_REVISION",
    "download_checkpoint",
    "load_hf_config",
    "load_pretrained",
]

QWEN3_0_6B_BASE_REPO = "Qwen/Qwen3-0.6B-Base"
# Commit of the Hub repository used for every M2 result (lastModified 2025-07-26).
QWEN3_0_6B_BASE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"

# Everything needed for inference; README/LICENSE are tiny and kept for provenance.
CHECKPOINT_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
]

_TORCH_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def download_checkpoint(
    repo_id: str = QWEN3_0_6B_BASE_REPO,
    revision: str = QWEN3_0_6B_BASE_REVISION,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download (or reuse from the HF cache) the pinned checkpoint; return its directory.

    Only safetensors and JSON/text files are fetched. Authentication, if ever
    needed, comes from the user's own ``huggingface-cli login`` / ``HF_TOKEN``;
    nothing is read or printed here.
    """
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=CHECKPOINT_FILES,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(path)


def load_hf_config(model_dir: str | Path) -> dict:
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _dtype_name(dtype: torch.dtype) -> str:
    for name, value in _TORCH_DTYPES.items():
        if value == dtype:
            return name
    raise ValueError(f"unsupported dtype {dtype}; use one of {list(_TORCH_DTYPES)}")


def load_pretrained(
    model_dir: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    strict: bool = True,
) -> tuple[Qwen3ForCausalLM, LoadReport]:
    """Instantiate our model from ``model_dir`` (``config.json`` + safetensors).

    ``dtype`` is the compute dtype of OUR model; checkpoint tensors (bfloat16 for
    Qwen3-0.6B-Base) are converted explicitly. float32 is the correctness reference
    on CPU; bfloat16 halves memory but is not automatically faster on CPUs without
    native bf16 instructions.
    """
    model_dir = Path(model_dir)
    config = Qwen3Config.from_hf_config(load_hf_config(model_dir), dtype=_dtype_name(dtype))
    ckpt = SafetensorsCheckpoint(model_dir)

    model = Qwen3ForCausalLM(config, device="meta", dtype=dtype)  # no memory, no init
    model.to_empty(device=device)  # uninitialised storage; every value is overwritten below
    model.model.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta, device=device)
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight  # to_empty may break sharing
    report = load_into_model(model, ckpt, strict=strict)
    model.eval()
    return model, report
