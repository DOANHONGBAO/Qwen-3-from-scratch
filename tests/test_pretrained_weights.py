"""Tier-3 tests on the REAL Qwen3-0.6B-Base files (never downloads; auto-skipped).

    export QWEN3_WEIGHTS_DIR=$(python -c \
        "from qwen3_scratch.pretrained import download_checkpoint as d; print(d())")
    python -m pytest -m weights            # header/tokenizer checks, < 1 GiB RAM
    python -m pytest -m "weights and slow" # also loads the model in float32 (~4 GiB peak RSS)

The full numerical comparison with Hugging Face is scripts/verify_pretrained.py.
"""

from __future__ import annotations

import dataclasses
import unicodedata

import pytest
import torch

from qwen3_scratch.config import QWEN3_0_6B_BASE, Qwen3Config

pytestmark = pytest.mark.weights

N_PARAMS = 596_049_920


def test_config_is_the_official_one(weights_dir) -> None:
    cfg = Qwen3Config.from_json(weights_dir / "config.json")
    assert cfg == dataclasses.replace(QWEN3_0_6B_BASE, dtype="float32")


def test_checkpoint_header_and_mapping_coverage(weights_dir) -> None:
    pytest.importorskip("safetensors")
    from qwen3_scratch.weight_loader import SafetensorsCheckpoint, build_name_map, check_checkpoint

    ckpt = SafetensorsCheckpoint(weights_dir)
    assert len(ckpt) == len(build_name_map(QWEN3_0_6B_BASE)) == 310
    assert "lm_head.weight" not in ckpt  # tied: the head is the embedding matrix
    assert {ckpt.dtype_of(n) for n in ckpt} == {"BF16"}
    assert sum(torch.Size(ckpt.shape_of(n)).numel() for n in ckpt) == N_PARAMS
    report = check_checkpoint(ckpt, QWEN3_0_6B_BASE)
    assert report.ok, report.summary()
    norm = ckpt["model.layers.0.self_attn.q_norm.weight"]
    assert norm.shape == (128,) and not torch.all(norm == 1)  # trained, not default init


def test_official_tokenizer_facts(weights_dir) -> None:
    pytest.importorskip("tokenizers")
    from qwen3_scratch.tokenizer import Qwen3Tokenizer

    tok = Qwen3Tokenizer.from_pretrained(weights_dir)
    assert tok.vocab_size == 151_669 < QWEN3_0_6B_BASE.vocab_size
    assert tok.eos_token_id == tok.pad_token_id == 151_643
    assert tok.bos_token_id is None and not tok.add_bos_token
    assert tok.token_to_id("<|im_start|>") == 151_644 and tok.token_to_id("<|im_end|>") == 151_645
    assert tok.encode("<|endoftext|>") == [151_643]
    assert len(tok.special_tokens) == 14
    # declared only in tokenizer_config.json (tokenizer.json has 22 added tokens)
    assert tok.registered_from_config == [
        "<tool_response>",
        "</tool_response>",
        "<think>",
        "</think>",
    ]
    assert tok.token_to_id("<think>") == 151_667 and tok.token_to_id("</think>") == 151_668


@pytest.mark.parametrize(
    "text",
    [
        "Artificial intelligence is",
        "Trí tuệ nhân tạo (AI) đang thay đổi thế giới.",
        "Giá: 1.234.567,89 ₫ — 50% off!!",
        "東京 café naïve 🙂 Ελληνικά",
        "",
        "def f(x):\n\treturn x  # tab",
    ],
)
def test_official_tokenizer_round_trip(weights_dir, text) -> None:
    pytest.importorskip("tokenizers")
    from qwen3_scratch.tokenizer import Qwen3Tokenizer

    tok = Qwen3Tokenizer.from_pretrained(weights_dir)
    assert tok.decode(tok.encode(text)) == unicodedata.normalize("NFC", text)


def test_tokenizer_matches_transformers(weights_dir) -> None:
    transformers = pytest.importorskip("transformers")
    from qwen3_scratch.tokenizer import Qwen3Tokenizer

    ours = Qwen3Tokenizer.from_pretrained(weights_dir)
    ref = transformers.AutoTokenizer.from_pretrained(weights_dir)
    texts = [
        "Hello world",
        "Xin chào Việt Nam",
        "x<|endoftext|>y",
        "  spaces  ",
        "3.14159",
        "<think>reasoning</think> answer",
        "<tool_response>{}</tool_response>",
    ]
    for text in texts:
        assert ours.encode(text) == ref(text, add_special_tokens=True).input_ids, text


@pytest.mark.slow
def test_float32_model_loads_and_is_deterministic(weights_dir) -> None:
    from qwen3_scratch.pretrained import load_pretrained
    from qwen3_scratch.tokenizer import Qwen3Tokenizer

    model, report = load_pretrained(weights_dir, dtype=torch.float32)
    assert report.ok and report.num_parameters == N_PARAMS and model.weights_tied
    tok = Qwen3Tokenizer.from_pretrained(weights_dir)
    ids = torch.tensor([tok.encode("The quick brown fox jumps over the lazy dog")])
    with torch.inference_mode():
        a = model(ids).logits
        b = model(ids).logits
    assert torch.isfinite(a).all() and torch.equal(a, b)
