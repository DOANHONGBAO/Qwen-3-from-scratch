"""Tokenizer adapter tests on a toy Qwen-like tokenizer (CPU tier, no download).

The real Qwen3 tokenizer is tested in tests/test_pretrained_weights.py (weights tier).
"""

from __future__ import annotations

import unicodedata

import pytest
import torch

tokenizer_mod = pytest.importorskip("qwen3_scratch.tokenizer")
pytest.importorskip("tokenizers")
Qwen3Tokenizer = tokenizer_mod.Qwen3Tokenizer
from qwen3_scratch.synthetic import TOY_CORPUS  # noqa: E402

TEXTS = [
    "Artificial intelligence is",
    "Trí tuệ nhân tạo là gì?",  # Vietnamese with combining marks
    "1234567890 3.14 -42",
    "café 東京 🙂 Ελληνικά",
    "!?.,;:()[]{}\"'",
    "  leading and trailing spaces  ",
    "line1\nline2\ttab",
    "",
]


@pytest.fixture(scope="module")
def tok(toy_tokenizer_dir):
    return Qwen3Tokenizer.from_pretrained(toy_tokenizer_dir)


@pytest.mark.parametrize("text", TEXTS + TOY_CORPUS)
def test_round_trip(tok, text) -> None:
    ids = tok.encode(text)
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == unicodedata.normalize("NFC", text)


def test_empty_string(tok) -> None:
    assert tok.encode("") == [] and tok.decode([]) == ""


def test_nfc_normalisation_is_applied(tok) -> None:
    decomposed = unicodedata.normalize("NFD", "Việt Nam")
    assert decomposed != "Việt Nam"
    assert tok.encode(decomposed) == tok.encode("Việt Nam")
    assert tok.decode(tok.encode(decomposed)) == "Việt Nam"


def test_special_tokens_and_no_bos(tok) -> None:
    eos = tok.eos_token_id
    assert eos is not None and tok.pad_token_id == eos and tok.bos_token_id is None
    assert not tok.add_bos_token
    assert tok.encode("<|endoftext|>") == [eos]
    ids = tok.encode("Hello<|endoftext|>")
    assert ids[-1] == eos
    assert tok.decode(ids, skip_special_tokens=True) == "Hello"
    assert tok.decode(ids) == "Hello<|endoftext|>"
    assert set(tok.special_tokens) == {"<|endoftext|>", "<|im_start|>", "<|im_end|>"}


def test_config_only_added_tokens_are_registered(tok, toy_tokenizer_dir) -> None:
    """tokenizer_config.json may declare added tokens that tokenizer.json lacks (Qwen3!)."""
    import json

    from tokenizers import Tokenizer

    raw = Tokenizer.from_file(str(toy_tokenizer_dir / "tokenizer.json"))
    assert raw.token_to_id("<think>") is None  # absent from tokenizer.json
    cfg = json.loads((toy_tokenizer_dir / "tokenizer_config.json").read_text())
    declared = {v["content"]: int(k) for k, v in cfg["added_tokens_decoder"].items()}
    assert tok.registered_from_config == ["<think>", "</think>"]
    assert tok.vocab_size == raw.get_vocab_size(with_added_tokens=True) + 2
    for content in ("<think>", "</think>"):
        assert tok.token_to_id(content) == declared[content]
    ids = tok.encode("a<think>b</think>")
    assert declared["<think>"] in ids and declared["</think>"] in ids
    assert tok.decode(ids) == "a<think>b</think>"
    assert "<think>" not in tok.special_tokens  # declared non-special


def test_inconsistent_config_ids_rejected(toy_tokenizer_dir, tmp_path) -> None:
    import json
    import shutil

    shutil.copy(toy_tokenizer_dir / "tokenizer.json", tmp_path / "tokenizer.json")
    cfg = json.loads((toy_tokenizer_dir / "tokenizer_config.json").read_text())
    entry = cfg["added_tokens_decoder"].pop(max(cfg["added_tokens_decoder"], key=int))
    cfg["added_tokens_decoder"]["99999"] = entry  # not contiguous
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="declares 99999"):
        Qwen3Tokenizer.from_pretrained(tmp_path)


def test_reject_special_token_text(tok) -> None:
    with pytest.raises(ValueError, match="special-token"):
        tok.encode("a <|im_start|> b", allow_special_tokens=False)
    assert tok.encode("plain", allow_special_tokens=False)


def test_batch_padding(tok) -> None:
    ids, mask = tok.encode_batch(["a", "a longer sentence"], padding=True, padding_side="left")
    assert len(ids[0]) == len(ids[1]) and mask[0][0] == 0 and mask[1] == [1] * len(ids[1])
    assert ids[0][0] == tok.pad_token_id
    ids_r, mask_r = tok.encode_batch(["a", "a longer sentence"], padding=True, padding_side="right")
    assert mask_r[0][-1] == 0 and ids_r[0][0] != tok.pad_token_id
    plain, _ = tok.encode_batch(["a", "bb"])
    assert plain == [tok.encode("a"), tok.encode("bb")]
    with pytest.raises(ValueError):
        tok.encode_batch(["a"], padding=True, padding_side="middle")


def test_decode_accepts_tensor_and_validates(tok) -> None:
    ids = tok.encode("Xin chào")
    assert tok.decode(torch.tensor(ids)) == "Xin chào"
    with pytest.raises(ValueError, match="outside"):
        tok.decode([tok.vocab_size])
    with pytest.raises(TypeError):
        tok.decode([1.5])
    with pytest.raises(TypeError):
        tok.encode(123)


def test_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        Qwen3Tokenizer(tmp_path / "tokenizer.json")


def test_matches_transformers_fast_tokenizer(toy_tokenizer_dir, tok) -> None:
    transformers = pytest.importorskip("transformers")
    ref = transformers.AutoTokenizer.from_pretrained(toy_tokenizer_dir)
    for text in TEXTS + TOY_CORPUS + ["x<|endoftext|>y", "a<think>b</think>c"]:
        assert tok.encode(text) == ref.encode(text, add_special_tokens=False), text
        ids = tok.encode(text)
        assert tok.decode(ids) == ref.decode(ids, clean_up_tokenization_spaces=False)


def test_decode_can_skip_padded_vocab_rows(tok) -> None:
    ids = tok.encode("ok")
    assert tok.decode([*ids, tok.vocab_size + 5], skip_out_of_vocab=True) == "ok"
