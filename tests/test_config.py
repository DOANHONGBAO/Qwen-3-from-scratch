"""Tests for the framework-independent Qwen3Config (tier 1: local CPU)."""

from __future__ import annotations

import dataclasses
import json

import pytest

from qwen3_scratch.config import (
    PRESETS,
    QWEN3_0_6B_BASE,
    RASCHKA_KEY_MAP,
    TINY_QWEN3,
    Qwen3Config,
)

HF_CONFIG_PATH = "configs/hf/Qwen3-0.6B-Base.config.json"


@pytest.fixture()
def hf_config(repo_root) -> dict:
    return json.loads((repo_root / HF_CONFIG_PATH).read_text())


def test_official_config_maps_to_preset(hf_config) -> None:
    cfg = Qwen3Config.from_hf_config(hf_config, dtype="bfloat16")
    assert cfg == QWEN3_0_6B_BASE


def test_qwen3_specific_properties_of_0_6b() -> None:
    cfg = QWEN3_0_6B_BASE
    # head_dim is independent of hidden_size / num_heads (1024 / 16 = 64).
    assert cfg.head_dim == 128 != cfg.hidden_size // cfg.num_attention_heads
    assert cfg.q_proj_dim == 2048  # wider than hidden_size
    assert cfg.kv_proj_dim == 1024
    assert cfg.num_queries_per_kv == 2
    assert cfg.tie_word_embeddings and cfg.qk_norm and not cfg.attention_bias


def test_parameter_count_matches_model_card() -> None:
    # Model card: 0.6B parameters in total, 0.44B non-embedding.
    total = QWEN3_0_6B_BASE.estimate_num_parameters()
    non_emb = QWEN3_0_6B_BASE.estimate_num_parameters(include_embeddings=False)
    assert total == 596_049_920
    assert round(non_emb / 1e9, 2) == 0.44
    assert round(total / 1e9, 1) == 0.6


def test_untied_embeddings_add_lm_head() -> None:
    untied = dataclasses.replace(TINY_QWEN3, tie_word_embeddings=False)
    diff = untied.estimate_num_parameters() - TINY_QWEN3.estimate_num_parameters()
    assert diff == TINY_QWEN3.vocab_size * TINY_QWEN3.hidden_size


def test_tiny_preserves_qwen3_structure() -> None:
    cfg = TINY_QWEN3
    assert cfg.num_queries_per_kv > 1  # real GQA, not MHA
    assert cfg.head_dim != cfg.hidden_size // cfg.num_attention_heads
    assert cfg.qk_norm and cfg.tie_word_embeddings


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_json_roundtrip(tmp_path, name: str) -> None:
    path = tmp_path / f"{name}.json"
    PRESETS[name].to_json(path)
    assert Qwen3Config.from_json(path) == PRESETS[name]


@pytest.mark.parametrize("name", ["qwen3_0.6b_base", "tiny_qwen3"])
def test_committed_config_files_match_presets(repo_root, name: str) -> None:
    loaded = Qwen3Config.from_json(repo_root / "configs" / f"{name}.json")
    expected = {"qwen3_0.6b_base": QWEN3_0_6B_BASE, "tiny_qwen3": TINY_QWEN3}[name]
    assert loaded == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_key_value_heads", 3),  # 4 % 3 != 0
        ("head_dim", 31),  # odd -> RoPE impossible
        ("hidden_size", 0),
        ("vocab_size", -1),
        ("num_hidden_layers", 1.5),
        ("num_attention_heads", True),
        ("rope_theta", 0.0),
        ("rms_norm_eps", 0.0),
        ("dtype", "int8"),
    ],
)
def test_invalid_values_raise(field: str, value) -> None:
    with pytest.raises(ValueError):
        dataclasses.replace(TINY_QWEN3, **{field: value})


def test_unknown_keys_rejected() -> None:
    data = TINY_QWEN3.to_dict() | {"emb_dim": 64}
    with pytest.raises(ValueError, match="Unknown config keys"):
        Qwen3Config.from_dict(data)


@pytest.mark.parametrize(
    ("key", "value", "exc"),
    [
        ("model_type", "llama", ValueError),
        ("hidden_act", "gelu", NotImplementedError),
        ("use_sliding_window", True, NotImplementedError),
        ("rope_scaling", {"type": "yarn", "factor": 4.0}, NotImplementedError),
    ],
)
def test_unsupported_hf_features_raise(hf_config, key, value, exc) -> None:
    with pytest.raises(exc):
        Qwen3Config.from_hf_config(hf_config | {key: value})


def test_raschka_key_map_covers_architecture_fields() -> None:
    field_names = {f.name for f in dataclasses.fields(Qwen3Config)}
    assert set(RASCHKA_KEY_MAP) <= field_names
