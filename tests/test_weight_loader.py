"""Weight loading tests with small synthetic safetensors checkpoints (CPU tier, no download)."""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch

safetensors_torch = pytest.importorskip("safetensors.torch", reason="safetensors not installed")

from qwen3_scratch.config import QWEN3_0_6B_BASE, QWEN3_TINY, TINY_QWEN3  # noqa: E402
from qwen3_scratch.model import Qwen3ForCausalLM  # noqa: E402
from qwen3_scratch.pretrained import load_pretrained  # noqa: E402
from qwen3_scratch.weight_loader import (  # noqa: E402
    CheckpointError,
    SafetensorsCheckpoint,
    build_name_map,
    check_checkpoint,
    expected_shapes,
    load_into_model,
)


def _hf_config_dict(c, **extra):
    d = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": c.vocab_size,
        "hidden_size": c.hidden_size,
        "intermediate_size": c.intermediate_size,
        "num_hidden_layers": c.num_hidden_layers,
        "num_attention_heads": c.num_attention_heads,
        "num_key_value_heads": c.num_key_value_heads,
        "head_dim": c.head_dim,
        "max_position_embeddings": c.max_position_embeddings,
        "rope_theta": c.rope_theta,
        "rms_norm_eps": c.rms_norm_eps,
        "tie_word_embeddings": c.tie_word_embeddings,
        "attention_bias": False,
        "hidden_act": "silu",
        "rope_scaling": None,
        "use_sliding_window": False,
        "torch_dtype": "bfloat16",
    }
    d.update(extra)
    return d


def _source(c=QWEN3_TINY, seed=0):
    torch.manual_seed(seed)
    m = Qwen3ForCausalLM(c)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.add_(0.1 * torch.randn(p.shape))
    return m


def _state(m, drop_lm_head=True, dtype=torch.bfloat16):
    sd = {k: v.detach().to(dtype).clone() for k, v in m.state_dict().items()}
    if drop_lm_head and m.config.tie_word_embeddings:
        sd.pop("lm_head.weight")  # like the official tied checkpoint
    return sd


def _write(tmp_path, c, sd, **cfg_extra):
    safetensors_torch.save_file(sd, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(_hf_config_dict(c, **cfg_extra)))
    return tmp_path


# ------------------------------------------------------------- mapping
def test_name_map_and_shapes_for_official_config() -> None:
    names = build_name_map(QWEN3_0_6B_BASE)
    assert len(names) == 1 + 28 * 11 + 1  # embed + 11 per layer + final norm; tied: no lm_head
    shapes = expected_shapes(QWEN3_0_6B_BASE)
    assert shapes["model.layers.0.self_attn.q_proj.weight"] == (2048, 1024)
    assert shapes["model.layers.0.self_attn.k_proj.weight"] == (1024, 1024)
    assert shapes["model.layers.27.self_attn.o_proj.weight"] == (1024, 2048)
    assert shapes["model.layers.0.self_attn.q_norm.weight"] == (128,)
    assert shapes["model.layers.0.mlp.down_proj.weight"] == (1024, 3072)
    total = sum(torch.Size(s).numel() for s in shapes.values())
    assert total == 596_049_920 == QWEN3_0_6B_BASE.estimate_num_parameters()


def test_name_map_matches_model_parameters() -> None:
    for c in (QWEN3_TINY, TINY_QWEN3, dataclasses.replace(QWEN3_TINY, tie_word_embeddings=False)):
        m = Qwen3ForCausalLM(c, device="meta")
        params = {n: tuple(p.shape) for n, p in m.named_parameters()}  # tied lm_head deduped
        shapes = expected_shapes(c)
        assert params == shapes


def test_unsupported_configs() -> None:
    with pytest.raises(NotImplementedError):
        build_name_map(dataclasses.replace(QWEN3_TINY, attention_bias=True))
    with pytest.raises(NotImplementedError):
        build_name_map(dataclasses.replace(QWEN3_TINY, qk_norm=False))


# ---------------------------------------------------------- happy path
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_load_pretrained_round_trip(tmp_path, dtype) -> None:
    src = _source()
    _write(tmp_path, QWEN3_TINY, _state(src))
    model, report = load_pretrained(tmp_path, dtype=dtype)
    assert report.ok and report.tied and not report.lm_head_in_checkpoint
    assert len(report.mapped) == len(build_name_map(QWEN3_TINY))
    assert report.num_parameters == QWEN3_TINY.estimate_num_parameters()
    assert report.num_checkpoint_values == report.num_parameters
    if dtype != torch.bfloat16:
        assert report.dtype_conversions == {f"bfloat16->{str(dtype)[6:]}": len(report.mapped)}
    else:
        assert report.dtype_conversions == {}
    assert model.weights_tied and not model.training
    assert all(p.dtype == dtype for p in model.parameters())
    for name, p in model.state_dict().items():  # values == bf16 checkpoint values
        assert torch.equal(p, src.state_dict()[name].to(torch.bfloat16).to(dtype)), name
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        ref = Qwen3ForCausalLM(QWEN3_TINY, dtype=dtype)
        ref.load_state_dict({k: v.to(dtype) for k, v in _state(src, drop_lm_head=False).items()})
        assert torch.equal(model(ids).logits, ref.eval()(ids).logits)
    assert model.model.rotary_emb.inv_freq.dtype == torch.float32  # RoPE rebuilt, not garbage


def test_untied_checkpoint(tmp_path) -> None:
    c = dataclasses.replace(QWEN3_TINY, tie_word_embeddings=False)
    src = _source(c)
    _write(tmp_path, c, _state(src))
    model, report = load_pretrained(tmp_path)
    assert not model.weights_tied and report.ok and "lm_head.weight" in report.mapped
    assert torch.equal(model.lm_head.weight, src.lm_head.weight.to(torch.bfloat16).float())


def test_tied_checkpoint_with_identical_lm_head_is_accepted(tmp_path) -> None:
    src = _source()
    _write(tmp_path, QWEN3_TINY, _state(src, drop_lm_head=False))
    model, report = load_pretrained(tmp_path)
    assert report.lm_head_in_checkpoint and report.ok and model.weights_tied


def test_tied_checkpoint_with_different_lm_head_is_rejected(tmp_path) -> None:
    sd = _state(_source(), drop_lm_head=False)
    sd["lm_head.weight"] = sd["lm_head.weight"] + 1
    _write(tmp_path, QWEN3_TINY, sd)
    with pytest.raises(CheckpointError, match="differs"):
        load_pretrained(tmp_path)


def test_sharded_checkpoint(tmp_path) -> None:
    sd = _state(_source())
    names = sorted(sd)
    half = len(names) // 2
    shards = {
        "model-00001-of-00002.safetensors": names[:half],
        "model-00002-of-00002.safetensors": names[half:],
    }
    weight_map = {}
    for fname, keys in shards.items():
        safetensors_torch.save_file({k: sd[k] for k in keys}, str(tmp_path / fname))
        weight_map.update(dict.fromkeys(keys, fname))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (tmp_path / "config.json").write_text(json.dumps(_hf_config_dict(QWEN3_TINY)))
    ckpt = SafetensorsCheckpoint(tmp_path)
    assert len(ckpt) == len(sd)
    _, report = load_pretrained(tmp_path)
    assert report.ok


# ------------------------------------------------------------ failures
def test_missing_key_is_reported_and_model_untouched(tmp_path) -> None:
    sd = _state(_source())
    del sd["model.layers.1.self_attn.k_norm.weight"]
    _write(tmp_path, QWEN3_TINY, sd)
    with pytest.raises(CheckpointError, match=r"missing 1: \['model.layers.1.self_attn.k_norm"):
        load_pretrained(tmp_path)


def test_unexpected_key_strict_and_non_strict(tmp_path) -> None:
    sd = _state(_source())
    sd["model.layers.0.self_attn.rotary_emb.inv_freq"] = torch.ones(16)
    _write(tmp_path, QWEN3_TINY, sd)
    with pytest.raises(CheckpointError, match="unexpected 1"):
        load_pretrained(tmp_path)
    _, report = load_pretrained(tmp_path, strict=False)
    assert report.unexpected == ["model.layers.0.self_attn.rotary_emb.inv_freq"]


def test_shape_mismatch_is_reported_even_non_strict(tmp_path) -> None:
    sd = _state(_source())
    sd["model.layers.0.mlp.up_proj.weight"] = torch.zeros(10, 128, dtype=torch.bfloat16)
    _write(tmp_path, QWEN3_TINY, sd)
    with pytest.raises(CheckpointError, match=r"up_proj.weight: checkpoint \(10, 128\)"):
        load_pretrained(tmp_path, strict=False)


def test_config_mismatch_is_detected(tmp_path) -> None:
    """A checkpoint from a different model size fails on shapes, not silently."""
    _write(tmp_path, QWEN3_TINY, _state(_source()), hidden_size=64, head_dim=16)
    with pytest.raises(CheckpointError, match="shape mismatch"):
        load_pretrained(tmp_path)


def test_check_checkpoint_reads_no_data() -> None:
    sd = _state(_source())
    report = check_checkpoint(sd, QWEN3_TINY)
    assert report.ok and not report.mapped
    with pytest.raises(CheckpointError):
        check_checkpoint([("a", torch.zeros(1))], QWEN3_TINY)


def test_non_float_tensor_rejected() -> None:
    sd = _state(_source())
    sd["model.norm.weight"] = torch.ones(QWEN3_TINY.hidden_size, dtype=torch.int32)
    with pytest.raises(CheckpointError, match="non-floating"):
        load_into_model(Qwen3ForCausalLM(QWEN3_TINY), sd)


def test_untied_model_parameter_assigned_twice_detected() -> None:
    """If two names pointed at the same parameter, the second copy would be lost."""
    c = dataclasses.replace(QWEN3_TINY, tie_word_embeddings=False)
    m = Qwen3ForCausalLM(c)
    m.lm_head.weight = m.model.embed_tokens.weight  # accidental sharing in an untied model
    with pytest.raises(CheckpointError, match="twice"):
        load_into_model(m, _state(_source(c)))


def test_pickle_checkpoints_refused(tmp_path) -> None:
    torch.save({"x": torch.zeros(1)}, tmp_path / "pytorch_model.bin")
    with pytest.raises(CheckpointError, match="refusing"):
        SafetensorsCheckpoint(tmp_path)
    with pytest.raises(CheckpointError):
        SafetensorsCheckpoint(tmp_path / "pytorch_model.bin")


def test_duplicate_tensor_across_shards(tmp_path) -> None:
    for f in ("a.safetensors", "b.safetensors"):
        safetensors_torch.save_file({"model.norm.weight": torch.ones(2)}, str(tmp_path / f))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.norm.weight": "a.safetensors", "x": "b.safetensors"}})
    )
    with pytest.raises(CheckpointError, match="more than one shard"):
        SafetensorsCheckpoint(tmp_path)


def test_unsupported_hf_config_rejected(tmp_path) -> None:
    _write(tmp_path, QWEN3_TINY, _state(_source()), rope_scaling={"type": "yarn", "factor": 4})
    with pytest.raises(NotImplementedError, match="RoPE scaling"):
        load_pretrained(tmp_path)
