"""End-to-end parity of Qwen3ForCausalLM with Hugging Face ``Qwen3ForCausalLM``.

Both models are built from configuration only (no downloads). HF's random init
is the starting point; its ``state_dict`` is then copied into ours with
``load_state_dict(strict=True)`` - the same seed alone would NOT give identical
weights in two implementations. HF runs with ``attn_implementation="eager"`` in
``eval()`` mode (no dropout). Intermediates come from forward hooks on
``model.embed_tokens``, each ``model.layers[i]`` and ``model.norm``.
Requires the ``hf`` extra.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

modeling = pytest.importorskip(
    "transformers.models.qwen3.modeling_qwen3", reason="transformers not installed (hf extra)"
)
transformers = pytest.importorskip("transformers")

from qwen3_scratch.config import QWEN3_TINY, TINY_QWEN3, Qwen3Config  # noqa: E402
from qwen3_scratch.model import Qwen3ForCausalLM  # noqa: E402

CONFIGS = {
    "qwen3-tiny": QWEN3_TINY,
    "tiny-D!=H/N": TINY_QWEN3,
    "untied": dataclasses.replace(QWEN3_TINY, tie_word_embeddings=False),
}


def _hf_config(c: Qwen3Config):
    return transformers.Qwen3Config(
        vocab_size=c.vocab_size,
        hidden_size=c.hidden_size,
        intermediate_size=c.intermediate_size,
        num_hidden_layers=c.num_hidden_layers,
        num_attention_heads=c.num_attention_heads,
        num_key_value_heads=c.num_key_value_heads,
        head_dim=c.head_dim,
        max_position_embeddings=c.max_position_embeddings,
        rope_theta=c.rope_theta,
        rms_norm_eps=c.rms_norm_eps,
        attention_bias=c.attention_bias,
        attention_dropout=c.attention_dropout,
        tie_word_embeddings=c.tie_word_embeddings,
        initializer_range=c.initializer_range,
        attn_implementation="eager",
    )


def _pair(c: Qwen3Config, seed: int = 0):
    torch.manual_seed(seed)
    hf = modeling.Qwen3ForCausalLM(_hf_config(c)).eval()
    with torch.no_grad():  # make norm weights non-trivial so they are really compared
        for name, p in hf.named_parameters():
            if "norm" in name:
                p.add_(0.1 * torch.randn(p.shape))
    ours = Qwen3ForCausalLM(c).eval()
    result = ours.load_state_dict(hf.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    return hf, ours


def _ids(c, b=2, t=11, seed=1):
    return torch.randint(0, c.vocab_size, (b, t), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("name", CONFIGS)
def test_parameter_coverage_shapes_counts_and_tying(name) -> None:
    c = CONFIGS[name]
    hf, ours = _pair(c)
    hf_params = {n: p.shape for n, p in hf.named_parameters(remove_duplicate=False)}
    our_params = {n: p.shape for n, p in ours.named_parameters(remove_duplicate=False)}
    assert hf_params == our_params
    for n, p in hf.state_dict().items():
        assert torch.equal(ours.state_dict()[n], p), n
    n_hf = sum(p.numel() for p in hf.parameters())
    assert ours.num_parameters() == n_hf == c.estimate_num_parameters()
    hf_tied = hf.lm_head.weight is hf.model.embed_tokens.weight
    assert ours.weights_tied == hf_tied == c.tie_word_embeddings


def _hf_trace(hf, ids, **kwargs):
    seen = {"layers": []}
    hooks = [
        hf.model.embed_tokens.register_forward_hook(lambda m, i, o: seen.__setitem__("emb", o))
    ]
    for layer in hf.model.layers:
        hooks.append(
            layer.register_forward_hook(
                lambda m, i, o: seen["layers"].append(o[0] if isinstance(o, tuple) else o)
            )
        )
    hooks.append(hf.model.norm.register_forward_hook(lambda m, i, o: seen.__setitem__("norm", o)))
    out = hf(input_ids=ids, **kwargs)
    for h in hooks:
        h.remove()
    return out, seen


@pytest.mark.parametrize("name", CONFIGS)
def test_forward_intermediates_logits_and_loss_bit_identical(name) -> None:
    c = CONFIGS[name]
    hf, ours = _pair(c)
    ids = _ids(c)
    labels = ids.clone()
    labels[0, :3] = -100  # some ignored targets
    with torch.no_grad():
        ref, seen = _hf_trace(hf, ids, labels=labels)
        out = ours(ids, labels=labels, output_hidden_states=True)
    assert torch.equal(out.hidden_states[0], seen["emb"])
    assert len(seen["layers"]) == c.num_hidden_layers
    for i, h in enumerate(seen["layers"]):
        assert torch.equal(out.hidden_states[i + 1], h), f"layer {i}"
    assert torch.equal(out.last_hidden_state, seen["norm"])
    assert torch.equal(out.logits, ref.logits)
    assert torch.equal(out.loss, ref.loss)


def test_gradients_bit_identical() -> None:
    hf, ours = _pair(QWEN3_TINY, seed=3)
    ids = _ids(QWEN3_TINY, seed=4)
    hf(input_ids=ids, labels=ids).loss.backward()
    ours(ids, labels=ids).loss.backward()
    hf_named = dict(hf.named_parameters())
    for name, p in ours.named_parameters():
        assert torch.equal(p.grad, hf_named[name].grad), name


def test_padded_batch_matches_on_real_tokens() -> None:
    hf, ours = _pair(QWEN3_TINY, seed=5)
    ids = _ids(QWEN3_TINY, b=2, t=8, seed=6)
    mask = torch.tensor([[1] * 8, [0, 0, 0, 1, 1, 1, 1, 1]])
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        ref = hf(input_ids=ids, attention_mask=mask, position_ids=pos).logits
        out = ours(ids, attention_mask=mask, position_ids=pos).logits
    real = mask.bool()
    assert torch.equal(out[real], ref[real])
    # Fully masked pad rows differ by design (ours: zero attention, HF: uniform).


def test_bfloat16_forward_bit_identical() -> None:
    hf, ours = _pair(QWEN3_TINY, seed=7)
    inv_freq_fp32 = hf.model.rotary_emb.inv_freq.clone()
    hf, ours = hf.to(torch.bfloat16), ours.to(torch.bfloat16)
    # HF quirk: Module.to(bfloat16) also rounds HF's inv_freq buffer to bfloat16
    # (a model created directly in bfloat16 keeps it float32). Ours stays float32
    # (regression fix, D-027); restore HF's buffer to compare like with like.
    assert hf.model.rotary_emb.inv_freq.dtype == torch.bfloat16
    assert ours.model.rotary_emb.inv_freq.dtype == torch.float32
    hf.model.rotary_emb.inv_freq = inv_freq_fp32
    ids = _ids(QWEN3_TINY, seed=8)
    with torch.no_grad():
        assert torch.equal(ours(ids).logits, hf(input_ids=ids).logits)


def test_loss_bit_identical_on_larger_batch() -> None:
    # Same row order as HF's loss (labels padded with -100, all T positions flattened).
    hf, ours = _pair(QWEN3_TINY, seed=9)
    ids = _ids(QWEN3_TINY, b=4, t=64, seed=10)
    with torch.no_grad():
        assert torch.equal(ours(ids, labels=ids).loss, hf(input_ids=ids, labels=ids).loss)
