"""Tests for Qwen3Model / Qwen3ForCausalLM (M1.6). CPU tier, random weights."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch.testing import assert_close

from qwen3_scratch.config import QWEN3_0_6B_BASE, QWEN3_TINY, TINY_QWEN3
from qwen3_scratch.layers import Qwen3DecoderLayer, RMSNorm, RotaryEmbedding
from qwen3_scratch.model import Qwen3ForCausalLM, Qwen3Model


def _model(cfg=QWEN3_TINY, seed=0, **overrides) -> Qwen3ForCausalLM:
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    torch.manual_seed(seed)
    m = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():  # non-trivial norm weights so every norm matters
        for name, p in m.named_parameters():
            if "norm" in name:
                p.add_(0.1 * torch.randn(p.shape))
    return m


def _ids(b, t, vocab=QWEN3_TINY.vocab_size, seed=0):
    return torch.randint(0, vocab, (b, t), generator=torch.Generator().manual_seed(seed))


# ------------------------------------------------- 3. construction / names
def test_structure_and_parameter_names() -> None:
    m = _model()
    assert isinstance(m.model, Qwen3Model)
    assert len(m.model.layers) == QWEN3_TINY.num_hidden_layers
    assert all(isinstance(layer, Qwen3DecoderLayer) for layer in m.model.layers)
    assert isinstance(m.model.norm, RMSNorm)
    assert isinstance(m.model.rotary_emb, RotaryEmbedding)
    assert m.lm_head.bias is None
    keys = set(m.state_dict())
    assert {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"} <= keys
    assert "model.layers.1.self_attn.q_norm.weight" in keys
    assert not any("rotary" in k or "inv_freq" in k for k in keys)  # buffers not saved


def test_no_absolute_position_embedding() -> None:
    m = _model()
    embeddings = [mod for mod in m.modules() if isinstance(mod, torch.nn.Embedding)]
    assert embeddings == [m.model.embed_tokens]


def test_init_follows_hf_scheme() -> None:
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(QWEN3_TINY)
    for name, p in m.named_parameters():
        if "norm" in name:
            assert torch.equal(p, torch.ones_like(p)), name
        else:
            assert abs(p.std().item() - 0.02) < 0.004, name
            assert abs(p.mean().item()) < 0.004, name


# --------------------------------------------- 10, 11, 12. tying and counts
def test_weight_tying_shares_one_parameter() -> None:
    m = _model()
    assert m.weights_tied
    assert m.lm_head.weight is m.model.embed_tokens.weight
    assert m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr()
    names = [n for n, _ in m.named_parameters()]
    assert "lm_head.weight" not in names  # registered once (under the embedding)
    with torch.no_grad():
        m.model.embed_tokens.weight[3, 0] = 42.0
    assert m.lm_head.weight[3, 0] == 42.0
    m(_ids(1, 4), labels=_ids(1, 4)).loss.backward()
    assert m.lm_head.weight.grad is m.model.embed_tokens.weight.grad  # one gradient


def test_untied_weights_are_independent() -> None:
    m = _model(tie_word_embeddings=False)
    assert not m.weights_tied
    assert m.lm_head.weight.data_ptr() != m.model.embed_tokens.weight.data_ptr()
    before = m.lm_head.weight.clone()
    with torch.no_grad():
        m.model.embed_tokens.weight.zero_()
    assert torch.equal(m.lm_head.weight, before)


def test_parameter_counts() -> None:
    tied, untied = _model(), _model(tie_word_embeddings=False)
    v, h = QWEN3_TINY.vocab_size, QWEN3_TINY.hidden_size
    assert tied.num_parameters() == QWEN3_TINY.estimate_num_parameters() == 426_752
    assert untied.num_parameters() - tied.num_parameters() == v * h == 32_768
    assert tied.num_parameters(trainable_only=True) == tied.num_parameters()
    for p in tied.model.embed_tokens.parameters():
        p.requires_grad_(False)
    assert tied.num_parameters(trainable_only=True) == 426_752 - v * h


def test_qwen3_0_6b_parameter_count_on_meta_device() -> None:
    # Never allocated: the meta device only records shapes.
    m = Qwen3ForCausalLM(QWEN3_0_6B_BASE, device="meta")
    assert m.num_parameters() == 596_049_920 == QWEN3_0_6B_BASE.estimate_num_parameters()
    assert m.weights_tied


# ----------------------------------------------- 5, 6, 7, 8, 9. shapes
@pytest.mark.parametrize("cfg", [QWEN3_TINY, TINY_QWEN3], ids=["qwen3-tiny", "tiny(D!=H/N)"])
@pytest.mark.parametrize(("b", "t"), [(1, 1), (1, 7), (3, 5), (2, 32)])
def test_output_shapes(cfg, b, t) -> None:
    m = _model(cfg)
    out = m(_ids(b, t, cfg.vocab_size), output_hidden_states=True)
    assert out.logits.shape == (b, t, cfg.vocab_size)
    assert out.last_hidden_state.shape == (b, t, cfg.hidden_size)
    assert len(out.hidden_states) == cfg.num_hidden_layers + 1
    assert all(h.shape == (b, t, cfg.hidden_size) for h in out.hidden_states)
    assert out.loss is None and torch.isfinite(out.logits).all()


def test_forward_matches_manual_composition() -> None:
    m = _model()
    ids = _ids(2, 6)
    out = m(ids, output_hidden_states=True)
    h = m.model.embed_tokens(ids)
    assert torch.equal(out.hidden_states[0], h)
    cos, sin = m.model.rotary_emb(torch.arange(6)[None])
    for i, layer in enumerate(m.model.layers):
        h = layer(h, cos, sin)
        assert torch.equal(out.hidden_states[i + 1], h)
    h = m.model.norm(h)
    assert torch.equal(out.last_hidden_state, h)
    assert torch.equal(out.logits, h @ m.model.embed_tokens.weight.T)  # tied LM head


def test_inputs_embeds_equivalent_to_ids() -> None:
    m = _model()
    ids = _ids(2, 5)
    a = m(ids).logits
    b = m(inputs_embeds=m.model.embed_tokens(ids)).logits
    assert torch.equal(a, b)


def test_batch_items_are_independent() -> None:
    m = _model()
    ids = _ids(3, 6)
    full = m(ids).logits
    for i in range(3):
        assert_close(full[i : i + 1], m(ids[i : i + 1]).logits, rtol=1e-5, atol=1e-5)


# ------------------------------------------------------ 13. determinism
def test_deterministic() -> None:
    ids = _ids(2, 8)
    a, b = _model(seed=3), _model(seed=3)
    assert torch.equal(a(ids).logits, a(ids).logits)
    assert torch.equal(a(ids).logits, b(ids).logits)
    assert not torch.equal(a(ids).logits, _model(seed=4)(ids).logits)


# ------------------------------------------------------ 14. gradients
def test_gradients_reach_every_parameter() -> None:
    m = _model()
    ids = _ids(2, 6)
    m(ids, labels=ids).loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


# ------------------------------- 19, 20, 21. masking, leakage, positions
def test_no_future_leakage_full_model() -> None:
    """Mandatory: identical prefixes, different futures -> identical prefix logits."""
    m = _model()
    prefix = _ids(1, 5, seed=1)
    a = torch.cat([prefix, _ids(1, 4, seed=2)], dim=1)
    b = torch.cat([prefix, _ids(1, 4, seed=3)], dim=1)
    assert not torch.equal(a[:, 5:], b[:, 5:])
    logits = m(torch.cat([a, b])).logits
    assert_close(logits[0, :5], logits[1, :5], rtol=1e-6, atol=1e-6)
    assert not torch.allclose(logits[0, 5:], logits[1, 5:])


def test_no_future_leakage_in_gradients() -> None:
    m = _model()
    embeds = m.model.embed_tokens(_ids(1, 6)).detach().requires_grad_()
    m(inputs_embeds=embeds).logits[0, 2].sum().backward()
    assert embeds.grad[0, 3:].abs().max() == 0
    assert embeds.grad[0, :3].abs().sum() > 0


def test_changing_one_token_affects_later_positions_only() -> None:
    m = _model()
    ids = _ids(1, 8, seed=5)
    changed = ids.clone()
    changed[0, 3] = (ids[0, 3] + 1) % QWEN3_TINY.vocab_size
    a, b = m(ids).logits, m(changed).logits
    assert torch.equal(a[0, :3], b[0, :3])  # earlier positions: bitwise unchanged
    for t in range(3, 8):  # this and every later prediction changes
        assert not torch.allclose(a[0, t], b[0, t])


def test_all_ones_padding_mask_equals_no_mask() -> None:
    m = _model()
    ids = _ids(2, 6)
    assert torch.equal(
        m(ids).logits, m(ids, attention_mask=torch.ones(2, 6, dtype=torch.long)).logits
    )


def test_left_padding_with_positions_matches_unpadded() -> None:
    # A left-padded sequence with position_ids = cumsum(mask) - 1 must give the same
    # logits on its real tokens as the unpadded sequence on its own.
    m = _model()
    real = _ids(1, 5, seed=6)
    padded = torch.cat([torch.zeros(1, 3, dtype=torch.long), real], dim=1)  # pad id 0 (arbitrary)
    mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    out = m(padded, attention_mask=mask, position_ids=pos).logits
    ref = m(real).logits
    assert_close(out[0, 3:], ref[0], rtol=1e-5, atol=1e-5)
    assert torch.isfinite(out).all()  # fully-masked pad rows give finite values


def test_pad_tokens_are_not_masked_automatically() -> None:
    m = _model()
    ids = torch.tensor([[0, 0, 5, 6]])
    assert not torch.allclose(m(ids).logits[0, 2:], m(ids[:, 2:]).logits[0])


def test_relative_positions_shift_invariance() -> None:
    # RoPE: shifting all positions by the same offset changes logits only through
    # float32 table rounding.
    m = _model()
    ids = _ids(1, 6)
    a = m(ids).logits
    b = m(ids, position_ids=torch.arange(50, 56)).logits
    assert_close(a, b, rtol=1e-3, atol=1e-3)


def test_same_rope_tables_for_all_layers() -> None:
    m = _model()
    seen = []
    for layer in m.model.layers:
        layer.register_forward_pre_hook(lambda _m, args: seen.append(args[1]))
    m(_ids(1, 4))
    assert len(seen) == 2 and all(s is seen[0] for s in seen)  # one cos table, shared


# ---------------------------------------------------- 26. invalid input
@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"input_ids": torch.tensor([1, 2, 3])}, ValueError),  # [T] instead of [B, T]
        ({"input_ids": torch.tensor([[0.0, 1.0]])}, TypeError),
        ({"input_ids": torch.tensor([[1, 256]])}, ValueError),  # out of vocab
        ({"input_ids": torch.tensor([[-1, 2]])}, ValueError),
        ({"input_ids": torch.zeros(1, 0, dtype=torch.long)}, ValueError),
        ({"input_ids": torch.zeros(1, 129, dtype=torch.long)}, ValueError),  # > max positions
        ({}, ValueError),  # neither ids nor embeds
        (
            {
                "input_ids": torch.zeros(1, 3, dtype=torch.long),
                "position_ids": torch.tensor([0, 1]),
            },
            ValueError,
        ),
        (
            {
                "input_ids": torch.zeros(1, 3, dtype=torch.long),
                "position_ids": torch.tensor([126, 127, 128]),
            },
            ValueError,
        ),
        (
            {"input_ids": torch.zeros(1, 3, dtype=torch.long), "attention_mask": torch.zeros(1, 3)},
            TypeError,
        ),
        (
            {
                "input_ids": torch.zeros(1, 3, dtype=torch.long),
                "attention_mask": torch.tensor([[1, 2, 1]]),
            },
            ValueError,
        ),
        (
            {
                "input_ids": torch.zeros(1, 3, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            },
            ValueError,
        ),
        ({"inputs_embeds": torch.zeros(1, 3, 7)}, ValueError),
    ],
)
def test_invalid_inputs(kwargs, exc) -> None:
    with pytest.raises(exc):
        _model()(**kwargs)


def test_attention_dropout_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        Qwen3ForCausalLM(dataclasses.replace(QWEN3_TINY, attention_dropout=0.1))
