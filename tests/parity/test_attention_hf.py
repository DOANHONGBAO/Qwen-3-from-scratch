"""Parity of GroupedQueryAttention with Hugging Face ``Qwen3Attention`` (eager).

How the reference is driven (no downloads, small random config):

* ``attn_implementation="eager"`` so attention probabilities are materialised;
  the module is in ``eval()`` (dropout off; Qwen3's ``attention_dropout`` is 0 anyway).
* HF's attention module does NOT build a causal mask itself: called with
  ``attention_mask=None`` it attends to the future (checked below). We pass the
  additive causal mask ``[B, 1, T, T]`` (0 / finfo.min) that ``Qwen3Model`` would build.
* ``position_embeddings`` come from HF ``Qwen3RotaryEmbedding`` for explicit
  ``position_ids`` (with a non-zero offset).
* Intermediates: projection and norm outputs and the ``o_proj`` input via forward
  hooks; probabilities from HF's returned ``attn_weights``. RoPE outputs, grouped
  KV and logits are not exposed by HF, so they are recomputed from the hooked
  tensors with HF's own ``apply_rotary_pos_emb`` / ``repeat_kv`` functions and
  ``Qwen3Attention.scaling`` (documented equivalent reference).
"""

from __future__ import annotations

import pytest
import torch

modeling = pytest.importorskip(
    "transformers.models.qwen3.modeling_qwen3", reason="transformers not installed (hf extra)"
)
transformers = pytest.importorskip("transformers")

from qwen3_scratch.layers import (  # noqa: E402
    GroupedQueryAttention,
    bool_to_additive_mask,
    causal_mask,
    make_position_ids,
)

HIDDEN, HEAD_DIM, T, B, OFFSET = 64, 32, 7, 2, 3
CONFIGS = [(4, 2), (4, 4), (4, 1), (16, 8)]
IDS = ["GQA", "MHA", "MQA", "0.6B-heads"]


def _setup(nq: int, nkv: int, dtype: torch.dtype, seed: int = 0):
    cfg = transformers.Qwen3Config(
        hidden_size=HIDDEN,
        num_attention_heads=nq,
        num_key_value_heads=nkv,
        head_dim=HEAD_DIM,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        vocab_size=32,
        attn_implementation="eager",
    )
    torch.manual_seed(seed)
    hf = modeling.Qwen3Attention(cfg, layer_idx=0).to(dtype).eval()
    with torch.no_grad():
        hf.q_norm.weight.copy_(1 + 0.3 * torch.randn(HEAD_DIM))
        hf.k_norm.weight.copy_(1 - 0.3 * torch.randn(HEAD_DIM))
    ours = GroupedQueryAttention(HIDDEN, nq, nkv, HEAD_DIM, dtype=dtype)
    ours.load_state_dict(hf.state_dict(), strict=True)  # explicit 1:1 key mapping
    x = torch.randn(B, T, HIDDEN, generator=torch.Generator().manual_seed(seed + 1)).to(dtype)
    pos = make_position_ids(T, offset=OFFSET, batch_size=B)
    cos, sin = modeling.Qwen3RotaryEmbedding(cfg)(x, pos)
    allowed = causal_mask(pos, pos)  # [B, T, T]
    mask = bool_to_additive_mask(allowed, dtype)[:, None]  # [B, 1, T, T]
    return hf, ours, x, cos, sin, mask


def _hf_trace(hf, x, cos, sin, mask):
    captured = {}
    hooks = [
        getattr(hf, n).register_forward_hook(
            lambda _m, _i, out, n=n: captured.__setitem__(n, out.detach())
        )
        for n in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm")
    ]
    hooks.append(
        hf.o_proj.register_forward_hook(lambda _m, inp, _o: captured.__setitem__("ctx", inp[0]))
    )
    with torch.no_grad():
        out, weights = hf(x, (cos, sin), mask)
    for h in hooks:
        h.remove()
    q_n = captured["q_norm"].transpose(1, 2)
    k_n = captured["k_norm"].transpose(1, 2)
    q_r, k_r = modeling.apply_rotary_pos_emb(q_n, k_n, cos, sin)
    k_g = modeling.repeat_kv(k_r, hf.num_key_value_groups)
    v = captured["v_proj"].view(B, T, -1, HEAD_DIM).transpose(1, 2)
    v_g = modeling.repeat_kv(v, hf.num_key_value_groups)
    logits = torch.matmul(q_r, k_g.transpose(2, 3)) * hf.scaling + mask
    return {
        "q_proj": captured["q_proj"],
        "k_proj": captured["k_proj"],
        "v_proj": captured["v_proj"],
        "q_normed": q_n,
        "k_normed": k_n,
        "q_rope": q_r,
        "k_rope": k_r,
        "k_grouped": k_g,
        "v_grouped": v_g,
        "logits": logits,
        "probs": weights,
        "context": captured["ctx"],
        "output": out,
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(("nq", "nkv"), CONFIGS, ids=IDS)
def test_every_intermediate_bit_identical(nq, nkv, dtype) -> None:
    hf, ours, x, cos, sin, mask = _setup(nq, nkv, dtype)
    ref = _hf_trace(hf, x, cos, sin, mask)
    with torch.no_grad():
        _, tr = ours(x, cos, sin, return_trace=True)
    allowed = mask == 0
    for name in (
        "q_proj",
        "k_proj",
        "v_proj",
        "q_normed",
        "k_normed",
        "q_rope",
        "k_rope",
        "k_grouped",
        "v_grouped",
        "probs",
        "context",
        "output",
    ):
        ours_t, ref_t = getattr(tr, name), ref[name]
        assert ours_t.shape == ref_t.shape, name
        assert torch.equal(ours_t, ref_t), name
    # Logits: identical where allowed; masked entries are -inf (ours) vs ~finfo.min (HF).
    allowed_b = allowed.expand_as(tr.scores)
    assert torch.equal(tr.scores[allowed_b], ref["logits"][allowed_b])
    assert torch.isneginf(tr.scores[~allowed_b]).all()


@pytest.mark.parametrize(("nq", "nkv"), CONFIGS, ids=IDS)
def test_gradients_bit_identical_fp32(nq, nkv) -> None:
    hf, ours, x, cos, sin, mask = _setup(nq, nkv, torch.float32, seed=5)
    upstream = torch.randn(B, T, HIDDEN, generator=torch.Generator().manual_seed(9))
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    ours(xa, cos, sin).backward(upstream)
    hf(xb, (cos, sin), mask)[0].backward(upstream)
    assert torch.equal(xa.grad, xb.grad)
    for (name, p_ours), (name_hf, p_hf) in zip(
        sorted(ours.named_parameters()), sorted(hf.named_parameters()), strict=True
    ):
        assert name == name_hf
        assert torch.equal(p_ours.grad, p_hf.grad), name


def test_hf_module_without_mask_is_not_causal() -> None:
    hf, ours, x, cos, sin, _ = _setup(4, 2, torch.float32)
    with torch.no_grad():
        _, weights = hf(x, (cos, sin), None)
        _, tr = ours(x, cos, sin, return_trace=True)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert (weights[..., future] > 0).any()  # HF alone attends to the future
    assert (tr.probs[..., future] == 0).all()  # ours is causal by default
