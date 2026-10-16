"""Parity of Qwen3DecoderLayer with Hugging Face ``Qwen3DecoderLayer`` (eager).

The HF layer is driven like ``Qwen3Model`` drives it: explicit
``position_embeddings`` from ``Qwen3RotaryEmbedding`` and the additive causal mask
``[B, 1, T, T]``. Intermediates come from forward hooks on ``input_layernorm``,
``self_attn``, ``post_attention_layernorm`` and ``mlp``; the first residual
``h1 = x + attn`` is recomputed from the hooked attention output (HF does not
expose it). Requires the ``hf`` extra; no downloads.
"""

from __future__ import annotations

import pytest
import torch

modeling = pytest.importorskip(
    "transformers.models.qwen3.modeling_qwen3", reason="transformers not installed (hf extra)"
)
transformers = pytest.importorskip("transformers")

from qwen3_scratch.layers import (  # noqa: E402
    Qwen3DecoderLayer,
    bool_to_additive_mask,
    causal_mask,
    make_position_ids,
)

B, T, H, INTER, D = 2, 7, 64, 160, 32


def _setup(nq: int, nkv: int, dtype: torch.dtype, seed: int = 0):
    cfg = transformers.Qwen3Config(
        hidden_size=H,
        intermediate_size=INTER,
        num_attention_heads=nq,
        num_key_value_heads=nkv,
        head_dim=D,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        vocab_size=32,
        attn_implementation="eager",
    )
    torch.manual_seed(seed)
    hf = modeling.Qwen3DecoderLayer(cfg, layer_idx=0).to(dtype).eval()
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if "norm" in name:
                p.copy_(1 + 0.2 * torch.randn(p.shape))
    ours = Qwen3DecoderLayer(H, INTER, nq, nkv, D, dtype=dtype)
    missing, unexpected = ours.load_state_dict(hf.state_dict(), strict=True)
    assert not missing and not unexpected
    gen = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(B, T, H, generator=gen).to(dtype)
    pos = make_position_ids(T, offset=5, batch_size=B)
    cos, sin = modeling.Qwen3RotaryEmbedding(cfg)(x, pos)
    mask = bool_to_additive_mask(causal_mask(pos, pos), dtype)[:, None]
    return hf, ours, x, pos, cos, sin, mask


def _hf_forward(hf, x, pos, cos, sin, mask):
    seen = {}

    def keep(name):
        def hook(_m, _inp, out):
            seen[name] = out[0] if isinstance(out, tuple) else out

        return hook

    hooks = [
        getattr(hf, n).register_forward_hook(keep(n))
        for n in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")
    ]
    out = hf(x, attention_mask=mask, position_embeddings=(cos, sin), position_ids=pos)
    for h in hooks:
        h.remove()
    seen["residual_1"] = x + seen["self_attn"]
    seen["output"] = out[0] if isinstance(out, tuple) else out
    return seen


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(("nq", "nkv"), [(4, 2), (4, 4), (4, 1)], ids=["GQA", "MHA", "MQA"])
def test_all_intermediates_bit_identical(nq, nkv, dtype) -> None:
    hf, ours, x, pos, cos, sin, mask = _setup(nq, nkv, dtype)
    with torch.no_grad():
        ref = _hf_forward(hf, x, pos, cos, sin, mask)
        _, tr = ours(x, cos, sin, return_trace=True)
    pairs = {
        "input_layernorm": tr.input_norm,
        "self_attn": tr.attn_output,
        "residual_1": tr.residual_1,
        "post_attention_layernorm": tr.post_attn_norm,
        "mlp": tr.mlp_output,
        "output": tr.output,
    }
    for name, value in pairs.items():
        assert value.shape == ref[name].shape == (B, T, H), name
        assert torch.equal(value, ref[name]), name


def test_gradients_bit_identical_fp32() -> None:
    hf, ours, x, pos, cos, sin, mask = _setup(4, 2, torch.float32, seed=3)
    u = torch.randn(B, T, H, generator=torch.Generator().manual_seed(4))
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    ours(xa, cos, sin).backward(u)
    out = hf(xb, attention_mask=mask, position_embeddings=(cos, sin), position_ids=pos)
    (out[0] if isinstance(out, tuple) else out).backward(u)
    assert torch.equal(xa.grad, xb.grad)
    ours_g = dict(ours.named_parameters())
    for name, p in hf.named_parameters():
        assert torch.equal(ours_g[name].grad, p.grad), name
