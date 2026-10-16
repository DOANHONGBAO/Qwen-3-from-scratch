"""QK-Norm parity with the normalisation inside HF ``Qwen3Attention``.

HF applies ``q_norm``/``k_norm`` to ``q_proj(x).view(B, T, N, D)`` and only then
transposes to ``(B, N, T, D)``. We normalise after the transpose. This test checks
that both layouts give bit-identical results with identical weights.
Requires the ``hf`` extra; skipped otherwise. No downloads.
"""

from __future__ import annotations

import pytest
import torch

modeling = pytest.importorskip(
    "transformers.models.qwen3.modeling_qwen3", reason="transformers not installed (hf extra)"
)
transformers = pytest.importorskip("transformers")

from qwen3_scratch.layers import QKNorm  # noqa: E402


def _hf_attention(hidden: int, nq: int, nkv: int, d: int):
    cfg = transformers.Qwen3Config(
        hidden_size=hidden,
        num_attention_heads=nq,
        num_key_value_heads=nkv,
        head_dim=d,
        num_hidden_layers=1,
        vocab_size=32,
        rms_norm_eps=1e-6,
    )
    return modeling.Qwen3Attention(cfg, layer_idx=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("nq", "nkv", "d"), [(4, 2, 32), (16, 8, 128), (4, 4, 16), (4, 1, 16)])
def test_matches_hf_q_norm_k_norm(dtype, nq, nkv, d) -> None:
    torch.manual_seed(0)
    hidden = 64
    hf = _hf_attention(hidden, nq, nkv, d).to(dtype)
    with torch.no_grad():  # non-trivial norm weights
        hf.q_norm.weight.copy_(1 + 0.3 * torch.randn(d))
        hf.k_norm.weight.copy_(1 - 0.3 * torch.randn(d))
    ours = QKNorm(d, 1e-6, dtype=dtype)
    ours.load_state_dict(
        {"q_norm.weight": hf.q_norm.weight, "k_norm.weight": hf.k_norm.weight}, strict=True
    )

    x = torch.randn(2, 5, hidden).to(dtype)
    with torch.no_grad():
        q_btnd = hf.q_proj(x).view(2, 5, nq, d)
        k_btnd = hf.k_proj(x).view(2, 5, nkv, d)
        hf_q = hf.q_norm(q_btnd).transpose(1, 2)  # exactly HF's forward order
        hf_k = hf.k_norm(k_btnd).transpose(1, 2)
        our_q, our_k = ours(q_btnd.transpose(1, 2), k_btnd.transpose(1, 2))
    assert torch.equal(our_q, hf_q) and torch.equal(our_k, hf_k)


def test_hf_has_no_v_norm() -> None:
    hf = _hf_attention(64, 4, 2, 16)
    names = {n for n, _ in hf.named_parameters()}
    assert {"q_norm.weight", "k_norm.weight"} <= names
    assert not any("v_norm" in n for n in names)
