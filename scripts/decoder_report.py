#!/usr/bin/env python
"""Numerical error report for one Qwen3DecoderLayer (M1.5).

    python scripts/decoder_report.py

Qwen3-0.6B layer sizes (H=1024, I=3072, 16/8 heads, D=128), random weights
(std 0.02; norm weights 1 + 0.1*N(0,1)), x ~ N(0, 1) of shape (1, 64, 1024),
positions 0..63. Compares float32/bfloat16 against the same layer in float64
(float64 RoPE tables) and against HF Qwen3DecoderLayer (eager, if installed).
Metrics: max_abs, mean_abs, rel_l2 = ||a-b|| / ||b||. CPU only, fixed seeds.
"""

from __future__ import annotations

import torch

from qwen3_scratch.config import QWEN3_0_6B_BASE as CFG
from qwen3_scratch.layers import (
    Qwen3DecoderLayer,
    RotaryEmbedding,
    bool_to_additive_mask,
    causal_mask,
    compute_inv_freq,
    make_position_ids,
    rope_cos_sin,
)

T = 64


def metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    a, b = a.detach().double(), b.detach().double()
    d = (a - b).abs()
    return d.max().item(), d.mean().item(), (d.norm() / b.norm()).item()


def main() -> None:
    try:
        import transformers
        from transformers.models.qwen3 import modeling_qwen3 as hf_mod
    except ImportError:
        transformers = hf_mod = None

    gen = torch.Generator().manual_seed(0)
    ref = Qwen3DecoderLayer.from_config(CFG, dtype=torch.float64)
    state = {
        n: (1 + 0.1 * torch.randn(p.shape, generator=gen, dtype=torch.float64))
        if "norm" in n
        else 0.02 * torch.randn(p.shape, generator=gen, dtype=torch.float64)
        for n, p in ref.named_parameters()
    }
    ref.load_state_dict(state)
    x64 = torch.randn(1, T, CFG.hidden_size, generator=gen, dtype=torch.float64)
    u64 = torch.randn(1, T, CFG.hidden_size, generator=gen, dtype=torch.float64)
    pos = make_position_ids(T, batch_size=1)
    xr = x64.clone().requires_grad_()
    inv64 = compute_inv_freq(CFG.head_dim, CFG.rope_theta, dtype=torch.float64)
    out64, tr64 = ref(xr, *rope_cos_sin(pos, inv64), return_trace=True)
    out64.backward(u64)

    print(
        f"torch {torch.__version__}; transformers: "
        f"{transformers.__version__ if transformers else 'not installed'}\n"
    )
    print("| dtype | ours vs | tensor | max_abs | mean_abs | rel_l2 |")
    print("|---|---|---|---|---|---|")
    for dtype in (torch.float32, torch.bfloat16):
        name = str(dtype).replace("torch.", "")
        layer = Qwen3DecoderLayer.from_config(CFG, dtype=dtype)
        layer.load_state_dict({k: v.to(dtype) for k, v in state.items()})
        x = x64.to(dtype).requires_grad_()
        cos, sin = RotaryEmbedding.from_config(CFG)(pos, dtype=dtype)
        out, tr = layer(x, cos, sin, return_trace=True)
        out.backward(u64.to(dtype))
        rows = [
            ("float64 layer", "attn_output", tr.attn_output, tr64.attn_output),
            ("float64 layer", "mlp_output", tr.mlp_output, tr64.mlp_output),
            ("float64 layer", "output", out, out64),
            ("float64 layer", "dL/dx", x.grad, xr.grad),
        ]
        if hf_mod is not None:
            cfg = transformers.Qwen3Config(
                hidden_size=CFG.hidden_size,
                intermediate_size=CFG.intermediate_size,
                num_attention_heads=CFG.num_attention_heads,
                num_key_value_heads=CFG.num_key_value_heads,
                head_dim=CFG.head_dim,
                rope_theta=CFG.rope_theta,
                rms_norm_eps=CFG.rms_norm_eps,
                attn_implementation="eager",
            )
            hf = hf_mod.Qwen3DecoderLayer(cfg, layer_idx=0).to(dtype).eval()
            hf.load_state_dict({k: v.to(dtype) for k, v in state.items()})
            xh = x64.to(dtype).requires_grad_()
            mask = bool_to_additive_mask(causal_mask(pos, pos), dtype)[:, None]
            hout = hf(xh, attention_mask=mask, position_embeddings=(cos, sin), position_ids=pos)
            hout = hout[0] if isinstance(hout, tuple) else hout
            hout.backward(u64.to(dtype))
            rows += [
                ("HF Qwen3DecoderLayer", "output", out, hout),
                ("HF Qwen3DecoderLayer", "dL/dx", x.grad, xh.grad),
            ]
        for ref_name, tensor, a, b in rows:
            mx, mean, rel = metrics(a, b)
            print(f"| {name} | {ref_name} | {tensor} | {mx:.3e} | {mean:.3e} | {rel:.3e} |")


if __name__ == "__main__":
    main()
