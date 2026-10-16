#!/usr/bin/env python
"""Numerical error report for GroupedQueryAttention (M1.4).

    python scripts/attention_report.py

Qwen3-0.6B attention sizes (H=1024, Nq=16, Nkv=8, D=128, theta=1e6), random
weights (std 0.02, HF initializer scale), QK-Norm weights 1 + 0.1*N(0,1),
input x ~ N(0, 1) of shape (1, 128, 1024), positions 1000..1127.

Compares our module (float32 and bfloat16) against:
  * the same module in float64 (float64 RoPE tables) = high-precision reference,
  * Hugging Face Qwen3Attention, eager, with the additive causal mask (if installed).
Metrics: max_abs; max_rel = max|a-b| / max|ref|; mean_abs. Also input-gradient errors.
CPU only, fixed seeds, no downloads.
"""

from __future__ import annotations

import torch

from qwen3_scratch.config import QWEN3_0_6B_BASE as CFG
from qwen3_scratch.layers import (
    GroupedQueryAttention,
    RotaryEmbedding,
    bool_to_additive_mask,
    causal_mask,
    compute_inv_freq,
    make_position_ids,
    rope_cos_sin,
)

T, OFFSET = 128, 1000


def metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    a, b = a.double(), b.double()
    diff = (a - b).abs()
    return diff.max().item(), (diff.max() / b.abs().max()).item(), diff.mean().item()


def build(dtype: torch.dtype, state: dict) -> GroupedQueryAttention:
    m = GroupedQueryAttention.from_config(CFG, dtype=dtype)
    m.load_state_dict({k: v.to(dtype) for k, v in state.items()})
    return m


def main() -> None:
    try:
        import transformers
        from transformers.models.qwen3 import modeling_qwen3 as hf_mod
    except ImportError:
        transformers = hf_mod = None

    gen = torch.Generator().manual_seed(0)
    ref64 = GroupedQueryAttention.from_config(CFG, dtype=torch.float64)
    state = {}
    for name, p in ref64.named_parameters():
        if "norm" in name:
            state[name] = 1 + 0.1 * torch.randn(p.shape, generator=gen, dtype=torch.float64)
        else:
            state[name] = 0.02 * torch.randn(p.shape, generator=gen, dtype=torch.float64)
    ref64.load_state_dict(state)
    x64 = torch.randn(1, T, CFG.hidden_size, generator=gen, dtype=torch.float64)
    up64 = torch.randn(1, T, CFG.hidden_size, generator=gen, dtype=torch.float64)
    pos = make_position_ids(T, offset=OFFSET, batch_size=1)
    inv64 = compute_inv_freq(CFG.head_dim, CFG.rope_theta, dtype=torch.float64)

    xr = x64.clone().requires_grad_()
    out64, tr64 = ref64(xr, *rope_cos_sin(pos, inv64), return_trace=True)
    out64.backward(up64)
    grad64 = xr.grad

    print(f"torch {torch.__version__}; transformers available: {transformers is not None}")
    print(
        f"shape (1, {T}, {CFG.hidden_size}), heads {CFG.num_attention_heads}/"
        f"{CFG.num_key_value_heads}, head_dim {CFG.head_dim}, "
        f"positions {OFFSET}..{OFFSET + T - 1}\n"
    )
    print("| dtype | ours vs | tensor | max_abs | max_rel | mean_abs |")
    print("|---|---|---|---|---|---|")
    for dtype in (torch.float32, torch.bfloat16):
        name = str(dtype).replace("torch.", "")
        ours = build(dtype, state)
        x = x64.to(dtype).requires_grad_()
        cos, sin = RotaryEmbedding.from_config(CFG)(pos, dtype=dtype)
        out, tr = ours(x, cos, sin, return_trace=True)
        out.backward(up64.to(dtype))
        rows = [
            ("float64 module", "probs", tr.probs, tr64.probs),
            ("float64 module", "context", tr.context, tr64.context),
            ("float64 module", "output", out, out64),
            ("float64 module", "dL/dx", x.grad, grad64),
        ]
        if hf_mod is not None:
            cfg = transformers.Qwen3Config(
                hidden_size=CFG.hidden_size,
                num_attention_heads=CFG.num_attention_heads,
                num_key_value_heads=CFG.num_key_value_heads,
                head_dim=CFG.head_dim,
                rope_theta=CFG.rope_theta,
                rms_norm_eps=CFG.rms_norm_eps,
                attn_implementation="eager",
            )
            hf = hf_mod.Qwen3Attention(cfg, layer_idx=0).to(dtype).eval()
            hf.load_state_dict({k: v.to(dtype) for k, v in state.items()})
            xh = x64.to(dtype).requires_grad_()
            hcos, hsin = hf_mod.Qwen3RotaryEmbedding(cfg)(xh, pos)
            mask = bool_to_additive_mask(causal_mask(pos, pos), dtype)[:, None]
            hout, hw = hf(xh, (hcos, hsin), mask)
            hout.backward(up64.to(dtype))
            rows += [
                ("HF Qwen3Attention", "probs", tr.probs, hw),
                ("HF Qwen3Attention", "output", out, hout),
                ("HF Qwen3Attention", "dL/dx", x.grad, xh.grad),
            ]
        for ref, tensor, a, b in rows:
            max_abs, max_rel, mean_abs = metrics(a.detach(), b.detach())
            print(f"| {name} | {ref} | {tensor} | {max_abs:.3e} | {max_rel:.3e} | {mean_abs:.3e} |")


if __name__ == "__main__":
    main()
