#!/usr/bin/env python
"""Numerical error report for SwiGLUFeedForward (M1.2).

    python scripts/swiglu_report.py

Uses Qwen3-0.6B sizes (H=1024, I=3072) with random weights (std 0.02, like the HF
initializer) and inputs x ~ N(0, 1), shape (2, 64, 1024). Compares against:
  * the textbook formula in float64 on the SAME (already rounded) inputs/weights,
  * Hugging Face Qwen3MLP with identical weights (only if `transformers` is installed).

Metrics: max_abs = max|a-b|; scaled = max|a-b| / max|b|; rel_l2 = ||a-b||_2 / ||b||_2.
Elementwise relative error is not reported: matmul outputs contain values near
zero whose relative error is meaningless. CPU only, fixed seeds.
"""

from __future__ import annotations

import torch

from qwen3_scratch.config import QWEN3_0_6B_BASE as CFG
from qwen3_scratch.layers import SwiGLUFeedForward, swiglu_textbook


def errors(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float, float]:
    a, b = a.double(), b.double()
    diff = (a - b).abs()
    return (
        diff.max().item(),
        (diff.max() / b.abs().max()).item(),
        (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item(),
        100.0 * (diff > 0).double().mean().item(),
    )


def main() -> None:
    try:
        import transformers
        from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
    except ImportError:
        transformers = Qwen3MLP = None  # noqa: N806

    gen = torch.Generator().manual_seed(0)
    h, i = CFG.hidden_size, CFG.intermediate_size
    weights64 = {
        "gate_proj.weight": 0.02 * torch.randn(i, h, generator=gen, dtype=torch.float64),
        "up_proj.weight": 0.02 * torch.randn(i, h, generator=gen, dtype=torch.float64),
        "down_proj.weight": 0.02 * torch.randn(h, i, generator=gen, dtype=torch.float64),
    }
    x64 = torch.randn(2, 64, h, generator=gen, dtype=torch.float64)

    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        name = str(dtype).replace("torch.", "")
        sd = {k: v.to(dtype) for k, v in weights64.items()}
        x = x64.to(dtype)
        ours = SwiGLUFeedForward(h, i, dtype=dtype)
        ours.load_state_dict(sd)
        with torch.no_grad():
            y = ours(x)
            ref = swiglu_textbook(x.double(), *(sd[k].double() for k in weights64))
            rows.append((name, "textbook float64", *errors(y, ref)))
            if Qwen3MLP is not None:
                hf_cfg = transformers.Qwen3Config(
                    hidden_size=h, intermediate_size=i, hidden_act="silu"
                )
                hf = Qwen3MLP(hf_cfg).to(dtype)
                hf.load_state_dict(sd)
                rows.append((name, "HF Qwen3MLP", *errors(y, hf(x))))

    print(f"torch {torch.__version__}; transformers available: {Qwen3MLP is not None}")
    print(f"shape (2, 64, {h}) -> I = {i}\n")
    print("| dtype | ours vs | max_abs | max_abs / max abs(ref) | rel_l2 | elements differing |")
    print("|---|---|---|---|---|---|")
    for name, ref, max_abs, scaled, rel_l2, frac in rows:
        print(f"| {name} | {ref} | {max_abs:.3e} | {scaled:.3e} | {rel_l2:.3e} | {frac:.1f}% |")


if __name__ == "__main__":
    main()
