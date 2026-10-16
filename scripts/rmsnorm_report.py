#!/usr/bin/env python
"""Numerical error report for qwen3_scratch RMSNorm (M1.1).

    python scripts/rmsnorm_report.py

Compares our RMSNorm against:
  * the textbook formula evaluated in float64 on the SAME (already rounded) inputs,
  * torch.nn.RMSNorm (eps passed explicitly, same weights),
  * Hugging Face Qwen3RMSNorm (only if `transformers` is installed),
  * the alternative "scale in float32, then cast" order used by the educational
    reference (Raschka, Appendix C), to quantify discrepancy X5.

Error definitions (elementwise, over all outputs):
  max_abs = max |a - b|
  max_rel = max |a - b| / |b|  over elements with |b| >= 1e-3 (avoids 0/0 noise)
  elements differing = share of outputs that are not bit-identical
CPU only, fixed seeds, no downloads.
"""

from __future__ import annotations

import torch

from qwen3_scratch.layers import RMSNorm, rms_norm_textbook

EPS = 1e-6


def errors(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    a, b = a.double(), b.double()
    diff = (a - b).abs()
    mask = b.abs() >= 1e-3
    rel = (diff[mask] / b.abs()[mask]).max().item() if mask.any() else 0.0
    return diff.max().item(), rel, 100.0 * (diff > 0).double().mean().item()


def scale_then_cast(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Alternative precision order: multiply by weight in float32, then cast."""
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return (h * weight.float()).to(x.dtype)


def main() -> None:
    gen = torch.Generator().manual_seed(0)
    rows: list[tuple[str, str, str, float, float, float]] = []
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    except ImportError:
        Qwen3RMSNorm = None  # noqa: N806

    for d, label in [(1024, "hidden (B=4,T=128,d=1024)"), (128, "QK head (B=4,H=16,T=128,d=128)")]:
        shape = (4, 128, d) if d == 1024 else (4, 16, 128, d)
        x64 = 2.0 * torch.randn(*shape, generator=gen, dtype=torch.float64)
        w64 = 1.0 + 0.2 * torch.randn(d, generator=gen, dtype=torch.float64)

        for dtype in (torch.float32, torch.bfloat16):
            name = str(dtype).replace("torch.", "")
            x, w = x64.to(dtype), w64.to(dtype)
            ours = RMSNorm(d, eps=EPS, dtype=dtype)
            with torch.no_grad():
                ours.weight.copy_(w)
                y = ours(x)

                ref64 = rms_norm_textbook(x.double(), w.double(), EPS)
                rows.append((label, name, "textbook float64", *errors(y, ref64)))

                tnn = torch.nn.RMSNorm(d, eps=EPS, dtype=dtype)
                tnn.weight.copy_(w)
                rows.append((label, name, "torch.nn.RMSNorm", *errors(y, tnn(x))))

                if Qwen3RMSNorm is not None:
                    hf = Qwen3RMSNorm(d, eps=EPS).to(dtype)
                    hf.weight.copy_(w)
                    rows.append((label, name, "HF Qwen3RMSNorm", *errors(y, hf(x))))

                if dtype != torch.float32:
                    alt = scale_then_cast(x, w, EPS)
                    rows.append((label, name, "scale-then-cast order", *errors(y, alt)))

    print(f"torch {torch.__version__}; transformers available: {Qwen3RMSNorm is not None}\n")
    print("| tensor | dtype | ours vs | max_abs | max_rel | elements differing |")
    print("|---|---|---|---|---|---|")
    for label, dtype, ref, max_abs, max_rel, frac in rows:
        print(f"| {label} | {dtype} | {ref} | {max_abs:.3e} | {max_rel:.3e} | {frac:.1f}% |")


if __name__ == "__main__":
    main()
