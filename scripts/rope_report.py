#!/usr/bin/env python
"""Numerical error report for RoPE (M1.3).

    python scripts/rope_report.py

Qwen3-0.6B settings (head_dim 128, theta 1e6), positions 0..32767, CPU only.

1. Table accuracy: HF-style float32 angles/cos/sin vs float64 ground truth.
2. Rotated q (B=1, H=16, T=512 at several offsets): ours vs float64 truth,
   vs Hugging Face (if `transformers` is installed), and vs the alternative
   precision order of the educational reference (rotate in float32, cast once).
3. Relative-position property with float32 tables: |score(m+s, n+s) - score(m, n)|.
"""

from __future__ import annotations

import torch

from qwen3_scratch.config import QWEN3_0_6B_BASE as CFG
from qwen3_scratch.layers import (
    RotaryEmbedding,
    apply_rope,
    compute_inv_freq,
    make_position_ids,
    rope_cos_sin,
)


def metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    a, b = a.double(), b.double()
    diff = (a - b).abs()
    rel_l2 = (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()
    return diff.max().item(), rel_l2, 100.0 * (diff > 0).double().mean().item()


def main() -> None:
    try:
        import transformers
        from transformers.models.qwen3.modeling_qwen3 import (
            Qwen3RotaryEmbedding,
            apply_rotary_pos_emb,
        )
    except ImportError:
        transformers = None

    d, theta, n_pos = CFG.head_dim, CFG.rope_theta, CFG.max_position_embeddings
    rope = RotaryEmbedding(d, theta)
    inv64 = compute_inv_freq(d, theta, dtype=torch.float64)
    all_pos = make_position_ids(n_pos)

    print(f"torch {torch.__version__}; transformers available: {transformers is not None}")
    print(f"head_dim={d}, rope_theta={theta:g}, positions 0..{n_pos - 1}\n")

    # 1. tables
    cos32, sin32 = rope(all_pos)
    cos64, sin64 = rope_cos_sin(all_pos, inv64)
    angle32 = all_pos[:, None].float() * rope.inv_freq
    angle64 = all_pos[:, None].double() * inv64
    print("## 1. float32 tables vs float64 truth (all positions)\n")
    print("| quantity | max_abs error | at position <= 4095 |")
    print("|---|---|---|")
    for name, a, b in [
        ("inv_freq", rope.inv_freq, inv64),
        ("angle (rad)", angle32, angle64),
        ("cos", cos32, cos64),
        ("sin", sin32, sin64),
    ]:
        err = (a.double() - b).abs()
        short = err[:4096].max().item() if err.dim() == 2 else err.max().item()
        print(f"| {name} | {err.max().item():.3e} | {short:.3e} |")

    # 2. rotations
    print("\n## 2. rotated q, shape (1, 16, 512, 128)\n")
    print("| offset | dtype | ours vs | max_abs | rel_l2 | elements differing |")
    print("|---|---|---|---|---|---|")
    gen = torch.Generator().manual_seed(0)
    q64 = torch.randn(1, 16, 512, d, generator=gen, dtype=torch.float64)
    for offset in (0, 8192, 32256):
        pos = make_position_ids(512, offset=offset, batch_size=1)
        truth_tables = rope_cos_sin(pos, inv64)
        for dtype in (torch.float32, torch.bfloat16):
            name = str(dtype).replace("torch.", "")
            q = q64.to(dtype)
            cos, sin = rope(pos, dtype=dtype)
            ours = apply_rope(q, cos, sin)
            truth = apply_rope(q.double(), *truth_tables)
            rows = [("float64 truth", *metrics(ours, truth))]
            if transformers is not None:
                cfg = transformers.Qwen3Config(
                    hidden_size=2048,
                    num_attention_heads=16,
                    num_key_value_heads=8,
                    head_dim=d,
                    rope_theta=theta,
                    max_position_embeddings=n_pos,
                )
                hf_cos, hf_sin = Qwen3RotaryEmbedding(cfg)(q, pos)
                hf_q, _ = apply_rotary_pos_emb(q, q[:, :1], hf_cos, hf_sin)
                rows.append(("HF apply_rotary_pos_emb", *metrics(ours, hf_q)))
            if dtype != torch.float32:
                c32, s32 = rope(pos)  # float32 tables, rotate in float32, cast once
                alt = apply_rope(q.float(), c32, s32).to(dtype)
                rows.append(("rotate-in-fp32 order", *metrics(ours, alt)))
            for ref, max_abs, rel, frac in rows:
                print(f"| {offset} | {name} | {ref} | {max_abs:.3e} | {rel:.3e} | {frac:.1f}% |")

    # 3. relative-position property with float32 tables
    print("\n## 3. relative-position property, float32 tables, float64 vectors\n")
    print("| shift s | max over 200 random (q, k) of abs(score(m+s, n+s) - score(m, n)) |")
    print("|---|---|")
    q = torch.randn(200, 1, 1, d, generator=gen, dtype=torch.float64)
    k = torch.randn(200, 1, 1, d, generator=gen, dtype=torch.float64)

    def score(pq: int, pk: int) -> torch.Tensor:
        cq, sq = rope(torch.tensor([pq]), dtype=torch.float64)
        ck, sk = rope(torch.tensor([pk]), dtype=torch.float64)
        return (apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum(dim=-1).flatten()

    base = score(3, 10)
    for shift in (1, 100, 1000, 8000, 32000):
        print(f"| {shift} | {(score(3 + shift, 10 + shift) - base).abs().max().item():.3e} |")


if __name__ == "__main__":
    main()
