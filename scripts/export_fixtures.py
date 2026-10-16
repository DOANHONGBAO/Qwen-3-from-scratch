#!/usr/bin/env python
"""Export small deterministic reference fixtures for future C++ parity tests.

    python scripts/export_fixtures.py            # (re)write all fixtures
    python scripts/export_fixtures.py --check    # verify committed fixtures are current

Fixtures are tiny JSON files under ``tests/fixtures/`` (a few KB each), so they are
reviewable in Git and need no binary format. Every float is stored as a JSON
number holding the exact float32 value (Python ``repr`` of the widened double),
so it round-trips bit-exactly. Each file records the generator seed, shapes,
dtype and the tolerance the C++ port must meet.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from qwen3_scratch.config import Qwen3Config
from qwen3_scratch.layers import (
    GroupedQueryAttention,
    RotaryEmbedding,
    SwiGLUFeedForward,
    apply_rope_to_qk,
    make_position_ids,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def _values(t: torch.Tensor) -> list:
    return t.detach().to(torch.float64).tolist()


def _tensor(t: torch.Tensor) -> dict[str, Any]:
    return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "data": _values(t)}


def swiglu_fixture() -> dict[str, Any]:
    """SwiGLU with H=4, I=6 on a (2, 3, 4) input, float32."""
    seed, h, i = 1234, 4, 6
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 3, h, generator=gen)
    mlp = SwiGLUFeedForward(h, i)
    with torch.no_grad():
        for p in mlp.parameters():
            p.copy_(0.5 * torch.randn(p.shape, generator=gen))
        y = mlp(x)
    return {
        "component": "swiglu",
        "description": "down_proj(silu(gate_proj(x)) * up_proj(x)); weights (out, in)",
        "seed": seed,
        "config": {"hidden_size": h, "intermediate_size": i, "bias": False},
        "tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "inputs": {"x": _tensor(x)},
        "parameters": {name: _tensor(p) for name, p in mlp.named_parameters()},
        "outputs": {"y": _tensor(y)},
    }


def rope_fixture() -> dict[str, Any]:
    """RoPE, head_dim=8, theta=10000, Nq=2, Nkv=1, positions 5..7 (decode-like offset)."""
    seed, d, theta = 4321, 8, 10_000.0
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, 2, 3, d, generator=gen)
    k = torch.randn(1, 1, 3, d, generator=gen)
    rope = RotaryEmbedding(d, theta)
    pos = make_position_ids(3, offset=5)
    cos, sin = rope(pos)
    q_rot, k_rot = apply_rope_to_qk(q, k, cos, sin)
    return {
        "component": "rope",
        "description": "split-half RoPE: y = x*cos + rotate_half(x)*sin, float32 tables",
        "seed": seed,
        "config": {"head_dim": d, "rope_theta": theta, "position_ids": pos.tolist()},
        "tolerance": {"rtol": 1e-6, "atol": 1e-6},
        "inputs": {"q": _tensor(q), "k": _tensor(k)},
        "parameters": {"inv_freq": _tensor(rope.inv_freq)},
        "outputs": {
            "cos": _tensor(cos),
            "sin": _tensor(sin),
            "q_rot": _tensor(q_rot),
            "k_rot": _tensor(k_rot),
        },
    }


def attention_fixture() -> dict[str, Any]:
    """GQA self-attention: H=8, Nq=4, Nkv=2, D=4, T=3, positions 2..4, causal, float32."""
    seed, h, nq, nkv, d, theta = 2024, 8, 4, 2, 4, 10_000.0
    gen = torch.Generator().manual_seed(seed)
    attn = GroupedQueryAttention(h, nq, nkv, d)
    with torch.no_grad():
        for name, p in attn.named_parameters():
            base = 1.0 if "norm" in name else 0.0
            p.copy_(base + 0.5 * torch.randn(p.shape, generator=gen))
        x = torch.randn(1, 3, h, generator=gen)
        pos = make_position_ids(3, offset=2)
        cos, sin = RotaryEmbedding(d, theta)(pos)
        y, tr = attn(x, cos, sin, return_trace=True)
    return {
        "component": "attention",
        "description": (
            "Qwen3 GQA: q/k/v proj -> per-head q_norm/k_norm -> RoPE (split-half) -> "
            "query head h uses kv head h // (Nq/Nkv) -> causal softmax(q k^T * D^-0.5) v "
            "-> o_proj. probs/context/output are expected values."
        ),
        "seed": seed,
        "config": {
            "hidden_size": h,
            "num_attention_heads": nq,
            "num_key_value_heads": nkv,
            "head_dim": d,
            "rope_theta": theta,
            "rms_norm_eps": 1e-6,
            "position_ids": pos.tolist(),
        },
        "tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "inputs": {"x": _tensor(x)},
        "parameters": {name: _tensor(p) for name, p in attn.named_parameters()},
        "outputs": {
            "q_rope": _tensor(tr.q_rope),
            "k_rope": _tensor(tr.k_rope),
            "probs": _tensor(tr.probs),
            "context": _tensor(tr.context),
            "output": _tensor(y),
        },
    }


def model_fixture() -> dict[str, Any]:
    """Whole Tiny Qwen3 forward: embeddings, every layer, final norm, logits, loss."""
    from qwen3_scratch.model import Qwen3ForCausalLM

    seed = 7
    cfg = Qwen3Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    gen = torch.Generator().manual_seed(seed)
    model = Qwen3ForCausalLM(cfg, init_weights=False)
    with torch.no_grad():
        for name, p in model.named_parameters():  # tied matrix appears once
            base = 1.0 if "norm" in name else 0.0
            p.copy_(base + 0.3 * torch.randn(p.shape, generator=gen))
        ids = torch.tensor([[3, 14, 1, 5, 9]])
        out = model(ids, labels=ids, output_hidden_states=True)
    return {
        "component": "qwen3_for_causal_lm",
        "description": (
            "Tiny Qwen3ForCausalLM forward. lm_head.weight is TIED to "
            "model.embed_tokens.weight (not stored twice). position_ids = 0..T-1, "
            "attention_mask = all ones (causal only). hidden_states[0] = embeddings, "
            "hidden_states[i+1] = output of layer i (before the final norm). "
            "loss = mean next-token cross-entropy with labels = input_ids."
        ),
        "seed": seed,
        "config": cfg.to_dict(),
        "tolerance": {"rtol": 1e-5, "atol": 1e-5},
        "inputs": {
            "input_ids": ids.tolist(),
            "position_ids": [list(range(ids.shape[1]))],
            "attention_mask": torch.ones_like(ids).tolist(),
            "labels": ids.tolist(),
        },
        "parameters": {name: _tensor(p) for name, p in model.named_parameters()},
        "outputs": {
            **{f"hidden_states_{i}": _tensor(h) for i, h in enumerate(out.hidden_states)},
            "last_hidden_state": _tensor(out.last_hidden_state),
            "logits": _tensor(out.logits),
            "loss": _tensor(out.loss.reshape(1)),
        },
    }


FIXTURES: dict[str, Callable[[], dict[str, Any]]] = {
    "attention_small.json": attention_fixture,
    "model_tiny.json": model_fixture,
    "rope_small.json": rope_fixture,
    "swiglu_small.json": swiglu_fixture,
}


def render(builder: Callable[[], dict[str, Any]]) -> str:
    return json.dumps(builder(), indent=1) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if fixtures are stale")
    args = parser.parse_args()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    stale = []
    for name, builder in FIXTURES.items():
        path = FIXTURE_DIR / name
        text = render(builder)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(name)
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(FIXTURE_DIR.parents[1])} ({len(text)} bytes)")
    if stale:
        print("stale fixtures:", ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
