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


def _tiny_model():
    """The fixture model shared by model_tiny.json and kv_cache_tiny.json."""
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
    return seed, cfg, model.eval()


def model_fixture() -> dict[str, Any]:
    """Whole Tiny Qwen3 forward: embeddings, every layer, final norm, logits, loss."""
    seed, cfg, model = _tiny_model()
    with torch.no_grad():
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


def kv_cache_fixture() -> dict[str, Any]:
    """Prefill + cached single-token decode on the model_tiny.json weights."""
    seed, cfg, model = _tiny_model()
    prompt = torch.tensor([[3, 14, 1, 5, 9]])
    decode_ids = [7, 2, 11]  # fed one at a time after the prefill (teacher forcing)
    outputs: dict[str, Any] = {}
    with torch.no_grad():
        cache = model.new_cache()
        out = model(prompt, past_key_values=cache, use_cache=True)
        outputs["prefill_logits"] = _tensor(out.logits)
        for layer in range(cfg.num_hidden_layers):
            k, v = cache.get(layer)
            outputs[f"prefill_key_layer{layer}"] = _tensor(k)
            outputs[f"prefill_value_layer{layer}"] = _tensor(v)
        for step, tok in enumerate(decode_ids):
            out = model(torch.tensor([[tok]]), past_key_values=cache, use_cache=True)
            outputs[f"decode_logits_step{step}"] = _tensor(out.logits)
        for layer in range(cfg.num_hidden_layers):
            k, v = cache.get(layer)
            outputs[f"final_key_layer{layer}"] = _tensor(k)
            outputs[f"final_value_layer{layer}"] = _tensor(v)
        full = model(torch.tensor([prompt[0].tolist() + decode_ids])).logits
        outputs["full_forward_logits"] = _tensor(full)
    n_prompt = prompt.shape[1]
    return {
        "component": "kv_cache_decode",
        "description": (
            "Weights = model_tiny.json (same seed). 1) prefill: input_ids = prompt, "
            "position_ids 0..4, cache empty -> prefill_logits and per-layer K/V. "
            "2) decode: feed decode_ids one at a time, position_ids = 5, 6, 7 "
            "(= cache length before the step), attending over all cached keys "
            "-> decode_logits_step{i}. Cache layout per layer: key/value "
            "[B, num_key_value_heads, T_cached, head_dim], row-major, keys stored AFTER "
            "QK-Norm and RoPE, values raw. full_forward_logits = uncached forward over "
            "prompt + decode_ids: its last 3 rows equal the decode logits up to rounding."
        ),
        "seed": seed,
        "config": cfg.to_dict(),
        "tolerance": {"rtol": 1e-5, "atol": 1e-5},
        "inputs": {
            "prompt_ids": prompt.tolist(),
            "prompt_position_ids": [list(range(n_prompt))],
            "decode_ids": decode_ids,
            "decode_position_ids": [n_prompt + i for i in range(len(decode_ids))],
        },
        "parameters": {},
        "outputs": outputs,
    }


def sampling_fixture() -> dict[str, Any]:
    """Deterministic parts of token selection: temperature, top-k, top-p, greedy."""
    from qwen3_scratch.generation import apply_top_k, apply_top_p

    gen = torch.Generator().manual_seed(11)
    logits = (2.0 * torch.randn(2, 10, generator=gen)).to(torch.float32)
    temperature, k, p = 0.7, 4, 0.8
    scaled = logits / temperature
    probs = lambda z: torch.softmax(z, dim=-1)  # noqa: E731
    return {
        "component": "sampling_filters",
        "description": (
            "Rows of logits [B, V] (float32). Order: logits / temperature -> top-k "
            "(keep values >= k-th largest; ties kept) -> top-p (drop a token when the "
            "probability mass of strictly better tokens is already >= p; top-1 always "
            "kept) -> softmax. Removed tokens have probability exactly 0. greedy_ids = "
            "argmax of the raw logits (lowest index on ties). The random draw "
            "(torch.multinomial) is NOT part of the contract: RNGs differ across languages."
        ),
        "seed": 11,
        "config": {"temperature": temperature, "top_k": k, "top_p": p},
        "tolerance": {"rtol": 1e-6, "atol": 1e-7},
        "inputs": {"logits": _tensor(logits)},
        "parameters": {},
        "outputs": {
            "probs_temperature": _tensor(probs(scaled)),
            "probs_top_k": _tensor(probs(apply_top_k(scaled, k))),
            "probs_top_p": _tensor(probs(apply_top_p(scaled, p))),
            "probs_top_k_top_p": _tensor(probs(apply_top_p(apply_top_k(scaled, k), p))),
            "greedy_ids": _tensor(logits.argmax(-1).to(torch.float32)),
        },
    }


def weight_map_fixture() -> dict[str, Any]:
    """Checkpoint tensor names and shapes for Qwen3-0.6B-Base (the loader contract)."""
    from qwen3_scratch.config import QWEN3_0_6B_BASE
    from qwen3_scratch.weight_loader import build_name_map, expected_shapes

    shapes = expected_shapes(QWEN3_0_6B_BASE)
    return {
        "component": "checkpoint_name_map",
        "description": (
            "Every tensor of the official Qwen/Qwen3-0.6B-Base model.safetensors "
            "(revision da87bfb), in loading order, with its shape; all stored as BF16, "
            "row-major, nn.Linear weights as [out_features, in_features] (no transpose). "
            "lm_head.weight is absent: tied to model.embed_tokens.weight."
        ),
        "seed": None,
        "config": QWEN3_0_6B_BASE.to_dict(),
        "tolerance": {"rtol": 0.0, "atol": 0.0},
        "inputs": {
            "tensors": [[n, list(shapes[n])] for n in build_name_map(QWEN3_0_6B_BASE)],
            "num_values": sum(torch.Size(s).numel() for s in shapes.values()),
        },
        "parameters": {},
        "outputs": {},
    }


FIXTURES: dict[str, Callable[[], dict[str, Any]]] = {
    "attention_small.json": attention_fixture,
    "checkpoint_name_map.json": weight_map_fixture,
    "kv_cache_tiny.json": kv_cache_fixture,
    "model_tiny.json": model_fixture,
    "sampling_small.json": sampling_fixture,
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
