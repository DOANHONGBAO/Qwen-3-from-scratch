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

from qwen3_scratch.layers import SwiGLUFeedForward

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


FIXTURES: dict[str, Callable[[], dict[str, Any]]] = {
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
