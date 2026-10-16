"""The educational CLI demo must run on CPU and print the key sections."""

from __future__ import annotations

import subprocess
import sys


def test_demo_tiny_qwen3_runs(repo_root) -> None:
    out = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "demo_tiny_qwen3.py"), "--new-tokens", "4"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    for text in (
        "RANDOM weights",
        "parameters: total 426,752",
        "logits",
        "(1, 13, 256)",
        "loss =",
        "greedy generation, 4 new tokens",
        "deterministic (same result twice): True",
    ):
        assert text in out.stdout, text
