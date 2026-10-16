#!/usr/bin/env python
"""Measure this machine and estimate Qwen3-0.6B-Base memory needs (M2.1/M2.2).

    python scripts/hardware_report.py            # prints a table, writes results/m2/hardware.json

No download, no model load: the estimates are analytic (from the config).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from qwen3_scratch.config import QWEN3_0_6B_BASE
from qwen3_scratch.kv_cache import kv_cache_bytes
from qwen3_scratch.sysinfo import GiB, system_summary

REPO = Path(__file__).resolve().parents[1]

# Overheads used for the estimates (rationale in docs/hardware_assessment.md):
FRAMEWORK_OVERHEAD = 0.45 * GiB  # Python + torch CPU runtime + allocator slack (measured ~0.3-0.5)
SAFETY_FRACTION = 0.85  # never plan to use more than 85% of MemAvailable


def estimates() -> dict:
    cfg = QWEN3_0_6B_BASE
    n = cfg.estimate_num_parameters()  # 596,049,920 (tied)
    v, h = cfg.vocab_size, cfg.hidden_size
    out = {"num_parameters": n}
    for name, b in (("bfloat16", 2), ("float32", 4)):
        weights = n * b
        largest = v * h * b  # embedding: largest single tensor (converted one at a time)
        out[name] = {
            "weights_gib": round(weights / GiB, 3),
            "load_peak_gib": round((weights + largest + FRAMEWORK_OVERHEAD) / GiB, 3),
            # activations for T tokens are small next to weights; logits dominate: T * V * 4 B
            "logits_per_1k_tokens_gib": round(1024 * v * 4 / GiB, 3),
            "two_models_gib": round((2 * weights + largest + FRAMEWORK_OVERHEAD) / GiB, 3),
            "kv_cache_mib": {
                str(t): round(kv_cache_bytes(cfg, t, bytes_per_element=b) / 2**20, 1)
                for t in (128, 512, 2048, 8192, 32768)
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(REPO / "results" / "m2" / "hardware.json"))
    args = parser.parse_args()

    info = system_summary()
    disk = shutil.disk_usage(Path.home())
    info["disk_free_gib_home"] = round(disk.free / GiB, 1)
    hf_cache = Path.home() / ".cache" / "huggingface"
    info["hf_cache_gib"] = (
        round(sum(f.stat().st_size for f in hf_cache.rglob("*") if f.is_file()) / GiB, 3)
        if hf_cache.exists()
        else 0.0
    )
    est = estimates()
    avail = info["mem_available_gib"] * SAFETY_FRACTION
    plan = {}
    for dtype in ("float32", "bfloat16"):
        e = est[dtype]
        plan[dtype] = {
            "one_model_fits": e["load_peak_gib"] <= avail,
            "two_models_fit": e["two_models_gib"] <= avail,
        }
    info["estimates"] = est
    info["plan"] = {"usable_gib": round(avail, 2), **plan}

    print(
        json.dumps(
            {
                k: info[k]
                for k in (
                    "cpu",
                    "mem_total_gib",
                    "mem_available_gib",
                    "swap_total_gib",
                    "cgroup_limit_gib",
                    "disk_free_gib_home",
                    "hf_cache_gib",
                )
            },
            indent=2,
        )
    )
    print(f"\nusable RAM for planning ({SAFETY_FRACTION:.0%} of MemAvailable): {avail:.2f} GiB")
    for dtype in ("float32", "bfloat16"):
        e = est[dtype]
        print(
            f"{dtype:9s} weights {e['weights_gib']:.2f} GiB | load peak ~{e['load_peak_gib']:.2f} "
            f"GiB | two models ~{e['two_models_gib']:.2f} GiB | one fits: "
            f"{plan[dtype]['one_model_fits']} | two fit: {plan[dtype]['two_models_fit']}"
        )
        print(f"          KV cache MiB by context length: {e['kv_cache_mib']}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
