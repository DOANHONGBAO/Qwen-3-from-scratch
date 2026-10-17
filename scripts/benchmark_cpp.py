#!/usr/bin/env python
"""CPU benchmark of the C++ Qwen3 (``qwen3_bench``) on a real checkpoint (M4).

    python scripts/benchmark_cpp.py --download                    # pinned Qwen3-0.6B-Base
    python scripts/benchmark_cpp.py --model-dir <dir> --threads 1,2,4

The script configures and builds a separate optimised copy of the C++ code in
``build/cpp-native`` (``-DQWEN3_NATIVE=ON``: ``-march=native``, OpenMP), then runs
``qwen3_bench`` once per (threads, prompt length) case. Each run: 1 warm-up, then
``--repeats`` timed repetitions of prefill + (N - 1) cached decode steps; medians are
reported, never extrapolated. Prompts are synthetic token ids (no tokenizer), greedy
decoding, EOS ignored, checkpoint load time reported separately.

Output: ``results/m4/benchmark_cpp.json`` and ``.csv`` (small; safe to commit).
Time shares come from the built-in profiler (``cpp/include/qwen3/profiler.hpp``).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BUILD = REPO / "build" / "cpp-native"
DEFAULT_OUT = REPO / "results" / "m4" / "benchmark_cpp.json"
GiB = 1024**3


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def build(build_dir: Path) -> Path:
    gen = (
        ["-G", "Ninja"]
        if subprocess.run(["which", "ninja"], capture_output=True).returncode == 0
        else []
    )
    run(
        [
            "cmake",
            "-S",
            str(REPO / "cpp"),
            "-B",
            str(build_dir),
            *gen,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DQWEN3_NATIVE=ON",
            "-DQWEN3_BUILD_TESTS=OFF",
        ],
        stdout=subprocess.DEVNULL,
    )
    run(["cmake", "--build", str(build_dir), "--parallel", "--target", "qwen3_bench"])
    return build_dir / "qwen3_bench"


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def compiler(build_dir: Path) -> str | None:
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return None
    for line in cache.read_text().splitlines():
        if line.startswith("CMAKE_CXX_COMPILER:"):
            exe = line.split("=", 1)[1]
            out = subprocess.run([exe, "--version"], capture_output=True, text=True)
            return out.stdout.splitlines()[0] if out.stdout else exe
    return None


def git_commit() -> str | None:
    out = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    )
    return out.stdout.strip() or None


def shares(profile: dict[str, float]) -> dict[str, float]:
    """Exclusive time per part (attention.core is nested inside layer.attention)."""
    core = profile.get("attention.core", 0.0)
    parts = {
        "attention_projections": profile.get("layer.attention", 0.0) - core,
        "attention_core": core,
        "mlp": profile.get("layer.mlp", 0.0),
        "lm_head": profile.get("lm_head", 0.0),
        "norms": profile.get("layer.norm", 0.0),
    }
    total = sum(parts.values()) or 1.0
    return {k: round(v / total, 4) for k, v in parts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir")
    parser.add_argument("--download", action="store_true", help="use the pinned Qwen3-0.6B-Base")
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD))
    parser.add_argument("--binary", help="use this qwen3_bench instead of building one")
    parser.add_argument("--threads", default="1,2,4", help="comma-separated thread counts")
    parser.add_argument("--prompt-lens", default="16,128", help="comma-separated")
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true", help="skip the free-RAM guard")
    args = parser.parse_args()

    if args.download:
        from qwen3_scratch.pretrained import download_checkpoint

        args.model_dir = str(download_checkpoint())
    if not args.model_dir:
        parser.error("pass --model-dir or --download")

    from qwen3_scratch.config import Qwen3Config
    from qwen3_scratch.sysinfo import mem_info

    cfg = Qwen3Config.from_json(Path(args.model_dir) / "config.json")
    need = cfg.estimate_num_parameters() * 4 + int(0.5 * GiB)
    avail = mem_info().get("MemAvailable", 0)
    print(f"C++ model needs ~{need / GiB:.2f} GiB; MemAvailable {avail / GiB:.2f} GiB")
    if need > 0.85 * avail and not args.force:
        sys.exit("not enough free RAM (15% margin). Close other programs or pass --force.")

    build_dir = Path(args.build_dir)
    binary = Path(args.binary) if args.binary else build(build_dir)
    threads = [int(t) for t in args.threads.split(",")]
    prompt_lens = [int(p) for p in args.prompt_lens.split(",")]

    cases = []
    for t in threads:
        for p in prompt_lens:
            cmd = [
                str(binary),
                "--model-dir",
                args.model_dir,
                "--threads",
                str(t),
                "--prompt-len",
                str(p),
                "--new-tokens",
                str(args.new_tokens),
                "--repeats",
                str(args.repeats),
            ]
            out = run(cmd, stdout=subprocess.PIPE).stdout
            row = json.loads(next(line for line in out.splitlines() if line.startswith("{")))
            row["time_shares"] = shares(row.pop("profile_seconds"))
            row.pop("profile_total_seconds", None)
            cases.append(row)
            print(
                f"threads {row['threads']}, prompt {p}: prefill "
                f"{row['prefill_tokens_per_second']:.1f} tok/s, decode "
                f"{row['decode_tokens_per_second']:.2f} tok/s, "
                f"peak RSS {row['peak_rss_gib']:.2f} GiB, "
                f"shares {row['time_shares']}"
            )

    report = {
        "status": "RUN",
        "model_dir_name": Path(args.model_dir).name,
        "build": {
            "flags": "Release, -march=native, OpenMP",
            "compiler": compiler(build_dir),
            "commit": git_commit(),
        },
        "machine": {
            "cpu": cpu_name(),
            "logical_cpus": os.cpu_count(),
            "platform": platform.platform(),
            "mem_total_gib": round(mem_info().get("MemTotal", 0) / GiB, 2),
        },
        "settings": {
            "new_tokens": args.new_tokens,
            "repeats": args.repeats,
            "statistic": "median",
            "warmup": 1,
            "prompt": "synthetic ids, greedy, no EOS",
        },
        "cases": cases,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "threads",
                "prompt_len",
                "new_tokens",
                "prefill_tok_s",
                "decode_tok_s",
                "peak_rss_gib",
                "load_s",
            ]
        )
        for c in cases:
            w.writerow(
                [
                    c["threads"],
                    c["prompt_len"],
                    c["new_tokens"],
                    round(c["prefill_tokens_per_second"], 2),
                    round(c["decode_tokens_per_second"], 2),
                    round(c["peak_rss_gib"], 3),
                    round(c["load_seconds"], 2),
                ]
            )
    print(f"wrote {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
