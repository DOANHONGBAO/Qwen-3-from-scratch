#!/usr/bin/env python
"""Lightweight environment validation for qwen3-scratch.

Usage:
    python scripts/check_env.py          # human-readable report
    python scripts/check_env.py --json   # machine-readable report

Exit code 0 means every REQUIRED check passed. GPU absence is reported but is
not a failure: local development is CPU-only by design.
No network access, no credentials, no model downloads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

EXPECTED_ENV = "qwen3-scratch"
SUPPORTED_PYTHON = ((3, 11), (3, 13))  # inclusive range, matches pyproject requires-python


def preflight() -> list[str]:
    """Standard-library-only checks that explain the most common setup mistakes.

    Runs before anything from the project is imported, so it works even when
    the package or torch is missing. Returns a list of problems (empty = OK).
    """
    problems: list[str] = []
    env_name = os.environ.get("CONDA_DEFAULT_ENV")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    print(f"python executable : {sys.executable}")
    print(f"python version    : {sys.version.split()[0]}")
    print(f"conda env         : {env_name} ({conda_prefix})")

    if env_name != EXPECTED_ENV:
        problems.append(
            f"Active conda env is {env_name!r}, expected {EXPECTED_ENV!r} "
            "(note: hyphen, not underscore). Run: conda activate qwen3-scratch"
        )
    if conda_prefix and not os.path.realpath(sys.prefix).startswith(os.path.realpath(conda_prefix)):
        problems.append(
            f"`python` is not from the active conda env (sys.prefix={sys.prefix}). "
            "The env probably has no Python installed, so the system Python is used. "
            "Recreate it: conda env create -f environment.yml"
        )
    low, high = SUPPORTED_PYTHON
    if not low <= sys.version_info[:2] <= high:
        problems.append(
            f"Python {sys.version.split()[0]} is unsupported; this project needs "
            f"{low[0]}.{low[1]}-{high[0]}.{high[1]} (environment.yml pins 3.11)."
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args()

    results: list[tuple[str, str, str]] = []  # (status, check, detail)

    def record(ok: bool, name: str, detail: str, required: bool = True) -> None:
        status = "PASSED" if ok else ("FAILED" if required else "INFO")
        results.append((status, name, detail))

    problems = preflight()
    for problem in problems:
        print(f"WARNING {problem}")

    try:
        import qwen3_scratch
        from qwen3_scratch.config import PRESETS
        from qwen3_scratch.env import (
            collect_environment,
            python_version_ok,
            torch_cpu_sanity_check,
        )
    except ImportError as exc:
        print(f"FAILED  import qwen3_scratch: {exc}")
        if problems:
            print("Fix the warnings above first (wrong env / wrong Python), then:")
            print("  conda env create -f environment.yml   # or: bash scripts/setup_env.sh")
            print("  conda activate qwen3-scratch")
        else:
            print("The interpreter is correct but the project is not installed in it. Run:")
            print(
                "  python -m pip install torch==2.14.1 "
                "--index-url https://download.pytorch.org/whl/cpu"
            )
            print('  python -m pip install -e ".[dev]"')
        print("  python scripts/check_env.py")
        return 1

    info = collect_environment()
    record(python_version_ok(), "python >= 3.11", info["python"])
    record(
        info.get("conda_env") == "qwen3-scratch",
        "conda env 'qwen3-scratch' active",
        str(info.get("conda_env")),
        required=False,
    )
    record(True, "import qwen3_scratch", qwen3_scratch.__version__)
    record(info.get("torch") is not None, "import torch", str(info.get("torch")))

    if info.get("torch") is not None:
        first = torch_cpu_sanity_check(seed=0)
        second = torch_cpu_sanity_check(seed=0)
        record(first == second, "torch CPU deterministic matmul", f"{first:.6f}")
        record(
            bool(info.get("cuda_available")),
            "CUDA available (optional)",
            str(info.get("cuda_device_name", "no CUDA device - expected on local CPU machine")),
            required=False,
        )

    for name, cfg in PRESETS.items():
        record(True, f"config preset '{name}'", f"{cfg.estimate_num_parameters():,} params")

    record(info.get("cmake") is not None, "cmake (optional, M3)", str(info.get("cmake")), False)
    record(info.get("cxx") is not None, "C++ compiler (optional, M3)", str(info.get("cxx")), False)

    if args.json:
        payload = {
            "environment": info,
            "checks": [{"status": s, "check": c, "detail": d} for s, c, d in results],
        }
        print(json.dumps(payload, indent=2))
    else:
        for status, name, detail in results:
            print(f"{status:<7} {name:<40} {detail}")

    return 1 if any(s == "FAILED" for s, _, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())
