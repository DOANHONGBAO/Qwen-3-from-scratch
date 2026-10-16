"""Environment introspection used by ``scripts/check_env.py`` and the smoke tests.

Collects versions and hardware facts without touching the network, credentials,
or model weights.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from typing import Any

MIN_PYTHON = (3, 11)


def collect_environment() -> dict[str, Any]:
    """Return a JSON-serialisable description of the runtime environment."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cmake": shutil.which("cmake"),
        "cxx": shutil.which("g++") or shutil.which("clang++"),
    }
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "torch_cuda_build": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "torch_num_threads": torch.get_num_threads(),
            }
        )
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:  # pragma: no cover - reported, not raised
        info["torch"] = None
    return info


def python_version_ok(version: tuple[int, int] | None = None) -> bool:
    version = version or sys.version_info[:2]
    return tuple(version) >= MIN_PYTHON


def torch_cpu_sanity_check(seed: int = 0) -> float:
    """Run a tiny deterministic CPU computation and return its scalar result.

    Two calls with the same seed must return the identical value; this checks
    that torch is importable, functional, and seedable.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(8, 8, generator=gen, dtype=torch.float32)
    b = torch.randn(8, 8, generator=gen, dtype=torch.float32)
    return float((a @ b).sum())
