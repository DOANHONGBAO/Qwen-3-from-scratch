"""Small, dependency-free system probes used by the M2 scripts (Linux /proc based).

Memory terms (do not confuse them):

* ``MemAvailable`` - what the kernel estimates can be allocated without swapping.
* RSS ("resident set size") - memory this process currently holds in RAM.
* Peak RSS (``VmHWM``) - the largest RSS so far; the number that decides OOM.
  It can be reset on Linux by writing ``5`` to ``/proc/self/clear_refs``.

Model weight size is NOT peak process memory: peak = weights + activations +
temporary copies during loading + framework overhead.
"""

from __future__ import annotations

import os
import platform
import re
from pathlib import Path

__all__ = [
    "cgroup_memory_limit",
    "cpu_info",
    "library_versions",
    "mem_info",
    "peak_rss_bytes",
    "reset_peak_rss",
    "rss_breakdown",
    "rss_bytes",
    "system_summary",
]

GiB = 1024**3


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def mem_info() -> dict[str, int]:
    """``/proc/meminfo`` values in bytes (MemTotal, MemAvailable, SwapTotal, SwapFree, ...)."""
    text = _read("/proc/meminfo") or ""
    out = {}
    for line in text.splitlines():
        m = re.match(r"(\w+):\s+(\d+)\s*kB", line)
        if m:
            out[m.group(1)] = int(m.group(2)) * 1024
    return out


def _status_kb(field: str) -> int | None:
    text = _read("/proc/self/status") or ""
    m = re.search(rf"^{field}:\s+(\d+)\s*kB", text, re.MULTILINE)
    return int(m.group(1)) * 1024 if m else None


def rss_bytes() -> int | None:
    return _status_kb("VmRSS")


def peak_rss_bytes() -> int | None:
    return _status_kb("VmHWM")


def rss_breakdown() -> dict[str, int | None]:
    """Anonymous vs file-backed resident memory (bytes).

    Memory-mapped checkpoint pages count as ``RssFile``: they are part of RSS but the
    kernel can drop them under pressure; ``RssAnon`` (tensors, Python heap) cannot be
    dropped without swap.
    """
    return {k: _status_kb(k) for k in ("VmRSS", "RssAnon", "RssFile", "VmHWM")}


def reset_peak_rss() -> bool:
    """Reset VmHWM to the current RSS (Linux >= 4.0). Returns False if unsupported."""
    try:
        Path("/proc/self/clear_refs").write_text("5", encoding="ascii")
        return True
    except OSError:
        return False


def cgroup_memory_limit() -> int | None:
    """cgroup v2/v1 memory limit in bytes, or None when unlimited / unknown."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        text = _read(path)
        if text is None:
            continue
        text = text.strip()
        if text == "max":
            return None
        try:
            value = int(text)
        except ValueError:
            continue
        return None if value >= 1 << 60 else value
    return None


def cpu_info() -> dict:
    text = _read("/proc/cpuinfo") or ""
    model = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
    flags = re.search(r"^flags\s*:\s*(.+)$", text, re.MULTILINE)
    flag_set = set(flags.group(1).split()) if flags else set()
    interesting = ["avx", "avx2", "fma", "f16c", "avx512f", "avx512_bf16", "amx_bf16", "avx_vnni"]
    return {
        "model": model.group(1).strip() if model else platform.processor(),
        "logical_cpus": os.cpu_count(),
        "flags": {f: f in flag_set for f in interesting},
    }


def library_versions() -> dict[str, str | None]:
    import importlib

    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("torch", "transformers", "safetensors", "tokenizers", "huggingface_hub", "numpy"):
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:  # noqa: BLE001 - optional packages
            out[name] = None
    return out


def system_summary() -> dict:
    """Everything the M2 reports record about the machine they ran on."""
    mem = mem_info()
    info = {
        "platform": platform.platform(),
        "hostname_hidden": True,  # deliberately not recorded
        "cpu": cpu_info(),
        "mem_total_gib": round(mem.get("MemTotal", 0) / GiB, 2),
        "mem_available_gib": round(mem.get("MemAvailable", 0) / GiB, 2),
        "swap_total_gib": round(mem.get("SwapTotal", 0) / GiB, 2),
        "swap_free_gib": round(mem.get("SwapFree", 0) / GiB, 2),
        "cgroup_limit_gib": (
            round(cgroup_memory_limit() / GiB, 2) if cgroup_memory_limit() else None
        ),
        "versions": library_versions(),
    }
    try:
        import torch

        info["torch_threads"] = torch.get_num_threads()
        info["torch_interop_threads"] = torch.get_num_interop_threads()
        info["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass
    return info
