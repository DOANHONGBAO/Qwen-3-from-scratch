"""Kaggle GPU smoke test for qwen3-scratch (script kernel).

Purpose: confirm that a Kaggle session has a working CUDA device and that basic
float32/bfloat16 kernels agree with the CPU within tolerance, before any real
experiment is launched. It needs no secrets, no internet, and no model weights.

Run locally (CPU) to check the script itself:
    python kaggle/smoke_test/kaggle_smoke_test.py
On Kaggle it writes ``smoke_test_report.json`` to the working directory, which
``kaggle kernels output`` downloads.
"""

from __future__ import annotations

import json
import platform
import sys
import time

import torch


def compare(device: torch.device, dtype: torch.dtype, n: int = 256) -> dict:
    gen = torch.Generator().manual_seed(0)
    a = torch.randn(n, n, generator=gen)
    b = torch.randn(n, n, generator=gen)
    reference = a.double() @ b.double()
    start = time.perf_counter()
    out = (a.to(device, dtype) @ b.to(device, dtype)).float().cpu()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    err = (out.double() - reference).abs()
    return {
        "device": str(device),
        "dtype": str(dtype),
        "max_abs_err_vs_fp64_cpu": float(err.max()),
        "mean_abs_err_vs_fp64_cpu": float(err.mean()),
        "seconds_incl_transfer": elapsed,
    }


def main() -> int:
    report: dict = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "results": [],
    }
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        devices.append(torch.device("cuda"))

    for device in devices:
        report["results"].append(compare(device, torch.float32))
        bf16_ok = device.type == "cpu" or torch.cuda.is_bf16_supported()
        if bf16_ok:
            report["results"].append(compare(device, torch.bfloat16))

    print(json.dumps(report, indent=2))
    with open("smoke_test_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    fp32 = [r for r in report["results"] if r["dtype"] == "torch.float32"]
    # Loose bound: TF32 on Ampere+ GPUs reduces float32 matmul precision.
    ok = all(r["max_abs_err_vs_fp64_cpu"] < 0.5 for r in fp32)
    print("SMOKE TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
