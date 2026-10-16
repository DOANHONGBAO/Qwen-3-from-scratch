"""Kaggle kernel: run the M2 parity check and benchmark with THIS repository's model.

Fallback for machines that cannot run the experiments locally (see
docs/hardware_assessment.md). The kernel:

1. clones the repository at a pinned commit (QWEN3_SCRATCH_COMMIT below; set by
   ``kaggle/prepare_kernel.sh qwen3_inference`` from your local HEAD),
2. installs the pinned Hugging Face reference libraries (torch is Kaggle's own build;
   its version is recorded in every report),
3. downloads the pinned Qwen/Qwen3-0.6B-Base revision (internet must be enabled;
   no token needed for this public model),
4. runs scripts/hardware_report.py, scripts/verify_pretrained.py (float32) and
   scripts/benchmark_generation.py on CPU, or on whatever the kernel was given,
5. copies the small JSON/CSV/PNG results to /kaggle/working/results.

Inference always runs on qwen3_scratch.Qwen3ForCausalLM; transformers is only the
reference in the parity phase. No credentials are read or written.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/DOANHONGBAO/Qwen-3-from-scratch.git"
QWEN3_SCRATCH_COMMIT = "COMMIT_SHA"  # replaced by prepare_kernel.sh
PINNED = [
    "transformers==5.20.0",
    "safetensors==0.8.0",
    "tokenizers==0.23.3",
    "huggingface-hub==2.2.0",
]
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "kaggle_run"
SRC = WORK / "src_repo"
OUT = WORK / "results"


def sh(*cmd: str, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    if not SRC.exists():
        sh("git", "clone", "--quiet", REPO_URL, str(SRC))
    sh("git", "checkout", "--quiet", QWEN3_SCRATCH_COMMIT, cwd=SRC)
    sh(sys.executable, "-m", "pip", "install", "--quiet", *PINNED)
    sh(sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "-e", str(SRC))
    OUT.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    sh(py, "scripts/hardware_report.py", "--out", str(OUT / "hardware.json"), cwd=SRC)
    code = "from qwen3_scratch.pretrained import download_checkpoint as d; print(d())"
    ckpt = (
        subprocess.run([py, "-c", code], check=True, capture_output=True, text=True, cwd=SRC)
        .stdout.strip()
        .splitlines()[-1]
    )
    sh(
        py,
        "scripts/verify_pretrained.py",
        "--model-dir",
        ckpt,
        "--dtype",
        "float32",
        "--out",
        str(OUT),
        cwd=SRC,
    )
    sh(
        py,
        "scripts/benchmark_generation.py",
        "--model-dir",
        ckpt,
        "--dtype",
        "float32",
        "--tag",
        "kaggle_fp32",
        "--out-dir",
        str(OUT),
        "--plot",
        cwd=SRC,
    )
    for f in OUT.glob("*.safetensors"):  # reference tensors are not needed as outputs
        f.unlink()
    (OUT / "commit.txt").write_text(QWEN3_SCRATCH_COMMIT + "\n")
    shutil.make_archive(str(WORK / "results"), "zip", OUT)
    print("done:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
