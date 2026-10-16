"""Tier-2 (GPU) and tier-3 (pretrained weights) smoke tests.

Both are skipped automatically on the local CPU machine and in CI. They exist
now so that the test tiers are wired up before the model is implemented.
"""

from __future__ import annotations

import json

import pytest

from qwen3_scratch.config import QWEN3_0_6B_BASE, Qwen3Config


@pytest.mark.gpu
def test_cuda_matmul_matches_cpu() -> None:
    import torch

    gen = torch.Generator().manual_seed(0)
    a = torch.randn(64, 64, generator=gen)
    b = torch.randn(64, 64, generator=gen)
    cpu = a @ b
    gpu = (a.cuda() @ b.cuda()).cpu()
    # TF32 may be enabled on Ampere+; tolerances chosen accordingly.
    torch.testing.assert_close(gpu, cpu, rtol=1e-2, atol=1e-2)


@pytest.mark.weights
def test_local_checkpoint_config_matches_preset(weights_dir) -> None:
    """Checks a locally downloaded Qwen3-0.6B-Base directory. Never downloads."""
    config_path = weights_dir / "config.json"
    assert config_path.is_file(), f"missing {config_path}"
    hf = json.loads(config_path.read_text())
    assert Qwen3Config.from_hf_config(hf, dtype="bfloat16") == QWEN3_0_6B_BASE
