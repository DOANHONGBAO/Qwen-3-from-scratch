"""Shared pytest configuration.

Test tiers (see docs/architecture.md, "Testing conventions"):

1. Local CPU tests  - unmarked; always run, including in GitHub Actions.
2. GPU tests        - ``@pytest.mark.gpu``; auto-skipped without CUDA (run on Kaggle).
3. Weight tests     - ``@pytest.mark.weights``; auto-skipped unless the environment
                      variable ``QWEN3_WEIGHTS_DIR`` points to an existing local
                      directory. Tests never download checkpoints.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    has_cuda = _cuda_available()
    weights_dir = os.environ.get("QWEN3_WEIGHTS_DIR")
    has_weights = bool(weights_dir) and Path(weights_dir).is_dir()

    skip_gpu = pytest.mark.skip(reason="CUDA not available (GPU tier runs on Kaggle)")
    skip_weights = pytest.mark.skip(reason="QWEN3_WEIGHTS_DIR not set to an existing directory")
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)
        if "weights" in item.keywords and not has_weights:
            item.add_marker(skip_weights)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def weights_dir() -> Path:
    return Path(os.environ["QWEN3_WEIGHTS_DIR"])


# ------------------------------------------------------------------ toy tokenizer
@pytest.fixture(scope="session")
def toy_tokenizer_dir(tmp_path_factory) -> Path:
    """A small byte-level BPE tokenizer with Qwen-like settings (NFC, no BOS, eos=pad).

    Used to test the adapter in CI without downloading the real 151k-token tokenizer.
    """
    pytest.importorskip("tokenizers", reason="tokenizers not installed (hf extra)")
    from qwen3_scratch.synthetic import build_toy_tokenizer

    return build_toy_tokenizer(tmp_path_factory.mktemp("toy_tokenizer"))
