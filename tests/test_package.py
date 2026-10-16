"""Package import checks and environment smoke tests (tier 1: local CPU)."""

from __future__ import annotations

import importlib
import re
import subprocess
import sys

import pytest

import qwen3_scratch
from qwen3_scratch.env import collect_environment, python_version_ok, torch_cpu_sanity_check


@pytest.mark.parametrize("module", ["qwen3_scratch", "qwen3_scratch.config", "qwen3_scratch.env"])
def test_modules_import(module: str) -> None:
    importlib.import_module(module)


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", qwen3_scratch.__version__)


def test_config_module_does_not_import_torch() -> None:
    # The config must stay framework-independent so it can mirror the C++ struct.
    code = "import sys, qwen3_scratch.config; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_python_version_policy() -> None:
    assert python_version_ok((3, 11))
    assert python_version_ok((3, 12))
    assert not python_version_ok((3, 10))


def test_torch_cpu_sanity_is_deterministic() -> None:
    assert torch_cpu_sanity_check(seed=123) == torch_cpu_sanity_check(seed=123)
    assert torch_cpu_sanity_check(seed=1) != torch_cpu_sanity_check(seed=2)


def test_collect_environment_has_expected_keys() -> None:
    info = collect_environment()
    for key in ("python", "platform", "torch", "cuda_available"):
        assert key in info
    assert info["torch"] is not None


def test_check_env_script_succeeds(repo_root) -> None:
    script = repo_root / "scripts" / "check_env.py"
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "FAILED" not in out.stdout
