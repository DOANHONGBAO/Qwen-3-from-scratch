"""The committed C++ parity fixtures must match what the Python modules produce.

Inputs and parameters must be bit-identical (they come from a seeded generator);
outputs are compared with the tolerance stored in the fixture, because matmul
kernels may round differently on another CPU.
"""

from __future__ import annotations

import importlib.util
import json

import pytest
import torch
from torch.testing import assert_close


def _load_exporter(repo_root):
    spec = importlib.util.spec_from_file_location(
        "export_fixtures", repo_root / "scripts" / "export_fixtures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_tensor(entry: dict) -> torch.Tensor:
    t = torch.tensor(entry["data"], dtype=torch.float64)
    assert list(t.shape) == entry["shape"]
    return t


def _fixture_names(repo_root) -> list[str]:
    return sorted(_load_exporter(repo_root).FIXTURES)


def pytest_generate_tests(metafunc):
    if "fixture_name" in metafunc.fixturenames:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        metafunc.parametrize("fixture_name", _fixture_names(root))


def test_fixture_is_current(repo_root, fixture_name) -> None:
    exporter = _load_exporter(repo_root)
    path = repo_root / "tests" / "fixtures" / fixture_name
    assert path.is_file(), f"run: python scripts/export_fixtures.py ({fixture_name} missing)"
    committed = json.loads(path.read_text(encoding="utf-8"))
    fresh = exporter.FIXTURES[fixture_name]()
    assert committed["config"] == fresh["config"]
    for group in ("inputs", "parameters"):
        assert committed[group].keys() == fresh[group].keys()
        for key in committed[group]:
            assert committed[group][key] == fresh[group][key], f"{group}.{key} changed"
    tol = committed["tolerance"]
    for key in committed["outputs"]:
        assert_close(
            _as_tensor(fresh["outputs"][key]), _as_tensor(committed["outputs"][key]), **tol
        )


@pytest.mark.parametrize("entry_kind", ["inputs", "parameters", "outputs"])
def test_fixture_values_are_exact_float32(repo_root, fixture_name, entry_kind) -> None:
    data = json.loads((repo_root / "tests" / "fixtures" / fixture_name).read_text())
    for entry in data[entry_kind].values():
        if entry["dtype"] != "float32":
            continue
        t = _as_tensor(entry)
        assert torch.equal(t.to(torch.float32).to(torch.float64), t)
