"""Smoke-test the explanatory notebooks by executing their code cells.

No Jupyter dependency: code cells are concatenated and run with ``exec`` in a
fresh namespace, which is enough to catch notebooks drifting from the package API.
Every ``notebooks/*.ipynb`` is collected automatically.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted(
    p.name for p in (Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb")
)


# Notebooks that need optional extras (the hf extra provides safetensors/tokenizers).
REQUIRES = {"07_pretrained_qwen3.ipynb": ["safetensors", "tokenizers"]}


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_code_cells_run(repo_root, name, monkeypatch) -> None:
    for module in REQUIRES.get(name, []):
        pytest.importorskip(module, reason=f"{name} needs {module} (hf extra)")
    monkeypatch.setenv("MPLBACKEND", "Agg")  # plots render off-screen if matplotlib exists
    nb = json.loads((repo_root / "notebooks" / name).read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert cells, f"{name} has no code cells"
    for cell in cells:
        assert not cell.get("outputs"), f"{name}: commit notebooks without outputs"
    source = "\n\n".join("".join(c["source"]) for c in cells)
    exec(compile(source, name, "exec"), {"__name__": "__notebook__"})
