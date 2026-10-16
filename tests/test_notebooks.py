"""Smoke-test the explanatory notebooks by executing their code cells.

No Jupyter dependency: code cells are concatenated and run with ``exec`` in a
fresh namespace, which is enough to catch notebooks drifting from the package API.
"""

from __future__ import annotations

import json

import pytest

NOTEBOOKS = ["00_environment_and_config.ipynb", "01_rmsnorm.ipynb"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_code_cells_run(repo_root, name) -> None:
    nb = json.loads((repo_root / "notebooks" / name).read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert cells, f"{name} has no code cells"
    for cell in cells:
        assert not cell.get("outputs"), f"{name}: commit notebooks without outputs"
    source = "\n\n".join("".join(c["source"]) for c in cells)
    exec(compile(source, name, "exec"), {"__name__": "__notebook__"})
