from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

NOTEBOOK_DIR = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "p0d2_hard_probe"
)


def _load_generator() -> ModuleType:
    path = NOTEBOOK_DIR / "build_p0d2h_notebook.py"
    spec = importlib.util.spec_from_file_location(
        "build_p0d2h_notebook",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p0d2h_notebook_matches_generator_and_has_valid_python() -> None:
    generator = _load_generator()
    path = NOTEBOOK_DIR / "p0d2_hard_probe_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook == generator.build_notebook()
    assert notebook["metadata"]["colab"]["name"] == path.name
    opening = "".join(notebook["cells"][0]["source"])
    assert (
        "https://colab.research.google.com/github/donleaveher/"
        "plasticity-placement/blob/agent%2Fadd-lora-evaluation/"
        "notebooks/p0d2_hard_probe/p0d2_hard_probe_colab.ipynb"
        in opening
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(
                "".join(cell["source"]),
                filename=f"{path.name}:cell-{index}",
            )


def test_p0d2h_notebook_is_read_only_and_uses_independent_namespace() -> None:
    notebook = json.loads(
        (NOTEBOOK_DIR / "p0d2_hard_probe_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert source.count("RUN_HARD_PROBE = False") == 1
    assert "/plasticity-p0d/hard-probe/v1/pipelines" in source
    assert "budget_match-" in source
    assert "'plasticity-p0d2h', action" in source
    assert "'plasticity-p0d2', 'run'" not in source
    assert "'plasticity-train-lora'" not in source
    assert "plan['expected_unit_count'] != 288" in source
    assert "plan['expected_probe_row_count'] != 5376" in source
    assert "automatic_training_started" not in source
