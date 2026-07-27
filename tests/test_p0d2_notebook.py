from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "p0d2_budget_match"


def _load_generator() -> ModuleType:
    path = NOTEBOOK_DIR / "build_p0d2_notebook.py"
    spec = importlib.util.spec_from_file_location("build_p0d2_notebook", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p0d2_notebook_matches_generator_and_has_valid_python() -> None:
    generator = _load_generator()
    path = NOTEBOOK_DIR / "p0d2_budget_match_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook == generator.build_notebook()
    assert notebook["metadata"]["colab"]["name"] == path.name
    opening_markdown = "".join(notebook["cells"][0]["source"])
    assert (
        "https://colab.research.google.com/github/donleaveher/"
        "plasticity-placement/blob/agent%2Fadd-lora-evaluation/"
        "notebooks/p0d2_budget_match/p0d2_budget_match_colab.ipynb"
        in opening_markdown
    )
    assert "alt=\"Open In Colab\"" in opening_markdown
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(
                "".join(cell["source"]),
                filename=f"{path.name}:cell-{index}",
            )


def test_p0d2_notebook_freezes_matrix_and_independent_drive_namespace() -> None:
    notebook = json.loads(
        (NOTEBOOK_DIR / "p0d2_budget_match_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert source.count("RUN_BUDGET_MATCH = False") == 1
    assert "/plasticity-p0d/budget-match/v1/pipelines" in source
    assert "/plasticity-p0d/lora-locus/v1/pipelines" not in source
    assert "'plasticity-p0d2', action" in source
    assert "'plasticity-p0c', 'run'" not in source
    assert "SOURCE_P0C_MANIFEST" in source
    assert "'pipeline_version': 'p0d2-budget-match-v1'" in source
    assert "plan['expected_unit_count'] != 504" in source
    assert "plan['expected_probe_row_count'] != 27456" in source
    assert "unit.get('budget_validation', {}).get('within_tolerance')" in source
    assert "automatic_narrow_scan_started" not in source
