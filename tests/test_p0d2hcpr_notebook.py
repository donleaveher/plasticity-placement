from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).parents[1] / "notebooks" / "p0d2h_composition_preserving_remediation"


def _module():
    path = NOTEBOOK_DIR / "build_p0d2hcpr_notebook.py"
    spec = importlib.util.spec_from_file_location("build_p0d2hcpr_notebook", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_notebook_is_colab_ready_and_all_code_cells_parse(tmp_path: Path) -> None:
    module = _module()
    module.OUTPUT_DIR = tmp_path
    module.main()
    notebook = json.loads((tmp_path / module.NOTEBOOK_NAME).read_text())
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"CPR-cell-{index}")


def test_notebook_has_safe_defaults_and_complete_lifecycle() -> None:
    notebook = _module().build_notebook()
    all_code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "RUN_PLAN = True" in all_code
    assert "CPR_PIPELINE_ATTEMPT = 'pipeline-cpr2'" in all_code
    assert "CPR_ATTEMPT = 'cpr2'" in all_code
    assert "completed.stderr" in all_code
    for value in ("RUN_AUTHORIZE", "RUN_TRAIN", "RUN_QUALIFY"):
        assert f"{value} = False" in all_code
    for action in ("plan", "authorize", "train", "qualify"):
        assert f"'plasticity-p0d2hcpr', '{action}'" in all_code
    assert "--same-runtime-summary" in all_code
    assert "--source-code-revision-lock" in all_code
    assert "mappings_per_adapter_authorized" in all_code
