from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

NOTEBOOK_DIR = (
    Path(__file__).parents[1] / "notebooks" / "p0d2h_cbr_resumable_evaluation"
)


def _module():
    path = NOTEBOOK_DIR / "build_p0d2hcbr_resumable_notebook.py"
    spec = importlib.util.spec_from_file_location(
        "build_p0d2hcbr_resumable_notebook", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resumable_notebook_is_colab_ready_and_parses(tmp_path: Path) -> None:
    module = _module()
    module.OUTPUT_DIR = tmp_path
    module.main()
    notebook = json.loads((tmp_path / module.NOTEBOOK_NAME).read_text())

    assert notebook["metadata"]["accelerator"] == "GPU"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"CBR-resume-cell-{index}")


def test_resumable_notebook_has_safe_controls_and_resume_lifecycle() -> None:
    notebook = _module().build_notebook()
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    assert len(code_cells) == 4
    config, controls, lifecycle, results = code_cells
    stages = (
        "RUN_INSPECT_RESUME",
        "RUN_AUTHORIZE_RESUME",
        "RUN_EVALUATE_RESUMABLE",
        "RUN_AGGREGATE",
        "RUN_VERIFY",
    )
    assert "RUN_INSPECT_RESUME = True" in controls
    for name in stages[1:]:
        assert f"{name} = False" in controls
    for name in (
        *stages,
        "APPROVER",
        "SHARD_SIZE",
        "TARGET_CURRICULUM",
        "TARGET_PLACEMENT",
        "TARGET_SEED",
    ):
        assert f"{name} =" not in config
        assert f"{name} =" not in lifecycle
        assert f"{name} =" not in results
    positions = [lifecycle.index(f"if {name}:") for name in stages]
    assert positions == sorted(positions)
    for command in (
        "resumable-evaluation-template",
        "authorize-resumable-evaluation",
        "evaluate-resumable",
        "aggregate",
        "verify",
    ):
        assert f"'plasticity-p0d2hcbr', '{command}'" in lifecycle
    expected_revision = (
        "REQUESTED_RESUMABLE_CODE_REVISION = "
        "'99462a4a7742b82f0cee395d4da891d1ed6e48db'"
    )
    assert expected_revision in config
    assert "PYTHONUNBUFFERED" in lifecycle
    assert "completed_shards" in lifecycle


def test_resumable_builder_matches_checked_in_notebook() -> None:
    module = _module()
    checked_in = json.loads((NOTEBOOK_DIR / module.NOTEBOOK_NAME).read_text())

    assert checked_in == module.build_notebook()
