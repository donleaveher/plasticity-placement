from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

NOTEBOOK_DIR = (
    Path(__file__).parents[1] / "notebooks" / "p0d2h_cbr_budget_recovery_corrected"
)


def _module():
    path = NOTEBOOK_DIR / "build_p0d2hcbr_recovery_notebook.py"
    spec = importlib.util.spec_from_file_location("build_p0d2hcbr_recovery_notebook", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_recovery_notebook_is_colab_ready_and_parses(tmp_path: Path) -> None:
    module = _module()
    module.OUTPUT_DIR = tmp_path
    module.main()
    notebook = json.loads((tmp_path / module.NOTEBOOK_NAME).read_text())

    assert notebook["metadata"]["accelerator"] == "GPU"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"CBR-recovery-cell-{index}")


def test_recovery_notebook_has_one_safe_control_cell_and_ordered_lifecycle() -> None:
    notebook = _module().build_notebook()
    code_cells = [
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert len(code_cells) == 4
    config, controls, lifecycle, results = code_cells
    stage_names = (
        "RUN_INSPECT_DEVIATION",
        "RUN_AUTHORIZE_DEVIATION",
        "RUN_EVALUATE_V1",
        "RUN_AGGREGATE_V1",
        "RUN_VERIFY_V1",
        "RUN_PLAN_V2",
        "RUN_AUTHORIZE_V2",
        "RUN_TRAIN_V2",
        "RUN_EVALUATE_V2",
        "RUN_AGGREGATE_V2",
        "RUN_VERIFY_V2",
    )
    assert "RUN_INSPECT_DEVIATION = True" in controls
    for name in stage_names[1:]:
        assert f"{name} = False" in controls
    for name in (*stage_names, "APPROVER", "TARGET_CURRICULUM", "TARGET_PLACEMENT", "TARGET_SEED"):
        assert f"{name} =" not in config
        assert f"{name} =" not in lifecycle
        assert f"{name} =" not in results
    positions = [lifecycle.index(f"if {name}:") for name in stage_names]
    assert positions == sorted(positions)
    for command in (
        "budget-deviation-template",
        "authorize-budget-deviation",
        "plan-corrected",
        "authorize",
        "train",
        "evaluate",
        "aggregate",
        "verify",
    ):
        assert f"'plasticity-p0d2hcbr', '{command}'" in lifecycle
    assert "EXPERIMENT_CODE_REVISION = 'f4a71efe6bc717d345843b025c95fceeb6071954'" in config
    assert "REQUESTED_RECOVERY_CODE_REVISION = '04437d33de7af53d6bdd408665a3be941c440965'" in config
    assert "CBR2_CODE_REVISION_LOCK" in config
    assert "if v2_stage_selected:" in lifecycle
    assert "if not RUN_PLAN_V2:" in lifecycle
    assert "placement_comparison_scope" in results


def test_recovery_builder_matches_checked_in_notebook() -> None:
    module = _module()
    checked_in = json.loads((NOTEBOOK_DIR / module.NOTEBOOK_NAME).read_text())

    assert checked_in == module.build_notebook()
