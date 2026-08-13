from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).parents[1] / "notebooks" / "p0d2h_cbr_v2"


def _module():
    path = NOTEBOOK_DIR / "build_p0d2hcbr_v2_notebook.py"
    spec = importlib.util.spec_from_file_location("build_p0d2hcbr_v2_notebook", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_notebook_is_colab_ready_and_parses(tmp_path: Path) -> None:
    module = _module()
    module.OUTPUT_DIR = tmp_path
    module.main()
    notebook = json.loads((tmp_path / module.NOTEBOOK_NAME).read_text())

    assert notebook["metadata"]["accelerator"] == "GPU"
    assert module.COLAB_URL in "".join(notebook["cells"][0]["source"])
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"CBR-v2-cell-{index}")


def test_v2_notebook_has_safe_controls_and_ordered_lifecycle() -> None:
    notebook = _module().build_notebook()
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    assert len(code_cells) == 4
    config, controls, lifecycle, results = code_cells
    stages = (
        "RUN_PLAN_V2",
        "RUN_AUTHORIZE_V2",
        "RUN_TRAIN_V2",
        "RUN_STATUS_V2",
        "RUN_INSPECT_RESUME_V2",
        "RUN_AUTHORIZE_RESUME_V2",
        "RUN_EVALUATE_RESUMABLE_V2",
        "RUN_AGGREGATE_V2",
        "RUN_VERIFY_V2",
    )
    assert "RUN_PLAN_V2 = True" in controls
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
    assert "RUN_EVALUATE_V2" not in controls
    assert "RUN_EVALUATE_V2" not in lifecycle
    for command in (
        "plan-corrected",
        "authorize",
        "train",
        "resumable-evaluation-template",
        "authorize-resumable-evaluation",
        "evaluate-resumable",
        "aggregate",
        "verify",
    ):
        assert f"'plasticity-p0d2hcbr', '{command}'" in lifecycle
    assert "CBR-v2 state:" in lifecycle
    assert "completed_shards" in lifecycle
    assert "PYTHONUNBUFFERED" in lifecycle


def test_v2_notebook_preserves_experiment_identity_and_switches_revision() -> None:
    notebook = _module().build_notebook()
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    config, _, lifecycle, _ = code_cells
    assert (
        "SOURCE_CBR_RECOVERY_CODE_REVISION = "
        "'04437d33de7af53d6bdd408665a3be941c440965'"
    ) in config
    assert (
        "REQUESTED_CORRECTED_CODE_REVISION = "
        "'99462a4a7742b82f0cee395d4da891d1ed6e48db'"
    ) in config
    assert (
        "REQUESTED_RESUMABLE_CODE_REVISION = "
        "'99462a4a7742b82f0cee395d4da891d1ed6e48db'"
    ) in config
    assert "active_revision = corrected_revision if corrected_stage_selected else" in lifecycle
    assert "f'code-{corrected_revision[:10]}_cbr1-" in lifecycle
    assert "CBR2_PIPELINE_ATTEMPT = 'pipeline-cbr2r1'" in config
    assert "CBR2_RESUME_ATTEMPT = 'resume1'" in config
    assert "!= SOURCE_CBR_RECOVERY_CODE_REVISION" in lifecycle
    assert "CBR2_RESUME_EVALUATIONS_ROOT" in lifecycle


def test_v2_builder_matches_checked_in_notebook() -> None:
    module = _module()
    checked_in = json.loads((NOTEBOOK_DIR / module.NOTEBOOK_NAME).read_text())

    assert checked_in == module.build_notebook()
