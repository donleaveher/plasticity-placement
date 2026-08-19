from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

NOTEBOOK_DIR = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "p0d2h_route_remediation"
)


def _load_generator() -> ModuleType:
    path = NOTEBOOK_DIR / "build_p0d2hrr_notebook.py"
    spec = importlib.util.spec_from_file_location(
        "build_p0d2hrr_notebook",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_notebook() -> dict[str, object]:
    return json.loads(
        (
            NOTEBOOK_DIR / "p0d2h_route_remediation_colab.ipynb"
        ).read_text(encoding="utf-8")
    )


def test_notebook_matches_generator_and_code_cells_parse() -> None:
    generator = _load_generator()
    path = NOTEBOOK_DIR / "p0d2h_route_remediation_colab.ipynb"
    notebook = _load_notebook()
    assert notebook == generator.build_notebook()
    assert notebook["metadata"]["colab"]["name"] == path.name
    opening = "".join(notebook["cells"][0]["source"])
    assert (
        "https://colab.research.google.com/github/donleaveher/"
        "plasticity-placement/blob/agent%2Fadd-lora-evaluation/"
        "notebooks/p0d2h_route_remediation/"
        "p0d2h_route_remediation_colab.ipynb"
        in opening
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(
                "".join(cell["source"]),
                filename=f"{path.name}:cell-{index}",
            )
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_notebook_defaults_to_preflight_and_separates_all_gates() -> None:
    notebook = _load_notebook()
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    all_code = "\n".join(code_cells)
    assert "RUN_PLAN = True" in all_code
    assert "ADOPT_AUTHORIZATION = False" in all_code
    assert "RUN_TRAINING = False" in all_code
    assert "RUN_LOCKED_EVALUATION = False" in all_code
    assert "RUN_AGGREGATE = False" in all_code
    assert "requested_stage_count > 1" in all_code
    assert "AUTHORIZATION_PATH.resolve().is_relative_to(RR_OUTPUT.resolve())" in all_code
    assert "decision') != 'pending'" in all_code
    assert "this notebook will not create it" in all_code

    for action in (
        "environment",
        "plan",
        "authorize",
        "train",
        "evaluate",
        "aggregate",
        "status",
    ):
        assert f"'plasticity-p0d2hrr', {action!r}" in all_code or (
            action in {"plan", "authorize", "train", "evaluate", "aggregate", "status"}
            and f"rr_command({action!r})" in all_code
        )
    for forbidden in (
        "plasticity-train-lora",
        "train_lora(",
        "load_adapter_model(",
        "decision = 'approved'",
        '"decision": "approved"',
        "allows_rlvr = True",
    ):
        assert forbidden not in all_code


def test_notebook_locks_code_source_and_complete_evaluation_matrix() -> None:
    notebook = _load_notebook()
    all_code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "code_revision.txt" in all_code
    assert "'checkout', '--detach', CODE_REVISION" in all_code
    assert "SOURCE_SNAPSHOT_BEFORE = source_snapshot()" in all_code
    assert "SOURCE_SNAPSHOT_AFTER = assert_source_unchanged()" in all_code
    assert "SOURCE_SNAPSHOT_AFTER != SOURCE_SNAPSHOT_BEFORE" in all_code
    assert "audits['data'].get('train_example_count') != 480" in all_code
    assert "audits['data'].get('dev_example_count') != 96" in all_code
    assert "audits['forced_choice'].get('decision_count') != 1152" in all_code
    assert "audits['crd'].get('decision_count') != 1536" in all_code
    assert "automatic_narrow_scan_started" in all_code
    assert "automatic_rlvr_started" in all_code

    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "## 5. Independent authorization handoff" in markdown
    assert "## 7. Run the single frozen LoRA training job (GPU)" in markdown
    assert "## 8. Run the one locked external evaluation (GPU)" in markdown
    assert "no automatic next stage" in markdown
