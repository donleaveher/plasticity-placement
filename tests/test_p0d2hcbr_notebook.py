from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).parents[1] / "notebooks" / "p0d2h_counterbalanced_binding_remediation"


def _module():
    path = NOTEBOOK_DIR / "build_p0d2hcbr_notebook.py"
    spec = importlib.util.spec_from_file_location("build_p0d2hcbr_notebook", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_cbr_notebook_is_colab_ready_and_parses(tmp_path: Path) -> None:
    module = _module()
    module.OUTPUT_DIR = tmp_path
    module.main()
    notebook = json.loads((tmp_path / module.NOTEBOOK_NAME).read_text())

    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"CBR-cell-{index}")


def test_cbr_notebook_has_dedicated_safe_controls_and_atomic_lifecycle() -> None:
    notebook = _module().build_notebook()
    code_cells = [
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert len(code_cells) == 4
    config, controls, lifecycle, results = code_cells
    all_code = "\n".join(code_cells)
    for text in (
        "RUN_PLAN = True",
        "RUN_AUTHORIZE = False",
        "RUN_TRAIN = False",
        "RUN_EVALUATE = False",
        "RUN_AGGREGATE = False",
        "RUN_VERIFY = False",
        "APPROVER = ''",
        "TARGET_CURRICULUM = ''",
        "TARGET_PLACEMENT = ''",
        "TARGET_SEED = None",
    ):
        assert text in controls
    for variable in (
        "RUN_PLAN",
        "RUN_AUTHORIZE",
        "RUN_TRAIN",
        "RUN_EVALUATE",
        "RUN_AGGREGATE",
        "RUN_VERIFY",
        "APPROVER",
        "TARGET_CURRICULUM",
        "TARGET_PLACEMENT",
        "TARGET_SEED",
    ):
        assert f"{variable} =" not in config
        assert f"{variable} =" not in lifecycle
        assert f"{variable} =" not in results
    assert "REQUESTED_CODE_REVISION = 'f4a71efe6bc717d345843b025c95fceeb6071954'" in config
    assert lifecycle.index("if RUN_PLAN:") < lifecycle.index("if RUN_AUTHORIZE:")
    assert lifecycle.index("if RUN_AUTHORIZE:") < lifecycle.index("if RUN_TRAIN:")
    assert lifecycle.index("if RUN_TRAIN:") < lifecycle.index("if RUN_EVALUATE:")
    assert lifecycle.index("if RUN_EVALUATE:") < lifecycle.index("if RUN_AGGREGATE:")
    assert lifecycle.index("if RUN_AGGREGATE:") < lifecycle.index("if RUN_VERIFY:")
    for action in ("plan", "authorize", "train", "evaluate", "aggregate", "verify"):
        assert f"'plasticity-p0d2hcbr', '{action}'" in all_code
    assert "--source-code-revision-lock" in lifecycle
    assert "mappings_per_adapter_authorized" in results


def test_builder_matches_checked_in_notebook() -> None:
    module = _module()
    checked_in = json.loads((NOTEBOOK_DIR / module.NOTEBOOK_NAME).read_text())

    assert checked_in == module.build_notebook()
