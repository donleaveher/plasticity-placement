from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).parents[1] / "notebooks" / "p0d2h_route_state_handoff"


def _module():
    path = NOTEBOOK_DIR / "build_p0d2hrsh_notebook.py"
    spec = importlib.util.spec_from_file_location("build_p0d2hrsh_notebook", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_handoff_notebook_is_colab_ready_parses_and_matches_builder(tmp_path: Path) -> None:
    module = _module()
    checked_in = json.loads((NOTEBOOK_DIR / module.NOTEBOOK_NAME).read_text())
    assert checked_in == module.build_notebook()
    module.OUTPUT_DIR = tmp_path
    module.main()
    notebook = json.loads((tmp_path / module.NOTEBOOK_NAME).read_text())
    assert notebook["metadata"]["accelerator"] == "GPU"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"RSH-cell-{index}")


def test_handoff_notebook_has_isolated_safe_controls_and_atomic_lifecycle() -> None:
    notebook = _module().build_notebook()
    code_cells = [
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert len(code_cells) == 4
    config, controls, combined, results = code_cells
    all_code = "\n".join(code_cells)
    assert "RUN_PLAN = True" in controls
    assert "RUN_AUTHORIZE = False" in controls
    assert "RUN_AUDIT = False" in controls
    assert "APPROVER = ''" in controls
    for variable in ("RUN_PLAN", "RUN_AUTHORIZE", "RUN_AUDIT", "APPROVER"):
        assert f"{variable} =" not in config
        assert f"{variable} =" not in combined
        assert f"{variable} =" not in results
    assert (
        combined.index("RESOLVED_CODE_REVISION = subprocess.run")
        < combined.index("AUDIT_OUTPUT = PIPELINE_ROOT")
        < combined.index("if RUN_PLAN:")
        < combined.index("if RUN_AUTHORIZE:")
        < combined.index("if RUN_AUDIT:")
    )
    assert "Authorization already adopted; validating idempotent completion" in combined
    assert "preflight/replay_audit.json" in "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )
    for action in ("plan", "authorize", "run"):
        assert f"'plasticity-p0d2hrsh', '{action}'" in all_code
    assert "'plasticity-p0d2hrsh', 'verify'" in results
    for artifact in (
        "paired_records.jsonl",
        "adapter_off.jsonl",
        "adapter_on.jsonl",
        "raw_score_checkpoint.json",
        "audit_manifest.json",
    ):
        assert artifact in results
    assert "audit_manifest_sha256" in results
    assert "'plasticity-p0d2hrsh', 'train'" not in all_code
    assert "historical_rtb_decision_changed" in all_code
    assert "training_authorized" in all_code
    assert "mappings_per_adapter_authorized" in all_code
