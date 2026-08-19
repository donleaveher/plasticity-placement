from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).parents[1] / "notebooks" / "p0d2h_receipt_action_binding"
EXPECTED_CODE_REVISION = "25f72d5f53010b32a3b4ebb569dbd1863b7aeaa0"


def _module():
    path = NOTEBOOK_DIR / "build_p0d2hrab_notebook.py"
    spec = importlib.util.spec_from_file_location("build_p0d2hrab_notebook", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_binding_notebook_is_colab_ready_parses_and_matches_builder(tmp_path: Path) -> None:
    module = _module()
    assert module.CODE_REVISION == EXPECTED_CODE_REVISION
    checked_in = json.loads((NOTEBOOK_DIR / module.NOTEBOOK_NAME).read_text())
    assert checked_in == module.build_notebook()
    module.OUTPUT_DIR = tmp_path
    module.main()
    notebook = json.loads((tmp_path / module.NOTEBOOK_NAME).read_text())
    assert notebook["metadata"]["accelerator"] == "GPU"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"RAB-cell-{index}")


def test_binding_notebook_has_isolated_controls_and_atomic_lifecycle() -> None:
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
        < combined.rindex("if RUN_AUDIT:")
    )
    assert "Authorization already adopted; validating idempotent completion" in combined
    assert "if RUN_AUDIT:\n    subprocess.run(['nvidia-smi']" in combined
    assert "if RUN_AUDIT and environment_payload['environment']['cuda_available']" in combined
    assert "rsh_quadrant_diagnostic.md" in combined
    assert "display(Markdown(quadrant_report.read_text()))" in combined
    for action in ("plan", "authorize", "run", "verify"):
        assert f"'plasticity-p0d2hrab', '{action}'" in all_code
    assert "'plasticity-p0d2hrab', 'train'" not in all_code
    for boundary in (
        "historical_rsh_decision_changed",
        "training_authorized",
        "mappings_per_adapter_authorized",
    ):
        assert boundary in all_code
    for artifact in (
        "paired_records.jsonl",
        "adapter_off.jsonl",
        "adapter_on.jsonl",
        "raw_score_checkpoint.json",
        "audit_manifest.json",
    ):
        assert artifact in results
    assert "audit_manifest_sha256" in results


def test_binding_notebook_uses_completed_rsh_and_direct_colab_url() -> None:
    module = _module()
    notebook_text = json.dumps(module.build_notebook())

    assert "route-state-handoff-rsh1" in notebook_text
    assert "code-b536c34c2b_rtb-d3cc84c2ed" in notebook_text
    assert module.COLAB_URL.startswith("https://colab.research.google.com/github/")
    assert "agent%2Fadd-lora-evaluation" in module.COLAB_URL
