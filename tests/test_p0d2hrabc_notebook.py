from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).parents[1] / "notebooks" / "p0d2h_rab_counterbalancing"
EXPECTED_CODE_REVISION = "8fc2c1bb0c762690cbff19aaa9238781a50a5d1d"


def _module():
    path = NOTEBOOK_DIR / "build_p0d2hrabc_notebook.py"
    spec = importlib.util.spec_from_file_location("build_p0d2hrabc_notebook", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_counterbalancing_notebook_matches_builder_and_parses(tmp_path: Path) -> None:
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
            ast.parse("".join(cell["source"]), filename=f"RABC-cell-{index}")


def test_counterbalancing_notebook_is_safe_and_atomic() -> None:
    notebook = _module().build_notebook()
    code_cells = [
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert len(code_cells) == 4
    config, controls, combined, results = code_cells
    all_code = "\n".join(code_cells)
    for text in (
        "RUN_PLAN = True",
        "RUN_AUTHORIZE = False",
        "RUN_AUDIT = False",
        "APPROVER = ''",
    ):
        assert text in controls
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
    for action in ("plan", "authorize", "run", "verify"):
        assert f"'plasticity-p0d2hrabc', '{action}'" in all_code
    assert "'plasticity-p0d2hrabc', 'train'" not in all_code
    for boundary in (
        "historical_rab_decision_changed",
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
    for source_artifact in (
        "preregistration.json",
        "binding_probes.jsonl",
        "bank_audit.json",
        "token_audit.json",
    ):
        assert source_artifact in combined
    assert "analysis_status" in results
    assert "tie_diagnostics" in results
    assert "audit_manifest_sha256" in results


def test_counterbalancing_notebook_uses_completed_rab_and_direct_colab_url() -> None:
    module = _module()
    notebook_text = json.dumps(module.build_notebook())

    assert "receipt-action-binding-rab1" in notebook_text
    assert "code-25f72d5f53_rsh-4d405f751a" in notebook_text
    assert module.COLAB_URL.startswith("https://colab.research.google.com/github/")
    assert "agent%2Fadd-lora-evaluation" in module.COLAB_URL
