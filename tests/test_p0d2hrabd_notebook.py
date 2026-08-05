from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).parents[1] / "notebooks" / "p0d2h_rab_error_topology"
EXPECTED_CODE_REVISION = "242bfbc8155e00d8640ca2c8a3a3698ad57c6d45"


def _module():
    path = NOTEBOOK_DIR / "build_p0d2hrabd_notebook.py"
    spec = importlib.util.spec_from_file_location("build_p0d2hrabd_notebook", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_diagnostic_notebook_is_colab_ready_and_matches_builder(tmp_path: Path) -> None:
    module = _module()
    assert module.CODE_REVISION == EXPECTED_CODE_REVISION
    assert re.fullmatch(r"[0-9a-f]{40}", module.CODE_REVISION)
    checked_in = json.loads((NOTEBOOK_DIR / module.NOTEBOOK_NAME).read_text())
    assert checked_in == module.build_notebook()
    module.OUTPUT_DIR = tmp_path
    module.main()
    notebook = json.loads((tmp_path / module.NOTEBOOK_NAME).read_text())
    assert "accelerator" not in notebook["metadata"]
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"RABD-cell-{index}")


def test_diagnostic_notebook_has_isolated_safe_controls() -> None:
    notebook = _module().build_notebook()
    code_cells = [
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert len(code_cells) == 4
    config, controls, combined, results = code_cells
    all_code = "\n".join(code_cells)
    assert "RUN_PLAN = True" in controls
    assert "RUN_DIAGNOSTIC = False" in controls
    for variable in ("RUN_PLAN", "RUN_DIAGNOSTIC"):
        assert f"{variable} =" not in config
        assert f"{variable} =" not in combined
        assert f"{variable} =" not in results
    assert "APPROVER" not in controls
    assert "RUN_AUTHORIZE" not in all_code
    assert (
        combined.index("RESOLVED_CODE_REVISION = subprocess.run")
        < combined.index("DIAGNOSTIC_OUTPUT = PIPELINE_ROOT")
        < combined.index("if RUN_PLAN:")
        < combined.index("if RUN_DIAGNOSTIC:")
    )
    for action in ("plan", "run", "verify"):
        assert f"'plasticity-p0d2hrabd', '{action}'" in all_code
    assert "'plasticity-p0d2hrabd', 'train'" not in all_code
    assert "nvidia-smi" not in all_code
    for boundary in (
        "historical_rab_decision_changed",
        "inference_authorized",
        "training_authorized",
        "mappings_per_adapter_authorized",
    ):
        assert boundary in all_code
    for artifact in (
        "cell_records.jsonl",
        "unit_records.jsonl",
        "factor_slices.json",
        "audit_manifest.json",
    ):
        assert artifact in results


def test_diagnostic_notebook_uses_exact_completed_rab_and_direct_colab_url() -> None:
    module = _module()
    notebook_text = json.dumps(module.build_notebook())

    assert "receipt-action-binding-rab1" in notebook_text
    assert "code-25f72d5f53_rsh-4d405f751a" in notebook_text
    assert module.COLAB_URL.startswith("https://colab.research.google.com/github/")
    assert "agent%2Fadd-lora-evaluation" in module.COLAB_URL
