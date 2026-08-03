from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).parents[1] / "notebooks" / "p0d2h_cpr_qualification_recovery"


def _module():
    path = NOTEBOOK_DIR / "build_p0d2hcpr_recovery_notebook.py"
    spec = importlib.util.spec_from_file_location("build_p0d2hcpr_recovery_notebook", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_recovery_notebook_is_colab_ready_and_parses(tmp_path: Path) -> None:
    module = _module()
    module.OUTPUT_DIR = tmp_path
    module.main()
    notebook = json.loads((tmp_path / module.NOTEBOOK_NAME).read_text())
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"CPR-recovery-cell-{index}")


def test_recovery_notebook_is_single_use_and_cannot_train_or_scan() -> None:
    notebook = _module().build_notebook()
    all_code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "RUN_RECOVERY = False" in all_code
    assert "REQUESTED_CODE_REVISION = None" not in all_code
    assert "REQUESTED_CODE_REVISION = 'b4dcbdf952d519b97bbe87a6fd0e066c8a21581c'" in all_code
    assert "composition-remediation-cpr2" in all_code
    assert "--recovery-authorization" in all_code
    assert "analysis_code_sha256" in all_code
    assert "allowed_recovery_runs': 1" in all_code
    assert "allows_training': False" in all_code
    assert "allows_mappings_per_adapter_scan': False" in all_code
    assert "'plasticity-p0d2hcpr', 'train'" not in all_code
    assert "'plasticity-p0d2hcpr', 'plan'" not in all_code
