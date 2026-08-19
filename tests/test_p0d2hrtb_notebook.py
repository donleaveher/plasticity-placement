from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).parents[1] / "notebooks" / "p0d2h_route_transfer_bridge"


def _module():
    path = NOTEBOOK_DIR / "build_p0d2hrtb_notebook.py"
    spec = importlib.util.spec_from_file_location("build_p0d2hrtb_notebook", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_bridge_notebook_is_colab_ready_and_parses(tmp_path: Path) -> None:
    module = _module()
    checked_in = json.loads((NOTEBOOK_DIR / module.NOTEBOOK_NAME).read_text())
    assert checked_in == module.build_notebook()
    module.OUTPUT_DIR = tmp_path
    module.main()
    notebook = json.loads((tmp_path / module.NOTEBOOK_NAME).read_text())
    assert notebook["metadata"]["accelerator"] == "GPU"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"RTB-cell-{index}")


def test_bridge_notebook_has_isolated_controls_and_recovery_lifecycle() -> None:
    notebook = _module().build_notebook()
    code_cells = [
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    all_code = "\n".join(code_cells)
    assert len(code_cells) == 4
    config, controls, combined, results = code_cells
    assert (
        combined.index("RESOLVED_EXPERIMENT_CODE_REVISION = prepare_checkout")
        < combined.index("AUDIT_OUTPUT = PIPELINE_ROOT")
        < combined.index("if RUN_PLAN:")
        < combined.index("recovery_template_result = run")
        < combined.index("if RUN_AUDIT:")
    )
    assert "RUN_PLAN = True" in controls
    assert "RUN_AUTHORIZE = False" in controls
    assert "RUN_RECOVER_ZERO_ARTIFACT = False" in controls
    assert "RUN_AUDIT = False" in controls
    assert "APPROVER = ''" in controls
    assert "ORIGINAL_RUNTIME_TERMINATED = False" in controls
    assert "recovery_template != observed_recovery_template" in combined
    assert "Recovery evidence changed after inspection" in combined
    assert "Recovery already adopted; validating idempotent completion" in combined
    for variable in (
        "RUN_PLAN",
        "RUN_AUTHORIZE",
        "RUN_RECOVER_ZERO_ARTIFACT",
        "RUN_AUDIT",
        "APPROVER",
        "ORIGINAL_RUNTIME_TERMINATED",
    ):
        assert f"{variable} =" not in config
        assert f"{variable} =" not in combined
        assert f"{variable} =" not in results
    assert "EXPERIMENT_CODE_REVISION = 'dd004f54e39598e4b67ae525bb46a38c293378b1'" in config
    assert (
        "RECOVERY_CODE_REVISION = '8678cc8c2510e50015fb451f558d47901f918716'" in config
    )
    assert "EXPERIMENT_REPO_DIR" in config
    assert "RECOVERY_REPO_DIR" in config
    assert "f'{revision_ref}^{{commit}}'" in all_code
    assert "f'{{revision_ref}}^{{commit}}'" not in all_code
    assert "composition-remediation-cpr2" in all_code
    assert "same-runtime-q2/summary.json" in all_code
    assert "manifest['state'] == 'complete'" in all_code
    assert "audit_manifest['artifacts']['summary']" in all_code
    assert "Bridge summary hash does not match complete manifests" in all_code
    for action in (
        "plan",
        "authorize",
        "zero-artifact-recovery-template",
        "recover-zero-artifact",
        "run",
    ):
        assert f"'plasticity-p0d2hrtb', '{action}'" in all_code
    assert "'plasticity-p0d2hrtb', 'train'" not in all_code
    assert "mappings_per_adapter_authorized" in all_code
