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
    module.OUTPUT_DIR = tmp_path
    module.main()
    notebook = json.loads((tmp_path / module.NOTEBOOK_NAME).read_text())
    assert notebook["metadata"]["accelerator"] == "GPU"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"RTB-cell-{index}")


def test_bridge_notebook_has_safe_three_pass_lifecycle() -> None:
    notebook = _module().build_notebook()
    all_code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "RUN_PLAN = True" in all_code
    assert "RUN_AUTHORIZE = False" in all_code
    assert "RUN_AUDIT = False" in all_code
    assert "REQUESTED_CODE_REVISION = 'dd004f54e39598e4b67ae525bb46a38c293378b1'" in all_code
    assert "f'{revision_ref}^{{commit}}'" in all_code
    assert "f'{{revision_ref}}^{{commit}}'" not in all_code
    assert "composition-remediation-cpr2" in all_code
    assert "same-runtime-q2/summary.json" in all_code
    assert "manifest['state'] == 'complete'" in all_code
    assert "audit_manifest['artifacts']['summary']" in all_code
    assert "Bridge summary hash does not match complete manifests" in all_code
    for action in ("plan", "authorize", "run"):
        assert f"'plasticity-p0d2hrtb', '{action}'" in all_code
    assert "'plasticity-p0d2hrtb', 'train'" not in all_code
    assert "mappings_per_adapter_authorized" in all_code
