from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).parents[1] / "notebooks" / "p0d2h_rab_label_disentanglement"


def _module():
    path = NOTEBOOK_DIR / "build_p0d2hrabx_notebook.py"
    spec = importlib.util.spec_from_file_location("build_p0d2hrabx_notebook", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_label_notebook_matches_builder_and_parses(tmp_path: Path) -> None:
    module = _module()
    checked_in = json.loads((NOTEBOOK_DIR / module.NOTEBOOK_NAME).read_text())
    assert checked_in == module.build_notebook()
    module.OUTPUT_DIR = tmp_path
    module.main()
    notebook = json.loads((tmp_path / module.NOTEBOOK_NAME).read_text())
    assert notebook["metadata"]["accelerator"] == "GPU"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"RABX-cell-{index}")


def test_label_notebook_has_safe_dedicated_controls_and_atomic_lifecycle() -> None:
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
        "RUN_RECOVERY_INSPECTION = False",
        "RUN_RECOVER_ZERO_ARTIFACT = False",
        "RUN_AUDIT = False",
        "APPROVER = ''",
        "ORIGINAL_RUNTIME_TERMINATED = False",
    ):
        assert text in controls
    for variable in (
        "RUN_PLAN",
        "RUN_AUTHORIZE",
        "RUN_RECOVERY_INSPECTION",
        "RUN_RECOVER_ZERO_ARTIFACT",
        "RUN_AUDIT",
        "APPROVER",
        "ORIGINAL_RUNTIME_TERMINATED",
    ):
        assert f"{variable} =" not in config
        assert f"{variable} =" not in combined
        assert f"{variable} =" not in results
    assert (
        combined.index("RESOLVED_EXPERIMENT_CODE_REVISION = prepare_checkout")
        < combined.index("AUDIT_OUTPUT = PIPELINE_ROOT")
        < combined.index("if RUN_PLAN:")
        < combined.index("if RUN_AUTHORIZE:")
        < combined.index("if RUN_RECOVERY_INSPECTION:")
        < combined.index("if RUN_RECOVER_ZERO_ARTIFACT:")
        < combined.rindex("if RUN_AUDIT:")
    )
    assert "if RUN_AUDIT:\n    subprocess.run(['nvidia-smi']" in combined
    assert "Authorization already adopted; validating idempotent completion" in combined
    for action in ("plan", "authorize", "run", "verify"):
        assert f"'plasticity-p0d2hrabx', '{action}'" in all_code
    for action in ("zero-artifact-recovery-template", "recover-zero-artifact"):
        assert f"'plasticity-p0d2hrabx', '{action}'" in all_code
    assert "RECOVERY_REPO_DIR" in combined
    assert "EXPERIMENT_CODE_REVISION = '58eaa8c4dde431e393925662b6c1e913bba02611'" in config
    assert "RECOVERY_CODE_REVISION = '31bd1e9811e983370586be15e39f83a80509eb65'" in config
    assert "RECOVERY_ATTEMPT = 'rfix2'" in config
    assert "recovery_code_revision-{RECOVERY_ATTEMPT}.txt" in combined
    assert "'plasticity-p0d2hrabx', 'train'" not in all_code
    for boundary in (
        "historical_rab_decision_changed",
        "historical_rabc_attribution_changed",
        "training_authorized",
        "mappings_per_adapter_authorized",
    ):
        assert boundary in all_code


def test_label_notebook_uses_completed_rabc_and_direct_colab_url() -> None:
    module = _module()
    notebook_text = json.dumps(module.build_notebook())

    assert "rab-counterbalancing-rabc1" in notebook_text
    assert "code-8fc2c1bb0c_rab-9368c8c29c" in notebook_text
    assert module.COLAB_URL.startswith("https://colab.research.google.com/github/")
    assert "agent%2Fadd-lora-evaluation" in module.COLAB_URL
