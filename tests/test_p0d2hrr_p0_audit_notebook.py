from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

NOTEBOOK_DIR = (
    Path(__file__).resolve().parents[1] / "notebooks" / "p0d2h_route_remediation_p0_audit"
)


def _load_generator() -> ModuleType:
    path = NOTEBOOK_DIR / "build_p0d2hrr_p0_audit_notebook.py"
    spec = importlib.util.spec_from_file_location(
        "build_p0d2hrr_p0_audit_notebook",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p0_audit_notebook_matches_generator_and_parses() -> None:
    generator = _load_generator()
    path = NOTEBOOK_DIR / "p0d2h_route_remediation_p0_audit_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook == generator.build_notebook()
    assert notebook["metadata"]["accelerator"] == "CPU"
    assert notebook["metadata"]["colab"]["name"] == path.name
    opening = "".join(notebook["cells"][0]["source"])
    assert (
        "https://colab.research.google.com/github/donleaveher/"
        "plasticity-placement/blob/agent%2Fadd-lora-evaluation/"
        "notebooks/p0d2h_route_remediation_p0_audit/"
        "p0d2h_route_remediation_p0_audit_colab.ipynb" in opening
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(
                "".join(cell["source"]),
                filename=f"{path.name}:cell-{index}",
            )
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_p0_audit_notebook_is_read_only_cpu_and_source_locked() -> None:
    notebook = json.loads(
        (NOTEBOOK_DIR / "p0d2h_route_remediation_p0_audit_colab.ipynb").read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "REPO_DIR = Path('/content/plasticity-placement-rr-p0-audit')" in code
    assert "EXPECTED_RR_RUN_ID = 'p0d2hrr-902b15bc17'" in code
    assert "'ae456ee26d9ac12c1ba35c0372574d7ae7f276eabba9800ae9af3360f6868986'" in code
    assert "EXPECTED_BASE_CRD_RUN_ID = 'p0d2hcrd-route-decomposition-8527b64f36'" in code
    assert "model.get('use_4bit') is True" in code
    assert "ANALYSIS_PIPELINE_ATTEMPT = 'pipeline-p0a1'" in code
    assert "route-remediation-followup-analysis/v1/pipelines" in code
    assert "'plasticity-p0d2hrr', 'audit-p0'" in code
    assert "'--historical-preregistration-sha256'" in code
    assert "'plasticity-p0d2hrr', 'train'" not in code
    assert "'plasticity-p0d2hrr', 'evaluate'" not in code
    assert "'plasticity-train-lora'" not in code
    assert "load_adapter_model" not in code
    assert "nvidia-smi" not in code
    assert "RR_CODE_REVISION_LOCK" in code
    assert "BASE_CRD_CODE_REVISION_LOCK" in code
    assert "'--base-crd-code-revision-lock'" in code
    assert "ANALYSIS_CODE_REVISION_LOCK" in code
    assert "SOURCE_SNAPSHOT_BEFORE = source_snapshot()" in code
    assert "SOURCE_SNAPSHOT_AFTER = source_snapshot()" in code
    assert "SOURCE_FC_OUTPUT.resolve()" in code
    assert "Frozen audit sources are not mutually independent" in code
    assert "Source artifacts changed during read-only audit" in code
    assert "source_gate_status_changed" in code
    assert "next_stage_eligibility_changed" in code
    assert "training_authorized" in code
    assert "current_artifact_graph_checks_passed" in code
    assert "historical_attempt_uniqueness_verified" in code

    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    assert "CPU-only" in markdown
    assert "read-only" in markdown
    assert "It does not load a model" in markdown
    assert "Do not start another LoRA" in markdown
