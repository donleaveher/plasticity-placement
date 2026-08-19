from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

NOTEBOOK_DIR = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "p0d2h_invalid_audit"
)


def _load_generator() -> ModuleType:
    path = NOTEBOOK_DIR / "build_p0d2hc_invalid_audit_notebook.py"
    spec = importlib.util.spec_from_file_location(
        "build_p0d2hc_invalid_audit_notebook",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_invalid_audit_notebook_matches_generator_and_parses() -> None:
    generator = _load_generator()
    path = NOTEBOOK_DIR / "p0d2h_invalid_output_audit_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook == generator.build_notebook()
    assert notebook["metadata"]["colab"]["name"] == path.name
    opening = "".join(notebook["cells"][0]["source"])
    assert (
        "https://colab.research.google.com/github/donleaveher/"
        "plasticity-placement/blob/agent%2Fadd-lora-evaluation/"
        "notebooks/p0d2h_invalid_audit/"
        "p0d2h_invalid_output_audit_colab.ipynb"
        in opening
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(
                "".join(cell["source"]),
                filename=f"{path.name}:cell-{index}",
            )


def test_invalid_audit_notebook_is_cpu_only_and_read_only() -> None:
    notebook = json.loads(
        (
            NOTEBOOK_DIR / "p0d2h_invalid_output_audit_colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "SOURCE_PIPELINE_ATTEMPT = 'pipeline-c1'" in code
    assert "SOURCE_CALIBRATION_ATTEMPT = 'c1'" in code
    assert "AUDIT_PIPELINE_ATTEMPT = 'pipeline-a1'" in code
    assert "/hard-probe-calibration-analysis/v1/pipelines" in code
    assert "'plasticity-p0d2hc', 'audit-invalid'" in code
    assert "'plasticity-p0d2hc', 'run'" not in code
    assert "'plasticity-train-lora'" not in code
    assert "nvidia-smi" not in code
    assert "raw_tree_hash() != SOURCE_RAW_HASH" in code
    assert "Source manifest changed during read-only audit" in code
    assert "strict_gate_status_changed" in code
    assert "next_stage_eligibility_changed" in code

    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "CPU-only, read-only supplementary analysis" in markdown
    assert "inference, changes no frozen gate" in markdown
