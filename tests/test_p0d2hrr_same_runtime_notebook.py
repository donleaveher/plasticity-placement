from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

NOTEBOOK_DIR = (
    Path(__file__).resolve().parents[1] / "notebooks" / "p0d2h_route_remediation_same_runtime"
)


def _load_generator() -> ModuleType:
    path = NOTEBOOK_DIR / "build_p0d2hrr_same_runtime_notebook.py"
    spec = importlib.util.spec_from_file_location("build_same_runtime_notebook", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_notebook() -> dict[str, object]:
    return json.loads(
        (NOTEBOOK_DIR / "p0d2h_route_remediation_same_runtime_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )


def test_same_runtime_notebook_matches_generator_and_parses() -> None:
    generator = _load_generator()
    notebook = _load_notebook()
    assert notebook == generator.build_notebook()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"same-runtime-cell-{index}")
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_same_runtime_notebook_freezes_runtime_contrast_and_boundaries() -> None:
    notebook = _load_notebook()
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    normalized_markdown = " ".join(markdown.split())
    assert "RUN_AUDIT = True" in code
    assert "EXPECTED_RR_RUN_ID = 'p0d2hrr-902b15bc17'" in code
    assert "'same-runtime-audit'" in code
    assert "--combined-noninferiority-margin" in code
    assert "COMBINED_NONINFERIORITY_MARGIN = 0.02" in code
    assert "SENTINEL_SCORE_TOLERANCE = 1e-5" in code
    assert "settings-{settings_sha256[:10]}" in code
    assert "'checkout', '--detach', ANALYSIS_CODE_REVISION" in code
    assert "AUDIT_MANIFEST_PATH.is_file()" in code
    assert "manifest.get('identity') != EXPECTED_IDENTITY" in code
    assert "sha256(artifact_path.read_bytes()).hexdigest()" in code
    assert "summary.get('source_snapshot_after') != snapshot" in code
    assert "summary.get('mappings_per_adapter_authorized') is not False" in code
    assert "one loaded model object" in normalized_markdown
    assert "never starts training or 1/4/8" in normalized_markdown
    for forbidden in ("train_lora(", "plasticity-train-lora", "RUN_TRAINING"):
        assert forbidden not in code
