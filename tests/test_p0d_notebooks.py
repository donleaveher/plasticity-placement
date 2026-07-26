from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "p0d_lora_locus"
STAGES = ("band_scan", "narrow_scan")


def _load_generator() -> ModuleType:
    path = NOTEBOOK_DIR / "build_p0d_notebooks.py"
    spec = importlib.util.spec_from_file_location("build_p0d_notebooks", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p0d_notebooks_match_generator_and_have_valid_python() -> None:
    generator = _load_generator()
    for stage in STAGES:
        path = NOTEBOOK_DIR / f"p0d_{stage}_colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook == generator.build_notebook(stage)
        assert notebook["metadata"]["colab"]["name"] == path.name
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse(
                    "".join(cell["source"]),
                    filename=f"{path.name}:cell-{index}",
                )


def test_p0d_notebooks_are_stage_isolated_and_use_independent_drive_namespace() -> None:
    for stage in STAGES:
        path = NOTEBOOK_DIR / f"p0d_{stage}_colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert f"RUN_{stage.upper()} = False" in source
        other = next(name for name in STAGES if name != stage)
        assert f"RUN_{other.upper()} = False" not in source
        assert "/plasticity-p0d/lora-locus/v1/pipelines" in source
        assert "plasticity-p0d" in source
        assert "SOURCE_P0C_MANIFEST" in source
        assert "'plasticity-p0c', 'run'" not in source
        assert "'pipeline_version': 'p0d-lora-locus-v1'" in source
        assert "'environment_fingerprint': ENVIRONMENT_FINGERPRINT" in source


def test_band_and_narrow_notebook_gates_are_frozen() -> None:
    band = json.loads(
        (NOTEBOOK_DIR / "p0d_band_scan_colab.ipynb").read_text(encoding="utf-8")
    )
    band_source = "\n".join(
        "".join(cell["source"]) for cell in band["cells"]
    )
    assert "plan['expected_unit_count'] != 288" in band_source
    assert "summary['probe_row_count'] != 16224" in band_source
    assert "automatic" not in band_source.casefold()

    narrow = json.loads(
        (NOTEBOOK_DIR / "p0d_narrow_scan_colab.ipynb").read_text(encoding="utf-8")
    )
    narrow_source = "\n".join(
        "".join(cell["source"]) for cell in narrow["cells"]
    )
    assert "frozen-narrow-conditions.json" in narrow_source
    assert "source_band_run_id" in narrow_source
    assert "candidate_status.get('status') != 'manual_freeze_required'" in narrow_source
    assert "RUN_NARROW_SCAN = False" in narrow_source
