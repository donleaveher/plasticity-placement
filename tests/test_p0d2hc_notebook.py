from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

NOTEBOOK_DIR = (
    Path(__file__).resolve().parents[1] / "notebooks" / "p0d2h_calibration"
)


def _load_generator() -> ModuleType:
    path = NOTEBOOK_DIR / "build_p0d2hc_notebook.py"
    spec = importlib.util.spec_from_file_location(
        "build_p0d2hc_notebook",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p0d2hc_notebook_matches_generator_and_has_valid_python() -> None:
    generator = _load_generator()
    path = NOTEBOOK_DIR / "p0d2h_calibration_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook == generator.build_notebook()
    assert notebook["metadata"]["colab"]["name"] == path.name
    opening = "".join(notebook["cells"][0]["source"])
    assert (
        "https://colab.research.google.com/github/donleaveher/"
        "plasticity-placement/blob/agent%2Fadd-lora-evaluation/"
        "notebooks/p0d2h_calibration/p0d2h_calibration_colab.ipynb"
        in opening
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(
                "".join(cell["source"]),
                filename=f"{path.name}:cell-{index}",
            )


def test_p0d2hc_notebook_is_base_only_and_independently_namespaced() -> None:
    notebook = json.loads(
        (
            NOTEBOOK_DIR / "p0d2h_calibration_colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "P0D2H_PIPELINE_ATTEMPT = 'pipeline-r1'" in code
    assert "P0D2HC_PIPELINE_ATTEMPT = 'pipeline-c1'" in code
    assert "CALIBRATION_ATTEMPT = 'c1'" in code
    assert "/hard-probe-calibration/v1/pipelines" in code
    assert "'plasticity-p0d2hc', action" in code
    assert "'plasticity-train-lora'" not in code
    assert "'plasticity-p0d2', 'run'" not in code
    assert "['no_write', 'external', 'answer_copy_oracle']" in code
    assert "expected_models * 1152" in code
    assert "0 GPU memory is expected here" in code
    assert "FORMAL GPU RUN STARTING" in code
    assert "p0d2hc_command('run')" in code
    assert "p0d2hc_command('audit')" in code

    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "performs no LoRA training" in markdown
    assert "## 6. Formal base-only evaluation (GPU)" in markdown
    assert "## 9. Review results and next-stage eligibility" in markdown
