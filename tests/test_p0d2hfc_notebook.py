from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

NOTEBOOK_DIR = (
    Path(__file__).resolve().parents[1] / "notebooks" / "p0d2h_forced_choice"
)


def _load_generator() -> ModuleType:
    path = NOTEBOOK_DIR / "build_p0d2hfc_notebook.py"
    spec = importlib.util.spec_from_file_location(
        "build_p0d2hfc_notebook",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notebook_matches_generator_and_all_code_cells_parse() -> None:
    generator = _load_generator()
    path = NOTEBOOK_DIR / "p0d2h_forced_choice_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook == generator.build_notebook()
    assert notebook["metadata"]["colab"]["name"] == path.name
    opening = "".join(notebook["cells"][0]["source"])
    assert (
        "https://colab.research.google.com/github/donleaveher/"
        "plasticity-placement/blob/agent%2Fadd-lora-evaluation/"
        "notebooks/p0d2h_forced_choice/"
        "p0d2h_forced_choice_colab.ipynb"
        in opening
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(
                "".join(cell["source"]),
                filename=f"{path.name}:cell-{index}",
            )


def test_notebook_freezes_cpu_preflight_before_first_gpu_model_load() -> None:
    notebook = json.loads(
        (
            NOTEBOOK_DIR / "p0d2h_forced_choice_colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    code_cells = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    all_code = "\n".join(code_cells)
    assert "SOURCE_PIPELINE_ATTEMPT = 'pipeline-c1'" in all_code
    assert "FORCED_CHOICE_PIPELINE_ATTEMPT = 'pipeline-f1'" in all_code
    assert "/hard-probe-forced-choice/v1/pipelines" in all_code
    assert "plan['expected_decision_row_count'] != 2304" in all_code
    assert "audit.get('candidate_count') != 9216" in all_code
    assert "'plasticity-p0d2hfc', action" in all_code
    for forbidden in (
        "plasticity-train-lora",
        "load_adapter_model",
        "activate_adapter",
        "plasticity-p0d2hfc', 'train",
    ):
        assert forbidden not in all_code

    formal_index = next(
        index
        for index, cell in enumerate(code_cells)
        if "FORMAL GPU SCORING STARTING" in cell
    )
    preformal = "\n".join(code_cells[:formal_index])
    assert "p0d2hfc_command('run')" not in preformal
    assert "AutoModelForCausalLM" not in preformal
    assert "load_base_model" not in preformal
    assert "p0d2hfc_command('audit')" in preformal

    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "## 6. Formal full-string candidate scoring (GPU)" in markdown
    assert "no automatic next stage" in markdown
