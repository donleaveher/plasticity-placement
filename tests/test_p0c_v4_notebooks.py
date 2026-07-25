from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "p0c_v4"
STAGES = ("smoke", "calibration", "pilot", "confirmatory")


def _load_generator() -> ModuleType:
    path = NOTEBOOK_DIR / "build_p0c_v4_notebooks.py"
    spec = importlib.util.spec_from_file_location("build_p0c_v4_notebooks", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v4_notebooks_are_valid_and_match_the_shared_generator() -> None:
    generator = _load_generator()
    for stage in STAGES:
        path = NOTEBOOK_DIR / f"p0c_{stage}_v4_colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook == generator.build_notebook(stage)
        assert notebook["metadata"]["colab"]["name"] == path.name
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                ast.parse(
                    "".join(cell["source"]),
                    filename=f"{path.name}:cell-{index}",
                )


def test_each_v4_notebook_runs_only_its_named_experiment() -> None:
    for stage in STAGES:
        path = NOTEBOOK_DIR / f"p0c_{stage}_v4_colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        notebook_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        enabled_flags = [name for name in STAGES if f"RUN_{name.upper()} = False" in source]
        assert enabled_flags == [stage]
        assert "'pipeline_version': 'p0c-v4'" in source
        assert "/plasticity-p0c/v4/pipelines" in source
        assert f"/notebooks/p0c_v4/{path.name}" in notebook_text
        assert "PROVENANCE_KEY = f'code-{CODE_HASH[:10]}_cfg-{SETTINGS_FINGERPRINT}'" in source
        assert "f'env-{ENVIRONMENT_FINGERPRINT}'" not in source
        assert "'environment_fingerprint': ENVIRONMENT_FINGERPRINT" in source
