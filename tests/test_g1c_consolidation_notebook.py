from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

NOTEBOOK_DIR = (
    Path(__file__).resolve().parents[1] / "notebooks" / "pathmem_g1c_consolidation"
)


def _builder() -> ModuleType:
    path = NOTEBOOK_DIR / "build_notebook.py"
    spec = importlib.util.spec_from_file_location("pathmem_g1c_notebook_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_notebook_matches_builder_is_cpu_and_parses() -> None:
    builder = _builder()
    notebook_path = NOTEBOOK_DIR / builder.NOTEBOOK_NAME
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook == builder.build_notebook()
    assert notebook["metadata"]["accelerator"] == "CPU"
    assert notebook["metadata"]["colab"]["name"] == builder.NOTEBOOK_NAME
    opening = "".join(notebook["cells"][0]["source"])
    assert opening.startswith("[Open this notebook in Google Colab](https://colab")
    assert builder.COLAB_URL in opening
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"G1-C-Colab-cell-{index}")
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_notebook_has_one_safe_control_cell_and_ordered_lifecycle() -> None:
    notebook = _builder().build_notebook()
    controls = [
        cell for cell in notebook["cells"] if "user-controls" in cell["metadata"]["tags"]
    ]
    assert len(controls) == 1
    control_source = "".join(controls[0]["source"])
    assert "RUN_INSPECT = True" in control_source
    assert "RUN_PREPARE = False" in control_source
    assert "RUN_VERIFY = False" in control_source
    assert "USE_GOOGLE_DRIVE = False" in control_source
    assert 'RUN_LABEL = "g0-v2-r-opcd-plan-v1"' in control_source

    tags = [cell["metadata"]["tags"][0] for cell in notebook["cells"]]
    assert tags.index("user-controls") < tags.index("setup")
    assert tags.index("setup") < tags.index("identity-and-paths")
    assert tags.index("identity-and-paths") < tags.index("lifecycle")
    assert tags.index("lifecycle") < tags.index("results")


def test_user_controls_are_assigned_only_in_the_control_cell() -> None:
    notebook = _builder().build_notebook()
    names = (
        "RUN_INSPECT =",
        "RUN_PREPARE =",
        "RUN_VERIFY =",
        "USE_GOOGLE_DRIVE =",
        "RUN_LABEL =",
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        is_control = "user-controls" in cell["metadata"]["tags"]
        assert is_control or not any(name in source for name in names)


def test_notebook_exposes_only_the_cpu_planning_lifecycle() -> None:
    notebook = _builder().build_notebook()
    code_source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for command in ("inspect", "prepare", "verify"):
        assert f'["{command}", *parent_arguments' in code_source
    assert '"--g0-v1-manifest"' in code_source
    assert '"plasticity_placement.pathmem_consolidation"' in code_source
    assert '"uv", "sync", "--frozen", "--no-dev"' in code_source
    assert "CODE_REVISION_LOCK" in code_source
    assert '"checkout", "--detach", CHECKOUT_REVISION' in code_source
    for forbidden in (
        "nvidia-smi",
        '"--extra", "train"',
        "RUN_G1C",
        "RUN_P0",
        "kill_or_reserve_access_authorized = True",
        "path_contrast_authorized = True",
    ):
        assert forbidden not in code_source
