from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

NOTEBOOK_DIR = (
    Path(__file__).resolve().parents[1] / "notebooks" / "pathmem_g1c_execution"
)
CONTROL_NAMES = {
    "RUN_INSPECT",
    "RUN_AUTHORIZE",
    "RUN_PREFLIGHT",
    "RUN_TRAIN",
    "RUN_EVALUATE",
    "RUN_AGGREGATE",
    "RUN_VERIFY",
    "USE_GOOGLE_DRIVE",
    "APPROVER",
    "RUN_LABEL",
    "MAX_UNITS",
    "TARGET_UNIT_ID",
}


def _builder() -> ModuleType:
    path = NOTEBOOK_DIR / "build_notebook.py"
    spec = importlib.util.spec_from_file_location("pathmem_g1c_execution_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assigned_names(source: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def test_generated_notebook_matches_builder_is_gpu_and_parses() -> None:
    builder = _builder()
    path = NOTEBOOK_DIR / builder.NOTEBOOK_NAME
    notebook = json.loads(path.read_text(encoding="utf-8"))

    assert notebook == builder.build_notebook()
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["colab"]["gpuType"] == "A100"
    assert builder.COLAB_URL in "".join(notebook["cells"][0]["source"])
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"G1-C-exec-cell-{index}")
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_one_control_cell_has_safe_defaults_and_all_controls() -> None:
    notebook = _builder().build_notebook()
    controls = [
        cell for cell in notebook["cells"] if "user-controls" in cell["metadata"]["tags"]
    ]
    assert len(controls) == 1
    source = "".join(controls[0]["source"])
    assert _assigned_names(source) == CONTROL_NAMES
    assert "RUN_INSPECT = True" in source
    for name in (
        "RUN_AUTHORIZE",
        "RUN_PREFLIGHT",
        "RUN_TRAIN",
        "RUN_EVALUATE",
        "RUN_AGGREGATE",
        "RUN_VERIFY",
        "USE_GOOGLE_DRIVE",
    ):
        assert f"{name} = False" in source
    assert 'APPROVER = ""' in source
    assert "MAX_UNITS = 1" in source


def test_controls_are_assigned_only_in_control_cell_and_lifecycle_is_ordered() -> None:
    notebook = _builder().build_notebook()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assigned_controls = _assigned_names("".join(cell["source"])) & CONTROL_NAMES
        if "user-controls" in cell["metadata"]["tags"]:
            assert assigned_controls == CONTROL_NAMES
        else:
            assert not assigned_controls

    tags = [cell["metadata"]["tags"][0] for cell in notebook["cells"]]
    assert tags.index("user-controls") < tags.index("setup")
    assert tags.index("setup") < tags.index("identity-and-paths")
    assert tags.index("identity-and-paths") < tags.index("lifecycle")
    assert tags.index("lifecycle") < tags.index("results")


def test_notebook_exposes_exact_gated_execution_lifecycle() -> None:
    notebook = _builder().build_notebook()
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    for command in (
        "inspect",
        "authorization-template",
        "adopt-authorization",
        "preflight",
        "train",
        "evaluate",
        "aggregate",
        "verify",
    ):
        assert f'"{command}"' in source
    assert '"plasticity_placement.pathmem_consolidation_exec"' in source
    assert '"--extra", "train", "--extra", "colab"' in source
    assert 'subprocess.run(["nvidia-smi"], check=True)' in source
    assert '"checkout", "--detach", CHECKOUT_REVISION' in source
    assert "CODE_REVISION_LOCK" in source
    assert "APPROVAL_PATH" in source
    assert '"--max-units", str(MAX_UNITS)' in source
    assert '"--unit-id", TARGET_UNIT_ID' in source


def test_gpu_install_is_stage_gated_and_forbidden_permissions_remain_false() -> None:
    notebook = _builder().build_notebook()
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert 'gpu_stage = SELECTED_STAGE in {"preflight", "train", "evaluate"}' in source
    assert source.count('"--extra", "train", "--extra", "colab"') == 1
    assert source.count('subprocess.run(["nvidia-smi"], check=True)') == 1
    for permission in (
        "p0_authorized",
        "path_contrast_authorized",
        "kill_or_reserve_access_authorized",
    ):
        assert f'"{permission}": False' in source
        assert f'"{permission}": True' not in source
    for forbidden in ("RUN_P0", "RUN_PATH_CONTRAST", "RUN_KILL", "RUN_HPO"):
        assert forbidden not in source


def test_target_unit_selector_accepts_only_frozen_g1c_unit_ids() -> None:
    builder = _builder()
    for item_index in range(1, 13):
        for terminal_state in ("A", "B"):
            unit_id = f"g1c:pmv1-interface_dev-{item_index:02d}:{terminal_state}"
            assert re.fullmatch(builder.TARGET_UNIT_PATTERN, unit_id)
    for invalid in (
        "g1c:pmv1-interface_dev-00:A",
        "g1c:pmv1-interface_dev-13:B",
        "g1c:pmv1-interface_dev-01:C",
        "pmv1-interface_dev-01:A",
        "g1c:pmv1-interface_dev-01:A/../other",
    ):
        assert re.fullmatch(builder.TARGET_UNIT_PATTERN, invalid) is None

    notebook_source = "\n".join(
        "".join(cell["source"])
        for cell in builder.build_notebook()["cells"]
        if cell["cell_type"] == "code"
    )
    assert builder.TARGET_UNIT_PATTERN in notebook_source
