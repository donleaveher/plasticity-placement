from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

from plasticity_placement.pathmem.io import json_hash
from plasticity_placement.pathmem_ropcd_p0.planner import compile_p0_plan

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPOSITORY_ROOT / "notebooks/pathmem_ropcd_p0_execution/build_notebook.py"
NOTEBOOK_PATH = (
    REPOSITORY_ROOT / "notebooks/pathmem_ropcd_p0_execution/pathmem_ropcd_p0_execution_colab.ipynb"
)


def _builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ropcd_p0_notebook_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _handoff() -> dict:
    identity = {
        "schema_version": "pathmem-g1c-to-ropcd-p0-handoff-v1",
        "passed": True,
        "run_id": "r" * 64,
        "source_manifest_id": "s" * 64,
        "plan_id": "p" * 64,
        "g1c_authorization_id": "a" * 64,
        "g1c_implementation_sha256": "i" * 64,
        "g1c_git_revision": "1" * 40,
        "summary_sha256": "u" * 64,
        "g1c_gate": {"passed": True},
        "p0_runner_implementation_review_eligible": True,
        "p0_authorized": False,
        "path_contrast_computed": False,
    }
    return {**identity, "handoff_id": json_hash(identity)}


def test_generated_notebook_is_exact_parseable_and_starts_with_colab_url() -> None:
    builder = _builder()
    expected = builder.build_notebook()
    observed = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert observed == expected
    assert observed["cells"][0]["cell_type"] == "markdown"
    assert "colab.research.google.com" in "".join(observed["cells"][0]["source"])
    for cell in observed["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_notebook_has_one_safe_control_cell_and_ordered_lifecycle() -> None:
    notebook = _builder().build_notebook()
    controls = [cell for cell in notebook["cells"] if "user-controls" in cell["metadata"]["tags"]]
    assert len(controls) == 1
    source = "".join(controls[0]["source"])
    assert "RUN_INSPECT = True" in source
    for flag in (
        "RUN_PREPARE_PLAN",
        "RUN_VERIFY_PLAN",
        "RUN_AUTHORIZE",
        "RUN_PREFLIGHT",
        "RUN_TRAIN",
        "RUN_EVALUATE",
        "RUN_AGGREGATE",
        "RUN_VERIFY",
    ):
        assert f"{flag} = False" in source
    all_source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    positions = [
        all_source.index(token)
        for token in (
            "if RUN_INSPECT:",
            "elif RUN_PREPARE_PLAN:",
            "elif RUN_VERIFY_PLAN:",
            "elif RUN_AUTHORIZE:",
            "elif RUN_PREFLIGHT:",
            "elif RUN_TRAIN:",
            "elif RUN_EVALUATE:",
            "elif RUN_AGGREGATE:",
        )
    ]
    assert positions == sorted(positions)
    assert "pre_authorization_stages" in all_source
    assert "CODE_REVISION_LOCK" in all_source
    assert 'decision": "approved' not in source


def test_target_selector_accepts_all_planned_ids_and_rejects_paths() -> None:
    builder = _builder()
    plan = compile_p0_plan(_handoff())
    assert all(re.fullmatch(builder.TARGET_UNIT_PATTERN, unit["unit_id"]) for unit in plan["units"])
    for invalid in (
        "p0r:pmv1-kill-01:seed-41:ABA",
        "p0r:pmv1-smoke-05:seed-41:ABA",
        "p0r:pmv1-smoke-01:seed-42:ABA",
        "p0r:pmv1-smoke-01:seed-41:OTHER",
        "p0r:pmv1-smoke-01:seed-41:ABA/../other",
    ):
        assert re.fullmatch(builder.TARGET_UNIT_PATTERN, invalid) is None


def test_gpu_install_is_stage_gated_and_forbidden_permissions_stay_false() -> None:
    notebook = _builder().build_notebook()
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert 'gpu_stage = SELECTED_STAGE in {"preflight", "train", "evaluate"}' in source
    assert 'sync_command += ["--extra", "train", "--extra", "colab"]' in source
    for permission in (
        "p1_authorized",
        "kill_or_reserve_access_authorized",
        "learned_router_authorized",
        "rl_controller_authorized",
    ):
        assert f'"{permission}": False' in source
    for forbidden in ("RUN_P1", "RUN_KILL", "RUN_RESERVE", "RUN_RL", "RUN_HPO"):
        assert forbidden not in source
