from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

from plasticity_placement.pathmem.io import json_hash
from plasticity_placement.pathmem_ropcd_p1.planner import compile_p1_plan

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPOSITORY_ROOT / "notebooks/pathmem_ropcd_p1_execution/build_notebook.py"
NOTEBOOK_PATH = (
    REPOSITORY_ROOT
    / "notebooks/pathmem_ropcd_p1_execution/pathmem_ropcd_p1_execution_colab.ipynb"
)


def _builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ropcd_p1_notebook_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _handoff() -> dict:
    identity = {
        "schema_version": "pathmem-ropcd-g2-to-p1-handoff-v1",
        "passed": True,
        "repair_id": "r" * 64,
        "source_run_id": "s" * 64,
        "source_plan_id": "p" * 64,
        "source_matrix_sha256": "m" * 64,
        "g2_summary_sha256": "g" * 64,
        "g2_gate": {"gate": "G2", "passed": True},
        "repair_implementation_sha256": "i" * 64,
        "p1_implementation_review_eligible": True,
        "p1_authorized": False,
        "kill_or_reserve_accessed": False,
    }
    return {**identity, "handoff_id": json_hash(identity)}


def test_generated_p1_notebook_is_exact_and_parseable() -> None:
    builder = _builder()
    expected = builder.build_notebook()
    observed = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert observed == expected
    assert "colab.research.google.com" in "".join(observed["cells"][0]["source"])
    for cell in observed["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_p1_notebook_has_one_safe_control_cell_and_two_approvals() -> None:
    notebook = _builder().build_notebook()
    controls = [cell for cell in notebook["cells"] if "user-controls" in cell["metadata"]["tags"]]
    assert len(controls) == 1
    source = "".join(controls[0]["source"])
    assert "RUN_INSPECT = True" in source
    for flag in (
        "RUN_PREPARE_PLAN",
        "RUN_VERIFY_PLAN",
        "RUN_BENCHMARK_AUTHORIZE",
        "RUN_BENCHMARK",
        "RUN_BENCHMARK_AGGREGATE",
        "RUN_BENCHMARK_VERIFY",
        "RUN_AUTHORIZE",
        "RUN_PREFLIGHT",
        "RUN_TRAIN",
        "RUN_EVALUATE",
        "RUN_AGGREGATE",
        "RUN_VERIFY",
    ):
        assert f"{flag} = False" in source
    all_source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "BENCHMARK_APPROVAL" in all_source
    assert "RUN_APPROVAL" in all_source
    assert "RESOURCE_PROFILE" in all_source
    assert "CODE_REVISION_LOCK" in all_source
    assert 'decision": "approved' not in source


def test_p1_target_selector_covers_every_planned_unit_and_rejects_escape() -> None:
    builder = _builder()
    plan = compile_p1_plan(_handoff())
    assert all(
        re.fullmatch(builder.TARGET_UNIT_PATTERN, unit["unit_id"])
        for unit in plan["units"]
    )
    for invalid in (
        "p1r:pmv1-smoke-01:seed-41:ABA",
        "p1r:pmv1-kill-13:seed-41:ABA",
        "p1r:pmv1-kill-01:seed-44:ABA",
        "p1r:pmv1-kill-01:seed-41:OTHER",
        "p1r:pmv1-kill-01:seed-41:ABA/../other",
    ):
        assert re.fullmatch(builder.TARGET_UNIT_PATTERN, invalid) is None


def test_p1_gpu_stages_are_explicit_and_downstream_permissions_remain_false() -> None:
    notebook = _builder().build_notebook()
    source = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert (
        'gpu_stage = SELECTED_STAGE in {"benchmark-run", "preflight", "train", "evaluate"}'
        in source
    )
    assert 'sync_command += ["--extra", "train", "--extra", "colab"]' in source
    for permission in (
        "reserve_access_authorized",
        "p1b_authorized",
        "p2_authorized",
        "learned_router_authorized",
        "rl_controller_authorized",
    ):
        assert f'"{permission}": False' in source
    for forbidden in ("RUN_P1B", "RUN_P2", "RUN_RESERVE", "RUN_RL", "RUN_HPO"):
        assert forbidden not in source
