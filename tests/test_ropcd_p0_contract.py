from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from plasticity_placement.pathmem.io import json_hash
from plasticity_placement.pathmem_ropcd_p0.cli import build_parser
from plasticity_placement.pathmem_ropcd_p0.config import (
    EXECUTION_ORDER_SEED,
    EXPECTED_ROWS,
    EXPECTED_UNITS,
    P0_ROPCD_RECIPE,
    PANEL_SEED,
    TRAINING_SEED,
    execution_permissions,
    planning_permissions,
)
from plasticity_placement.pathmem_ropcd_p0.identity import (
    build_authorization_template,
    verify_external_approval,
)
from plasticity_placement.pathmem_ropcd_p0.manifest import (
    P0ExecutionManifest,
    UnitState,
)
from plasticity_placement.pathmem_ropcd_p0.planner import (
    audit_p0_plan,
    compile_p0_plan,
)
from plasticity_placement.pathmem_ropcd_p0.source import _verify_git_implementation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


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


def _plan() -> dict:
    return compile_p0_plan(_handoff())


def test_p0_plan_freezes_qualified_recipe_scope_and_counts() -> None:
    plan = _plan()
    assert audit_p0_plan(plan)["passed"] is True
    assert plan["recipe"] == P0_ROPCD_RECIPE.to_dict()
    assert P0_ROPCD_RECIPE.training_seed == TRAINING_SEED
    assert P0_ROPCD_RECIPE.panel_seed == PANEL_SEED
    assert P0_ROPCD_RECIPE.execution_order_seed == EXECUTION_ORDER_SEED
    assert plan["recipe"]["model"]["name"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert plan["counts"] == {
        "items": 4,
        "root_adapters": 4,
        "units": EXPECTED_UNITS,
        "core_units": 40,
        "technical_duplicate_units": 4,
        "optimizer_steps": 2_112,
        "teacher_student_prompt_pairs": 176,
        "evaluation_rows": EXPECTED_ROWS,
    }
    assert EXPECTED_ROWS == 2_168
    assert plan["permissions"] == planning_permissions()
    assert all(value is False for value in plan["permissions"].values())
    assert {unit["item_id"] for unit in plan["units"]} == {
        "pmv1-smoke-01",
        "pmv1-smoke-02",
        "pmv1-smoke-03",
        "pmv1-smoke-04",
    }


def test_p0_plan_has_exact_parent_dag_and_matched_event_exposure() -> None:
    plan = _plan()
    assert [unit["sequence_index"] for unit in plan["units"]] == list(range(EXPECTED_UNITS))
    assert plan["scope"]["execution_order_sha256"] == json_hash(
        [unit["unit_id"] for unit in plan["units"]]
    )
    by_item_logical = {(unit["item_id"], unit["logical_name"]): unit for unit in plan["units"]}
    expected_parents = {
        "A1": None,
        "B1": None,
        "AB": "A1",
        "BA": "B1",
        "ABA": "AB",
        "BAA": "BA",
        "BAB": "BA",
        "ABB": "AB",
        "A2": None,
        "B2": None,
    }
    for item_id in sorted({unit["item_id"] for unit in plan["units"]}):
        item_units = [unit for unit in plan["units"] if unit["item_id"] == item_id]
        assert len({unit["root_adapter_seed"] for unit in item_units}) == 1
        for logical_name, parent in expected_parents.items():
            unit = by_item_logical[(item_id, logical_name)]
            assert unit["parent_logical_name"] == parent
            assert len(unit["rollout_prompt_pairs"]) == 4
            assert len(unit["core_probe_ids"]) == 14
            assert len(unit["qualification_probe_ids"]) == 4
            assert len(unit["unrelated_probe_ids"]) == 8
        assert (
            by_item_logical[(item_id, "ABA")]["ordered_exposure_sha256"]
            == by_item_logical[(item_id, "BAA")]["ordered_exposure_sha256"]
        )
        assert (
            by_item_logical[(item_id, "BAB")]["ordered_exposure_sha256"]
            == by_item_logical[(item_id, "ABB")]["ordered_exposure_sha256"]
        )
        duplicate = next(
            unit
            for unit in plan["units"]
            if unit["item_id"] == item_id and unit["technical_duplicate"]
        )
        original = by_item_logical[(item_id, duplicate["duplicate_of"])]
        assert duplicate["parent_unit_id"] == original["parent_unit_id"]
        assert duplicate["ordered_exposure_sha256"] == original["ordered_exposure_sha256"]
        assert duplicate["step_seed_namespace"] == original["step_seed_namespace"]
    assert len({unit["root_adapter_seed"] for unit in plan["units"]}) == 4
    positions = {unit["unit_id"]: unit["sequence_index"] for unit in plan["units"]}
    assert all(
        unit["parent_unit_id"] is None or positions[unit["parent_unit_id"]] < unit["sequence_index"]
        for unit in plan["units"]
    )


def test_p0_plan_rejects_privilege_or_authority_tampering() -> None:
    plan = _plan()
    plan["permissions"]["p1_authorized"] = True
    with pytest.raises(ValueError, match="identity changed"):
        audit_p0_plan(plan)
    handoff = _handoff()
    handoff["p0_authorized"] = True
    with pytest.raises((ValueError, PermissionError), match="handoff|downstream|identity"):
        compile_p0_plan(handoff)


def test_p0_authorization_is_exact_and_keeps_downstream_disabled(tmp_path: Path) -> None:
    plan_root = tmp_path / "plan"
    plan_root.mkdir()
    (plan_root / "manifest.json").write_text("{}", encoding="utf-8")
    (plan_root / "p0_plan.json").write_text("{}", encoding="utf-8")
    report = {
        "passed": True,
        "manifest_id": "m" * 64,
        "plan_id": "p" * 64,
        "g1c_handoff_id": "h" * 64,
        "g1c_run_id": "r" * 64,
    }
    template = build_authorization_template(
        plan_root=plan_root,
        output_root=tmp_path / "run",
        plan_report=report,
    )
    assert template["permissions"] == execution_permissions()
    assert template["permissions"]["p0_training_authorized"] is True
    assert template["permissions"]["smoke_path_contrast_authorized"] is True
    for field in (
        "p1_authorized",
        "kill_or_reserve_access_authorized",
        "learned_router_authorized",
        "rl_controller_authorized",
        "automatic_hyperparameter_search_authorized",
    ):
        assert template["permissions"][field] is False
    approval = json.loads(json.dumps(template))
    approval.update(
        {
            "decision": "approved",
            "approved_by": "responsible-researcher",
            "approved_at": datetime.now(UTC).isoformat(),
        }
    )
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    verified = verify_external_approval(approval_path, expected_template=template)
    assert verified["authorization_id"] == json_hash(approval)
    approval["permissions"]["p1_authorized"] = True
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    with pytest.raises(ValueError, match="binding changed"):
        verify_external_approval(approval_path, expected_template=template)


def test_manifest_enforces_exact_units_transitions_and_safe_state(tmp_path: Path) -> None:
    plan = _plan()
    authorization = {"authorization_id": "a" * 64}
    manifest = P0ExecutionManifest.load_or_create(
        tmp_path / "manifest.json",
        plan_manifest_id="m" * 64,
        plan=plan,
        authorization=authorization,
        implementation_sha256="i" * 64,
    )
    unit_id = plan["units"][0]["unit_id"]
    assert len(manifest.payload["roots"]) == 4
    assert all(root["state"] == "pending" for root in manifest.payload["roots"].values())
    manifest.mark_gpu_inference_started()
    assert manifest.payload["gpu_inference_started"] is True
    assert manifest.unit_state(unit_id) is UnitState.PLANNED
    manifest.mark_unit(unit_id, UnitState.TRAINING)
    manifest.mark_unit(unit_id, UnitState.TRAINED, adapter_sha256="b" * 64)
    assert manifest.payload["training_started"] is True
    with pytest.raises(ValueError, match="invalid R-OPCD P0 transition"):
        manifest.mark_unit(unit_id, UnitState.TRAINING)
    manifest.payload["p1_authorized"] = True
    with pytest.raises(ValueError, match="unauthorized downstream"):
        manifest.save()


def test_recorded_git_implementation_verifies_objects_not_worktree() -> None:
    revision = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    relative = "src/plasticity_placement/pathmem_consolidation_exec/config.py"
    payload = subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), "show", f"{revision}:{relative}"],
        check=True,
        capture_output=True,
    ).stdout
    implementation = {
        "git_revision": revision,
        "source_files": {relative: sha256(payload).hexdigest()},
    }
    _verify_git_implementation(implementation)
    implementation["source_files"][relative] = "0" * 64
    with pytest.raises(ValueError, match="source hash changed"):
        _verify_git_implementation(implementation)


def test_cli_exposes_only_bounded_p0_lifecycle() -> None:
    parser = build_parser()
    common = [
        "--bundle",
        "bundle",
        "--g0-v1-manifest",
        "parent.json",
        "--g1c-run",
        "g1c",
        "--plan",
        "plan",
        "--output",
        "run",
    ]
    for command in (
        "inspect",
        "prepare-plan",
        "verify-plan",
        "authorization-template",
        "preflight",
        "aggregate",
        "verify",
    ):
        assert parser.parse_args([command, *common]).command == command
    training = parser.parse_args(["train", *common, "--max-units", "2"])
    assert training.max_units == 2
    assert training.unit_id is None
