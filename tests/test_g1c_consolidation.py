from __future__ import annotations

import json
from pathlib import Path

import pytest

from plasticity_placement.pathmem.manifest import verify_g0_bundle
from plasticity_placement.pathmem_consolidation.cli import build_parser, main
from plasticity_placement.pathmem_consolidation.config import (
    G1C_THRESHOLDS,
    OPERATOR_ID,
    build_g0v2_contract,
    evaluate_g1c_gate,
)
from plasticity_placement.pathmem_consolidation.manifest import (
    verify_g0v2_bundle,
    verify_parent_g0_v1,
    write_g0v2_bundle,
)
from plasticity_placement.pathmem_consolidation.planner import (
    audit_g1c_plan,
    compile_g1c_plan,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = REPOSITORY_ROOT / "notebooks/pathmem_g1_p0/protocol_snapshot"
PARENT_MANIFEST = PROTOCOL_ROOT / "experiments/g0/manifest.json"


def _parent() -> dict[str, str]:
    return verify_parent_g0_v1(PARENT_MANIFEST)


def test_g0v2_contract_separates_consolidation_from_rl_control() -> None:
    parent = _parent()
    contract = build_g0v2_contract(
        parent_manifest_id=parent["manifest_id"],
        parent_manifest_sha256=parent["manifest_sha256"],
    )
    assert contract["operator"]["operator_id"] == OPERATOR_ID
    assert contract["operator"]["learned_router"] is False
    assert contract["authorization"] == {
        "safe_default": "planning_only",
        "training_authorized": False,
        "gpu_inference_authorized": False,
        "g1c_execution_authorized": False,
        "p0_authorized": False,
        "path_contrast_authorized": False,
        "kill_or_reserve_access_authorized": False,
        "rl_controller_authorized": False,
    }
    thresholds = {row["metric"]: row for row in contract["qualification"]["thresholds"]}
    assert thresholds["retention_h12_current_top1"]["value"] == 0.75
    assert thresholds["router_false_activation_rate"]["value"] == 0.02
    assert thresholds["restore_max_candidate_delta"]["value"] == 1e-5


def test_g1c_gate_requires_the_exact_metric_set() -> None:
    passing = {threshold.metric: threshold.value for threshold in G1C_THRESHOLDS}
    assert evaluate_g1c_gate(passing).passed is True
    failing = {**passing, "retention_h12_current_top1": 0.74}
    result = evaluate_g1c_gate(failing)
    assert result.passed is False
    assert result.blockers == ("retention_h12_current_top1",)
    with pytest.raises(ValueError, match="metric set changed"):
        evaluate_g1c_gate({**passing, "extra": 1.0})


def test_plan_compiles_balanced_privileged_pairs_retention_and_swaps() -> None:
    parent = _parent()
    plan = compile_g1c_plan(
        parent_manifest_id=parent["manifest_id"],
        parent_manifest_sha256=parent["manifest_sha256"],
    )
    audit = audit_g1c_plan(plan)
    assert audit["passed"] is True
    assert plan["action_unit_counts"] == {
        "act_k2": 6,
        "act_n7": 6,
        "act_p3": 6,
        "act_v9": 6,
    }
    assert plan["scope"]["included_splits"] == ["interface_dev"]
    assert plan["scope"]["path_contrast"] is False
    units = {unit["unit_id"]: unit for unit in plan["units"]}
    for unit in units.values():
        assert len(unit["rollout_prompt_pairs"]) == 4
        assert len(unit["retention_distractor_unit_ids"]) == 12
        assert all(
            units[distractor]["item_id"] != unit["item_id"]
            for distractor in unit["retention_distractor_unit_ids"]
        )
        wrong_swap = units[unit["wrong_swap_unit_id"]]
        assert wrong_swap["item_id"] != unit["item_id"]
        assert wrong_swap["current_action"] != unit["current_action"]
        for pair in unit["rollout_prompt_pairs"]:
            assert "<pathmem_current_state>" in pair["privileged_teacher_prompt"]
            assert "<pathmem_current_state>" not in pair["student_prompt"]


def test_g0v2_bundle_is_immutable_regenerable_and_tamper_evident(tmp_path: Path) -> None:
    output = tmp_path / "g0-v2"
    manifest_path = write_g0v2_bundle(output, PARENT_MANIFEST)
    first = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = verify_g0v2_bundle(output, PARENT_MANIFEST)
    assert report["passed"] is True
    write_g0v2_bundle(output, PARENT_MANIFEST)
    second = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert first == second

    contract_path = output / "contract.json"
    contract_path.write_text(contract_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    tampered = verify_g0v2_bundle(output, PARENT_MANIFEST)
    assert tampered["passed"] is False
    assert "artifact hash mismatch" in tampered["error"]


def test_cli_defaults_to_inspection_and_requires_explicit_output(tmp_path: Path) -> None:
    parser = build_parser()
    inspected = parser.parse_args(
        ["inspect", "--g0-v1-manifest", str(PARENT_MANIFEST), "--compact"]
    )
    assert inspected.command == "inspect"
    assert not hasattr(inspected, "output")
    output = tmp_path / "bundle"
    assert (
        main(
            [
                "prepare",
                "--g0-v1-manifest",
                str(PARENT_MANIFEST),
                "--output",
                str(output),
            ]
        )
        == 0
    )


def test_frozen_g0v1_snapshot_still_verifies() -> None:
    report = verify_g0_bundle(PROTOCOL_ROOT / "experiments/g0", PROTOCOL_ROOT)
    assert report["passed"] is True
