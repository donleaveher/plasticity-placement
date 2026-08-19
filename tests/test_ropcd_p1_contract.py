from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from plasticity_placement.pathmem.io import json_hash
from plasticity_placement.pathmem_ropcd_p1.cli import build_parser
from plasticity_placement.pathmem_ropcd_p1.config import (
    BENCHMARK_PROFILE_SCHEMA_VERSION,
    EXPECTED_ROWS,
    EXPECTED_UNITS,
    P1_ROPCD_RECIPE,
    TRAINING_SEEDS,
    benchmark_permissions,
    execution_permissions,
    planning_permissions,
)
from plasticity_placement.pathmem_ropcd_p1.identity import (
    build_execution_authorization_template,
    implementation_identity,
)
from plasticity_placement.pathmem_ropcd_p1.manifest import P1ExecutionManifest, UnitState
from plasticity_placement.pathmem_ropcd_p1.planner import audit_p1_plan, compile_p1_plan


def _handoff() -> dict[str, Any]:
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


def _plan() -> dict[str, Any]:
    return compile_p1_plan(_handoff())


def test_p1_plan_freezes_scale_recipe_seeds_and_safe_permissions() -> None:
    plan = _plan()
    assert audit_p1_plan(plan)["passed"] is True
    assert plan["recipe"] == P1_ROPCD_RECIPE.to_dict()
    assert plan["counts"] == {
        "items": 12,
        "training_seeds": 3,
        "item_seed_cells": 36,
        "root_adapters": 36,
        "units": EXPECTED_UNITS,
        "core_units": 360,
        "technical_duplicate_units": 36,
        "optimizer_steps": 19_008,
        "teacher_student_prompt_pairs": 1_584,
        "evaluation_rows": EXPECTED_ROWS,
        "hardware_benchmark_units": 4,
        "hardware_benchmark_optimizer_steps": 192,
    }
    assert EXPECTED_UNITS == 396
    assert EXPECTED_ROWS == 18_840
    assert {unit["training_seed"] for unit in plan["units"]} == set(TRAINING_SEEDS)
    assert {unit["item_split"] for unit in plan["units"]} == {"kill"}
    assert {unit["item_split"] for unit in plan["benchmark_units"]} == {"hardware_dev"}
    assert plan["permissions"] == planning_permissions()
    assert all(value is False for value in plan["permissions"].values())


def test_p1_plan_has_36_isolated_lineages_and_balanced_duplicates() -> None:
    plan = _plan()
    cells = Counter((unit["item_id"], unit["training_seed"]) for unit in plan["units"])
    assert len(cells) == 36
    assert set(cells.values()) == {11}
    assert len({unit["root_id"] for unit in plan["units"]}) == 36
    assert Counter(
        unit["duplicate_of"] for unit in plan["units"] if unit["technical_duplicate"]
    ) == {"ABA": 9, "BAA": 9, "BAB": 9, "ABB": 9}
    by_cell = {
        (unit["item_id"], unit["training_seed"], unit["logical_name"]): unit
        for unit in plan["units"]
    }
    for item, seed in cells:
        assert (
            by_cell[(item, seed, "ABA")]["ordered_exposure_sha256"]
            == by_cell[(item, seed, "BAA")]["ordered_exposure_sha256"]
        )
        assert (
            by_cell[(item, seed, "BAB")]["ordered_exposure_sha256"]
            == by_cell[(item, seed, "ABB")]["ordered_exposure_sha256"]
        )


def test_p1_authority_is_split_between_benchmark_and_formal_execution() -> None:
    benchmark = benchmark_permissions()
    formal = execution_permissions()
    assert benchmark["hardware_dev_benchmark_training_authorized"] is True
    assert benchmark["p1_training_authorized"] is False
    assert benchmark["kill_path_contrast_authorized"] is False
    assert formal["hardware_dev_benchmark_training_authorized"] is False
    assert formal["p1_training_authorized"] is True
    assert formal["kill_path_contrast_authorized"] is True
    for permissions in (benchmark, formal):
        for field in (
            "reserve_access_authorized",
            "p1b_authorized",
            "p2_authorized",
            "learned_router_authorized",
            "rl_controller_authorized",
            "automatic_hyperparameter_search_authorized",
        ):
            assert permissions[field] is False


def test_formal_authorization_requires_exact_current_resource_profile(tmp_path: Path) -> None:
    plan_root = tmp_path / "plan"
    plan_root.mkdir()
    (plan_root / "manifest.json").write_text("{}", encoding="utf-8")
    (plan_root / "p1_plan.json").write_text("{}", encoding="utf-8")
    _, implementation_sha256 = implementation_identity()
    profile_identity = {
        "schema_version": BENCHMARK_PROFILE_SCHEMA_VERSION,
        "verified": True,
        "benchmark_id": "b" * 64,
        "authorization_id": "a" * 64,
        "plan_id": "p" * 64,
        "g2_handoff_id": "h" * 64,
        "implementation_sha256": implementation_sha256,
        "hardware_split": "hardware_dev",
        "measurements": {},
        "projection": {"p1_wall_seconds": 123.0},
        "scientific_outcomes_computed": False,
        "formal_p1_authorized": False,
        "p1b_authorized": False,
        "p2_authorized": False,
        "recipe_sha256": "q" * 64,
    }
    profile = {**profile_identity, "profile_id": json_hash(profile_identity)}
    profile_path = tmp_path / "resource_profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    report = {
        "passed": True,
        "manifest_id": "m" * 64,
        "plan_id": "p" * 64,
        "g2_handoff_id": "h" * 64,
        "g2_repair_id": "r" * 64,
    }
    template = build_execution_authorization_template(
        plan_root=plan_root,
        output_root=tmp_path / "run",
        plan_report=report,
        resource_profile_path=profile_path,
    )
    assert template["resource_profile"]["profile_id"] == profile["profile_id"]
    assert template["permissions"] == execution_permissions()
    profile["plan_id"] = "x" * 64
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(PermissionError, match="verified resource profile"):
        build_execution_authorization_template(
            plan_root=plan_root,
            output_root=tmp_path / "run",
            plan_report=report,
            resource_profile_path=profile_path,
        )


def test_p1_manifest_tracks_item_seed_roots_and_blocks_downstream(tmp_path: Path) -> None:
    plan = _plan()
    manifest = P1ExecutionManifest.load_or_create(
        tmp_path / "manifest.json",
        plan_manifest_id="m" * 64,
        plan=plan,
        authorization={
            "authorization_id": "a" * 64,
            "resource_profile": {"profile_id": "p" * 64},
        },
        implementation_sha256="i" * 64,
    )
    assert len(manifest.payload["roots"]) == 36
    unit_id = plan["units"][0]["unit_id"]
    manifest.mark_unit(unit_id, UnitState.TRAINING)
    manifest.mark_unit(unit_id, UnitState.TRAINED)
    assert manifest.payload["kill_split_accessed"] is True
    manifest.payload["p1b_authorized"] = True
    with pytest.raises(ValueError, match="unauthorized downstream"):
        manifest.save()


def test_p1_cli_exposes_bounded_lifecycle_only() -> None:
    choices = build_parser()._subparsers._group_actions[0].choices
    assert set(choices) == {
        "inspect",
        "prepare-plan",
        "verify-plan",
        "benchmark-authorization-template",
        "benchmark-adopt-authorization",
        "benchmark-run",
        "benchmark-aggregate",
        "benchmark-verify",
        "authorization-template",
        "adopt-authorization",
        "preflight",
        "train",
        "evaluate",
        "aggregate",
        "verify",
    }
    assert not ({"p1b", "p2", "reserve", "router", "rl"} & set(choices))
