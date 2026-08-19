from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from plasticity_placement.pathmem.io import json_hash
from plasticity_placement.pathmem_consolidation.manifest import write_g0v2_bundle
from plasticity_placement.pathmem_consolidation_exec.cli import build_parser
from plasticity_placement.pathmem_consolidation_exec.config import (
    G1C_EXECUTION_RECIPE,
    RECIPE_ID,
    RopcdExecutionRecipe,
)
from plasticity_placement.pathmem_consolidation_exec.identity import (
    build_authorization_template,
    verify_external_approval,
    verify_source_bundle,
)
from plasticity_placement.pathmem_consolidation_exec.routing import (
    ExactKeyRouter,
    RouteTarget,
    extract_route_key,
)
from plasticity_placement.pathmem_consolidation_exec.runtime import (
    adopt_run_authorization,
    inspect_execution,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PARENT_MANIFEST = (
    REPOSITORY_ROOT
    / "notebooks/pathmem_g1_p0/protocol_snapshot/experiments/g0/manifest.json"
)


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    write_g0v2_bundle(root, PARENT_MANIFEST)
    return root


def test_frozen_recipe_is_exact_and_has_no_downstream_authority() -> None:
    recipe = G1C_EXECUTION_RECIPE
    assert recipe.recipe_id == RECIPE_ID
    assert recipe.model.name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert recipe.model.revision == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    assert recipe.rank == 16
    assert recipe.alpha == 32
    assert recipe.optimizer_steps_per_unit == 48
    assert recipe.checkpoint_interval_steps == 8
    assert recipe.trajectory_kl_weight == recipe.candidate_kl_weight == 0.5
    assert recipe.rollout_do_sample is True
    assert recipe.checkpoint_selection == "fixed_final_step"
    with pytest.raises(ValueError, match="optimizer recipe changed"):
        RopcdExecutionRecipe(learning_rate=2e-4)


def test_source_bundle_still_verifies_and_inspection_is_safe(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    output = tmp_path / "run"
    source = verify_source_bundle(bundle, PARENT_MANIFEST)
    assert source["plan_id"] == (
        "9eea51b86368aaf1b96deadd9883ddaa033c14f2f700cc8336484bd21bc6911e"
    )
    inspected = inspect_execution(
        bundle_root=bundle,
        parent_manifest=PARENT_MANIFEST,
        output_root=output,
    )
    assert inspected["safe_default"] == "inspection_only"
    assert inspected["training_started"] is False
    assert inspected["gpu_inference_started"] is False
    assert inspected["p0_authorized"] is False
    assert not output.exists()


def test_external_authorization_is_exact_idempotent_and_tamper_evident(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    output = tmp_path / "run"
    template = build_authorization_template(
        bundle_root=bundle,
        parent_manifest=PARENT_MANIFEST,
        output_root=output,
    )
    assert template["decision"] == "pending_human_review"
    assert template["permissions"] == {
        "g1c_training_authorized": True,
        "g1c_gpu_inference_authorized": True,
        "p0_authorized": False,
        "path_contrast_authorized": False,
        "kill_or_reserve_access_authorized": False,
        "rl_controller_authorized": False,
        "automatic_hyperparameter_search_authorized": False,
    }
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
    first = adopt_run_authorization(
        bundle_root=bundle,
        parent_manifest=PARENT_MANIFEST,
        output_root=output,
        approval_path=approval_path,
    )
    second = adopt_run_authorization(
        bundle_root=bundle,
        parent_manifest=PARENT_MANIFEST,
        output_root=output,
        approval_path=approval_path,
    )
    assert first == second
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["units"]) == 24
    assert manifest["training_started"] is False
    assert manifest["p0_authorized"] is False

    approval["permissions"]["p0_authorized"] = True
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    with pytest.raises(ValueError, match="template binding changed"):
        verify_external_approval(approval_path, expected_template=template)


def test_exact_key_router_rejects_ambiguity_and_false_matches() -> None:
    router = ExactKeyRouter()
    target = RouteTarget(
        unit_id="g1c:item:A",
        memory_key="key-interface_dev-01",
        adapter_dir="/tmp/adapter",
        adapter_sha256="a" * 64,
    )
    router.apply(target)
    prompt = "Context: ctx. Key: key-interface_dev-01. Return one action."
    assert extract_route_key(prompt) == "key-interface_dev-01"
    assert router.resolve_prompt(prompt) == target
    assert router.resolve_prompt("Key: control-key-1. Return one action.") is None
    with pytest.raises(ValueError, match="ambiguous"):
        extract_route_key("Key: key-a. Then Key: key-b.")
    replacement = RouteTarget(
        unit_id="g1c:item:B",
        memory_key=target.memory_key,
        adapter_dir="/tmp/other",
        adapter_sha256="b" * 64,
    )
    router.apply(replacement)
    assert router.resolve_key(target.memory_key) == replacement
    snapshot = router.snapshot()
    assert len(snapshot["router_sha256"]) == 64
    assert snapshot["application_order"] == [
        target.unit_id,
        replacement.unit_id,
    ]


def test_cli_has_bounded_explicit_lifecycle() -> None:
    parser = build_parser()
    common = [
        "--bundle",
        "bundle",
        "--g0-v1-manifest",
        "parent.json",
        "--output",
        "run",
    ]
    for command in (
        "inspect",
        "authorization-template",
        "preflight",
        "aggregate",
        "verify",
    ):
        assert parser.parse_args([command, *common]).command == command
    train = parser.parse_args(["train", *common, "--max-units", "2"])
    assert train.max_units == 2
    assert train.unit_id is None
