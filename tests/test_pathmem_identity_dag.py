from __future__ import annotations

from pathlib import Path

import pytest

from plasticity_placement.pathmem.compiler import compile_history_families
from plasticity_placement.pathmem.dag import build_core_dag_plan, resolve_node_identity
from plasticity_placement.pathmem.generator import build_item_bank, compile_event_blocks
from plasticity_placement.pathmem.identity import (
    ResultIdentityInput,
    RuntimeIdentity,
    block_seed,
    event_block_hash,
    ordered_minibatch_hash,
    root_adapter_seed,
    verify_parent_lineage,
)
from plasticity_placement.pathmem.run_manifest import (
    PathRunManifest,
    UnitState,
    register_verified_artifact,
)
from plasticity_placement.pathmem.schema import OperatorKind


def _runtime(item_id: str, *, training_seed: int = 11) -> RuntimeIdentity:
    return RuntimeIdentity(
        experiment_seed=20260813,
        item_id=item_id,
        training_seed=training_seed,
        operator=OperatorKind.LORA_ADAMW_RESET,
        base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        base_model_revision="revision",
        root_adapter_sha256="a" * 64,
        trainer_impl_sha256="b" * 64,
        trainer_config_sha256="c" * 64,
        tokenizer_sha256="d" * 64,
        library_lock_sha256="e" * 64,
        code_and_patch_sha256="f" * 64,
        resolved_precision="float32",
    )


def test_block_rng_depends_on_block_identity_not_path_position() -> None:
    item = build_item_bank()[0]
    block = compile_event_blocks(item)[0]
    block_hash = event_block_hash(block)
    assert block_seed(7, block_hash, "A1") == block_seed(7, block_hash, "A1")
    assert block_seed(7, block_hash, "A1") != block_seed(8, block_hash, "A1")
    assert ordered_minibatch_hash(block, 7) == ordered_minibatch_hash(block, 7)


def test_core_dag_has_ten_nodes_and_shared_prefixes() -> None:
    item = build_item_bank()[0]
    blocks = compile_event_blocks(item)
    families = compile_history_families(item, blocks)
    runtime = _runtime(item.item_id)
    plan = build_core_dag_plan(families, blocks, runtime)
    assert len(plan.nodes) == 10
    nodes = {node.logical_name: node for node in plan.nodes}
    assert nodes["AB"].parent_logical_name == "A1"
    assert nodes["BA"].parent_logical_name == "B1"
    assert nodes["ABA"].parent_logical_name == "AB"
    assert nodes["ABB"].parent_logical_name == "AB"
    assert nodes["A2"].current_only_anchor is True
    assert nodes["B2"].current_only_anchor is True
    assert build_core_dag_plan(families, blocks, runtime) == plan


def test_resolved_node_and_result_identity_cover_parent_and_access_mode() -> None:
    item = build_item_bank()[0]
    blocks = compile_event_blocks(item)
    families = compile_history_families(item, blocks)
    runtime = _runtime(item.item_id)
    dag = build_core_dag_plan(families, blocks, runtime)
    first = dag.nodes[0]
    node_a = resolve_node_identity(
        plan=first,
        dag=dag,
        runtime=runtime,
        parent_state_sha256=runtime.root_state_sha256,
    )
    node_b = resolve_node_identity(
        plan=first,
        dag=dag,
        runtime=runtime,
        parent_state_sha256="1" * 64,
    )
    assert node_a.node_id != node_b.node_id
    assert node_a.node_id == resolve_node_identity(
        plan=first,
        dag=dag,
        runtime=runtime,
        parent_state_sha256=runtime.root_state_sha256,
    ).node_id
    common = {
        "node_id": node_a.node_id,
        "probe_bank_sha256": "2" * 64,
        "renderer_and_prompt_sha256": "3" * 64,
        "external_state_sha256": "4" * 64,
        "lifecycle_trace_sha256": "5" * 64,
        "scorer_sha256": "6" * 64,
        "evaluation_precision": "float32",
    }
    on = ResultIdentityInput(**common, access_mode="memory_on")
    off = ResultIdentityInput(**common, access_mode="memory_off")
    assert on.result_id != off.result_id


def test_root_seed_and_lineage_validation_are_operator_specific_and_strict() -> None:
    item_id = build_item_bank()[0].item_id
    reset = root_adapter_seed(1, item_id, 2, OperatorKind.LORA_ADAMW_RESET)
    carry = root_adapter_seed(1, item_id, 2, OperatorKind.LORA_ADAMW_CARRY)
    assert reset != carry
    verify_parent_lineage(
        expected_parent_sha256="a" * 64,
        observed_parent_sha256="a" * 64,
        expected_root_sha256="b" * 64,
        observed_root_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="parent hash mismatch"):
        verify_parent_lineage(
            expected_parent_sha256="a" * 64,
            observed_parent_sha256="c" * 64,
            expected_root_sha256="b" * 64,
            observed_root_sha256="b" * 64,
        )


def test_path_run_manifest_freezes_plans_transitions_and_technical_duplicates(
    tmp_path: Path,
) -> None:
    item = build_item_bank()[0]
    blocks = compile_event_blocks(item)
    runtime = _runtime(item.item_id)
    dag = build_core_dag_plan(compile_history_families(item, blocks), blocks, runtime)
    source_attempt = f"{item.item_id}::seed-{runtime.training_seed}::ABA"
    manifest_path = tmp_path / "run" / "manifest.json"
    manifest = PathRunManifest.load_or_create(
        manifest_path,
        phase="P0",
        g0_manifest_id="9" * 64,
        run_identity={"code": "frozen"},
        dag_plans=(dag,),
        technical_duplicates=(source_attempt,),
    )
    assert len(manifest.payload["units"]) == 11
    duplicate_id = f"{source_attempt}::technical-duplicate"
    assert manifest.payload["units"][duplicate_id]["node_plan_id"] == (
        manifest.payload["units"][source_attempt]["node_plan_id"]
    )
    manifest.mark_unit(source_attempt, UnitState.TRAINING)
    manifest.mark_unit(source_attempt, UnitState.TRAINED)
    manifest.mark_unit(source_attempt, UnitState.SCORING)
    manifest.mark_unit(source_attempt, UnitState.VERIFIED, result_id="a" * 64)
    assert manifest.unit_state(source_attempt) is UnitState.VERIFIED
    assert manifest.payload["training_started"] is True
    with pytest.raises(ValueError, match="invalid unit transition"):
        manifest.mark_unit(source_attempt, UnitState.TRAINED)
    reloaded = PathRunManifest.load_or_create(
        manifest_path,
        phase="P0",
        g0_manifest_id="9" * 64,
        run_identity={"code": "frozen"},
        dag_plans=(dag,),
        technical_duplicates=(source_attempt,),
    )
    assert reloaded.unit_state(source_attempt) is UnitState.VERIFIED


def test_artifact_index_rejects_lineage_or_node_collisions(tmp_path: Path) -> None:
    artifact = tmp_path / "adapter.safetensors"
    artifact.write_bytes(b"verified adapter")
    index = tmp_path / "artifact-index.json"
    artifact_hash = register_verified_artifact(
        index,
        node_id="1" * 64,
        artifact_path=artifact,
        expected_parent_sha256="2" * 64,
        observed_parent_sha256="2" * 64,
        expected_root_sha256="3" * 64,
        observed_root_sha256="3" * 64,
    )
    assert len(artifact_hash) == 64
    assert register_verified_artifact(
        index,
        node_id="1" * 64,
        artifact_path=artifact,
        expected_parent_sha256="2" * 64,
        observed_parent_sha256="2" * 64,
        expected_root_sha256="3" * 64,
        observed_root_sha256="3" * 64,
    ) == artifact_hash
    replacement = tmp_path / "replacement.safetensors"
    replacement.write_bytes(b"different")
    with pytest.raises(ValueError, match="different artifact"):
        register_verified_artifact(
            index,
            node_id="1" * 64,
            artifact_path=replacement,
            expected_parent_sha256="2" * 64,
            observed_parent_sha256="2" * 64,
            expected_root_sha256="3" * 64,
            observed_root_sha256="3" * 64,
        )
