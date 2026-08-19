from __future__ import annotations

import random
from collections import Counter
from typing import Any

from plasticity_placement.pathmem.backends import render_fresh_session_probe
from plasticity_placement.pathmem.compiler import compile_bank
from plasticity_placement.pathmem.generator import canonical_external_note
from plasticity_placement.pathmem.identity import (
    block_seed,
    event_block_hash,
    ordered_minibatch_hash,
)
from plasticity_placement.pathmem.io import json_hash
from plasticity_placement.pathmem.schema import AccessMode, EventBlock, Probe, ProbeCategory
from plasticity_placement.pathmem_ropcd_p0.config import (
    DUPLICATE_PATH_BY_ITEM_INDEX,
    EXECUTION_ORDER_SEED,
    EXPECTED_ITEMS,
    EXPECTED_ROWS,
    EXPECTED_UNITS,
    EXPERIMENT_SEED,
    FINAL_PATHS,
    OPERATOR_ID,
    P0_ROPCD_RECIPE,
    PANEL_SEED,
    PHASE,
    PLAN_SCHEMA_VERSION,
    SPLIT,
    TRAINING_SEED,
    planning_permissions,
)


def compile_p0_plan(g1c_handoff: dict[str, Any]) -> dict[str, Any]:
    _require_g1c_handoff(g1c_handoff)
    bank = compile_bank()
    items = tuple(
        sorted((item for item in bank.items if item.split == SPLIT), key=lambda x: x.item_id)
    )
    if len(items) != EXPECTED_ITEMS:
        raise ValueError(f"R-OPCD P0 requires {EXPECTED_ITEMS} smoke items")
    blocks_by_item = {
        item.item_id: tuple(
            sorted(
                (
                    block
                    for block in bank.event_blocks
                    if block.block_id.startswith(f"{item.item_id}:")
                ),
                key=lambda block: block.block_id,
            )
        )
        for item in items
    }
    probes_by_item = {
        item.item_id: tuple(
            sorted(
                (probe for probe in bank.probes if probe.item_id == item.item_id),
                key=lambda probe: probe.probe_id,
            )
        )
        for item in items
    }
    units: list[dict[str, Any]] = []
    sequence_index = 0
    for item_index, item in enumerate(items):
        blocks = blocks_by_item[item.item_id]
        probes = probes_by_item[item.item_id]
        core_specs = (
            ("A1", None, "A1"),
            ("B1", None, "B1"),
            ("AB", "A1", "B1"),
            ("BA", "B1", "A1"),
            ("ABA", "AB", "A2"),
            ("BAA", "BA", "A2"),
            ("BAB", "BA", "B2"),
            ("ABB", "AB", "B2"),
            ("A2", None, "A2"),
            ("B2", None, "B2"),
        )
        by_logical: dict[str, dict[str, Any]] = {}
        for logical_name, parent_logical_name, block_suffix in core_specs:
            unit = _build_unit(
                item=item,
                blocks=blocks,
                probes=probes,
                logical_name=logical_name,
                parent_logical_name=parent_logical_name,
                block_suffix=block_suffix,
                sequence_index=sequence_index,
                technical_duplicate=False,
                duplicate_of=None,
            )
            units.append(unit)
            by_logical[logical_name] = unit
            sequence_index += 1
        duplicate_of = DUPLICATE_PATH_BY_ITEM_INDEX[item_index]
        original = by_logical[duplicate_of]
        units.append(
            _build_unit(
                item=item,
                blocks=blocks,
                probes=probes,
                logical_name=f"DUP-{duplicate_of}",
                parent_logical_name=str(original["parent_logical_name"]),
                block_suffix=str(original["block_suffix"]),
                sequence_index=sequence_index,
                technical_duplicate=True,
                duplicate_of=duplicate_of,
            )
        )
        sequence_index += 1
    units = _topological_execution_order(units)
    identity = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "phase": PHASE,
        "operator_id": OPERATOR_ID,
        "g1c_handoff": _handoff_identity(g1c_handoff),
        "recipe": P0_ROPCD_RECIPE.to_dict(),
        "recipe_sha256": json_hash(P0_ROPCD_RECIPE.to_dict()),
        "scope": {
            "included_splits": [SPLIT],
            "excluded_splits": [
                "interface_dev",
                "kill",
                "confirmatory",
                "hardware_dev",
                "reserve",
            ],
            "training_seed": TRAINING_SEED,
            "panel_seed": PANEL_SEED,
            "execution_order_seed": EXECUTION_ORDER_SEED,
            "execution_order_sha256": json_hash([str(unit["unit_id"]) for unit in units]),
            "primary_path_pairs": [["ABA", "BAA"], ["BAB", "ABB"]],
            "optimizer_state_lifecycle": "reset_at_semantic_consolidation",
            "side_memory_lifecycle": "isolated_child_initialized_from_exact_parent",
            "router": "deterministic_exact_key_to_selected_final_module",
        },
        "counts": {
            "items": EXPECTED_ITEMS,
            "root_adapters": EXPECTED_ITEMS,
            "units": len(units),
            "core_units": 40,
            "technical_duplicate_units": 4,
            "optimizer_steps": len(units) * P0_ROPCD_RECIPE.optimizer_steps_per_unit,
            "teacher_student_prompt_pairs": sum(
                len(unit["rollout_prompt_pairs"]) for unit in units
            ),
            "evaluation_rows": EXPECTED_ROWS,
        },
        "permissions": planning_permissions(),
        "claim_boundaries": [
            "engineering smoke only",
            "P0 cannot establish a scientific path-dependence result",
            "P0 cannot change epsilon_JS",
            "P0 does not authorize P1, kill/reserve access, learned routing, or RL control",
        ],
        "units": units,
    }
    plan = {**identity, "plan_id": json_hash(identity)}
    audit_p0_plan(plan)
    return plan


def audit_p0_plan(plan: dict[str, Any]) -> dict[str, Any]:
    identity = {key: value for key, value in plan.items() if key != "plan_id"}
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION or plan.get("plan_id") != json_hash(
        identity
    ):
        raise ValueError("R-OPCD P0 plan identity changed")
    scope = plan.get("scope", {})
    if scope.get("included_splits") != [SPLIT] or scope.get("primary_path_pairs") != [
        ["ABA", "BAA"],
        ["BAB", "ABB"],
    ]:
        raise ValueError("R-OPCD P0 scope changed")
    if plan.get("permissions") != planning_permissions():
        raise ValueError("R-OPCD P0 CPU plan contains execution authority")
    units = plan.get("units")
    if not isinstance(units, list) or len(units) != EXPECTED_UNITS:
        raise ValueError("R-OPCD P0 unit count changed")
    unit_by_id = {unit.get("unit_id"): unit for unit in units if isinstance(unit, dict)}
    if len(unit_by_id) != EXPECTED_UNITS or None in unit_by_id:
        raise ValueError("R-OPCD P0 unit IDs are incomplete or duplicated")
    item_counts = Counter(str(unit["item_id"]) for unit in units)
    if set(item_counts.values()) != {11} or len(item_counts) != EXPECTED_ITEMS:
        raise ValueError("R-OPCD P0 requires eleven isolated units per item")
    duplicates = [unit for unit in units if unit.get("technical_duplicate") is True]
    if len(duplicates) != EXPECTED_ITEMS or Counter(
        unit["duplicate_of"] for unit in duplicates
    ) != Counter(FINAL_PATHS):
        raise ValueError("R-OPCD P0 technical duplicates are not balanced")
    expected_order = _topological_execution_order([dict(unit) for unit in units])
    if [unit["unit_id"] for unit in units] != [unit["unit_id"] for unit in expected_order]:
        raise ValueError("R-OPCD P0 execution order changed")
    if [unit["sequence_index"] for unit in units] != list(range(EXPECTED_UNITS)):
        raise ValueError("R-OPCD P0 sequence indexes changed")
    if plan.get("scope", {}).get("execution_order_sha256") != json_hash(
        [str(unit["unit_id"]) for unit in units]
    ):
        raise ValueError("R-OPCD P0 execution-order hash changed")
    for unit in units:
        pairs = unit.get("rollout_prompt_pairs")
        if not isinstance(pairs, list) or len(pairs) != 4:
            raise ValueError("each R-OPCD P0 unit requires four prompt pairs")
        if any(
            "<pathmem_current_state>" not in pair["privileged_teacher_prompt"]
            or "<pathmem_current_state>" in pair["student_prompt"]
            for pair in pairs
        ):
            raise ValueError("R-OPCD P0 privileged context boundary changed")
        if unit.get("root_adapter_seed") != _root_seed(str(unit["item_id"])):
            raise ValueError("R-OPCD P0 item root seed changed")
        parent_id = unit.get("parent_unit_id")
        if parent_id is not None:
            parent = unit_by_id.get(parent_id)
            if parent is None or parent["item_id"] != unit["item_id"]:
                raise ValueError("R-OPCD P0 parent lineage escaped its item")
            if int(parent["sequence_index"]) >= int(unit["sequence_index"]):
                raise ValueError("R-OPCD P0 parent must precede its child")
    expected_counts = {
        "items": EXPECTED_ITEMS,
        "root_adapters": EXPECTED_ITEMS,
        "units": EXPECTED_UNITS,
        "core_units": 40,
        "technical_duplicate_units": 4,
        "optimizer_steps": EXPECTED_UNITS * P0_ROPCD_RECIPE.optimizer_steps_per_unit,
        "teacher_student_prompt_pairs": EXPECTED_UNITS * 4,
        "evaluation_rows": EXPECTED_ROWS,
    }
    if plan.get("counts") != expected_counts:
        raise ValueError("R-OPCD P0 plan counts changed")
    return {
        "passed": True,
        "plan_id": plan["plan_id"],
        "g1c_run_id": plan["g1c_handoff"]["run_id"],
        "counts": expected_counts,
        "training_started": False,
        "gpu_inference_started": False,
        "path_contrast_computed": False,
        "p1_authorized": False,
    }


def _build_unit(
    *,
    item: Any,
    blocks: tuple[EventBlock, ...],
    probes: tuple[Probe, ...],
    logical_name: str,
    parent_logical_name: str | None,
    block_suffix: str,
    sequence_index: int,
    technical_duplicate: bool,
    duplicate_of: str | None,
) -> dict[str, Any]:
    block = next(
        candidate for candidate in blocks if candidate.block_id.endswith(f":{block_suffix}")
    )
    terminal_state = block.state_label
    current_action = item.state_a_action if terminal_state == "A" else item.state_b_action
    obsolete_action = item.state_b_action if terminal_state == "A" else item.state_a_action
    qualification = tuple(
        probe
        for probe in probes
        if probe.terminal_state == terminal_state and probe.category is ProbeCategory.QUALIFICATION
    )
    core = tuple(
        probe for probe in probes if probe.terminal_state == terminal_state and probe.core_endpoint
    )
    unrelated = tuple(
        probe
        for probe in probes
        if probe.terminal_state == terminal_state and probe.category is ProbeCategory.UNRELATED
    )
    if len(qualification) != 4 or len(core) != 14 or len(unrelated) != 8:
        raise ValueError(f"R-OPCD P0 probe panel changed: {item.item_id}/{terminal_state}")
    evaluation_qualification = _panel_order(
        qualification, item.item_id, terminal_state, "qualification"
    )
    evaluation_core = _panel_order(core, item.item_id, terminal_state, "core")
    evaluation_unrelated = _panel_order(unrelated, item.item_id, terminal_state, "unrelated")
    teacher_context = canonical_external_note(item, terminal_state)
    prompt_pairs = [
        {
            "probe_id": probe.probe_id,
            "student_prompt": render_fresh_session_probe(
                probe, persistent_state="unused", access_mode=AccessMode.OFF
            ),
            "privileged_teacher_prompt": render_fresh_session_probe(
                probe, persistent_state=teacher_context, access_mode=AccessMode.ON
            ),
            "action_choices": list(probe.action_choices),
            "current_action": current_action,
            "obsolete_action": obsolete_action,
        }
        for probe in qualification
    ]
    unit_id = _unit_id(item.item_id, logical_name)
    parent_unit_id = (
        None if parent_logical_name is None else _unit_id(item.item_id, parent_logical_name)
    )
    block_sha256 = event_block_hash(block)
    occurrence_id = block.block_id.rsplit(":", 1)[-1]
    unit_block_seed = block_seed(TRAINING_SEED, block_sha256, occurrence_id)
    step_seed_namespace = json_hash(
        {
            "training_seed": TRAINING_SEED,
            "event_block_sha256": block_sha256,
            "occurrence_id": occurrence_id,
        }
    )
    ordered_exposure = [
        {
            "step": step,
            "pair_index": step % len(prompt_pairs),
            "probe_id": prompt_pairs[step % len(prompt_pairs)]["probe_id"],
            "step_seed": int(json_hash([TRAINING_SEED, step_seed_namespace, step])[:8], 16),
        }
        for step in range(P0_ROPCD_RECIPE.optimizer_steps_per_unit)
    ]
    return {
        "unit_id": unit_id,
        "sequence_index": sequence_index,
        "item_id": item.item_id,
        "content_domain": item.content_domain.value,
        "logical_name": logical_name,
        "parent_logical_name": parent_logical_name,
        "parent_unit_id": parent_unit_id,
        "block_id": block.block_id,
        "block_suffix": block_suffix,
        "source_event_block_sha256": json_hash(block.to_dict()),
        "ordered_minibatch_sha256": ordered_minibatch_hash(block, TRAINING_SEED),
        "block_seed": unit_block_seed,
        "step_seed_namespace": step_seed_namespace,
        "ordered_exposure_sha256": json_hash(ordered_exposure),
        "root_adapter_seed": _root_seed(item.item_id),
        "terminal_state": terminal_state,
        "memory_key": item.memory_key,
        "route_key": item.memory_key,
        "side_memory_namespace": f"pathmem/p0-r-opcd/{item.item_id}/{logical_name}",
        "current_action": current_action,
        "obsolete_action": obsolete_action,
        "action_choices": list(item.action_choices),
        "canonical_teacher_context": teacher_context,
        "rollout_prompt_pairs": prompt_pairs,
        "core_probe_ids": [probe.probe_id for probe in evaluation_core],
        "qualification_probe_ids": [probe.probe_id for probe in evaluation_qualification],
        "unrelated_probe_ids": [probe.probe_id for probe in evaluation_unrelated],
        "technical_duplicate": technical_duplicate,
        "duplicate_of": duplicate_of,
    }


def _unit_id(item_id: str, logical_name: str) -> str:
    return f"p0r:{item_id}:seed-{TRAINING_SEED}:{logical_name}"


def _root_seed(item_id: str) -> int:
    return int(
        json_hash(
            {
                "experiment_seed": EXPERIMENT_SEED,
                "item_id": item_id,
                "training_seed": TRAINING_SEED,
                "operator": OPERATOR_ID,
            }
        )[:16],
        16,
    )


def _panel_order(
    probes: tuple[Probe, ...], item_id: str, terminal_state: str, panel: str
) -> tuple[Probe, ...]:
    ordered = list(sorted(probes, key=lambda probe: probe.probe_id))
    seed = int(
        json_hash(
            {
                "panel_seed": PANEL_SEED,
                "item_id": item_id,
                "terminal_state": terminal_state,
                "panel": panel,
            }
        )[:16],
        16,
    )
    random.Random(seed).shuffle(ordered)
    return tuple(ordered)


def _topological_execution_order(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    remaining = {str(unit["unit_id"]): dict(unit) for unit in units}
    emitted: set[str] = set()
    ordered: list[dict[str, Any]] = []
    rng = random.Random(EXECUTION_ORDER_SEED)
    while remaining:
        ready = sorted(
            (
                unit_id
                for unit_id, unit in remaining.items()
                if unit["parent_unit_id"] is None or unit["parent_unit_id"] in emitted
            )
        )
        if not ready:
            raise ValueError("R-OPCD P0 unit graph is cyclic")
        selected = ready[rng.randrange(len(ready))]
        unit = remaining.pop(selected)
        unit["sequence_index"] = len(ordered)
        ordered.append(unit)
        emitted.add(selected)
    return ordered


def _handoff_identity(handoff: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "run_id",
        "source_manifest_id",
        "plan_id",
        "g1c_authorization_id",
        "g1c_implementation_sha256",
        "g1c_git_revision",
        "summary_sha256",
        "handoff_id",
    )
    return {key: handoff[key] for key in keys}


def _require_g1c_handoff(handoff: dict[str, Any]) -> None:
    if handoff.get("passed") is not True or handoff.get("g1c_gate", {}).get("passed") is not True:
        raise PermissionError("R-OPCD P0 planning requires a passing G1-C handoff")
    if handoff.get("p0_runner_implementation_review_eligible") is not True:
        raise PermissionError("G1-C handoff is not eligible for P0 implementation review")
    if (
        handoff.get("p0_authorized") is not False
        or handoff.get("path_contrast_computed") is not False
    ):
        raise ValueError("G1-C source contains downstream P0/path state")
    expected = {key: value for key, value in handoff.items() if key != "handoff_id"}
    if handoff.get("handoff_id") != json_hash(expected):
        raise ValueError("G1-C handoff identity changed")
