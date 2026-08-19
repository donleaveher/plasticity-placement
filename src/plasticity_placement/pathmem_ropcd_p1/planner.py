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
from plasticity_placement.pathmem_ropcd_p1.config import (
    DUPLICATE_PATHS,
    EXECUTION_ORDER_SEED,
    EXPECTED_BENCHMARK_UNITS,
    EXPECTED_CORE_UNITS,
    EXPECTED_DUPLICATE_UNITS,
    EXPECTED_ITEMS,
    EXPECTED_ROOTS,
    EXPECTED_ROWS,
    EXPECTED_UNITS,
    EXPERIMENT_SEED,
    FINAL_PATHS,
    HARDWARE_SPLIT,
    OPERATOR_ID,
    P1_ROPCD_RECIPE,
    PANEL_SEED,
    PHASE,
    PLAN_SCHEMA_VERSION,
    SPLIT,
    TRAINING_SEEDS,
    planning_permissions,
)


def compile_p1_plan(g2_handoff: dict[str, Any]) -> dict[str, Any]:
    _require_g2_handoff(g2_handoff)
    bank = compile_bank()
    items = tuple(
        sorted((item for item in bank.items if item.split == SPLIT), key=lambda item: item.item_id)
    )
    hardware_items = tuple(
        sorted(
            (item for item in bank.items if item.split == HARDWARE_SPLIT),
            key=lambda item: item.item_id,
        )
    )
    if len(items) != EXPECTED_ITEMS or len(hardware_items) != EXPECTED_BENCHMARK_UNITS:
        raise ValueError("R-OPCD P1 split cardinality changed")
    blocks_by_item = _blocks_by_item(bank.event_blocks)
    probes_by_item = _probes_by_item(bank.probes)
    units: list[dict[str, Any]] = []
    cell_index = 0
    for item in items:
        for training_seed in TRAINING_SEEDS:
            cell_units = _build_cell(
                item=item,
                blocks=blocks_by_item[item.item_id],
                probes=probes_by_item[item.item_id],
                training_seed=training_seed,
                duplicate_of=DUPLICATE_PATHS[cell_index % len(DUPLICATE_PATHS)],
            )
            units.extend(cell_units)
            cell_index += 1
    units = _topological_execution_order(units)
    benchmark_units = [
        _build_unit(
            item=item,
            blocks=blocks_by_item[item.item_id],
            probes=probes_by_item[item.item_id],
            training_seed=TRAINING_SEEDS[0],
            logical_name="A2",
            parent_logical_name=None,
            block_suffix="A2",
            technical_duplicate=False,
            duplicate_of=None,
            unit_prefix="p1bmk",
            namespace_prefix="pathmem/p1-r-opcd-hardware-benchmark",
        )
        for item in hardware_items
    ]
    for index, unit in enumerate(benchmark_units):
        unit["sequence_index"] = index
    identity = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "phase": PHASE,
        "operator_id": OPERATOR_ID,
        "g2_handoff": _handoff_identity(g2_handoff),
        "recipe": P1_ROPCD_RECIPE.to_dict(),
        "recipe_sha256": json_hash(P1_ROPCD_RECIPE.to_dict()),
        "scope": {
            "included_splits": [SPLIT],
            "hardware_benchmark_split": HARDWARE_SPLIT,
            "excluded_scientific_splits": [
                "interface_dev",
                "smoke",
                "confirmatory",
                "hardware_dev",
                "reserve",
            ],
            "training_seeds": list(TRAINING_SEEDS),
            "panel_seed": PANEL_SEED,
            "execution_order_seed": EXECUTION_ORDER_SEED,
            "execution_order_sha256": json_hash(
                [str(unit["unit_id"]) for unit in units]
            ),
            "primary_path_pairs": [["ABA", "BAA"], ["BAB", "ABB"]],
            "optimizer_state_lifecycle": "reset_at_semantic_consolidation",
            "side_memory_lifecycle": "isolated_child_initialized_from_exact_parent",
            "router": "deterministic_exact_key_to_selected_final_module",
            "resource_profile_required_before_execution_authorization": True,
        },
        "analysis": {
            "primary_endpoint": "item_clustered_mean_endpoint_js_nats",
            "epsilon_js_nats": 0.02,
            "bootstrap_replicates": 10_000,
            "bootstrap_seed": EXPERIMENT_SEED,
            "seed_aggregation": "average_within_item_before_item_bootstrap",
            "decision_rule": {
                "meaningful_violation": "lower_95_gt_epsilon",
                "approximate_consistency": "upper_95_lt_epsilon",
                "inconclusive": "interval_intersects_epsilon",
            },
            "intention_to_audit": True,
            "failed_paths_filtered": False,
        },
        "counts": {
            "items": EXPECTED_ITEMS,
            "training_seeds": len(TRAINING_SEEDS),
            "item_seed_cells": EXPECTED_ROOTS,
            "root_adapters": EXPECTED_ROOTS,
            "units": len(units),
            "core_units": EXPECTED_CORE_UNITS,
            "technical_duplicate_units": EXPECTED_DUPLICATE_UNITS,
            "optimizer_steps": len(units) * P1_ROPCD_RECIPE.optimizer_steps_per_unit,
            "teacher_student_prompt_pairs": sum(
                len(unit["rollout_prompt_pairs"]) for unit in units
            ),
            "evaluation_rows": EXPECTED_ROWS,
            "hardware_benchmark_units": len(benchmark_units),
            "hardware_benchmark_optimizer_steps": (
                len(benchmark_units) * P1_ROPCD_RECIPE.optimizer_steps_per_unit
            ),
        },
        "permissions": planning_permissions(),
        "claim_boundaries": [
            "P1 is an investment and kill decision, not an ICLR-level result",
            "P1 cannot authorize P1b or P2 automatically",
            "hardware_dev measurements are resource evidence only",
            "reserve and confirmatory data remain unopened",
            "learned routing, RL control, and automatic search remain unauthorized",
        ],
        "benchmark_units": benchmark_units,
        "units": units,
    }
    plan = {**identity, "plan_id": json_hash(identity)}
    audit_p1_plan(plan)
    return plan


def audit_p1_plan(plan: dict[str, Any]) -> dict[str, Any]:
    identity = {key: value for key, value in plan.items() if key != "plan_id"}
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION or plan.get("plan_id") != json_hash(
        identity
    ):
        raise ValueError("R-OPCD P1 plan identity changed")
    if plan.get("permissions") != planning_permissions():
        raise ValueError("R-OPCD P1 CPU plan contains execution authority")
    scope = plan.get("scope", {})
    if (
        scope.get("included_splits") != [SPLIT]
        or scope.get("training_seeds") != list(TRAINING_SEEDS)
        or scope.get("primary_path_pairs") != [["ABA", "BAA"], ["BAB", "ABB"]]
    ):
        raise ValueError("R-OPCD P1 scope changed")
    units = plan.get("units")
    if not isinstance(units, list) or len(units) != EXPECTED_UNITS:
        raise ValueError("R-OPCD P1 unit count changed")
    unit_by_id = {unit.get("unit_id"): unit for unit in units if isinstance(unit, dict)}
    if len(unit_by_id) != EXPECTED_UNITS or None in unit_by_id:
        raise ValueError("R-OPCD P1 unit IDs are incomplete or duplicated")
    cells = Counter((str(unit["item_id"]), int(unit["training_seed"])) for unit in units)
    if len(cells) != EXPECTED_ROOTS or set(cells.values()) != {11}:
        raise ValueError("R-OPCD P1 requires eleven units per item-seed cell")
    duplicates = [unit for unit in units if unit.get("technical_duplicate") is True]
    if len(duplicates) != EXPECTED_DUPLICATE_UNITS or Counter(
        str(unit["duplicate_of"]) for unit in duplicates
    ) != Counter({path: 9 for path in FINAL_PATHS}):
        raise ValueError("R-OPCD P1 technical duplicates are not path-balanced")
    if [unit["sequence_index"] for unit in units] != list(range(EXPECTED_UNITS)):
        raise ValueError("R-OPCD P1 sequence indexes changed")
    expected_order = _topological_execution_order([dict(unit) for unit in units])
    if [unit["unit_id"] for unit in units] != [unit["unit_id"] for unit in expected_order]:
        raise ValueError("R-OPCD P1 execution order changed")
    if scope.get("execution_order_sha256") != json_hash(
        [str(unit["unit_id"]) for unit in units]
    ):
        raise ValueError("R-OPCD P1 execution-order hash changed")
    for unit in units:
        _audit_unit(unit, unit_by_id)
    benchmark_units = plan.get("benchmark_units")
    if not isinstance(benchmark_units, list) or len(benchmark_units) != EXPECTED_BENCHMARK_UNITS:
        raise ValueError("R-OPCD P1 hardware benchmark unit count changed")
    if any(unit.get("item_split") != HARDWARE_SPLIT for unit in benchmark_units):
        raise ValueError("R-OPCD P1 benchmark escaped hardware_dev")
    expected_counts = {
        "items": EXPECTED_ITEMS,
        "training_seeds": len(TRAINING_SEEDS),
        "item_seed_cells": EXPECTED_ROOTS,
        "root_adapters": EXPECTED_ROOTS,
        "units": EXPECTED_UNITS,
        "core_units": EXPECTED_CORE_UNITS,
        "technical_duplicate_units": EXPECTED_DUPLICATE_UNITS,
        "optimizer_steps": EXPECTED_UNITS * P1_ROPCD_RECIPE.optimizer_steps_per_unit,
        "teacher_student_prompt_pairs": EXPECTED_UNITS * 4,
        "evaluation_rows": EXPECTED_ROWS,
        "hardware_benchmark_units": EXPECTED_BENCHMARK_UNITS,
        "hardware_benchmark_optimizer_steps": (
            EXPECTED_BENCHMARK_UNITS * P1_ROPCD_RECIPE.optimizer_steps_per_unit
        ),
    }
    if plan.get("counts") != expected_counts:
        raise ValueError("R-OPCD P1 plan counts changed")
    return {
        "passed": True,
        "plan_id": plan["plan_id"],
        "g2_repair_id": plan["g2_handoff"]["repair_id"],
        "counts": expected_counts,
        "training_started": False,
        "gpu_inference_started": False,
        "kill_path_contrast_computed": False,
        "p1b_authorized": False,
        "p2_authorized": False,
    }


def _build_cell(
    *,
    item: Any,
    blocks: tuple[EventBlock, ...],
    probes: tuple[Probe, ...],
    training_seed: int,
    duplicate_of: str,
) -> list[dict[str, Any]]:
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
    units = [
        _build_unit(
            item=item,
            blocks=blocks,
            probes=probes,
            training_seed=training_seed,
            logical_name=logical_name,
            parent_logical_name=parent_logical_name,
            block_suffix=block_suffix,
            technical_duplicate=False,
            duplicate_of=None,
            unit_prefix="p1r",
            namespace_prefix="pathmem/p1-r-opcd",
        )
        for logical_name, parent_logical_name, block_suffix in core_specs
    ]
    original = next(unit for unit in units if unit["logical_name"] == duplicate_of)
    units.append(
        _build_unit(
            item=item,
            blocks=blocks,
            probes=probes,
            training_seed=training_seed,
            logical_name=f"DUP-{duplicate_of}",
            parent_logical_name=str(original["parent_logical_name"]),
            block_suffix=str(original["block_suffix"]),
            technical_duplicate=True,
            duplicate_of=duplicate_of,
            unit_prefix="p1r",
            namespace_prefix="pathmem/p1-r-opcd",
        )
    )
    return units


def _build_unit(
    *,
    item: Any,
    blocks: tuple[EventBlock, ...],
    probes: tuple[Probe, ...],
    training_seed: int,
    logical_name: str,
    parent_logical_name: str | None,
    block_suffix: str,
    technical_duplicate: bool,
    duplicate_of: str | None,
    unit_prefix: str,
    namespace_prefix: str,
) -> dict[str, Any]:
    block = next(candidate for candidate in blocks if candidate.block_id.endswith(block_suffix))
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
        raise ValueError(f"R-OPCD P1 probe panel changed: {item.item_id}/{terminal_state}")
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
    unit_id = _unit_id(unit_prefix, item.item_id, training_seed, logical_name)
    parent_unit_id = (
        None
        if parent_logical_name is None
        else _unit_id(unit_prefix, item.item_id, training_seed, parent_logical_name)
    )
    block_sha256 = event_block_hash(block)
    occurrence_id = block.block_id.rsplit(":", 1)[-1]
    namespace = json_hash(
        {
            "training_seed": training_seed,
            "event_block_sha256": block_sha256,
            "occurrence_id": occurrence_id,
        }
    )
    exposure = [
        {
            "step": step,
            "pair_index": step % len(prompt_pairs),
            "probe_id": prompt_pairs[step % len(prompt_pairs)]["probe_id"],
            "step_seed": int(json_hash([training_seed, namespace, step])[:8], 16),
        }
        for step in range(P1_ROPCD_RECIPE.optimizer_steps_per_unit)
    ]
    return {
        "unit_id": unit_id,
        "sequence_index": -1,
        "item_id": item.item_id,
        "item_split": item.split,
        "content_domain": item.content_domain.value,
        "training_seed": training_seed,
        "root_id": f"{item.item_id}:seed-{training_seed}",
        "root_adapter_seed": _root_seed(item.item_id, training_seed, unit_prefix),
        "logical_name": logical_name,
        "parent_logical_name": parent_logical_name,
        "parent_unit_id": parent_unit_id,
        "block_id": block.block_id,
        "block_suffix": block_suffix,
        "source_event_block_sha256": json_hash(block.to_dict()),
        "ordered_minibatch_sha256": ordered_minibatch_hash(block, training_seed),
        "block_seed": block_seed(training_seed, block_sha256, occurrence_id),
        "step_seed_namespace": namespace,
        "ordered_exposure_sha256": json_hash(exposure),
        "terminal_state": terminal_state,
        "memory_key": item.memory_key,
        "route_key": item.memory_key,
        "side_memory_namespace": (
            f"{namespace_prefix}/{item.item_id}/seed-{training_seed}/{logical_name}"
        ),
        "current_action": current_action,
        "obsolete_action": obsolete_action,
        "action_choices": list(item.action_choices),
        "canonical_teacher_context": teacher_context,
        "rollout_prompt_pairs": prompt_pairs,
        "core_probe_ids": [
            probe.probe_id
            for probe in _panel_order(core, item.item_id, terminal_state, "core")
        ],
        "qualification_probe_ids": [
            probe.probe_id
            for probe in _panel_order(
                qualification, item.item_id, terminal_state, "qualification"
            )
        ],
        "unrelated_probe_ids": [
            probe.probe_id
            for probe in _panel_order(unrelated, item.item_id, terminal_state, "unrelated")
        ],
        "technical_duplicate": technical_duplicate,
        "duplicate_of": duplicate_of,
    }


def _audit_unit(unit: dict[str, Any], unit_by_id: dict[Any, dict[str, Any]]) -> None:
    pairs = unit.get("rollout_prompt_pairs")
    if not isinstance(pairs, list) or len(pairs) != 4:
        raise ValueError("each R-OPCD P1 unit requires four prompt pairs")
    if any(
        "<pathmem_current_state>" not in pair["privileged_teacher_prompt"]
        or "<pathmem_current_state>" in pair["student_prompt"]
        for pair in pairs
    ):
        raise ValueError("R-OPCD P1 privileged context boundary changed")
    if int(unit["training_seed"]) not in TRAINING_SEEDS:
        raise ValueError("R-OPCD P1 unit uses an unplanned training seed")
    if unit.get("root_adapter_seed") != _root_seed(
        str(unit["item_id"]), int(unit["training_seed"]), "p1r"
    ):
        raise ValueError("R-OPCD P1 root seed changed")
    parent_id = unit.get("parent_unit_id")
    if parent_id is not None:
        parent = unit_by_id.get(parent_id)
        if (
            parent is None
            or parent["item_id"] != unit["item_id"]
            or parent["training_seed"] != unit["training_seed"]
            or int(parent["sequence_index"]) >= int(unit["sequence_index"])
        ):
            raise ValueError("R-OPCD P1 parent lineage or ordering changed")


def _unit_id(prefix: str, item_id: str, training_seed: int, logical_name: str) -> str:
    return f"{prefix}:{item_id}:seed-{training_seed}:{logical_name}"


def _root_seed(item_id: str, training_seed: int, purpose: str) -> int:
    return int(
        json_hash(
            {
                "experiment_seed": EXPERIMENT_SEED,
                "item_id": item_id,
                "training_seed": training_seed,
                "operator": OPERATOR_ID,
                "purpose": purpose,
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
            unit_id
            for unit_id, unit in remaining.items()
            if unit["parent_unit_id"] is None or unit["parent_unit_id"] in emitted
        )
        if not ready:
            raise ValueError("R-OPCD P1 unit graph is cyclic")
        selected = ready[rng.randrange(len(ready))]
        unit = remaining.pop(selected)
        unit["sequence_index"] = len(ordered)
        ordered.append(unit)
        emitted.add(selected)
    return ordered


def _blocks_by_item(blocks: tuple[EventBlock, ...]) -> dict[str, tuple[EventBlock, ...]]:
    item_ids = {block.block_id.split(":", 1)[0] for block in blocks}
    return {
        item_id: tuple(
            sorted(
                (block for block in blocks if block.block_id.startswith(f"{item_id}:")),
                key=lambda block: block.block_id,
            )
        )
        for item_id in item_ids
    }


def _probes_by_item(probes: tuple[Probe, ...]) -> dict[str, tuple[Probe, ...]]:
    item_ids = {probe.item_id for probe in probes}
    return {
        item_id: tuple(
            sorted(
                (probe for probe in probes if probe.item_id == item_id),
                key=lambda probe: probe.probe_id,
            )
        )
        for item_id in item_ids
    }


def _handoff_identity(handoff: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "repair_id",
        "source_run_id",
        "source_plan_id",
        "source_matrix_sha256",
        "g2_summary_sha256",
        "g2_gate",
        "repair_implementation_sha256",
        "handoff_id",
    )
    return {key: handoff[key] for key in keys}


def _require_g2_handoff(handoff: dict[str, Any]) -> None:
    identity = {key: value for key, value in handoff.items() if key != "handoff_id"}
    if handoff.get("handoff_id") != json_hash(identity):
        raise ValueError("R-OPCD G2-to-P1 handoff identity changed")
    if (
        handoff.get("passed") is not True
        or handoff.get("g2_gate", {}).get("passed") is not True
        or handoff.get("p1_implementation_review_eligible") is not True
        or handoff.get("p1_authorized") is not False
        or handoff.get("kill_or_reserve_accessed") is not False
    ):
        raise PermissionError("R-OPCD P1 requires a passing, non-authorizing G2 handoff")
