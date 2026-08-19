from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from plasticity_placement.pathmem.backends import render_fresh_session_probe
from plasticity_placement.pathmem.compiler import compile_bank
from plasticity_placement.pathmem.generator import ACTIONS, canonical_external_note
from plasticity_placement.pathmem.io import json_hash
from plasticity_placement.pathmem.schema import AccessMode, Probe, ProbeCategory, SemanticItem
from plasticity_placement.pathmem_consolidation.config import (
    EXECUTION_ORDER_SEED,
    G1C_GATE_SCHEMA_VERSION,
    OPERATOR_ID,
    QUALIFICATION_SPLIT,
    RETENTION_HORIZON,
    build_g0v2_contract,
)

G1C_PLAN_SCHEMA_VERSION = "pathmem-g1c-consolidation-plan-v1"


@dataclass(frozen=True, slots=True)
class RolloutPromptPair:
    probe_id: str
    student_prompt: str
    privileged_teacher_prompt: str
    action_choices: tuple[str, str, str, str]
    current_action: str
    obsolete_action: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action_choices"] = list(self.action_choices)
        return payload


@dataclass(frozen=True, slots=True)
class ConsolidationUnit:
    unit_id: str
    sequence_index: int
    item_id: str
    content_domain: str
    terminal_state: str
    memory_key: str
    current_action: str
    obsolete_action: str
    action_choices: tuple[str, str, str, str]
    side_memory_namespace: str
    route_key: str
    canonical_teacher_context: str
    rollout_prompt_pairs: tuple[RolloutPromptPair, ...]
    immediate_probe_ids: tuple[str, ...]
    core_probe_ids: tuple[str, ...]
    near_neighbor_probe_ids: tuple[str, ...]
    unrelated_probe_ids: tuple[str, ...]
    retention_distractor_unit_ids: tuple[str, ...]
    wrong_swap_unit_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "sequence_index": self.sequence_index,
            "item_id": self.item_id,
            "content_domain": self.content_domain,
            "terminal_state": self.terminal_state,
            "memory_key": self.memory_key,
            "current_action": self.current_action,
            "obsolete_action": self.obsolete_action,
            "action_choices": list(self.action_choices),
            "side_memory_namespace": self.side_memory_namespace,
            "route_key": self.route_key,
            "canonical_teacher_context": self.canonical_teacher_context,
            "rollout_prompt_pairs": [pair.to_dict() for pair in self.rollout_prompt_pairs],
            "immediate_probe_ids": list(self.immediate_probe_ids),
            "core_probe_ids": list(self.core_probe_ids),
            "near_neighbor_probe_ids": list(self.near_neighbor_probe_ids),
            "unrelated_probe_ids": list(self.unrelated_probe_ids),
            "retention_distractor_unit_ids": list(self.retention_distractor_unit_ids),
            "wrong_swap_unit_id": self.wrong_swap_unit_id,
        }


@dataclass(frozen=True, slots=True)
class _UnitSeed:
    item: SemanticItem
    terminal_state: str
    unit_id: str
    current_action: str
    obsolete_action: str
    teacher_context: str
    rollout_pairs: tuple[RolloutPromptPair, ...]
    immediate_probe_ids: tuple[str, ...]
    core_probe_ids: tuple[str, ...]
    near_neighbor_probe_ids: tuple[str, ...]
    unrelated_probe_ids: tuple[str, ...]


def compile_g1c_plan(
    *,
    parent_manifest_id: str,
    parent_manifest_sha256: str,
) -> dict[str, Any]:
    contract = build_g0v2_contract(
        parent_manifest_id=parent_manifest_id,
        parent_manifest_sha256=parent_manifest_sha256,
    )
    bank = compile_bank()
    items = tuple(
        sorted(
            (item for item in bank.items if item.split == QUALIFICATION_SPLIT),
            key=lambda item: item.item_id,
        )
    )
    if len(items) != 12:
        raise ValueError(f"G1-C requires 12 interface_dev items, found {len(items)}")

    probes_by_item: dict[str, list[Probe]] = {item.item_id: [] for item in items}
    for probe in bank.probes:
        if probe.item_id in probes_by_item:
            probes_by_item[probe.item_id].append(probe)

    seeds = tuple(
        _build_unit_seed(item, terminal_state, probes_by_item[item.item_id])
        for item in items
        for terminal_state in ("A", "B")
    )
    ordered = tuple(
        sorted(
            seeds,
            key=lambda seed: json_hash(
                {"execution_order_seed": EXECUTION_ORDER_SEED, "unit_id": seed.unit_id}
            ),
        )
    )
    units = tuple(
        _finalize_unit(seed, sequence_index, ordered)
        for sequence_index, seed in enumerate(ordered)
    )
    counts = {
        "items": len(items),
        "consolidation_units": len(units),
        "teacher_student_prompt_pairs": sum(len(unit.rollout_prompt_pairs) for unit in units),
        "immediate_probe_assignments": sum(len(unit.immediate_probe_ids) for unit in units),
        "core_probe_assignments": sum(len(unit.core_probe_ids) for unit in units),
        "near_neighbor_probe_assignments": sum(
            len(unit.near_neighbor_probe_ids) for unit in units
        ),
        "unrelated_probe_assignments": sum(len(unit.unrelated_probe_ids) for unit in units),
        "retention_anchor_units": len(units),
        "retention_distractor_applications": sum(
            len(unit.retention_distractor_unit_ids) for unit in units
        ),
        "wrong_swap_controls": len(units),
    }
    identity = {
        "schema_version": G1C_PLAN_SCHEMA_VERSION,
        "gate_schema_version": G1C_GATE_SCHEMA_VERSION,
        "operator_id": OPERATOR_ID,
        "contract_sha256": json_hash(contract),
        "parent_g0_v1": contract["parent_g0_v1"],
        "scope": {
            "split": QUALIFICATION_SPLIT,
            "included_splits": [QUALIFICATION_SPLIT],
            "excluded_splits": [
                "smoke",
                "kill",
                "confirmatory",
                "hardware_dev",
                "reserve",
            ],
            "path_contrast": False,
            "retention_horizon_subsequent_consolidations": RETENTION_HORIZON,
            "retention_execution": {
                "parent_state": "fresh cloned root per anchor",
                "sequence": [
                    "consolidate anchor",
                    "score immediate panel",
                    "apply 12 listed cross-key consolidations",
                    "rescore retained anchor",
                ],
            },
        },
        "execution_order_seed": EXECUTION_ORDER_SEED,
        "counts": counts,
        "action_unit_counts": dict(sorted(Counter(unit.current_action for unit in units).items())),
        "authorization": contract["authorization"],
        "units": [unit.to_dict() for unit in units],
    }
    plan = {**identity, "plan_id": json_hash(identity)}
    audit_g1c_plan(plan)
    return plan


def audit_g1c_plan(plan: dict[str, Any]) -> dict[str, Any]:
    identity = {key: value for key, value in plan.items() if key != "plan_id"}
    if plan.get("schema_version") != G1C_PLAN_SCHEMA_VERSION:
        raise ValueError("G1-C plan schema changed")
    if plan.get("plan_id") != json_hash(identity):
        raise ValueError("G1-C plan identity mismatch")
    scope = plan.get("scope")
    if not isinstance(scope, dict) or scope.get("included_splits") != [QUALIFICATION_SPLIT]:
        raise ValueError("G1-C plan escaped interface_dev")
    if scope.get("path_contrast") is not False:
        raise ValueError("G1-C plan cannot authorize a path contrast")
    authorization = plan.get("authorization")
    if not isinstance(authorization, dict) or any(
        authorization.get(field) is not False
        for field in (
            "training_authorized",
            "gpu_inference_authorized",
            "g1c_execution_authorized",
            "p0_authorized",
            "path_contrast_authorized",
            "kill_or_reserve_access_authorized",
            "rl_controller_authorized",
        )
    ):
        raise ValueError("G1-C CPU plan contains unauthorized execution state")

    units = plan.get("units")
    if not isinstance(units, list) or len(units) != 24:
        raise ValueError("G1-C plan requires 24 consolidation units")
    unit_by_id = {unit.get("unit_id"): unit for unit in units if isinstance(unit, dict)}
    if len(unit_by_id) != len(units) or None in unit_by_id:
        raise ValueError("G1-C unit identities must be complete and unique")
    for unit in units:
        prompt_pairs = unit.get("rollout_prompt_pairs")
        if not isinstance(prompt_pairs, list) or len(prompt_pairs) != 4:
            raise ValueError("each G1-C unit requires four rollout prompt pairs")
        for pair in prompt_pairs:
            teacher = pair.get("privileged_teacher_prompt", "")
            student = pair.get("student_prompt", "")
            if "<pathmem_current_state>" not in teacher:
                raise ValueError("G1-C teacher prompt is missing privileged current state")
            if "<pathmem_current_state>" in student:
                raise ValueError("G1-C student prompt leaked privileged current state")
        distractors = unit.get("retention_distractor_unit_ids")
        if not isinstance(distractors, list) or len(distractors) != RETENTION_HORIZON:
            raise ValueError("G1-C retention schedule length changed")
        for distractor_id in distractors:
            distractor = unit_by_id.get(distractor_id)
            if distractor is None or distractor["item_id"] == unit["item_id"]:
                raise ValueError("G1-C retention distractors must use another memory key")
        swap = unit_by_id.get(unit.get("wrong_swap_unit_id"))
        if (
            swap is None
            or swap["item_id"] == unit["item_id"]
            or swap["current_action"] == unit["current_action"]
        ):
            raise ValueError("G1-C wrong-swap control is not behaviorally discriminating")

    expected_action_counts = {action: 6 for action in ACTIONS}
    if plan.get("action_unit_counts") != expected_action_counts:
        raise ValueError("G1-C current-action coverage is not balanced")
    expected_counts = {
        "items": 12,
        "consolidation_units": 24,
        "teacher_student_prompt_pairs": 96,
        "immediate_probe_assignments": 96,
        "core_probe_assignments": 336,
        "near_neighbor_probe_assignments": 96,
        "unrelated_probe_assignments": 192,
        "retention_anchor_units": 24,
        "retention_distractor_applications": 288,
        "wrong_swap_controls": 24,
    }
    if plan.get("counts") != expected_counts:
        raise ValueError(f"G1-C plan counts changed: {plan.get('counts')}")
    return {
        "passed": True,
        "plan_id": plan["plan_id"],
        "operator_id": OPERATOR_ID,
        "split": QUALIFICATION_SPLIT,
        "counts": expected_counts,
        "action_unit_counts": expected_action_counts,
        "training_started": False,
        "gpu_inference_started": False,
        "path_contrast_computed": False,
    }


def _build_unit_seed(
    item: SemanticItem,
    terminal_state: str,
    item_probes: list[Probe],
) -> _UnitSeed:
    current = item.state_a_action if terminal_state == "A" else item.state_b_action
    obsolete = item.state_b_action if terminal_state == "A" else item.state_a_action
    terminal_probes = tuple(
        sorted(
            (probe for probe in item_probes if probe.terminal_state == terminal_state),
            key=lambda probe: probe.probe_id,
        )
    )
    qualification = _category(terminal_probes, ProbeCategory.QUALIFICATION)
    core = tuple(probe for probe in terminal_probes if probe.core_endpoint)
    near = _category(terminal_probes, ProbeCategory.NEAR_NEIGHBOR)
    unrelated = _category(terminal_probes, ProbeCategory.UNRELATED)
    if (len(qualification), len(core), len(near), len(unrelated)) != (4, 14, 4, 8):
        raise ValueError(f"G1-C probe panel changed for {item.item_id}/{terminal_state}")
    teacher_context = canonical_external_note(item, terminal_state)
    pairs = tuple(
        RolloutPromptPair(
            probe_id=probe.probe_id,
            student_prompt=render_fresh_session_probe(
                probe,
                persistent_state="unused",
                access_mode=AccessMode.OFF,
            ),
            privileged_teacher_prompt=render_fresh_session_probe(
                probe,
                persistent_state=teacher_context,
                access_mode=AccessMode.ON,
            ),
            action_choices=probe.action_choices,
            current_action=current,
            obsolete_action=obsolete,
        )
        for probe in qualification
    )
    return _UnitSeed(
        item=item,
        terminal_state=terminal_state,
        unit_id=f"g1c:{item.item_id}:{terminal_state}",
        current_action=current,
        obsolete_action=obsolete,
        teacher_context=teacher_context,
        rollout_pairs=pairs,
        immediate_probe_ids=tuple(probe.probe_id for probe in qualification),
        core_probe_ids=tuple(probe.probe_id for probe in core),
        near_neighbor_probe_ids=tuple(probe.probe_id for probe in near),
        unrelated_probe_ids=tuple(probe.probe_id for probe in unrelated),
    )


def _finalize_unit(
    seed: _UnitSeed,
    sequence_index: int,
    ordered: tuple[_UnitSeed, ...],
) -> ConsolidationUnit:
    following = (*ordered[sequence_index + 1 :], *ordered[:sequence_index])
    other_items = tuple(
        candidate for candidate in following if candidate.item.item_id != seed.item.item_id
    )
    distractors = other_items[:RETENTION_HORIZON]
    if len(distractors) != RETENTION_HORIZON:
        raise ValueError("not enough cross-key G1-C retention distractors")
    wrong_swap = next(
        candidate for candidate in other_items if candidate.current_action != seed.current_action
    )
    return ConsolidationUnit(
        unit_id=seed.unit_id,
        sequence_index=sequence_index,
        item_id=seed.item.item_id,
        content_domain=seed.item.content_domain.value,
        terminal_state=seed.terminal_state,
        memory_key=seed.item.memory_key,
        current_action=seed.current_action,
        obsolete_action=seed.obsolete_action,
        action_choices=seed.item.action_choices,
        side_memory_namespace=f"pathmem/g1c/{seed.item.item_id}/{seed.terminal_state}",
        route_key=seed.item.memory_key,
        canonical_teacher_context=seed.teacher_context,
        rollout_prompt_pairs=seed.rollout_pairs,
        immediate_probe_ids=seed.immediate_probe_ids,
        core_probe_ids=seed.core_probe_ids,
        near_neighbor_probe_ids=seed.near_neighbor_probe_ids,
        unrelated_probe_ids=seed.unrelated_probe_ids,
        retention_distractor_unit_ids=tuple(candidate.unit_id for candidate in distractors),
        wrong_swap_unit_id=wrong_swap.unit_id,
    )


def _category(probes: tuple[Probe, ...], category: ProbeCategory) -> tuple[Probe, ...]:
    return tuple(probe for probe in probes if probe.category is category)
