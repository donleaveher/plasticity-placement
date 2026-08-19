from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from plasticity_placement.p0c.compiler import ACTIONS
from plasticity_placement.p0c.domain import CompiledLesson, P0CProbe
from plasticity_placement.p0d2hcrd.config import (
    CANDIDATES_PER_ENDPOINT,
    ENDPOINTS,
    ROWS_PER_LESSON,
)

DECOMPOSITION_BANK_VERSION = "p0d2hcrd-counterbalanced-bank-v1"
PROMPT_RENDERERS = {
    "route_only": "p0d2hcrd-route-only-v1",
    "retrieval_only": "p0d2hcrd-retrieval-only-v1",
    "combined": "p0d2hcrd-combined-v1",
}
SLOT_CANDIDATES = ("slot_a", "slot_b")
OUTPUT_PREFIX = "\nChoose exactly one allowed answer and output only that token from:"


@dataclass(frozen=True, slots=True)
class DecompositionProbe:
    probe_id: str
    lesson_id: str
    pair_id: str
    lesson_type: str
    lesson_side: str
    endpoint: str
    prompt: str
    expected_candidate: str
    ordered_candidates: tuple[str, ...]
    route_variant: int
    target_slot: str
    current_marker: str | None
    slot_candidate_order: int | None
    slot_content_order: int | None
    action_panel_position: int | None
    source_probe_id: str
    source_row_key: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_decomposition_bank(
    selected: tuple[CompiledLesson, ...],
    hard_bank: dict[str, tuple[P0CProbe, ...]],
) -> tuple[dict[str, tuple[DecompositionProbe, ...]], dict[str, Any]]:
    if len(selected) != 24:
        raise ValueError("decomposition bank requires exactly 24 lessons")
    bank: dict[str, tuple[DecompositionProbe, ...]] = {}
    for item in selected:
        lesson_id = item.lesson.lesson_id
        source_by_variant = _source_conditional_probes(hard_bank[lesson_id])
        probes: list[DecompositionProbe] = []
        for route_variant in range(1, 5):
            source_probe = source_by_variant[route_variant]
            for slot_candidate_order in range(1, 3):
                for slot_content_order in range(1, 3):
                    probes.append(
                        _route_only_probe(
                            item,
                            source_probe,
                            route_variant,
                            slot_candidate_order,
                            slot_content_order,
                        )
                    )
            for action_position in range(1, 5):
                probes.append(
                    _retrieval_only_probe(
                        item,
                        source_probe,
                        route_variant,
                        action_position,
                    )
                )
                for slot_content_order in range(1, 3):
                    probes.append(
                        _combined_probe(
                            item,
                            source_probe,
                            route_variant,
                            action_position,
                            slot_content_order,
                        )
                    )
        bank[lesson_id] = tuple(probes)
    audit = audit_decomposition_bank(selected, bank)
    return bank, audit


def audit_decomposition_bank(
    selected: tuple[CompiledLesson, ...],
    bank: dict[str, tuple[DecompositionProbe, ...]],
) -> dict[str, Any]:
    lesson_ids = tuple(item.lesson.lesson_id for item in selected)
    if tuple(bank) != lesson_ids:
        raise ValueError("decomposition lesson order changed")
    records = [
        probe.to_dict()
        for lesson_id in lesson_ids
        for probe in bank[lesson_id]
    ]
    probe_ids = [str(record["probe_id"]) for record in records]
    if len(probe_ids) != len(set(probe_ids)):
        raise ValueError("decomposition probe IDs are not unique")
    endpoint_counts = {
        endpoint: sum(record["endpoint"] == endpoint for record in records)
        for endpoint in ENDPOINTS
    }
    expected_endpoint_counts = {
        endpoint: len(lesson_ids) * ROWS_PER_LESSON[endpoint]
        for endpoint in ENDPOINTS
    }
    if endpoint_counts != expected_endpoint_counts:
        raise ValueError("decomposition endpoint matrix is incomplete")
    candidate_count = sum(len(record["ordered_candidates"]) for record in records)
    expected_candidate_count = sum(
        expected_endpoint_counts[endpoint] * CANDIDATES_PER_ENDPOINT[endpoint]
        for endpoint in ENDPOINTS
    )
    if candidate_count != expected_candidate_count:
        raise ValueError("decomposition candidate matrix is incomplete")

    action_lesson_counts = {
        action: sum(item.lesson.desired_action == action for item in selected)
        for action in ACTIONS
    }
    if set(action_lesson_counts.values()) != {6}:
        raise ValueError("selected lessons are not balanced by expected action")

    leakage_failures: list[str] = []
    balance_failures: list[str] = []
    selected_by_id = {item.lesson.lesson_id: item for item in selected}
    for lesson_id, probes in bank.items():
        item = selected_by_id[lesson_id]
        by_endpoint = {
            endpoint: [probe for probe in probes if probe.endpoint == endpoint]
            for endpoint in ENDPOINTS
        }
        if {
            endpoint: len(values) for endpoint, values in by_endpoint.items()
        } != ROWS_PER_LESSON:
            balance_failures.append(f"{lesson_id}: endpoint row counts")
        for probe in by_endpoint["route_only"]:
            forbidden = (
                *ACTIONS,
                item.external_note,
                item.lesson.context_id,
                item.lesson.condition,
            )
            if any(value and value in probe.prompt for value in forbidden):
                leakage_failures.append(probe.probe_id)
        for probe in by_endpoint["retrieval_only"]:
            if "Routing table:" in probe.prompt or "Current marker:" in probe.prompt:
                leakage_failures.append(probe.probe_id)
        for probe in by_endpoint["combined"]:
            if (
                probe.prompt.count("[LIVE]") != 1
                or probe.prompt.count("[ARCHIVED]") != 1
            ):
                leakage_failures.append(probe.probe_id)

        for route_variant in range(1, 5):
            route_values = [
                probe
                for probe in by_endpoint["route_only"]
                if probe.route_variant == route_variant
            ]
            if {
                probe.slot_candidate_order for probe in route_values
            } != {1, 2} or {
                probe.slot_content_order for probe in route_values
            } != {1, 2}:
                balance_failures.append(f"{lesson_id}: route crossing {route_variant}")
            combined_values = [
                probe
                for probe in by_endpoint["combined"]
                if probe.route_variant == route_variant
            ]
            for content_order in (1, 2):
                positions = {
                    probe.action_panel_position
                    for probe in combined_values
                    if probe.slot_content_order == content_order
                }
                if positions != {1, 2, 3, 4}:
                    balance_failures.append(
                        f"{lesson_id}: combined positions "
                        f"{route_variant}/{content_order}"
                    )
        if {
            probe.action_panel_position
            for probe in by_endpoint["retrieval_only"]
        } != {1, 2, 3, 4}:
            balance_failures.append(f"{lesson_id}: retrieval positions")

    prompt_hashes = [sha256(record["prompt"].encode()).hexdigest() for record in records]
    if len(prompt_hashes) != len(set(prompt_hashes)):
        raise ValueError("decomposition prompts are not unique")
    all_checks_passed = not leakage_failures and not balance_failures
    payload_without_records = {
        "schema_version": "p0d2hcrd-bank-audit-v1",
        "bank_version": DECOMPOSITION_BANK_VERSION,
        "lesson_count": len(lesson_ids),
        "decision_count": len(records),
        "candidate_count": candidate_count,
        "endpoint_counts": endpoint_counts,
        "action_lesson_counts": action_lesson_counts,
        "leakage_failure_count": len(leakage_failures),
        "balance_failure_count": len(balance_failures),
        "leakage_failures": leakage_failures,
        "balance_failures": balance_failures,
        "all_checks_passed": all_checks_passed,
    }
    bank_sha256 = _json_hash(records)
    return {
        **payload_without_records,
        "bank_sha256": bank_sha256,
        "records": records,
    }


def _route_only_probe(
    item: CompiledLesson,
    source_probe: P0CProbe,
    route_variant: int,
    slot_candidate_order: int,
    slot_content_order: int,
) -> DecompositionProbe:
    target_slot, current_marker, routing = _routing(item, route_variant)
    candidate_order = (
        SLOT_CANDIDATES
        if slot_candidate_order == 1
        else tuple(reversed(SLOT_CANDIDATES))
    )
    opaque = {
        "slot_a": f"opaque_{_digest(item.lesson.lesson_id, route_variant, 'a')}",
        "slot_b": f"opaque_{_digest(item.lesson.lesson_id, route_variant, 'b')}",
    }
    slot_lines = _ordered_slot_lines(
        {
            slot: f"{slot} contains opaque record {opaque[slot]}."
            for slot in SLOT_CANDIDATES
        },
        slot_content_order,
    )
    prompt = (
        f"{routing}\n"
        + "\n".join(slot_lines)
        + "\nDo not retrieve or infer an action. Which slot does the current "
        "marker select?"
        + _decision(candidate_order, label="Answer")
    )
    return _probe(
        item,
        source_probe,
        endpoint="route_only",
        prompt=prompt,
        expected_candidate=target_slot,
        ordered_candidates=candidate_order,
        route_variant=route_variant,
        target_slot=target_slot,
        current_marker=current_marker,
        slot_candidate_order=slot_candidate_order,
        slot_content_order=slot_content_order,
        action_panel_position=None,
        suffix=f"r{route_variant:02d}_co{slot_candidate_order}_so{slot_content_order}",
    )


def _retrieval_only_probe(
    item: CompiledLesson,
    source_probe: P0CProbe,
    record_variant: int,
    action_position: int,
) -> DecompositionProbe:
    target_slot = _target_slot(record_variant)
    action_order = _action_order(
        item.lesson.desired_action,
        action_position,
        item.lesson.lesson_id,
        record_variant,
    )
    receipt = _digest(item.lesson.lesson_id, record_variant, "receipt")
    prompt = (
        "The routing stage has already been completed and independently "
        f"verified. Selection receipt {receipt} identifies {target_slot}. "
        "Do not perform a routing decision.\n"
        f"Selected live record: workspace {item.lesson.context_id}; condition: "
        f"{item.lesson.condition}.\n\n"
        "<verified_memory>\n"
        f"{item.external_note}\n"
        "</verified_memory>\n"
        "Retrieve the action for the supplied live record."
        + _decision(action_order, label="Action")
    )
    return _probe(
        item,
        source_probe,
        endpoint="retrieval_only",
        prompt=prompt,
        expected_candidate=item.lesson.desired_action,
        ordered_candidates=action_order,
        route_variant=record_variant,
        target_slot=target_slot,
        current_marker=None,
        slot_candidate_order=None,
        slot_content_order=None,
        action_panel_position=action_position,
        suffix=f"r{record_variant:02d}_ap{action_position}",
    )


def _combined_probe(
    item: CompiledLesson,
    source_probe: P0CProbe,
    route_variant: int,
    action_position: int,
    slot_content_order: int,
) -> DecompositionProbe:
    target_slot, current_marker, routing = _routing(item, route_variant)
    decoy_slot = "slot_b" if target_slot == "slot_a" else "slot_a"
    action_order = _action_order(
        item.lesson.desired_action,
        action_position,
        item.lesson.lesson_id,
        route_variant,
    )
    distractors = _rotated_distractors(item, route_variant)
    slots = {
        target_slot: (
            f"{target_slot} [LIVE] contains workspace {item.lesson.context_id}; "
            f"condition: {item.lesson.condition}; retrieve its verified mapping."
        ),
        decoy_slot: (
            f"{decoy_slot} [ARCHIVED] contains unrelated workspace "
            f"archive_{_digest(item.lesson.lesson_id, route_variant, 'archive')}; "
            f"obsolete labels mention {', '.join(distractors)}; never execute it."
        ),
    }
    prompt = (
        f"{routing}\n"
        + "\n".join(_ordered_slot_lines(slots, slot_content_order))
        + "\n\n<verified_memory>\n"
        f"{item.external_note}\n"
        "</verified_memory>\n"
        "First select the slot from the routing table. Then retrieve the action "
        "for the selected live record and ignore the archived slot."
        + _decision(action_order, label="Action")
    )
    return _probe(
        item,
        source_probe,
        endpoint="combined",
        prompt=prompt,
        expected_candidate=item.lesson.desired_action,
        ordered_candidates=action_order,
        route_variant=route_variant,
        target_slot=target_slot,
        current_marker=current_marker,
        slot_candidate_order=None,
        slot_content_order=slot_content_order,
        action_panel_position=action_position,
        suffix=f"r{route_variant:02d}_ap{action_position}_so{slot_content_order}",
    )


def _probe(
    item: CompiledLesson,
    source_probe: P0CProbe,
    *,
    endpoint: str,
    prompt: str,
    expected_candidate: str,
    ordered_candidates: tuple[str, ...],
    route_variant: int,
    target_slot: str,
    current_marker: str | None,
    slot_candidate_order: int | None,
    slot_content_order: int | None,
    action_panel_position: int | None,
    suffix: str,
) -> DecompositionProbe:
    lesson = item.lesson
    return DecompositionProbe(
        probe_id=f"{lesson.lesson_id}_crd_{endpoint}_{suffix}",
        lesson_id=lesson.lesson_id,
        pair_id=lesson.pair_id,
        lesson_type=lesson.lesson_type,
        lesson_side=lesson.lesson_id.rsplit("_", maxsplit=1)[-1],
        endpoint=endpoint,
        prompt=prompt,
        expected_candidate=expected_candidate,
        ordered_candidates=ordered_candidates,
        route_variant=route_variant,
        target_slot=target_slot,
        current_marker=current_marker,
        slot_candidate_order=slot_candidate_order,
        slot_content_order=slot_content_order,
        action_panel_position=action_panel_position,
        source_probe_id=source_probe.probe_id,
        source_row_key=(
            f"scale_canary::{lesson.lesson_id}::external::{source_probe.probe_id}"
        ),
    )


def _routing(
    item: CompiledLesson,
    route_variant: int,
) -> tuple[str, str, str]:
    marker_a = f"marker_{_digest(item.lesson.lesson_id, route_variant, 'ma')}"
    marker_b = f"marker_{_digest(item.lesson.lesson_id, route_variant, 'mb')}"
    target_slot = _target_slot(route_variant)
    current_marker = marker_a if target_slot == "slot_a" else marker_b
    routing = (
        f"Routing table: {marker_a} -> slot_a; {marker_b} -> slot_b. "
        f"Current marker: {current_marker}. Apply the table exactly once."
    )
    return target_slot, current_marker, routing


def _target_slot(route_variant: int) -> str:
    if route_variant not in {1, 2, 3, 4}:
        raise ValueError(f"unsupported route variant: {route_variant}")
    return "slot_a" if route_variant % 2 else "slot_b"


def _action_order(
    expected: str,
    expected_position: int,
    lesson_id: str,
    variant: int,
) -> tuple[str, ...]:
    if expected not in ACTIONS or expected_position not in {1, 2, 3, 4}:
        raise ValueError("invalid action-panel request")
    distractors = [action for action in ACTIONS if action != expected]
    offset = int(_digest(lesson_id, variant, "panel"), 16) % len(distractors)
    distractors = distractors[offset:] + distractors[:offset]
    result = distractors.copy()
    result.insert(expected_position - 1, expected)
    return tuple(result)


def _ordered_slot_lines(
    slots: dict[str, str],
    slot_content_order: int,
) -> tuple[str, str]:
    if slot_content_order == 1:
        order = SLOT_CANDIDATES
    elif slot_content_order == 2:
        order = tuple(reversed(SLOT_CANDIDATES))
    else:
        raise ValueError(f"unsupported slot content order: {slot_content_order}")
    return tuple(slots[slot] for slot in order)


def _decision(candidates: tuple[str, ...], *, label: str) -> str:
    return f"{OUTPUT_PREFIX} {', '.join(candidates)}.\n{label}:"


def _source_conditional_probes(
    probes: tuple[P0CProbe, ...],
) -> dict[int, P0CProbe]:
    selected = [
        probe for probe in probes if probe.category == "conditional_route"
    ]
    if len(selected) != 4:
        raise ValueError("source lesson requires four conditional-route probes")
    result: dict[int, P0CProbe] = {}
    for probe in selected:
        try:
            variant = int(probe.probe_id.rsplit("_", maxsplit=1)[-1])
        except ValueError as error:
            raise ValueError("source route probe has no numeric variant") from error
        result[variant] = probe
    if set(result) != {1, 2, 3, 4}:
        raise ValueError("source conditional-route variants changed")
    return result


def _rotated_distractors(
    item: CompiledLesson,
    route_variant: int,
) -> tuple[str, ...]:
    values = item.lesson.distractor_actions
    offset = (route_variant - 1) % len(values)
    return values[offset:] + values[:offset]


def _digest(*parts: object) -> str:
    return sha256(
        (DECOMPOSITION_BANK_VERSION + ":" + ":".join(map(str, parts))).encode()
    ).hexdigest()[:10]


def _json_hash(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
