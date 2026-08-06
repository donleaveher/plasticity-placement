from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from plasticity_placement.p0d2hrab.probes import ACTION_INSTRUCTION, BindingProbe
from plasticity_placement.p0d2hrabc.config import (
    BANK_VERSION,
    CANDIDATE_ROTATIONS,
    DISPLAY_ORDERS,
    ORIENTATIONS,
    RECEIPTS,
    AuditSpec,
)
from plasticity_placement.p0d2hrsh.probes import RECEIPT_PREFIX


@dataclass(frozen=True, slots=True)
class CounterbalancedProbe:
    probe_id: str
    source_probe_id: str
    unit_id: str
    pair_id: str
    route_variant: int
    orientation: str
    receipt: str
    cell: str
    display_order: str
    selected_display_position: int
    candidate_rotation: int
    expected_candidate_position: int
    lesson_a_id: str
    lesson_b_id: str
    slot_a_lesson_id: str
    slot_b_lesson_id: str
    slot_a_action: str
    slot_b_action: str
    expected_action: str
    counterfactual_action: str
    ordered_candidates: tuple[str, ...]
    prompt: str
    source_prompt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_counterbalanced_bank(
    source_probes: list[BindingProbe], spec: AuditSpec
) -> tuple[list[CounterbalancedProbe], dict[str, Any]]:
    if len(source_probes) != spec.source_prompt_count:
        raise ValueError("counterbalancing requires the exact 192-prompt RAB bank")
    probes: list[CounterbalancedProbe] = []
    for source in sorted(source_probes, key=lambda probe: probe.probe_id):
        slot_payloads = _slot_payloads(source.prompt)
        memory = _memory_block(source.prompt)
        for display_order in DISPLAY_ORDERS:
            for rotation in CANDIDATE_ROTATIONS:
                candidates = _rotate(source.ordered_candidates, rotation)
                prompt = _counterbalanced_prompt(
                    receipt=source.receipt,
                    slot_payloads=slot_payloads,
                    display_order=display_order,
                    memory=memory,
                    candidates=candidates,
                )
                selected_label = source.receipt
                probes.append(
                    CounterbalancedProbe(
                        probe_id=(f"{source.probe_id}_display_{display_order}_rotation_{rotation}"),
                        source_probe_id=source.probe_id,
                        unit_id=source.unit_id,
                        pair_id=source.pair_id,
                        route_variant=source.route_variant,
                        orientation=source.orientation,
                        receipt=source.receipt,
                        cell=source.cell,
                        display_order=display_order,
                        selected_display_position=display_order.index(selected_label),
                        candidate_rotation=rotation,
                        expected_candidate_position=candidates.index(source.expected_action),
                        lesson_a_id=source.lesson_a_id,
                        lesson_b_id=source.lesson_b_id,
                        slot_a_lesson_id=source.slot_a_lesson_id,
                        slot_b_lesson_id=source.slot_b_lesson_id,
                        slot_a_action=source.slot_a_action,
                        slot_b_action=source.slot_b_action,
                        expected_action=source.expected_action,
                        counterfactual_action=source.counterfactual_action,
                        ordered_candidates=candidates,
                        prompt=prompt,
                        source_prompt_sha256=_digest_text(source.prompt),
                    )
                )
    audit = audit_counterbalanced_bank(probes, source_probes, spec)
    return probes, audit


def audit_counterbalanced_bank(
    probes: list[CounterbalancedProbe],
    source_probes: list[BindingProbe],
    spec: AuditSpec,
) -> dict[str, Any]:
    by_source: dict[str, list[CounterbalancedProbe]] = {}
    by_unit: dict[str, list[CounterbalancedProbe]] = {}
    for probe in probes:
        by_source.setdefault(probe.source_probe_id, []).append(probe)
        by_unit.setdefault(probe.unit_id, []).append(probe)
    source_index = {probe.probe_id: probe for probe in source_probes}
    joint_counts = Counter(
        (probe.receipt, probe.selected_display_position, probe.expected_candidate_position)
        for probe in probes
    )
    prompt_multiplicities = Counter(_digest_text(probe.prompt) for probe in probes)
    expected_unit_factorial = {
        (orientation, receipt, display_order, rotation)
        for orientation in ORIENTATIONS
        for receipt in RECEIPTS
        for display_order in DISPLAY_ORDERS
        for rotation in CANDIDATE_ROTATIONS
    }
    checks = {
        "prompt_count": len(probes) == spec.prompt_count,
        "source_prompt_count": len(by_source) == spec.source_prompt_count,
        "unit_count": len(by_unit) == spec.unit_count,
        "unique_probe_ids": len({probe.probe_id for probe in probes}) == len(probes),
        "expected_textual_aliasing": len(prompt_multiplicities) == 384
        and set(prompt_multiplicities.values()) == {4},
        "eight_variants_per_source": all(len(rows) == 8 for rows in by_source.values()),
        "complete_unit_factorial": all(
            {
                (row.orientation, row.receipt, row.display_order, row.candidate_rotation)
                for row in rows
            }
            == expected_unit_factorial
            for rows in by_unit.values()
        ),
        "source_fields_preserved": all(
            _source_fields_match(row, source_index[row.source_probe_id]) for row in probes
        ),
        "candidate_sets_preserved": all(
            set(row.ordered_candidates) == set(source_index[row.source_probe_id].ordered_candidates)
            for row in probes
        ),
        "rotation_is_cyclic": all(
            row.ordered_candidates
            == _rotate(source_index[row.source_probe_id].ordered_candidates, row.candidate_rotation)
            for row in probes
        ),
        "ab_rotation_zero_reproduces_source": all(
            _at(rows, "AB", 0).prompt == source_index[source_id].prompt
            for source_id, rows in by_source.items()
        ),
        "display_order_only_reorders_slot_lines": all(
            _normalize_display(_at(rows, "AB", rotation).prompt)
            == _normalize_display(_at(rows, "BA", rotation).prompt)
            for rows in by_source.values()
            for rotation in CANDIDATE_ROTATIONS
        ),
        "selected_display_position_balanced": Counter(
            row.selected_display_position for row in probes
        )
        == {0: 768, 1: 768},
        "candidate_position_balanced": Counter(row.expected_candidate_position for row in probes)
        == {0: 384, 1: 384, 2: 384, 3: 384},
        "joint_confounders_crossed": all(count == 96 for count in joint_counts.values())
        and len(joint_counts) == 16,
    }
    payload = {
        "schema_version": "p0d2hrabc-bank-audit-v1",
        "bank_version": BANK_VERSION,
        "spec": spec.to_dict(),
        "factor_counts": {
            "orientation": dict(sorted(Counter(row.orientation for row in probes).items())),
            "receipt": dict(sorted(Counter(row.receipt for row in probes).items())),
            "display_order": dict(sorted(Counter(row.display_order for row in probes).items())),
            "selected_display_position": dict(
                sorted(Counter(row.selected_display_position for row in probes).items())
            ),
            "candidate_rotation": dict(
                sorted(Counter(row.candidate_rotation for row in probes).items())
            ),
            "expected_candidate_position": dict(
                sorted(Counter(row.expected_candidate_position for row in probes).items())
            ),
        },
        "prompt_text_count": len(prompt_multiplicities),
        "prompt_condition_count": len(probes),
        "prompt_text_multiplicity_counts": dict(
            sorted(Counter(prompt_multiplicities.values()).items())
        ),
        "prompt_aliasing_note": (
            "The four frozen RAB route variants already cycle the candidate panel. Crossing "
            "each variant with four new rotations yields 1536 condition IDs but 384 distinct "
            "prompt texts, each scored four times by the requested design."
        ),
        "joint_receipt_display_candidate_counts": {
            f"receipt_{receipt}_display_{position}_candidate_{candidate}": count
            for (receipt, position, candidate), count in sorted(joint_counts.items())
        },
        "checks": checks,
        "records_sha256": _json_hash([probe.to_dict() for probe in probes]),
        "all_checks_passed": all(checks.values()),
    }
    if payload["all_checks_passed"] is not True:
        raise ValueError(f"RAB counterbalanced bank audit failed: {checks}")
    return payload


def _counterbalanced_prompt(
    *,
    receipt: str,
    slot_payloads: dict[str, str],
    display_order: str,
    memory: str,
    candidates: tuple[str, ...],
) -> str:
    slots = " ".join(f"Slot {label} contains {slot_payloads[label]}" for label in display_order)
    return (
        f"{RECEIPT_PREFIX}{receipt}. {slots} {ACTION_INSTRUCTION}\n\n"
        f"<verified_memory>\n{memory}\n</verified_memory>\n"
        f"Only output one action token from: {', '.join(candidates)}.\nAction:"
    )


def _slot_payloads(prompt: str) -> dict[str, str]:
    start_a = prompt.index("Slot A contains ")
    start_b = prompt.index("Slot B contains ")
    instruction = prompt.index(f" {ACTION_INSTRUCTION}")
    if start_a > start_b:
        raise ValueError("frozen RAB source is not in AB display order")
    return {
        "A": prompt[start_a + len("Slot A contains ") : start_b].strip(),
        "B": prompt[start_b + len("Slot B contains ") : instruction].strip(),
    }


def _memory_block(prompt: str) -> str:
    start = prompt.index("<verified_memory>") + len("<verified_memory>")
    end = prompt.index("</verified_memory>")
    value = prompt[start:end].strip()
    if not value:
        raise ValueError("frozen RAB prompt has an empty verified-memory block")
    return value


def _rotate(values: tuple[str, ...], rotation: int) -> tuple[str, ...]:
    if len(values) != 4 or rotation not in CANDIDATE_ROTATIONS:
        raise ValueError("counterbalancing requires four candidates and rotations 0..3")
    return values[rotation:] + values[:rotation]


def _at(
    rows: list[CounterbalancedProbe], display_order: str, rotation: int
) -> CounterbalancedProbe:
    matches = [
        row
        for row in rows
        if row.display_order == display_order and row.candidate_rotation == rotation
    ]
    if len(matches) != 1:
        raise ValueError("counterbalanced source cell is not unique")
    return matches[0]


def _normalize_display(prompt: str) -> str:
    payloads = _slot_payloads_any_order(prompt)
    first = min(prompt.index("Slot A contains "), prompt.index("Slot B contains "))
    instruction = prompt.index(f" {ACTION_INSTRUCTION}")
    canonical_slots = f"Slot A contains {payloads['A']} Slot B contains {payloads['B']}"
    return prompt[:first] + canonical_slots + prompt[instruction:]


def _slot_payloads_any_order(prompt: str) -> dict[str, str]:
    instruction = prompt.index(f" {ACTION_INSTRUCTION}")
    starts = sorted(
        ((prompt.index(f"Slot {label} contains "), label) for label in ("A", "B")),
        key=lambda item: item[0],
    )
    result: dict[str, str] = {}
    for index, (start, label) in enumerate(starts):
        stop = starts[index + 1][0] if index + 1 < len(starts) else instruction
        result[label] = prompt[start + len(f"Slot {label} contains ") : stop].strip()
    return result


def _source_fields_match(row: CounterbalancedProbe, source: BindingProbe) -> bool:
    return all(
        getattr(row, field) == getattr(source, field)
        for field in (
            "unit_id",
            "pair_id",
            "route_variant",
            "orientation",
            "receipt",
            "cell",
            "lesson_a_id",
            "lesson_b_id",
            "slot_a_lesson_id",
            "slot_b_lesson_id",
            "slot_a_action",
            "slot_b_action",
            "expected_action",
            "counterfactual_action",
        )
    )


def _digest_text(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _json_hash(value: object) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
