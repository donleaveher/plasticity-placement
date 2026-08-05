from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from plasticity_placement.p0d2hrab.config import BANK_VERSION, CELLS, AuditSpec
from plasticity_placement.p0d2hrsh.probes import (
    RECEIPT_PREFIX,
    HandoffProbe,
    _receipt_prompt,
)

ACTION_INSTRUCTION = (
    "The routing receipt is authoritative; return the action for the selected slot."
)


@dataclass(frozen=True, slots=True)
class BindingProbe:
    probe_id: str
    unit_id: str
    pair_id: str
    route_variant: int
    orientation: str
    receipt: str
    cell: str
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
    source_a_probe_id: str
    source_b_probe_id: str
    source_a_prompt_sha256: str
    source_b_prompt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_binding_bank(
    handoff_probes: list[HandoffProbe], spec: AuditSpec
) -> tuple[list[BindingProbe], dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, HandoffProbe]] = {}
    for probe in handoff_probes:
        grouped.setdefault((probe.pair_id, probe.route_variant), {})[probe.lesson_side] = probe
    if len(grouped) != spec.unit_count:
        raise ValueError("receipt/action binding source units are incomplete")
    probes: list[BindingProbe] = []
    for (pair_id, variant), sides in sorted(grouped.items()):
        if set(sides) != {"a", "b"}:
            raise ValueError(f"receipt/action pair sides are incomplete: {pair_id}/r{variant}")
        lesson_a, lesson_b = sides["a"], sides["b"]
        if lesson_a.target_action == lesson_b.target_action or set(
            lesson_a.action_candidates
        ) != set(lesson_b.action_candidates):
            raise ValueError(f"receipt/action pair does not define two valid actions: {pair_id}")
        payloads = {
            "a": _valid_query_payload(lesson_a),
            "b": _valid_query_payload(lesson_b),
        }
        memories = {
            "a": _verified_memory(lesson_a.receipt_a_prompt),
            "b": _verified_memory(lesson_b.receipt_a_prompt),
        }
        candidate_order = lesson_a.action_candidates
        unit_id = f"{pair_id}_rab_r{variant:02d}"
        for orientation in ("canonical", "swapped"):
            assignments = ("a", "b") if orientation == "canonical" else ("b", "a")
            for receipt in ("A", "B"):
                selected_side = assignments[0 if receipt == "A" else 1]
                other_side = assignments[1 if receipt == "A" else 0]
                prompt = _binding_prompt(
                    receipt=receipt,
                    slot_a_payload=payloads[assignments[0]],
                    slot_b_payload=payloads[assignments[1]],
                    memory_a=memories["a"],
                    memory_b=memories["b"],
                    candidates=candidate_order,
                )
                probes.append(
                    BindingProbe(
                        probe_id=f"{unit_id}_{orientation}_{receipt}",
                        unit_id=unit_id,
                        pair_id=pair_id,
                        route_variant=variant,
                        orientation=orientation,
                        receipt=receipt,
                        cell=f"{orientation}_{receipt}",
                        lesson_a_id=lesson_a.lesson_id,
                        lesson_b_id=lesson_b.lesson_id,
                        slot_a_lesson_id=sides[assignments[0]].lesson_id,
                        slot_b_lesson_id=sides[assignments[1]].lesson_id,
                        slot_a_action=sides[assignments[0]].target_action,
                        slot_b_action=sides[assignments[1]].target_action,
                        expected_action=sides[selected_side].target_action,
                        counterfactual_action=sides[other_side].target_action,
                        ordered_candidates=candidate_order,
                        prompt=prompt,
                        source_a_probe_id=lesson_a.probe_id,
                        source_b_probe_id=lesson_b.probe_id,
                        source_a_prompt_sha256=_digest_text(lesson_a.receipt_a_prompt),
                        source_b_prompt_sha256=_digest_text(lesson_b.receipt_a_prompt),
                    )
                )
    audit = audit_binding_bank(probes, spec)
    return probes, audit


def audit_binding_bank(probes: list[BindingProbe], spec: AuditSpec) -> dict[str, Any]:
    cell_counts = Counter(probe.cell for probe in probes)
    units: dict[str, list[BindingProbe]] = {}
    for probe in probes:
        units.setdefault(probe.unit_id, []).append(probe)
    expected_positions = {
        cell: Counter(
            probe.ordered_candidates.index(probe.expected_action)
            for probe in probes
            if probe.cell == cell
        )
        for cell in CELLS
    }
    checks = {
        "prompt_count": len(probes) == spec.prompt_count,
        "unit_count": len(units) == spec.unit_count,
        "unique_probe_ids": len({probe.probe_id for probe in probes}) == len(probes),
        "unique_prompts": len({_digest_text(probe.prompt) for probe in probes}) == len(probes),
        "complete_cells": cell_counts == {cell: 48 for cell in CELLS},
        "complete_factorial": all(
            {(probe.orientation, probe.receipt) for probe in rows}
            == {
                (orientation, receipt)
                for orientation in ("canonical", "swapped")
                for receipt in ("A", "B")
            }
            for rows in units.values()
        ),
        "candidate_order_fixed_within_unit": all(
            len({probe.ordered_candidates for probe in rows}) == 1 for rows in units.values()
        ),
        "two_distinct_valid_actions": all(
            probe.expected_action != probe.counterfactual_action
            and {probe.expected_action, probe.counterfactual_action}
            <= set(probe.ordered_candidates)
            for probe in probes
        ),
        "candidate_position_balance": all(
            positions == {0: 12, 1: 12, 2: 12, 3: 12} for positions in expected_positions.values()
        ),
        "receipt_only_intervention": all(
            _receipt_prompt(_probe_at(rows, orientation, "A").prompt, "B")
            == _probe_at(rows, orientation, "B").prompt
            for rows in units.values()
            for orientation in ("canonical", "swapped")
        ),
        "binding_swap_reverses_actions": all(
            _probe_at(rows, "canonical", "A").expected_action
            == _probe_at(rows, "swapped", "B").expected_action
            and _probe_at(rows, "canonical", "B").expected_action
            == _probe_at(rows, "swapped", "A").expected_action
            for rows in units.values()
        ),
        "memory_order_fixed_within_unit": all(
            len({_memory_block(probe.prompt) for probe in rows}) == 1 for rows in units.values()
        ),
    }
    payload = {
        "schema_version": "p0d2hrab-bank-audit-v1",
        "bank_version": BANK_VERSION,
        "spec": spec.to_dict(),
        "cell_counts": dict(sorted(cell_counts.items())),
        "expected_candidate_positions": {
            cell: dict(sorted(positions.items())) for cell, positions in expected_positions.items()
        },
        "checks": checks,
        "records_sha256": _json_hash([probe.to_dict() for probe in probes]),
        "all_checks_passed": all(checks.values()),
    }
    if payload["all_checks_passed"] is not True:
        raise ValueError(f"receipt/action binding bank audit failed: {checks}")
    return payload


def _binding_prompt(
    *,
    receipt: str,
    slot_a_payload: str,
    slot_b_payload: str,
    memory_a: str,
    memory_b: str,
    candidates: tuple[str, ...],
) -> str:
    return (
        f"{RECEIPT_PREFIX}{receipt}. "
        f"Slot A contains {slot_a_payload} "
        f"Slot B contains {slot_b_payload} "
        f"{ACTION_INSTRUCTION}\n\n"
        f"<verified_memory>\n{memory_a}\n{memory_b}\n</verified_memory>\n"
        f"Only output one action token from: {', '.join(candidates)}.\nAction:"
    )


def _valid_query_payload(probe: HandoffProbe) -> str:
    prompt = probe.receipt_a_prompt
    start_a = prompt.index("Slot A contains ")
    start_b = prompt.index("Slot B contains ")
    instruction = prompt.index(f" {ACTION_INSTRUCTION}")
    if probe.target_slot == "A":
        return prompt[start_a + len("Slot A contains ") : start_b].strip()
    if probe.target_slot == "B":
        return prompt[start_b + len("Slot B contains ") : instruction].strip()
    raise ValueError(f"unsupported RSH target slot: {probe.target_slot}")


def _verified_memory(prompt: str) -> str:
    start = prompt.index("<verified_memory>") + len("<verified_memory>")
    end = prompt.index("</verified_memory>")
    value = prompt[start:end].strip()
    if not value or "\n" in value:
        raise ValueError("expected one frozen verified-memory statement per lesson")
    return value


def _memory_block(prompt: str) -> str:
    start = prompt.index("<verified_memory>") + len("<verified_memory>")
    end = prompt.index("</verified_memory>")
    return prompt[start:end].strip()


def _probe_at(rows: list[BindingProbe], orientation: str, receipt: str) -> BindingProbe:
    matches = [
        probe for probe in rows if probe.orientation == orientation and probe.receipt == receipt
    ]
    if len(matches) != 1:
        raise ValueError("receipt/action factorial cell is not unique")
    return matches[0]


def _digest_text(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _json_hash(value: object) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
