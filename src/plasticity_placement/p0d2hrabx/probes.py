from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from plasticity_placement.p0d2hrab.probes import BindingProbe
from plasticity_placement.p0d2hrabc.probes import _memory_block, _slot_payloads
from plasticity_placement.p0d2hrabx.config import (
    BANK_VERSION,
    CANDIDATE_ROTATIONS,
    CODEBOOKS,
    DISPLAY_ORDERS,
    ORIENTATIONS,
    RECEIPTS,
    AuditSpec,
)
from plasticity_placement.p0d2hrsh.probes import RECEIPT_PREFIX

CODEBOOK_TEXT = {
    "canonical": "Receipt A selects Slot A; Receipt B selects Slot B.",
    "crossed": "Receipt A selects Slot B; Receipt B selects Slot A.",
}
ACTION_INSTRUCTION = "Follow the routing codebook and return the action for the selected slot."


@dataclass(frozen=True, slots=True)
class LabelDisentanglementProbe:
    probe_id: str
    source_probe_id: str
    unit_id: str
    pair_id: str
    route_variant: int
    orientation: str
    receipt: str
    codebook: str
    selected_slot_label: str
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


def compile_label_disentanglement_bank(
    source_probes: list[BindingProbe], spec: AuditSpec
) -> tuple[list[LabelDisentanglementProbe], dict[str, Any]]:
    if len(source_probes) != spec.source_prompt_count:
        raise ValueError("label disentanglement requires the exact 192-prompt RAB bank")
    by_unit: dict[str, dict[tuple[str, str], BindingProbe]] = defaultdict(dict)
    for source in source_probes:
        key = (source.orientation, source.receipt)
        if key in by_unit[source.unit_id]:
            raise ValueError(f"duplicate RAB source cell: {source.unit_id}/{key}")
        by_unit[source.unit_id][key] = source
    expected_source_cells = {
        (orientation, receipt) for orientation in ORIENTATIONS for receipt in RECEIPTS
    }
    if len(by_unit) != spec.unit_count or any(
        set(rows) != expected_source_cells for rows in by_unit.values()
    ):
        raise ValueError(
            "RAB source units do not contain the complete orientation/receipt factorial"
        )

    probes: list[LabelDisentanglementProbe] = []
    for unit_id, source_cells in sorted(by_unit.items()):
        for orientation in ORIENTATIONS:
            reference = source_cells[(orientation, "A")]
            slot_payloads = _slot_payloads(reference.prompt)
            memory = _memory_block(reference.prompt)
            for receipt in RECEIPTS:
                for codebook in CODEBOOKS:
                    selected_label = selected_slot_label(receipt, codebook)
                    selected_source = source_cells[(orientation, selected_label)]
                    for display_order in DISPLAY_ORDERS:
                        for rotation in CANDIDATE_ROTATIONS:
                            candidates = _rotate(reference.ordered_candidates, rotation)
                            prompt = _label_disentanglement_prompt(
                                receipt=receipt,
                                codebook=codebook,
                                slot_payloads=slot_payloads,
                                display_order=display_order,
                                memory=memory,
                                candidates=candidates,
                            )
                            probes.append(
                                LabelDisentanglementProbe(
                                    probe_id=(
                                        f"{unit_id}_{orientation}_receipt_{receipt}_"
                                        f"codebook_{codebook}_display_{display_order}_"
                                        f"rotation_{rotation}"
                                    ),
                                    source_probe_id=selected_source.probe_id,
                                    unit_id=unit_id,
                                    pair_id=reference.pair_id,
                                    route_variant=reference.route_variant,
                                    orientation=orientation,
                                    receipt=receipt,
                                    codebook=codebook,
                                    selected_slot_label=selected_label,
                                    display_order=display_order,
                                    selected_display_position=display_order.index(selected_label),
                                    candidate_rotation=rotation,
                                    expected_candidate_position=candidates.index(
                                        selected_source.expected_action
                                    ),
                                    lesson_a_id=reference.lesson_a_id,
                                    lesson_b_id=reference.lesson_b_id,
                                    slot_a_lesson_id=reference.slot_a_lesson_id,
                                    slot_b_lesson_id=reference.slot_b_lesson_id,
                                    slot_a_action=reference.slot_a_action,
                                    slot_b_action=reference.slot_b_action,
                                    expected_action=selected_source.expected_action,
                                    counterfactual_action=selected_source.counterfactual_action,
                                    ordered_candidates=candidates,
                                    prompt=prompt,
                                    source_prompt_sha256=_digest_text(selected_source.prompt),
                                )
                            )
    audit = audit_label_disentanglement_bank(probes, source_probes, spec)
    return probes, audit


def selected_slot_label(receipt: str, codebook: str) -> str:
    if receipt not in RECEIPTS or codebook not in CODEBOOKS:
        raise ValueError("receipt and codebook must use frozen values")
    if codebook == "canonical":
        return receipt
    return "B" if receipt == "A" else "A"


def audit_label_disentanglement_bank(
    probes: list[LabelDisentanglementProbe],
    source_probes: list[BindingProbe],
    spec: AuditSpec,
) -> dict[str, Any]:
    by_unit: dict[str, list[LabelDisentanglementProbe]] = defaultdict(list)
    for probe in probes:
        by_unit[probe.unit_id].append(probe)
    source_index = {probe.probe_id: probe for probe in source_probes}
    expected_factorial = {
        (orientation, receipt, codebook, display, rotation)
        for orientation in ORIENTATIONS
        for receipt in RECEIPTS
        for codebook in CODEBOOKS
        for display in DISPLAY_ORDERS
        for rotation in CANDIDATE_ROTATIONS
    }
    prompt_multiplicities = Counter(_digest_text(probe.prompt) for probe in probes)
    joint = Counter(
        (
            probe.receipt,
            probe.selected_slot_label,
            probe.selected_display_position,
            probe.expected_candidate_position,
        )
        for probe in probes
    )
    semantic_groups: dict[
        tuple[str, str, str, str, int], list[LabelDisentanglementProbe]
    ] = defaultdict(list)
    for probe in probes:
        semantic_groups[
            (
                probe.unit_id,
                probe.orientation,
                probe.selected_slot_label,
                probe.display_order,
                probe.candidate_rotation,
            )
        ].append(probe)
    checks = {
        "prompt_count": len(probes) == spec.prompt_count,
        "unit_count": len(by_unit) == spec.unit_count,
        "unique_probe_ids": len({probe.probe_id for probe in probes}) == len(probes),
        "complete_unit_factorial": all(
            len(rows) == spec.conditions_per_unit
            and {
                (
                    row.orientation,
                    row.receipt,
                    row.codebook,
                    row.display_order,
                    row.candidate_rotation,
                )
                for row in rows
            }
            == expected_factorial
            for rows in by_unit.values()
        ),
        "receipt_slot_independently_crossed": Counter(
            (row.receipt, row.selected_slot_label) for row in probes
        )
        == {(receipt, slot): 768 for receipt in RECEIPTS for slot in RECEIPTS},
        "selected_slot_derived_from_codebook": all(
            row.selected_slot_label == selected_slot_label(row.receipt, row.codebook)
            for row in probes
        ),
        "selected_display_position_balanced": Counter(
            row.selected_display_position for row in probes
        )
        == {0: 1536, 1: 1536},
        "candidate_position_balanced": Counter(
            row.expected_candidate_position for row in probes
        )
        == {0: 768, 1: 768, 2: 768, 3: 768},
        "joint_primary_factors_crossed": len(joint) == 32
        and all(count == 96 for count in joint.values()),
        "semantic_equivalence_across_codebooks": all(
            len(rows) == 2
            and len({row.receipt for row in rows}) == 2
            and len({row.codebook for row in rows}) == 2
            and len({row.expected_action for row in rows}) == 1
            and len({row.counterfactual_action for row in rows}) == 1
            for rows in semantic_groups.values()
        ),
        "source_semantics_preserved": all(
            _source_semantics_match(row, source_index[row.source_probe_id]) for row in probes
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
        "expected_textual_aliasing": len(prompt_multiplicities) == 768
        and set(prompt_multiplicities.values()) == {4},
    }
    payload = {
        "schema_version": "p0d2hrabx-bank-audit-v1",
        "bank_version": BANK_VERSION,
        "spec": spec.to_dict(),
        "factor_counts": {
            "orientation": dict(sorted(Counter(row.orientation for row in probes).items())),
            "receipt": dict(sorted(Counter(row.receipt for row in probes).items())),
            "codebook": dict(sorted(Counter(row.codebook for row in probes).items())),
            "selected_slot_label": dict(
                sorted(Counter(row.selected_slot_label for row in probes).items())
            ),
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
        "prompt_condition_count": len(probes),
        "prompt_text_count": len(prompt_multiplicities),
        "prompt_text_multiplicity_counts": dict(
            sorted(Counter(prompt_multiplicities.values()).items())
        ),
        "joint_primary_factor_counts": {
            f"receipt_{receipt}_slot_{slot}_display_{display}_candidate_{candidate}": count
            for (receipt, slot, display, candidate), count in sorted(joint.items())
        },
        "checks": checks,
        "records_sha256": _json_hash([probe.to_dict() for probe in probes]),
        "all_checks_passed": all(checks.values()),
    }
    if payload["all_checks_passed"] is not True:
        raise ValueError(f"RAB label-disentanglement bank audit failed: {checks}")
    return payload


def _label_disentanglement_prompt(
    *,
    receipt: str,
    codebook: str,
    slot_payloads: dict[str, str],
    display_order: str,
    memory: str,
    candidates: tuple[str, ...],
) -> str:
    slots = " ".join(f"Slot {label} contains {slot_payloads[label]}" for label in display_order)
    return (
        f"Routing codebook: {CODEBOOK_TEXT[codebook]} {RECEIPT_PREFIX}{receipt}. "
        f"{slots} {ACTION_INSTRUCTION}\n\n"
        f"<verified_memory>\n{memory}\n</verified_memory>\n"
        f"Only output one action token from: {', '.join(candidates)}.\nAction:"
    )


def _rotate(values: tuple[str, ...], rotation: int) -> tuple[str, ...]:
    if len(values) != 4 or rotation not in CANDIDATE_ROTATIONS:
        raise ValueError("label disentanglement requires four candidates and rotations 0..3")
    return values[rotation:] + values[:rotation]


def _source_semantics_match(
    row: LabelDisentanglementProbe, source: BindingProbe
) -> bool:
    return all(
        getattr(row, field) == getattr(source, field)
        for field in (
            "unit_id",
            "pair_id",
            "route_variant",
            "orientation",
            "lesson_a_id",
            "lesson_b_id",
            "slot_a_lesson_id",
            "slot_b_lesson_id",
            "slot_a_action",
            "slot_b_action",
            "expected_action",
            "counterfactual_action",
        )
    ) and source.receipt == row.selected_slot_label


def _digest_text(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _json_hash(value: object) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
