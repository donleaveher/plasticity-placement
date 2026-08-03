from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from plasticity_placement.p0c.domain import Arm, CompiledLesson, P0CProbe
from plasticity_placement.p0d2hc.prompting import render_calibration_probe
from plasticity_placement.p0d2hcrd.probes import (
    DecompositionProbe,
    compile_decomposition_bank,
)
from plasticity_placement.p0d2hrtb.config import (
    ACTION_CELLS,
    BANK_VERSION,
    GRAMMARS,
    LEXICONS,
    PAYLOADS,
    AuditSpec,
)


@dataclass(frozen=True, slots=True)
class BridgeProbe:
    probe_id: str
    lesson_id: str
    pair_id: str
    lesson_type: str
    lesson_side: str
    cell: str
    response_type: str
    route_variant: int
    grammar: str | None
    lexicon: str | None
    payload: str
    target_slot: str
    expected_candidate: str
    ordered_candidates: tuple[str, ...]
    prompt: str
    source_probe_id: str
    route_anchor_probe_id: str | None
    route_anchor_prompt_sha256: str | None
    route_anchor_static_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_bridge_bank(
    selected: tuple[CompiledLesson, ...],
    hard_bank: dict[str, tuple[P0CProbe, ...]],
    spec: AuditSpec,
) -> tuple[list[BridgeProbe], dict[str, Any]]:
    if len(selected) != spec.lesson_count:
        raise ValueError("route-transfer bridge requires exactly 24 frozen lessons")
    decomposition_bank, decomposition_audit = compile_decomposition_bank(selected, hard_bank)
    if decomposition_audit["all_checks_passed"] is not True:
        raise ValueError("frozen CRD decomposition bank failed its own audit")
    probes: list[BridgeProbe] = []
    for lesson_index, item in enumerate(selected):
        sources = _conditional_sources(hard_bank[item.lesson.lesson_id])
        for variant in range(1, spec.route_variants_per_lesson + 1):
            source = sources[variant]
            route_anchor = _route_anchor(
                decomposition_bank[item.lesson.lesson_id],
                lesson_index=lesson_index,
                variant=variant,
            )
            for grammar in GRAMMARS:
                for lexicon in LEXICONS:
                    for payload in PAYLOADS:
                        probes.append(
                            _slot_probe(
                                item,
                                source,
                                route_anchor,
                                variant=variant,
                                grammar=grammar,
                                lexicon=lexicon,
                                payload=payload,
                            )
                        )
            probes.append(_external_action_probe(item, source, variant))
            probes.append(
                _forced_slot_action_probe(
                    item,
                    source,
                    variant=variant,
                )
            )
    audit = audit_bridge_bank(probes, spec)
    return probes, audit


def audit_bridge_bank(probes: list[BridgeProbe], spec: AuditSpec) -> dict[str, Any]:
    rows = [probe.to_dict() for probe in probes]
    ids = [probe.probe_id for probe in probes]
    cell_counts = Counter(probe.cell for probe in probes)
    factorial_cells = [
        _cell(grammar, lexicon, payload)
        for grammar in GRAMMARS
        for lexicon in LEXICONS
        for payload in PAYLOADS
    ]
    expected_counts = {cell: spec.rows_per_cell for cell in factorial_cells}
    expected_counts.update({cell: spec.rows_per_cell for cell in ACTION_CELLS})
    checks = {
        "decision_count": len(probes) == spec.decision_count,
        "unique_probe_ids": len(ids) == len(set(ids)),
        "unique_prompts": len({sha256(probe.prompt.encode()).hexdigest() for probe in probes})
        == len(probes),
        "complete_cells": dict(sorted(cell_counts.items()))
        == dict(sorted(expected_counts.items())),
        "lesson_balance": set(Counter(probe.lesson_id for probe in probes).values()) == {40},
        "target_balance": set(Counter(probe.target_slot for probe in probes).values()) == {480},
        "candidate_position_balance": all(
            set(
                Counter(
                    probe.ordered_candidates.index(probe.expected_candidate)
                    for probe in probes
                    if probe.cell == cell
                ).values()
            )
            == ({48} if cell not in ACTION_CELLS else {24})
            for cell in expected_counts
        ),
        "factorial_response_is_slot": all(
            probe.response_type == "slot" for probe in probes if probe.cell not in ACTION_CELLS
        ),
        "action_response_is_action": all(
            probe.response_type == "action" for probe in probes if probe.cell in ACTION_CELLS
        ),
        "exact_crd_route_anchor": all(
            sha256(probe.prompt.encode()).hexdigest() == probe.route_anchor_prompt_sha256
            for probe in probes
            if probe.cell == "slot_map_snake_opaque"
        ),
        "route_anchor_provenance_complete": all(
            (probe.route_anchor_probe_id is not None) == (probe.response_type == "slot")
            and (probe.route_anchor_prompt_sha256 is not None) == (probe.response_type == "slot")
            and (probe.route_anchor_static_sha256 is not None) == (probe.response_type == "slot")
            for probe in probes
        ),
        "slot_content_order_balance": all(
            Counter(
                _slot_content_order(probe.prompt, probe.lexicon)
                for probe in probes
                if probe.cell == cell
            )
            == {1: 48, 2: 48}
            for cell in factorial_cells
        ),
        "forced_action_payload_matches_external": _action_payloads_match(probes),
    }
    payload = {
        "schema_version": "p0d2hrtb-bank-audit-v1",
        "bank_version": BANK_VERSION,
        "spec": spec.to_dict(),
        "cell_counts": dict(sorted(cell_counts.items())),
        "checks": checks,
        "records_sha256": _json_hash(rows),
        "all_checks_passed": all(checks.values()),
    }
    if not payload["all_checks_passed"]:
        raise ValueError(f"route-transfer bridge bank audit failed: {checks}")
    return payload


def _slot_probe(
    item: CompiledLesson,
    source: P0CProbe,
    route_anchor: DecompositionProbe,
    *,
    variant: int,
    grammar: str,
    lexicon: str,
    payload: str,
) -> BridgeProbe:
    target_key = "a" if route_anchor.target_slot == "slot_a" else "b"
    decoy_key = "b" if target_key == "a" else "a"
    labels = _labels(lexicon)
    if grammar == "map":
        routing = route_anchor.prompt.split("\n", maxsplit=1)[0]
        if lexicon == "natural":
            routing = _naturalize_route_slots(routing)
    elif grammar == "default":
        current_marker = str(route_anchor.current_marker)
        routing = (
            f"Routing rule: marker {current_marker} selects "
            f"{_route_label(lexicon, target_key)}; every other marker selects "
            f"{_route_label(lexicon, decoy_key)}. Current marker: {current_marker}."
        )
    else:
        raise ValueError(f"unknown bridge grammar: {grammar}")
    slot_lines, memory = _payload(
        item,
        source,
        variant,
        labels,
        payload,
        content_order=int(route_anchor.slot_content_order),
    )
    candidates = route_anchor.ordered_candidates
    if lexicon == "natural":
        candidates = tuple("A" if value == "slot_a" else "B" for value in candidates)
    if grammar == "map" and lexicon == "snake" and payload == "opaque":
        prompt = route_anchor.prompt
    else:
        prompt = (
            f"{routing}\n"
            + "\n".join(slot_lines)
            + memory
            + "\nApply only the routing rule; do not retrieve or infer an action. "
            "Which slot is selected? Choose exactly one allowed answer and output only that "
            f"token from: {', '.join(candidates)}.\nAnswer:"
        )
    cell = _cell(grammar, lexicon, payload)
    return _probe(
        item,
        source,
        cell=cell,
        response_type="slot",
        route_variant=variant,
        grammar=grammar,
        lexicon=lexicon,
        payload=payload,
        target_slot=target_key,
        expected_candidate=labels[target_key],
        ordered_candidates=candidates,
        prompt=prompt,
        route_anchor_probe_id=route_anchor.probe_id,
        route_anchor_prompt_sha256=sha256(route_anchor.prompt.encode()).hexdigest(),
        route_anchor_static_sha256=route_anchor_static_sha256(route_anchor),
    )


def _external_action_probe(item: CompiledLesson, source: P0CProbe, variant: int) -> BridgeProbe:
    rendered, _, _ = render_calibration_probe(
        source,
        arm=Arm.EXTERNAL,
        external_note=item.external_note,
    )
    return _probe(
        item,
        source,
        cell="external_action",
        response_type="action",
        route_variant=variant,
        grammar="default",
        lexicon="natural",
        payload="external",
        target_slot="a" if variant % 2 else "b",
        expected_candidate=source.expected_action,
        ordered_candidates=source.action_choices,
        prompt=rendered.prompt,
        route_anchor_probe_id=None,
        route_anchor_prompt_sha256=None,
        route_anchor_static_sha256=None,
    )


def _forced_slot_action_probe(
    item: CompiledLesson, source: P0CProbe, *, variant: int
) -> BridgeProbe:
    target_key = "a" if variant % 2 else "b"
    labels = _labels("natural")
    rendered, _, _ = render_calibration_probe(
        source,
        arm=Arm.EXTERNAL,
        external_note=item.external_note,
    )
    route_prefix = (
        f"Routing rule: marker route-{variant} selects slot {labels[target_key]}; every other "
        f"marker selects slot {labels['b' if target_key == 'a' else 'a']}. "
        f"Current marker: route-{variant}. "
    )
    if not rendered.prompt.startswith(route_prefix):
        raise ValueError("frozen external conditional-route prefix changed")
    action_instruction = (
        "First apply the routing rule, then return the action for the selected slot."
    )
    if rendered.prompt.count(action_instruction) != 1:
        raise ValueError("frozen external conditional-route instruction changed")
    prompt = (
        f"The routing stage has been independently completed. Its signed receipt selects "
        f"slot {labels[target_key]}. "
        + rendered.prompt.removeprefix(route_prefix).replace(
            action_instruction,
            "The routing receipt is authoritative; return the action for the selected slot.",
            1,
        )
    )
    return _probe(
        item,
        source,
        cell="forced_slot_action",
        response_type="action",
        route_variant=variant,
        grammar=None,
        lexicon="natural",
        payload="external",
        target_slot=target_key,
        expected_candidate=source.expected_action,
        ordered_candidates=source.action_choices,
        prompt=prompt,
        route_anchor_probe_id=None,
        route_anchor_prompt_sha256=None,
        route_anchor_static_sha256=None,
    )


def _probe(
    item: CompiledLesson,
    source: P0CProbe,
    *,
    cell: str,
    response_type: str,
    route_variant: int,
    grammar: str | None,
    lexicon: str | None,
    payload: str,
    target_slot: str,
    expected_candidate: str,
    ordered_candidates: tuple[str, ...],
    prompt: str,
    route_anchor_probe_id: str | None,
    route_anchor_prompt_sha256: str | None,
    route_anchor_static_sha256: str | None,
) -> BridgeProbe:
    lesson = item.lesson
    return BridgeProbe(
        probe_id=f"{lesson.lesson_id}_rtb_{cell}_r{route_variant:02d}",
        lesson_id=lesson.lesson_id,
        pair_id=lesson.pair_id,
        lesson_type=lesson.lesson_type,
        lesson_side=lesson.lesson_id.rsplit("_", maxsplit=1)[-1],
        cell=cell,
        response_type=response_type,
        route_variant=route_variant,
        grammar=grammar,
        lexicon=lexicon,
        payload=payload,
        target_slot=target_slot,
        expected_candidate=expected_candidate,
        ordered_candidates=ordered_candidates,
        prompt=prompt,
        source_probe_id=source.probe_id,
        route_anchor_probe_id=route_anchor_probe_id,
        route_anchor_prompt_sha256=route_anchor_prompt_sha256,
        route_anchor_static_sha256=route_anchor_static_sha256,
    )


def _payload(
    item: CompiledLesson,
    source: P0CProbe,
    variant: int,
    labels: dict[str, str],
    payload: str,
    content_order: int,
) -> tuple[tuple[str, str], str]:
    if payload == "opaque":
        values = {
            key: (
                f"{_content_label('snake' if labels[key].startswith('slot_') else 'natural', key)} "
                "contains opaque record "
                f"opaque_{_digest(item.lesson.lesson_id, variant, key)}."
            )
            for key in ("a", "b")
        }
        return _ordered_values(values, content_order), ""
    if payload != "external":
        raise ValueError(f"unknown bridge payload: {payload}")
    values = _exact_external_slot_lines(source)
    if labels["a"] == "slot_a":
        values = {
            "a": values["a"].replace("Slot A", "slot_a", 1),
            "b": values["b"].replace("Slot B", "slot_b", 1),
        }
    memory = f"\n\n<verified_memory>\n{item.external_note}\n</verified_memory>\n"
    return _ordered_values(values, content_order), memory


def _route_anchor(
    probes: tuple[DecompositionProbe, ...], *, lesson_index: int, variant: int
) -> DecompositionProbe:
    candidate_order = 1 + ((lesson_index + variant) % 2)
    content_order = 1 + ((lesson_index + variant // 2) % 2)
    matches = [
        probe
        for probe in probes
        if probe.endpoint == "route_only"
        and probe.route_variant == variant
        and probe.slot_candidate_order == candidate_order
        and probe.slot_content_order == content_order
    ]
    if len(matches) != 1:
        raise ValueError("frozen CRD route-only anchor is not unique")
    return matches[0]


ROUTE_ANCHOR_STATIC_FIELDS = (
    "probe_id",
    "lesson_id",
    "pair_id",
    "lesson_type",
    "lesson_side",
    "endpoint",
    "expected_candidate",
    "ordered_candidates",
    "route_variant",
    "target_slot",
    "current_marker",
    "slot_candidate_order",
    "slot_content_order",
    "action_panel_position",
    "source_probe_id",
    "source_row_key",
)


def route_anchor_static_sha256(value: DecompositionProbe | dict[str, Any]) -> str:
    row = value.to_dict() if isinstance(value, DecompositionProbe) else value
    identity = {field: row[field] for field in ROUTE_ANCHOR_STATIC_FIELDS}
    identity["ordered_candidates"] = list(identity["ordered_candidates"])
    return _json_hash(identity)


def _ordered_values(values: dict[str, str], content_order: int) -> tuple[str, str]:
    if content_order == 1:
        return values["a"], values["b"]
    if content_order == 2:
        return values["b"], values["a"]
    raise ValueError("unsupported bridge slot-content order")


def _naturalize_route_slots(value: str) -> str:
    return value.replace("slot_a", "slot A").replace("slot_b", "slot B")


def _route_label(lexicon: str, key: str) -> str:
    return _labels(lexicon)[key] if lexicon == "snake" else f"slot {_labels(lexicon)[key]}"


def _content_label(lexicon: str, key: str) -> str:
    return _labels(lexicon)[key] if lexicon == "snake" else f"Slot {_labels(lexicon)[key]}"


def _exact_external_slot_lines(source: P0CProbe) -> dict[str, str]:
    start_a = source.prompt.index("Slot A contains ")
    start_b = source.prompt.index("Slot B contains ")
    end_b = source.prompt.index(
        " First apply the routing rule, then return the action for the selected slot."
    )
    return {
        "a": source.prompt[start_a : start_b - 1],
        "b": source.prompt[start_b:end_b],
    }


def _external_payload_segment(prompt: str) -> str:
    start = prompt.index("Slot A contains ")
    boundaries = (
        "First apply the routing rule, then return the action for the selected slot.",
        "The routing receipt is authoritative; return the action for the selected slot.",
    )
    boundary = next(value for value in boundaries if value in prompt)
    slot_payload = prompt[start : prompt.index(boundary)].rstrip()
    memory_start = prompt.index("<verified_memory>")
    memory_end = prompt.index("</verified_memory>") + len("</verified_memory>")
    return slot_payload + "\n\n" + prompt[memory_start:memory_end]


def _action_payloads_match(probes: list[BridgeProbe]) -> bool:
    external = {
        probe.source_probe_id: _external_payload_segment(probe.prompt)
        for probe in probes
        if probe.cell == "external_action"
    }
    forced = {
        probe.source_probe_id: _external_payload_segment(probe.prompt)
        for probe in probes
        if probe.cell == "forced_slot_action"
    }
    return external == forced and len(external) == 96


def _slot_content_order(prompt: str, lexicon: str | None) -> int:
    lexicon_value = str(lexicon)
    a = _content_label(lexicon_value, "a")
    b = _content_label(lexicon_value, "b")
    return 1 if prompt.index(f"{a} contains") < prompt.index(f"{b} contains") else 2


def _labels(lexicon: str) -> dict[str, str]:
    if lexicon == "snake":
        return {"a": "slot_a", "b": "slot_b"}
    if lexicon == "natural":
        return {"a": "A", "b": "B"}
    raise ValueError(f"unknown bridge lexicon: {lexicon}")


def _cell(grammar: str, lexicon: str, payload: str) -> str:
    return f"slot_{grammar}_{lexicon}_{payload}"


def _conditional_sources(probes: tuple[P0CProbe, ...]) -> dict[int, P0CProbe]:
    selected = [probe for probe in probes if probe.category == "conditional_route"]
    result = {int(probe.probe_id.rsplit("_", maxsplit=1)[-1]): probe for probe in selected}
    if set(result) != {1, 2, 3, 4}:
        raise ValueError("frozen conditional-route variants changed")
    return result


def _digest(*parts: object) -> str:
    return sha256((BANK_VERSION + ":" + ":".join(map(str, parts))).encode()).hexdigest()[:10]


def _json_hash(value: object) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
