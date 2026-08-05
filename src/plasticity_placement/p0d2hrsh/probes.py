from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

from plasticity_placement.p0d2hrsh.config import BANK_VERSION, PROMPT_KINDS, AuditSpec
from plasticity_placement.p0d2hrtb.probes import BridgeProbe, _external_payload_segment

RECEIPT_PREFIX = (
    "The routing stage has been independently completed. Its signed receipt selects slot "
)


@dataclass(frozen=True, slots=True)
class HandoffProbe:
    probe_id: str
    lesson_id: str
    pair_id: str
    lesson_type: str
    lesson_side: str
    route_variant: int
    source_probe_id: str
    target_slot: str
    target_action: str
    slot_candidates: tuple[str, ...]
    action_candidates: tuple[str, ...]
    slot_prompt: str
    direct_prompt: str
    receipt_a_prompt: str
    receipt_b_prompt: str
    rtb_slot_probe_id: str
    rtb_direct_probe_id: str
    rtb_oracle_probe_id: str
    rtb_slot_prompt_sha256: str
    rtb_direct_prompt_sha256: str
    rtb_oracle_prompt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def prompt(self, kind: str) -> str:
        return {
            "slot_readout": self.slot_prompt,
            "direct_action": self.direct_prompt,
            "receipt_A": self.receipt_a_prompt,
            "receipt_B": self.receipt_b_prompt,
        }[kind]

    def candidates(self, kind: str) -> tuple[str, ...]:
        return self.slot_candidates if kind == "slot_readout" else self.action_candidates


def compile_handoff_bank(
    bridge_probes: list[BridgeProbe], spec: AuditSpec
) -> tuple[list[HandoffProbe], dict[str, Any]]:
    index = {(probe.source_probe_id, probe.cell): probe for probe in bridge_probes}
    source_ids = sorted(
        probe.source_probe_id for probe in bridge_probes if probe.cell == "external_action"
    )
    if len(source_ids) != spec.unit_count or len(set(source_ids)) != spec.unit_count:
        raise ValueError("RTB external action source IDs are incomplete")
    probes: list[HandoffProbe] = []
    for source_id in source_ids:
        slot = index[(source_id, "slot_default_natural_external")]
        direct = index[(source_id, "external_action")]
        oracle = index[(source_id, "forced_slot_action")]
        if not (
            slot.lesson_id == direct.lesson_id == oracle.lesson_id
            and slot.route_variant == direct.route_variant == oracle.route_variant
            and slot.target_slot == direct.target_slot == oracle.target_slot
            and direct.expected_candidate == oracle.expected_candidate
            and direct.ordered_candidates == oracle.ordered_candidates
        ):
            raise ValueError(f"RTB handoff anchors differ for {source_id}")
        target_slot = str(slot.expected_candidate)
        if target_slot not in {"A", "B"}:
            raise ValueError(f"unexpected natural slot label for {source_id}")
        receipt_a = _receipt_prompt(oracle.prompt, "A")
        receipt_b = _receipt_prompt(oracle.prompt, "B")
        oracle_receipt = receipt_a if target_slot == "A" else receipt_b
        if oracle_receipt != oracle.prompt:
            raise ValueError(f"RTB oracle receipt is not reproducible for {source_id}")
        probes.append(
            HandoffProbe(
                probe_id=f"{direct.lesson_id}_rsh_r{direct.route_variant:02d}",
                lesson_id=direct.lesson_id,
                pair_id=direct.pair_id,
                lesson_type=direct.lesson_type,
                lesson_side=direct.lesson_side,
                route_variant=direct.route_variant,
                source_probe_id=source_id,
                target_slot=target_slot,
                target_action=direct.expected_candidate,
                slot_candidates=slot.ordered_candidates,
                action_candidates=direct.ordered_candidates,
                slot_prompt=slot.prompt,
                direct_prompt=direct.prompt,
                receipt_a_prompt=receipt_a,
                receipt_b_prompt=receipt_b,
                rtb_slot_probe_id=slot.probe_id,
                rtb_direct_probe_id=direct.probe_id,
                rtb_oracle_probe_id=oracle.probe_id,
                rtb_slot_prompt_sha256=_digest_text(slot.prompt),
                rtb_direct_prompt_sha256=_digest_text(direct.prompt),
                rtb_oracle_prompt_sha256=_digest_text(oracle.prompt),
            )
        )
    audit = audit_handoff_bank(probes, spec)
    return probes, audit


def audit_handoff_bank(probes: list[HandoffProbe], spec: AuditSpec) -> dict[str, Any]:
    prompt_records = [
        {
            "probe_id": probe.probe_id,
            "kind": kind,
            "prompt_sha256": _digest_text(probe.prompt(kind)),
            "candidates": list(probe.candidates(kind)),
        }
        for probe in probes
        for kind in PROMPT_KINDS
    ]
    checks = {
        "unit_count": len(probes) == spec.unit_count,
        "unique_probe_ids": len({probe.probe_id for probe in probes}) == spec.unit_count,
        "lesson_balance": set(Counter(probe.lesson_id for probe in probes).values()) == {4},
        "route_variant_balance": Counter(probe.route_variant for probe in probes)
        == {1: 24, 2: 24, 3: 24, 4: 24},
        "target_slot_balance": Counter(probe.target_slot for probe in probes) == {"A": 48, "B": 48},
        "prompt_count": len(prompt_records) == spec.unit_count * spec.prompt_kind_count,
        "unique_prompts": len({row["prompt_sha256"] for row in prompt_records})
        == len(prompt_records),
        "oracle_anchor_exact": all(
            _digest_text(
                probe.receipt_a_prompt if probe.target_slot == "A" else probe.receipt_b_prompt
            )
            == probe.rtb_oracle_prompt_sha256
            for probe in probes
        ),
        "receipt_payload_identity": all(
            _external_payload_segment(probe.direct_prompt)
            == _external_payload_segment(probe.receipt_a_prompt)
            == _external_payload_segment(probe.receipt_b_prompt)
            for probe in probes
        ),
        "receipt_only_slot_differs": all(
            _receipt_prompt(probe.receipt_a_prompt, "B") == probe.receipt_b_prompt
            and _receipt_prompt(probe.receipt_b_prompt, "A") == probe.receipt_a_prompt
            for probe in probes
        ),
        "candidate_domains": all(
            set(probe.slot_candidates) == {"A", "B"}
            and len(probe.action_candidates) == 4
            and probe.target_action in probe.action_candidates
            for probe in probes
        ),
    }
    payload = {
        "schema_version": "p0d2hrsh-bank-audit-v1",
        "bank_version": BANK_VERSION,
        "spec": spec.to_dict(),
        "checks": checks,
        "prompt_records_sha256": _json_hash(prompt_records),
        "records_sha256": _json_hash([probe.to_dict() for probe in probes]),
        "all_checks_passed": all(checks.values()),
    }
    if payload["all_checks_passed"] is not True:
        raise ValueError(f"route-state handoff bank audit failed: {checks}")
    return payload


def _receipt_prompt(prompt: str, slot: str) -> str:
    if slot not in {"A", "B"}:
        raise ValueError("receipt slot must be A or B")
    start = prompt.find(RECEIPT_PREFIX)
    if start < 0:
        raise ValueError("signed receipt prefix is missing")
    label_start = start + len(RECEIPT_PREFIX)
    if prompt[label_start : label_start + 2] not in {"A.", "B."}:
        raise ValueError("signed receipt slot is malformed")
    return prompt[:label_start] + slot + prompt[label_start + 1 :]


def _digest_text(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _json_hash(value: object) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
