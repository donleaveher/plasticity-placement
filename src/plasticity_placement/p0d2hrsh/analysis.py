from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable
from statistics import mean
from typing import Any

from plasticity_placement.p0d2hrsh.config import PROMPT_KINDS, AuditSpec


def analyze_handoff(raw_rows: list[dict[str, Any]], spec: AuditSpec) -> dict[str, Any]:
    records, scoring_integrity = build_handoff_records(raw_rows, spec)

    endpoints = {
        endpoint: _endpoint_summary(records, endpoint, spec)
        for endpoint in (
            "slot_correct",
            "direct_correct",
            "predicted_chain_correct",
            "oracle_correct",
            "wrong_target_selected",
        )
    }

    observed_rescue = _cluster_interval(
        records,
        lambda row: (
            row["adapter"]["predicted_chain_correct"]
            - row["adapter"]["direct_correct"]
            - row["base"]["predicted_chain_correct"]
            + row["base"]["direct_correct"]
        ),
        spec=spec,
        seed_label="observed_handoff_rescue",
    )
    conservative_rescue = _cluster_interval(
        records,
        lambda row: (
            row["adapter"]["predicted_chain_lower"]
            - row["adapter"]["direct_upper"]
            - row["base"]["predicted_chain_upper"]
            + row["base"]["direct_lower"]
        ),
        spec=spec,
        seed_label="conservative_handoff_rescue",
    )
    optimistic_rescue = _cluster_interval(
        records,
        lambda row: (
            row["adapter"]["predicted_chain_upper"]
            - row["adapter"]["direct_lower"]
            - row["base"]["predicted_chain_lower"]
            + row["base"]["direct_upper"]
        ),
        spec=spec,
        seed_label="optimistic_handoff_rescue",
    )
    chain_minus_oracle = _cluster_interval(
        records,
        lambda row: row["adapter"]["chain_minus_oracle_lower"],
        spec=spec,
        seed_label="chain_minus_oracle",
    )
    oracle_minus_wrong = _cluster_interval(
        records,
        lambda row: row["adapter"]["oracle_lower"]
        - row["adapter"]["wrong_target_upper"],
        spec=spec,
        seed_label="oracle_minus_wrong",
    )

    rescue_supported = (
        scoring_integrity
        and conservative_rescue["estimate"] >= spec.material_handoff_rescue
        and conservative_rescue["ci95"][0] > 0.0
    )
    chain_noninferior = (
        scoring_integrity
        and chain_minus_oracle["ci95"][0]
        > -spec.chain_oracle_noninferiority_margin
    )
    receipt_specific = (
        scoring_integrity
        and oracle_minus_wrong["estimate"] >= spec.wrong_slot_specificity_threshold
        and oracle_minus_wrong["ci95"][0] > 0.0
    )
    if not scoring_integrity:
        status = "scoring_integrity_failed"
    elif rescue_supported and chain_noninferior and receipt_specific:
        status = "handoff_supported"
    else:
        status = "handoff_not_supported"

    return {
        "endpoint_summaries": endpoints,
        "primary_handoff_rescue": {
            "estimand": "(predicted-chain - direct) ON - (predicted-chain - direct) OFF",
            "observed": observed_rescue,
            "conservative_compatible_tie_bound": conservative_rescue,
            "optimistic_compatible_tie_bound": optimistic_rescue,
            "material_threshold": spec.material_handoff_rescue,
            "supported": rescue_supported,
        },
        "safeguards": {
            "adapter_on_chain_minus_oracle_conservative": {
                **chain_minus_oracle,
                "noninferiority_margin": spec.chain_oracle_noninferiority_margin,
                "passed": chain_noninferior,
            },
            "adapter_on_oracle_minus_wrong_target_conservative": {
                **oracle_minus_wrong,
                "material_threshold": spec.wrong_slot_specificity_threshold,
                "passed": receipt_specific,
            },
        },
        "tie_diagnostics": {
            "slot_ties": _tie_count(raw_rows, "slot_readout"),
            "action_ties": sum(bool(row.get("tie")) for row in raw_rows)
            - _tie_count(raw_rows, "slot_readout"),
            "all_ties_propagated_to_bounds": scoring_integrity,
        },
        "decision": {
            "status": status,
            "scoring_integrity": scoring_integrity,
            "handoff_rescue_supported": rescue_supported,
            "predicted_chain_oracle_noninferior": chain_noninferior,
            "wrong_receipt_specificity_passed": receipt_specific,
            "historical_rtb_decision_changed": False,
            "training_authorized": False,
            "mappings_per_adapter_authorized": False,
            "next_action": "review_handoff_evidence_before_any_intervention",
        },
    }


def build_handoff_records(
    raw_rows: list[dict[str, Any]], spec: AuditSpec
) -> tuple[list[dict[str, Any]], bool]:
    index = {
        (str(row["probe_id"]), str(row["adapter_state"]), str(row["prompt_kind"])): row
        for row in raw_rows
    }
    expected_count = spec.unit_count * 2 * len(PROMPT_KINDS)
    if len(index) != len(raw_rows) or len(index) != expected_count:
        raise ValueError("route-state handoff raw matrix is incomplete or duplicated")
    probe_ids = sorted({str(row["probe_id"]) for row in raw_rows})
    if len(probe_ids) != spec.unit_count:
        raise ValueError("route-state handoff probe count differs from the frozen bank")

    records = [_derive_record(probe_id, index) for probe_id in probe_ids]
    return records, all(_valid_outcome(row) for row in raw_rows)


def _derive_record(
    probe_id: str,
    index: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    first = index[(probe_id, "adapter_off", "slot_readout")]
    record: dict[str, Any] = {
        "probe_id": probe_id,
        "lesson_id": str(first["lesson_id"]),
        "route_variant": int(first["route_variant"]),
        "target_slot": str(first["target_slot"]),
        "target_action": str(first["target_action"]),
    }
    for state, label in (("adapter_off", "base"), ("adapter_on", "adapter")):
        rows = {kind: index[(probe_id, state, kind)] for kind in PROMPT_KINDS}
        _require_static(rows)
        slot = rows["slot_readout"]
        predicted_slot = _resolved_prediction(slot)
        possible_slots = _top_candidates(slot)
        if not slot.get("tie"):
            possible_slots = {predicted_slot}
        if not possible_slots or not possible_slots <= {"A", "B"}:
            raise ValueError(f"invalid slot candidates for {probe_id}/{state}")
        predicted_receipt = rows[f"receipt_{predicted_slot}"]
        target_slot = str(first["target_slot"])
        oracle = rows[f"receipt_{target_slot}"]
        wrong = rows[f"receipt_{'B' if target_slot == 'A' else 'A'}"]
        target_action = str(first["target_action"])
        chain_bounds = [
            _target_bounds(rows[f"receipt_{candidate}"], target_action)
            for candidate in sorted(possible_slots)
        ]
        direct_bounds = _target_bounds(rows["direct_action"], target_action)
        oracle_bounds = _target_bounds(oracle, target_action)
        wrong_bounds = _target_bounds(wrong, target_action)
        record[label] = {
            "slot_predicted": predicted_slot,
            "slot_correct": float(predicted_slot == target_slot),
            "slot_possible_top_candidates": sorted(possible_slots),
            "predicted_chain_correct": float(
                _resolved_prediction(predicted_receipt) == target_action
            ),
            "predicted_chain_lower": min(bounds[0] for bounds in chain_bounds),
            "predicted_chain_upper": max(bounds[1] for bounds in chain_bounds),
            "direct_correct": float(
                _resolved_prediction(rows["direct_action"]) == target_action
            ),
            "direct_lower": direct_bounds[0],
            "direct_upper": direct_bounds[1],
            "oracle_correct": float(_resolved_prediction(oracle) == target_action),
            "oracle_lower": oracle_bounds[0],
            "oracle_upper": oracle_bounds[1],
            "wrong_target_selected": float(_resolved_prediction(wrong) == target_action),
            "wrong_target_lower": wrong_bounds[0],
            "wrong_target_upper": wrong_bounds[1],
            "chain_minus_oracle_lower": min(
                0.0
                if candidate == target_slot
                else _target_bounds(rows[f"receipt_{candidate}"], target_action)[0]
                - oracle_bounds[1]
                for candidate in possible_slots
            ),
        }
    return record


def _endpoint_summary(
    records: list[dict[str, Any]], endpoint: str, spec: AuditSpec
) -> dict[str, Any]:
    base = mean(float(row["base"][endpoint]) for row in records)
    adapter = mean(float(row["adapter"][endpoint]) for row in records)
    difference = _cluster_interval(
        records,
        lambda row: float(row["adapter"][endpoint]) - float(row["base"][endpoint]),
        spec=spec,
        seed_label=endpoint,
    )
    return {
        "unit_count": len(records),
        "base": base,
        "adapter": adapter,
        "adapter_minus_base": difference,
    }


def _cluster_interval(
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    *,
    spec: AuditSpec,
    seed_label: str,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        number = float(value(row))
        if not math.isfinite(number):
            raise ValueError("non-finite handoff estimand")
        grouped[str(row["lesson_id"])].append(number)
    labels = sorted(grouped)
    generator = random.Random(_derived_seed(spec.bootstrap_seed, seed_label))
    estimates: list[float] = []
    for _ in range(spec.bootstrap_samples):
        sampled = [labels[generator.randrange(len(labels))] for _ in labels]
        estimates.append(mean(number for label in sampled for number in grouped[label]))
    return {
        "estimate": mean(value(row) for row in rows),
        "ci95": [_percentile(estimates, 0.025), _percentile(estimates, 0.975)],
        "confidence": spec.confidence,
        "cluster_unit": "lesson_id",
        "cluster_count": len(labels),
        "observation_count": len(rows),
        "bootstrap_samples": spec.bootstrap_samples,
        "bootstrap_seed": _derived_seed(spec.bootstrap_seed, seed_label),
    }


def _valid_outcome(row: dict[str, Any]) -> bool:
    candidates = row.get("candidates")
    if (
        row.get("non_finite") is not False
        or not isinstance(candidates, list)
        or not candidates
        or not _top_candidates(row)
    ):
        return False
    if row.get("tie"):
        return row.get("error_status") == "tie" and row.get("predicted_candidate") is None
    return row.get("error_status") == "ok" and row.get("predicted_candidate") in {
        candidate.get("candidate") for candidate in candidates
    }


def _target_bounds(row: dict[str, Any], target: str) -> tuple[float, float]:
    if not row.get("tie"):
        value = float(_resolved_prediction(row) == target)
        return value, value
    top = _top_candidates(row)
    return (float(top == {target}), float(target in top))


def _top_candidates(row: dict[str, Any]) -> set[str]:
    return {
        str(candidate["candidate"])
        for candidate in row.get("candidates", [])
        if candidate.get("sum_rank") == 1
    }


def _resolved_prediction(row: dict[str, Any]) -> str:
    if not row.get("tie"):
        predicted = row.get("predicted_candidate")
        if not isinstance(predicted, str):
            raise ValueError("non-tied outcome has no predicted candidate")
        return predicted
    top = _top_candidates(row)
    ordered = [str(candidate) for candidate in row.get("ordered_candidates", [])]
    resolved = next((candidate for candidate in ordered if candidate in top), None)
    if resolved is None:
        raise ValueError("tied outcome cannot be resolved from frozen candidate order")
    return resolved


def _require_static(rows: dict[str, dict[str, Any]]) -> None:
    static = ("probe_id", "lesson_id", "route_variant", "target_slot", "target_action")
    reference = rows["slot_readout"]
    if any(
        row.get(field) != reference.get(field)
        for row in rows.values()
        for field in static
    ):
        raise ValueError(f"handoff raw static fields differ for {reference.get('probe_id')}")


def _tie_count(rows: list[dict[str, Any]], kind: str) -> int:
    return sum(row.get("prompt_kind") == kind and bool(row.get("tie")) for row in rows)


def _derived_seed(seed: int, label: str) -> int:
    return seed + sum((index + 1) * ord(char) for index, char in enumerate(label))


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty bootstrap distribution")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
