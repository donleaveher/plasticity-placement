from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from statistics import mean
from typing import Any

from plasticity_placement.p0d2hrab.analysis import _top_candidates, _valid_outcome
from plasticity_placement.p0d2hrabc.config import (
    CANDIDATE_ROTATIONS,
    DISPLAY_ORDERS,
    ORIENTATIONS,
    RECEIPTS,
    AuditSpec,
)

Row = dict[str, Any]
Predicate = Callable[[Row], bool]


def analyze_counterbalancing(
    raw_rows: list[Row], spec: AuditSpec
) -> tuple[dict[str, Any], list[Row]]:
    records, integrity = build_paired_records(raw_rows, spec)
    if not integrity["all_checks_passed"]:
        return _failed_analysis(integrity), records

    main_effects: dict[str, Any] = {}
    definitions = {
        "receipt_B_minus_A": (
            lambda row: row["receipt"] == "B",
            lambda row: row["receipt"] == "A",
        ),
        "selected_display_second_minus_first": (
            lambda row: row["selected_display_position"] == 1,
            lambda row: row["selected_display_position"] == 0,
        ),
        "candidate_position_0_minus_1_2_3": (
            lambda row: row["expected_candidate_position"] == 0,
            lambda row: row["expected_candidate_position"] != 0,
        ),
    }
    for name, (numerator, denominator) in definitions.items():
        main_effects[name] = {
            state: _effect(records, state, numerator, denominator, spec, f"{name}_{state}")
            for state in ("base", "adapter")
        }
        main_effects[name]["adapter_excess"] = _state_difference_effect(
            records, numerator, denominator, spec, f"{name}_adapter_excess"
        )

    stratified = {
        "receipt_B_minus_A": _stratified_effects(
            records,
            "adapter",
            definitions["receipt_B_minus_A"],
            ("display_order", "expected_candidate_position"),
            spec,
            "receipt",
        ),
        "selected_display_second_minus_first": _stratified_effects(
            records,
            "adapter",
            definitions["selected_display_second_minus_first"],
            ("receipt", "expected_candidate_position"),
            spec,
            "display",
        ),
        "candidate_position_0_minus_1_2_3": _stratified_effects(
            records,
            "adapter",
            definitions["candidate_position_0_minus_1_2_3"],
            ("receipt", "display_order"),
            spec,
            "candidate",
        ),
    }

    interactions = {
        "receipt_x_selected_display": _binary_interaction(
            records,
            lambda row: int(row["receipt"] == "B"),
            lambda row: int(row["selected_display_position"] == 1),
            spec,
            "receipt_x_display",
        ),
        "receipt_x_candidate_position_0": _binary_interaction(
            records,
            lambda row: int(row["receipt"] == "B"),
            lambda row: int(row["expected_candidate_position"] == 0),
            spec,
            "receipt_x_candidate0",
        ),
        "selected_display_x_candidate_position_0": _binary_interaction(
            records,
            lambda row: int(row["selected_display_position"] == 1),
            lambda row: int(row["expected_candidate_position"] == 0),
            spec,
            "display_x_candidate0",
        ),
        "receipt_x_selected_display_x_candidate_position_0": _three_way_interaction(records, spec),
    }

    flags = {
        "receipt_or_slot_label": _persistent_penalty(main_effects["receipt_B_minus_A"]["adapter"]),
        "serial_position": _persistent_penalty(
            main_effects["selected_display_second_minus_first"]["adapter"]
        ),
        "candidate_position_0": _persistent_penalty(
            main_effects["candidate_position_0_minus_1_2_3"]["adapter"]
        ),
        "composition_interaction": any(
            _excludes_zero(values["accuracy_midpoint"]["ci95"]) for values in interactions.values()
        ),
    }
    supported = [name for name, value in flags.items() if value]
    if not supported:
        attribution = "no_identified_penalty"
    elif len(supported) == 1:
        attribution = supported[0]
    else:
        attribution = "multiple_mechanisms"

    state_summaries = {state: _state_summary(records, state, spec) for state in ("base", "adapter")}
    adapter_change = _cluster_interval(
        records,
        lambda row: row["adapter"]["correct_midpoint"] - row["base"]["correct_midpoint"],
        spec,
        "overall_adapter_minus_base_accuracy",
    )
    transition_counts = Counter(_transition(row["base"], row["adapter"]) for row in records)
    analysis = {
        "analysis_status": "counterbalancing_complete",
        "integrity": integrity,
        "state_summaries": state_summaries,
        "adapter_minus_base_accuracy_midpoint": adapter_change,
        "correctness_transitions": dict(sorted(transition_counts.items())),
        "main_effects": main_effects,
        "stratified_adapter_effects": stratified,
        "interactions": interactions,
        "causal_attribution": {
            "status": attribution,
            "flags": flags,
            "rule": (
                "A main-factor penalty is supported only when its adapter accuracy contrast "
                "and conservative identified interval are below zero in the balanced factorial; "
                "crossed strata are reported to expose heterogeneity. Composition interaction "
                "is supported "
                "when a pair-cluster bootstrap interval for a two- or three-way contrast "
                "excludes zero."
            ),
            "receipt_or_slot_label_note": (
                "Receipt A/B and slot label A/B remain intentionally linked; this audit "
                "separates their joint label-binding effect from serial display position, "
                "but cannot distinguish receipt-token bias from slot-label bias."
            ),
        },
        "tie_diagnostics": {
            "raw_tie_count": sum(bool(row.get("tie")) for row in raw_rows),
            "paired_prompt_with_any_tie_count": sum(
                row["base"]["tie"] or row["adapter"]["tie"] for row in records
            ),
            "ties_resolved_by_candidate_order": False,
            "accuracy_reported_as_identified_bounds": True,
        },
        "historical_rab_decision_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
        "next_action": "review_counterbalanced_attribution_before_any_remediation_training",
    }
    return analysis, records


def build_paired_records(raw_rows: list[Row], spec: AuditSpec) -> tuple[list[Row], dict[str, Any]]:
    index = {(str(row.get("probe_id")), str(row.get("adapter_state"))): row for row in raw_rows}
    mismatches: list[dict[str, str]] = []
    if len(index) != len(raw_rows):
        mismatches.append({"scope": "matrix", "error": "duplicate raw row key"})
    probe_ids = sorted({str(row.get("probe_id")) for row in raw_rows})
    records: list[Row] = []
    for probe_id in probe_ids:
        try:
            base = index[(probe_id, "adapter_off")]
            adapter = index[(probe_id, "adapter_on")]
            records.append(_paired_record(base, adapter))
        except (KeyError, TypeError, ValueError) as error:
            mismatches.append({"scope": probe_id, "error": str(error)})
    unit_factors: dict[str, Counter[tuple[str, str, str, int]]] = defaultdict(Counter)
    for row in records:
        unit_factors[str(row["unit_id"])][
            (
                str(row["orientation"]),
                str(row["receipt"]),
                str(row["display_order"]),
                int(row["candidate_rotation"]),
            )
        ] += 1
    expected = {
        (orientation, receipt, display, rotation)
        for orientation in ORIENTATIONS
        for receipt in RECEIPTS
        for display in DISPLAY_ORDERS
        for rotation in CANDIDATE_ROTATIONS
    }
    joint = Counter(
        (
            str(row["receipt"]),
            int(row["selected_display_position"]),
            int(row["expected_candidate_position"]),
        )
        for row in records
    )
    checks = {
        "raw_decision_count": len(raw_rows) == spec.raw_decision_count,
        "paired_prompt_count": len(records) == spec.prompt_count,
        "unique_raw_keys": len(index) == len(raw_rows),
        "all_rows_reconstructed": not mismatches,
        "unit_count": len(unit_factors) == spec.unit_count,
        "complete_unit_factorial": all(
            set(counts) == expected and all(count == 1 for count in counts.values())
            for counts in unit_factors.values()
        ),
        "joint_confounders_crossed": len(joint) == 16
        and all(count == 96 for count in joint.values()),
    }
    integrity = {
        "checks": checks,
        "mismatches": mismatches[:100],
        "mismatch_count": len(mismatches),
        "all_checks_passed": all(checks.values()),
    }
    return records, integrity


def _paired_record(base: Row, adapter: Row) -> Row:
    static = (
        "probe_id",
        "source_probe_id",
        "unit_id",
        "pair_id",
        "route_variant",
        "orientation",
        "receipt",
        "cell",
        "display_order",
        "selected_display_position",
        "candidate_rotation",
        "expected_candidate_position",
        "expected_action",
        "counterfactual_action",
        "ordered_candidates",
        "formatted_prompt_sha256",
    )
    differences = [field for field in static if base.get(field) != adapter.get(field)]
    if differences:
        raise ValueError(f"OFF/ON static fields differ: {differences}")
    if not _valid_outcome(base) or not _valid_outcome(adapter):
        raise ValueError("candidate scorer contract is invalid")
    record = {field: base[field] for field in static}
    record["base"] = _state_record(base)
    record["adapter"] = _state_record(adapter)
    return record


def _state_record(row: Row) -> Row:
    expected = str(row["expected_action"])
    counterfactual = str(row["counterfactual_action"])
    top = _top_candidates(row)
    correct_lower = float(top == {expected})
    correct_upper = float(expected in top)
    scores = {
        str(candidate["candidate"]): float(candidate["sum_logprob"])
        for candidate in row["candidates"]
    }
    return {
        "predicted_action": row.get("predicted_candidate"),
        "top_candidates": sorted(top),
        "tie": bool(row["tie"]),
        "correct_lower": correct_lower,
        "correct_upper": correct_upper,
        "correct_midpoint": (correct_lower + correct_upper) / 2.0,
        "selected_minus_counterfactual_margin": scores[expected] - scores[counterfactual],
    }


def _state_summary(records: list[Row], state: str, spec: AuditSpec) -> Row:
    return {
        "observation_count": len(records),
        "accuracy_identified_interval": [
            mean(row[state]["correct_lower"] for row in records),
            mean(row[state]["correct_upper"] for row in records),
        ],
        "accuracy_midpoint": _cluster_interval(
            records,
            lambda row: row[state]["correct_midpoint"],
            spec,
            f"{state}_accuracy",
        ),
        "selected_minus_counterfactual_margin": _cluster_interval(
            records,
            lambda row: row[state]["selected_minus_counterfactual_margin"],
            spec,
            f"{state}_margin",
        ),
    }


def _effect(
    records: list[Row],
    state: str,
    numerator: Predicate,
    denominator: Predicate,
    spec: AuditSpec,
    label: str,
) -> Row:
    accuracy = _group_contrast(
        records,
        numerator,
        denominator,
        lambda row: row[state]["correct_midpoint"],
        spec,
        f"{label}_accuracy",
    )
    lower = _group_mean(records, numerator, lambda row: row[state]["correct_lower"]) - _group_mean(
        records, denominator, lambda row: row[state]["correct_upper"]
    )
    upper = _group_mean(records, numerator, lambda row: row[state]["correct_upper"]) - _group_mean(
        records, denominator, lambda row: row[state]["correct_lower"]
    )
    margin = _group_contrast(
        records,
        numerator,
        denominator,
        lambda row: row[state]["selected_minus_counterfactual_margin"],
        spec,
        f"{label}_margin",
    )
    return {
        "contrast_direction": "numerator_minus_denominator; negative means numerator disadvantage",
        "numerator_count": sum(numerator(row) for row in records),
        "denominator_count": sum(denominator(row) for row in records),
        "accuracy_midpoint": accuracy,
        "accuracy_identified_interval": [lower, upper],
        "selected_minus_counterfactual_margin": margin,
        "conservative_negative_penalty": accuracy["ci95"][1] < 0.0 and upper < 0.0,
    }


def _state_difference_effect(
    records: list[Row],
    numerator: Predicate,
    denominator: Predicate,
    spec: AuditSpec,
    label: str,
) -> Row:
    def change(row: Row) -> float:
        return row["adapter"]["correct_midpoint"] - row["base"]["correct_midpoint"]

    return {
        "accuracy_midpoint": _group_contrast(
            records, numerator, denominator, change, spec, f"{label}_accuracy"
        ),
        "interpretation": (
            "adapter-minus-base change in numerator minus the same change in denominator"
        ),
    }


def _stratified_effects(
    records: list[Row],
    state: str,
    groups: tuple[Predicate, Predicate],
    strata: tuple[str, ...],
    spec: AuditSpec,
    label: str,
) -> dict[str, Row]:
    values = sorted({tuple(row[field] for field in strata) for row in records}, key=str)
    result: dict[str, Row] = {}
    for value in values:
        subset = [row for row in records if tuple(row[field] for field in strata) == value]
        key = "|".join(f"{field}={item}" for field, item in zip(strata, value, strict=True))
        result[key] = _effect(subset, state, groups[0], groups[1], spec, f"{label}_{key}")
    return result


def _binary_interaction(
    records: list[Row],
    first: Callable[[Row], int],
    second: Callable[[Row], int],
    spec: AuditSpec,
    label: str,
) -> Row:
    weights = {(1, 1): 1.0, (1, 0): -1.0, (0, 1): -1.0, (0, 0): 1.0}
    return {
        "accuracy_midpoint": _cell_contrast(
            records,
            lambda row: (first(row), second(row)),
            weights,
            lambda row: row["adapter"]["correct_midpoint"],
            spec,
            label,
        ),
        "contrast": "difference_in_differences",
    }


def _three_way_interaction(records: list[Row], spec: AuditSpec) -> Row:
    weights = {
        (receipt, display, candidate): (1.0 if (receipt + display + candidate) % 2 == 1 else -1.0)
        for receipt in (0, 1)
        for display in (0, 1)
        for candidate in (0, 1)
    }
    return {
        "accuracy_midpoint": _cell_contrast(
            records,
            lambda row: (
                int(row["receipt"] == "B"),
                int(row["selected_display_position"] == 1),
                int(row["expected_candidate_position"] == 0),
            ),
            weights,
            lambda row: row["adapter"]["correct_midpoint"],
            spec,
            "receipt_x_display_x_candidate0",
        ),
        "contrast": "third_order_difference",
    }


def _persistent_penalty(effect: Row) -> bool:
    return bool(effect["conservative_negative_penalty"])


def _group_contrast(
    rows: list[Row],
    numerator: Predicate,
    denominator: Predicate,
    value: Callable[[Row], float],
    spec: AuditSpec,
    label: str,
) -> Row:
    by_pair: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_id"])].append(row)
    effects: dict[str, float] = {}
    for pair_id, pair_rows in by_pair.items():
        numerator_values = [float(value(row)) for row in pair_rows if numerator(row)]
        denominator_values = [float(value(row)) for row in pair_rows if denominator(row)]
        if not numerator_values or not denominator_values:
            raise ValueError(f"contrast group is empty for pair {pair_id}")
        effects[pair_id] = mean(numerator_values) - mean(denominator_values)
    return _bootstrap_pair_effects(effects, spec, label)


def _cell_contrast(
    rows: list[Row],
    cell: Callable[[Row], tuple[int, ...]],
    weights: dict[tuple[int, ...], float],
    value: Callable[[Row], float],
    spec: AuditSpec,
    label: str,
) -> Row:
    by_pair: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_id"])].append(row)
    effects: dict[str, float] = {}
    for pair_id, pair_rows in by_pair.items():
        grouped: dict[tuple[int, ...], list[float]] = defaultdict(list)
        for row in pair_rows:
            grouped[cell(row)].append(float(value(row)))
        if set(grouped) != set(weights):
            raise ValueError(f"interaction cells are incomplete for pair {pair_id}")
        effects[pair_id] = sum(weights[key] * mean(grouped[key]) for key in weights)
    return _bootstrap_pair_effects(effects, spec, label)


def _bootstrap_pair_effects(effects: dict[str, float], spec: AuditSpec, label: str) -> Row:
    labels = sorted(effects)
    if len(labels) != spec.pair_count:
        raise ValueError("pair-cluster count differs from frozen design")
    generator = random.Random(_derived_seed(spec.bootstrap_seed, label))
    samples = [
        mean(effects[labels[generator.randrange(len(labels))]] for _ in labels)
        for _ in range(spec.bootstrap_samples)
    ]
    return {
        "estimate": mean(effects.values()),
        "ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "confidence": spec.confidence,
        "cluster_unit": "pair_id",
        "cluster_count": len(labels),
        "bootstrap_samples": spec.bootstrap_samples,
        "bootstrap_seed": _derived_seed(spec.bootstrap_seed, label),
    }


def _cluster_interval(
    rows: list[Row], value: Callable[[Row], float], spec: AuditSpec, label: str
) -> Row:
    by_pair: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_id"])].append(float(value(row)))
    return _bootstrap_pair_effects(
        {pair_id: mean(values) for pair_id, values in by_pair.items()}, spec, label
    )


def _group_mean(rows: list[Row], predicate: Predicate, value: Callable[[Row], float]) -> float:
    values = [float(value(row)) for row in rows if predicate(row)]
    if not values:
        raise ValueError("contrast group is empty")
    return mean(values)


def _transition(base: Row, adapter: Row) -> str:
    def label(state: Row) -> str:
        if state["correct_lower"] == state["correct_upper"] == 1.0:
            return "C"
        if state["correct_lower"] == state["correct_upper"] == 0.0:
            return "W"
        return "T"

    return f"{label(base)}→{label(adapter)}"


def _failed_analysis(integrity: Row) -> Row:
    return {
        "analysis_status": "scoring_integrity_failed",
        "integrity": integrity,
        "causal_attribution": {"status": "scoring_integrity_failed", "flags": {}},
        "historical_rab_decision_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
        "next_action": "repair_scoring_integrity_without_interpreting_factor_effects",
    }


def _excludes_zero(interval: list[float]) -> bool:
    return interval[1] < 0.0 or interval[0] > 0.0


def _derived_seed(seed: int, label: str) -> int:
    return seed + sum((index + 1) * ord(char) for index, char in enumerate(label))


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
