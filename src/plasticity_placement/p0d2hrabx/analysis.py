from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

from plasticity_placement.p0d2hrab.analysis import _top_candidates, _valid_outcome
from plasticity_placement.p0d2hrabc.analysis import (
    _cell_contrast,
    _cluster_interval,
    _effect,
    _excludes_zero,
    _failed_analysis,
    _persistent_penalty,
    _state_difference_effect,
    _state_summary,
    _stratified_effects,
    _transition,
)
from plasticity_placement.p0d2hrabx.config import (
    CANDIDATE_ROTATIONS,
    CODEBOOKS,
    DISPLAY_ORDERS,
    ORIENTATIONS,
    RECEIPTS,
    AuditSpec,
)

Row = dict[str, Any]


def analyze_label_disentanglement(
    raw_rows: list[Row], spec: AuditSpec
) -> tuple[dict[str, Any], list[Row]]:
    records, integrity = build_paired_records(raw_rows, spec)
    if not integrity["all_checks_passed"]:
        return _failed_analysis(integrity), records

    definitions = {
        "receipt_B_minus_A": (
            lambda row: row["receipt"] == "B",
            lambda row: row["receipt"] == "A",
        ),
        "selected_slot_B_minus_A": (
            lambda row: row["selected_slot_label"] == "B",
            lambda row: row["selected_slot_label"] == "A",
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
    main_effects: dict[str, Any] = {}
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
            ("selected_slot_label", "selected_display_position", "expected_candidate_position"),
            spec,
            "receipt",
        ),
        "selected_slot_B_minus_A": _stratified_effects(
            records,
            "adapter",
            definitions["selected_slot_B_minus_A"],
            ("receipt", "selected_display_position", "expected_candidate_position"),
            spec,
            "slot_label",
        ),
        "selected_display_second_minus_first": _stratified_effects(
            records,
            "adapter",
            definitions["selected_display_second_minus_first"],
            ("receipt", "selected_slot_label", "expected_candidate_position"),
            spec,
            "display",
        ),
        "candidate_position_0_minus_1_2_3": _stratified_effects(
            records,
            "adapter",
            definitions["candidate_position_0_minus_1_2_3"],
            ("receipt", "selected_slot_label", "selected_display_position"),
            spec,
            "candidate",
        ),
    }

    factor_functions = {
        "receipt": lambda row: int(row["receipt"] == "B"),
        "slot_label": lambda row: int(row["selected_slot_label"] == "B"),
        "display": lambda row: int(row["selected_display_position"] == 1),
        "candidate0": lambda row: int(row["expected_candidate_position"] == 0),
    }
    interactions: dict[str, Any] = {}
    for order in (2, 3, 4):
        for names in combinations(factor_functions, order):
            key = "_x_".join(names)
            interactions[key] = _interaction(
                records,
                tuple(factor_functions[name] for name in names),
                spec,
                key,
            )

    flags = {
        "receipt_token": _persistent_penalty(main_effects["receipt_B_minus_A"]["adapter"]),
        "slot_label": _persistent_penalty(
            main_effects["selected_slot_B_minus_A"]["adapter"]
        ),
        "serial_position": _persistent_penalty(
            main_effects["selected_display_second_minus_first"]["adapter"]
        ),
        "candidate_position_0": _persistent_penalty(
            main_effects["candidate_position_0_minus_1_2_3"]["adapter"]
        ),
        "composition_interaction": any(
            _excludes_zero(values["accuracy_midpoint"]["ci95"])
            for values in interactions.values()
        ),
    }
    supported = [name for name, value in flags.items() if value]
    attribution = (
        "no_identified_penalty"
        if not supported
        else supported[0]
        if len(supported) == 1
        else "multiple_mechanisms"
    )
    state_summaries = {state: _state_summary(records, state, spec) for state in ("base", "adapter")}
    adapter_change = _cluster_interval(
        records,
        lambda row: row["adapter"]["correct_midpoint"] - row["base"]["correct_midpoint"],
        spec,
        "overall_adapter_minus_base_accuracy",
    )
    transition_counts = Counter(_transition(row["base"], row["adapter"]) for row in records)
    analysis = {
        "analysis_status": "label_disentanglement_complete",
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
                "A main-factor penalty requires a negative adapter midpoint confidence interval "
                "and a negative conservative identified interval. An interaction is supported "
                "when its pair-cluster midpoint interval excludes zero."
            ),
            "identifiability_note": (
                "The explicit canonical/crossed codebook independently balances receipt token "
                "and selected slot label. Effects remain specific to this rewritten routing "
                "interface and do not by themselves authorize training."
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
        "historical_rabc_attribution_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
        "next_action": "review_receipt_vs_slot_attribution_before_any_remediation_design",
    }
    return analysis, records


def build_paired_records(raw_rows: list[Row], spec: AuditSpec) -> tuple[list[Row], dict[str, Any]]:
    index = {(str(row.get("probe_id")), str(row.get("adapter_state"))): row for row in raw_rows}
    mismatches: list[dict[str, str]] = []
    if len(index) != len(raw_rows):
        mismatches.append({"scope": "matrix", "error": "duplicate raw row key"})
    records: list[Row] = []
    for probe_id in sorted({str(row.get("probe_id")) for row in raw_rows}):
        try:
            records.append(
                _paired_record(
                    index[(probe_id, "adapter_off")], index[(probe_id, "adapter_on")]
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            mismatches.append({"scope": probe_id, "error": str(error)})

    unit_factors: dict[str, Counter[tuple[str, str, str, str, int]]] = defaultdict(Counter)
    for row in records:
        unit_factors[str(row["unit_id"])][
            (
                str(row["orientation"]),
                str(row["receipt"]),
                str(row["codebook"]),
                str(row["display_order"]),
                int(row["candidate_rotation"]),
            )
        ] += 1
    expected = {
        (orientation, receipt, codebook, display, rotation)
        for orientation in ORIENTATIONS
        for receipt in RECEIPTS
        for codebook in CODEBOOKS
        for display in DISPLAY_ORDERS
        for rotation in CANDIDATE_ROTATIONS
    }
    joint = Counter(
        (
            str(row["receipt"]),
            str(row["selected_slot_label"]),
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
        "joint_primary_factors_crossed": len(joint) == 32
        and all(count == 96 for count in joint.values()),
    }
    return records, {
        "checks": checks,
        "mismatches": mismatches[:100],
        "mismatch_count": len(mismatches),
        "all_checks_passed": all(checks.values()),
    }


def _paired_record(base: Row, adapter: Row) -> Row:
    static = (
        "probe_id",
        "source_probe_id",
        "unit_id",
        "pair_id",
        "route_variant",
        "orientation",
        "receipt",
        "codebook",
        "selected_slot_label",
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
    lower = float(top == {expected})
    upper = float(expected in top)
    scores = {
        str(candidate["candidate"]): float(candidate["sum_logprob"])
        for candidate in row["candidates"]
    }
    return {
        "predicted_action": row.get("predicted_candidate"),
        "top_candidates": sorted(top),
        "tie": bool(row["tie"]),
        "correct_lower": lower,
        "correct_upper": upper,
        "correct_midpoint": (lower + upper) / 2.0,
        "selected_minus_counterfactual_margin": scores[expected] - scores[counterfactual],
    }


def _interaction(
    records: list[Row],
    functions: tuple[Any, ...],
    spec: AuditSpec,
    label: str,
) -> Row:
    weights = {
        cell: (1.0 if sum(cell) % 2 == len(cell) % 2 else -1.0)
        for cell in _binary_cells(len(functions))
    }
    return {
        "accuracy_midpoint": _cell_contrast(
            records,
            lambda row: tuple(function(row) for function in functions),
            weights,
            lambda row: row["adapter"]["correct_midpoint"],
            spec,
            label,
        ),
        "contrast": f"{len(functions)}-way factorial interaction",
    }


def _binary_cells(width: int) -> list[tuple[int, ...]]:
    if width == 0:
        return [()]
    return [(head, *tail) for head in (0, 1) for tail in _binary_cells(width - 1)]
