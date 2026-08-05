from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from statistics import mean, median
from typing import Any

from plasticity_placement.p0d2hrabd.config import (
    CATEGORIES,
    CELLS,
    FACTORS,
    DiagnosticSpec,
)


def analyze_error_topology(
    paired_records: list[dict[str, Any]],
    adapter_off_rows: list[dict[str, Any]],
    adapter_on_rows: list[dict[str, Any]],
    lesson_types: dict[str, str],
    spec: DiagnosticSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    cell_records, integrity = build_cell_records(
        paired_records,
        adapter_off_rows,
        adapter_on_rows,
        lesson_types,
        spec,
    )
    if integrity["all_checks_passed"] is not True:
        raise ValueError(f"RAB error-topology row reconstruction failed: {integrity['checks']}")
    unit_records = build_unit_records(cell_records, spec)
    correctness_transitions = Counter(row["correctness_transition"] for row in cell_records)
    margin_sign_transitions = Counter(row["margin_sign_transition"] for row in cell_records)
    taxonomy = {state: _taxonomy_summary(unit_records, state) for state in ("base", "adapter")}
    taxonomy_transition = Counter(
        f"{row['base']['category']}→{row['adapter']['category']}" for row in unit_records
    )
    margin_summary = {
        "base": _cluster_interval(
            cell_records,
            lambda row: float(row["base"]["selected_minus_counterfactual_margin"]),
            spec,
            "base_margin",
        ),
        "adapter": _cluster_interval(
            cell_records,
            lambda row: float(row["adapter"]["selected_minus_counterfactual_margin"]),
            spec,
            "adapter_margin",
        ),
        "adapter_minus_base": _cluster_interval(
            cell_records,
            lambda row: (
                float(row["adapter"]["selected_minus_counterfactual_margin"])
                - float(row["base"]["selected_minus_counterfactual_margin"])
            ),
            spec,
            "margin_difference",
        ),
        "median": {
            state: median(
                float(row[state]["selected_minus_counterfactual_margin"]) for row in cell_records
            )
            for state in ("base", "adapter")
        },
    }
    factor_slices = _factor_slices(cell_records)
    action_bias = _action_bias(cell_records)
    summary = {
        "schema_version": "p0d2hrabd-analysis-v1",
        "analysis_status": "descriptive_error_topology_complete",
        "integrity": integrity,
        "cell_count": len(cell_records),
        "unit_count": len(unit_records),
        "correctness_transitions": {
            label: correctness_transitions[label] for label in ("C→C", "C→W", "W→C", "W→W")
        },
        "margin_sign_transitions": dict(sorted(margin_sign_transitions.items())),
        "margin_summary": margin_summary,
        "unit_taxonomy": taxonomy,
        "unit_taxonomy_transition": dict(sorted(taxonomy_transition.items())),
        "correct_count_distribution": {
            state: dict(
                sorted(Counter(row[state]["correct_count"] for row in unit_records).items())
            )
            for state in ("base", "adapter")
        },
        "compliance": {
            state: {
                metric: mean(float(row[state][metric]) for row in unit_records)
                for metric in (
                    "receipt_pair_compliance",
                    "binding_swap_compliance",
                    "full_factorial_compliance",
                )
            }
            for state in ("base", "adapter")
        },
        "factor_slices": factor_slices,
        "action_bias": action_bias,
        "descriptive_only": True,
        "historical_rab_decision_changed": False,
        "inference_authorized": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
        "next_action": "review_error_topology_before_designing_any_remediation",
    }
    return summary, cell_records, unit_records


def build_cell_records(
    paired_records: list[dict[str, Any]],
    adapter_off_rows: list[dict[str, Any]],
    adapter_on_rows: list[dict[str, Any]],
    lesson_types: dict[str, str],
    spec: DiagnosticSpec,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paired = _unique_index(paired_records, "probe_id", "paired records")
    off = _unique_index(adapter_off_rows, "probe_id", "adapter OFF rows")
    on = _unique_index(adapter_on_rows, "probe_id", "adapter ON rows")
    checks = {
        "paired_record_count": len(paired) == spec.paired_record_count,
        "adapter_off_count": len(off) == spec.raw_row_count_per_state,
        "adapter_on_count": len(on) == spec.raw_row_count_per_state,
        "probe_id_sets_match": set(paired) == set(off) == set(on),
        "lesson_type_pairs_complete": len(lesson_types) == 12,
    }
    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, str]] = []
    if checks["probe_id_sets_match"]:
        for probe_id in sorted(paired):
            try:
                rows.append(
                    _cell_record(
                        paired[probe_id],
                        off[probe_id],
                        on[probe_id],
                        lesson_types,
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                mismatches.append({"probe_id": probe_id, "error": str(error)})
    unit_cells: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        unit_cells[row["unit_id"]][row["cell"]] += 1
    checks.update(
        {
            "row_reconstruction_complete": len(rows) == spec.paired_record_count and not mismatches,
            "unit_count": len(unit_cells) == spec.unit_count,
            "factorial_complete": all(
                counts == {cell: 1 for cell in CELLS} for counts in unit_cells.values()
            ),
            "transition_partition": len(rows)
            == sum(Counter(row["correctness_transition"] for row in rows).values()),
        }
    )
    return rows, {
        "schema_version": "p0d2hrabd-row-integrity-v1",
        "checks": checks,
        "mismatches": mismatches,
        "all_checks_passed": all(checks.values()),
    }


def build_unit_records(
    cell_records: list[dict[str, Any]], spec: DiagnosticSpec
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cell_records:
        grouped[row["unit_id"]].append(row)
    if len(grouped) != spec.unit_count:
        raise ValueError("RAB topology unit count differs from frozen source")
    records: list[dict[str, Any]] = []
    for unit_id, values in sorted(grouped.items()):
        by_cell = {row["cell"]: row for row in values}
        if set(by_cell) != set(CELLS) or len(values) != len(CELLS):
            raise ValueError(f"RAB topology unit factorial is incomplete: {unit_id}")
        for field in ("pair_id", "lesson_type", "route_variant"):
            if len({str(row[field]) for row in values}) != 1:
                raise ValueError(f"RAB topology unit has inconsistent {field}: {unit_id}")
        record: dict[str, Any] = {
            "unit_id": unit_id,
            "pair_id": values[0]["pair_id"],
            "lesson_type": values[0]["lesson_type"],
            "route_variant": values[0]["route_variant"],
        }
        for state in ("base", "adapter"):
            record[state] = _classify_unit(by_cell, state)
        record["taxonomy_transition"] = (
            f"{record['base']['category']}→{record['adapter']['category']}"
        )
        records.append(record)
    return records


def _cell_record(
    paired: dict[str, Any],
    off: dict[str, Any],
    on: dict[str, Any],
    lesson_types: dict[str, str],
) -> dict[str, Any]:
    static = (
        "probe_id",
        "unit_id",
        "pair_id",
        "route_variant",
        "orientation",
        "receipt",
        "cell",
        "expected_action",
        "counterfactual_action",
        "ordered_candidates",
    )
    if any(paired.get(field) != raw.get(field) for raw in (off, on) for field in static):
        raise ValueError("paired/raw static fields differ")
    if off.get("adapter_state") != "adapter_off" or on.get("adapter_state") != "adapter_on":
        raise ValueError("raw adapter states differ from file identity")
    pair_id = str(paired["pair_id"])
    if pair_id not in lesson_types:
        raise ValueError("lesson type is missing for pair")
    expected = str(paired["expected_action"])
    counterfactual = str(paired["counterfactual_action"])
    candidates = [str(value) for value in paired["ordered_candidates"]]
    if expected not in candidates or counterfactual not in candidates or expected == counterfactual:
        raise ValueError("selected/counterfactual candidate identity is invalid")
    record: dict[str, Any] = {
        field: paired[field] for field in static if field != "ordered_candidates"
    }
    record.update(
        {
            "ordered_candidates": candidates,
            "lesson_type": lesson_types[pair_id],
            "candidate_position": candidates.index(expected),
        }
    )
    for label, raw, paired_label in (
        ("base", off, "base"),
        ("adapter", on, "adapter"),
    ):
        candidate_index = _candidate_index(raw, candidates)
        resolved = _resolved_prediction(raw, candidates)
        source = paired[paired_label]
        if (
            resolved != source.get("predicted_action")
            or float(resolved == expected) != float(source.get("selected_correct"))
            or float(resolved == counterfactual) != float(source.get("counterfactual_selected"))
            or source.get("scoring_valid") is not True
        ):
            raise ValueError(f"paired/raw derived outcome differs for {label}")
        selected_score = float(candidate_index[expected]["sum_logprob"])
        counterfactual_score = float(candidate_index[counterfactual]["sum_logprob"])
        if not math.isfinite(selected_score) or not math.isfinite(counterfactual_score):
            raise ValueError("selected/counterfactual score is non-finite")
        record[label] = {
            "predicted_action": resolved,
            "selected_correct": float(resolved == expected),
            "counterfactual_selected": float(resolved == counterfactual),
            "other_action_selected": float(resolved not in {expected, counterfactual}),
            "choice_class": (
                "selected"
                if resolved == expected
                else "counterfactual"
                if resolved == counterfactual
                else "other"
            ),
            "selected_sum_logprob": selected_score,
            "counterfactual_sum_logprob": counterfactual_score,
            "selected_minus_counterfactual_margin": selected_score - counterfactual_score,
            "tie": bool(raw["tie"]),
            "top1_top2_margin": raw.get("top1_top2_margin"),
        }
        if label == "base":
            record["expected_token_count"] = int(candidate_index[expected]["token_count"])
        elif int(candidate_index[expected]["token_count"]) != record["expected_token_count"]:
            raise ValueError("OFF/ON expected candidate token counts differ")
    base_correct = bool(record["base"]["selected_correct"])
    adapter_correct = bool(record["adapter"]["selected_correct"])
    record["correctness_transition"] = (
        ("C" if base_correct else "W") + "→" + ("C" if adapter_correct else "W")
    )
    record["margin_sign_transition"] = (
        _margin_sign(record["base"]["selected_minus_counterfactual_margin"])
        + "→"
        + _margin_sign(record["adapter"]["selected_minus_counterfactual_margin"])
    )
    return record


def _classify_unit(by_cell: dict[str, dict[str, Any]], state: str) -> dict[str, Any]:
    rows = [by_cell[cell] for cell in CELLS]
    predictions = [row[state]["predicted_action"] for row in rows]
    correct = [bool(row[state]["selected_correct"]) for row in rows]
    counterfactual = [bool(row[state]["counterfactual_selected"]) for row in rows]
    tied = [bool(row[state]["tie"]) for row in rows]
    action_a = str(by_cell["canonical_A"]["expected_action"])
    action_b = str(by_cell["canonical_B"]["expected_action"])
    valid_actions = {action_a, action_b}
    if any(tied):
        category = "invalid_or_tied"
    elif any(prediction not in valid_actions for prediction in predictions):
        category = "other_action_intrusion"
    elif all(correct):
        category = "fully_compliant"
    elif all(counterfactual):
        category = "fully_inverted"
    elif len(set(predictions)) == 1:
        category = "single_action_locked"
    elif predictions[0] == predictions[1] and predictions[2] == predictions[3]:
        category = "receipt_invariant"
    elif predictions[0] == predictions[2] and predictions[1] == predictions[3]:
        category = "binding_invariant"
    else:
        category = "partial_mixed"
    signature = "".join(
        "a" if prediction == action_a else "b" if prediction == action_b else "o"
        for prediction in predictions
    )
    return {
        "category": category,
        "prediction_signature_CA_CB_SA_SB": signature,
        "expected_signature_CA_CB_SA_SB": "abba",
        "correct_count": sum(correct),
        "counterfactual_count": sum(counterfactual),
        "receipt_pair_compliance": mean(
            (
                float(correct[0] and correct[1]),
                float(correct[2] and correct[3]),
            )
        ),
        "binding_swap_compliance": mean(
            (
                float(correct[0] and correct[2]),
                float(correct[1] and correct[3]),
            )
        ),
        "full_factorial_compliance": float(all(correct)),
        "tie_count": sum(tied),
    }


def _factor_slices(cell_records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for factor in FACTORS:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in cell_records:
            grouped[str(row[factor])].append(row)
        result[factor] = [
            _slice_record(factor, level, rows)
            for level, rows in sorted(grouped.items(), key=lambda item: item[0])
        ]
    return result


def _slice_record(factor: str, level: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    transitions = Counter(row["correctness_transition"] for row in rows)
    base_margin = mean(float(row["base"]["selected_minus_counterfactual_margin"]) for row in rows)
    adapter_margin = mean(
        float(row["adapter"]["selected_minus_counterfactual_margin"]) for row in rows
    )
    return {
        "factor": factor,
        "level": level,
        "observation_count": len(rows),
        "base_accuracy": mean(float(row["base"]["selected_correct"]) for row in rows),
        "adapter_accuracy": mean(float(row["adapter"]["selected_correct"]) for row in rows),
        "adapter_minus_base_accuracy": mean(
            float(row["adapter"]["selected_correct"]) - float(row["base"]["selected_correct"])
            for row in rows
        ),
        "correctness_transitions": {
            label: transitions[label] for label in ("C→C", "C→W", "W→C", "W→W")
        },
        "base_mean_margin": base_margin,
        "adapter_mean_margin": adapter_margin,
        "adapter_minus_base_mean_margin": adapter_margin - base_margin,
    }


def _action_bias(cell_records: list[dict[str, Any]]) -> dict[str, Any]:
    actions = sorted({str(row["expected_action"]) for row in cell_records})
    expected = Counter(str(row["expected_action"]) for row in cell_records)
    return {
        "expected_counts": {action: expected[action] for action in actions},
        "states": {
            state: {
                "predicted_counts": dict(
                    sorted(
                        Counter(str(row[state]["predicted_action"]) for row in cell_records).items()
                    )
                ),
            }
            for state in ("base", "adapter")
        },
        "by_expected_action": [
            _slice_record(
                "expected_action",
                action,
                [row for row in cell_records if row["expected_action"] == action],
            )
            for action in actions
        ],
    }


def _taxonomy_summary(unit_records: list[dict[str, Any]], state: str) -> dict[str, Any]:
    counts = Counter(str(row[state]["category"]) for row in unit_records)
    error_counts = {
        category: counts[category] for category in CATEGORIES if category != "fully_compliant"
    }
    dominant = max(error_counts, key=lambda category: (error_counts[category], category))
    return {
        "counts": {category: counts[category] for category in CATEGORIES},
        "rates": {category: counts[category] / len(unit_records) for category in CATEGORIES},
        "dominant_noncompliant_category": dominant,
        "dominant_noncompliant_count": error_counts[dominant],
    }


def _candidate_index(
    raw: dict[str, Any], ordered_candidates: list[str]
) -> dict[str, dict[str, Any]]:
    candidates = raw.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("raw candidate scores are missing")
    names = [str(candidate.get("candidate")) for candidate in candidates]
    if names != ordered_candidates or len(set(names)) != 4:
        raise ValueError("raw candidate order differs from frozen paired row")
    if any(
        not isinstance(candidate.get(field), (int, float))
        or not math.isfinite(float(candidate[field]))
        for candidate in candidates
        for field in ("sum_logprob", "mean_logprob")
    ):
        raise ValueError("raw candidate scores are non-finite or malformed")
    return {str(candidate["candidate"]): candidate for candidate in candidates}


def _resolved_prediction(raw: dict[str, Any], ordered_candidates: list[str]) -> str:
    if raw.get("non_finite") is not False:
        raise ValueError("raw row contains non-finite scores")
    top = {
        str(candidate["candidate"])
        for candidate in raw["candidates"]
        if candidate.get("sum_rank") == 1
    }
    if not top:
        raise ValueError("raw row has no top candidate")
    candidates = raw["candidates"]
    maximum = max(float(candidate["sum_logprob"]) for candidate in candidates)
    score_top = {
        str(candidate["candidate"])
        for candidate in candidates
        if float(candidate["sum_logprob"]) == maximum
    }
    if top != score_top:
        raise ValueError("raw sum ranks differ from candidate scores")
    tied = len(top) > 1
    if bool(raw.get("tie")) != tied:
        raise ValueError("raw tie flag differs from top-ranked candidates")
    if tied:
        if raw.get("predicted_candidate") is not None or raw.get("error_status") != "tie":
            raise ValueError("raw tie contract is inconsistent")
        resolved = next((candidate for candidate in ordered_candidates if candidate in top), None)
        if resolved is None:
            raise ValueError("raw tie cannot be resolved from frozen order")
        return resolved
    if len(top) != 1 or raw.get("error_status") != "ok":
        raise ValueError("raw unique-winner contract is inconsistent")
    predicted = str(raw.get("predicted_candidate"))
    if predicted != next(iter(top)):
        raise ValueError("raw prediction differs from top-ranked candidate")
    return predicted


def _cluster_interval(
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    spec: DiagnosticSpec,
    seed_label: str,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        number = float(value(row))
        if not math.isfinite(number):
            raise ValueError("non-finite topology margin estimand")
        grouped[str(row["pair_id"])].append(number)
    labels = sorted(grouped)
    generator = random.Random(_derived_seed(spec.bootstrap_seed, seed_label))
    estimates = [
        mean(
            number
            for label in [labels[generator.randrange(len(labels))] for _ in labels]
            for number in grouped[label]
        )
        for _ in range(spec.bootstrap_samples)
    ]
    return {
        "estimate": mean(value(row) for row in rows),
        "ci95": [_percentile(estimates, 0.025), _percentile(estimates, 0.975)],
        "confidence": spec.confidence,
        "cluster_unit": "pair_id",
        "cluster_count": len(labels),
        "observation_count": len(rows),
        "bootstrap_samples": spec.bootstrap_samples,
        "bootstrap_seed": _derived_seed(spec.bootstrap_seed, seed_label),
    }


def _unique_index(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result = {str(row[key]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{label} contain duplicate {key}")
    return result


def _margin_sign(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


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
