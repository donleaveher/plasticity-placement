from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from statistics import mean
from typing import Any

from plasticity_placement.p0d2hrab.config import CELLS, ORIENTATIONS, RECEIPTS, AuditSpec


def analyze_binding(raw_rows: list[dict[str, Any]], spec: AuditSpec) -> dict[str, Any]:
    records, scoring_integrity = build_binding_records(raw_rows, spec)
    endpoints = {
        endpoint: _endpoint_summary(records, endpoint, spec)
        for endpoint in (
            "selected_correct",
            "counterfactual_selected",
            "specificity",
        )
    }
    on_binding = _cluster_interval(
        records,
        lambda row: row["adapter"]["selected_lower"],
        spec=spec,
        seed_label="adapter_on_binding_lower",
    )
    on_specificity = _cluster_interval(
        records,
        lambda row: row["adapter"]["specificity_lower"],
        spec=spec,
        seed_label="adapter_on_specificity_lower",
    )
    noninferiority = _cluster_interval(
        records,
        lambda row: row["adapter"]["selected_lower"] - row["base"]["selected_upper"],
        spec=spec,
        seed_label="adapter_noninferiority",
    )
    orientation_bounds = {
        orientation: _cluster_interval(
            [row for row in records if row["orientation"] == orientation],
            lambda row: row["adapter"]["selected_lower"],
            spec=spec,
            seed_label=f"adapter_on_{orientation}_lower",
        )
        for orientation in ORIENTATIONS
    }
    receipt_bounds = {
        receipt: _cluster_interval(
            [row for row in records if row["receipt"] == receipt],
            lambda row: row["adapter"]["selected_lower"],
            spec=spec,
            seed_label=f"adapter_on_receipt_{receipt}_lower",
        )
        for receipt in RECEIPTS
    }
    compliance = {
        "receipt_pair": _compliance_summary(records, ("unit_id", "orientation"), spec),
        "binding_swap": _compliance_summary(records, ("unit_id", "receipt"), spec),
        "full_factorial": _compliance_summary(records, ("unit_id",), spec),
    }

    binding_gate = (
        scoring_integrity
        and on_binding["estimate"] >= spec.binding_accuracy_threshold
        and on_binding["ci95"][0] > spec.binding_accuracy_ci_floor
    )
    specificity_gate = (
        scoring_integrity
        and on_specificity["estimate"] >= spec.specificity_threshold
        and on_specificity["ci95"][0] > 0.0
    )
    orientation_gate = scoring_integrity and all(
        values["estimate"] >= spec.binding_accuracy_threshold
        for values in orientation_bounds.values()
    )
    noninferiority_gate = (
        scoring_integrity and noninferiority["ci95"][0] > -spec.adapter_noninferiority_margin
    )
    if not scoring_integrity:
        status = "scoring_integrity_failed"
    elif binding_gate and specificity_gate and orientation_gate and noninferiority_gate:
        status = "binding_supported"
    else:
        status = "binding_not_supported"
    return {
        "endpoint_summaries": endpoints,
        "adapter_on_conservative_binding_accuracy": {
            **on_binding,
            "accuracy_threshold": spec.binding_accuracy_threshold,
            "ci_floor": spec.binding_accuracy_ci_floor,
            "passed": binding_gate,
        },
        "adapter_on_conservative_specificity": {
            **on_specificity,
            "material_threshold": spec.specificity_threshold,
            "passed": specificity_gate,
        },
        "orientation_conservative_accuracy": {
            orientation: {
                **values,
                "accuracy_threshold": spec.binding_accuracy_threshold,
                "passed": scoring_integrity
                and values["estimate"] >= spec.binding_accuracy_threshold,
            }
            for orientation, values in orientation_bounds.items()
        },
        "receipt_conservative_accuracy": receipt_bounds,
        "adapter_on_minus_off_noninferiority": {
            **noninferiority,
            "noninferiority_margin": spec.adapter_noninferiority_margin,
            "passed": noninferiority_gate,
        },
        "compliance": compliance,
        "tie_diagnostics": {
            "tie_count": sum(bool(row.get("tie")) for row in raw_rows),
            "all_ties_propagated_to_bounds": scoring_integrity,
        },
        "decision": {
            "status": status,
            "scoring_integrity": scoring_integrity,
            "binding_accuracy_passed": binding_gate,
            "causal_specificity_passed": specificity_gate,
            "orientation_robustness_passed": orientation_gate,
            "adapter_noninferiority_passed": noninferiority_gate,
            "historical_rsh_decision_changed": False,
            "training_authorized": False,
            "mappings_per_adapter_authorized": False,
            "next_action": "review_binding_evidence_before_any_end_to_end_qualification",
        },
    }


def build_binding_records(
    raw_rows: list[dict[str, Any]], spec: AuditSpec
) -> tuple[list[dict[str, Any]], bool]:
    index = {(str(row["probe_id"]), str(row["adapter_state"])): row for row in raw_rows}
    if len(index) != len(raw_rows) or len(index) != spec.raw_decision_count:
        raise ValueError("receipt/action raw matrix is incomplete or duplicated")
    probe_ids = sorted({str(row["probe_id"]) for row in raw_rows})
    if len(probe_ids) != spec.prompt_count:
        raise ValueError("receipt/action prompt count differs from the frozen bank")
    records = [_derive_record(probe_id, index) for probe_id in probe_ids]
    cell_counts = Counter(str(row["cell"]) for row in records)
    unit_cells: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        unit_cells[str(row["unit_id"])][str(row["cell"])] += 1
    factorial_integrity = (
        cell_counts == {cell: 48 for cell in CELLS}
        and len(unit_cells) == spec.unit_count
        and all(counts == {cell: 1 for cell in CELLS} for counts in unit_cells.values())
    )
    return records, factorial_integrity and all(_valid_outcome(row) for row in raw_rows)


def _derive_record(
    probe_id: str,
    index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    base = index[(probe_id, "adapter_off")]
    adapter = index[(probe_id, "adapter_on")]
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
    if any(base.get(field) != adapter.get(field) for field in static):
        raise ValueError(f"receipt/action OFF/ON static fields differ: {probe_id}")
    expected = str(base["expected_action"])
    counterfactual = str(base["counterfactual_action"])
    record: dict[str, Any] = {field: base[field] for field in static}
    for row, label in ((base, "base"), (adapter, "adapter")):
        valid = _valid_outcome(row)
        selected = _candidate_bounds(row, expected) if valid else (0.0, 1.0)
        other = _candidate_bounds(row, counterfactual) if valid else (0.0, 1.0)
        predicted = _resolved_prediction(row) if valid else None
        record[label] = {
            "predicted_action": predicted,
            "scoring_valid": valid,
            "selected_correct": float(predicted == expected),
            "selected_lower": selected[0],
            "selected_upper": selected[1],
            "counterfactual_selected": float(predicted == counterfactual),
            "counterfactual_lower": other[0],
            "counterfactual_upper": other[1],
            "specificity": float(predicted == expected) - float(predicted == counterfactual),
            "specificity_lower": selected[0] - other[1],
            "specificity_upper": selected[1] - other[0],
        }
    return record


def _endpoint_summary(
    records: list[dict[str, Any]], endpoint: str, spec: AuditSpec
) -> dict[str, Any]:
    return {
        "observation_count": len(records),
        "base": mean(float(row["base"][endpoint]) for row in records),
        "adapter": mean(float(row["adapter"][endpoint]) for row in records),
        "adapter_minus_base": _cluster_interval(
            records,
            lambda row: row["adapter"][endpoint] - row["base"][endpoint],
            spec=spec,
            seed_label=f"endpoint_{endpoint}",
        ),
    }


def _compliance_summary(
    records: list[dict[str, Any]], group_fields: tuple[str, ...], spec: AuditSpec
) -> dict[str, Any]:
    groups: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[tuple(row[field] for field in group_fields)].append(row)
    rows: list[dict[str, Any]] = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        rows.append(
            {
                "pair_id": values[0]["pair_id"],
                "group": list(key),
                "base_observed": float(all(row["base"]["selected_correct"] for row in values)),
                "adapter_observed": float(
                    all(row["adapter"]["selected_correct"] for row in values)
                ),
                "base_lower": min(row["base"]["selected_lower"] for row in values),
                "base_upper": min(row["base"]["selected_upper"] for row in values),
                "adapter_lower": min(row["adapter"]["selected_lower"] for row in values),
            }
        )
    label = "_".join(group_fields)
    return {
        "group_fields": list(group_fields),
        "group_count": len(rows),
        "observed": {
            "base": mean(row["base_observed"] for row in rows),
            "adapter": mean(row["adapter_observed"] for row in rows),
        },
        "adapter_conservative": _cluster_interval(
            rows,
            lambda row: row["adapter_lower"],
            spec=spec,
            seed_label=f"compliance_{label}_adapter",
        ),
        "adapter_minus_base_conservative": _cluster_interval(
            rows,
            lambda row: row["adapter_lower"] - row["base_upper"],
            spec=spec,
            seed_label=f"compliance_{label}_difference",
        ),
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
            raise ValueError("non-finite receipt/action estimand")
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


def _valid_outcome(row: dict[str, Any]) -> bool:
    candidates = row.get("candidates")
    ordered = row.get("ordered_candidates")
    if (
        row.get("non_finite") is not False
        or not isinstance(candidates, list)
        or len(candidates) != 4
        or not isinstance(ordered, list)
        or len(ordered) != 4
        or len(set(map(str, ordered))) != 4
    ):
        return False
    names = [str(candidate.get("candidate")) for candidate in candidates]
    if names != [str(candidate) for candidate in ordered]:
        return False
    if any(
        not isinstance(candidate.get(field), (int, float))
        or not math.isfinite(float(candidate[field]))
        for candidate in candidates
        for field in ("sum_logprob", "mean_logprob")
    ):
        return False
    top = _top_candidates(row)
    maximum = max(float(candidate["sum_logprob"]) for candidate in candidates)
    score_top = {
        str(candidate["candidate"])
        for candidate in candidates
        if float(candidate["sum_logprob"]) == maximum
    }
    if not top or not top <= set(names) or top != score_top:
        return False
    tied = len(top) > 1
    if bool(row.get("tie")) != tied:
        return False
    if tied:
        return row.get("error_status") == "tie" and row.get("predicted_candidate") is None
    return (
        row.get("error_status") == "ok"
        and row.get("predicted_candidate") == next(iter(top))
    )


def _candidate_bounds(row: dict[str, Any], candidate: str) -> tuple[float, float]:
    if not row.get("tie"):
        value = float(_resolved_prediction(row) == candidate)
        return value, value
    top = _top_candidates(row)
    return float(top == {candidate}), float(candidate in top)


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
            raise ValueError("non-tied receipt/action outcome has no prediction")
        return predicted
    top = _top_candidates(row)
    ordered = [str(candidate) for candidate in row.get("ordered_candidates", [])]
    resolved = next((candidate for candidate in ordered if candidate in top), None)
    if resolved is None:
        raise ValueError("receipt/action tie cannot be resolved from frozen candidate order")
    return resolved


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
