from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from statistics import mean, median
from typing import Any

from plasticity_placement.p0d2hrabdl.config import (
    CATEGORIES,
    CELLS,
    EXPECTED_ADAPTER_TAXONOMY,
    EXPECTED_BASE_TAXONOMY,
    EXPECTED_TRANSITION_COUNTS,
    FACTORS,
    TRANSITIONS,
    LocalizationSpec,
)


def analyze_localization(
    summary: dict[str, Any],
    source_factor_slices: dict[str, Any],
    cell_records: list[dict[str, Any]],
    unit_records: list[dict[str, Any]],
    spec: LocalizationSpec,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    integrity = validate_source_rows(
        summary,
        source_factor_slices,
        cell_records,
        unit_records,
        spec,
    )
    if integrity["all_checks_passed"] is not True:
        raise ValueError(f"RAB localization source rows failed integrity: {integrity['checks']}")
    cell_localization = build_cell_transition_localization(cell_records)
    taxonomy_migrations = build_taxonomy_migrations(unit_records)
    margin_profiles = build_margin_transition_profiles(cell_records, spec)
    unit_hotspots = build_unit_hotspots(cell_records, unit_records)
    analysis = {
        "schema_version": "p0d2hrabdl-analysis-v1",
        "analysis_status": "descriptive_error_localization_complete",
        "integrity": integrity,
        "transition_totals": dict(EXPECTED_TRANSITION_COUNTS),
        "top_adverse_levels": {
            factor: values["adverse_ranking"][:3]
            for factor, values in cell_localization["factors"].items()
        },
        "taxonomy_focus": {
            "base_single_action_locked_destinations": taxonomy_migrations[
                "base_single_action_locked_destinations"
            ],
            "adapter_receipt_invariant_origins": taxonomy_migrations[
                "adapter_receipt_invariant_origins"
            ],
            "adapter_partial_mixed_origins": taxonomy_migrations["adapter_partial_mixed_origins"],
        },
        "margin_transition_profiles": margin_profiles["profiles"],
        "highest_adverse_units": unit_hotspots[:10],
        "descriptive_only": True,
        "historical_rab_decision_changed": False,
        "historical_rabd_status_changed": False,
        "inference_authorized": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
        "next_action": "review_localized_barriers_before_preregistering_any_intervention",
    }
    return analysis, cell_localization, taxonomy_migrations, margin_profiles, unit_hotspots


def validate_source_rows(
    summary: dict[str, Any],
    source_factor_slices: dict[str, Any],
    cell_records: list[dict[str, Any]],
    unit_records: list[dict[str, Any]],
    spec: LocalizationSpec,
) -> dict[str, Any]:
    cell_ids = [str(row.get("probe_id")) for row in cell_records]
    unit_ids = [str(row.get("unit_id")) for row in unit_records]
    source_analysis = summary.get("analysis", {})
    transitions = Counter(str(row.get("correctness_transition")) for row in cell_records)
    base_taxonomy = Counter(str(row.get("base", {}).get("category")) for row in unit_records)
    adapter_taxonomy = Counter(str(row.get("adapter", {}).get("category")) for row in unit_records)
    cells_by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cell_records:
        cells_by_unit[str(row.get("unit_id"))].append(row)
    unit_index = {str(row.get("unit_id")): row for row in unit_records}
    factor_checks = _validate_factor_slices(source_factor_slices, cell_records)
    finite_margins = all(
        math.isfinite(float(row[state]["selected_minus_counterfactual_margin"]))
        for row in cell_records
        for state in ("base", "adapter")
    )
    unit_join_consistent = set(cells_by_unit) == set(unit_index) and all(
        _unit_join_matches(unit_index[unit_id], rows) for unit_id, rows in cells_by_unit.items()
    )
    checks = {
        "source_run_id": summary.get("run_id") == spec.source_run_id,
        "source_status": source_analysis.get("analysis_status")
        == "descriptive_error_topology_complete",
        "cell_count": len(cell_records) == spec.cell_count,
        "cell_ids_unique": len(set(cell_ids)) == len(cell_ids),
        "unit_count": len(unit_records) == spec.unit_count,
        "unit_ids_unique": len(set(unit_ids)) == len(unit_ids),
        "pair_count": len({str(row.get("pair_id")) for row in cell_records}) == spec.pair_count,
        "transition_totals": {label: transitions[label] for label in TRANSITIONS}
        == EXPECTED_TRANSITION_COUNTS,
        "summary_transition_match": source_analysis.get("correctness_transitions")
        == EXPECTED_TRANSITION_COUNTS,
        "base_taxonomy": {category: base_taxonomy[category] for category in CATEGORIES}
        == EXPECTED_BASE_TAXONOMY,
        "adapter_taxonomy": {category: adapter_taxonomy[category] for category in CATEGORIES}
        == EXPECTED_ADAPTER_TAXONOMY,
        "summary_taxonomy_match": source_analysis.get("unit_taxonomy", {})
        .get("base", {})
        .get("counts")
        == EXPECTED_BASE_TAXONOMY
        and source_analysis.get("unit_taxonomy", {}).get("adapter", {}).get("counts")
        == EXPECTED_ADAPTER_TAXONOMY,
        "unit_cell_factorial": all(
            len(rows) == 4 and {str(row.get("cell")) for row in rows} == set(CELLS)
            for rows in cells_by_unit.values()
        ),
        "unit_join_consistent": unit_join_consistent,
        "finite_margins": finite_margins,
        "factor_names": set(source_factor_slices) == set(FACTORS),
        "factor_slices_consistent": factor_checks["all_checks_passed"],
        "historical_boundaries": summary.get("historical_rab_decision_changed") is False
        and summary.get("inference_authorized") is False
        and summary.get("training_authorized") is False
        and summary.get("mappings_per_adapter_authorized") is False,
    }
    return {
        "schema_version": "p0d2hrabdl-source-integrity-v1",
        "checks": checks,
        "factor_checks": factor_checks,
        "all_checks_passed": all(checks.values()),
    }


def build_cell_transition_localization(cell_records: list[dict[str, Any]]) -> dict[str, Any]:
    global_counts = Counter(str(row["correctness_transition"]) for row in cell_records)
    adverse_total = global_counts["C→W"] + global_counts["W→W"]
    result: dict[str, Any] = {
        "schema_version": "p0d2hrabdl-cell-localization-v1",
        "global_transition_counts": {label: global_counts[label] for label in TRANSITIONS},
        "adverse_definition": "C→W or W→W",
        "adverse_total": adverse_total,
        "factors": {},
    }
    for factor in FACTORS:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in cell_records:
            grouped[str(row[factor])].append(row)
        levels = [
            _transition_slice(factor, level, rows, global_counts, adverse_total)
            for level, rows in sorted(grouped.items())
        ]
        ranking = sorted(
            levels,
            key=lambda row: (
                -int(row["adverse_count"]),
                -int(row["correctness_transitions"]["C→W"]),
                str(row["level"]),
            ),
        )
        result["factors"][factor] = {
            "levels": levels,
            "adverse_ranking": [
                {
                    "rank": rank,
                    "level": row["level"],
                    "observation_count": row["observation_count"],
                    "adverse_count": row["adverse_count"],
                    "adverse_share": row["adverse_share_of_global"],
                    "C→W": row["correctness_transitions"]["C→W"],
                    "W→W": row["correctness_transitions"]["W→W"],
                }
                for rank, row in enumerate(ranking, start=1)
            ],
        }
    return result


def build_taxonomy_migrations(unit_records: list[dict[str, Any]]) -> dict[str, Any]:
    transitions = Counter(str(row["taxonomy_transition"]) for row in unit_records)
    matrix = {
        base: {adapter: transitions[f"{base}→{adapter}"] for adapter in CATEGORIES}
        for base in CATEGORIES
    }
    return {
        "schema_version": "p0d2hrabdl-taxonomy-migrations-v1",
        "matrix": matrix,
        "nonzero_transitions": [
            {"base": base, "adapter": adapter, "count": matrix[base][adapter]}
            for base in CATEGORIES
            for adapter in CATEGORIES
            if matrix[base][adapter]
        ],
        "base_single_action_locked_destinations": matrix["single_action_locked"],
        "adapter_receipt_invariant_origins": {
            base: matrix[base]["receipt_invariant"] for base in CATEGORIES
        },
        "adapter_partial_mixed_origins": {
            base: matrix[base]["partial_mixed"] for base in CATEGORIES
        },
    }


def build_margin_transition_profiles(
    cell_records: list[dict[str, Any]], spec: LocalizationSpec
) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for transition in TRANSITIONS:
        rows = [row for row in cell_records if row["correctness_transition"] == transition]
        profiles[transition] = _margin_profile(rows, spec, transition)
    return {
        "schema_version": "p0d2hrabdl-margin-profiles-v1",
        "profiles": profiles,
        "descriptive_only": True,
    }


def build_unit_hotspots(
    cell_records: list[dict[str, Any]], unit_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cell_records:
        grouped[str(row["unit_id"])].append(row)
    unit_index = {str(row["unit_id"]): row for row in unit_records}
    hotspots: list[dict[str, Any]] = []
    for unit_id, rows in grouped.items():
        transitions = Counter(str(row["correctness_transition"]) for row in rows)
        source = unit_index[unit_id]
        per_cell = [
            {
                "cell": row["cell"],
                "correctness_transition": row["correctness_transition"],
                "base_margin": row["base"]["selected_minus_counterfactual_margin"],
                "adapter_margin": row["adapter"]["selected_minus_counterfactual_margin"],
                "margin_change": row["adapter"]["selected_minus_counterfactual_margin"]
                - row["base"]["selected_minus_counterfactual_margin"],
                "expected_action": row["expected_action"],
                "candidate_position": row["candidate_position"],
            }
            for row in sorted(rows, key=lambda value: CELLS.index(str(value["cell"])))
        ]
        hotspots.append(
            {
                "unit_id": unit_id,
                "pair_id": source["pair_id"],
                "lesson_type": source["lesson_type"],
                "route_variant": source["route_variant"],
                "taxonomy_transition": source["taxonomy_transition"],
                "base_category": source["base"]["category"],
                "adapter_category": source["adapter"]["category"],
                "C→W": transitions["C→W"],
                "W→W": transitions["W→W"],
                "W→C": transitions["W→C"],
                "C→C": transitions["C→C"],
                "adverse_count": transitions["C→W"] + transitions["W→W"],
                "mean_margin_change": mean(float(row["margin_change"]) for row in per_cell),
                "cells": per_cell,
            }
        )
    hotspots.sort(
        key=lambda row: (
            -int(row["adverse_count"]),
            -int(row["C→W"]),
            -int(row["W→W"]),
            -int(row["W→C"]),
            str(row["unit_id"]),
        )
    )
    for rank, row in enumerate(hotspots, start=1):
        row["adverse_rank"] = rank
    return hotspots


def _transition_slice(
    factor: str,
    level: str,
    rows: list[dict[str, Any]],
    global_counts: Counter[str],
    adverse_total: int,
) -> dict[str, Any]:
    counts = Counter(str(row["correctness_transition"]) for row in rows)
    adverse = counts["C→W"] + counts["W→W"]
    base_margins = [float(row["base"]["selected_minus_counterfactual_margin"]) for row in rows]
    adapter_margins = [
        float(row["adapter"]["selected_minus_counterfactual_margin"]) for row in rows
    ]
    margin_changes = [
        adapter - base for base, adapter in zip(base_margins, adapter_margins, strict=True)
    ]
    return {
        "factor": factor,
        "level": level,
        "observation_count": len(rows),
        "correctness_transitions": {label: counts[label] for label in TRANSITIONS},
        "transition_rates": {label: counts[label] / len(rows) for label in TRANSITIONS},
        "adverse_count": adverse,
        "adverse_rate": adverse / len(rows),
        "adverse_share_of_global": adverse / adverse_total,
        "C→W_share_of_global": counts["C→W"] / global_counts["C→W"],
        "W→W_share_of_global": counts["W→W"] / global_counts["W→W"],
        "rescue_share_of_global": counts["W→C"] / global_counts["W→C"],
        "net_correctness_change": counts["W→C"] - counts["C→W"],
        "base_mean_margin": mean(base_margins),
        "adapter_mean_margin": mean(adapter_margins),
        "mean_margin_change": mean(margin_changes),
        "adapter_margin_sign_counts": _sign_counts(adapter_margins),
        "margin_change_sign_counts": _sign_counts(margin_changes),
        "margin_change_by_transition": {
            transition: _descriptive_transition_margin(rows, transition)
            for transition in TRANSITIONS
        },
    }


def _margin_profile(
    rows: list[dict[str, Any]], spec: LocalizationSpec, transition: str
) -> dict[str, Any]:
    base = [float(row["base"]["selected_minus_counterfactual_margin"]) for row in rows]
    adapter = [float(row["adapter"]["selected_minus_counterfactual_margin"]) for row in rows]
    changes = [after - before for before, after in zip(base, adapter, strict=True)]
    return {
        "observation_count": len(rows),
        "base_mean": mean(base),
        "base_median": median(base),
        "adapter_mean": mean(adapter),
        "adapter_median": median(adapter),
        "mean_change": mean(changes),
        "median_change": median(changes),
        "base_sign_counts": _sign_counts(base),
        "adapter_sign_counts": _sign_counts(adapter),
        "change_sign_counts": _sign_counts(changes),
        "mean_change_pair_cluster_ci": _cluster_interval(
            rows,
            lambda row: (
                float(row["adapter"]["selected_minus_counterfactual_margin"])
                - float(row["base"]["selected_minus_counterfactual_margin"])
            ),
            spec,
            f"transition_{transition}",
        ),
    }


def _validate_factor_slices(
    source_factor_slices: dict[str, Any], cell_records: list[dict[str, Any]]
) -> dict[str, Any]:
    differences: list[dict[str, Any]] = []
    for factor in FACTORS:
        expected: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in cell_records:
            expected[str(row[factor])].append(row)
        observed_rows = source_factor_slices.get(factor, [])
        observed = {str(row.get("level")): row for row in observed_rows}
        if len(observed) != len(observed_rows) or set(observed) != set(expected):
            differences.append({"factor": factor, "error": "level set or uniqueness differs"})
            continue
        for level, rows in expected.items():
            transitions = Counter(str(row["correctness_transition"]) for row in rows)
            source = observed[level]
            base_margin = mean(
                float(row["base"]["selected_minus_counterfactual_margin"]) for row in rows
            )
            adapter_margin = mean(
                float(row["adapter"]["selected_minus_counterfactual_margin"]) for row in rows
            )
            if (
                source.get("observation_count") != len(rows)
                or source.get("correctness_transitions")
                != {label: transitions[label] for label in TRANSITIONS}
                or not _close(source.get("base_mean_margin"), base_margin)
                or not _close(source.get("adapter_mean_margin"), adapter_margin)
                or not _close(
                    source.get("adapter_minus_base_mean_margin"),
                    adapter_margin - base_margin,
                )
            ):
                differences.append(
                    {
                        "factor": factor,
                        "level": level,
                        "error": "source slice counts or margins differ",
                    }
                )
    return {
        "checked_factors": list(FACTORS),
        "differences": differences,
        "all_checks_passed": not differences,
    }


def _unit_join_matches(unit: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    if any(
        str(row[field]) != str(unit[field])
        for row in rows
        for field in ("unit_id", "pair_id", "lesson_type", "route_variant")
    ):
        return False
    transitions = Counter(str(row["correctness_transition"]) for row in rows)
    return (
        sum(transitions.values()) == 4
        and unit.get("taxonomy_transition")
        == f"{unit.get('base', {}).get('category')}→{unit.get('adapter', {}).get('category')}"
    )


def _descriptive_transition_margin(rows: list[dict[str, Any]], transition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["correctness_transition"] == transition]
    if not selected:
        return {"observation_count": 0, "mean_change": None, "adapter_negative_count": 0}
    changes = [
        float(row["adapter"]["selected_minus_counterfactual_margin"])
        - float(row["base"]["selected_minus_counterfactual_margin"])
        for row in selected
    ]
    return {
        "observation_count": len(selected),
        "mean_change": mean(changes),
        "adapter_negative_count": sum(
            float(row["adapter"]["selected_minus_counterfactual_margin"]) < 0 for row in selected
        ),
    }


def _close(observed: Any, expected: float) -> bool:
    return isinstance(observed, (int, float)) and math.isclose(
        float(observed), expected, rel_tol=1e-12, abs_tol=1e-12
    )


def _cluster_interval(
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    spec: LocalizationSpec,
    seed_label: str,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        number = float(value(row))
        if not math.isfinite(number):
            raise ValueError("non-finite localization estimand")
        grouped[str(row["pair_id"])].append(number)
    labels = sorted(grouped)
    if not labels:
        raise ValueError("localization bootstrap has no pair clusters")
    seed = _derived_seed(spec.bootstrap_seed, seed_label)
    generator = random.Random(seed)
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
        "bootstrap_seed": seed,
    }


def _sign_counts(values: list[float]) -> dict[str, int]:
    counts = Counter(
        "positive" if value > 0 else "negative" if value < 0 else "zero" for value in values
    )
    return {label: counts[label] for label in ("positive", "zero", "negative")}


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
