from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from plasticity_placement.p0d2hcbr.config import GateSpec
from plasticity_placement.p0d2hcbr.data import (
    CODEBOOKS,
    DISPLAY_ORDERS,
    ORIENTATIONS,
    RECEIPTS,
    ROTATIONS,
)
from plasticity_placement.p0d2hcpr.qualification import _tie_accuracy_bounds
from plasticity_placement.p0d2hrab.analysis import _top_candidates, _valid_outcome
from plasticity_placement.p0d2hrabc.analysis import (
    _cell_contrast,
    _cluster_interval,
    _effect,
    _state_difference_effect,
    _state_summary,
    _transition,
)
from plasticity_placement.p0d2hrr.paired_audit import _cluster_bootstrap

Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class FactorialSpec:
    pair_count: int = 24
    unit_count: int = 24
    conditions_per_unit: int = 64
    prompt_count: int = 1_536
    raw_decision_count: int = 3_072
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20260810
    confidence: float = 0.95


def analyze_factorial(
    raw_rows: list[Row], spec: FactorialSpec | None = None
) -> tuple[dict[str, Any], list[Row]]:
    spec = spec or FactorialSpec()
    records, integrity = build_factorial_records(raw_rows, spec)
    if not integrity["all_checks_passed"]:
        return {
            "analysis_status": "factorial_integrity_failed",
            "integrity": integrity,
        }, records
    definitions: dict[str, tuple[Callable[[Row], bool], Callable[[Row], bool]]] = {
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
    return {
        "analysis_status": "cbr_factorial_complete",
        "integrity": integrity,
        "state_summaries": {
            state: _state_summary(records, state, spec) for state in ("base", "adapter")
        },
        "adapter_minus_base_accuracy_midpoint": _cluster_interval(
            records,
            lambda row: row["adapter"]["correct_midpoint"] - row["base"]["correct_midpoint"],
            spec,
            "cbr_overall_adapter_minus_base",
        ),
        "correctness_transitions": dict(
            sorted(Counter(_transition(row["base"], row["adapter"]) for row in records).items())
        ),
        "main_effects": main_effects,
        "interactions": interactions,
        "tie_diagnostics": {
            "raw_tie_count": sum(bool(row.get("tie")) for row in raw_rows),
            "paired_prompt_with_any_tie_count": sum(
                row["base"]["tie"] or row["adapter"]["tie"] for row in records
            ),
            "ties_resolved_by_candidate_order": False,
        },
    }, records


def build_factorial_records(
    raw_rows: list[Row], spec: FactorialSpec
) -> tuple[list[Row], dict[str, Any]]:
    index = {(str(row.get("probe_id")), str(row.get("adapter_state"))): row for row in raw_rows}
    mismatches: list[dict[str, str]] = []
    if len(index) != len(raw_rows):
        mismatches.append({"scope": "matrix", "error": "duplicate raw row key"})
    records: list[Row] = []
    for probe_id in sorted({str(row.get("probe_id")) for row in raw_rows}):
        try:
            records.append(
                _paired_record(index[(probe_id, "adapter_off")], index[(probe_id, "adapter_on")])
            )
        except (KeyError, TypeError, ValueError) as error:
            mismatches.append({"scope": probe_id, "error": str(error)})
    by_unit: dict[str, Counter[tuple[str, str, str, str, int]]] = defaultdict(Counter)
    for row in records:
        by_unit[str(row["unit_id"])][
            (
                str(row["orientation"]),
                str(row["receipt"]),
                str(row["codebook"]),
                str(row["display_order"]),
                int(row["candidate_rotation"]),
            )
        ] += 1
    expected = set(
        (orientation, receipt, codebook, display, rotation)
        for orientation in ORIENTATIONS
        for receipt in RECEIPTS
        for codebook in CODEBOOKS
        for display in DISPLAY_ORDERS
        for rotation in ROTATIONS
    )
    joint = Counter(
        (
            row["receipt"],
            row["selected_slot_label"],
            row["selected_display_position"],
            row["expected_candidate_position"],
        )
        for row in records
    )
    checks = {
        "raw_decision_count": len(raw_rows) == spec.raw_decision_count,
        "paired_prompt_count": len(records) == spec.prompt_count,
        "unique_raw_keys": len(index) == len(raw_rows),
        "all_rows_reconstructed": not mismatches,
        "unit_count": len(by_unit) == spec.unit_count,
        "complete_unit_factorial": all(
            set(counts) == expected and set(counts.values()) == {1} for counts in by_unit.values()
        ),
        "joint_primary_factors_crossed": len(joint) == 32
        and set(joint.values()) == {spec.prompt_count // 32},
    }
    return records, {
        "checks": checks,
        "mismatches": mismatches[:100],
        "mismatch_count": len(mismatches),
        "all_checks_passed": all(checks.values()),
    }


def classify_unit(
    *,
    heldout: dict[str, Any],
    rab: dict[str, Any],
    paired: dict[str, Any],
    guardrail_pairs: list[Row],
    gates: GateSpec,
) -> dict[str, Any]:
    if heldout.get("analysis_status") != "cbr_factorial_complete":
        return _failed_decision("heldout_integrity_failed")
    adapter_state = heldout["state_summaries"]["adapter"]
    heldout_lower = float(adapter_state["accuracy_identified_interval"][0])
    factor_checks: dict[str, bool] = {}
    factor_intervals: dict[str, list[float]] = {}
    for name, effect in heldout["main_effects"].items():
        interval = [float(value) for value in effect["adapter"]["accuracy_midpoint"]["ci95"]]
        factor_intervals[name] = interval
        factor_checks[name] = (
            interval[0] >= -gates.factor_equivalence_margin
            and interval[1] <= gates.factor_equivalence_margin
        )
    rab_binding = rab["adapter_on_conservative_binding_accuracy"]
    fc = paired["forced_choice"]["external_conditional_route"]
    endpoints = paired["crd"]["by_endpoint"]
    retrieval = endpoints["retrieval_only"]
    combined = endpoints["combined"]
    fc_lower, _ = _tie_accuracy_bounds(fc, "adapter")
    retrieval_lower, _ = _tie_accuracy_bounds(retrieval, "adapter")
    combined_adapter_lower, _ = _tie_accuracy_bounds(combined, "adapter")
    _, combined_base_upper = _tie_accuracy_bounds(combined, "base")
    combined_tie_delta = combined_adapter_lower - combined_base_upper
    guardrail = _cluster_bootstrap(
        guardrail_pairs,
        lambda row: float(bool(row["base_correct"]) and not bool(row["adapter_correct"])),
        bootstrap_samples=gates.bootstrap_samples,
        bootstrap_seed=gates.bootstrap_seed + 701,
    )
    compatible_ties = all(
        (not row[f"{state}_tie"] or row[f"{state}_tie_expected_compatible"])
        and row[f"{state}_error_status"] in {"ok", "tie"}
        for row in guardrail_pairs
        for state in ("base", "adapter")
    )
    checks = {
        "scoring_integrity": compatible_ties,
        "heldout_tie_worst_accuracy": heldout_lower >= gates.binding_min_accuracy,
        "historical_rab_binding_accuracy": float(rab_binding["estimate"])
        >= gates.binding_min_accuracy,
        "all_four_factors_equivalent": all(factor_checks.values()),
        "conditional_route": fc_lower >= gates.conditional_route_min_accuracy,
        "retrieval_preserved": retrieval_lower >= gates.retrieval_only_min_accuracy,
        "combined_cluster_noninferior": combined["paired_accuracy_difference"]["ci95"][0]
        >= -gates.combined_noninferiority_margin,
        "combined_tie_noninferior": combined_tie_delta >= -gates.combined_noninferiority_margin,
        "guardrail_c_to_w": guardrail["ci95"][1] <= gates.guardrail_c_to_w_max,
    }
    if not checks["scoring_integrity"]:
        status = "scoring_integrity_failed"
    elif all(checks.values()):
        status = "binding_remediation_candidate_review_required"
    elif checks["heldout_tie_worst_accuracy"] and not (
        checks["combined_cluster_noninferior"] and checks["guardrail_c_to_w"]
    ):
        status = "remediation_with_behavioral_drift"
    else:
        status = "unit_not_qualified"
    return {
        "status": status,
        "checks": checks,
        "factor_equivalence_checks": factor_checks,
        "factor_intervals": factor_intervals,
        "values": {
            "heldout_tie_worst_accuracy": heldout_lower,
            "historical_rab_binding_accuracy": rab_binding["estimate"],
            "conditional_route_tie_worst_accuracy": fc_lower,
            "retrieval_tie_worst_accuracy": retrieval_lower,
            "combined_tie_worst_delta": combined_tie_delta,
            "guardrail_c_to_w": guardrail,
        },
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }


def _failed_decision(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "checks": {},
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
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
    records: list[Row], functions: tuple[Any, ...], spec: FactorialSpec, label: str
) -> Row:
    cells = _binary_cells(len(functions))
    weights = {cell: (1.0 if sum(cell) % 2 == len(cell) % 2 else -1.0) for cell in cells}
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
