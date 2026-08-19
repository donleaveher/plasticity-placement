from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable
from statistics import mean
from typing import Any

from plasticity_placement.pathmem.scoring import (
    CandidateDistribution,
    jensen_shannon_nats,
    write_qualified,
)
from plasticity_placement.pathmem_ropcd_p1.config import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONSEQUENCE_THRESHOLD,
    EPSILON_JS_NATS,
    EXPECTED_ITEMS,
    EXPECTED_ROWS,
    EXPECTED_UNITS,
    REQUIRED_ARM_COUNTS,
    TECHNICAL_DEFECT_TOLERANCE_NATS,
    TECHNICAL_TOLERANCE,
    TRAINING_SEEDS,
)


def summarize_p1(
    rows: list[dict[str, Any]],
    *,
    verified_units: int,
    integrity_checks: dict[str, bool],
) -> dict[str, Any]:
    by_arm = _rows_by_arm(rows)
    counts = {arm: len(by_arm.get(arm, [])) for arm in REQUIRED_ARM_COUNTS}
    endpoints = _endpoint_metrics(by_arm.get("parametric_path", []))
    external = _endpoint_metrics(by_arm.get("external_latest", []))
    icl = _endpoint_metrics(by_arm.get("icl_history", []))
    both = _endpoint_metrics(by_arm.get("both", []))
    consequence = _consequence_metrics(by_arm.get("parametric_path", []))
    duplicate = _duplicate_metrics(
        by_arm.get("parametric_path", []), by_arm.get("technical_duplicate", [])
    )
    utility = _utility_metrics(
        by_arm.get("parametric_path", []), by_arm.get("adapter_disabled", [])
    )
    unrelated = _unrelated_metrics(
        by_arm.get("base_unrelated", []), by_arm.get("parametric_unrelated", [])
    )
    qualification = _qualification_metrics(
        by_arm.get("path_qualification", []), by_arm.get("parametric_path", [])
    )
    rescore_delta = _paired_max_delta(
        by_arm.get("parametric_path", []), by_arm.get("parametric_rescore", [])
    )
    disable_delta = _paired_max_delta(
        by_arm.get("base_before", []), by_arm.get("adapter_disabled", [])
    )
    d_interval = item_bootstrap_interval(endpoints["per_item"], value="D")
    c_interval = item_bootstrap_interval(consequence["per_item"], value="C")
    external_interval = item_bootstrap_interval(external["per_item"], value="D")
    duplicate_interval = item_bootstrap_interval(duplicate["per_item"], value="D")
    decision = _decision(d_interval)
    invalid_rows = [
        row
        for row in rows
        if row.get("error_status") != "ok"
        or row.get("tie") is not False
        or row.get("non_finite") is not False
    ]
    base_integrity = {
        "planned_units_complete_no_extras": verified_units == EXPECTED_UNITS,
        "expected_row_count": len(rows) == EXPECTED_ROWS,
        "arm_row_counts_exact": counts == REQUIRED_ARM_COUNTS,
        "row_identities_unique": _row_ids_unique(rows),
        "all_rows_valid": not invalid_rows,
        "all_prompts_untruncated": all(
            row.get("candidate_audit", {}).get("all_candidates_valid") is True
            for row in rows
        ),
        "same_adapter_rescore_within_tolerance": rescore_delta <= TECHNICAL_TOLERANCE,
        "adapter_disable_within_tolerance": disable_delta <= TECHNICAL_TOLERANCE,
        "external_prompts_byte_identical": _external_prompts_identical(
            by_arm.get("external_latest", [])
        ),
        "external_defect_upper_95_below_tolerance": (
            external_interval["upper_95"] < TECHNICAL_DEFECT_TOLERANCE_NATS
        ),
        "technical_duplicate_upper_95_below_tolerance": (
            duplicate_interval["upper_95"] < TECHNICAL_DEFECT_TOLERANCE_NATS
        ),
        "path_write_qualification_complete_without_filtering": (
            len(qualification["per_cell_path"]) == EXPECTED_ITEMS * len(TRAINING_SEEDS) * 4
        ),
        "all_item_seed_cells_retained": len(endpoints["per_cell"]) == (
            EXPECTED_ITEMS * len(TRAINING_SEEDS)
        ),
    }
    overlap = set(base_integrity) & set(integrity_checks)
    if overlap:
        raise ValueError(f"duplicate R-OPCD P1 integrity checks: {sorted(overlap)}")
    all_integrity = {**base_integrity, **integrity_checks}
    blockers = sorted(name for name, passed in all_integrity.items() if not passed)
    gate_passed = not blockers
    return {
        "phase": "P1-R-OPCD",
        "scope": "12-item investment/kill test; not a submission-level result",
        "gate": {
            "gate": "G3",
            "passed": gate_passed,
            "checks": [[name, passed] for name, passed in sorted(all_integrity.items())],
            "blockers": blockers,
        },
        "decision": {
            **decision,
            "valid": gate_passed,
            "p1b_implementation_review_eligible": (
                gate_passed and decision["classification"] == "meaningful_violation"
            ),
            "p1b_authorized": False,
            "p2_authorized": False,
        },
        "primary": {
            "D_A_mean_nats": mean(value["D_A"] for value in endpoints["per_item"]),
            "D_B_mean_nats": mean(value["D_B"] for value in endpoints["per_item"]),
            "D_mean_nats": d_interval["mean"],
            "D_95_ci_nats": [d_interval["lower_95"], d_interval["upper_95"]],
            "epsilon_js_nats": EPSILON_JS_NATS,
            "per_item": endpoints["per_item"],
            "per_item_seed": endpoints["per_cell"],
            "leave_one_item_out": _leave_one_out(endpoints["per_item"], "D"),
        },
        "consequence": {
            "C_mean": c_interval["mean"],
            "C_95_ci": [c_interval["lower_95"], c_interval["upper_95"]],
            "threshold": CONSEQUENCE_THRESHOLD,
            "consequential": (
                gate_passed
                and decision["classification"] == "meaningful_violation"
                and c_interval["lower_95"] > CONSEQUENCE_THRESHOLD
            ),
            "per_item": consequence["per_item"],
        },
        "secondary": {
            "external_D_mean_nats": external_interval["mean"],
            "external_D_upper_95_nats": external_interval["upper_95"],
            "icl_D_mean_nats": mean(value["D"] for value in icl["per_item"]),
            "both_D_mean_nats": mean(value["D"] for value in both["per_item"]),
            "technical_duplicate_D_mean_nats": duplicate_interval["mean"],
            "technical_duplicate_D_upper_95_nats": duplicate_interval["upper_95"],
            "same_adapter_rescore_max_candidate_delta": rescore_delta,
            "adapter_disable_max_candidate_delta": disable_delta,
            "utility": utility,
            "unrelated": unrelated,
            "path_write_qualification_rate": mean(
                qualification["per_cell_path"].values()
            ),
            "joint_path_qualification_rate": mean(qualification["joint"].values()),
            "invalid_row_count": len(invalid_rows),
        },
        "path_write_qualification": qualification,
        "counts": {
            "rows": len(rows),
            "planned_units": EXPECTED_UNITS,
            "verified_units": verified_units,
            "by_arm": counts,
        },
        "intention_to_audit": True,
        "failed_paths_filtered": False,
        "kill_path_contrast_computed": True,
        "p1b_authorized": False,
        "p2_authorized": False,
    }


def item_bootstrap_interval(
    values: list[dict[str, Any]], *, value: str
) -> dict[str, float]:
    observations = [float(row[value]) for row in values]
    if len(observations) != EXPECTED_ITEMS or any(
        not math.isfinite(candidate) or candidate < 0 for candidate in observations
    ):
        raise ValueError("R-OPCD P1 bootstrap requires twelve finite item values")
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = sorted(
        mean(observations[rng.randrange(len(observations))] for _ in observations)
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    return {
        "mean": mean(observations),
        "lower_95": _percentile(estimates, 0.025),
        "upper_95": _percentile(estimates, 0.975),
    }


def _endpoint_metrics(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    indexed = _path_index(rows)
    per_cell: list[dict[str, Any]] = []
    items = sorted({str(row["item_id"]) for row in rows})
    for item_id in items:
        for seed in TRAINING_SEEDS:
            d_a = _pair_probe_mean(indexed, item_id, seed, "ABA", "BAA", _js)
            d_b = _pair_probe_mean(indexed, item_id, seed, "BAB", "ABB", _js)
            per_cell.append(
                {
                    "item_id": item_id,
                    "training_seed": seed,
                    "D_A": d_a,
                    "D_B": d_b,
                    "D": mean((d_a, d_b)),
                }
            )
    per_item = _average_seeds(per_cell, ("D_A", "D_B", "D"))
    return {"per_cell": per_cell, "per_item": per_item}


def _consequence_metrics(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    indexed = _path_index(rows)
    per_cell: list[dict[str, Any]] = []
    for item_id in sorted({str(row["item_id"]) for row in rows}):
        for seed in TRAINING_SEEDS:
            c_a = _pair_probe_mean(indexed, item_id, seed, "ABA", "BAA", _current_delta)
            c_b = _pair_probe_mean(indexed, item_id, seed, "BAB", "ABB", _current_delta)
            per_cell.append(
                {"item_id": item_id, "training_seed": seed, "C": mean((c_a, c_b))}
            )
    return {"per_cell": per_cell, "per_item": _average_seeds(per_cell, ("C",))}


def _duplicate_metrics(
    originals: list[dict[str, Any]], duplicates: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    indexed = _path_index(originals)
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in duplicates:
        key = (
            str(row["item_id"]),
            int(row["training_seed"]),
            str(row["duplicate_of"]),
            str(row["probe_id"]),
        )
        if key not in indexed:
            raise ValueError(f"R-OPCD P1 duplicate has no original: {key}")
        grouped[(key[0], key[1])].append(_js(indexed[key], row))
    if len(grouped) != EXPECTED_ITEMS * len(TRAINING_SEEDS) or any(
        len(values) != 14 for values in grouped.values()
    ):
        raise ValueError("R-OPCD P1 duplicate panels changed")
    per_cell = [
        {"item_id": item_id, "training_seed": seed, "D": mean(values)}
        for (item_id, seed), values in sorted(grouped.items())
    ]
    return {"per_cell": per_cell, "per_item": _average_seeds(per_cell, ("D",))}


def _utility_metrics(
    on_rows: list[dict[str, Any]], off_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    on = _path_index(on_rows)
    off = _path_index(off_rows)
    if set(on) != set(off):
        raise ValueError("R-OPCD P1 utility panels are not matched")
    grouped: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for key, row in on.items():
        grouped[key[:3]].append(_current_probability(row) - _current_probability(off[key]))
    per_path = {
        f"{item}:seed-{seed}:{path}": mean(values)
        for (item, seed, path), values in sorted(grouped.items())
    }
    asymmetries: dict[str, dict[str, float]] = {}
    for item in sorted({key[0] for key in grouped}):
        for seed in TRAINING_SEEDS:
            values = {
                path: mean(grouped[(item, seed, path)])
                for path in ("ABA", "BAA", "BAB", "ABB")
            }
            asymmetries[f"{item}:seed-{seed}"] = {
                "A_path_A": values["ABA"] - values["BAA"],
                "A_path_B": values["BAB"] - values["ABB"],
            }
    return {"per_cell_path_U": per_path, "per_cell_asymmetry": asymmetries}


def _unrelated_metrics(
    base: list[dict[str, Any]], parametric: list[dict[str, Any]]
) -> dict[str, Any]:
    first = _simple_index(base)
    second = _simple_index(parametric)
    if set(first) != set(second):
        raise ValueError("R-OPCD P1 unrelated panels are not matched")
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for key in first:
        grouped[(key[0], key[1])].append(
            100.0 * (float(second[key]["correct"]) - float(first[key]["correct"]))
        )
    per_cell = {
        f"{item}:seed-{seed}": mean(values)
        for (item, seed), values in sorted(grouped.items())
    }
    return {"aggregate_regression_pp": mean(per_cell.values()), "per_cell_regression_pp": per_cell}


def _qualification_metrics(
    rows: list[dict[str, Any]], references: list[dict[str, Any]]
) -> dict[str, Any]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["item_id"]), int(row["training_seed"]), str(row["logical_name"]))
        grouped[key].append(row)
    reference: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in references:
        key = (str(row["item_id"]), int(row["training_seed"]), str(row["logical_name"]))
        reference[key].append(row)
    per_path: dict[str, bool] = {}
    for key, values in sorted(grouped.items()):
        if len(values) != 4 or len(reference[key]) != 14:
            raise ValueError(f"R-OPCD P1 qualification panel changed: {key}")
        obsolete = str(reference[key][0]["obsolete_action"])
        per_path[f"{key[0]}:seed-{key[1]}:{key[2]}"] = write_qualified(
            [_distribution(row) for row in values],
            current_action=str(values[0]["expected_action"]),
            obsolete_action=obsolete,
        )
    joint: dict[str, bool] = {}
    for item in sorted({key[0] for key in grouped}):
        for seed in TRAINING_SEEDS:
            for label, paths in (("A", ("ABA", "BAA")), ("B", ("BAB", "ABB"))):
                joint[f"{item}:seed-{seed}:{label}"] = all(
                    per_path[f"{item}:seed-{seed}:{path}"] for path in paths
                )
    return {"per_cell_path": per_path, "joint": joint}


def _path_index(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    indexed = {
        (
            str(row["item_id"]),
            int(row["training_seed"]),
            str(row.get("duplicate_of", row["logical_name"])),
            str(row["probe_id"]),
        ): row
        for row in rows
    }
    if len(indexed) != len(rows):
        raise ValueError("R-OPCD P1 path rows are duplicated")
    return indexed


def _simple_index(rows: list[dict[str, Any]]) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    return {
        (
            str(row["item_id"]),
            int(row["training_seed"]),
            str(row["unit_id"]),
            str(row["probe_id"]),
        ): row
        for row in rows
    }


def _pair_probe_mean(
    indexed: dict[tuple[str, int, str, str], dict[str, Any]],
    item: str,
    seed: int,
    left: str,
    right: str,
    metric: Callable[[dict[str, Any], dict[str, Any]], float],
) -> float:
    left_rows = {key[3]: row for key, row in indexed.items() if key[:3] == (item, seed, left)}
    right_rows = {key[3]: row for key, row in indexed.items() if key[:3] == (item, seed, right)}
    if len(left_rows) != 14 or set(left_rows) != set(right_rows):
        raise ValueError(f"incomplete R-OPCD P1 endpoint panel: {item}/seed-{seed}/{left}")
    return mean(metric(left_rows[probe], right_rows[probe]) for probe in left_rows)


def _average_seeds(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in sorted({str(row["item_id"]) for row in rows}):
        selected = [row for row in rows if row["item_id"] == item]
        observed_seeds = {row["training_seed"] for row in selected}
        if len(selected) != len(TRAINING_SEEDS) or observed_seeds != set(TRAINING_SEEDS):
            raise ValueError(f"R-OPCD P1 seed aggregation changed: {item}")
        result.append(
            {
                "item_id": item,
                **{
                    field: mean(float(row[field]) for row in selected)
                    for field in fields
                },
            }
        )
    if len(result) != EXPECTED_ITEMS:
        raise ValueError("R-OPCD P1 item aggregation changed")
    return result


def _distribution(row: dict[str, Any]) -> CandidateDistribution:
    return CandidateDistribution(
        tuple(row["ordered_actions"]),
        tuple(float(value) for value in row["candidate_probabilities"]),
    )


def _js(left: dict[str, Any], right: dict[str, Any]) -> float:
    return jensen_shannon_nats(_distribution(left), _distribution(right))


def _current_probability(row: dict[str, Any]) -> float:
    return _distribution(row).probability(str(row["expected_action"]))


def _current_delta(left: dict[str, Any], right: dict[str, Any]) -> float:
    return abs(_current_probability(left) - _current_probability(right))


def _paired_max_delta(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    first = _simple_index(left)
    second = _simple_index(right)
    if not first or len(first) != len(left) or set(first) != set(second):
        return math.inf
    return max(
        max(
            abs(a - b)
            for a, b in zip(
                first[key]["candidate_scores"],
                second[key]["candidate_scores"],
                strict=True,
            )
        )
        for key in first
    )


def _external_prompts_identical(rows: list[dict[str, Any]]) -> bool:
    grouped: dict[tuple[str, int, str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (
            str(row["item_id"]),
            int(row["training_seed"]),
            str(row["terminal_state"]),
            str(row["probe_id"]),
        )
        grouped[key].add(str(row["raw_prompt_sha256"]))
    return bool(grouped) and all(len(values) == 1 for values in grouped.values())


def _decision(interval: dict[str, float]) -> dict[str, Any]:
    if interval["lower_95"] > EPSILON_JS_NATS:
        classification = "meaningful_violation"
    elif interval["upper_95"] < EPSILON_JS_NATS:
        classification = "approximate_consistency"
    else:
        classification = "inconclusive"
    return {
        "classification": classification,
        "rule": "lower_95>epsilon meaningful; upper_95<epsilon approximate; else inconclusive",
    }


def _leave_one_out(values: list[dict[str, Any]], field: str) -> dict[str, float]:
    return {
        str(removed["item_id"]): mean(float(row[field]) for row in values if row is not removed)
        for removed in values
    }


def _rows_by_arm(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["arm"])].append(row)
    return dict(grouped)


def _row_ids_unique(rows: list[dict[str, Any]]) -> bool:
    identities = [row.get("row_id") for row in rows]
    return bool(identities) and None not in identities and len(identities) == len(set(identities))


def _percentile(sorted_values: list[float], probability: float) -> float:
    index = max(0, min(len(sorted_values) - 1, math.ceil(probability * len(sorted_values)) - 1))
    return sorted_values[index]
