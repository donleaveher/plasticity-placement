from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean
from typing import Any

from plasticity_placement.pathmem.scoring import (
    CandidateDistribution,
    jensen_shannon_nats,
    write_qualified,
)
from plasticity_placement.pathmem_ropcd_p0.config import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPECTED_ROWS,
    EXPECTED_UNITS,
    TECHNICAL_DEFECT_TOLERANCE_NATS,
    TECHNICAL_TOLERANCE,
)

REQUIRED_ARM_COUNTS = {
    "no_memory": 112,
    "p_latest": 112,
    "base_before": 224,
    "parametric_path": 224,
    "parametric_rescore": 224,
    "adapter_disabled": 224,
    "external_latest": 224,
    "icl_history": 224,
    "both": 224,
    "technical_duplicate": 56,
    "base_unrelated": 128,
    "parametric_unrelated": 128,
    "path_qualification": 64,
}


def summarize_p0(
    rows: list[dict[str, Any]],
    *,
    verified_units: int,
    integrity_checks: dict[str, bool],
) -> dict[str, Any]:
    by_arm = _rows_by_arm(rows)
    observed_counts = {arm: len(by_arm.get(arm, [])) for arm in REQUIRED_ARM_COUNTS}
    external_defects = _paired_item_defects(by_arm.get("external_latest", []))
    icl_defects = _paired_item_defects(by_arm.get("icl_history", []))
    path_defects = _paired_item_defects(by_arm.get("parametric_path", []))
    both_defects = _paired_item_defects(by_arm.get("both", []))
    duplicate_defects = _duplicate_item_defects(
        by_arm.get("parametric_path", []), by_arm.get("technical_duplicate", [])
    )
    external_upper = bootstrap_upper_95(external_defects)
    duplicate_upper = bootstrap_upper_95(duplicate_defects)
    rescore_delta = _paired_max_delta(
        by_arm.get("parametric_path", []), by_arm.get("parametric_rescore", [])
    )
    disable_delta = _paired_max_delta(
        by_arm.get("base_before", []), by_arm.get("adapter_disabled", [])
    )
    path_qualification = _path_qualification(by_arm.get("path_qualification", []))
    joint_pass = _joint_path_pass(path_qualification)
    unrelated = _unrelated_regression(
        by_arm.get("base_unrelated", []), by_arm.get("parametric_unrelated", [])
    )
    margins = _current_stale_margins(by_arm.get("parametric_path", []))
    arm_diagnostics = {
        arm: {
            "current_top1": mean(bool(row.get("correct")) for row in values),
            "obsolete_intrusion": mean(bool(row.get("obsolete_intrusion")) for row in values),
        }
        for arm, values in sorted(by_arm.items())
    }
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
        "arm_row_counts_exact": observed_counts == REQUIRED_ARM_COUNTS,
        "row_identities_unique": _row_ids_unique(rows),
        "all_prompts_untruncated": all(
            row.get("candidate_audit", {}).get("all_candidates_valid") is True for row in rows
        ),
        "all_rows_valid": not invalid_rows,
        "same_adapter_rescore_within_tolerance": rescore_delta <= TECHNICAL_TOLERANCE,
        "adapter_disable_within_tolerance": disable_delta <= TECHNICAL_TOLERANCE,
        "external_prompts_byte_identical": _external_prompts_identical(
            by_arm.get("external_latest", [])
        ),
        "external_defect_upper_95_below_tolerance": (
            external_upper < TECHNICAL_DEFECT_TOLERANCE_NATS
        ),
        "technical_duplicate_upper_95_below_tolerance": (
            duplicate_upper < TECHNICAL_DEFECT_TOLERANCE_NATS
        ),
        "path_write_qualification_reported_without_filtering": (
            len(path_qualification) == 16 and len(joint_pass) == 8
        ),
        "unrelated_panels_complete": len(unrelated["per_path_regression_pp"]) == 16,
    }
    overlap = set(base_integrity) & set(integrity_checks)
    if overlap:
        raise ValueError(f"duplicate R-OPCD P0 integrity checks: {sorted(overlap)}")
    all_integrity = {**base_integrity, **integrity_checks}
    gate_checks = [[name, passed] for name, passed in sorted(all_integrity.items())]
    blockers = [name for name, passed in gate_checks if not passed]
    return {
        "phase": "P0-R-OPCD",
        "scope": "engineering smoke only; no scientific path claim or epsilon change",
        "gate": {
            "gate": "G2",
            "passed": not blockers,
            "checks": gate_checks,
            "blockers": blockers,
        },
        "integrity_checks": all_integrity,
        "metrics": {
            "external_item_mean_defect_nats": mean(external_defects),
            "external_defect_upper_95_nats": external_upper,
            "icl_item_mean_defect_nats": mean(icl_defects),
            "parametric_item_mean_defect_nats": mean(path_defects),
            "both_item_mean_defect_nats": mean(both_defects),
            "technical_duplicate_item_mean_defect_nats": mean(duplicate_defects),
            "technical_duplicate_upper_95_nats": duplicate_upper,
            "same_adapter_rescore_max_candidate_delta": rescore_delta,
            "adapter_disable_max_candidate_delta": disable_delta,
            "path_write_qualification_rate": mean(path_qualification.values()),
            "joint_path_pass_rate": mean(joint_pass.values()),
            "unrelated_regression_pp": unrelated["aggregate_regression_pp"],
            "current_stale_margin_nats": mean(margins.values()),
            "invalid_row_count": len(invalid_rows),
        },
        "path_write_qualification": path_qualification,
        "joint_path_pass": joint_pass,
        "per_path_unrelated_regression_pp": unrelated["per_path_regression_pp"],
        "per_path_current_stale_margin_nats": margins,
        "arm_diagnostics": arm_diagnostics,
        "counts": {
            "rows": len(rows),
            "planned_units": EXPECTED_UNITS,
            "verified_units": verified_units,
            "by_arm": observed_counts,
        },
        "path_contrast_computed": True,
        "p1_authorized": False,
        "scientific_result_authorized": False,
    }


def bootstrap_upper_95(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("bootstrap inputs must be finite and non-negative")
    if len(values) == 1:
        return values[0]
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = sorted(
        mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    return estimates[math.ceil(0.95 * len(estimates)) - 1]


def _rows_by_arm(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["arm"])].append(row)
    return dict(grouped)


def _distribution(row: dict[str, Any]) -> CandidateDistribution:
    actions = row.get("ordered_actions")
    probabilities = row.get("candidate_probabilities")
    if not isinstance(actions, list) or not isinstance(probabilities, list):
        raise ValueError("R-OPCD P0 row lacks a candidate distribution")
    return CandidateDistribution(tuple(actions), tuple(float(value) for value in probabilities))


def _paired_item_defects(rows: list[dict[str, Any]]) -> list[float]:
    indexed = {
        (str(row["item_id"]), str(row["logical_name"]), str(row["probe_id"])): row for row in rows
    }
    defects: list[float] = []
    for item_id in sorted({str(row["item_id"]) for row in rows}):
        pair_means: list[float] = []
        for left_path, right_path in (("ABA", "BAA"), ("BAB", "ABB")):
            left = {
                probe_id: row
                for (candidate, path, probe_id), row in indexed.items()
                if candidate == item_id and path == left_path
            }
            right = {
                probe_id: row
                for (candidate, path, probe_id), row in indexed.items()
                if candidate == item_id and path == right_path
            }
            if len(left) != 14 or set(left) != set(right):
                raise ValueError(f"incomplete R-OPCD endpoint panel: {item_id}/{left_path}")
            pair_means.append(
                mean(
                    jensen_shannon_nats(
                        _distribution(left[probe_id]), _distribution(right[probe_id])
                    )
                    for probe_id in left
                )
            )
        defects.append(mean(pair_means))
    if len(defects) != 4:
        raise ValueError("R-OPCD P0 endpoint defect item count changed")
    return defects


def _duplicate_item_defects(
    originals: list[dict[str, Any]], duplicates: list[dict[str, Any]]
) -> list[float]:
    indexed = {
        (str(row["item_id"]), str(row["logical_name"]), str(row["probe_id"])): row
        for row in originals
    }
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in duplicates:
        key = (str(row["item_id"]), str(row["duplicate_of"]), str(row["probe_id"]))
        if key not in indexed:
            raise ValueError(f"R-OPCD P0 duplicate has no original: {key}")
        grouped[key[0]].append(jensen_shannon_nats(_distribution(indexed[key]), _distribution(row)))
    if len(grouped) != 4 or any(len(values) != 14 for values in grouped.values()):
        raise ValueError("R-OPCD P0 duplicate panels changed")
    return [mean(grouped[item_id]) for item_id in sorted(grouped)]


def _paired_max_delta(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    def key(row: dict[str, Any]) -> tuple[str, str]:
        return str(row["unit_id"]), str(row["probe_id"])

    first = {key(row): row for row in left}
    second = {key(row): row for row in right}
    if not first or len(first) != len(left) or set(first) != set(second):
        return math.inf
    return max(
        max(
            abs(a - b)
            for a, b in zip(
                first[item]["candidate_scores"], second[item]["candidate_scores"], strict=True
            )
        )
        for item in first
    )


def _path_qualification(rows: list[dict[str, Any]]) -> dict[str, bool]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("probe_category") == "qualification":
            grouped[(str(row["item_id"]), str(row["logical_name"]))].append(row)
    result: dict[str, bool] = {}
    for (item_id, path), values in sorted(grouped.items()):
        if len(values) != 4:
            raise ValueError(f"R-OPCD P0 path qualification panel changed: {item_id}/{path}")
        result[f"{item_id}:{path}"] = write_qualified(
            tuple(_distribution(row) for row in sorted(values, key=lambda row: row["probe_id"])),
            current_action=str(values[0]["expected_action"]),
            obsolete_action=str(values[0]["obsolete_action"]),
        )
    return result


def _joint_path_pass(path_pass: dict[str, bool]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    item_ids = sorted({key.rsplit(":", 1)[0] for key in path_pass})
    for item_id in item_ids:
        for left, right in (("ABA", "BAA"), ("BAB", "ABB")):
            result[f"{item_id}:{left}-{right}"] = bool(
                path_pass[f"{item_id}:{left}"] and path_pass[f"{item_id}:{right}"]
            )
    return result


def _unrelated_regression(
    base_rows: list[dict[str, Any]], parametric_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[str, str, str]:
        return str(row["item_id"]), str(row["logical_name"]), str(row["probe_id"])

    base = {key(row): row for row in base_rows}
    parametric = {key(row): row for row in parametric_rows}
    if len(base) != 128 or set(base) != set(parametric):
        raise ValueError("R-OPCD P0 unrelated pairing changed")
    grouped: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for item in base:
        grouped[f"{item[0]}:{item[1]}"].append(
            (bool(base[item]["correct"]), bool(parametric[item]["correct"]))
        )
    per_path = {
        path: 100.0
        * (mean(float(left) for left, _ in pairs) - mean(float(right) for _, right in pairs))
        for path, pairs in sorted(grouped.items())
    }
    return {
        "aggregate_regression_pp": 100.0
        * (
            mean(float(row["correct"]) for row in base_rows)
            - mean(float(row["correct"]) for row in parametric_rows)
        ),
        "per_path_regression_pp": per_path,
    }


def _current_stale_margins(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        obsolete = row.get("obsolete_action")
        if obsolete is None:
            continue
        distribution = _distribution(row)
        current_probability = distribution.probability(str(row["expected_action"]))
        obsolete_probability = distribution.probability(str(obsolete))
        grouped[f"{row['item_id']}:{row['logical_name']}"].append(
            math.log(max(current_probability, 1e-300)) - math.log(max(obsolete_probability, 1e-300))
        )
    if len(grouped) != 16:
        raise ValueError("R-OPCD P0 current/stale panels changed")
    return {key: mean(values) for key, values in sorted(grouped.items())}


def _external_prompts_identical(rows: list[dict[str, Any]]) -> bool:
    grouped: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        grouped[(str(row["item_id"]), str(row["terminal_state"]), str(row["probe_id"]))].add(
            str(row["raw_prompt_sha256"])
        )
    return bool(grouped) and all(len(hashes) == 1 for hashes in grouped.values())


def _row_ids_unique(rows: list[dict[str, Any]]) -> bool:
    identities = [row.get("row_id") for row in rows]
    return bool(identities) and None not in identities and len(identities) == len(set(identities))
