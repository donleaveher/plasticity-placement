from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable
from statistics import mean
from typing import Any

from plasticity_placement.pathmem.gates import evaluate_g1_gate
from plasticity_placement.pathmem.scoring import (
    TECHNICAL_TOLERANCE_JS_NATS,
    CandidateDistribution,
    jensen_shannon_nats,
)
from plasticity_placement.pathmem_exec.scoring import maximum_candidate_delta

G1_EXPECTED_ATTEMPTS = 24
G1_EXPECTED_ROWS = 960
P0_EXPECTED_ARTIFACTS = 44
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260816


def summarize_g1(
    rows: list[dict[str, Any]],
    *,
    verified_attempts: int,
) -> dict[str, Any]:
    by_arm = _rows_by_arm(rows)
    required_counts = {
        "answer_copy": 96,
        "external_latest": 96,
        "bare_base": 96,
        "parametric_current": 96,
        "parametric_rescore": 96,
        "adapter_disabled": 96,
        "base_unrelated": 192,
        "parametric_unrelated": 192,
    }
    observed_counts = {arm: len(by_arm.get(arm, ())) for arm in required_counts}
    parametric = by_arm.get("parametric_current", [])
    per_action = {
        action: _accuracy(row for row in parametric if row["expected_action"] == action)
        for action in sorted({str(row["expected_action"]) for row in parametric})
    }
    metrics = {
        "answer_copy_top1": _accuracy(by_arm.get("answer_copy", [])),
        "parser_validity": _rate(bool(row.get("parser_valid")) for row in rows),
        "external_current_top1": _accuracy(by_arm.get("external_latest", [])),
        "external_obsolete_intrusion": _rate(
            bool(row.get("obsolete_intrusion")) for row in by_arm.get("external_latest", [])
        ),
        "parametric_current_top1": _accuracy(parametric),
        "parametric_per_action_min_top1": min(per_action.values(), default=0.0),
        "parametric_unrelated_regression_pp": 100.0
        * (
            _accuracy(by_arm.get("base_unrelated", []))
            - _accuracy(by_arm.get("parametric_unrelated", []))
        ),
        "adapter_disable_max_candidate_delta": _paired_max_delta(
            by_arm.get("bare_base", []),
            by_arm.get("adapter_disabled", []),
        ),
    }
    same_adapter_rescore_delta = _paired_max_delta(
        by_arm.get("parametric_current", []),
        by_arm.get("parametric_rescore", []),
    )
    integrity = {
        "planned_attempts_complete": verified_attempts == G1_EXPECTED_ATTEMPTS,
        "expected_row_count": len(rows) == G1_EXPECTED_ROWS,
        "arm_row_counts_exact": observed_counts == required_counts,
        "result_identities_unique": _result_ids_unique(rows),
        "all_prompts_untruncated": all(
            bool(row.get("candidate_audit", {}).get("all_candidates_valid")) for row in rows
        ),
        "model_revision_locked": _single_value(rows, "model_revision"),
        "precision_locked": {row.get("evaluation_precision") for row in rows} == {"nf4-bfloat16"},
        "same_adapter_rescore_within_tolerance": same_adapter_rescore_delta <= 1e-5,
    }
    gate = evaluate_g1_gate(metrics).to_dict()
    return {
        "phase": "G1",
        "scope": "interface qualification only; no path contrast is computed",
        "metrics": metrics,
        "per_action_top1": per_action,
        "gate": gate,
        "integrity_checks": integrity,
        "integrity_passed": all(integrity.values()),
        "authorization_ready": bool(gate["passed"] and all(integrity.values())),
        "counts": {
            "rows": len(rows),
            "verified_attempts": verified_attempts,
            "by_arm": observed_counts,
        },
        "diagnostics": {"same_adapter_rescore_max_candidate_delta": same_adapter_rescore_delta},
    }


def summarize_p0(
    rows: list[dict[str, Any]],
    *,
    verified_artifacts: int,
    planned_artifacts: int,
    lineage_verified: bool,
    exposure_verified: bool,
) -> dict[str, Any]:
    by_arm = _rows_by_arm(rows)
    required_counts = {
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
    }
    observed_counts = {arm: len(by_arm.get(arm, ())) for arm in required_counts}
    external_item_defects = _paired_item_defects(
        by_arm.get("external_latest", []),
        pairs=("ABA:BAA", "BAB:ABB"),
    )
    duplicate_item_defects = _duplicate_item_defects(
        by_arm.get("parametric_path", []),
        by_arm.get("technical_duplicate", []),
    )
    external_upper = bootstrap_upper_95(external_item_defects)
    duplicate_upper = bootstrap_upper_95(duplicate_item_defects)
    rescore_delta = _paired_max_delta(
        by_arm.get("parametric_path", []),
        by_arm.get("parametric_rescore", []),
    )
    disable_delta = _paired_max_delta(
        by_arm.get("base_before", []),
        by_arm.get("adapter_disabled", []),
    )
    invalid_rows = [row for row in rows if row.get("error_status") != "ok"]
    integrity = {
        "planned_units_complete_no_extras": (
            planned_artifacts == P0_EXPECTED_ARTIFACTS
            and verified_artifacts == P0_EXPECTED_ARTIFACTS
        ),
        "arm_row_counts_exact": observed_counts == required_counts,
        "lineage_and_environment_hashes_verified": lineage_verified,
        "token_data_step_exposure_exact": exposure_verified,
        "same_adapter_rescore_within_tolerance": rescore_delta <= 1e-5,
        "external_prompts_byte_identical": _external_prompts_identical(
            by_arm.get("external_latest", [])
        ),
        "external_defect_upper_95_below_tolerance": (external_upper < TECHNICAL_TOLERANCE_JS_NATS),
        "technical_duplicate_upper_95_below_tolerance": (
            duplicate_upper < TECHNICAL_TOLERANCE_JS_NATS
        ),
        "adapter_disable_within_tolerance": disable_delta <= 1e-5,
        "invalid_tie_error_nonfinite_zero": not invalid_rows,
        "result_identities_unique": _result_ids_unique(rows),
        "all_prompts_untruncated": all(
            bool(row.get("candidate_audit", {}).get("all_candidates_valid")) for row in rows
        ),
    }
    return {
        "phase": "P0",
        "scope": "engineering smoke only; outputs cannot establish a scientific result",
        "passed": all(integrity.values()),
        "integrity_checks": integrity,
        "metrics": {
            "external_item_mean_defect_nats": mean(external_item_defects),
            "external_defect_upper_95_nats": external_upper,
            "technical_duplicate_item_mean_defect_nats": mean(duplicate_item_defects),
            "technical_duplicate_upper_95_nats": duplicate_upper,
            "same_adapter_rescore_max_candidate_delta": rescore_delta,
            "adapter_disable_max_candidate_delta": disable_delta,
            "invalid_row_count": len(invalid_rows),
        },
        "counts": {
            "rows": len(rows),
            "planned_artifacts": planned_artifacts,
            "verified_artifacts": verified_artifacts,
            "by_arm": {arm: len(arm_rows) for arm, arm_rows in sorted(by_arm.items())},
        },
    }


def bootstrap_upper_95(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("bootstrap values must be non-empty, finite, and non-negative")
    if len(values) == 1:
        return values[0]
    rng = random.Random(BOOTSTRAP_SEED)
    estimates = sorted(
        mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    return estimates[math.ceil(0.95 * len(estimates)) - 1]


def _rows_by_arm(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["arm"])].append(row)
    return dict(grouped)


def _rate(values: Iterable[bool]) -> float:
    observed = list(values)
    return sum(observed) / len(observed) if observed else 0.0


def _accuracy(rows: Iterable[dict[str, Any]]) -> float:
    return _rate(bool(row.get("correct")) for row in rows)


def _pair_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("attempt_id", "")), str(row["probe_id"])


def _paired_max_delta(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
) -> float:
    left = {_pair_key(row): row for row in first}
    right = {_pair_key(row): row for row in second}
    if not left or set(left) != set(right) or len(left) != len(first) or len(right) != len(second):
        return math.inf
    return max(maximum_candidate_delta(left[key], right[key]) for key in left)


def _distribution(row: dict[str, Any]) -> CandidateDistribution:
    probabilities = row.get("candidate_probabilities")
    actions = row.get("ordered_actions")
    if not isinstance(probabilities, list) or len(probabilities) != 4:
        raise ValueError("row lacks a valid candidate distribution")
    return CandidateDistribution(tuple(actions), tuple(float(value) for value in probabilities))


def _paired_item_defects(
    rows: list[dict[str, Any]],
    *,
    pairs: tuple[str, ...],
) -> list[float]:
    by_item_path_probe = {
        (str(row["item_id"]), str(row["logical_name"]), str(row["probe_id"])): row for row in rows
    }
    item_ids = sorted({str(row["item_id"]) for row in rows})
    defects: list[float] = []
    for item_id in item_ids:
        pair_means: list[float] = []
        for pair in pairs:
            first_path, second_path = pair.split(":")
            first_probes = {
                probe_id: row
                for (candidate_item, path, probe_id), row in by_item_path_probe.items()
                if candidate_item == item_id and path == first_path
            }
            second_probes = {
                probe_id: row
                for (candidate_item, path, probe_id), row in by_item_path_probe.items()
                if candidate_item == item_id and path == second_path
            }
            if not first_probes or set(first_probes) != set(second_probes):
                raise ValueError(f"incomplete endpoint panel for {item_id}/{pair}")
            pair_means.append(
                mean(
                    jensen_shannon_nats(
                        _distribution(first_probes[probe_id]),
                        _distribution(second_probes[probe_id]),
                    )
                    for probe_id in first_probes
                )
            )
        defects.append(mean(pair_means))
    if not defects:
        raise ValueError("no endpoint defects were available")
    return defects


def _duplicate_item_defects(
    original_rows: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
) -> list[float]:
    originals = {
        (str(row["item_id"]), str(row["logical_name"]), str(row["probe_id"])): row
        for row in original_rows
    }
    by_item: dict[str, list[float]] = defaultdict(list)
    for duplicate in duplicate_rows:
        key = (
            str(duplicate["item_id"]),
            str(duplicate["duplicate_of"]),
            str(duplicate["probe_id"]),
        )
        if key not in originals:
            raise ValueError(f"technical duplicate has no original: {key}")
        by_item[key[0]].append(
            jensen_shannon_nats(_distribution(originals[key]), _distribution(duplicate))
        )
    if not by_item or any(len(values) != 14 for values in by_item.values()):
        raise ValueError("technical duplicate panels must contain 14 probes per item")
    return [mean(by_item[item_id]) for item_id in sorted(by_item)]


def _external_prompts_identical(rows: list[dict[str, Any]]) -> bool:
    grouped: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        grouped[(str(row["item_id"]), str(row["terminal_state"]), str(row["probe_id"]))].add(
            str(row["raw_prompt_sha256"])
        )
    return bool(grouped) and all(len(hashes) == 1 for hashes in grouped.values())


def _result_ids_unique(rows: list[dict[str, Any]]) -> bool:
    result_ids = [row.get("result_identity", {}).get("result_id") for row in rows]
    return bool(result_ids) and None not in result_ids and len(set(result_ids)) == len(result_ids)


def _single_value(rows: list[dict[str, Any]], field: str) -> bool:
    values = {row.get(field) for row in rows}
    return len(values) == 1 and None not in values
