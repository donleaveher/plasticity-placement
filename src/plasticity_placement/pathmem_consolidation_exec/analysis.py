from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from plasticity_placement.pathmem_consolidation.config import evaluate_g1c_gate
from plasticity_placement.pathmem_consolidation_exec.config import SUMMARY_SCHEMA_VERSION

EXPECTED_TEACHER_ROWS = 96
EXPECTED_ROWS_BY_ARM = {
    "base_immediate": 96,
    "immediate": 96,
    "retained": 96,
    "retained_core": 336,
    "base_control": 288,
    "routed_control": 288,
    "access_off": 96,
    "restored_base": 96,
    "wrong_swap": 96,
}
EXPECTED_ROWS_PER_UNIT_BY_ARM = {
    arm: count // 24 for arm, count in EXPECTED_ROWS_BY_ARM.items()
}


def summarize_g1c(
    teacher_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    *,
    run_id: str,
    source_manifest_id: str,
    plan_id: str,
    integrity_checks: dict[str, bool],
) -> dict[str, Any]:
    teacher_valid = [row for row in teacher_rows if row.get("parser_valid") is True]
    teacher_metrics = {
        "teacher_current_top1": _accuracy(teacher_rows),
        "teacher_obsolete_intrusion": _mean_boolean(
            teacher_rows, "obsolete_intrusion"
        ),
        "teacher_parser_validity": len(teacher_valid) / max(len(teacher_rows), 1),
    }

    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        by_arm[str(row.get("arm"))].append(row)
    observed_counts = {arm: len(rows) for arm, rows in sorted(by_arm.items())}
    exact_counts = observed_counts == EXPECTED_ROWS_BY_ARM
    matrix_checks = _matrix_checks(teacher_rows, evaluation_rows)

    immediate = by_arm["immediate"]
    retained = by_arm["retained"]
    immediate_by_action = _accuracy_by(immediate, "expected_action")
    retained_by_action = _accuracy_by(retained, "expected_action")
    actions = sorted(set(immediate_by_action) | set(retained_by_action))
    exact_action_counts = Counter(str(row["expected_action"]) for row in immediate)
    balanced_actions = len(exact_action_counts) == 4 and set(exact_action_counts.values()) == {24}
    per_action_drops = {
        action: 100.0
        * (immediate_by_action.get(action, 0.0) - retained_by_action.get(action, 0.0))
        for action in actions
    }
    unrelated_base = [
        row for row in by_arm["base_control"] if row.get("probe_category") == "unrelated"
    ]
    unrelated_routed = [
        row
        for row in by_arm["routed_control"]
        if row.get("probe_category") == "unrelated"
    ]
    route_targets = [*immediate, *retained, *by_arm["retained_core"]]
    route_controls = by_arm["routed_control"]
    metrics = {
        **teacher_metrics,
        "immediate_current_top1": _accuracy(immediate),
        "immediate_per_action_min_top1": min(immediate_by_action.values(), default=0.0),
        "immediate_zero_action_count": float(
            sum(value == 0.0 for value in immediate_by_action.values())
        ),
        "retention_h12_current_top1": _accuracy(retained),
        "retention_h12_aggregate_drop_pp": 100.0
        * (_accuracy(immediate) - _accuracy(retained)),
        "retention_h12_per_action_max_drop_pp": max(
            per_action_drops.values(), default=math.inf
        ),
        "unrelated_regression_pp": 100.0
        * (_accuracy(unrelated_base) - _accuracy(unrelated_routed)),
        "router_miss_rate": _mean_boolean(route_targets, "route_miss"),
        "router_false_activation_rate": _mean_boolean(
            route_controls, "false_activation"
        ),
        "wrong_swap_sensitivity_reported": float(
            len(by_arm["wrong_swap"]) == EXPECTED_ROWS_BY_ARM["wrong_swap"]
        ),
        "access_off_max_candidate_delta": _paired_max_delta(
            by_arm["base_immediate"], by_arm["access_off"]
        ),
        "restore_max_candidate_delta": _paired_max_delta(
            by_arm["base_immediate"], by_arm["restored_base"]
        ),
        "integrity_passed": float(
            exact_counts
            and len(teacher_rows) == EXPECTED_TEACHER_ROWS
            and balanced_actions
            and all(matrix_checks.values())
            and integrity_checks
            and all(integrity_checks.values())
        ),
    }
    gate_result = evaluate_g1c_gate(metrics)
    gate = {
        "gate": gate_result.gate,
        "passed": gate_result.passed,
        "checks": [[metric, passed] for metric, passed in gate_result.checks],
        "blockers": list(gate_result.blockers),
    }
    wrong_swap_accuracy = _accuracy(by_arm["wrong_swap"])
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_id": run_id,
        "source_manifest_id": source_manifest_id,
        "plan_id": plan_id,
        "counts": {
            "teacher_rows": len(teacher_rows),
            "evaluation_rows": len(evaluation_rows),
            "rows_by_arm": observed_counts,
        },
        "metrics": metrics,
        "teacher_metrics": teacher_metrics,
        "per_action": {
            action: {
                "immediate_top1": immediate_by_action.get(action, 0.0),
                "retained_top1": retained_by_action.get(action, 0.0),
                "drop_pp": per_action_drops[action],
            }
            for action in actions
        },
        "per_item": _per_item_report(immediate, retained),
        "routing": {
            "wrong_swap_accuracy": wrong_swap_accuracy,
            "correct_minus_wrong_swap_pp": 100.0
            * (_accuracy(immediate) - wrong_swap_accuracy),
            "target_query_count": len(route_targets),
            "control_query_count": len(route_controls),
        },
        "integrity_checks": {
            **integrity_checks,
            "exact_result_matrix": exact_counts,
            "exact_teacher_matrix": len(teacher_rows) == EXPECTED_TEACHER_ROWS,
            "balanced_immediate_action_matrix": balanced_actions,
            **matrix_checks,
        },
        "gate": gate,
        "p0_runner_implementation_review_eligible": gate["passed"],
        "p0_authorized": False,
        "path_contrast_authorized": False,
        "rl_controller_authorized": False,
    }
    return summary


def summarize_teacher_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {
        "current_top1": _accuracy(rows),
        "obsolete_intrusion": _mean_boolean(rows, "obsolete_intrusion"),
        "parser_validity": sum(row.get("parser_valid") is True for row in rows)
        / max(len(rows), 1),
    }
    checks = {
        "exact_96_rows": len(rows) == EXPECTED_TEACHER_ROWS,
        "current_top1_at_least_95pct": metrics["current_top1"] >= 0.95,
        "obsolete_intrusion_at_most_5pct": metrics["obsolete_intrusion"] <= 0.05,
        "parser_validity_exactly_100pct": metrics["parser_validity"] == 1.0,
    }
    return {
        "gate": "G1-C-teacher",
        "passed": all(checks.values()),
        "metrics": metrics,
        "checks": checks,
        "blockers": [name for name, passed in checks.items() if not passed],
    }


def _matrix_checks(
    teacher_rows: list[dict[str, Any]], evaluation_rows: list[dict[str, Any]]
) -> dict[str, bool]:
    teacher_units = Counter(str(row.get("unit_id")) for row in teacher_rows)
    evaluation_units = Counter(str(row.get("unit_id")) for row in evaluation_rows)
    per_unit_arm = Counter(
        (str(row.get("unit_id")), str(row.get("arm"))) for row in evaluation_rows
    )
    unit_ids = set(teacher_units)
    exact_per_unit = len(unit_ids) == 24 and all(
        per_unit_arm[(unit_id, arm)] == count
        for unit_id in unit_ids
        for arm, count in EXPECTED_ROWS_PER_UNIT_BY_ARM.items()
    )

    teacher_row_ids = [row.get("row_id") for row in teacher_rows]
    evaluation_row_ids = [row.get("row_id") for row in evaluation_rows]
    teacher_pairs = {
        (str(row.get("unit_id")), str(row.get("probe_id"))) for row in teacher_rows
    }

    def pairs(arm: str) -> set[tuple[str, str]]:
        return {
            (str(row.get("unit_id")), str(row.get("probe_id")))
            for row in evaluation_rows
            if row.get("arm") == arm
        }

    qualification_arms = (
        "base_immediate",
        "immediate",
        "retained",
        "access_off",
        "restored_base",
        "wrong_swap",
    )
    row_keys = [
        (str(row.get("unit_id")), str(row.get("probe_id")), str(row.get("arm")))
        for row in evaluation_rows
    ]
    return {
        "exact_24_unit_matrix": (
            unit_ids == set(evaluation_units)
            and set(teacher_units.values()) == {4}
            and exact_per_unit
        ),
        "unique_teacher_row_ids": (
            all(isinstance(value, str) and value for value in teacher_row_ids)
            and len(set(teacher_row_ids)) == len(teacher_row_ids)
        ),
        "unique_evaluation_row_ids": (
            all(isinstance(value, str) and value for value in evaluation_row_ids)
            and len(set(evaluation_row_ids)) == len(evaluation_row_ids)
        ),
        "unique_evaluation_unit_probe_arms": len(set(row_keys)) == len(row_keys),
        "qualification_panels_paired": all(
            pairs(arm) == teacher_pairs for arm in qualification_arms
        ),
        "control_panels_paired": pairs("base_control") == pairs("routed_control"),
    }
def _accuracy(rows: list[dict[str, Any]]) -> float:
    return sum(row.get("correct") is True for row in rows) / max(len(rows), 1)


def _accuracy_by(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    counts: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for row in rows:
        key = str(row.get(field))
        counts[key] += 1
        correct[key] += row.get("correct") is True
    return {key: correct[key] / count for key, count in sorted(counts.items())}


def _mean_boolean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(row.get(field) is True for row in rows) / max(len(rows), 1)


def _paired_max_delta(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> float:
    def keyed(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[float]]:
        return {
            (str(row["unit_id"]), str(row["probe_id"])): [
                float(value) for value in row["candidate_scores"]
            ]
            for row in rows
        }

    left = keyed(first)
    right = keyed(second)
    if not left or set(left) != set(right):
        return math.inf
    return max(
        abs(a - b)
        for key in left
        for a, b in zip(left[key], right[key], strict=True)
    )


def _per_item_report(
    immediate: list[dict[str, Any]], retained: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    item_ids = sorted({str(row["item_id"]) for row in [*immediate, *retained]})
    return {
        item_id: {
            "immediate_top1": _accuracy(
                [row for row in immediate if row["item_id"] == item_id]
            ),
            "retained_top1": _accuracy(
                [row for row in retained if row["item_id"] == item_id]
            ),
        }
        for item_id in item_ids
    }
