from __future__ import annotations

import json
from typing import Any

from plasticity_placement.pathmem_consolidation_exec.analysis import (
    EXPECTED_ROWS_BY_ARM,
    summarize_g1c,
    summarize_teacher_gate,
)

ACTIONS = ("act_k2", "act_n7", "act_p3", "act_v9")


def _row(
    *,
    arm: str,
    index: int,
    expected: str,
    correct: bool = True,
    category: str = "qualification",
    scores: list[float] | None = None,
) -> dict[str, Any]:
    predicted = expected if correct else ACTIONS[(ACTIONS.index(expected) + 1) % 4]
    return {
        "row_id": f"{arm}:{index:04d}",
        "unit_id": f"unit-{index % 24:02d}",
        "item_id": f"item-{index % 12:02d}",
        "probe_id": f"probe-{index:04d}",
        "arm": arm,
        "probe_category": category,
        "expected_action": expected,
        "predicted_action": predicted,
        "correct": correct,
        "obsolete_intrusion": False,
        "parser_valid": True,
        "route_miss": False,
        "false_activation": False,
        "candidate_scores": scores or [-0.1, -2.0, -3.0, -4.0],
        "candidate_audit": {"all_candidates_valid": True},
        "non_finite": False,
    }


def _teacher_rows() -> list[dict[str, Any]]:
    return [
        _row(arm="teacher", index=index, expected=ACTIONS[index % 4])
        for index in range(96)
    ]


def _evaluation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm, count in EXPECTED_ROWS_BY_ARM.items():
        for index in range(count):
            category = (
                "unrelated"
                if arm in {"base_control", "routed_control"} and index % 12 < 8
                else "near_neighbor"
                if arm in {"base_control", "routed_control"}
                else "qualification"
            )
            rows.append(
                _row(
                    arm=arm,
                    index=index,
                    expected=ACTIONS[index % 4],
                    correct=arm != "wrong_swap",
                    category=category,
                )
            )
    return rows


def test_teacher_gate_requires_exact_complete_matrix() -> None:
    passing = summarize_teacher_gate(_teacher_rows())
    assert passing["passed"] is True
    missing = summarize_teacher_gate(_teacher_rows()[:-1])
    assert missing["passed"] is False
    assert missing["checks"]["exact_96_rows"] is False


def test_g1c_summary_passes_exact_matrix_and_retains_all_boundaries() -> None:
    summary = summarize_g1c(
        _teacher_rows(),
        _evaluation_rows(),
        run_id="r" * 64,
        source_manifest_id="s" * 64,
        plan_id="p" * 64,
        integrity_checks={"complete": True, "source_unchanged": True},
    )
    assert summary["gate"]["passed"] is True
    assert summary["metrics"]["immediate_current_top1"] == 1.0
    assert summary["metrics"]["retention_h12_current_top1"] == 1.0
    assert summary["metrics"]["unrelated_regression_pp"] == 0.0
    assert summary["metrics"]["access_off_max_candidate_delta"] == 0.0
    assert summary["routing"]["wrong_swap_accuracy"] == 0.0
    assert summary["p0_runner_implementation_review_eligible"] is True
    assert summary["p0_authorized"] is False
    assert summary["path_contrast_authorized"] is False
    assert summary["rl_controller_authorized"] is False
    assert json.loads(json.dumps(summary)) == summary
    assert isinstance(summary["gate"]["checks"], list)
    assert isinstance(summary["gate"]["blockers"], list)


def test_g1c_summary_fails_retention_and_incomplete_matrix() -> None:
    rows = _evaluation_rows()
    for row in rows:
        if row["arm"] == "retained":
            row["correct"] = False
    summary = summarize_g1c(
        _teacher_rows(),
        rows[:-1],
        run_id="r" * 64,
        source_manifest_id="s" * 64,
        plan_id="p" * 64,
        integrity_checks={"complete": True},
    )
    assert summary["gate"]["passed"] is False
    assert "retention_h12_current_top1" in summary["gate"]["blockers"]
    assert "integrity_passed" in summary["gate"]["blockers"]
    assert summary["p0_runner_implementation_review_eligible"] is False


def test_g1c_summary_rejects_duplicate_rows_and_skewed_units() -> None:
    rows = _evaluation_rows()
    rows[1]["row_id"] = rows[0]["row_id"]
    rows[2]["unit_id"] = rows[0]["unit_id"]
    summary = summarize_g1c(
        _teacher_rows(),
        rows,
        run_id="r" * 64,
        source_manifest_id="s" * 64,
        plan_id="p" * 64,
        integrity_checks={"complete": True},
    )
    assert summary["gate"]["passed"] is False
    assert summary["integrity_checks"]["unique_evaluation_row_ids"] is False
    assert summary["integrity_checks"]["exact_24_unit_matrix"] is False
    assert summary["integrity_checks"]["qualification_panels_paired"] is False
