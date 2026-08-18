from __future__ import annotations

import math
from typing import Any

from plasticity_placement.pathmem_ropcd_p0.analysis import summarize_p0

ACTIONS = ("act_a", "act_b", "act_c", "act_d")
PATHS = ("ABA", "BAA", "BAB", "ABB")
DUPLICATES = ("ABA", "BAA", "BAB", "ABB")


def _probabilities(expected: str) -> list[float]:
    probabilities = [0.05, 0.05, 0.05, 0.05]
    probabilities[ACTIONS.index(expected)] = 0.85
    return probabilities


def _row(
    *,
    arm: str,
    row_id: str,
    item_id: str,
    logical_name: str,
    probe_id: str,
    terminal_state: str,
    expected: str,
    obsolete: str | None,
    category: str,
    raw_prompt_sha256: str | None = None,
) -> dict[str, Any]:
    probabilities = _probabilities(expected)
    scores = [math.log(value) for value in probabilities]
    return {
        "row_id": row_id,
        "unit_id": f"{item_id}:{logical_name}",
        "item_id": item_id,
        "logical_name": logical_name,
        "probe_id": probe_id,
        "terminal_state": terminal_state,
        "probe_category": category,
        "arm": arm,
        "ordered_actions": list(ACTIONS),
        "candidate_probabilities": probabilities,
        "candidate_scores": scores,
        "expected_action": expected,
        "obsolete_action": obsolete,
        "predicted_action": expected,
        "correct": True,
        "error_status": "ok",
        "tie": False,
        "non_finite": False,
        "parser_valid": True,
        "raw_prompt_sha256": raw_prompt_sha256 or row_id,
        "candidate_audit": {"all_candidates_valid": True},
    }


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counter = 0

    def add(**kwargs: Any) -> dict[str, Any]:
        nonlocal counter
        counter += 1
        row = _row(row_id=f"row-{counter:05d}", **kwargs)
        rows.append(row)
        return row

    for item_index in range(4):
        item_id = f"item-{item_index}"
        for terminal_state in ("A", "B"):
            expected, obsolete = (
                (ACTIONS[0], ACTIONS[1]) if terminal_state == "A" else (ACTIONS[1], ACTIONS[0])
            )
            for probe_index in range(14):
                probe_id = f"{item_id}:{terminal_state}:core-{probe_index:02d}"
                for arm, logical_name in (("no_memory", "N"), ("p_latest", f"{terminal_state}2")):
                    add(
                        arm=arm,
                        item_id=item_id,
                        logical_name=logical_name,
                        probe_id=probe_id,
                        terminal_state=terminal_state,
                        expected=expected,
                        obsolete=obsolete,
                        category="exact",
                    )
        for path in PATHS:
            terminal_state = path[-1]
            expected, obsolete = (
                (ACTIONS[0], ACTIONS[1]) if terminal_state == "A" else (ACTIONS[1], ACTIONS[0])
            )
            for probe_index in range(14):
                probe_id = f"{item_id}:{terminal_state}:core-{probe_index:02d}"
                external_hash = f"external:{item_id}:{terminal_state}:{probe_id}"
                for arm in (
                    "base_before",
                    "parametric_path",
                    "parametric_rescore",
                    "adapter_disabled",
                    "external_latest",
                    "icl_history",
                    "both",
                ):
                    add(
                        arm=arm,
                        item_id=item_id,
                        logical_name=path,
                        probe_id=probe_id,
                        terminal_state=terminal_state,
                        expected=expected,
                        obsolete=obsolete,
                        category="exact",
                        raw_prompt_sha256=(external_hash if arm == "external_latest" else None),
                    )
            for probe_index in range(8):
                probe_id = f"{item_id}:{terminal_state}:unrelated-{probe_index:02d}"
                for arm in ("base_unrelated", "parametric_unrelated"):
                    add(
                        arm=arm,
                        item_id=item_id,
                        logical_name=path,
                        probe_id=probe_id,
                        terminal_state=terminal_state,
                        expected=ACTIONS[2],
                        obsolete=None,
                        category="unrelated",
                    )
            for probe_index in range(4):
                add(
                    arm="path_qualification",
                    item_id=item_id,
                    logical_name=path,
                    probe_id=f"{item_id}:{terminal_state}:qualification-{probe_index:02d}",
                    terminal_state=terminal_state,
                    expected=expected,
                    obsolete=obsolete,
                    category="qualification",
                )
        duplicate_of = DUPLICATES[item_index]
        terminal_state = duplicate_of[-1]
        expected, obsolete = (
            (ACTIONS[0], ACTIONS[1]) if terminal_state == "A" else (ACTIONS[1], ACTIONS[0])
        )
        for probe_index in range(14):
            row = add(
                arm="technical_duplicate",
                item_id=item_id,
                logical_name=f"DUP-{duplicate_of}",
                probe_id=f"{item_id}:{terminal_state}:core-{probe_index:02d}",
                terminal_state=terminal_state,
                expected=expected,
                obsolete=obsolete,
                category="exact",
            )
            row["duplicate_of"] = duplicate_of
    return rows


def test_p0_summary_passes_complete_engineering_matrix() -> None:
    summary = summarize_p0(
        _rows(),
        verified_units=44,
        integrity_checks={"lineage_verified": True, "authorization_bound": True},
    )
    assert summary["gate"]["passed"] is True
    assert summary["counts"]["rows"] == 2_168
    assert summary["metrics"]["external_defect_upper_95_nats"] == 0.0
    assert summary["metrics"]["icl_item_mean_defect_nats"] == 0.0
    assert summary["metrics"]["both_item_mean_defect_nats"] == 0.0
    assert summary["metrics"]["technical_duplicate_upper_95_nats"] == 0.0
    assert summary["metrics"]["path_write_qualification_rate"] == 1.0
    assert summary["metrics"]["joint_path_pass_rate"] == 1.0
    assert summary["metrics"]["unrelated_regression_pp"] == 0.0
    assert summary["arm_diagnostics"]["parametric_path"]["current_top1"] == 1.0
    assert len(summary["path_write_qualification"]) == 16
    assert len(summary["joint_path_pass"]) == 8
    assert summary["p1_authorized"] is False
    assert summary["scientific_result_authorized"] is False


def test_p0_summary_preserves_failed_paths_and_fails_integrity() -> None:
    rows = _rows()
    target = next(row for row in rows if row["arm"] == "path_qualification")
    target["candidate_probabilities"] = [0.05, 0.85, 0.05, 0.05]
    target["candidate_scores"] = [math.log(value) for value in target["candidate_probabilities"]]
    rescore = next(row for row in rows if row["arm"] == "parametric_rescore")
    rescore["candidate_scores"][0] += 0.1
    summary = summarize_p0(
        rows,
        verified_units=44,
        integrity_checks={"lineage_verified": True},
    )
    assert summary["gate"]["passed"] is False
    assert summary["metrics"]["path_write_qualification_rate"] < 1.0
    assert len(summary["path_write_qualification"]) == 16
    assert "same_adapter_rescore_within_tolerance" in summary["gate"]["blockers"]
