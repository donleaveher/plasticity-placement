from __future__ import annotations

from typing import Any

from plasticity_placement.pathmem.io import json_hash
from plasticity_placement.pathmem_exec.analysis import summarize_g1, summarize_p0

ACTIONS = ["act_n7", "act_p3", "act_v9", "act_k2"]


def _row(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    attempt_id: str,
    probe_id: str,
    item_id: str = "item",
    logical_name: str = "control",
    expected_action: str = ACTIONS[0],
    raw_prompt_sha256: str | None = None,
) -> dict[str, Any]:
    row = {
        "arm": arm,
        "attempt_id": attempt_id,
        "probe_id": probe_id,
        "item_id": item_id,
        "logical_name": logical_name,
        "terminal_state": "A" if logical_name in {"ABA", "BAA", "A2"} else "B",
        "expected_action": expected_action,
        "predicted_action": expected_action,
        "correct": True,
        "obsolete_intrusion": False,
        "parser_valid": True,
        "error_status": "ok",
        "tie": False,
        "non_finite": False,
        "ordered_actions": ACTIONS,
        "candidate_probabilities": [0.7, 0.1, 0.1, 0.1],
        "candidates": [
            {"action": action, "sum_logprob": score}
            for action, score in zip(ACTIONS, (-0.1, -2.0, -3.0, -4.0), strict=True)
        ],
        "candidate_audit": {"all_candidates_valid": True},
        "model_revision": "frozen-revision",
        "evaluation_precision": "nf4-bfloat16",
        "raw_prompt_sha256": raw_prompt_sha256 or json_hash([attempt_id, probe_id, arm]),
        "result_identity": {"result_id": json_hash([len(rows), attempt_id, probe_id, arm])},
    }
    rows.append(row)
    return row


def _passing_g1_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(96):
        _row(rows, arm="answer_copy", attempt_id=f"copy-{index}", probe_id=f"q-{index}")
        _row(
            rows,
            arm="external_latest",
            attempt_id=f"external-{index}",
            probe_id=f"q-{index}",
        )
    for attempt_index in range(24):
        attempt_id = f"anchor-{attempt_index}"
        for probe_index in range(4):
            probe_id = f"qualification-{probe_index}"
            expected = ACTIONS[(attempt_index * 4 + probe_index) % 4]
            for arm in (
                "bare_base",
                "parametric_current",
                "parametric_rescore",
                "adapter_disabled",
            ):
                _row(
                    rows,
                    arm=arm,
                    attempt_id=attempt_id,
                    probe_id=probe_id,
                    expected_action=expected,
                )
        for probe_index in range(8):
            probe_id = f"unrelated-{probe_index}"
            for arm in ("base_unrelated", "parametric_unrelated"):
                _row(rows, arm=arm, attempt_id=attempt_id, probe_id=probe_id)
    return rows


def test_g1_summary_passes_only_with_exact_rows_and_metrics() -> None:
    summary = summarize_g1(_passing_g1_rows(), verified_attempts=24)
    assert summary["gate"]["passed"] is True
    assert summary["integrity_passed"] is True
    assert summary["authorization_ready"] is True
    assert summary["counts"]["rows"] == 960
    assert summary["metrics"]["adapter_disable_max_candidate_delta"] == 0.0


def test_g1_summary_blocks_authorization_on_missing_row() -> None:
    rows = _passing_g1_rows()
    rows.pop()
    summary = summarize_g1(rows, verified_attempts=24)
    assert summary["gate"]["passed"] is True
    assert summary["integrity_passed"] is False
    assert summary["authorization_ready"] is False


def _passing_p0_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    duplicate_paths = ("ABA", "BAA", "BAB", "ABB")
    for item_index in range(4):
        item_id = f"smoke-{item_index}"
        for state in ("A", "B"):
            for probe_index in range(14):
                probe_id = f"{item_id}:{state}:core-{probe_index}"
                _row(
                    rows,
                    arm="no_memory",
                    attempt_id=f"{item_id}:N:{state}",
                    probe_id=probe_id,
                    item_id=item_id,
                    logical_name="N",
                )
                _row(
                    rows,
                    arm="p_latest",
                    attempt_id=f"{item_id}:{state}2",
                    probe_id=probe_id,
                    item_id=item_id,
                    logical_name=f"{state}2",
                )
        for logical_name in ("ABA", "BAA", "BAB", "ABB"):
            state = logical_name[-1]
            attempt_id = f"{item_id}:{logical_name}"
            for probe_index in range(14):
                probe_id = f"{item_id}:{state}:core-{probe_index}"
                external_prompt = json_hash([item_id, state, probe_id, "external"])
                for arm in (
                    "base_before",
                    "parametric_path",
                    "parametric_rescore",
                    "adapter_disabled",
                    "external_latest",
                    "icl_history",
                    "both",
                ):
                    _row(
                        rows,
                        arm=arm,
                        attempt_id=attempt_id,
                        probe_id=probe_id,
                        item_id=item_id,
                        logical_name=logical_name,
                        raw_prompt_sha256=(external_prompt if arm == "external_latest" else None),
                    )
        duplicate_of = duplicate_paths[item_index]
        state = duplicate_of[-1]
        for probe_index in range(14):
            row = _row(
                rows,
                arm="technical_duplicate",
                attempt_id=f"{item_id}:DUP-{duplicate_of}",
                probe_id=f"{item_id}:{state}:core-{probe_index}",
                item_id=item_id,
                logical_name=f"DUP-{duplicate_of}",
            )
            row["duplicate_of"] = duplicate_of
    return rows


def test_p0_summary_passes_exact_44_artifact_smoke() -> None:
    summary = summarize_p0(
        _passing_p0_rows(),
        verified_artifacts=44,
        planned_artifacts=44,
        lineage_verified=True,
        exposure_verified=True,
    )
    assert summary["passed"] is True
    assert summary["counts"]["rows"] == 1848
    assert summary["metrics"]["external_defect_upper_95_nats"] == 0.0
    assert summary["metrics"]["technical_duplicate_upper_95_nats"] == 0.0


def test_p0_summary_fails_if_external_prompt_bytes_differ() -> None:
    rows = _passing_p0_rows()
    external = [row for row in rows if row["arm"] == "external_latest"]
    external[0]["raw_prompt_sha256"] = "changed"
    summary = summarize_p0(
        rows,
        verified_artifacts=44,
        planned_artifacts=44,
        lineage_verified=True,
        exposure_verified=True,
    )
    assert summary["passed"] is False
    assert summary["integrity_checks"]["external_prompts_byte_identical"] is False
