from __future__ import annotations

from plasticity_placement.p0d2hcrd.integrity_audit import (
    build_scoring_integrity_audit,
)
from plasticity_placement.p0d2hcrd.scoring import rank_candidate_scores


def _candidate(name: str, token_id: int, score: float) -> dict[str, object]:
    return {
        "candidate": name,
        "continuation": f" {name}",
        "token_ids": [token_id],
        "token_count": 1,
        "sum_logprob": score,
        "mean_logprob": score,
        "token_logprobs": [score],
    }


def _row(
    candidates: list[dict[str, object]],
    *,
    probe_id: str = "probe-1",
    expected_candidate: str = "slot_a",
) -> dict[str, object]:
    outcome = rank_candidate_scores(candidates)
    return {
        "lesson_id": "lesson-1",
        "pair_id": "pair-1",
        "probe_id": probe_id,
        "endpoint": "route_only",
        "lesson_type": "factual",
        "lesson_side": "a",
        "route_variant": "variant-1",
        "target_slot": "slot_a",
        "expected_candidate": expected_candidate,
        "ordered_candidates": [candidate["candidate"] for candidate in candidates],
        "slot_candidate_order": "ab",
        "slot_content_order": None,
        "action_panel_position": None,
        **outcome,
        "correct": outcome["predicted_candidate"] == expected_candidate,
    }


def test_integrity_audit_localizes_distinct_exact_score_tie() -> None:
    report, records = build_scoring_integrity_audit(
        [
            _row(
                [
                    _candidate("slot_a", 11, -1.0),
                    _candidate("slot_b", 22, -1.0),
                ]
            )
        ]
    )
    assert report["classification"] == ("distinct_candidate_exact_score_ties_observed")
    assert report["structural_checks_passed"] is True
    assert report["tie_record_count"] == 1
    assert report["endpoint_sensitivity"]["route_only"] == {
        "decision_count": 1,
        "correct_count": 0,
        "tie_count": 1,
        "official_accuracy": 0.0,
        "accuracy_if_all_ties_incorrect": 0.0,
        "accuracy_if_all_ties_correct": 1.0,
    }
    assert records[0]["tie_origin"] == ("distinct_candidates_exact_score_tie")
    assert len(records[0]["top_sum_candidates"]) == 2
    assert records[0]["checks"]["all_structural_checks_passed"] is True


def test_integrity_audit_flags_candidate_token_collision() -> None:
    report, records = build_scoring_integrity_audit(
        [
            _row(
                [
                    _candidate("slot_a", 11, -1.0),
                    _candidate("slot_b", 11, -1.0),
                ]
            )
        ]
    )
    assert report["classification"] == "structural_inconsistency_detected"
    assert report["structural_checks_passed"] is False
    assert records[0]["checks"]["candidate_token_sequences_unique"] is False
    assert records[0]["tie_origin"] == ("candidate_or_score_structure_inconsistency")


def test_integrity_audit_reports_clean_rows_without_changing_gate() -> None:
    report, records = build_scoring_integrity_audit(
        [
            _row(
                [
                    _candidate("slot_a", 11, -0.1),
                    _candidate("slot_b", 22, -2.0),
                ]
            )
        ]
    )
    assert report["classification"] == "no_scoring_anomalies"
    assert report["anomaly_record_count"] == 0
    assert report["source_gate_status_changed"] is False
    assert report["next_stage_eligibility_changed"] is False
    assert records == []
