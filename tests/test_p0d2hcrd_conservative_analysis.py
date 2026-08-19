from __future__ import annotations

from typing import Any

from plasticity_placement.p0d2hcrd.conservative_analysis import (
    build_conservative_tie_analysis,
    conservative_diagnostic_gate,
)
from plasticity_placement.p0d2hcrd.scoring import rank_candidate_scores


def _candidate(name: str, token_id: int, score: float) -> dict[str, Any]:
    return {
        "candidate": name,
        "continuation": name,
        "token_ids": [token_id],
        "token_count": 1,
        "sum_logprob": score,
        "mean_logprob": score,
        "token_logprobs": [score],
    }


def _row(
    lesson_id: str,
    endpoint: str,
    index: int,
    *,
    correct: bool,
    tie: bool = False,
) -> dict[str, Any]:
    candidate_count = 2 if endpoint == "route_only" else 4
    names = [f"candidate-{value}" for value in range(candidate_count)]
    expected = names[0]
    scores = [-4.0 - value for value in range(candidate_count)]
    if correct:
        scores[0] = -0.1
    else:
        scores[1] = -0.1
    if tie:
        scores[0] = scores[1] = -0.1
    candidates = [_candidate(name, 100 + value, scores[value]) for value, name in enumerate(names)]
    outcome = rank_candidate_scores(candidates)
    return {
        "lesson_id": lesson_id,
        "pair_id": f"pair-{lesson_id}",
        "probe_id": f"{lesson_id}-{endpoint}-{index}",
        "endpoint": endpoint,
        "lesson_type": "procedure_recovery",
        "lesson_side": "a",
        "route_variant": 1,
        "target_slot": "slot_b",
        "expected_candidate": expected,
        "ordered_candidates": names,
        "slot_candidate_order": None,
        "slot_content_order": 1,
        "action_panel_position": 1,
        **outcome,
        "correct": outcome["predicted_candidate"] == expected,
    }


def _matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lesson_index in range(24):
        lesson_id = f"L{lesson_index:02d}"
        for index in range(16):
            rows.append(
                _row(
                    lesson_id,
                    "route_only",
                    index,
                    correct=index < 10,
                )
            )
            rows.append(
                _row(
                    lesson_id,
                    "retrieval_only",
                    index,
                    correct=True,
                )
            )
        for index in range(32):
            rows.append(
                _row(
                    lesson_id,
                    "combined",
                    index,
                    correct=True,
                    tie=lesson_index < 2 and index == 0,
                )
            )
    return rows


def test_conservative_bounds_allow_finite_ties_without_tie_breaking() -> None:
    result = build_conservative_tie_analysis(
        _matrix(),
        {"all_checks_passed": True},
        {"all_checks_passed": True, "input_truncated_count": 0},
        bootstrap_samples=100,
    )
    assert result["tie_summary"]["finite_exact_tie_count"] == 2
    assert result["tie_summary"]["expected_in_top_set_count"] == 2
    assert result["endpoint_summary"]["route_only"]["accuracy_lower"] == 0.625
    assert result["endpoint_summary"]["route_only"]["accuracy_upper"] == 0.625
    assert result["endpoint_summary"]["combined"]["accuracy_lower"] < 1.0
    assert result["endpoint_summary"]["combined"]["accuracy_upper"] == 1.0
    assert result["diagnostic_gate"]["status"] == "route_bottleneck_supported"
    assert result["diagnostic_gate"]["bound_statuses"]["match"] is True
    assert result["diagnostic_gate"]["training_complexity_review_eligible"] is False


def test_conservative_gate_preserves_bound_ambiguity() -> None:
    gate = conservative_diagnostic_gate(
        {
            "route_only": {"accuracy_lower": 0.89, "accuracy_upper": 0.91},
            "retrieval_only": {
                "accuracy_lower": 1.0,
                "accuracy_upper": 1.0,
            },
            "combined": {"accuracy_lower": 0.95, "accuracy_upper": 0.96},
        },
        {"all_checks_passed": True},
        {"all_checks_passed": True, "input_truncated_count": 0},
        structural_checks_passed=True,
        fatal_error_count=0,
    )
    assert gate["status"] == "tie_interval_ambiguous"
    assert gate["bound_statuses"] == {
        "lower": "route_bottleneck_supported",
        "upper": "no_component_bottleneck_detected",
        "match": False,
    }


def test_conservative_gate_keeps_non_tie_errors_fatal() -> None:
    gate = conservative_diagnostic_gate(
        {
            endpoint: {
                "accuracy_lower": 1.0,
                "accuracy_upper": 1.0,
            }
            for endpoint in ("route_only", "retrieval_only", "combined")
        },
        {"all_checks_passed": True},
        {"all_checks_passed": True, "input_truncated_count": 0},
        structural_checks_passed=True,
        fatal_error_count=1,
    )
    assert gate["status"] == "scoring_integrity_failed"
    assert gate["integrity_passed"] is False
