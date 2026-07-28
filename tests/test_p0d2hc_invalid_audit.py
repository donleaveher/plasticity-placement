from __future__ import annotations

from typing import Any

import pytest

from plasticity_placement.p0d2h.probes import HARD_CATEGORIES
from plasticity_placement.p0d2hc.config import CALIBRATION_ARMS
from plasticity_placement.p0d2hc.invalid_audit import (
    build_invalid_output_audit,
    classify_output_row,
)


@pytest.mark.parametrize(
    ("text", "tokens", "invalid_class", "semantic_correct"),
    [
        ("", 0, "empty_output", False),
        (
            "Answer: act_n7",
            4,
            "expected_action_with_extra_text",
            True,
        ),
        (
            "Answer: act_p3",
            4,
            "wrong_action_with_extra_text",
            False,
        ),
        (
            "act_n7 or act_p3",
            6,
            "multiple_actions_including_expected",
            False,
        ),
        (
            "act_p3 or act_v9",
            6,
            "multiple_actions_excluding_expected",
            False,
        ),
        (
            "The action is act_",
            8,
            "no_allowed_action_at_generation_limit",
            False,
        ),
        (
            "I cannot decide",
            4,
            "no_allowed_action_other",
            False,
        ),
    ],
)
def test_invalid_taxonomy_is_mutually_exclusive(
    text: str,
    tokens: int,
    invalid_class: str,
    semantic_correct: bool,
) -> None:
    classified = classify_output_row(
        _row(
            generated_text=text,
            generated_tokens=tokens,
            invalid=True,
            correct=False,
        ),
        max_new_tokens=8,
    )
    assert classified["invalid_class"] == invalid_class
    assert classified["semantic_correct"] is semantic_correct


def test_semantic_recovery_keeps_strict_exact_result_separate() -> None:
    strict = classify_output_row(
        _row(
            generated_text=" ACT_N7 ",
            generated_tokens=1,
            invalid=False,
            correct=True,
        ),
        max_new_tokens=8,
    )
    recovered = classify_output_row(
        _row(
            generated_text="The answer is act_n7.",
            generated_tokens=6,
            invalid=True,
            correct=False,
        ),
        max_new_tokens=8,
    )
    assert strict["strict_correct"] is True
    assert strict["invalid_class"] is None
    assert recovered["strict_correct"] is False
    assert recovered["semantic_correct"] is True


def test_invalid_audit_reports_lesson_clustered_semantic_gain() -> None:
    rows = []
    for arm in CALIBRATION_ARMS:
        for category in HARD_CATEGORIES:
            rows.append(
                _row(
                    arm=arm,
                    category=category,
                    probe_id=f"{arm}-{category}",
                    generated_text=(
                        "Answer: act_n7"
                        if arm == "external"
                        else "act_n7"
                    ),
                    generated_tokens=4 if arm == "external" else 1,
                    invalid=arm == "external",
                    correct=arm != "external",
                )
            )
    report, invalid_records = build_invalid_output_audit(
        rows,
        max_new_tokens=8,
        bootstrap_samples=20,
    )
    external = report["overall_by_model_arm"]["scale_canary"][
        "external"
    ]
    assert report["analysis_status"] == "post_hoc_supplementary"
    assert report["strict_gate_status_changed"] is False
    assert report["next_stage_eligibility_changed"] is False
    assert report["invalid_record_count"] == 4
    assert external["strict_accuracy"] == 0.0
    assert external["strict_invalid_rate"] == 1.0
    assert external["semantic_accuracy"] == 1.0
    assert external["semantic_recovery_gain"] == 1.0
    assert external["invalid_semantic_recovery_rate"] == 1.0
    assert len(invalid_records) == 4


def test_invalid_audit_rejects_stored_parser_mismatch() -> None:
    with pytest.raises(ValueError, match="differs"):
        classify_output_row(
            _row(
                generated_text="act_n7",
                generated_tokens=1,
                invalid=True,
                correct=False,
            ),
            max_new_tokens=8,
        )


def _row(
    *,
    arm: str = "external",
    category: str = "binding_decoys",
    probe_id: str = "probe-1",
    generated_text: str,
    generated_tokens: int,
    invalid: bool,
    correct: bool,
) -> dict[str, Any]:
    return {
        "calibration_model_id": "scale_canary",
        "calibration_model_role": "scale_canary",
        "arm": arm,
        "lesson_id": "lesson-1",
        "probe_id": probe_id,
        "category": category,
        "pair_id": "pair-1",
        "lesson_type": "fact_mapping",
        "expected_action": "act_n7",
        "generated_text": generated_text,
        "generated_tokens": generated_tokens,
        "invalid": invalid,
        "correct": correct,
    }
