from __future__ import annotations

from copy import deepcopy

from plasticity_placement.p0d2h.probes import HARD_CATEGORIES
from plasticity_placement.p0d2hfc.analysis import (
    _cross_scale_gate,
    _model_gate,
)


def _arm(accuracy: float, category_accuracy: float) -> dict:
    return {
        "accuracy": accuracy,
        "tie_error_nonfinite_rate": 0.0,
        "categories": {
            category: {"accuracy": category_accuracy}
            for category in HARD_CATEGORIES
        },
    }


def _passing_summary() -> dict:
    return {
        "no_write": _arm(0.35, 0.35),
        "external": _arm(0.85, 0.75),
        "answer_copy_oracle": _arm(0.95, 0.90),
    }


def _contrasts(lower: float = 0.001) -> dict:
    return {"external_minus_no_write": {"ci95": [lower, 0.8]}}


def _audit() -> dict:
    return {"all_checks_passed": True, "input_truncated_count": 0}


def test_gate_boundary_values_pass_but_ci_lower_must_be_strict() -> None:
    passed = _model_gate(
        "scale_canary",
        _passing_summary(),
        _contrasts(),
        _audit(),
    )
    assert passed["status"] == "training_complexity_review_eligible"
    assert passed["eligible"] is True

    failed = _model_gate(
        "scale_canary",
        _passing_summary(),
        _contrasts(0.0),
        _audit(),
    )
    assert failed["status"] == "external_contrast_failed"
    assert failed["eligible"] is False


def test_conditional_route_failure_cannot_hide_behind_overall_external() -> None:
    summary = _passing_summary()
    summary["external"]["accuracy"] = 0.90
    summary["external"]["categories"]["conditional_route"]["accuracy"] = 0.74
    gate = _model_gate(
        "scale_canary",
        summary,
        _contrasts(),
        _audit(),
    )
    assert gate["status"] == "category_calibration_failed"
    assert gate["conditional_route_external_accuracy"] == 0.74


def test_any_tie_error_or_nonfinite_rate_fails_integrity_gate() -> None:
    summary = _passing_summary()
    summary["external"]["tie_error_nonfinite_rate"] = 1 / 384
    gate = _model_gate(
        "scale_canary",
        summary,
        _contrasts(),
        _audit(),
    )
    assert gate["status"] == "scoring_integrity_failed"
    assert gate["eligible"] is False


def test_cross_scale_status_supports_bounded_capacity_interpretation() -> None:
    source_summary = _passing_summary()
    source_summary["external"]["accuracy"] = 0.50
    summaries = {
        "source_model": source_summary,
        "scale_canary": deepcopy(_passing_summary()),
    }
    gates = {
        "source_model": _model_gate(
            "source_model",
            source_summary,
            _contrasts(),
            _audit(),
        ),
        "scale_canary": _model_gate(
            "scale_canary",
            summaries["scale_canary"],
            _contrasts(),
            _audit(),
        ),
    }
    cross_scale = _cross_scale_gate(summaries, gates)
    assert cross_scale["status"] == "scale_capacity_bottleneck_supported"
    assert cross_scale["next_stage_status"] == (
        "training_complexity_review_eligible"
    )
    assert cross_scale["eligible_model_ids"] == ["scale_canary"]
    assert cross_scale["automatic_training_started"] is False
