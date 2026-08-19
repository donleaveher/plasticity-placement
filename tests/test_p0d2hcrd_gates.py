from __future__ import annotations

import pytest

from plasticity_placement.p0d2hcrd.analysis import diagnostic_gate


def _summary(
    route: float = 0.90,
    retrieval: float = 0.90,
    combined: float = 0.75,
    invalid: float = 0.0,
) -> dict[str, dict[str, float]]:
    return {
        "route_only": {
            "accuracy": route,
            "tie_error_nonfinite_rate": invalid,
        },
        "retrieval_only": {
            "accuracy": retrieval,
            "tie_error_nonfinite_rate": invalid,
        },
        "combined": {
            "accuracy": combined,
            "tie_error_nonfinite_rate": invalid,
        },
    }


@pytest.mark.parametrize(
    ("route", "retrieval", "combined", "status"),
    [
        (0.89, 0.90, 0.75, "route_bottleneck_supported"),
        (0.90, 0.89, 0.75, "retrieval_bottleneck_supported"),
        (0.89, 0.89, 0.75, "shared_component_failure"),
        (0.90, 0.90, 0.74, "composition_bottleneck_supported"),
        (0.90, 0.90, 0.75, "no_component_bottleneck_detected"),
    ],
)
def test_diagnostic_statuses_are_prospectively_exhaustive(
    route: float,
    retrieval: float,
    combined: float,
    status: str,
) -> None:
    gate = diagnostic_gate(
        _summary(route, retrieval, combined),
        {"all_checks_passed": True},
        {"all_checks_passed": True, "input_truncated_count": 0},
    )
    assert gate["status"] == status
    assert gate["training_complexity_review_eligible"] is False
    assert gate["automatic_training_started"] is False
    assert gate["automatic_narrow_scan_started"] is False


@pytest.mark.parametrize(
    ("bank", "token", "invalid"),
    [
        (False, True, 0.0),
        (True, False, 0.0),
        (True, True, 1 / 1_536),
    ],
)
def test_any_integrity_failure_overrides_component_interpretation(
    bank: bool,
    token: bool,
    invalid: float,
) -> None:
    gate = diagnostic_gate(
        _summary(invalid=invalid),
        {"all_checks_passed": bank},
        {"all_checks_passed": token, "input_truncated_count": 0},
    )
    assert gate["status"] == "scoring_integrity_failed"

