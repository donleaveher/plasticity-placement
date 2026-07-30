from __future__ import annotations

from types import SimpleNamespace

import pytest

from plasticity_placement.p0d2hcrd.scoring import rank_candidate_scores
from plasticity_placement.p0d2hrr.analysis import (
    _route_remediation_gate,
    _verify_ranked_outcome,
)
from plasticity_placement.p0d2hrr.cli import build_parser
from plasticity_placement.p0d2hrr.config import PilotSpec


def _endpoint_summary(
    route: float = 0.91,
    retrieval: float = 1.0,
    combined: float = 0.98,
) -> dict[str, dict[str, float]]:
    return {
        "route_only": {"accuracy": route},
        "retrieval_only": {"accuracy": retrieval},
        "combined": {"accuracy": combined},
    }


def _rows(count: int = 8) -> list[dict[str, str]]:
    return [{"error_status": "ok"} for _ in range(count)]


def test_route_remediation_gate_requires_original_gate_and_guardrails() -> None:
    config = SimpleNamespace(spec=PilotSpec())
    gate = _route_remediation_gate(
        config,
        {"eligible": True, "conditional_route_external_accuracy": 0.75},
        _endpoint_summary(),
        _rows(),
        _rows(),
    )
    assert gate["status"] == "route_remediation_passed_review_required"
    assert gate["training_complexity_review_eligible"] is True
    assert gate["automatic_training_started"] is False

    route_boundary = _route_remediation_gate(
        config,
        {"eligible": True, "conditional_route_external_accuracy": 0.75},
        _endpoint_summary(route=0.625),
        _rows(),
        _rows(),
    )
    assert route_boundary["status"] == "route_guardrail_failed"
    assert route_boundary["checks"]["route_only_improved"] is False

    original_failed = _route_remediation_gate(
        config,
        {"eligible": False, "conditional_route_external_accuracy": 0.80},
        _endpoint_summary(),
        _rows(),
        _rows(),
    )
    assert original_failed["status"] == "original_calibration_gate_failed"
    assert original_failed["training_complexity_review_eligible"] is False


def test_any_tie_or_error_blocks_route_remediation_pass() -> None:
    config = SimpleNamespace(spec=PilotSpec())
    crd_rows = _rows()
    crd_rows[0]["error_status"] = "tie"
    gate = _route_remediation_gate(
        config,
        {"eligible": True, "conditional_route_external_accuracy": 0.80},
        _endpoint_summary(),
        _rows(),
        crd_rows,
    )
    assert gate["checks"]["crd_scoring_integrity"] is False
    assert gate["training_complexity_review_eligible"] is False


def test_cli_has_no_automatic_run_or_scan_command() -> None:
    parser = build_parser()
    commands = next(
        action.choices
        for action in parser._actions
        if hasattr(action, "choices") and isinstance(action.choices, dict)
    )
    assert set(commands) == {
        "environment",
        "plan",
        "authorize",
        "train",
        "evaluate",
        "aggregate",
        "status",
    }


def test_aggregate_recomputes_candidate_ranking() -> None:
    candidates = [
        {
            "candidate": "slot_a",
            "sum_logprob": -0.1,
            "mean_logprob": -0.1,
        },
        {
            "candidate": "slot_b",
            "sum_logprob": -1.0,
            "mean_logprob": -1.0,
        },
    ]
    ranked = rank_candidate_scores(candidates)
    row = {
        **ranked,
        "expected_candidate": "slot_a",
        "correct": True,
    }
    _verify_ranked_outcome(row, "p0d2hrr-dev-decision-row-v1")
    row["predicted_candidate"] = "slot_b"
    with pytest.raises(ValueError, match="ranking differs"):
        _verify_ranked_outcome(row, "p0d2hrr-dev-decision-row-v1")
