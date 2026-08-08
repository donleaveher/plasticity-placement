from __future__ import annotations

from plasticity_placement.p0d2hcbr.analysis import analyze_factorial, classify_unit
from plasticity_placement.p0d2hcbr.config import ExperimentSpec, GateSpec
from plasticity_placement.p0d2hcbr.data import compile_heldout_bank


def test_factorial_analysis_accepts_complete_unbiased_matrix() -> None:
    analysis, records = analyze_factorial(_synthetic_rows("correct"))

    assert len(records) == 1_536
    assert analysis["integrity"]["all_checks_passed"] is True
    assert analysis["state_summaries"]["adapter"]["accuracy_identified_interval"] == [1.0, 1.0]
    assert all(
        values["adapter"]["accuracy_midpoint"]["ci95"] == [0.0, 0.0]
        for values in analysis["main_effects"].values()
    )


def test_factorial_analysis_detects_literal_receipt_penalty() -> None:
    analysis, _ = analyze_factorial(_synthetic_rows("receipt"))

    interval = analysis["main_effects"]["receipt_B_minus_A"]["adapter"]["accuracy_midpoint"]["ci95"]
    assert interval[1] < 0.0


def test_factorial_analysis_rejects_incomplete_matrix() -> None:
    analysis, _ = analyze_factorial(_synthetic_rows("correct")[:-2])

    assert analysis["analysis_status"] == "factorial_integrity_failed"


def test_unit_gate_requires_binding_invariance_and_preservation() -> None:
    heldout, _ = analyze_factorial(_synthetic_rows("correct"))
    metric = _paired_metric()
    paired = {
        "forced_choice": {"external_conditional_route": metric},
        "crd": {
            "by_endpoint": {
                name: _paired_metric() for name in ("route_only", "retrieval_only", "combined")
            }
        },
    }
    guardrail = [
        {
            "lesson_id": f"lesson-{index:02d}",
            "base_correct": True,
            "adapter_correct": True,
            "base_tie": False,
            "adapter_tie": False,
            "base_tie_expected_compatible": True,
            "adapter_tie_expected_compatible": True,
            "base_error_status": "ok",
            "adapter_error_status": "ok",
        }
        for index in range(24)
    ]
    decision = classify_unit(
        heldout=heldout,
        rab={"adapter_on_conservative_binding_accuracy": {"estimate": 1.0}},
        paired=paired,
        guardrail_pairs=guardrail,
        gates=GateSpec(),
    )

    assert decision["status"] == "binding_remediation_candidate_review_required"
    assert all(decision["checks"].values())
    assert decision["mappings_per_adapter_authorized"] is False


def _synthetic_rows(mode: str) -> list[dict[str, object]]:
    probes, _ = compile_heldout_bank(ExperimentSpec().data)
    rows: list[dict[str, object]] = []
    for probe in probes:
        static = {key: value for key, value in probe.to_dict().items() if key not in {"prompt"}}
        static["ordered_candidates"] = list(probe.ordered_candidates)
        static["formatted_prompt_sha256"] = "a" * 64
        rows.append(_raw_row(static, "adapter_off", probe.expected_action))
        predicted = (
            probe.counterfactual_action
            if mode == "receipt" and probe.receipt == "B"
            else probe.expected_action
        )
        rows.append(_raw_row(static, "adapter_on", predicted))
    return rows


def _raw_row(static: dict[str, object], state: str, predicted: str) -> dict[str, object]:
    candidates = list(static["ordered_candidates"])
    scores = {candidate: -float(index + 2) for index, candidate in enumerate(candidates)}
    scores[predicted] = 0.0
    unique = sorted(set(scores.values()), reverse=True)
    return {
        **static,
        "adapter_state": state,
        "predicted_candidate": predicted,
        "candidates": [
            {
                "candidate": candidate,
                "sum_rank": unique.index(scores[candidate]) + 1,
                "mean_rank": unique.index(scores[candidate]) + 1,
                "sum_logprob": scores[candidate],
                "mean_logprob": scores[candidate],
            }
            for candidate in candidates
        ],
        "tie": False,
        "non_finite": False,
        "error_status": "ok",
    }


def _paired_metric() -> dict[str, object]:
    tie = {
        "bounds_valid": True,
        "all_ties_incorrect_accuracy": 1.0,
        "all_ties_compatible_accuracy": 1.0,
    }
    return {
        "tie_sensitivity": {"base": tie, "adapter": tie},
        "paired_accuracy_difference": {"estimate": 0.0, "ci95": [0.0, 0.0]},
    }
