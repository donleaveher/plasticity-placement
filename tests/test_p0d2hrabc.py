from __future__ import annotations

from collections import Counter

import pytest

from plasticity_placement.p0c.compiler import compile_bank
from plasticity_placement.p0d2h.probes import compile_hard_probe_bank
from plasticity_placement.p0d2hrab.config import AuditSpec as BindingSpec
from plasticity_placement.p0d2hrab.probes import compile_binding_bank
from plasticity_placement.p0d2hrabc.analysis import analyze_counterbalancing
from plasticity_placement.p0d2hrabc.config import AuditSpec
from plasticity_placement.p0d2hrabc.probes import compile_counterbalanced_bank
from plasticity_placement.p0d2hrsh.config import AuditSpec as HandoffSpec
from plasticity_placement.p0d2hrsh.probes import compile_handoff_bank
from plasticity_placement.p0d2hrtb.config import AuditSpec as BridgeSpec
from plasticity_placement.p0d2hrtb.probes import compile_bridge_bank

SELECTED_IDS = (
    "F_pair01_a",
    "F_pair01_b",
    "F_pair04_a",
    "F_pair04_b",
    "F_pair06_a",
    "F_pair06_b",
    "RF_pair01_a",
    "RF_pair01_b",
    "RF_pair04_a",
    "RF_pair04_b",
    "RF_pair05_a",
    "RF_pair05_b",
    "P_pair01_a",
    "P_pair01_b",
    "P_pair02_a",
    "P_pair02_b",
    "P_pair03_a",
    "P_pair03_b",
    "P_pair04_a",
    "P_pair04_b",
    "P_pair05_a",
    "P_pair05_b",
    "P_pair06_a",
    "P_pair06_b",
)


def _binding_bank():
    by_id = {item.lesson.lesson_id: item for item in compile_bank()}
    selected = tuple(by_id[lesson_id] for lesson_id in SELECTED_IDS)
    hard_bank = compile_hard_probe_bank(selected)
    bridge, _ = compile_bridge_bank(selected, hard_bank, BridgeSpec())
    handoff, _ = compile_handoff_bank(bridge, HandoffSpec())
    probes, _ = compile_binding_bank(handoff, BindingSpec())
    return probes


def test_counterbalanced_bank_crosses_all_factors_and_preserves_source_prompt() -> None:
    source = _binding_bank()
    probes, audit = compile_counterbalanced_bank(source, AuditSpec())

    assert len(probes) == 1_536
    assert audit["all_checks_passed"] is True
    assert audit["prompt_text_count"] == 384
    assert audit["prompt_text_multiplicity_counts"] == {4: 384}
    assert Counter(probe.selected_display_position for probe in probes) == {0: 768, 1: 768}
    assert Counter(probe.expected_candidate_position for probe in probes) == {
        0: 384,
        1: 384,
        2: 384,
        3: 384,
    }
    source_by_id = {probe.probe_id: probe for probe in source}
    assert all(
        probe.prompt == source_by_id[probe.source_probe_id].prompt
        for probe in probes
        if probe.display_order == "AB" and probe.candidate_rotation == 0
    )


def test_counterbalancing_spec_is_frozen_and_cannot_authorize_training() -> None:
    spec = AuditSpec()

    assert spec.prompt_count == 1_536
    assert spec.raw_decision_count == 3_072
    assert spec.allows_training is False
    assert spec.allows_mappings_per_adapter_scan is False
    with pytest.raises(ValueError, match="frozen"):
        AuditSpec(prompt_count=1_535)


@pytest.mark.parametrize(
    ("mode", "expected_status"),
    (
        ("receipt_b", "receipt_or_slot_label"),
        ("serial", "serial_position"),
        ("candidate0", "candidate_position_0"),
        ("interaction", "composition_interaction"),
        ("correct", "no_identified_penalty"),
    ),
)
def test_analysis_identifies_counterbalanced_failure_modes(mode: str, expected_status: str) -> None:
    analysis, records = analyze_counterbalancing(_synthetic_rows(mode), AuditSpec())

    assert len(records) == 1_536
    assert analysis["integrity"]["all_checks_passed"] is True
    assert analysis["causal_attribution"]["status"] == expected_status
    assert analysis["training_authorized"] is False
    assert analysis["mappings_per_adapter_authorized"] is False


def test_analysis_does_not_break_ties_by_candidate_order() -> None:
    analysis, _ = analyze_counterbalancing(_synthetic_rows("tie"), AuditSpec())

    assert analysis["analysis_status"] == "counterbalancing_complete"
    assert analysis["state_summaries"]["adapter"]["accuracy_identified_interval"] == [
        0.0,
        1.0,
    ]
    assert analysis["tie_diagnostics"]["ties_resolved_by_candidate_order"] is False
    assert analysis["causal_attribution"]["status"] == "no_identified_penalty"


def test_analysis_rejects_incomplete_factorial() -> None:
    rows = _synthetic_rows("correct")[:-2]

    analysis, _ = analyze_counterbalancing(rows, AuditSpec())

    assert analysis["analysis_status"] == "scoring_integrity_failed"
    assert analysis["causal_attribution"]["status"] == "scoring_integrity_failed"


def _synthetic_rows(mode: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair_index in range(12):
        pair_id = f"pair-{pair_index:02d}"
        for route_variant in range(4):
            unit_id = f"{pair_id}-r{route_variant}"
            base_candidates = ("act_a", "act_b", "act_c", "act_d")
            for orientation in ("canonical", "swapped"):
                for receipt in ("A", "B"):
                    expected = (
                        "act_a"
                        if (orientation, receipt) in {("canonical", "A"), ("swapped", "B")}
                        else "act_b"
                    )
                    counterfactual = "act_b" if expected == "act_a" else "act_a"
                    for display_order in ("AB", "BA"):
                        selected_position = display_order.index(receipt)
                        for rotation in range(4):
                            candidates = base_candidates[rotation:] + base_candidates[:rotation]
                            candidate_position = candidates.index(expected)
                            probe_id = (
                                f"{unit_id}-{orientation}-{receipt}-{display_order}-{rotation}"
                            )
                            static = {
                                "probe_id": probe_id,
                                "source_probe_id": f"{unit_id}-{orientation}-{receipt}",
                                "unit_id": unit_id,
                                "pair_id": pair_id,
                                "route_variant": route_variant,
                                "orientation": orientation,
                                "receipt": receipt,
                                "cell": f"{orientation}_{receipt}",
                                "display_order": display_order,
                                "selected_display_position": selected_position,
                                "candidate_rotation": rotation,
                                "expected_candidate_position": candidate_position,
                                "expected_action": expected,
                                "counterfactual_action": counterfactual,
                                "ordered_candidates": list(candidates),
                                "formatted_prompt_sha256": "a" * 64,
                            }
                            rows.append(_raw_row(static, "adapter_off", expected))
                            rows.append(
                                _raw_row(
                                    static,
                                    "adapter_on",
                                    _adapter_prediction(
                                        mode,
                                        expected,
                                        counterfactual,
                                        receipt,
                                        selected_position,
                                        candidate_position,
                                    ),
                                )
                            )
    return rows


def _adapter_prediction(
    mode: str,
    expected: str,
    counterfactual: str,
    receipt: str,
    selected_position: int,
    candidate_position: int,
) -> str | None:
    if mode == "tie":
        return None
    adverse = {
        "correct": False,
        "receipt_b": receipt == "B",
        "serial": selected_position == 1,
        "candidate0": candidate_position == 0,
        "interaction": (receipt == "B") != (selected_position == 1),
    }[mode]
    return counterfactual if adverse else expected


def _raw_row(static: dict[str, object], state: str, predicted: str | None) -> dict[str, object]:
    candidates = list(static["ordered_candidates"])
    tie = predicted is None
    expected = str(static["expected_action"])
    counterfactual = str(static["counterfactual_action"])
    top = {expected, counterfactual} if tie else {str(predicted)}
    scores = {candidate: -float(index + 2) for index, candidate in enumerate(candidates)}
    for candidate in top:
        scores[candidate] = 0.0
    unique_scores = sorted(set(scores.values()), reverse=True)
    return {
        **static,
        "adapter_state": state,
        "predicted_candidate": predicted,
        "candidates": [
            {
                "candidate": candidate,
                "sum_rank": unique_scores.index(scores[candidate]) + 1,
                "mean_rank": unique_scores.index(scores[candidate]) + 1,
                "sum_logprob": scores[candidate],
                "mean_logprob": scores[candidate],
            }
            for candidate in candidates
        ],
        "tie": tie,
        "non_finite": False,
        "error_status": "tie" if tie else "ok",
    }
