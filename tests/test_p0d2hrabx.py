from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from plasticity_placement.p0c.compiler import compile_bank
from plasticity_placement.p0d2h.probes import compile_hard_probe_bank
from plasticity_placement.p0d2hrab.config import AuditSpec as BindingSpec
from plasticity_placement.p0d2hrab.probes import compile_binding_bank
from plasticity_placement.p0d2hrabx.analysis import analyze_label_disentanglement
from plasticity_placement.p0d2hrabx.config import AuditSpec
from plasticity_placement.p0d2hrabx.probes import compile_label_disentanglement_bank
from plasticity_placement.p0d2hrabx.runtime import (
    MANIFEST_SCHEMA_VERSION,
    authorization_template,
    authorize_audit,
)
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


def test_label_bank_independently_crosses_receipt_and_selected_slot() -> None:
    probes, audit = compile_label_disentanglement_bank(_binding_bank(), AuditSpec())

    assert len(probes) == 3_072
    assert audit["all_checks_passed"] is True
    assert audit["prompt_text_count"] == 768
    assert audit["prompt_text_multiplicity_counts"] == {4: 768}
    assert Counter((row.receipt, row.selected_slot_label) for row in probes) == {
        (receipt, slot): 768 for receipt in ("A", "B") for slot in ("A", "B")
    }
    assert Counter(row.selected_display_position for row in probes) == {0: 1536, 1: 1536}
    assert Counter(row.expected_candidate_position for row in probes) == {
        0: 768,
        1: 768,
        2: 768,
        3: 768,
    }


def test_label_spec_is_frozen_and_never_authorizes_training() -> None:
    spec = AuditSpec()

    assert spec.expected_rabc_run_id == "p0d2hrabc-c4f96876bb"
    assert spec.prompt_count == 3_072
    assert spec.raw_decision_count == 6_144
    assert spec.allows_training is False
    assert spec.allows_mappings_per_adapter_scan is False
    with pytest.raises(ValueError, match="frozen"):
        AuditSpec(prompt_count=3_071)


def test_label_authorization_is_external_safe_and_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    preregistration_sha256 = "a" * 64
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": "label-test",
        "state": "planned",
        "config": {"preregistration_sha256": preregistration_sha256},
        "authorization": None,
        "errors": [],
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    approval = {
        **authorization_template(preregistration_sha256),
        "decision": "approved",
        "approved_by": "reviewer",
        "approved_at": "2026-08-07T00:00:00+00:00",
    }
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(approval))

    first = authorize_audit(output, approval_path)
    second = authorize_audit(output, approval_path)

    assert first == second == output / "manifest.json"
    adopted = json.loads((output / "authorization.json").read_text())
    assert adopted["allows_training"] is False
    assert adopted["allows_historical_reclassification"] is False


@pytest.mark.parametrize(
    ("mode", "expected_flag"),
    (("receipt", "receipt_token"), ("slot", "slot_label")),
)
def test_analysis_separates_receipt_token_from_slot_label(
    mode: str, expected_flag: str
) -> None:
    analysis, records = analyze_label_disentanglement(_synthetic_rows(mode), AuditSpec())

    assert len(records) == 3_072
    assert analysis["integrity"]["all_checks_passed"] is True
    assert analysis["causal_attribution"]["status"] == expected_flag
    assert analysis["causal_attribution"]["flags"][expected_flag] is True
    other = "slot_label" if expected_flag == "receipt_token" else "receipt_token"
    assert analysis["causal_attribution"]["flags"][other] is False
    assert analysis["training_authorized"] is False
    assert analysis["mappings_per_adapter_authorized"] is False


def test_analysis_rejects_incomplete_label_factorial() -> None:
    analysis, _ = analyze_label_disentanglement(_synthetic_rows("correct")[:-2], AuditSpec())

    assert analysis["analysis_status"] == "scoring_integrity_failed"


def _synthetic_rows(mode: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair_index in range(12):
        pair_id = f"pair-{pair_index:02d}"
        for route_variant in range(4):
            unit_id = f"{pair_id}-r{route_variant}"
            base_candidates = ("act_a", "act_b", "act_c", "act_d")
            route_candidates = (
                base_candidates[route_variant:] + base_candidates[:route_variant]
            )
            for orientation in ("canonical", "swapped"):
                actions = (
                    {"A": "act_a", "B": "act_b"}
                    if orientation == "canonical"
                    else {"A": "act_b", "B": "act_a"}
                )
                for receipt in ("A", "B"):
                    for codebook in ("canonical", "crossed"):
                        selected = (
                            receipt
                            if codebook == "canonical"
                            else "B"
                            if receipt == "A"
                            else "A"
                        )
                        expected = actions[selected]
                        counterfactual = actions["B" if selected == "A" else "A"]
                        for display_order in ("AB", "BA"):
                            selected_position = display_order.index(selected)
                            for rotation in range(4):
                                candidates = (
                                    route_candidates[rotation:] + route_candidates[:rotation]
                                )
                                candidate_position = candidates.index(expected)
                                probe_id = (
                                    f"{unit_id}-{orientation}-{receipt}-{codebook}-"
                                    f"{display_order}-{rotation}"
                                )
                                static = {
                                    "probe_id": probe_id,
                                    "source_probe_id": f"{unit_id}-{orientation}-{selected}",
                                    "unit_id": unit_id,
                                    "pair_id": pair_id,
                                    "route_variant": route_variant,
                                    "orientation": orientation,
                                    "receipt": receipt,
                                    "codebook": codebook,
                                    "selected_slot_label": selected,
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
                                adverse = (
                                    (mode == "receipt" and receipt == "B")
                                    or (mode == "slot" and selected == "B")
                                )
                                predicted = counterfactual if adverse else expected
                                rows.append(_raw_row(static, "adapter_on", predicted))
    return rows


def _raw_row(
    static: dict[str, object], state: str, predicted: str
) -> dict[str, object]:
    candidates = list(static["ordered_candidates"])
    scores = {candidate: -float(index + 2) for index, candidate in enumerate(candidates)}
    scores[predicted] = 0.0
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
        "tie": False,
        "non_finite": False,
        "error_status": "ok",
    }
