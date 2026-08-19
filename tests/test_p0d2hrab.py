from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from plasticity_placement.p0c.compiler import compile_bank
from plasticity_placement.p0d2h.probes import compile_hard_probe_bank
from plasticity_placement.p0d2hrab.analysis import analyze_binding
from plasticity_placement.p0d2hrab.config import CELLS, AuditSpec
from plasticity_placement.p0d2hrab.probes import compile_binding_bank
from plasticity_placement.p0d2hrab.runtime import (
    MANIFEST_SCHEMA_VERSION,
    _load_handoff_probes,
    _validate_rsh_result,
    authorization_template,
    authorize_audit,
    build_rsh_quadrant_diagnostic,
    verify_complete_result,
)
from plasticity_placement.p0d2hrr.io import file_hash
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


def _handoff_bank():
    by_id = {item.lesson.lesson_id: item for item in compile_bank()}
    selected = tuple(by_id[lesson_id] for lesson_id in SELECTED_IDS)
    hard_bank = compile_hard_probe_bank(selected)
    bridge, _ = compile_bridge_bank(selected, hard_bank, BridgeSpec())
    probes, _ = compile_handoff_bank(bridge, HandoffSpec())
    return probes


def test_binding_bank_has_two_valid_mappings_and_balanced_counterfactuals() -> None:
    probes, audit = compile_binding_bank(_handoff_bank(), AuditSpec())

    assert len(probes) == 192
    assert audit["all_checks_passed"] is True
    assert Counter(probe.cell for probe in probes) == {cell: 48 for cell in CELLS}
    assert all(
        probe.expected_action != probe.counterfactual_action
        and probe.expected_action in probe.ordered_candidates
        and probe.counterfactual_action in probe.ordered_candidates
        and probe.lesson_a_id not in probe.prompt  # provenance IDs must not leak into prompts
        for probe in probes
    )
    units: dict[str, list] = {}
    for probe in probes:
        units.setdefault(probe.unit_id, []).append(probe)
    assert all(
        next(
            probe.expected_action
            for probe in rows
            if probe.orientation == "canonical" and probe.receipt == "A"
        )
        == next(
            probe.expected_action
            for probe in rows
            if probe.orientation == "swapped" and probe.receipt == "B"
        )
        for rows in units.values()
    )


def test_json_roundtrip_restores_frozen_candidate_tuples(tmp_path: Path) -> None:
    rsh_output = tmp_path / "rsh"
    (rsh_output / "preflight").mkdir(parents=True)
    source = _handoff_bank()
    (rsh_output / "preflight" / "handoff_probes.jsonl").write_text(
        "\n".join(json.dumps(probe.to_dict()) for probe in source) + "\n"
    )

    loaded = _load_handoff_probes(rsh_output)
    probes, audit = compile_binding_bank(loaded, AuditSpec())

    assert isinstance(loaded[0].action_candidates, tuple)
    assert isinstance(probes[0].ordered_candidates, tuple)
    assert audit["all_checks_passed"] is True


def test_binding_spec_rejects_post_result_changes() -> None:
    with pytest.raises(ValueError, match="frozen"):
        AuditSpec(binding_accuracy_threshold=0.70)
    with pytest.raises(ValueError, match="frozen"):
        AuditSpec(specificity_threshold=0.10)


def test_binding_analysis_supports_causal_two_valid_slot_readout() -> None:
    result = analyze_binding(_synthetic_rows(mode="correct"), AuditSpec())

    assert result["decision"]["status"] == "binding_supported"
    assert result["adapter_on_conservative_binding_accuracy"]["estimate"] == 1.0
    assert result["adapter_on_conservative_specificity"]["estimate"] == 1.0
    assert result["adapter_on_minus_off_noninferiority"]["estimate"] == 0.0
    assert result["compliance"]["full_factorial"]["observed"]["adapter"] == 1.0
    assert result["decision"]["training_authorized"] is False
    assert result["decision"]["mappings_per_adapter_authorized"] is False


def test_binding_analysis_rejects_receipt_invariant_action_prior() -> None:
    result = analyze_binding(_synthetic_rows(mode="fixed_a"), AuditSpec())

    assert result["decision"]["status"] == "binding_not_supported"
    assert result["adapter_on_conservative_binding_accuracy"]["estimate"] == 0.5
    assert result["adapter_on_conservative_specificity"]["estimate"] == 0.0
    assert result["decision"]["causal_specificity_passed"] is False


def test_binding_analysis_propagates_action_ties_without_fabricated_winner() -> None:
    result = analyze_binding(_synthetic_rows(mode="adapter_tie"), AuditSpec())

    assert result["decision"]["status"] == "binding_not_supported"
    assert result["decision"]["scoring_integrity"] is True
    assert result["adapter_on_conservative_binding_accuracy"]["estimate"] == 0.0
    assert result["adapter_on_conservative_specificity"]["estimate"] == -1.0
    assert result["tie_diagnostics"]["tie_count"] == 192


def test_binding_analysis_classifies_nonfinite_scores_as_integrity_failure() -> None:
    rows = _synthetic_rows(mode="correct")
    rows[0]["non_finite"] = True
    rows[0]["error_status"] = "non_finite"
    rows[0]["predicted_candidate"] = None
    rows[0]["candidates"] = []

    result = analyze_binding(rows, AuditSpec())

    assert result["decision"]["status"] == "scoring_integrity_failed"
    assert result["decision"]["scoring_integrity"] is False


def test_binding_analysis_rejects_per_unit_factorial_corruption() -> None:
    rows = _synthetic_rows(mode="correct")
    first_unit = str(rows[0]["unit_id"])
    second_unit = next(str(row["unit_id"]) for row in rows if row["unit_id"] != first_unit)
    for row in rows:
        if row["unit_id"] == first_unit and row["cell"] == "canonical_A":
            row.update({"orientation": "canonical", "receipt": "B", "cell": "canonical_B"})
        elif row["unit_id"] == second_unit and row["cell"] == "canonical_B":
            row.update({"orientation": "canonical", "receipt": "A", "cell": "canonical_A"})

    result = analyze_binding(rows, AuditSpec())

    assert result["decision"]["status"] == "scoring_integrity_failed"
    assert result["decision"]["scoring_integrity"] is False


@pytest.mark.parametrize("corruption", ("candidate_order", "tie_flag", "nonfinite_score"))
def test_binding_analysis_rejects_malformed_scorer_contract(corruption: str) -> None:
    rows = _synthetic_rows(mode="correct")
    row = rows[0]
    if corruption == "candidate_order":
        row["candidates"] = list(reversed(row["candidates"]))
    elif corruption == "tie_flag":
        for candidate in row["candidates"]:
            candidate["sum_rank"] = 1
            candidate["sum_logprob"] = 0.0
            candidate["mean_logprob"] = 0.0
    elif corruption == "nonfinite_score":
        row["candidates"][0]["sum_logprob"] = float("inf")
    else:
        raise AssertionError(corruption)

    result = analyze_binding(rows, AuditSpec())

    assert result["decision"]["status"] == "scoring_integrity_failed"
    assert result["decision"]["scoring_integrity"] is False


def test_compliance_difference_subtracts_base_upper_under_ties() -> None:
    rows = _synthetic_rows(mode="correct")
    for row in rows:
        if row["adapter_state"] == "adapter_off":
            row["tie"] = True
            row["error_status"] = "tie"
            row["predicted_candidate"] = None
            for candidate in row["candidates"]:
                candidate["sum_rank"] = 1
                candidate["sum_logprob"] = 0.0
                candidate["mean_logprob"] = 0.0

    result = analyze_binding(rows, AuditSpec())

    difference = result["compliance"]["receipt_pair"][
        "adapter_minus_base_conservative"
    ]
    assert difference["estimate"] == 0.0
    assert difference["ci95"] == [0.0, 0.0]


def test_rsh_quadrant_diagnostic_reports_paired_patterns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import plasticity_placement.p0d2hrab.runtime as runtime

    output = tmp_path / "rsh"
    output.mkdir()
    (output / "summary.json").write_text("{}")
    rows = []
    for index in range(96):
        rows.append(
            {
                "probe_id": f"probe-{index:03d}",
                "base": _rsh_state(index, both=82, oracle_only=5),
                "adapter": _rsh_state(index, both=76, oracle_only=13),
            }
        )
    (output / "paired_records.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    monkeypatch.setattr(runtime, "_validate_rsh_result", lambda _: {})

    diagnostic = build_rsh_quadrant_diagnostic(output)

    assert diagnostic["all_checks_passed"] is True
    assert diagnostic["state_results"]["base"]["counts"] == {
        "both_target": 82,
        "oracle_only_target": 5,
        "wrong_only_target": 0,
        "neither_target": 9,
    }
    assert diagnostic["state_results"]["adapter"]["counts"] == {
        "both_target": 76,
        "oracle_only_target": 13,
        "wrong_only_target": 0,
        "neither_target": 7,
    }


def test_rsh_source_validation_rejects_preflight_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import plasticity_placement.p0d2hrab.runtime as runtime

    output = tmp_path / "rsh"
    output.mkdir()
    for name in ("manifest.json", "preregistration.json", "summary.json"):
        (output / name).write_text("{}")
    monkeypatch.setattr(runtime, "verify_rsh_result", lambda _: output / "summary.json")
    monkeypatch.setattr(
        runtime,
        "_verify_rsh_preflight",
        lambda *_: (_ for _ in ()).throw(ValueError("preflight mutation")),
    )

    with pytest.raises(ValueError, match="preflight mutation"):
        _validate_rsh_result(output)


def test_binding_authorization_is_external_and_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    preregistration_sha256 = "a" * 64
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": "binding-test",
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
    assert json.loads(first.read_text())["state"] == "authorized"


def test_complete_result_verification_rejects_mutated_artifact(tmp_path: Path) -> None:
    output = _complete_result_fixture(tmp_path)

    assert verify_complete_result(output) == output / "summary.json"
    (output / "paired_records.jsonl").write_text("mutated\n")
    with pytest.raises(ValueError, match="published artifacts changed"):
        verify_complete_result(output)


def _synthetic_rows(*, mode: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair_index in range(12):
        pair_id = f"pair-{pair_index:02d}"
        for variant in range(1, 5):
            unit_id = f"{pair_id}-r{variant}"
            action_a = "act_a"
            action_b = "act_b"
            candidates = (action_a, action_b, "act_c", "act_d")
            for orientation in ("canonical", "swapped"):
                for receipt in ("A", "B"):
                    if orientation == "canonical":
                        expected = action_a if receipt == "A" else action_b
                    else:
                        expected = action_b if receipt == "A" else action_a
                    other = action_b if expected == action_a else action_a
                    for state in ("adapter_off", "adapter_on"):
                        tie = mode == "adapter_tie" and state == "adapter_on"
                        predicted = (
                            None
                            if tie
                            else action_a
                            if mode == "fixed_a" and state == "adapter_on"
                            else expected
                        )
                        rows.append(
                            {
                                "probe_id": f"{unit_id}-{orientation}-{receipt}",
                                "unit_id": unit_id,
                                "pair_id": pair_id,
                                "route_variant": variant,
                                "orientation": orientation,
                                "receipt": receipt,
                                "cell": f"{orientation}_{receipt}",
                                "expected_action": expected,
                                "counterfactual_action": other,
                                "ordered_candidates": list(candidates),
                                "adapter_state": state,
                                "predicted_candidate": predicted,
                                "candidates": [
                                    {
                                        "candidate": candidate,
                                        "sum_rank": 1
                                        if tie or candidate == predicted
                                        else 2 + candidates.index(candidate),
                                        "sum_logprob": 0.0
                                        if tie or candidate == predicted
                                        else -float(2 + candidates.index(candidate)),
                                        "mean_logprob": 0.0
                                        if tie or candidate == predicted
                                        else -float(2 + candidates.index(candidate)),
                                    }
                                    for candidate in candidates
                                ],
                                "tie": tie,
                                "non_finite": False,
                                "error_status": "tie" if tie else "ok",
                            }
                        )
    return rows


def _rsh_state(index: int, *, both: int, oracle_only: int) -> dict[str, object]:
    oracle = index < both + oracle_only
    wrong = index < both
    return {
        "oracle_correct": float(oracle),
        "wrong_target_selected": float(wrong),
        "oracle_lower": float(oracle),
        "oracle_upper": float(oracle),
        "wrong_target_lower": float(wrong),
        "wrong_target_upper": float(wrong),
    }


def _complete_result_fixture(tmp_path: Path) -> Path:
    output = tmp_path / "complete"
    (output / "results").mkdir(parents=True)
    run_id = "p0d2hrab-fixture"
    preregistration_sha256 = "b" * 64
    (output / "results" / "adapter_off.jsonl").write_text("{}\n" * 192)
    (output / "results" / "adapter_on.jsonl").write_text("{}\n" * 192)
    (output / "paired_records.jsonl").write_text("{}\n" * 192)
    (output / "report.md").write_text("# report\n")
    summary = {
        "run_id": run_id,
        "analysis": {"decision": {"status": "binding_not_supported"}},
        "historical_rsh_decision_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    (output / "summary.json").write_text(json.dumps(summary))
    raw_artifacts = {
        "adapter_off": file_hash(output / "results" / "adapter_off.jsonl"),
        "adapter_on": file_hash(output / "results" / "adapter_on.jsonl"),
    }
    checkpoint = {
        "schema_version": "p0d2hrab-raw-score-checkpoint-v1",
        "run_id": run_id,
        "adapter_off_count": 192,
        "adapter_on_count": 192,
        "artifacts": raw_artifacts,
    }
    (output / "raw_score_checkpoint.json").write_text(json.dumps(checkpoint))
    artifacts = {
        "summary": file_hash(output / "summary.json"),
        "report": file_hash(output / "report.md"),
        "paired_records": file_hash(output / "paired_records.jsonl"),
        **raw_artifacts,
        "raw_score_checkpoint": file_hash(output / "raw_score_checkpoint.json"),
    }
    audit = {
        "schema_version": "p0d2hrab-audit-manifest-v1",
        "run_id": run_id,
        "preregistration_sha256": preregistration_sha256,
        "artifacts": artifacts,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    (output / "audit_manifest.json").write_text(json.dumps(audit))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "state": "complete",
        "config": {"preregistration_sha256": preregistration_sha256},
        "result": {
            "summary_sha256": artifacts["summary"],
            "audit_manifest_sha256": file_hash(output / "audit_manifest.json"),
            "decision": "binding_not_supported",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    return output
