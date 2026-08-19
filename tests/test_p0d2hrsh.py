from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from plasticity_placement.p0c.compiler import compile_bank
from plasticity_placement.p0d2h.probes import compile_hard_probe_bank
from plasticity_placement.p0d2hrr.io import file_hash
from plasticity_placement.p0d2hrsh.analysis import analyze_handoff
from plasticity_placement.p0d2hrsh.config import PROMPT_KINDS, AuditSpec
from plasticity_placement.p0d2hrsh.probes import compile_handoff_bank
from plasticity_placement.p0d2hrsh.runtime import (
    MANIFEST_SCHEMA_VERSION,
    _audit_handoff_anchors,
    _build_replay_audit,
    authorization_template,
    authorize_audit,
    verify_complete_result,
)
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


def _selected():
    by_id = {item.lesson.lesson_id: item for item in compile_bank()}
    return tuple(by_id[lesson_id] for lesson_id in SELECTED_IDS)


def test_handoff_bank_reuses_exact_rtb_anchors_and_freezes_both_receipts() -> None:
    selected = _selected()
    hard_bank = compile_hard_probe_bank(selected)
    bridge, _ = compile_bridge_bank(selected, hard_bank, BridgeSpec())

    probes, audit = compile_handoff_bank(bridge, AuditSpec())

    assert len(probes) == 96
    assert audit["all_checks_passed"] is True
    assert Counter(probe.target_slot for probe in probes) == {"A": 48, "B": 48}
    assert all(
        probe.prompt("slot_readout")
        == next(item.prompt for item in bridge if item.probe_id == probe.rtb_slot_probe_id)
        and probe.prompt("direct_action")
        == next(item.prompt for item in bridge if item.probe_id == probe.rtb_direct_probe_id)
        and probe.prompt(f"receipt_{probe.target_slot}")
        == next(item.prompt for item in bridge if item.probe_id == probe.rtb_oracle_probe_id)
        for probe in probes
    )
    repeated, repeated_audit = compile_handoff_bank(bridge, AuditSpec())
    assert probes == repeated
    assert audit == repeated_audit


def test_handoff_spec_rejects_post_result_changes() -> None:
    with pytest.raises(ValueError, match="frozen"):
        AuditSpec(material_handoff_rescue=0.05)
    with pytest.raises(ValueError, match="frozen"):
        AuditSpec(chain_oracle_noninferiority_margin=0.10)


def test_handoff_analysis_supports_material_tie_robust_rescue() -> None:
    rows = _synthetic_rows(slot_tie=False)

    result = analyze_handoff(rows, AuditSpec())

    assert result["decision"]["status"] == "handoff_supported"
    assert result["primary_handoff_rescue"]["supported"] is True
    assert (
        result["primary_handoff_rescue"]["conservative_compatible_tie_bound"]["estimate"]
        == 1.0
    )
    assert result["safeguards"]["adapter_on_chain_minus_oracle_conservative"]["passed"]
    assert result["safeguards"]["adapter_on_oracle_minus_wrong_target_conservative"]["passed"]
    assert result["decision"]["training_authorized"] is False
    assert result["decision"]["mappings_per_adapter_authorized"] is False


def test_handoff_analysis_propagates_slot_tie_to_conservative_chain() -> None:
    rows = _synthetic_rows(slot_tie=True)

    result = analyze_handoff(rows, AuditSpec())

    primary = result["primary_handoff_rescue"]
    assert primary["observed"]["estimate"] == 1.0
    assert primary["conservative_compatible_tie_bound"]["estimate"] == 0.0
    assert primary["supported"] is False
    assert result["decision"]["status"] == "handoff_not_supported"
    assert result["tie_diagnostics"]["slot_ties"] == 96


def test_chain_oracle_bound_cancels_shared_tied_receipt() -> None:
    rows = _synthetic_rows(slot_tie=False)
    for row in rows:
        if (
            row["adapter_state"] == "adapter_on"
            and row["prompt_kind"] == f"receipt_{row['target_slot']}"
        ):
            row["tie"] = True
            row["error_status"] = "tie"
            row["predicted_candidate"] = None
            for candidate in row["candidates"]:
                candidate["sum_rank"] = 1

    result = analyze_handoff(rows, AuditSpec())

    safeguard = result["safeguards"]["adapter_on_chain_minus_oracle_conservative"]
    assert safeguard["estimate"] == 0.0
    assert safeguard["ci95"] == [0.0, 0.0]
    assert safeguard["passed"] is True


def test_handoff_authorization_is_external_and_idempotent(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    preregistration_sha256 = "a" * 64
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": "handoff-test",
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
        "approved_at": "2026-08-06T00:00:00+00:00",
    }
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(approval))

    first = authorize_audit(output, approval_path)
    second = authorize_audit(output, approval_path)

    assert first == second == output / "manifest.json"
    assert json.loads(first.read_text())["state"] == "authorized"


def test_replay_audit_accepts_only_reviewed_compatible_tie_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import plasticity_placement.p0d2hrsh.runtime as runtime

    rtb_output = tmp_path / "rtb"
    qualification = tmp_path / "q2" / "summary.json"
    qualification.parent.mkdir(parents=True)
    qualification.write_text("{}")
    rtb_output.mkdir()
    (rtb_output / "summary.json").write_text("{}")
    primary = _primary_replay_rows()
    external, q2 = _external_replay_rows()
    (rtb_output / "paired_records.jsonl").write_text(
        "\n".join(json.dumps(row) for row in [*primary, *external]) + "\n"
    )
    (qualification.parent / "paired_records.jsonl").write_text(
        "\n".join(json.dumps(row) for row in q2) + "\n"
    )
    monkeypatch.setattr(
        runtime,
        "_validate_rtb_result",
        lambda _: {
            "summary": {"run_id": "rtb-source"},
            "rtb_identity": {"qualification_summary": str(qualification)},
        },
    )

    audit = _build_replay_audit(rtb_output)

    assert audit["all_checks_passed"] is True
    assert audit["violation_count"] == 26
    assert audit["tie_counts"] == {"base": 18, "adapter": 8}
    assert audit["q2_to_rtb_decision_drift"] == {"base": 3, "adapter": 1}


def test_handoff_anchor_audit_rejects_drift_from_bound_rtb_rows() -> None:
    selected = _selected()
    hard_bank = compile_hard_probe_bank(selected)
    bridge, _ = compile_bridge_bank(selected, hard_bank, BridgeSpec())
    probes, _ = compile_handoff_bank(bridge, AuditSpec())
    probe = probes[0]
    anchors = (
        (
            "slot_readout",
            probe.rtb_slot_probe_id,
            probe.target_slot,
            list(probe.slot_candidates),
        ),
        (
            "direct_action",
            probe.rtb_direct_probe_id,
            probe.target_action,
            list(probe.action_candidates),
        ),
        (
            f"receipt_{probe.target_slot}",
            probe.rtb_oracle_probe_id,
            probe.target_action,
            list(probe.action_candidates),
        ),
    )
    token_records = [
        {"probe_id": probe.probe_id, "prompt_kind": kind, "prompt_sha256": f"hash-{kind}"}
        for kind, _, _, _ in anchors
    ]
    pairs = [
        {
            "probe_id": rtb_probe_id,
            "source_probe_id": probe.source_probe_id,
            "expected_candidate": expected,
            "ordered_candidates": candidates,
            "prompt_sha256": f"hash-{kind}",
        }
        for kind, rtb_probe_id, expected, candidates in anchors
    ]

    assert _audit_handoff_anchors([probe], token_records, pairs)["all_checks_passed"]
    pairs[0]["prompt_sha256"] = "mutated"
    drift = _audit_handoff_anchors([probe], token_records, pairs)
    assert drift["all_checks_passed"] is False
    assert drift["mismatches"] == [
        {
            "probe_id": probe.probe_id,
            "prompt_kind": "slot_readout",
            "rtb_probe_id": probe.rtb_slot_probe_id,
        }
    ]


def test_complete_result_verification_rejects_mutated_published_artifact(
    tmp_path: Path,
) -> None:
    output = _complete_result_fixture(tmp_path)

    assert verify_complete_result(output) == output / "summary.json"
    (output / "report.md").write_text("mutated")
    with pytest.raises(ValueError, match="published artifacts changed"):
        verify_complete_result(output)


def _synthetic_rows(*, slot_tie: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(96):
        target_slot = "A" if index % 2 == 0 else "B"
        other_slot = "B" if target_slot == "A" else "A"
        target_action = "act_target"
        for state in ("adapter_off", "adapter_on"):
            for kind in PROMPT_KINDS:
                if kind == "slot_readout":
                    tie = slot_tie and state == "adapter_on"
                    predicted = None if tie else target_slot
                    candidates = _candidates(
                        (target_slot, other_slot), predicted=target_slot, tie=tie
                    )
                    ordered_candidates = [target_slot, other_slot]
                else:
                    receipt_slot = kind.removeprefix("receipt_")
                    if kind == "direct_action":
                        correct = state == "adapter_off"
                    else:
                        correct = receipt_slot == target_slot
                    predicted = target_action if correct else "act_wrong"
                    tie = False
                    candidates = _candidates(
                        (target_action, "act_wrong"), predicted=predicted, tie=False
                    )
                    ordered_candidates = [target_action, "act_wrong"]
                rows.append(
                    {
                        "probe_id": f"probe-{index:03d}",
                        "lesson_id": f"lesson-{index // 4:02d}",
                        "route_variant": index % 4 + 1,
                        "target_slot": target_slot,
                        "target_action": target_action,
                        "adapter_state": state,
                        "prompt_kind": kind,
                        "predicted_candidate": predicted,
                        "ordered_candidates": ordered_candidates,
                        "candidates": candidates,
                        "tie": tie,
                        "non_finite": False,
                        "error_status": "tie" if tie else "ok",
                    }
                )
    return rows


def _primary_replay_rows() -> list[dict[str, object]]:
    cells = (
        "slot_default_natural_external",
        "slot_default_natural_opaque",
        "slot_default_snake_external",
        "slot_default_snake_opaque",
        "slot_map_natural_external",
        "slot_map_natural_opaque",
        "slot_map_snake_external",
        "slot_map_snake_opaque",
    )
    rows: list[dict[str, object]] = []
    for cell in cells:
        for index in range(96):
            global_index = len(rows)
            base_tie = global_index < 18
            adapter_tie = 18 <= global_index < 26
            rows.append(
                {
                    "probe_id": f"{cell}-{index:03d}",
                    "cell": cell,
                    "base_correct": False,
                    "adapter_correct": True,
                    "base_error_status": "tie" if base_tie else "ok",
                    "adapter_error_status": "tie" if adapter_tie else "ok",
                    "base_tie": base_tie,
                    "adapter_tie": adapter_tie,
                    "base_tie_expected_compatible": base_tie,
                    "adapter_tie_expected_compatible": adapter_tie,
                    "base_signed_expected_margin": 0.0 if base_tie else -1.0,
                    "adapter_signed_expected_margin": 0.0 if adapter_tie else 1.0,
                }
            )
    return rows


def _external_replay_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    external: list[dict[str, object]] = []
    q2: list[dict[str, object]] = []
    for index in range(96):
        source_id = f"source-{index:03d}"
        common = {
            "prompt_sha256": f"prompt-{index:03d}",
            "expected_candidate": "act_target",
            "ordered_candidates": ["act_target", "act_wrong"],
        }
        q2.append(
            {
                **common,
                "matrix": "forced_choice",
                "category": "conditional_route",
                "probe_id": source_id,
                "base_correct": True,
                "adapter_correct": True,
                "base_signed_expected_margin": 0.10,
                "adapter_signed_expected_margin": 0.20,
            }
        )
        external.append(
            {
                **common,
                "probe_id": f"external-{index:03d}",
                "source_probe_id": source_id,
                "cell": "external_action",
                "base_correct": index >= 3,
                "adapter_correct": index != 0,
                "base_signed_expected_margin": 0.11,
                "adapter_signed_expected_margin": 0.18,
            }
        )
    return external, q2


def _candidates(
    values: tuple[str, str], *, predicted: str, tie: bool
) -> list[dict[str, object]]:
    return [
        {
            "candidate": value,
            "sum_rank": 1 if tie or value == predicted else 2,
        }
        for value in values
    ]


def _complete_result_fixture(tmp_path: Path) -> Path:
    output = tmp_path / "complete"
    (output / "results").mkdir(parents=True)
    run_id = "p0d2hrsh-fixture"
    preregistration_sha256 = "b" * 64
    (output / "results" / "adapter_off.jsonl").write_text("{}\n" * 384)
    (output / "results" / "adapter_on.jsonl").write_text("{}\n" * 384)
    (output / "paired_records.jsonl").write_text("{}\n" * 96)
    (output / "report.md").write_text("# report\n")
    summary = {
        "run_id": run_id,
        "analysis": {"decision": {"status": "handoff_not_supported"}},
        "historical_rtb_decision_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    (output / "summary.json").write_text(json.dumps(summary))
    raw_artifacts = {
        "adapter_off": file_hash(output / "results" / "adapter_off.jsonl"),
        "adapter_on": file_hash(output / "results" / "adapter_on.jsonl"),
    }
    checkpoint = {
        "schema_version": "p0d2hrsh-raw-score-checkpoint-v1",
        "run_id": run_id,
        "adapter_off_count": 384,
        "adapter_on_count": 384,
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
        "schema_version": "p0d2hrsh-audit-manifest-v1",
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
            "decision": "handoff_not_supported",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    return output
