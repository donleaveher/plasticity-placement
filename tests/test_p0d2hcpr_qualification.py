from __future__ import annotations

import json
from hashlib import sha256

import pytest

from plasticity_placement.p0d2hcpr.config import GateSpec
from plasticity_placement.p0d2hcpr.preflight import _validate_failed_same_runtime
from plasticity_placement.p0d2hcpr.qualification import (
    RECOVERABLE_FAILURE_SIGNATURE,
    RECOVERY_AUTHORIZATION_SCHEMA_VERSION,
    RECOVERY_AUTHORIZATION_SCOPE,
    _claim_qualification,
    _source_snapshot,
    _tie_accuracy_bounds,
    classify_result,
)
from plasticity_placement.p0d2hrr.io import file_hash
from plasticity_placement.p0d2hrr.paired_audit import _tie_bounds


def _metric(
    base: float,
    adapter: float,
    ci: tuple[float, float],
    *,
    base_interval: tuple[float, float] | None = None,
    adapter_interval: tuple[float, float] | None = None,
) -> dict[str, object]:
    return {
        "base_accuracy": base,
        "adapter_accuracy": adapter,
        "paired_accuracy_difference": {"ci95": list(ci)},
        "tie_sensitivity": {
            "base": {
                "bounds_valid": True,
                "all_ties_incorrect_accuracy": (base_interval or (base, base))[0],
                "all_ties_compatible_accuracy": (base_interval or (base, base))[1],
            },
            "adapter": {
                "bounds_valid": True,
                "all_ties_incorrect_accuracy": (adapter_interval or (adapter, adapter))[0],
                "all_ties_compatible_accuracy": (adapter_interval or (adapter, adapter))[1],
            },
        },
    }


def _analysis(*, conditional: float = 0.78, combined_ci: tuple[float, float] = (-0.01, 0.03)):
    return {
        "forced_choice": {"external_conditional_route": _metric(0.64, conditional, (-0.01, 0.04))},
        "crd": {
            "by_endpoint": {
                "route_only": _metric(
                    0.50,
                    0.67,
                    (0.12, 0.20),
                    adapter_interval=(0.65, 0.67),
                ),
                "retrieval_only": _metric(1.0, 1.0, (0.0, 0.0)),
                "combined": _metric(
                    0.79,
                    0.79,
                    combined_ci,
                    base_interval=(0.79, 0.792),
                    adapter_interval=(0.789, 0.79),
                ),
            }
        },
    }


def _pairs(*, compatible: bool = True):
    return [
        {
            "base_tie": True,
            "adapter_tie": True,
            "base_tie_expected_compatible": compatible,
            "adapter_tie_expected_compatible": compatible,
            "base_error_status": "tie",
            "adapter_error_status": "tie",
        }
    ]


def test_candidate_accepts_only_expected_compatible_ties_and_never_authorizes_148() -> None:
    decision = classify_result(_analysis(), _pairs(), GateSpec())
    assert decision["status"] == "reading_qualification_candidate_review_required"
    assert decision["mappings_per_adapter_authorized"] is False


def test_incompatible_tie_fails_integrity() -> None:
    decision = classify_result(_analysis(), _pairs(compatible=False), GateSpec())
    assert decision["status"] == "scoring_integrity_failed"


def test_route_gain_with_combined_regression_confirms_interference() -> None:
    decision = classify_result(_analysis(combined_ci=(-0.08, -0.04)), _pairs(), GateSpec())
    assert decision["status"] == "composition_interference_confirmed"
    assert decision["mappings_per_adapter_authorized"] is False


def test_combined_interval_crossing_zero_is_not_confirmed_interference() -> None:
    decision = classify_result(_analysis(combined_ci=(-0.08, 0.04)), _pairs(), GateSpec())
    assert decision["status"] == "qualification_failed"
    assert decision["diagnostics"]["combined_material_regression_supported"] is False


def test_classifier_consumes_real_paired_audit_tie_schema() -> None:
    produced = _tie_bounds(
        [
            {
                "adapter_correct": True,
                "adapter_tie": False,
                "adapter_tie_expected_compatible": False,
                "adapter_error_status": "ok",
            },
            {
                "adapter_correct": False,
                "adapter_tie": True,
                "adapter_tie_expected_compatible": True,
                "adapter_error_status": "tie",
            },
        ],
        "adapter",
    )

    assert _tie_accuracy_bounds({"tie_sensitivity": {"adapter": produced}}, "adapter") == (
        0.5,
        1.0,
    )


def test_classifier_rejects_obsolete_accuracy_interval_schema() -> None:
    metric = {"tie_sensitivity": {"adapter": {"accuracy_interval": [0.5, 1.0]}}}
    with pytest.raises(ValueError, match="missing or invalid"):
        _tie_accuracy_bounds(metric, "adapter")


def test_locked_qualification_can_only_be_claimed_once(tmp_path) -> None:
    _claim_qualification(tmp_path, tmp_path.parent / "analysis", "run", "a" * 64)
    with pytest.raises(PermissionError, match="already been claimed"):
        _claim_qualification(tmp_path, tmp_path.parent / "analysis-2", "run-2", "b" * 64)


def test_one_recovery_can_be_claimed_with_external_code_bound_authorization(tmp_path) -> None:
    output = tmp_path / "cpr"
    original_output = tmp_path / "qualification-q1"
    recovery_output = tmp_path / "qualification-q2"
    adapter_sha256 = "a" * 64
    analysis_code_sha256 = "b" * 64
    _claim_qualification(output, original_output, "run-q1", adapter_sha256)
    original_claim = output / "qualification_claim.json"
    authorization_path = tmp_path / "approvals" / "recovery.json"
    authorization_path.parent.mkdir()
    authorization_path.write_text(
        json.dumps(
            {
                "schema_version": RECOVERY_AUTHORIZATION_SCHEMA_VERSION,
                "decision": "approved",
                "scope": RECOVERY_AUTHORIZATION_SCOPE,
                "failure_signature": RECOVERABLE_FAILURE_SIGNATURE,
                "original_claim_sha256": file_hash(original_claim),
                "original_analysis_output": str(original_output.resolve()),
                "recovery_analysis_output": str(recovery_output.resolve()),
                "analysis_code_sha256": analysis_code_sha256,
                "allowed_recovery_runs": 1,
                "allows_training": False,
                "allows_mappings_per_adapter_scan": False,
                "approved_by": "test-approver",
                "approved_at": "2026-08-03T00:00:00+00:00",
            }
        )
    )

    context = _claim_qualification(
        output,
        recovery_output.resolve(),
        "run-q2",
        adapter_sha256,
        analysis_code_sha256=analysis_code_sha256,
        recovery_authorization=authorization_path,
    )

    assert context["mode"] == "approved_post_inference_classification_failure_recovery"
    assert context["analysis_code_sha256"] == analysis_code_sha256
    assert (output / "qualification_recovery_authorization.json").is_file()
    assert (output / "qualification_recovery_claim.json").is_file()
    with pytest.raises(PermissionError, match="already been claimed"):
        _claim_qualification(
            output,
            tmp_path / "qualification-q3",
            "run-q3",
            adapter_sha256,
            analysis_code_sha256=analysis_code_sha256,
            recovery_authorization=authorization_path,
        )


def test_recovery_completes_claim_after_interruption_following_authorization_adoption(
    tmp_path,
) -> None:
    output = tmp_path / "cpr"
    original_output = tmp_path / "qualification-q1"
    recovery_output = (tmp_path / "qualification-q2").resolve()
    adapter_sha256 = "a" * 64
    analysis_code_sha256 = "b" * 64
    _claim_qualification(output, original_output, "run-q1", adapter_sha256)
    original_claim = output / "qualification_claim.json"
    authorization = {
        "schema_version": RECOVERY_AUTHORIZATION_SCHEMA_VERSION,
        "decision": "approved",
        "scope": RECOVERY_AUTHORIZATION_SCOPE,
        "failure_signature": RECOVERABLE_FAILURE_SIGNATURE,
        "original_claim_sha256": file_hash(original_claim),
        "original_analysis_output": str(original_output.resolve()),
        "recovery_analysis_output": str(recovery_output),
        "analysis_code_sha256": analysis_code_sha256,
        "allowed_recovery_runs": 1,
        "allows_training": False,
        "allows_mappings_per_adapter_scan": False,
        "approved_by": "test-approver",
        "approved_at": "2026-08-03T00:00:00+00:00",
    }
    authorization_path = tmp_path / "external-recovery.json"
    authorization_path.write_text(json.dumps(authorization))
    adopted_path = output / "qualification_recovery_authorization.json"
    adopted_path.write_text(json.dumps(authorization, indent=2, sort_keys=True) + "\n")

    context = _claim_qualification(
        output,
        recovery_output,
        "run-q2",
        adapter_sha256,
        analysis_code_sha256=analysis_code_sha256,
        recovery_authorization=authorization_path,
    )

    claim = json.loads((output / "qualification_recovery_claim.json").read_text())
    assert context["recovery_authorization_sha256"] == file_hash(adopted_path)
    assert claim["recovery_authorization_sha256"] == file_hash(adopted_path)


def test_prior_same_runtime_manifest_must_bind_failed_summary(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    summary = {
        "schema_version": "p0d2hrr-same-runtime-audit-v1",
        "run_id": "same-runtime-run",
        "source": {
            "route_remediation": {
                "run_id": "rr-run",
                "adapter_sha256": "a" * 64,
                "current_artifact_checks_passed": True,
            }
        },
        "runtime_identity": {"sentinel": {"passed": True}},
        "source_artifacts_modified": False,
        "source_gate_status_changed": False,
        "decision": {
            "status": "scoring_integrity_failed",
            "mappings_per_adapter_authorized": False,
        },
    }
    summary_path.write_text(json.dumps(summary))
    summary_hash = sha256(summary_path.read_bytes()).hexdigest()
    audit_manifest = {
        "schema_version": "p0d2hrr-same-runtime-audit-manifest-v1",
        "run_id": "same-runtime-run",
        "artifacts": {"summary": {"sha256": summary_hash}},
        "source_artifacts_modified": False,
        "mappings_per_adapter_authorized": False,
    }
    manifest_path = tmp_path / "audit_manifest.json"
    manifest_path.write_text(json.dumps(audit_manifest))

    observed = _validate_failed_same_runtime(
        summary_path,
        {"run_id": "rr-run", "adapter_sha256": "a" * 64},
    )
    assert observed["run_id"] == "same-runtime-run"

    audit_manifest["artifacts"]["summary"]["sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(audit_manifest))
    with pytest.raises(ValueError, match="does not bind"):
        _validate_failed_same_runtime(
            summary_path,
            {"run_id": "rr-run", "adapter_sha256": "a" * 64},
        )


def test_source_snapshot_detects_adapter_weight_mutation(tmp_path, monkeypatch) -> None:
    output = tmp_path / "cpr"
    adapter = output / "adapter"
    adapter.mkdir(parents=True)
    weight = adapter / "adapter_model.safetensors"
    weight.write_bytes(b"before")
    source = tmp_path / "source"
    source.mkdir()
    frozen_source = tmp_path / "frozen-source"
    frozen_source.mkdir()
    same_runtime = tmp_path / "same-runtime"
    same_runtime.mkdir()
    paths = (
        output / "manifest.json",
        output / "preregistration.json",
        output / "authorization.json",
        adapter / "training_metadata.json",
        output / "qualification_claim.json",
        source / "manifest.json",
        source / "code_revision.txt",
        frozen_source / "manifest.json",
        same_runtime / "summary.json",
        same_runtime / "audit_manifest.json",
    )
    for path in paths:
        path.write_text("source")
    identity = {
        "source_output": str(source),
        "source_code_revision_lock": str(source / "code_revision.txt"),
        "same_runtime_summary": str(same_runtime / "summary.json"),
        "same_runtime_manifest": str(same_runtime / "audit_manifest.json"),
        "source_context": {
            "source_manifest_path": str(frozen_source / "manifest.json"),
        },
    }
    monkeypatch.setattr(
        "plasticity_placement.p0d2hcpr.qualification._adapter_hash",
        lambda adapter_dir: sha256((adapter_dir / weight.name).read_bytes()).hexdigest(),
    )

    before = _source_snapshot(output, identity)
    weight.write_bytes(b"after")
    after = _source_snapshot(output, identity)

    assert before["adapter_bundle"]["sha256"] != after["adapter_bundle"]["sha256"]

    source_audit = source / "preflight" / "audit.json"
    source_audit.parent.mkdir()
    source_audit.write_text("before")
    before_source_change = _source_snapshot(output, identity)
    source_audit.write_text("after")
    after_source_change = _source_snapshot(output, identity)
    assert (
        before_source_change["rr_output_tree"]["sha256"]
        != after_source_change["rr_output_tree"]["sha256"]
    )
