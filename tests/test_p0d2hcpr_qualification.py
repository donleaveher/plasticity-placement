from __future__ import annotations

import json
from hashlib import sha256

import pytest

from plasticity_placement.p0d2hcpr.config import GateSpec
from plasticity_placement.p0d2hcpr.preflight import _validate_failed_same_runtime
from plasticity_placement.p0d2hcpr.qualification import (
    _claim_qualification,
    _source_snapshot,
    classify_result,
)


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
            "base": {"accuracy_interval": list(base_interval or (base, base))},
            "adapter": {"accuracy_interval": list(adapter_interval or (adapter, adapter))},
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


def test_locked_qualification_can_only_be_claimed_once(tmp_path) -> None:
    _claim_qualification(tmp_path, tmp_path.parent / "analysis", "run", "a" * 64)
    with pytest.raises(PermissionError, match="already been claimed"):
        _claim_qualification(tmp_path, tmp_path.parent / "analysis-2", "run-2", "b" * 64)


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
