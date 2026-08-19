from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from plasticity_placement.p0d2hrr.io import file_hash, json_hash
from plasticity_placement.p0d2hrtb.recovery import (
    ADOPTED_AUTHORIZATION_NAME,
    RECOVERY_CLAIM_NAME,
    recover_zero_artifact_attempt,
    recovery_authorization_template,
)
from plasticity_placement.p0d2hrtb.runtime import _source_snapshot

NOW = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    output = tmp_path / "pipeline" / "runs" / "rtb"
    cpr = tmp_path / "cpr"
    qualification = tmp_path / "qualification" / "summary.json"
    (cpr / "adapter").mkdir(parents=True)
    (cpr / "adapter" / "weights.bin").write_bytes(b"adapter")
    _json_write(qualification, {"run_id": "q2"})

    probe_path = output / "preflight" / "bridge_probes.jsonl"
    bank_path = output / "preflight" / "bank_audit.json"
    token_path = output / "preflight" / "token_audit.json"
    probe_path.parent.mkdir(parents=True)
    probe_path.write_text('{"probe_id":"p1"}\n')
    _json_write(bank_path, {"all_checks_passed": True})
    _json_write(token_path, {"all_checks_passed": True})
    source_snapshot = _source_snapshot(cpr, qualification.parent)
    identity = {
        "schema_version": "p0d2hrtb-preregistration-v1",
        "code_sha256": "a" * 64,
        "spec": {},
        "cpr_output": str(cpr.resolve()),
        "qualification_summary": str(qualification.resolve()),
        "source_snapshot": source_snapshot,
        "probe_sha256": file_hash(probe_path),
        "bank_audit_sha256": file_hash(bank_path),
        "token_audit_sha256": file_hash(token_path),
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    identity["preregistration_sha256"] = json_hash(identity)
    _json_write(output / "preregistration.json", identity)
    original_authorization = {
        "schema_version": "p0d2hrtb-authorization-v1",
        "decision": "approved",
        "scope": "single_frozen_cpr_route_transfer_bridge_audit",
        "preregistration_sha256": identity["preregistration_sha256"],
        "approved_by": "original-approver",
        "approved_at": "2026-08-03T00:00:00+00:00",
        "allowed_audit_runs": 1,
        "allows_training": False,
        "allows_prompt_revision_after_results": False,
        "allows_mappings_per_adapter_scan": False,
    }
    original_authorization_path = output / "authorization.json"
    _json_write(original_authorization_path, original_authorization)
    manifest = {
        "schema_version": "p0d2hrtb-manifest-v1",
        "run_id": "p0d2hrtb-test",
        "state": "running",
        "created_at": "2026-08-04T09:00:00+00:00",
        "updated_at": "2026-08-04T09:34:13+00:00",
        "config": identity,
        "authorization": {
            "sha256": file_hash(original_authorization_path),
            "approved_by": "original-approver",
        },
        "errors": [],
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    _json_write(output / "manifest.json", manifest)
    lock = tmp_path / "pipeline" / "code_revision.txt"
    lock.write_text("d" * 40 + "\n")
    return output, lock, manifest


def _approval(tmp_path: Path, template: dict[str, object]) -> Path:
    path = tmp_path / "external-approvals" / "zero-artifact.json"
    approved = {
        **template,
        "decision": "approved",
        "approved_by": "recovery-approver",
        "approved_at": "2026-08-05T12:00:00+00:00",
        "original_runtime_confirmed_terminated": True,
    }
    _json_write(path, approved)
    return path


def test_zero_artifact_recovery_requires_external_bound_approval(tmp_path: Path) -> None:
    output, lock, _ = _fixture(tmp_path)
    template = recovery_authorization_template(output, lock, now=NOW)
    approval = _approval(tmp_path, template)

    path = recover_zero_artifact_attempt(output, approval, lock, now=NOW)

    manifest = json.loads(path.read_text())
    claim = json.loads((output / RECOVERY_CLAIM_NAME).read_text())
    assert manifest["state"] == "authorized"
    assert manifest["recovery"]["claim_sha256"] == file_hash(output / RECOVERY_CLAIM_NAME)
    assert claim["experiment_code_revision"] == "d" * 40
    assert claim["allowed_identical_retries"] == 1
    assert claim["original_runtime_confirmed_terminated"] is True
    assert claim["training_authorized"] is False
    assert claim["mappings_per_adapter_authorized"] is False
    assert (output / ADOPTED_AUTHORIZATION_NAME).is_file()

    repeated = recover_zero_artifact_attempt(output, approval, lock, now=NOW)
    assert repeated == path


def test_zero_artifact_recovery_completes_after_claim_write_interruption(
    tmp_path: Path,
) -> None:
    output, lock, running_manifest = _fixture(tmp_path)
    template = recovery_authorization_template(output, lock, now=NOW)
    approval = _approval(tmp_path, template)
    recover_zero_artifact_attempt(output, approval, lock, now=NOW)
    claim_hash = file_hash(output / RECOVERY_CLAIM_NAME)
    _json_write(output / "manifest.json", running_manifest)

    recover_zero_artifact_attempt(output, approval, lock, now=NOW)

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["state"] == "authorized"
    assert manifest["recovery"]["claim_sha256"] == claim_hash


def test_zero_artifact_recovery_rejects_any_persisted_result(tmp_path: Path) -> None:
    output, lock, _ = _fixture(tmp_path)
    result = output / "results" / "partial.jsonl.tmp"
    result.parent.mkdir()
    result.write_text("partial")

    with pytest.raises(PermissionError, match="found result artifacts"):
        recovery_authorization_template(output, lock, now=NOW)


def test_zero_artifact_recovery_rejects_nonstale_attempt(tmp_path: Path) -> None:
    output, lock, manifest = _fixture(tmp_path)
    manifest["updated_at"] = "2026-08-05T11:55:00+00:00"
    _json_write(output / "manifest.json", manifest)

    with pytest.raises(PermissionError, match="not stale enough"):
        recovery_authorization_template(output, lock, now=NOW)


def test_zero_artifact_recovery_rejects_changed_approval(tmp_path: Path) -> None:
    output, lock, _ = _fixture(tmp_path)
    template = recovery_authorization_template(output, lock, now=NOW)
    template["experiment_code_revision"] = "e" * 40
    approval = _approval(tmp_path, template)

    with pytest.raises(PermissionError, match="authorization differs"):
        recover_zero_artifact_attempt(output, approval, lock, now=NOW)


def test_zero_artifact_recovery_requires_runtime_termination_attestation(
    tmp_path: Path,
) -> None:
    output, lock, _ = _fixture(tmp_path)
    template = recovery_authorization_template(output, lock, now=NOW)
    approval = _approval(tmp_path, template)
    payload = json.loads(approval.read_text())
    payload["original_runtime_confirmed_terminated"] = False
    _json_write(approval, payload)

    with pytest.raises(PermissionError, match="authorization differs"):
        recover_zero_artifact_attempt(output, approval, lock, now=NOW)


def test_zero_artifact_recovery_cannot_be_consumed_twice(tmp_path: Path) -> None:
    output, lock, _ = _fixture(tmp_path)
    template = recovery_authorization_template(output, lock, now=NOW)
    approval = _approval(tmp_path, template)
    recover_zero_artifact_attempt(output, approval, lock, now=NOW)
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["state"] = "running"
    manifest["updated_at"] = "2026-08-04T09:34:13+00:00"
    _json_write(output / "manifest.json", manifest)

    with pytest.raises(PermissionError, match="already consumed"):
        recovery_authorization_template(output, lock, now=NOW)
