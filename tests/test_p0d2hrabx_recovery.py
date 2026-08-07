from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import plasticity_placement.p0d2hrabx.recovery as recovery
from plasticity_placement.p0d2hrabx.recovery import (
    ADOPTED_AUTHORIZATION_NAME,
    RECOVERY_CLAIM_NAME,
    recover_zero_artifact_attempt,
    recovery_authorization_template,
)
from plasticity_placement.p0d2hrabx.runtime import MANIFEST_SCHEMA_VERSION
from plasticity_placement.p0d2hrr.io import file_hash, json_hash

NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
EXPERIMENT_REVISION = "5" * 40


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, object]]:
    output = tmp_path / "pipeline" / "runs" / "rabx"
    output.mkdir(parents=True)
    source_snapshot = {"rabc_output_tree": "a" * 64, "rab_source_snapshot": {}}
    identity = {
        "schema_version": "p0d2hrabx-preregistration-v1",
        "code_sha256": EXPERIMENT_REVISION,
        "rabc_output": str((tmp_path / "rabc").resolve()),
        "source_snapshot": source_snapshot,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    identity["preregistration_sha256"] = json_hash(identity)
    _json_write(output / "preregistration.json", identity)
    authorization = {
        "decision": "approved",
        "approved_by": "original-approver",
        "approved_at": "2026-08-07T00:00:00+00:00",
        "allows_training": False,
    }
    _json_write(output / "authorization.json", authorization)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": "p0d2hrabx-test",
        "state": "running",
        "created_at": "2026-08-08T04:00:00+00:00",
        "updated_at": "2026-08-08T05:00:00+00:00",
        "config": identity,
        "authorization": {
            "sha256": file_hash(output / "authorization.json"),
            "approved_by": "original-approver",
        },
        "errors": [],
        "historical_rab_decision_changed": False,
        "historical_rabc_attribution_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    _json_write(output / "manifest.json", manifest)
    lock = tmp_path / "pipeline" / "code_revision.txt"
    lock.write_text(EXPERIMENT_REVISION + "\n")
    monkeypatch.setattr(recovery, "_verify_preflight", lambda *_: None)
    monkeypatch.setattr(
        recovery,
        "_validate_rabc_source",
        lambda *_: {"snapshot": source_snapshot},
    )
    return output, lock, manifest


def _approval(tmp_path: Path, template: dict[str, object]) -> Path:
    path = tmp_path / "external" / "zero-artifact.json"
    approved = {
        **template,
        "decision": "approved",
        "approved_by": "recovery-approver",
        "approved_at": "2026-08-08T12:00:00+00:00",
        "original_runtime_confirmed_terminated": True,
    }
    _json_write(path, approved)
    return path


def test_zero_artifact_recovery_restores_same_attempt_to_authorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, lock, _ = _fixture(tmp_path, monkeypatch)
    template = recovery_authorization_template(output, lock, now=NOW)
    approval = _approval(tmp_path, template)

    path = recover_zero_artifact_attempt(output, approval, lock, now=NOW)

    manifest = json.loads(path.read_text())
    claim = json.loads((output / RECOVERY_CLAIM_NAME).read_text())
    assert manifest["state"] == "authorized"
    assert manifest["run_id"] == "p0d2hrabx-test"
    assert manifest["recovery"]["claim_sha256"] == file_hash(
        output / RECOVERY_CLAIM_NAME
    )
    assert claim["experiment_code_revision"] == EXPERIMENT_REVISION
    assert claim["allowed_identical_retries"] == 1
    assert claim["training_authorized"] is False
    assert (output / ADOPTED_AUTHORIZATION_NAME).is_file()
    assert recover_zero_artifact_attempt(output, approval, lock, now=NOW) == path


def test_zero_artifact_recovery_rejects_persisted_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, lock, _ = _fixture(tmp_path, monkeypatch)
    result = output / "results" / "partial.jsonl.tmp"
    result.parent.mkdir()
    result.write_text("partial")

    with pytest.raises(PermissionError, match="found result artifacts"):
        recovery_authorization_template(output, lock, now=NOW)


def test_zero_artifact_recovery_rejects_nonstale_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, lock, manifest = _fixture(tmp_path, monkeypatch)
    manifest["updated_at"] = "2026-08-08T11:55:00+00:00"
    _json_write(output / "manifest.json", manifest)

    with pytest.raises(PermissionError, match="retry after"):
        recovery_authorization_template(output, lock, now=NOW)


def test_zero_artifact_recovery_requires_runtime_termination_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, lock, _ = _fixture(tmp_path, monkeypatch)
    template = recovery_authorization_template(output, lock, now=NOW)
    approval = _approval(tmp_path, template)
    payload = json.loads(approval.read_text())
    payload["original_runtime_confirmed_terminated"] = False
    _json_write(approval, payload)

    with pytest.raises(PermissionError, match="authorization differs"):
        recover_zero_artifact_attempt(output, approval, lock, now=NOW)


def test_zero_artifact_recovery_rejects_revision_lock_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, lock, _ = _fixture(tmp_path, monkeypatch)
    lock.write_text("6" * 40 + "\n")

    with pytest.raises(ValueError, match="revision lock differs"):
        recovery_authorization_template(output, lock, now=NOW)
