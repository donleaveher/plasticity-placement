from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.runtime import current_code_hash
from plasticity_placement.p0d2hrabx.config import AuditSpec
from plasticity_placement.p0d2hrabx.runtime import (
    _load_manifest,
    _save_manifest,
    _validate_rabc_source,
    _verify_preflight,
)
from plasticity_placement.p0d2hrr.io import (
    file_hash,
    immutable_json_write,
    json_hash,
    read_json_object,
)

RECOVERY_AUTHORIZATION_SCHEMA_VERSION = "p0d2hrabx-zero-artifact-recovery-authorization-v1"
RECOVERY_CLAIM_SCHEMA_VERSION = "p0d2hrabx-zero-artifact-recovery-claim-v1"
RECOVERY_SCOPE = "single_stale_running_zero_artifact_identical_retry"
FAILURE_SIGNATURE = "stale_running_without_persisted_score_or_analysis_artifact"
MIN_STALE_SECONDS = 6 * 60 * 60
ADOPTED_AUTHORIZATION_NAME = "zero_artifact_recovery_authorization.json"
RECOVERY_CLAIM_NAME = "zero_artifact_recovery_claim.json"
TOP_LEVEL_RESULT_NAMES = (
    "raw_score_checkpoint.json",
    "paired_records.jsonl",
    "summary.json",
    "report.md",
    "audit_manifest.json",
)


def inspect_zero_artifact_recovery(
    output_dir: Path,
    experiment_code_revision_lock: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    experiment_code_revision_lock = experiment_code_revision_lock.resolve()
    manifest = _load_manifest(output_dir)
    if manifest.get("state") != "running":
        raise PermissionError("zero-artifact recovery requires stale running state")
    if isinstance(manifest.get("recovery"), dict):
        raise PermissionError("the single zero-artifact recovery was already consumed")
    if manifest.get("errors") != []:
        raise PermissionError("zero-artifact recovery requires an empty manifest error list")
    updated_at = _parse_timestamp(manifest.get("updated_at"), "manifest updated_at")
    observed_at = now or datetime.now(UTC)
    if observed_at.tzinfo is None:
        raise ValueError("recovery inspection time must be timezone-aware")
    age_seconds = (observed_at.astimezone(UTC) - updated_at.astimezone(UTC)).total_seconds()
    if age_seconds < MIN_STALE_SECONDS:
        remaining = int(MIN_STALE_SECONDS - age_seconds)
        raise PermissionError(
            "running RABX attempt is not stale enough for recovery; "
            f"retry after {remaining} seconds"
        )

    present = _present_result_artifacts(output_dir)
    if present:
        raise PermissionError(f"zero-artifact recovery found result artifacts: {present}")
    identity = manifest.get("config")
    if not isinstance(identity, dict):
        raise ValueError("RABX manifest has no preregistered identity")
    _verify_preflight(output_dir, identity)
    source = _validate_rabc_source(Path(str(identity["rabc_output"])), AuditSpec())
    if source["snapshot"] != identity.get("source_snapshot"):
        raise ValueError("RABC/RAB source artifacts changed before zero-artifact recovery")

    original_authorization_path = output_dir / "authorization.json"
    authorization_identity = manifest.get("authorization")
    if (
        not isinstance(authorization_identity, dict)
        or not original_authorization_path.is_file()
        or authorization_identity.get("sha256") != file_hash(original_authorization_path)
    ):
        raise ValueError("original RABX authorization is missing or changed")
    original_authorization = read_json_object(
        original_authorization_path, "original RABX authorization"
    )
    if original_authorization.get("decision") != "approved":
        raise ValueError("original RABX authorization is not approved")

    experiment_revision = _read_locked_revision(experiment_code_revision_lock)
    if experiment_revision != identity.get("code_sha256"):
        raise ValueError("experiment revision lock differs from preregistered RABX code")
    evidence = {
        "failure_signature": FAILURE_SIGNATURE,
        "manifest_state": "running",
        "manifest_errors": [],
        "running_updated_at": manifest["updated_at"],
        "result_artifacts_present": [],
        "result_paths_checked": ["results/**", *TOP_LEVEL_RESULT_NAMES],
        "source_snapshot_sha256": json_hash(source["snapshot"]),
        "original_authorization_sha256": file_hash(original_authorization_path),
    }
    return {
        "eligible": True,
        "output": str(output_dir),
        "run_id": manifest["run_id"],
        "preregistration_sha256": identity["preregistration_sha256"],
        "running_manifest_sha256": file_hash(output_dir / "manifest.json"),
        "experiment_code_revision": experiment_revision,
        "experiment_code_revision_lock_sha256": file_hash(experiment_code_revision_lock),
        "recovery_code_sha256": current_code_hash(),
        "stale_age_seconds": age_seconds,
        "minimum_stale_seconds": MIN_STALE_SECONDS,
        "evidence": evidence,
        "evidence_sha256": json_hash(evidence),
        "historical_rab_decision_changed": False,
        "historical_rabc_attribution_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }


def recovery_authorization_template(
    output_dir: Path,
    experiment_code_revision_lock: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    inspection = inspect_zero_artifact_recovery(output_dir, experiment_code_revision_lock, now=now)
    return {
        "schema_version": RECOVERY_AUTHORIZATION_SCHEMA_VERSION,
        "decision": "pending",
        "scope": RECOVERY_SCOPE,
        "failure_signature": FAILURE_SIGNATURE,
        "output": inspection["output"],
        "run_id": inspection["run_id"],
        "preregistration_sha256": inspection["preregistration_sha256"],
        "running_manifest_sha256": inspection["running_manifest_sha256"],
        "running_updated_at": inspection["evidence"]["running_updated_at"],
        "zero_artifact_evidence_sha256": inspection["evidence_sha256"],
        "experiment_code_revision": inspection["experiment_code_revision"],
        "experiment_code_revision_lock_sha256": inspection[
            "experiment_code_revision_lock_sha256"
        ],
        "recovery_code_sha256": inspection["recovery_code_sha256"],
        "allowed_identical_retries": 1,
        "original_runtime_confirmed_terminated": False,
        "allows_training": False,
        "allows_prompt_or_threshold_changes": False,
        "allows_new_result_cells": False,
        "allows_historical_reclassification": False,
        "allows_mappings_per_adapter_scan": False,
        "approved_by": "TBD",
        "approved_at": "TBD",
    }


def recover_zero_artifact_attempt(
    output_dir: Path,
    authorization_path: Path,
    experiment_code_revision_lock: Path,
    *,
    now: datetime | None = None,
) -> Path:
    output_dir = output_dir.resolve()
    authorization_path = authorization_path.resolve()
    experiment_code_revision_lock = experiment_code_revision_lock.resolve()
    if authorization_path == output_dir or authorization_path.is_relative_to(output_dir):
        raise ValueError("zero-artifact recovery authorization must be external")

    manifest = _load_manifest(output_dir)
    if manifest.get("state") == "authorized" and isinstance(manifest.get("recovery"), dict):
        _validate_completed_recovery(output_dir, authorization_path, manifest)
        return output_dir / "manifest.json"

    template = recovery_authorization_template(output_dir, experiment_code_revision_lock, now=now)
    authorization = read_json_object(authorization_path, "zero-artifact recovery authorization")
    required = {
        **template,
        "decision": "approved",
        "approved_by": authorization.get("approved_by"),
        "approved_at": authorization.get("approved_at"),
        "original_runtime_confirmed_terminated": True,
    }
    differences = {
        key: {"expected": value, "observed": authorization.get(key)}
        for key, value in required.items()
        if authorization.get(key) != value
    }
    if differences:
        raise PermissionError(f"zero-artifact recovery authorization differs: {differences}")
    approver = authorization.get("approved_by")
    approved_at = authorization.get("approved_at")
    if not isinstance(approver, str) or not approver.strip() or approver == "TBD":
        raise ValueError("zero-artifact recovery requires a named approver")
    _parse_timestamp(approved_at, "zero-artifact recovery approved_at")

    adopted_path = output_dir / ADOPTED_AUTHORIZATION_NAME
    if adopted_path.exists():
        adopted = read_json_object(adopted_path, "adopted zero-artifact authorization")
        if adopted != authorization:
            raise PermissionError("adopted zero-artifact authorization differs")
        authorization_sha256 = file_hash(adopted_path)
    else:
        authorization_sha256 = immutable_json_write(
            adopted_path,
            authorization,
            "zero-artifact recovery authorization",
        )

    claim_path = output_dir / RECOVERY_CLAIM_NAME
    static_claim = {
        "schema_version": RECOVERY_CLAIM_SCHEMA_VERSION,
        "mode": "approved_stale_running_zero_artifact_identical_retry",
        "failure_signature": FAILURE_SIGNATURE,
        "output": template["output"],
        "run_id": template["run_id"],
        "preregistration_sha256": template["preregistration_sha256"],
        "running_manifest_sha256": template["running_manifest_sha256"],
        "running_updated_at": template["running_updated_at"],
        "zero_artifact_evidence_sha256": template["zero_artifact_evidence_sha256"],
        "experiment_code_revision": template["experiment_code_revision"],
        "experiment_code_revision_lock_sha256": template[
            "experiment_code_revision_lock_sha256"
        ],
        "recovery_code_sha256": template["recovery_code_sha256"],
        "recovery_authorization_sha256": authorization_sha256,
        "allowed_identical_retries": 1,
        "original_runtime_confirmed_terminated": True,
        "historical_rab_decision_changed": False,
        "historical_rabc_attribution_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    if claim_path.exists():
        claim = read_json_object(claim_path, "zero-artifact recovery claim")
        differences = {
            key: {"expected": value, "observed": claim.get(key)}
            for key, value in static_claim.items()
            if claim.get(key) != value
        }
        if differences:
            raise PermissionError(f"existing zero-artifact claim differs: {differences}")
    else:
        claim = {
            **static_claim,
            "recovered_at": (now or datetime.now(UTC)).astimezone(UTC).isoformat(),
        }
        immutable_json_write(claim_path, claim, "zero-artifact recovery claim")
    claim_sha256 = file_hash(claim_path)

    manifest = _load_manifest(output_dir)
    if manifest.get("state") != "running":
        raise PermissionError("RABX state changed during zero-artifact recovery")
    manifest["state"] = "authorized"
    manifest["recovery"] = {
        "mode": claim["mode"],
        "claim_sha256": claim_sha256,
        "authorization_sha256": authorization_sha256,
        "recovered_at": claim["recovered_at"],
        "allowed_identical_retries": 1,
        "original_runtime_confirmed_terminated": True,
        "experiment_code_revision": claim["experiment_code_revision"],
        "recovery_code_sha256": claim["recovery_code_sha256"],
        "historical_rab_decision_changed": False,
        "historical_rabc_attribution_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    _save_manifest(output_dir, manifest)
    return output_dir / "manifest.json"


def _validate_completed_recovery(
    output_dir: Path, authorization_path: Path, manifest: dict[str, Any]
) -> None:
    adopted_path = output_dir / ADOPTED_AUTHORIZATION_NAME
    claim_path = output_dir / RECOVERY_CLAIM_NAME
    if not adopted_path.is_file() or not claim_path.is_file():
        raise ValueError("authorized recovery is missing immutable governance artifacts")
    external = read_json_object(authorization_path, "external zero-artifact authorization")
    adopted = read_json_object(adopted_path, "adopted zero-artifact authorization")
    if external != adopted:
        raise PermissionError("external and adopted zero-artifact authorizations differ")
    recovery = manifest["recovery"]
    if (
        recovery.get("authorization_sha256") != file_hash(adopted_path)
        or recovery.get("claim_sha256") != file_hash(claim_path)
        or recovery.get("allowed_identical_retries") != 1
        or recovery.get("original_runtime_confirmed_terminated") is not True
        or recovery.get("training_authorized") is not False
        or recovery.get("mappings_per_adapter_authorized") is not False
    ):
        raise ValueError("authorized zero-artifact recovery provenance is invalid")


def _present_result_artifacts(output_dir: Path) -> list[str]:
    present: set[str] = set()
    results_dir = output_dir / "results"
    if results_dir.exists():
        present.update(
            str(path.relative_to(output_dir)) for path in results_dir.rglob("*") if path.is_file()
        )
    for name in TOP_LEVEL_RESULT_NAMES:
        present.update(
            str(path.relative_to(output_dir))
            for path in output_dir.glob(name + "*")
            if path.is_file()
        )
    return sorted(present)


def _read_locked_revision(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"experiment code-revision lock is missing: {path}")
    revision = path.read_text().strip()
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("experiment code-revision lock is not a full Git commit")
    return revision


def _parse_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return parsed
