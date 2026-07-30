from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from plasticity_placement.p0d2hrr.config import AUTHORIZATION_SCOPE
from plasticity_placement.p0d2hrr.io import (
    immutable_json_write,
    read_json_object,
)

AUTHORIZATION_SCHEMA_VERSION = "p0d2hrr-authorization-v1"


def authorization_template(preregistration_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "decision": "pending",
        "scope": AUTHORIZATION_SCOPE,
        "preregistration_sha256": preregistration_sha256,
        "approved_by": "TBD",
        "approved_at": "TBD",
        "allowed_training_runs": 1,
        "allowed_locked_external_evaluations": 1,
        "allows_hyperparameter_search": False,
        "allows_mappings_per_adapter_scan": False,
        "allows_narrow_scan": False,
        "allows_rlvr": False,
        "notes": (
            "This template is not an approval. A separately authorized reviewer "
            "must create an approved copy outside the experiment output."
        ),
    }


def validate_authorization(
    payload: dict[str, Any],
    *,
    preregistration_sha256: str,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "decision",
        "scope",
        "preregistration_sha256",
        "approved_by",
        "approved_at",
        "allowed_training_runs",
        "allowed_locked_external_evaluations",
        "allows_hyperparameter_search",
        "allows_mappings_per_adapter_scan",
        "allows_narrow_scan",
        "allows_rlvr",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"authorization is missing {sorted(missing)}")
    if payload["schema_version"] != AUTHORIZATION_SCHEMA_VERSION:
        raise ValueError("authorization schema changed")
    if payload["decision"] != "approved":
        raise PermissionError("training is not approved")
    if payload["scope"] != AUTHORIZATION_SCOPE:
        raise PermissionError("authorization scope does not match the pilot")
    if payload["preregistration_sha256"] != preregistration_sha256:
        raise PermissionError("authorization does not bind this preregistration")
    approver = payload["approved_by"]
    if not isinstance(approver, str) or not approver.strip() or approver == "TBD":
        raise ValueError("authorization requires a named approver")
    approved_at = payload["approved_at"]
    if not isinstance(approved_at, str) or approved_at == "TBD":
        raise ValueError("authorization requires an approval timestamp")
    try:
        datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("approved_at must be an ISO-8601 timestamp") from error
    if payload["allowed_training_runs"] != 1:
        raise PermissionError("pilot authorization must permit exactly one training run")
    if payload["allowed_locked_external_evaluations"] != 1:
        raise PermissionError(
            "pilot authorization must permit exactly one locked external evaluation"
        )
    forbidden = (
        "allows_hyperparameter_search",
        "allows_mappings_per_adapter_scan",
        "allows_narrow_scan",
        "allows_rlvr",
    )
    if any(payload[field] is not False for field in forbidden):
        raise PermissionError("authorization attempts to expand the pilot scope")
    return payload


def adopt_authorization(
    output_dir: Path,
    authorization_path: Path,
    *,
    preregistration_sha256: str,
) -> tuple[dict[str, Any], str]:
    source = authorization_path.resolve()
    output = output_dir.resolve()
    if source == output or source.is_relative_to(output):
        raise ValueError("authorization must be authored outside the experiment output directory")
    payload = validate_authorization(
        read_json_object(authorization_path, "authorization"),
        preregistration_sha256=preregistration_sha256,
    )
    digest = immutable_json_write(
        output_dir / "authorization.json",
        payload,
        "authorization",
    )
    return payload, digest
