from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from plasticity_placement.p0d2hcbr.config import AUTHORIZATION_SCOPE
from plasticity_placement.p0d2hrr.io import immutable_json_write, read_json_object

SCHEMA_VERSION = "p0d2hcbr-authorization-v1"


def template(preregistration_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "pending",
        "scope": AUTHORIZATION_SCOPE,
        "preregistration_sha256": preregistration_sha256,
        "approved_by": "TBD",
        "approved_at": "TBD",
        "allowed_training_runs": 12,
        "allowed_locked_evaluations": 12,
        "allows_hyperparameter_search": False,
        "allows_checkpoint_selection": False,
        "allows_mappings_per_adapter_scan": False,
        "allows_full_parameter_training": False,
        "notes": "This template is not an approval and must be approved outside the run output.",
    }


def adopt(
    output_dir: Path, source: Path, preregistration_sha256: str
) -> tuple[dict[str, Any], str]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    if source == output_dir or source.is_relative_to(output_dir):
        raise ValueError("authorization must be authored outside the CBR output")
    payload = read_json_object(source, "CBR authorization")
    expected = template(preregistration_sha256)
    required = {
        **expected,
        "decision": "approved",
        "approved_by": payload.get("approved_by"),
        "approved_at": payload.get("approved_at"),
        "notes": payload.get("notes", expected["notes"]),
    }
    differences = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in required.items()
        if payload.get(key) != value
    }
    if differences:
        raise PermissionError(f"CBR authorization differs: {differences}")
    if payload.get("approved_by") in {None, "", "TBD"} or payload.get("approved_at") == "TBD":
        raise ValueError("CBR authorization requires approver and timestamp")
    approved_at = datetime.fromisoformat(str(payload["approved_at"]).replace("Z", "+00:00"))
    if approved_at.tzinfo is None:
        raise ValueError("CBR approval timestamp must be timezone-aware")
    digest = immutable_json_write(output_dir / "authorization.json", payload, "CBR authorization")
    return payload, digest
