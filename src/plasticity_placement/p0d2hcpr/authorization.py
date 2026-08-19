from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from plasticity_placement.p0d2hcpr.config import AUTHORIZATION_SCOPE
from plasticity_placement.p0d2hrr.io import immutable_json_write, read_json_object

SCHEMA_VERSION = "p0d2hcpr-authorization-v1"


def template(preregistration_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "pending",
        "scope": AUTHORIZATION_SCOPE,
        "preregistration_sha256": preregistration_sha256,
        "approved_by": "TBD",
        "approved_at": "TBD",
        "allowed_training_runs": 1,
        "allowed_locked_qualifications": 1,
        "allows_hyperparameter_search": False,
        "allows_mappings_per_adapter_scan": False,
        "allows_kl_weight_scan": False,
        "notes": "This template is not an approval and must be approved outside the run output.",
    }


def adopt(
    output_dir: Path, source: Path, preregistration_sha256: str
) -> tuple[dict[str, Any], str]:
    if source.resolve() == output_dir.resolve() or source.resolve().is_relative_to(
        output_dir.resolve()
    ):
        raise ValueError("authorization must be created outside the CPR output")
    payload = read_json_object(source, "CPR authorization")
    required = set(template(preregistration_sha256)) - {"notes"}
    if missing := required - payload.keys():
        raise ValueError(f"CPR authorization is missing {sorted(missing)}")
    if payload["schema_version"] != SCHEMA_VERSION or payload["scope"] != AUTHORIZATION_SCOPE:
        raise PermissionError("CPR authorization identity changed")
    if (
        payload["decision"] != "approved"
        or payload["preregistration_sha256"] != preregistration_sha256
    ):
        raise PermissionError("CPR authorization does not approve this preregistration")
    if payload["approved_by"] in ("", "TBD") or payload["approved_at"] == "TBD":
        raise ValueError("CPR authorization requires approver and timestamp")
    datetime.fromisoformat(str(payload["approved_at"]).replace("Z", "+00:00"))
    if payload["allowed_training_runs"] != 1 or payload["allowed_locked_qualifications"] != 1:
        raise PermissionError("CPR authorization must permit exactly one run and qualification")
    forbidden = (
        "allows_hyperparameter_search",
        "allows_mappings_per_adapter_scan",
        "allows_kl_weight_scan",
    )
    if any(payload[name] is not False for name in forbidden):
        raise PermissionError("CPR authorization attempts to expand scope")
    digest = immutable_json_write(output_dir / "authorization.json", payload, "CPR authorization")
    return payload, digest
