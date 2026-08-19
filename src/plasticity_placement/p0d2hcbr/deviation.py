from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.runtime import current_code_hash
from plasticity_placement.p0d2hcbr.preflight import load_config
from plasticity_placement.p0d2hrr.io import (
    file_hash,
    immutable_json_write,
    json_hash,
    read_json_object,
)

SCHEMA_VERSION = "p0d2hcbr-budget-deviation-authorization-v1"
SCOPE = "existing_cbr_v1_full_matrix_budget_deviation_evaluation"
ADOPTED_NAME = "budget_deviation_authorization.json"
PLACEMENT_SCOPE = "exploratory_parameter_count_confounded"


def deviation_template(output_dir: Path) -> dict[str, Any]:
    from plasticity_placement.p0d2hcbr.runtime import audit_all_training_units

    output_dir = output_dir.resolve()
    manifest, _, identity = load_config(output_dir)
    if manifest.state not in {"trained", "evaluated", "complete"}:
        raise PermissionError(
            f"budget-deviation evaluation requires a fully trained CBR matrix: {manifest.state}"
        )
    audit = audit_all_training_units(output_dir)
    if audit["all_checks_passed"] is True:
        raise ValueError("CBR parameter budgets already satisfy the frozen tolerance")
    return _template_payload(manifest.payload, identity, audit)


def adopt_budget_deviation(output_dir: Path, authorization_path: Path) -> Path:
    output_dir = output_dir.resolve()
    authorization_path = authorization_path.resolve()
    if authorization_path == output_dir or authorization_path.is_relative_to(output_dir):
        raise ValueError("budget-deviation authorization must be authored outside CBR output")
    manifest, _, _ = load_config(output_dir)
    if isinstance(manifest.payload.get("budget_deviation"), dict):
        validate_adopted_deviation(output_dir)
        return output_dir / ADOPTED_NAME
    if manifest.state != "trained":
        raise PermissionError(
            f"budget-deviation authorization requires state=trained, found {manifest.state}"
        )
    if manifest.payload.get("evaluation_claims"):
        raise PermissionError("budget-deviation authorization requires zero evaluation claims")
    expected = deviation_template(output_dir)
    payload = read_json_object(authorization_path, "CBR budget-deviation authorization")
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
        raise PermissionError(f"CBR budget-deviation authorization differs: {differences}")
    if payload.get("approved_by") in {None, "", "TBD"} or payload.get("approved_at") == "TBD":
        raise ValueError("budget-deviation authorization requires approver and timestamp")
    approved_at = datetime.fromisoformat(str(payload["approved_at"]).replace("Z", "+00:00"))
    if approved_at.tzinfo is None:
        raise ValueError("budget-deviation approval timestamp must be timezone-aware")
    adopted_path = output_dir / ADOPTED_NAME
    digest = immutable_json_write(
        adopted_path, payload, "CBR budget-deviation authorization"
    )
    manifest.payload["budget_deviation"] = {
        "authorization_sha256": digest,
        "recovery_code_sha256": payload["recovery_code_sha256"],
        "budget_audit_sha256": payload["budget_audit_sha256"],
        "placement_comparison_scope": PLACEMENT_SCOPE,
        "approved_by": payload["approved_by"],
        "approved_at": payload["approved_at"],
        "additional_training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    manifest.save()
    return adopted_path


def validate_adopted_deviation(
    output_dir: Path, audit: dict[str, Any] | None = None
) -> dict[str, Any]:
    from plasticity_placement.p0d2hcbr.runtime import audit_all_training_units

    output_dir = output_dir.resolve()
    manifest, _, identity = load_config(output_dir)
    recorded = manifest.payload.get("budget_deviation")
    if not isinstance(recorded, dict):
        raise PermissionError("CBR budget deviation has not been externally authorized")
    adopted_path = output_dir / ADOPTED_NAME
    if not adopted_path.is_file() or file_hash(adopted_path) != recorded.get(
        "authorization_sha256"
    ):
        raise PermissionError("CBR budget-deviation authorization changed")
    audit = audit or audit_all_training_units(output_dir)
    if audit["all_checks_passed"] is True:
        raise ValueError("recorded budget deviation no longer exists")
    payload = read_json_object(adopted_path, "CBR budget-deviation authorization")
    expected = _template_payload(manifest.payload, identity, audit)
    fixed_keys = set(expected) - {"decision", "approved_by", "approved_at", "notes"}
    differences = {
        key: {"expected": expected[key], "observed": payload.get(key)}
        for key in fixed_keys
        if payload.get(key) != expected[key]
    }
    if (
        differences
        or payload.get("decision") != "approved"
        or payload.get("approved_by") in {None, "", "TBD"}
        or payload.get("approved_at") in {None, "TBD"}
        or recorded.get("recovery_code_sha256") != current_code_hash()
        or recorded.get("budget_audit_sha256") != json_hash(audit)
        or recorded.get("placement_comparison_scope") != PLACEMENT_SCOPE
        or recorded.get("additional_training_authorized") is not False
        or recorded.get("mappings_per_adapter_authorized") is not False
    ):
        raise PermissionError(
            f"CBR budget-deviation authorization provenance differs: {differences}"
        )
    return {
        "authorization_sha256": recorded["authorization_sha256"],
        "recovery_code_sha256": recorded["recovery_code_sha256"],
        "budget_audit_sha256": recorded["budget_audit_sha256"],
        "placement_comparison_scope": PLACEMENT_SCOPE,
    }


def evaluation_code_sha256(output_dir: Path, identity: dict[str, Any]) -> str:
    observed = current_code_hash()
    if observed == identity["code_sha256"]:
        return observed
    recovery = validate_adopted_deviation(output_dir)
    if recovery["recovery_code_sha256"] != observed:
        raise ValueError("evaluation code differs from the authorized recovery code")
    return observed


def _template_payload(
    manifest: dict[str, Any], identity: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "pending",
        "scope": SCOPE,
        "cbr_run_id": manifest["run_id"],
        "preregistration_sha256": identity["preregistration_sha256"],
        "experiment_code_sha256": identity["code_sha256"],
        "recovery_code_sha256": current_code_hash(),
        "budget_audit_sha256": json_hash(audit),
        "observed_budget_comparisons": audit["comparisons"],
        "frozen_relative_tolerance": audit["relative_tolerance"],
        "placement_comparison_scope": PLACEMENT_SCOPE,
        "allowed_locked_evaluations": 12,
        "allows_additional_training": False,
        "allows_hyperparameter_search": False,
        "allows_checkpoint_selection": False,
        "allows_mappings_per_adapter_scan": False,
        "approved_by": "TBD",
        "approved_at": "TBD",
        "notes": (
            "Authorizes full descriptive evaluation of the existing 12 adapters. "
            "Cross-placement conclusions remain exploratory and parameter-count-confounded."
        ),
    }
