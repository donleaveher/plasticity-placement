from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.io import canonical_json_bytes, file_hash, json_hash
from plasticity_placement.pathmem_ropcd_p0.repair import (
    REPAIR_AUTHORIZATION_SCHEMA_VERSION,
    REPAIR_MANIFEST_SCHEMA_VERSION,
    REPAIR_RULE_ID,
    _repair_report,
    _repaired_summary,
    _source_snapshot,
    repair_permissions,
)
from plasticity_placement.pathmem_ropcd_p0.source import _verify_git_implementation
from plasticity_placement.pathmem_ropcd_p1.config import G2_HANDOFF_SCHEMA_VERSION


def verify_g2_handoff(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    p0_plan_root: Path,
    p0_run_root: Path,
    repair_root: Path,
) -> dict[str, Any]:
    context, rows, integrity = _source_snapshot(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=p0_plan_root,
        output_root=p0_run_root,
    )
    authorization = _verify_recorded_repair_authorization(
        repair_root=repair_root,
        p0_run_root=p0_run_root,
        context=context,
        rows=rows,
    )
    expected_summary = _repaired_summary(context, rows, integrity, authorization)
    summary_path = repair_root / "aggregate/summary.json"
    report_path = repair_root / "aggregate/report.md"
    observed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if canonical_json_bytes(observed_summary) != canonical_json_bytes(expected_summary):
        raise ValueError("recorded R-OPCD G2 repaired summary regeneration mismatch")
    if report_path.read_text(encoding="utf-8") != _repair_report(expected_summary):
        raise ValueError("recorded R-OPCD G2 repaired report regeneration mismatch")
    repair_manifest = _verify_recorded_repair_manifest(
        repair_root=repair_root,
        context=context,
        rows=rows,
        authorization=authorization,
    )
    gate = expected_summary.get("gate")
    if not isinstance(gate, dict) or gate.get("gate") != "G2" or gate.get("passed") is not True:
        raise PermissionError("R-OPCD P1 requires a passing repaired G2 gate")
    identity = {
        "schema_version": G2_HANDOFF_SCHEMA_VERSION,
        "passed": True,
        "repair_id": repair_manifest["repair_id"],
        "source_run_id": context.manifest.payload["run_id"],
        "source_plan_id": context.bundle["plan"]["plan_id"],
        "source_matrix_sha256": json_hash(rows),
        "g2_summary_sha256": file_hash(summary_path),
        "g2_gate": gate,
        "repair_implementation_sha256": repair_manifest["implementation_sha256"],
        "p1_implementation_review_eligible": True,
        "p1_authorized": False,
        "kill_or_reserve_accessed": False,
    }
    return {**identity, "handoff_id": json_hash(identity)}


def _verify_recorded_repair_authorization(
    *,
    repair_root: Path,
    p0_run_root: Path,
    context: Any,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    path = repair_root / "authorization.json"
    observed = json.loads(path.read_text(encoding="utf-8"))
    authorization_id = observed.pop("authorization_id", None)
    expected_keys = {
        "schema_version",
        "decision",
        "scope",
        "source",
        "repair_rule",
        "implementation",
        "implementation_sha256",
        "source_output_root",
        "repair_output_root",
        "permissions",
        "approved_by",
        "approved_at",
    }
    if set(observed) != expected_keys:
        raise ValueError("recorded R-OPCD G2 repair authorization field set changed")
    expected_source = {
        "run_id": context.manifest.payload["run_id"],
        "plan_id": context.bundle["plan"]["plan_id"],
        "plan_manifest_id": context.manifest.payload["identity"]["plan_manifest_id"],
        "authorization_id": context.manifest.payload["identity"]["authorization_id"],
        "execution_implementation_sha256": context.manifest.payload["identity"][
            "implementation_sha256"
        ],
        "matrix_sha256": json_hash(rows),
        "rows": len(rows),
        "source_execution_manifest_sha256": file_hash(p0_run_root / "manifest.json"),
        "source_execution_authorization_sha256": file_hash(
            p0_run_root / "authorization.json"
        ),
    }
    expected_rule = {
        "repair_rule_id": REPAIR_RULE_ID,
        "missing_field": "path_qualification.obsolete_action",
        "recovery_source": "matching_parametric_path_core_rows",
        "required_reference_rows_per_item_path": 14,
        "required_qualification_rows_per_item_path": 4,
        "probability_outcomes_used_for_label_recovery": False,
        "failed_paths_filtered": False,
    }
    checks = (
        observed.get("schema_version") == REPAIR_AUTHORIZATION_SCHEMA_VERSION,
        observed.get("decision") == "approved",
        observed.get("scope") == "ropcd_p0_qualification_label_analysis_repair_only",
        observed.get("source") == expected_source,
        observed.get("repair_rule") == expected_rule,
        observed.get("source_output_root") == str(p0_run_root.resolve()),
        observed.get("repair_output_root") == str(repair_root.resolve()),
        observed.get("permissions") == repair_permissions(),
        isinstance(observed.get("approved_by"), str)
        and bool(observed["approved_by"].strip()),
        _timezone_aware(observed.get("approved_at")),
    )
    if not all(checks):
        raise ValueError("recorded R-OPCD G2 repair authorization binding changed")
    implementation = observed.get("implementation")
    if not isinstance(implementation, dict) or observed.get(
        "implementation_sha256"
    ) != json_hash(implementation):
        raise ValueError("recorded R-OPCD G2 repair implementation changed")
    _verify_git_implementation(implementation)
    identity = {**observed, "approved_by": observed["approved_by"].strip()}
    expected_id = json_hash(identity)
    if authorization_id != expected_id:
        raise ValueError("recorded R-OPCD G2 repair authorization identity changed")
    return {**identity, "authorization_id": expected_id}


def _verify_recorded_repair_manifest(
    *,
    repair_root: Path,
    context: Any,
    rows: list[dict[str, Any]],
    authorization: dict[str, Any],
) -> dict[str, Any]:
    path = repair_root / "repair_manifest.json"
    observed = json.loads(path.read_text(encoding="utf-8"))
    repair_id = observed.pop("repair_id", None)
    expected_keys = {
        "schema_version",
        "repair_rule_id",
        "source_run_id",
        "source_plan_id",
        "source_matrix_sha256",
        "source_execution_implementation_sha256",
        "source_execution_manifest_sha256",
        "source_execution_authorization_sha256",
        "authorization_id",
        "implementation",
        "implementation_sha256",
        "files",
        "permissions",
        "source_mutated",
        "training_started",
        "gpu_inference_started",
        "p1_authorized",
    }
    if set(observed) != expected_keys:
        raise ValueError("recorded R-OPCD G2 repair manifest field set changed")
    if repair_id != json_hash(observed):
        raise ValueError("recorded R-OPCD G2 repair manifest identity changed")
    implementation = observed.get("implementation")
    if not isinstance(implementation, dict) or observed.get(
        "implementation_sha256"
    ) != json_hash(implementation):
        raise ValueError("recorded R-OPCD G2 repair manifest implementation changed")
    _verify_git_implementation(implementation)
    expected_files = {
        "aggregate/summary.json": file_hash(repair_root / "aggregate/summary.json"),
        "aggregate/report.md": file_hash(repair_root / "aggregate/report.md"),
    }
    checks = (
        observed.get("schema_version") == REPAIR_MANIFEST_SCHEMA_VERSION,
        observed.get("repair_rule_id") == REPAIR_RULE_ID,
        observed.get("source_run_id") == context.manifest.payload["run_id"],
        observed.get("source_plan_id") == context.bundle["plan"]["plan_id"],
        observed.get("source_matrix_sha256") == json_hash(rows),
        observed.get("source_execution_implementation_sha256")
        == context.manifest.payload["identity"]["implementation_sha256"],
        observed.get("authorization_id") == authorization["authorization_id"],
        observed.get("source_execution_manifest_sha256")
        == authorization["source"]["source_execution_manifest_sha256"],
        observed.get("source_execution_authorization_sha256")
        == authorization["source"]["source_execution_authorization_sha256"],
        observed.get("implementation") == authorization["implementation"],
        observed.get("implementation_sha256") == authorization["implementation_sha256"],
        observed.get("files") == expected_files,
        observed.get("permissions") == repair_permissions(),
        observed.get("source_mutated") is False,
        observed.get("training_started") is False,
        observed.get("gpu_inference_started") is False,
        observed.get("p1_authorized") is False,
    )
    if not all(checks):
        raise ValueError("recorded R-OPCD G2 repair manifest binding changed")
    return {**observed, "repair_id": repair_id}


def _timezone_aware(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None
