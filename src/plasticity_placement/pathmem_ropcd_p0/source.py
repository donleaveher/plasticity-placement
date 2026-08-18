from __future__ import annotations

import json
import subprocess
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.io import canonical_json_bytes, file_hash, json_hash
from plasticity_placement.pathmem_consolidation_exec.analysis import summarize_g1c
from plasticity_placement.pathmem_consolidation_exec.config import (
    AUTHORIZATION_SCHEMA_VERSION as G1C_AUTHORIZATION_SCHEMA_VERSION,
)
from plasticity_placement.pathmem_consolidation_exec.config import (
    G1C_EXECUTION_RECIPE,
)
from plasticity_placement.pathmem_consolidation_exec.identity import verify_source_bundle
from plasticity_placement.pathmem_consolidation_exec.manifest import G1CExecutionManifest
from plasticity_placement.pathmem_consolidation_exec.runtime import (
    RunContext,
    _integrity_checks,
    _load_result_matrix,
    _render_report,
)

HANDOFF_SCHEMA_VERSION = "pathmem-g1c-to-ropcd-p0-handoff-v1"
G1C_PERMISSIONS = {
    "g1c_training_authorized": True,
    "g1c_gpu_inference_authorized": True,
    "p0_authorized": False,
    "path_contrast_authorized": False,
    "kill_or_reserve_access_authorized": False,
    "rl_controller_authorized": False,
    "automatic_hyperparameter_search_authorized": False,
}


def verify_g1c_handoff(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
) -> dict[str, Any]:
    source = verify_source_bundle(bundle_root, parent_manifest)
    authorization = _verify_recorded_authorization(g1c_run_root, source)
    manifest = G1CExecutionManifest.load(g1c_run_root / "manifest.json")
    identity = manifest.payload["identity"]
    expected_planned = {
        str(unit["unit_id"]): {
            "unit_id": str(unit["unit_id"]),
            "sequence_index": int(unit["sequence_index"]),
            "item_id": str(unit["item_id"]),
            "terminal_state": str(unit["terminal_state"]),
            "memory_key": str(unit["memory_key"]),
            "unit_sha256": json_hash(unit),
        }
        for unit in source["plan"]["units"]
    }
    expected_source = {
        key: source[key]
        for key in (
            "manifest_id",
            "manifest_sha256",
            "plan_id",
            "plan_sha256",
            "parent_g0_v1_manifest_id",
        )
    }
    if identity.get("source") != expected_source:
        raise ValueError("G1-C manifest source binding changed")
    if identity.get("authorization_id") != authorization["authorization_id"]:
        raise ValueError("G1-C manifest authorization binding changed")
    if identity.get("recipe") != G1C_EXECUTION_RECIPE.to_dict():
        raise ValueError("G1-C manifest recipe changed")
    if identity.get("recipe_sha256") != json_hash(G1C_EXECUTION_RECIPE.to_dict()):
        raise ValueError("G1-C manifest recipe hash changed")
    if identity.get("implementation_sha256") != authorization["implementation_sha256"]:
        raise ValueError("G1-C manifest implementation binding changed")
    if identity.get("planned_units") != expected_planned:
        raise ValueError("G1-C manifest planned units changed")
    manifest.require_all_verified()

    context = RunContext(
        source=source,
        authorization=authorization,
        manifest=manifest,
        recipe=G1C_EXECUTION_RECIPE,
    )
    teacher_rows, evaluation_rows = _load_result_matrix(g1c_run_root, context)
    integrity = _integrity_checks(bundle_root, g1c_run_root, context, evaluation_rows)
    if not integrity or not all(integrity.values()):
        failed = sorted(key for key, passed in integrity.items() if not passed)
        raise ValueError(f"G1-C transitive integrity failed: {failed}")
    expected_summary = summarize_g1c(
        teacher_rows,
        evaluation_rows,
        run_id=str(manifest.payload["run_id"]),
        source_manifest_id=str(source["manifest_id"]),
        plan_id=str(source["plan_id"]),
        integrity_checks=integrity,
    )
    summary_path = g1c_run_root / "aggregate/summary.json"
    report_path = g1c_run_root / "aggregate/report.md"
    observed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if canonical_json_bytes(observed_summary) != canonical_json_bytes(expected_summary):
        raise ValueError("G1-C handoff summary regeneration mismatch")
    if report_path.read_text(encoding="utf-8") != _render_report(expected_summary):
        raise ValueError("G1-C handoff report regeneration mismatch")
    aggregate = manifest.payload.get("aggregate", {})
    if (
        aggregate.get("state") != "complete"
        or aggregate.get("summary_sha256") != file_hash(summary_path)
        or aggregate.get("report_sha256") != file_hash(report_path)
        or canonical_json_bytes(aggregate.get("gate"))
        != canonical_json_bytes(expected_summary["gate"])
    ):
        raise ValueError("G1-C handoff aggregate binding changed")
    if expected_summary["gate"].get("passed") is not True:
        raise PermissionError("R-OPCD P0 requires a passing G1-C gate")
    if expected_summary.get("p0_runner_implementation_review_eligible") is not True:
        raise PermissionError("G1-C did not permit P0 implementation review")
    if any(
        manifest.payload.get(field) is not False
        for field in ("p0_authorized", "path_contrast_computed", "kill_or_reserve_accessed")
    ):
        raise ValueError("G1-C handoff contains downstream execution state")
    handoff_identity = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "passed": True,
        "run_id": manifest.payload["run_id"],
        "source_manifest_id": source["manifest_id"],
        "plan_id": source["plan_id"],
        "g1c_authorization_id": authorization["authorization_id"],
        "g1c_implementation_sha256": authorization["implementation_sha256"],
        "g1c_git_revision": authorization["implementation"]["git_revision"],
        "summary_sha256": file_hash(summary_path),
        "g1c_gate": expected_summary["gate"],
        "p0_runner_implementation_review_eligible": True,
        "p0_authorized": False,
        "path_contrast_computed": False,
    }
    return {**handoff_identity, "handoff_id": json_hash(handoff_identity)}


def _verify_recorded_authorization(
    g1c_run_root: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    path = g1c_run_root / "authorization.json"
    authorization = json.loads(path.read_text(encoding="utf-8"))
    authorization_id = authorization.pop("authorization_id", None)
    required = {
        "schema_version",
        "decision",
        "scope",
        "source",
        "recipe",
        "recipe_sha256",
        "implementation",
        "implementation_sha256",
        "output_root",
        "permissions",
        "approved_by",
        "approved_at",
    }
    if set(authorization) != required:
        raise ValueError("recorded G1-C authorization field set changed")
    expected_source = {
        key: source[key]
        for key in (
            "manifest_id",
            "manifest_sha256",
            "plan_id",
            "plan_sha256",
            "parent_g0_v1_manifest_id",
        )
    }
    checks = (
        authorization.get("schema_version") == G1C_AUTHORIZATION_SCHEMA_VERSION,
        authorization.get("decision") == "approved",
        authorization.get("scope") == "g1c_r_opcd_cuda_qualification_only",
        authorization.get("source") == expected_source,
        authorization.get("recipe") == G1C_EXECUTION_RECIPE.to_dict(),
        authorization.get("recipe_sha256") == json_hash(G1C_EXECUTION_RECIPE.to_dict()),
        authorization.get("permissions") == G1C_PERMISSIONS,
        authorization.get("output_root") == str(g1c_run_root.resolve()),
        isinstance(authorization.get("approved_by"), str)
        and bool(authorization["approved_by"].strip()),
        _is_timezone_aware_iso8601(authorization.get("approved_at")),
    )
    if not all(checks):
        raise ValueError("recorded G1-C authorization binding changed")
    implementation = authorization.get("implementation")
    if not isinstance(implementation, dict):
        raise ValueError("recorded G1-C implementation identity is missing")
    if authorization.get("implementation_sha256") != json_hash(implementation):
        raise ValueError("recorded G1-C implementation hash changed")
    _verify_git_implementation(implementation)
    expected_id = json_hash({**authorization, "approved_by": authorization["approved_by"].strip()})
    if authorization_id != expected_id:
        raise ValueError("recorded G1-C authorization identity changed")
    return {**authorization, "authorization_id": expected_id}


def _verify_git_implementation(implementation: dict[str, Any]) -> None:
    revision = implementation.get("git_revision")
    source_files = implementation.get("source_files")
    if not isinstance(revision, str) or not revision or not isinstance(source_files, dict):
        raise ValueError("recorded G1-C Git identity is incomplete")
    repository_root = Path(__file__).resolve().parents[3]
    exists = subprocess.run(
        ["git", "-C", str(repository_root), "cat-file", "-e", f"{revision}^{{commit}}"],
        check=False,
        capture_output=True,
    )
    if exists.returncode != 0:
        raise ValueError(f"recorded G1-C Git revision is unavailable: {revision}")
    for relative_path, expected_hash in sorted(source_files.items()):
        completed = subprocess.run(
            ["git", "-C", str(repository_root), "show", f"{revision}:{relative_path}"],
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0 or sha256(completed.stdout).hexdigest() != expected_hash:
            raise ValueError(f"recorded G1-C source hash changed: {relative_path}")


def _is_timezone_aware_iso8601(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None
