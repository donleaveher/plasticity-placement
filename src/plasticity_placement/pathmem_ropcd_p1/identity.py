from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.io import file_hash, immutable_json_write, json_hash
from plasticity_placement.pathmem_ropcd_p1.config import (
    AUTHORIZATION_SCHEMA_VERSION,
    BENCHMARK_AUTHORIZATION_SCHEMA_VERSION,
    BENCHMARK_PROFILE_SCHEMA_VERSION,
    P1_ROPCD_RECIPE,
    benchmark_permissions,
    execution_permissions,
)


def implementation_identity() -> tuple[dict[str, Any], str]:
    repository_root = Path(__file__).resolve().parents[3]
    roots = (
        repository_root / "src/plasticity_placement/pathmem_ropcd_p1",
        repository_root / "src/plasticity_placement/pathmem_ropcd_p0",
        repository_root / "src/plasticity_placement/pathmem_consolidation_exec",
        repository_root / "src/plasticity_placement/pathmem_consolidation",
        repository_root / "src/plasticity_placement/pathmem",
    )
    sources = {
        path.relative_to(repository_root).as_posix(): file_hash(path)
        for root in roots
        for path in sorted(root.glob("*.py"))
    }
    dependencies = (
        repository_root / "src/plasticity_placement/p0d2hfc/scoring.py",
        repository_root / "src/plasticity_placement/pathmem_exec/artifacts.py",
        repository_root / "src/plasticity_placement/pathmem_exec/environment.py",
        repository_root / "src/plasticity_placement/training/model_utils.py",
        repository_root / "pyproject.toml",
        repository_root / "uv.lock",
    )
    sources.update(
        {path.relative_to(repository_root).as_posix(): file_hash(path) for path in dependencies}
    )
    completed = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip() if completed.returncode == 0 else "NO_GIT_REVISION"
    payload = {"git_revision": revision, "source_files": sources}
    return payload, json_hash(payload)


def build_benchmark_authorization_template(
    *, plan_root: Path, output_root: Path, plan_report: dict[str, Any]
) -> dict[str, Any]:
    return _template(
        schema_version=BENCHMARK_AUTHORIZATION_SCHEMA_VERSION,
        scope="ropcd_p1_hardware_dev_resource_benchmark_only",
        plan_root=plan_root,
        output_root=output_root,
        plan_report=plan_report,
        permissions=benchmark_permissions(),
    )


def build_execution_authorization_template(
    *,
    plan_root: Path,
    output_root: Path,
    plan_report: dict[str, Any],
    resource_profile_path: Path,
) -> dict[str, Any]:
    profile = json.loads(resource_profile_path.read_text(encoding="utf-8"))
    profile_identity = {key: value for key, value in profile.items() if key != "profile_id"}
    _, current_implementation_sha256 = implementation_identity()
    if (
        profile.get("schema_version") != BENCHMARK_PROFILE_SCHEMA_VERSION
        or profile.get("profile_id") != json_hash(profile_identity)
        or profile.get("verified") is not True
        or profile.get("plan_id") != plan_report.get("plan_id")
        or profile.get("g2_handoff_id") != plan_report.get("g2_handoff_id")
        or profile.get("implementation_sha256") != current_implementation_sha256
        or profile.get("scientific_outcomes_computed") is not False
        or profile.get("formal_p1_authorized") is not False
        or profile.get("p1b_authorized") is not False
        or profile.get("p2_authorized") is not False
    ):
        raise PermissionError("R-OPCD P1 authorization requires a verified resource profile")
    template = _template(
        schema_version=AUTHORIZATION_SCHEMA_VERSION,
        scope="ropcd_specific_p1_kill_split_only",
        plan_root=plan_root,
        output_root=output_root,
        plan_report=plan_report,
        permissions=execution_permissions(),
    )
    return {
        **template,
        "resource_profile": {
            "profile_id": profile["profile_id"],
            "profile_sha256": file_hash(resource_profile_path),
            "benchmark_plan_id": profile["plan_id"],
            "benchmark_implementation_sha256": profile["implementation_sha256"],
            "projected_p1_wall_seconds": profile["projection"]["p1_wall_seconds"],
        },
    }


def verify_external_approval(
    approval_path: Path, *, expected_template: dict[str, Any]
) -> dict[str, Any]:
    return verify_external_approval_object(
        json.loads(approval_path.read_text(encoding="utf-8")),
        expected_template=expected_template,
    )


def verify_external_approval_object(
    approval: dict[str, Any], *, expected_template: dict[str, Any]
) -> dict[str, Any]:
    expected_keys = set(expected_template) | {"approved_by", "approved_at"}
    if set(approval) != expected_keys:
        raise ValueError("R-OPCD P1 authorization field set changed")
    differences = {
        key: {"expected": value, "observed": approval.get(key)}
        for key, value in expected_template.items()
        if key != "decision" and approval.get(key) != value
    }
    if differences:
        raise ValueError(f"R-OPCD P1 authorization binding changed: {differences}")
    if approval.get("decision") != "approved":
        raise PermissionError("R-OPCD P1 operation requires decision=approved")
    approved_by = approval.get("approved_by")
    approved_at = approval.get("approved_at")
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise ValueError("R-OPCD P1 authorization requires a human identifier")
    if not isinstance(approved_at, str):
        raise ValueError("R-OPCD P1 authorization requires an approval time")
    try:
        parsed = datetime.fromisoformat(approved_at)
    except ValueError as error:
        raise ValueError("R-OPCD P1 approval time is not ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError("R-OPCD P1 approval time must include a timezone")
    identity = {**approval, "approved_by": approved_by.strip()}
    return {**identity, "authorization_id": json_hash(identity)}


def adopt_authorization(output_root: Path, approval: dict[str, Any]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "authorization.json"
    immutable_json_write(path, approval, "R-OPCD P1 authorization")
    return path


def verify_adopted_authorization(
    output_root: Path, *, expected_template: dict[str, Any]
) -> dict[str, Any]:
    payload = json.loads((output_root / "authorization.json").read_text(encoding="utf-8"))
    authorization_id = payload.pop("authorization_id", None)
    expected = verify_external_approval_object(payload, expected_template=expected_template)
    if authorization_id != expected["authorization_id"]:
        raise ValueError("adopted R-OPCD P1 authorization identity changed")
    return expected


def _template(
    *,
    schema_version: str,
    scope: str,
    plan_root: Path,
    output_root: Path,
    plan_report: dict[str, Any],
    permissions: dict[str, bool],
) -> dict[str, Any]:
    if plan_report.get("passed") is not True or not plan_report.get("manifest_id"):
        raise PermissionError("R-OPCD P1 authorization requires a verified immutable plan")
    implementation, implementation_sha256 = implementation_identity()
    return {
        "schema_version": schema_version,
        "decision": "pending_human_review",
        "scope": scope,
        "plan": {
            "manifest_id": plan_report["manifest_id"],
            "manifest_sha256": file_hash(plan_root / "manifest.json"),
            "plan_id": plan_report["plan_id"],
            "plan_sha256": file_hash(plan_root / "p1_plan.json"),
            "g2_handoff_id": plan_report["g2_handoff_id"],
            "g2_repair_id": plan_report["g2_repair_id"],
        },
        "recipe": P1_ROPCD_RECIPE.to_dict(),
        "recipe_sha256": json_hash(P1_ROPCD_RECIPE.to_dict()),
        "implementation": implementation,
        "implementation_sha256": implementation_sha256,
        "output_root": str(output_root.resolve()),
        "permissions": permissions,
    }
