from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.io import file_hash, immutable_json_write, json_hash
from plasticity_placement.pathmem_ropcd_p0.config import (
    AUTHORIZATION_SCHEMA_VERSION,
    P0_ROPCD_RECIPE,
    execution_permissions,
)


def implementation_identity() -> tuple[dict[str, Any], str]:
    repository_root = Path(__file__).resolve().parents[3]
    source_roots = (
        repository_root / "src/plasticity_placement/pathmem_ropcd_p0",
        repository_root / "src/plasticity_placement/pathmem_consolidation_exec",
        repository_root / "src/plasticity_placement/pathmem_consolidation",
        repository_root / "src/plasticity_placement/pathmem",
    )
    sources = {
        path.relative_to(repository_root).as_posix(): file_hash(path)
        for source_root in source_roots
        for path in sorted(source_root.glob("*.py"))
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
    revision = _git_output(repository_root, ("rev-parse", "HEAD"))
    payload = {"git_revision": revision, "source_files": sources}
    return payload, json_hash(payload)


def build_authorization_template(
    *,
    plan_root: Path,
    output_root: Path,
    plan_report: dict[str, Any],
) -> dict[str, Any]:
    if plan_report.get("passed") is not True:
        raise PermissionError("R-OPCD P0 authorization requires a verified CPU plan")
    implementation, implementation_sha256 = implementation_identity()
    return {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "decision": "pending_human_review",
        "scope": "ropcd_specific_p0_smoke_only",
        "plan": {
            "manifest_id": plan_report["manifest_id"],
            "manifest_sha256": file_hash(plan_root / "manifest.json"),
            "plan_id": plan_report["plan_id"],
            "plan_sha256": file_hash(plan_root / "p0_plan.json"),
            "g1c_handoff_id": plan_report["g1c_handoff_id"],
            "g1c_run_id": plan_report["g1c_run_id"],
        },
        "recipe": P0_ROPCD_RECIPE.to_dict(),
        "recipe_sha256": json_hash(P0_ROPCD_RECIPE.to_dict()),
        "implementation": implementation,
        "implementation_sha256": implementation_sha256,
        "output_root": str(output_root.resolve()),
        "permissions": execution_permissions(),
    }


def verify_external_approval(
    approval_path: Path,
    *,
    expected_template: dict[str, Any],
) -> dict[str, Any]:
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    return verify_external_approval_object(approval, expected_template=expected_template)


def verify_external_approval_object(
    approval: dict[str, Any],
    *,
    expected_template: dict[str, Any],
) -> dict[str, Any]:
    expected_keys = set(expected_template) | {"approved_by", "approved_at"}
    if set(approval) != expected_keys:
        raise ValueError(
            "R-OPCD P0 authorization field set changed: "
            f"missing={sorted(expected_keys - set(approval))} "
            f"extra={sorted(set(approval) - expected_keys)}"
        )
    differences = {
        key: {"expected": value, "observed": approval.get(key)}
        for key, value in expected_template.items()
        if key != "decision" and approval.get(key) != value
    }
    if differences:
        raise ValueError(f"R-OPCD P0 authorization binding changed: {differences}")
    if approval.get("decision") != "approved":
        raise PermissionError("R-OPCD P0 execution requires decision=approved")
    approved_by = approval.get("approved_by")
    approved_at = approval.get("approved_at")
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise ValueError("R-OPCD P0 authorization requires a human identifier")
    if not isinstance(approved_at, str):
        raise ValueError("R-OPCD P0 authorization requires an approval time")
    try:
        parsed = datetime.fromisoformat(approved_at)
    except ValueError as error:
        raise ValueError("R-OPCD P0 approval time is not ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError("R-OPCD P0 approval time must include a timezone")
    identity = {**approval, "approved_by": approved_by.strip()}
    return {**identity, "authorization_id": json_hash(identity)}


def adopt_authorization(output_root: Path, approval: dict[str, Any]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "authorization.json"
    immutable_json_write(path, approval, "R-OPCD P0 authorization")
    return path


def verify_adopted_authorization(
    output_root: Path,
    *,
    expected_template: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads((output_root / "authorization.json").read_text(encoding="utf-8"))
    authorization_id = payload.pop("authorization_id", None)
    expected = verify_external_approval_object(payload, expected_template=expected_template)
    if authorization_id != expected["authorization_id"]:
        raise ValueError("adopted R-OPCD P0 authorization identity changed")
    return expected


def _git_output(root: Path, arguments: tuple[str, ...]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "NO_GIT_REVISION"
