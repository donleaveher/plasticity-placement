from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.io import (
    canonical_json_bytes,
    file_hash,
    immutable_json_write,
    json_hash,
)
from plasticity_placement.pathmem_ropcd_p0.config import PLAN_MANIFEST_SCHEMA_VERSION
from plasticity_placement.pathmem_ropcd_p0.identity import implementation_identity
from plasticity_placement.pathmem_ropcd_p0.planner import audit_p0_plan, compile_p0_plan
from plasticity_placement.pathmem_ropcd_p0.source import (
    _verify_git_implementation,
    verify_g1c_handoff,
)


def inspect_plan(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
) -> dict[str, Any]:
    handoff = verify_g1c_handoff(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
    )
    plan = compile_p0_plan(handoff)
    return {
        "plan_id": plan["plan_id"],
        "g1c_handoff_id": handoff["handoff_id"],
        "g1c_run_id": handoff["run_id"],
        "counts": plan["counts"],
        "recipe_sha256": plan["recipe_sha256"],
        "permissions": plan["permissions"],
        "training_started": False,
        "gpu_inference_started": False,
        "path_contrast_computed": False,
    }


def prepare_plan_bundle(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    plan_root: Path,
) -> dict[str, Any]:
    handoff = verify_g1c_handoff(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
    )
    plan = compile_p0_plan(handoff)
    implementation, implementation_sha256 = implementation_identity()
    plan_root.mkdir(parents=True, exist_ok=True)
    handoff_path = plan_root / "g1c_handoff.json"
    plan_path = plan_root / "p0_plan.json"
    immutable_json_write(handoff_path, handoff, "R-OPCD P0 G1-C handoff")
    immutable_json_write(plan_path, plan, "R-OPCD P0 plan")
    manifest_identity = {
        "schema_version": PLAN_MANIFEST_SCHEMA_VERSION,
        "g1c_handoff_id": handoff["handoff_id"],
        "plan_id": plan["plan_id"],
        "implementation": implementation,
        "implementation_sha256": implementation_sha256,
        "files": {
            "g1c_handoff.json": file_hash(handoff_path),
            "p0_plan.json": file_hash(plan_path),
        },
        "permissions": plan["permissions"],
    }
    manifest = {**manifest_identity, "manifest_id": json_hash(manifest_identity)}
    immutable_json_write(plan_root / "manifest.json", manifest, "R-OPCD P0 plan manifest")
    return verify_plan_bundle(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
    )


def verify_plan_bundle(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    plan_root: Path,
) -> dict[str, Any]:
    return _verify_plan_bundle(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
        require_current_implementation=True,
    )


def verify_recorded_plan_bundle(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    plan_root: Path,
) -> dict[str, Any]:
    """Verify an immutable old plan against its recorded Git objects."""
    return _verify_plan_bundle(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
        require_current_implementation=False,
    )


def _verify_plan_bundle(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    plan_root: Path,
    require_current_implementation: bool,
) -> dict[str, Any]:
    expected_files = {"g1c_handoff.json", "p0_plan.json", "manifest.json"}
    observed_files = {path.name for path in plan_root.iterdir() if path.is_file()}
    if observed_files != expected_files:
        raise ValueError(
            "R-OPCD P0 plan bundle file set changed: "
            f"missing={sorted(expected_files - observed_files)} "
            f"extra={sorted(observed_files - expected_files)}"
        )
    handoff = json.loads((plan_root / "g1c_handoff.json").read_text(encoding="utf-8"))
    plan = json.loads((plan_root / "p0_plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((plan_root / "manifest.json").read_text(encoding="utf-8"))
    manifest_id = manifest.pop("manifest_id", None)
    if manifest_id != json_hash(manifest):
        raise ValueError("R-OPCD P0 plan manifest identity changed")
    if manifest.get("schema_version") != PLAN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("R-OPCD P0 plan manifest schema changed")
    if manifest.get("files") != {
        "g1c_handoff.json": file_hash(plan_root / "g1c_handoff.json"),
        "p0_plan.json": file_hash(plan_root / "p0_plan.json"),
    }:
        raise ValueError("R-OPCD P0 plan bundle hashes changed")
    expected_handoff = verify_g1c_handoff(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
    )
    expected_plan = compile_p0_plan(expected_handoff)
    if canonical_json_bytes(handoff) != canonical_json_bytes(expected_handoff):
        raise ValueError("R-OPCD P0 G1-C handoff regeneration mismatch")
    if canonical_json_bytes(plan) != canonical_json_bytes(expected_plan):
        raise ValueError("R-OPCD P0 plan regeneration mismatch")
    audit_p0_plan(plan)
    if require_current_implementation:
        implementation, implementation_sha256 = implementation_identity()
        if (
            manifest.get("implementation") != implementation
            or manifest.get("implementation_sha256") != implementation_sha256
        ):
            raise ValueError("R-OPCD P0 planning implementation changed")
    else:
        implementation = manifest.get("implementation")
        if not isinstance(implementation, dict) or manifest.get(
            "implementation_sha256"
        ) != json_hash(implementation):
            raise ValueError("recorded R-OPCD P0 planning implementation changed")
        _verify_git_implementation(implementation)
    if manifest.get("permissions") != plan["permissions"]:
        raise ValueError("R-OPCD P0 planning permissions changed")
    return {
        "passed": True,
        "manifest_id": manifest_id,
        "plan_id": plan["plan_id"],
        "g1c_handoff_id": handoff["handoff_id"],
        "g1c_run_id": handoff["run_id"],
        "counts": plan["counts"],
        "permissions": plan["permissions"],
        "training_started": False,
        "gpu_inference_started": False,
        "path_contrast_computed": False,
    }


def load_plan_bundle(plan_root: Path) -> dict[str, Any]:
    return {
        "manifest": json.loads((plan_root / "manifest.json").read_text(encoding="utf-8")),
        "handoff": json.loads((plan_root / "g1c_handoff.json").read_text(encoding="utf-8")),
        "plan": json.loads((plan_root / "p0_plan.json").read_text(encoding="utf-8")),
    }
