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
from plasticity_placement.pathmem_ropcd_p1.config import PLAN_MANIFEST_SCHEMA_VERSION
from plasticity_placement.pathmem_ropcd_p1.identity import implementation_identity
from plasticity_placement.pathmem_ropcd_p1.planner import audit_p1_plan, compile_p1_plan
from plasticity_placement.pathmem_ropcd_p1.source import verify_g2_handoff


def inspect_plan(**source: Any) -> dict[str, Any]:
    handoff = verify_g2_handoff(**source)
    plan = compile_p1_plan(handoff)
    return _report(plan, handoff, manifest_id=None)


def prepare_plan_bundle(*, plan_root: Path, **source: Any) -> dict[str, Any]:
    handoff = verify_g2_handoff(**source)
    plan = compile_p1_plan(handoff)
    implementation, implementation_sha256 = implementation_identity()
    plan_root.mkdir(parents=True, exist_ok=True)
    handoff_path = plan_root / "g2_handoff.json"
    plan_path = plan_root / "p1_plan.json"
    immutable_json_write(handoff_path, handoff, "R-OPCD P1 G2 handoff")
    immutable_json_write(plan_path, plan, "R-OPCD P1 plan")
    identity = {
        "schema_version": PLAN_MANIFEST_SCHEMA_VERSION,
        "g2_handoff_id": handoff["handoff_id"],
        "plan_id": plan["plan_id"],
        "implementation": implementation,
        "implementation_sha256": implementation_sha256,
        "files": {
            "g2_handoff.json": file_hash(handoff_path),
            "p1_plan.json": file_hash(plan_path),
        },
        "permissions": plan["permissions"],
    }
    immutable_json_write(
        plan_root / "manifest.json",
        {**identity, "manifest_id": json_hash(identity)},
        "R-OPCD P1 plan manifest",
    )
    return verify_plan_bundle(plan_root=plan_root, **source)


def verify_plan_bundle(*, plan_root: Path, **source: Any) -> dict[str, Any]:
    expected_names = {"g2_handoff.json", "p1_plan.json", "manifest.json"}
    observed_names = {path.name for path in plan_root.iterdir() if path.is_file()}
    if observed_names != expected_names:
        raise ValueError(
            "R-OPCD P1 plan file set changed: "
            f"missing={sorted(expected_names - observed_names)} "
            f"extra={sorted(observed_names - expected_names)}"
        )
    handoff = json.loads((plan_root / "g2_handoff.json").read_text(encoding="utf-8"))
    plan = json.loads((plan_root / "p1_plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((plan_root / "manifest.json").read_text(encoding="utf-8"))
    manifest_id = manifest.pop("manifest_id", None)
    if manifest_id != json_hash(manifest):
        raise ValueError("R-OPCD P1 plan manifest identity changed")
    if manifest.get("schema_version") != PLAN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("R-OPCD P1 plan manifest schema changed")
    if manifest.get("files") != {
        "g2_handoff.json": file_hash(plan_root / "g2_handoff.json"),
        "p1_plan.json": file_hash(plan_root / "p1_plan.json"),
    }:
        raise ValueError("R-OPCD P1 plan hashes changed")
    expected_handoff = verify_g2_handoff(**source)
    expected_plan = compile_p1_plan(expected_handoff)
    if canonical_json_bytes(handoff) != canonical_json_bytes(expected_handoff):
        raise ValueError("R-OPCD P1 G2 handoff regeneration mismatch")
    if canonical_json_bytes(plan) != canonical_json_bytes(expected_plan):
        raise ValueError("R-OPCD P1 plan regeneration mismatch")
    audit_p1_plan(plan)
    implementation, implementation_sha256 = implementation_identity()
    if (
        manifest.get("implementation") != implementation
        or manifest.get("implementation_sha256") != implementation_sha256
    ):
        raise ValueError("R-OPCD P1 planning implementation changed")
    if manifest.get("permissions") != plan["permissions"]:
        raise ValueError("R-OPCD P1 planning permissions changed")
    return _report(plan, handoff, manifest_id=str(manifest_id))


def load_plan_bundle(plan_root: Path) -> dict[str, Any]:
    return {
        "manifest": json.loads((plan_root / "manifest.json").read_text(encoding="utf-8")),
        "handoff": json.loads((plan_root / "g2_handoff.json").read_text(encoding="utf-8")),
        "plan": json.loads((plan_root / "p1_plan.json").read_text(encoding="utf-8")),
    }


def _report(
    plan: dict[str, Any], handoff: dict[str, Any], manifest_id: str | None
) -> dict[str, Any]:
    return {
        "passed": True,
        "manifest_id": manifest_id,
        "plan_id": plan["plan_id"],
        "g2_handoff_id": handoff["handoff_id"],
        "g2_repair_id": handoff["repair_id"],
        "counts": plan["counts"],
        "permissions": plan["permissions"],
        "training_started": False,
        "gpu_inference_started": False,
        "kill_path_contrast_computed": False,
        "p1b_authorized": False,
        "p2_authorized": False,
    }
