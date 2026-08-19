from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.io import (
    file_hash,
    immutable_json_write,
    json_hash,
    read_json_object,
)
from plasticity_placement.pathmem_consolidation.config import build_g0v2_contract
from plasticity_placement.pathmem_consolidation.planner import (
    audit_g1c_plan,
    compile_g1c_plan,
)

G0V2_MANIFEST_VERSION = "pathmem-g0-v2-r-opcd-manifest-v1"
G0V1_MANIFEST_VERSION = "pathmem-g0-manifest-v1"
ARTIFACT_PATHS = ("contract.json", "g1c_plan.json")


def verify_parent_g0_v1(manifest_path: Path) -> dict[str, str]:
    manifest = read_json_object(manifest_path, "parent G0-v1 manifest")
    identity = {key: value for key, value in manifest.items() if key != "manifest_id"}
    manifest_id = manifest.get("manifest_id")
    if manifest.get("manifest_version") != G0V1_MANIFEST_VERSION:
        raise ValueError("parent manifest is not PathMem G0-v1")
    if not isinstance(manifest_id, str) or manifest_id != json_hash(identity):
        raise ValueError("parent G0-v1 manifest identity mismatch")
    gate = manifest.get("g0_gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise ValueError("parent G0-v1 gate did not pass")
    if any(
        manifest.get(field) is not False
        for field in (
            "training_started",
            "gpu_inference_started",
            "g1_authorized",
            "p0_authorized",
        )
    ):
        raise ValueError("parent G0-v1 manifest contains unauthorized execution state")

    parent_root = manifest_path.parent
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("parent G0-v1 artifact registry is missing")
    for relative_path, expected_hash in artifacts.items():
        path = parent_root / relative_path
        if not path.is_file() or file_hash(path) != expected_hash:
            raise ValueError(f"parent G0-v1 artifact hash mismatch: {relative_path}")

    protocol_root = manifest_path.parents[2]
    protocol_sources = manifest.get("protocol_sources")
    if not isinstance(protocol_sources, dict):
        raise ValueError("parent G0-v1 protocol registry is missing")
    for relative_path, expected_hash in protocol_sources.items():
        path = protocol_root / relative_path
        if not path.is_file() or file_hash(path) != expected_hash:
            raise ValueError(f"parent G0-v1 protocol hash mismatch: {relative_path}")
    return {
        "manifest_id": manifest_id,
        "manifest_sha256": file_hash(manifest_path),
    }


def write_g0v2_bundle(output_dir: Path, parent_manifest_path: Path) -> Path:
    parent = verify_parent_g0_v1(parent_manifest_path)
    contract = build_g0v2_contract(
        parent_manifest_id=parent["manifest_id"],
        parent_manifest_sha256=parent["manifest_sha256"],
    )
    plan = compile_g1c_plan(
        parent_manifest_id=parent["manifest_id"],
        parent_manifest_sha256=parent["manifest_sha256"],
    )
    artifact_hashes = {
        "contract.json": immutable_json_write(
            output_dir / "contract.json", contract, "G0-v2 R-OPCD contract"
        ),
        "g1c_plan.json": immutable_json_write(
            output_dir / "g1c_plan.json", plan, "G1-C consolidation plan"
        ),
    }
    plan_audit = audit_g1c_plan(plan)
    checks = (
        ("parent_g0_v1_bound", True),
        ("contract_frozen", True),
        ("plan_deterministic", True),
        ("interface_dev_only", plan_audit["split"] == "interface_dev"),
        ("balanced_action_coverage", set(plan_audit["action_unit_counts"].values()) == {6}),
        ("retention_schedule_complete", plan_audit["counts"]["retention_anchor_units"] == 24),
        ("training_not_started", True),
        ("gpu_inference_not_started", True),
        ("path_contrast_not_computed", True),
    )
    blockers = [name for name, passed in checks if not passed]
    if blockers:
        raise ValueError(f"G0-v2 planning gate failed: {blockers}")
    identity = {
        "manifest_version": G0V2_MANIFEST_VERSION,
        "parent_g0_v1": parent,
        "implementation_sources": _implementation_hashes(),
        "artifacts": artifact_hashes,
        "counts": plan["counts"],
        "g0_v2_gate": {
            "gate": "G0-v2-planning",
            "passed": True,
            "checks": checks,
            "blockers": [],
        },
        "implementation_stage": "cpu_contract_and_plan_only",
        "training_started": False,
        "gpu_inference_started": False,
        "g1c_authorized": False,
        "p0_authorized": False,
        "path_contrast_computed": False,
    }
    manifest = {**identity, "manifest_id": json_hash(identity)}
    manifest_path = output_dir / "manifest.json"
    immutable_json_write(manifest_path, manifest, "G0-v2 manifest")
    report = verify_g0v2_bundle(output_dir, parent_manifest_path)
    if not report["passed"]:
        raise ValueError(f"new G0-v2 bundle failed verification: {report}")
    return manifest_path


def verify_g0v2_bundle(output_dir: Path, parent_manifest_path: Path) -> dict[str, Any]:
    try:
        return _verify_g0v2_bundle(output_dir, parent_manifest_path)
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {"passed": False, "error": str(error)}


def _verify_g0v2_bundle(output_dir: Path, parent_manifest_path: Path) -> dict[str, Any]:
    expected_files = {"manifest.json", *ARTIFACT_PATHS}
    observed_files = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        raise ValueError(
            "unexpected G0-v2 file set: "
            f"missing={sorted(expected_files - observed_files)} "
            f"extra={sorted(observed_files - expected_files)}"
        )
    parent = verify_parent_g0_v1(parent_manifest_path)
    manifest = read_json_object(output_dir / "manifest.json", "G0-v2 manifest")
    identity = {key: value for key, value in manifest.items() if key != "manifest_id"}
    if manifest.get("manifest_version") != G0V2_MANIFEST_VERSION:
        raise ValueError("G0-v2 manifest version mismatch")
    if manifest.get("manifest_id") != json_hash(identity):
        raise ValueError("G0-v2 manifest identity mismatch")
    if manifest.get("parent_g0_v1") != parent:
        raise ValueError("G0-v2 parent binding mismatch")
    if manifest.get("implementation_sources") != _implementation_hashes():
        raise ValueError("G0-v2 implementation source hash mismatch")
    for relative_path, expected_hash in manifest.get("artifacts", {}).items():
        if relative_path not in ARTIFACT_PATHS:
            raise ValueError(f"unknown G0-v2 artifact: {relative_path}")
        if file_hash(output_dir / relative_path) != expected_hash:
            raise ValueError(f"G0-v2 artifact hash mismatch: {relative_path}")

    expected_contract = build_g0v2_contract(
        parent_manifest_id=parent["manifest_id"],
        parent_manifest_sha256=parent["manifest_sha256"],
    )
    expected_plan = compile_g1c_plan(
        parent_manifest_id=parent["manifest_id"],
        parent_manifest_sha256=parent["manifest_sha256"],
    )
    observed_contract = read_json_object(output_dir / "contract.json", "G0-v2 contract")
    observed_plan = read_json_object(output_dir / "g1c_plan.json", "G1-C plan")
    if observed_contract != expected_contract:
        raise ValueError("G0-v2 contract regeneration mismatch")
    if observed_plan != expected_plan:
        raise ValueError("G1-C plan regeneration mismatch")
    audit = audit_g1c_plan(observed_plan)
    gate = manifest.get("g0_v2_gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise ValueError("G0-v2 planning gate did not pass")
    if any(
        manifest.get(field) is not False
        for field in (
            "training_started",
            "gpu_inference_started",
            "g1c_authorized",
            "p0_authorized",
            "path_contrast_computed",
        )
    ):
        raise ValueError("G0-v2 bundle contains unauthorized execution state")
    return {
        "passed": True,
        "manifest_id": manifest["manifest_id"],
        "parent_g0_v1_manifest_id": parent["manifest_id"],
        "plan_id": audit["plan_id"],
        "counts": audit["counts"],
        "training_started": False,
        "gpu_inference_started": False,
        "path_contrast_computed": False,
    }


def _implementation_hashes() -> dict[str, str]:
    repository_root = Path(__file__).resolve().parents[3]
    package_root = Path(__file__).resolve().parent
    return {
        path.relative_to(repository_root).as_posix(): file_hash(path)
        for path in sorted(package_root.glob("*.py"))
    }
