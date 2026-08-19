from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.io import file_hash, immutable_json_write, json_hash
from plasticity_placement.pathmem_consolidation.manifest import verify_g0v2_bundle
from plasticity_placement.pathmem_consolidation_exec.config import (
    AUTHORIZATION_SCHEMA_VERSION,
    G1C_EXECUTION_RECIPE,
    RopcdExecutionRecipe,
)

CRITICAL_PACKAGES = (
    "accelerate",
    "bitsandbytes",
    "peft",
    "safetensors",
    "torch",
    "transformers",
)


def verify_source_bundle(bundle_root: Path, parent_manifest: Path) -> dict[str, Any]:
    report = verify_g0v2_bundle(bundle_root, parent_manifest)
    if report.get("passed") is not True:
        raise RuntimeError(f"G0-v2 source bundle verification failed: {report}")
    manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((bundle_root / "g1c_plan.json").read_text(encoding="utf-8"))
    if manifest.get("manifest_id") != report["manifest_id"]:
        raise ValueError("source manifest identity differs from verification report")
    if plan.get("plan_id") != report["plan_id"]:
        raise ValueError("source plan identity differs from verification report")
    if plan.get("authorization", {}).get("g1c_execution_authorized") is not False:
        raise ValueError("planning source unexpectedly authorizes execution")
    return {
        "bundle_root": str(bundle_root.resolve()),
        "manifest_id": report["manifest_id"],
        "manifest_sha256": file_hash(bundle_root / "manifest.json"),
        "plan_id": report["plan_id"],
        "plan_sha256": file_hash(bundle_root / "g1c_plan.json"),
        "parent_g0_v1_manifest_id": report["parent_g0_v1_manifest_id"],
        "counts": report["counts"],
        "plan": plan,
    }


def implementation_identity() -> tuple[dict[str, Any], str]:
    repository_root = Path(__file__).resolve().parents[3]
    source_roots = (
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
        repository_root / "src/plasticity_placement/pathmem/scoring.py",
        repository_root / "src/plasticity_placement/p0d2hfc/scoring.py",
        repository_root / "src/plasticity_placement/training/model_utils.py",
        repository_root / "src/plasticity_placement/pathmem_exec/artifacts.py",
        repository_root / "src/plasticity_placement/pathmem_exec/environment.py",
        repository_root / "pyproject.toml",
        repository_root / "uv.lock",
    )
    sources.update(
        {
            path.relative_to(repository_root).as_posix(): file_hash(path)
            for path in dependencies
        }
    )
    revision = _git_output(repository_root, ("rev-parse", "HEAD"))
    payload = {"git_revision": revision, "source_files": sources}
    return payload, json_hash(payload)


def build_authorization_template(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    output_root: Path,
    recipe: RopcdExecutionRecipe = G1C_EXECUTION_RECIPE,
) -> dict[str, Any]:
    source = verify_source_bundle(bundle_root, parent_manifest)
    implementation, implementation_sha256 = implementation_identity()
    return {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "decision": "pending_human_review",
        "scope": "g1c_r_opcd_cuda_qualification_only",
        "source": {
            key: source[key]
            for key in (
                "manifest_id",
                "manifest_sha256",
                "plan_id",
                "plan_sha256",
                "parent_g0_v1_manifest_id",
            )
        },
        "recipe": recipe.to_dict(),
        "recipe_sha256": json_hash(recipe.to_dict()),
        "implementation": implementation,
        "implementation_sha256": implementation_sha256,
        "output_root": str(output_root.resolve()),
        "permissions": {
            "g1c_training_authorized": True,
            "g1c_gpu_inference_authorized": True,
            "p0_authorized": False,
            "path_contrast_authorized": False,
            "kill_or_reserve_access_authorized": False,
            "rl_controller_authorized": False,
            "automatic_hyperparameter_search_authorized": False,
        },
    }


def verify_external_approval(
    approval_path: Path,
    *,
    expected_template: dict[str, Any],
) -> dict[str, Any]:
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    expected_keys = set(expected_template) | {"approved_by", "approved_at"}
    if set(approval) != expected_keys:
        raise ValueError(
            "authorization field set changed: "
            f"missing={sorted(expected_keys - set(approval))} "
            f"extra={sorted(set(approval) - expected_keys)}"
        )
    differences = {
        key: {"expected": value, "observed": approval.get(key)}
        for key, value in expected_template.items()
        if key != "decision" and approval.get(key) != value
    }
    if differences:
        raise ValueError(f"authorization template binding changed: {differences}")
    if approval.get("decision") != "approved":
        raise PermissionError("G1-C CUDA execution requires decision=approved")
    approved_by = approval.get("approved_by")
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise ValueError("authorization requires a human responsible-party identifier")
    approved_at = approval.get("approved_at")
    if not isinstance(approved_at, str):
        raise ValueError("authorization requires an ISO-8601 approval time")
    try:
        parsed = datetime.fromisoformat(approved_at)
    except ValueError as error:
        raise ValueError("authorization approval time is not ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError("authorization approval time must include a timezone")
    identity = {**approval, "approved_by": approved_by.strip()}
    return {**identity, "authorization_id": json_hash(identity)}


def adopt_authorization(
    output_root: Path,
    approval: dict[str, Any],
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "authorization.json"
    immutable_json_write(path, approval, "G1-C execution authorization")
    return path


def verify_adopted_authorization(
    output_root: Path,
    *,
    expected_template: dict[str, Any],
) -> dict[str, Any]:
    path = output_root / "authorization.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    authorization_id = payload.pop("authorization_id", None)
    expected = verify_external_approval_object(payload, expected_template=expected_template)
    if authorization_id != expected["authorization_id"]:
        raise ValueError("adopted authorization identity mismatch")
    return expected


def verify_external_approval_object(
    approval: dict[str, Any], *, expected_template: dict[str, Any]
) -> dict[str, Any]:
    temporary_payload = dict(approval)
    expected_keys = set(expected_template) | {"approved_by", "approved_at"}
    if set(temporary_payload) != expected_keys:
        raise ValueError("adopted authorization field set changed")
    for key, value in expected_template.items():
        if key == "decision":
            continue
        if temporary_payload.get(key) != value:
            raise ValueError(f"adopted authorization binding changed: {key}")
    if temporary_payload.get("decision") != "approved":
        raise PermissionError("adopted authorization is not approved")
    approved_by = temporary_payload.get("approved_by")
    approved_at = temporary_payload.get("approved_at")
    if not isinstance(approved_by, str) or not approved_by.strip():
        raise ValueError("adopted authorization approver is missing")
    if not isinstance(approved_at, str):
        raise ValueError("adopted authorization time is missing")
    parsed = datetime.fromisoformat(approved_at)
    if parsed.tzinfo is None:
        raise ValueError("adopted authorization time must include a timezone")
    identity = {**temporary_payload, "approved_by": approved_by.strip()}
    return {**identity, "authorization_id": json_hash(identity)}


def execution_environment(recipe: RopcdExecutionRecipe = G1C_EXECUTION_RECIPE) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoConfig
    except ImportError as error:
        raise RuntimeError("GPU dependencies are missing; install train/colab extras") from error
    if not torch.cuda.is_available():
        raise RuntimeError("formal G1-C execution requires an NVIDIA CUDA runtime")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("formal G1-C execution requires native CUDA BF16 support")
    model_config = AutoConfig.from_pretrained(
        recipe.model.name,
        revision=recipe.model.revision,
    )
    resolved_revision = str(
        getattr(model_config, "_commit_hash", "") or recipe.model.revision
    )
    if resolved_revision != recipe.model.revision:
        raise RuntimeError("resolved model revision differs from the frozen revision")
    from plasticity_placement.pathmem_exec.environment import tokenizer_identity
    from plasticity_placement.training.model_utils import load_tokenizer

    tokenizer = load_tokenizer(recipe.model.name, recipe.model.revision)
    tokenizer_payload, tokenizer_sha256 = tokenizer_identity(tokenizer)
    packages = {
        package: importlib.metadata.version(package) for package in CRITICAL_PACKAGES
    }
    implementation, implementation_sha256 = implementation_identity()
    identity = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda": {
            "cuda_version": str(torch.version.cuda),
            "cudnn_version": int(torch.backends.cudnn.version() or 0),
            "gpu_name": str(torch.cuda.get_device_name(0)),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "bf16_supported": True,
        },
        "model": {
            **recipe.model.to_dict(),
            "resolved_revision": resolved_revision,
            "tokenizer": tokenizer_payload,
            "tokenizer_sha256": tokenizer_sha256,
        },
        "resolved_precision": "nf4-bfloat16",
        "implementation": implementation,
        "implementation_sha256": implementation_sha256,
    }
    return {**identity, "environment_fingerprint": json_hash(identity)}


def _git_output(root: Path, arguments: tuple[str, ...]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "NO_GIT_REVISION"
