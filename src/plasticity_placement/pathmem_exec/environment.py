from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.io import file_hash, json_hash
from plasticity_placement.pathmem.manifest import verify_g0_bundle
from plasticity_placement.pathmem_exec.config import PhaseConfig
from plasticity_placement.training.model_utils import load_tokenizer

CRITICAL_PACKAGES = (
    "accelerate",
    "bitsandbytes",
    "peft",
    "safetensors",
    "torch",
    "transformers",
)


def verify_g0_or_raise(g0_dir: Path, protocol_root: Path) -> dict[str, Any]:
    report = verify_g0_bundle(g0_dir, protocol_root)
    if report.get("passed") is not True:
        raise RuntimeError(f"G0 verification failed: {report}")
    return report


def require_cuda_bf16() -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is missing; install the train dependencies") from error
    if not torch.cuda.is_available():
        raise RuntimeError("formal G1/P0 execution requires an NVIDIA CUDA runtime")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("formal G1/P0 execution requires native CUDA BF16 support")
    return {
        "cuda_available": True,
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "bf16_supported": True,
    }


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in CRITICAL_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required package is missing: {package}") from error
    return versions


def library_lock_identity(cuda: dict[str, Any]) -> tuple[dict[str, Any], str]:
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "cuda": cuda,
    }
    return payload, json_hash(payload)


def implementation_identity() -> tuple[dict[str, Any], str]:
    repository_root = Path(__file__).resolve().parents[3]
    source_roots = (
        repository_root / "src" / "plasticity_placement" / "pathmem",
        repository_root / "src" / "plasticity_placement" / "pathmem_exec",
    )
    sources = {
        path.relative_to(repository_root).as_posix(): file_hash(path)
        for root in source_roots
        for path in sorted(root.glob("*.py"))
    }
    dependency_sources = (
        repository_root / "src/plasticity_placement/training/config.py",
        repository_root / "src/plasticity_placement/training/data.py",
        repository_root / "src/plasticity_placement/training/lora.py",
        repository_root / "src/plasticity_placement/training/model_utils.py",
        repository_root / "src/plasticity_placement/p0d2hfc/scoring.py",
    )
    sources.update(
        {
            path.relative_to(repository_root).as_posix(): file_hash(path)
            for path in dependency_sources
        }
    )
    revision = _git_output(repository_root, ("rev-parse", "HEAD"), required=False)
    payload = {
        "git_revision": revision or "NO_GIT_REVISION",
        "source_files": sources,
    }
    return payload, json_hash(payload)


def tokenizer_identity(tokenizer: Any) -> tuple[dict[str, Any], str]:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_json = backend.to_str() if backend is not None else None
    if not isinstance(backend_json, str) or not backend_json:
        vocabulary = getattr(tokenizer, "get_vocab", lambda: {})()
        backend_json = str(sorted(vocabulary.items()))
    payload = {
        "backend_sha256": json_hash(backend_json),
        "class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "special_tokens_map": {
            str(key): str(value)
            for key, value in dict(getattr(tokenizer, "special_tokens_map", {})).items()
        },
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": (
            int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
        ),
    }
    return payload, json_hash(payload)


def resolve_and_verify_model(config: PhaseConfig) -> tuple[Any, dict[str, Any]]:
    try:
        from transformers import AutoConfig
    except ImportError as error:
        raise RuntimeError("Transformers is missing; install the train dependencies") from error
    model_config = AutoConfig.from_pretrained(
        config.model.name,
        revision=config.model.revision,
    )
    resolved = str(getattr(model_config, "_commit_hash", "") or config.model.revision)
    if resolved != config.model.revision:
        raise RuntimeError(
            f"resolved model revision changed: {resolved} != {config.model.revision}"
        )
    tokenizer = load_tokenizer(config.model.name, config.model.revision)
    tokenizer_payload, tokenizer_sha256 = tokenizer_identity(tokenizer)
    return tokenizer, {
        "model_name": config.model.name,
        "requested_revision": config.model.revision,
        "resolved_revision": resolved,
        "tokenizer": tokenizer_payload,
        "tokenizer_sha256": tokenizer_sha256,
    }


def execution_environment(config: PhaseConfig) -> dict[str, Any]:
    cuda = require_cuda_bf16()
    library_payload, library_sha256 = library_lock_identity(cuda)
    code_payload, code_sha256 = implementation_identity()
    _, model_payload = resolve_and_verify_model(config)
    payload = {
        "phase_config": config.to_dict(),
        "model": model_payload,
        "library_lock": library_payload,
        "library_lock_sha256": library_sha256,
        "implementation": code_payload,
        "code_and_patch_sha256": code_sha256,
        "resolved_precision": "nf4-bfloat16",
        "python_executable": sys.executable,
    }
    return {**payload, "environment_fingerprint": json_hash(payload)}


def _git_output(root: Path, arguments: tuple[str, ...], *, required: bool) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return completed.stdout.strip()
    if required:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return None
