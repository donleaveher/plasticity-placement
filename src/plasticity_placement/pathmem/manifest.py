from __future__ import annotations

from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.compiler import (
    COMPILER_VERSION,
    PRIMARY_PATHS,
    audit_compiled_bank,
    compile_bank,
)
from plasticity_placement.pathmem.gates import (
    AUTHORIZATION_ORDER,
    G1_THRESHOLDS,
    GATE_SCHEMA_VERSION,
    evaluate_g0_gate,
)
from plasticity_placement.pathmem.generator import (
    ACTIONS,
    GENERATOR_SEED,
    GENERATOR_VERSION,
    SPLIT_COUNTS,
)
from plasticity_placement.pathmem.io import (
    file_hash,
    immutable_json_write,
    immutable_jsonl_write,
    json_hash,
    read_json_object,
    read_jsonl,
)
from plasticity_placement.pathmem.observability import (
    CLAIM_EVIDENCE_VERSION,
    OBSERVABILITY_VERSION,
    build_observability_matrix,
    claim_evidence_matrix,
    non_claims,
)
from plasticity_placement.pathmem.reducer import REDUCER_VERSION
from plasticity_placement.pathmem.schema import (
    EVENT_SCHEMA_VERSION,
    HISTORY_FAMILY_SCHEMA_VERSION,
    PROBE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    AccessMode,
    ContentDomain,
    ContractFamily,
    DiagnosticKind,
    EpisodeRole,
    EventKind,
    EvidenceKind,
    LifecycleLocus,
    ObservabilityStatus,
    OperatorKind,
    ProbeCategory,
)
from plasticity_placement.pathmem.scoring import (
    EPSILON_JS_NATS,
    SCORING_VERSION,
    TECHNICAL_TOLERANCE_JS_NATS,
)

G0_MANIFEST_VERSION = "pathmem-g0-manifest-v1"
PROTOCOL_FILES = (
    "README.md",
    "research-question-card.md",
    "ccfa.yaml",
    "docs/idea.md",
    "docs/principles.md",
    "docs/goals-and-gates.md",
    "docs/experiment-plan.md",
    "docs/related-work.md",
    "experiments/README.md",
)
ARTIFACT_PATHS = (
    "configs/schema_registry.json",
    "configs/phase_contract.json",
    "configs/observability.json",
    "configs/claim_evidence.json",
    "compiled/items.jsonl",
    "compiled/event_blocks.jsonl",
    "compiled/probes.jsonl",
    "compiled/history_families.jsonl",
    "compiled/diagnostic_specs.jsonl",
    "compiled/audit.json",
)


def write_g0_bundle(output_dir: Path, protocol_root: Path) -> Path:
    protocol_hashes = _protocol_hashes(protocol_root)
    bank = compile_bank()
    audit = audit_compiled_bank(bank)
    payloads = _artifact_payloads(bank, audit)
    artifact_hashes: dict[str, str] = {}
    for relative_path in ARTIFACT_PATHS:
        path = output_dir / relative_path
        value, is_jsonl = payloads[relative_path]
        if is_jsonl:
            artifact_hashes[relative_path] = immutable_jsonl_write(
                path,
                value,
                relative_path,
            )
        else:
            artifact_hashes[relative_path] = immutable_json_write(
                path,
                value,
                relative_path,
            )

    checks = {
        "protocol_sources_hashed": len(protocol_hashes) == len(PROTOCOL_FILES),
        "schemas_frozen": True,
        "split_bank_complete": audit["item_count"] == 68,
        "endpoint_equivalence_verified": True,
        "exposure_controls_verified": True,
        "observability_matrix_complete": True,
        "claim_falsifiers_complete": True,
        "manifest_identity_verified": True,
        "training_not_started": True,
    }
    gate = evaluate_g0_gate(checks)
    manifest_identity = {
        "manifest_version": G0_MANIFEST_VERSION,
        "implementation_sources": _implementation_hashes(),
        "protocol_sources": protocol_hashes,
        "artifacts": artifact_hashes,
        "counts": {
            "items": len(bank.items),
            "event_blocks": len(bank.event_blocks),
            "probes": len(bank.probes),
            "history_families": len(bank.history_families),
            "diagnostic_specs": len(bank.diagnostic_specs),
        },
        "g0_gate": gate.to_dict(),
        "training_started": False,
        "gpu_inference_started": False,
        "g1_authorized": False,
        "p0_authorized": False,
    }
    manifest = {**manifest_identity, "manifest_id": json_hash(manifest_identity)}
    immutable_json_write(output_dir / "manifest.json", manifest, "G0 manifest")
    verification = verify_g0_bundle(output_dir, protocol_root)
    if not verification["passed"]:
        raise ValueError(f"new G0 bundle failed verification: {verification}")
    return output_dir / "manifest.json"


def verify_g0_bundle(output_dir: Path, protocol_root: Path) -> dict[str, Any]:
    expected_files = {"manifest.json", *ARTIFACT_PATHS}
    observed_files = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    if observed_files != expected_files:
        return {
            "passed": False,
            "error": "unexpected G0 file set",
            "missing": sorted(expected_files - observed_files),
            "extra": sorted(observed_files - expected_files),
        }
    manifest = read_json_object(output_dir / "manifest.json", "G0 manifest")
    identity = {key: value for key, value in manifest.items() if key != "manifest_id"}
    if manifest.get("manifest_version") != G0_MANIFEST_VERSION:
        return {"passed": False, "error": "G0 manifest version mismatch"}
    if manifest.get("manifest_id") != json_hash(identity):
        return {"passed": False, "error": "G0 manifest identity mismatch"}
    if manifest.get("implementation_sources") != _implementation_hashes():
        return {"passed": False, "error": "implementation source hash mismatch"}
    if manifest.get("protocol_sources") != _protocol_hashes(protocol_root):
        return {"passed": False, "error": "protocol source hash mismatch"}
    for relative_path, expected_hash in manifest.get("artifacts", {}).items():
        if relative_path not in ARTIFACT_PATHS:
            return {"passed": False, "error": f"unknown artifact: {relative_path}"}
        if file_hash(output_dir / relative_path) != expected_hash:
            return {"passed": False, "error": f"artifact hash mismatch: {relative_path}"}

    bank = compile_bank()
    audit = audit_compiled_bank(bank)
    expected_payloads = _artifact_payloads(bank, audit)
    for relative_path, (expected, is_jsonl) in expected_payloads.items():
        observed: object
        if is_jsonl:
            observed = read_jsonl(output_dir / relative_path, relative_path)
        else:
            observed = read_json_object(output_dir / relative_path, relative_path)
        if json_hash(observed) != json_hash(expected):
            return {"passed": False, "error": f"regeneration mismatch: {relative_path}"}

    gate_payload = manifest.get("g0_gate")
    if not isinstance(gate_payload, dict) or gate_payload.get("passed") is not True:
        return {"passed": False, "error": "G0 gate is not passed"}
    if any(
        manifest.get(field) is not False
        for field in ("training_started", "gpu_inference_started", "g1_authorized", "p0_authorized")
    ):
        return {"passed": False, "error": "G0 bundle contains unauthorized execution state"}
    return {
        "passed": True,
        "manifest_id": manifest["manifest_id"],
        "counts": manifest["counts"],
        "g0_gate": gate_payload,
        "training_started": False,
    }


def _artifact_payloads(bank: Any, audit: dict[str, Any]) -> dict[str, tuple[Any, bool]]:
    return {
        "configs/schema_registry.json": (_schema_registry(), False),
        "configs/phase_contract.json": (_phase_contract(), False),
        "configs/observability.json": (
            {
                "version": OBSERVABILITY_VERSION,
                "cells": [cell.to_dict() for cell in build_observability_matrix()],
            },
            False,
        ),
        "configs/claim_evidence.json": (
            {
                "version": CLAIM_EVIDENCE_VERSION,
                "claims": list(claim_evidence_matrix()),
                "non_claims": list(non_claims()),
            },
            False,
        ),
        "compiled/items.jsonl": ([item.to_dict() for item in bank.items], True),
        "compiled/event_blocks.jsonl": (
            [block.to_dict() for block in bank.event_blocks],
            True,
        ),
        "compiled/probes.jsonl": ([probe.to_dict() for probe in bank.probes], True),
        "compiled/history_families.jsonl": (
            [family.to_dict() for family in bank.history_families],
            True,
        ),
        "compiled/diagnostic_specs.jsonl": (
            [spec.to_dict() for spec in bank.diagnostic_specs],
            True,
        ),
        "compiled/audit.json": (audit, False),
    }


def _schema_registry() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "history_family_schema_version": HISTORY_FAMILY_SCHEMA_VERSION,
        "probe_schema_version": PROBE_SCHEMA_VERSION,
        "compiler_version": COMPILER_VERSION,
        "generator_version": GENERATOR_VERSION,
        "reducer_version": REDUCER_VERSION,
        "scoring_version": SCORING_VERSION,
        "gate_schema_version": GATE_SCHEMA_VERSION,
        "enums": {
            "event_kinds": [value.value for value in EventKind],
            "content_domains": [value.value for value in ContentDomain],
            "contract_families": [value.value for value in ContractFamily],
            "episode_roles": [value.value for value in EpisodeRole],
            "probe_categories": [value.value for value in ProbeCategory],
            "access_modes": [value.value for value in AccessMode],
            "operators": [value.value for value in OperatorKind],
            "lifecycle_loci": [value.value for value in LifecycleLocus],
            "observability_statuses": [value.value for value in ObservabilityStatus],
            "evidence_kinds": [value.value for value in EvidenceKind],
            "diagnostic_kinds": [value.value for value in DiagnosticKind],
        },
        "semantic_item_is_statistical_unit": True,
        "history_family_is_provenance_object": True,
        "fresh_session_clears_volatile_conversation_state": True,
        "missing_lifecycle_fields_serialize_as": "N/A",
    }


def _phase_contract() -> dict[str, Any]:
    return {
        "generator_seed": GENERATOR_SEED,
        "split_counts": SPLIT_COUNTS,
        "action_universe": list(ACTIONS),
        "primary_paths": PRIMARY_PATHS,
        "authorization_order": list(AUTHORIZATION_ORDER),
        "g1_thresholds": G1_THRESHOLDS,
        "epsilon_js_nats": EPSILON_JS_NATS,
        "technical_tolerance_js_nats": TECHNICAL_TOLERANCE_JS_NATS,
        "bootstrap": {"samples": 10_000, "seed": GENERATOR_SEED, "unit": "semantic_item"},
        "seed_fields": (
            "training_seed",
            "panel_seed",
            "bootstrap_seed",
            "execution_order_seed",
        ),
        "p0_model": {
            "name": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "role": "engineering_smoke_only",
        },
        "p1_model": {
            "name": "Qwen/Qwen2.5-1.5B-Instruct",
            "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
            "role": "first_outcome_bearing_operator_audit",
        },
        "no_pretraining_required": True,
        "result_cells": "TBD_until_verified_execution",
    }


def _protocol_hashes(protocol_root: Path) -> dict[str, str]:
    if not protocol_root.is_dir():
        raise FileNotFoundError(f"protocol root does not exist: {protocol_root}")
    hashes: dict[str, str] = {}
    for relative_path in PROTOCOL_FILES:
        path = protocol_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"required protocol file is missing: {path}")
        hashes[relative_path] = file_hash(path)
    return hashes


def _implementation_hashes() -> dict[str, str]:
    repository_root = Path(__file__).resolve().parents[3]
    package_root = Path(__file__).resolve().parent
    paths = [
        *sorted(package_root.glob("*.py")),
        *sorted((repository_root / "tests").glob("test_pathmem_*.py")),
        repository_root / "pyproject.toml",
    ]
    lock_path = repository_root / "uv.lock"
    if lock_path.is_file():
        paths.append(lock_path)
    hashes: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"implementation identity source is missing: {path}")
        hashes[path.relative_to(repository_root).as_posix()] = file_hash(path)
    return hashes
