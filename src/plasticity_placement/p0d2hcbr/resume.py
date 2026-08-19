from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.runtime import current_code_hash
from plasticity_placement.p0d2hcbr.config import unit_id
from plasticity_placement.p0d2hcbr.deviation import ADOPTED_NAME, PLACEMENT_SCOPE
from plasticity_placement.p0d2hcbr.preflight import load_config
from plasticity_placement.p0d2hcbr.runtime import audit_all_training_units
from plasticity_placement.p0d2hrr.io import (
    file_hash,
    immutable_json_write,
    json_hash,
    read_json_object,
)

SCHEMA_VERSION = "p0d2hcbr-resumable-evaluation-authorization-v1"
SCOPE = "unfinished_cbr_units_sharded_off_on_evaluation"
ADOPTED_RESUME_NAME = "resumable_evaluation_authorization.json"
DEFAULT_SHARD_SIZE = 64
MIN_SHARD_SIZE = 8
MAX_SHARD_SIZE = 256


def resumable_evaluation_template(
    output_dir: Path,
    analysis_root: Path,
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    analysis_root = analysis_root.resolve()
    _validate_shard_size(shard_size)
    _require_independent(output_dir, analysis_root)
    manifest, spec, identity = load_config(output_dir)
    if manifest.state not in {"trained", "evaluated"}:
        raise PermissionError(
            "resumable evaluation requires a fully trained, unaggregated CBR matrix: "
            f"{manifest.state}"
        )
    budget_audit = resumable_budget_audit(output_dir)
    eligible: list[str] = []
    interrupted_claims: dict[str, dict[str, Any]] = {}
    for curriculum in spec.curricula:
        for placement in spec.placements:
            for seed in spec.training.seeds:
                key = unit_id(curriculum, placement, seed)
                claim = manifest.payload.get("evaluation_claims", {}).get(key)
                if isinstance(claim, dict) and claim.get("state") == "complete":
                    continue
                if isinstance(claim, dict):
                    interrupted_claims[key] = _validate_zero_artifact_claim(
                        output_dir, key, claim
                    )
                destination = analysis_root / key
                if destination.exists():
                    raise FileExistsError(
                        f"resumable evaluation destination already exists: {destination}"
                    )
                eligible.append(key)
    if not eligible:
        raise ValueError("no unfinished CBR evaluation units remain")
    adopted_deviation = manifest.payload.get("budget_deviation")
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "pending",
        "scope": SCOPE,
        "cbr_run_id": manifest.payload["run_id"],
        "preregistration_sha256": identity["preregistration_sha256"],
        "experiment_code_sha256": identity["code_sha256"],
        "resumable_code_sha256": current_code_hash(),
        "budget_audit_sha256": json_hash(budget_audit),
        "budget_deviation_authorization_sha256": (
            adopted_deviation.get("authorization_sha256")
            if isinstance(adopted_deviation, dict)
            else None
        ),
        "analysis_root": str(analysis_root),
        "shard_size": shard_size,
        "eligible_unit_ids": eligible,
        "interrupted_zero_artifact_claims": interrupted_claims,
        "pair_order": "adapter_off_then_adapter_on_per_prompt",
        "runtime_scope": "one_loaded_model_per_process_segment",
        "allows_cross_runtime_unit_resume": True,
        "allowed_locked_evaluations": len(eligible),
        "allows_additional_training": False,
        "allows_hyperparameter_search": False,
        "allows_checkpoint_selection": False,
        "allows_mappings_per_adapter_scan": False,
        "approved_by": "TBD",
        "approved_at": "TBD",
        "notes": (
            "Authorizes immutable prompt-shard checkpoints for unfinished CBR units. "
            "Every prompt retains adjacent adapter-OFF/ON scoring and every shard "
            "requires an OFF-state sentinel before publication."
        ),
    }


def adopt_resumable_evaluation(
    output_dir: Path,
    authorization_path: Path,
    analysis_root: Path,
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> Path:
    output_dir = output_dir.resolve()
    authorization_path = authorization_path.resolve()
    analysis_root = analysis_root.resolve()
    _require_external(output_dir, authorization_path)
    manifest, _, _ = load_config(output_dir)
    if isinstance(manifest.payload.get("resumable_evaluation"), dict):
        validate_resumable_authorization(output_dir)
        return output_dir / ADOPTED_RESUME_NAME
    expected = resumable_evaluation_template(
        output_dir, analysis_root, shard_size=shard_size
    )
    payload = read_json_object(authorization_path, "CBR resumable evaluation authorization")
    required = {
        **expected,
        "decision": "approved",
        "approved_by": payload.get("approved_by"),
        "approved_at": payload.get("approved_at"),
        "notes": payload.get("notes", expected["notes"]),
    }
    differences = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in required.items()
        if payload.get(key) != value
    }
    if differences:
        raise PermissionError(f"resumable evaluation authorization differs: {differences}")
    _validate_approver(payload)
    adopted_path = output_dir / ADOPTED_RESUME_NAME
    digest = immutable_json_write(
        adopted_path, payload, "CBR resumable evaluation authorization"
    )
    archived_claims = _archive_interrupted_claims(
        output_dir,
        manifest.payload,
        payload["interrupted_zero_artifact_claims"],
    )
    manifest.payload["resumable_evaluation"] = {
        "state": "authorized",
        "authorization_sha256": digest,
        "resumable_code_sha256": payload["resumable_code_sha256"],
        "analysis_root": payload["analysis_root"],
        "shard_size": payload["shard_size"],
        "eligible_unit_ids": payload["eligible_unit_ids"],
        "archived_interrupted_claims": archived_claims,
        "runtime_scope": payload["runtime_scope"],
        "approved_by": payload["approved_by"],
        "approved_at": payload["approved_at"],
        "additional_training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    try:
        manifest.save()
    except Exception:
        _restore_archived_claims(archived_claims)
        raise
    return adopted_path


def validate_resumable_authorization(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    manifest, _, identity = load_config(output_dir)
    recorded = manifest.payload.get("resumable_evaluation")
    if not isinstance(recorded, dict):
        raise PermissionError("resumable CBR evaluation has not been authorized")
    adopted_path = output_dir / ADOPTED_RESUME_NAME
    if not adopted_path.is_file() or file_hash(adopted_path) != recorded.get(
        "authorization_sha256"
    ):
        raise PermissionError("resumable evaluation authorization changed")
    payload = read_json_object(adopted_path, "CBR resumable evaluation authorization")
    budget_audit = resumable_budget_audit(output_dir)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("decision") != "approved"
        or payload.get("scope") != SCOPE
        or payload.get("cbr_run_id") != manifest.payload["run_id"]
        or payload.get("preregistration_sha256") != identity["preregistration_sha256"]
        or payload.get("resumable_code_sha256") != current_code_hash()
        or payload.get("budget_audit_sha256") != json_hash(budget_audit)
        or payload.get("analysis_root") != recorded.get("analysis_root")
        or payload.get("shard_size") != recorded.get("shard_size")
        or payload.get("eligible_unit_ids") != recorded.get("eligible_unit_ids")
        or payload.get("allows_cross_runtime_unit_resume") is not True
        or payload.get("allows_additional_training") is not False
        or payload.get("allows_checkpoint_selection") is not False
        or payload.get("allows_mappings_per_adapter_scan") is not False
        or recorded.get("resumable_code_sha256") != current_code_hash()
        or recorded.get("additional_training_authorized") is not False
        or recorded.get("mappings_per_adapter_authorized") is not False
    ):
        raise PermissionError("resumable evaluation authorization provenance differs")
    _validate_shard_size(int(payload["shard_size"]))
    _validate_approver(payload)
    for key, archived in recorded.get("archived_interrupted_claims", {}).items():
        path = Path(archived["archived_claim_path"])
        if not path.is_file() or file_hash(path) != archived.get("claim_sha256"):
            raise PermissionError(f"archived interrupted claim changed: {key}")
    return {**payload, "budget_audit": budget_audit}


def resumable_budget_audit(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    audit = audit_all_training_units(output_dir)
    if audit["all_checks_passed"] is True:
        return audit
    manifest, _, _ = load_config(output_dir)
    recorded = manifest.payload.get("budget_deviation")
    adopted_path = output_dir / ADOPTED_NAME
    if (
        not isinstance(recorded, dict)
        or not adopted_path.is_file()
        or file_hash(adopted_path) != recorded.get("authorization_sha256")
        or recorded.get("budget_audit_sha256") != json_hash(audit)
        or recorded.get("placement_comparison_scope") != PLACEMENT_SCOPE
        or recorded.get("additional_training_authorized") is not False
        or recorded.get("mappings_per_adapter_authorized") is not False
    ):
        raise PermissionError("budget deviation is not valid for resumable evaluation")
    adopted = read_json_object(adopted_path, "CBR budget-deviation authorization")
    if adopted.get("decision") != "approved":
        raise PermissionError("budget deviation approval is not approved")
    return {
        **audit,
        "authorized_deviation": {
            "authorization_sha256": recorded["authorization_sha256"],
            "recovery_code_sha256": recorded["recovery_code_sha256"],
            "budget_audit_sha256": recorded["budget_audit_sha256"],
            "placement_comparison_scope": PLACEMENT_SCOPE,
        },
        "comparison_scope": PLACEMENT_SCOPE,
    }


def _validate_zero_artifact_claim(
    output_dir: Path, key: str, claim: dict[str, Any]
) -> dict[str, Any]:
    if claim.get("state") != "claimed" or claim.get("summary_sha256") is not None:
        raise ValueError(f"unfinished CBR claim is not zero-artifact resumable: {key}")
    analysis_output = Path(str(claim.get("analysis_output", ""))).resolve()
    artifacts = (
        sorted(str(path) for path in analysis_output.rglob("*") if path.is_file())
        if analysis_output.exists()
        else []
    )
    if artifacts:
        raise ValueError(f"interrupted CBR claim contains artifacts: {key}: {artifacts}")
    claim_path = output_dir / "claims" / f"evaluation-{key}.json"
    if not claim_path.is_file() or claim_path.is_symlink():
        raise ValueError(f"immutable CBR evaluation claim is missing or unsafe: {key}")
    if file_hash(claim_path) != claim.get("claim_sha256"):
        raise ValueError(f"immutable CBR evaluation claim changed: {key}")
    document = read_json_object(claim_path, f"CBR evaluation claim {key}")
    if (
        document.get("unit_id") != key
        or document.get("run_id") != claim.get("run_id")
        or document.get("analysis_output") != str(analysis_output)
    ):
        raise ValueError(f"CBR evaluation claim identity differs: {key}")
    return {
        "run_id": claim["run_id"],
        "claim_sha256": claim["claim_sha256"],
        "analysis_output": str(analysis_output),
        "artifact_count": 0,
    }


def _archive_interrupted_claims(
    output_dir: Path,
    manifest_payload: dict[str, Any],
    claims: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    archived: dict[str, dict[str, Any]] = {}
    root = output_dir / "claims" / "resumable-evaluation-recovery"
    moves: list[tuple[Path, Path, str, dict[str, Any]]] = []
    for key, expected in claims.items():
        active = manifest_payload.get("evaluation_claims", {}).get(key)
        if not isinstance(active, dict) or active.get("claim_sha256") != expected.get(
            "claim_sha256"
        ):
            raise RuntimeError(f"CBR claim changed before resumable adoption: {key}")
        source = output_dir / "claims" / f"evaluation-{key}.json"
        destination = root / f"interrupted-{key}-{active['run_id']}.json"
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"resumable source claim is missing or unsafe: {key}")
        if file_hash(source) != expected.get("claim_sha256"):
            raise ValueError(f"resumable source claim changed before archive: {key}")
        if destination.exists():
            raise FileExistsError(f"resumable claim archive already exists: {destination}")
        moves.append((source, destination, key, expected))
    moved: list[tuple[Path, Path]] = []
    try:
        for source, destination, key, expected in moves:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            moved.append((source, destination))
            archived[key] = {
                **expected,
                "original_claim_path": str(source),
                "archived_claim_path": str(destination),
            }
            del manifest_payload["evaluation_claims"][key]
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                destination.replace(source)
        raise
    return archived


def _restore_archived_claims(archived: dict[str, dict[str, Any]]) -> None:
    for value in reversed(list(archived.values())):
        source = Path(value["original_claim_path"])
        destination = Path(value["archived_claim_path"])
        if destination.exists() and not source.exists():
            destination.replace(source)


def _validate_approver(payload: dict[str, Any]) -> None:
    if payload.get("approved_by") in {None, "", "TBD"}:
        raise ValueError("resumable evaluation authorization requires an approver")
    approved_at = datetime.fromisoformat(
        str(payload.get("approved_at", "TBD")).replace("Z", "+00:00")
    )
    if approved_at.tzinfo is None:
        raise ValueError("resumable evaluation approval timestamp must be timezone-aware")


def _validate_shard_size(shard_size: int) -> None:
    if not MIN_SHARD_SIZE <= shard_size <= MAX_SHARD_SIZE:
        raise ValueError(
            f"shard_size must be between {MIN_SHARD_SIZE} and {MAX_SHARD_SIZE}"
        )


def _require_external(output_dir: Path, path: Path) -> None:
    if path == output_dir or path.is_relative_to(output_dir):
        raise ValueError("resumable evaluation authorization must be outside CBR output")


def _require_independent(output_dir: Path, analysis_root: Path) -> None:
    if (
        output_dir == analysis_root
        or output_dir.is_relative_to(analysis_root)
        or analysis_root.is_relative_to(output_dir)
    ):
        raise ValueError("resumable analysis root must be independent from CBR training output")
