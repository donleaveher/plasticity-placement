from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from plasticity_placement.p0c.runtime import current_code_hash
from plasticity_placement.p0d2hcpr.authorization import adopt
from plasticity_placement.p0d2hcpr.preflight import load_config
from plasticity_placement.p0d2hrr.io import file_hash, read_json_object
from plasticity_placement.training.lora import train_lora


def authorize_experiment(output_dir: Path, authorization_path: Path) -> Path:
    output_dir = output_dir.resolve()
    manifest, _, identity = load_config(output_dir)
    if manifest.state != "planned":
        raise ValueError(f"CPR authorization requires state=planned, found {manifest.state}")
    payload, digest = adopt(
        output_dir, authorization_path.resolve(), identity["preregistration_sha256"]
    )
    manifest.transition(
        "authorized",
        authorization={
            "sha256": digest,
            "approved_by": payload["approved_by"],
            "approved_at": payload["approved_at"],
            "scope": payload["scope"],
        },
    )
    return manifest.path


def train_experiment(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    manifest, spec, identity = load_config(output_dir)
    if manifest.state != "authorized":
        raise PermissionError(f"CPR training requires state=authorized, found {manifest.state}")
    _validate_execution(output_dir, manifest.payload, identity)
    adapter_dir = output_dir / "adapter"
    if adapter_dir.exists():
        raise FileExistsError("CPR adapter already exists; use a new attempt")
    config = spec.training.to_lora_config(
        data_path=output_dir / "preflight" / "composition_train.jsonl",
        output_dir=adapter_dir,
    )
    manifest.transition("training")
    try:
        summary = train_lora(config)
        if (
            summary.optimizer_steps != spec.training.max_steps
            or summary.training_data_sha256 != identity["data_hashes"]["train"]
            or summary.model_revision != spec.model_revision
        ):
            raise ValueError("CPR training summary differs from preregistration")
        metadata = adapter_dir / "training_metadata.json"
        manifest.transition(
            "trained",
            training={
                "summary": asdict(summary),
                "training_metadata_sha256": file_hash(metadata),
                "completed_training_runs": 1,
                "checkpoint_selection": spec.training.checkpoint_selection,
                "objective": spec.training.objective,
                "resumed_from_rr1": False,
            },
        )
    except BaseException as error:
        manifest.fail("training", error)
        raise
    return manifest.path


def _validate_execution(
    output_dir: Path, payload: dict[str, object], identity: dict[str, object]
) -> None:
    if current_code_hash() != identity["code_sha256"]:
        raise ValueError("code changed after CPR preregistration")
    authorization = read_json_object(output_dir / "authorization.json", "CPR authorization")
    recorded = payload.get("authorization")
    if not isinstance(recorded, dict) or recorded.get("sha256") != file_hash(
        output_dir / "authorization.json"
    ):
        raise PermissionError("CPR authorization changed")
    if authorization.get("preregistration_sha256") != identity["preregistration_sha256"]:
        raise PermissionError("CPR authorization no longer binds the preregistration")
