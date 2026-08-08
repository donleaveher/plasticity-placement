from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.runtime import current_code_hash
from plasticity_placement.p0d2hcbr.authorization import adopt
from plasticity_placement.p0d2hcbr.config import CURRICULA, PLACEMENTS, unit_id
from plasticity_placement.p0d2hcbr.preflight import load_config
from plasticity_placement.p0d2hrr.io import file_hash, read_json_object
from plasticity_placement.training.lora import train_lora


def authorize_experiment(output_dir: Path, authorization_path: Path) -> Path:
    output_dir = output_dir.resolve()
    manifest, _, identity = load_config(output_dir)
    if manifest.state != "planned":
        raise ValueError(f"CBR authorization requires state=planned, found {manifest.state}")
    payload, digest = adopt(
        output_dir, authorization_path.resolve(), identity["preregistration_sha256"]
    )
    manifest.payload["authorization"] = {
        "sha256": digest,
        "approved_by": payload["approved_by"],
        "approved_at": payload["approved_at"],
        "scope": payload["scope"],
    }
    manifest.payload["training_authorized"] = True
    manifest.set_state("authorized")
    return manifest.path


def train_unit(output_dir: Path, curriculum: str, placement: str, seed: int) -> Path:
    output_dir = output_dir.resolve()
    manifest, spec, identity = load_config(output_dir)
    if manifest.state not in {"authorized", "training"}:
        raise PermissionError(
            f"CBR training requires authorized/training state, found {manifest.state}"
        )
    _validate_execution(output_dir, manifest.payload, identity)
    key = unit_id(curriculum, placement, seed)
    entry = manifest.payload["training_units"][key]
    adapter_dir = output_dir / "adapters" / key
    if entry["state"] == "trained":
        _validate_trained_unit(output_dir, key, entry, spec, identity)
        return adapter_dir / "training_metadata.json"
    if entry["state"] != "pending" or adapter_dir.exists():
        raise FileExistsError(f"CBR unit is not a new pending unit: {key}")
    config = spec.training.to_lora_config(
        placement=placement,
        seed=seed,
        data_path=output_dir / "preflight" / f"train_{curriculum}.jsonl",
        output_dir=adapter_dir,
    )
    manifest.payload["state"] = "training"
    entry["state"] = "running"
    manifest.save()
    try:
        summary = train_lora(config)
        expected_data_hash = identity["data_hashes"][f"train_{curriculum}"]
        if (
            summary.optimizer_steps != spec.training.max_steps
            or summary.training_data_sha256 != expected_data_hash
            or summary.model_revision != spec.model_revision
        ):
            raise ValueError("CBR training summary differs from preregistration")
        metadata = adapter_dir / "training_metadata.json"
        entry.update(
            {
                "state": "trained",
                "training": {
                    "summary": asdict(summary),
                    "training_metadata_sha256": file_hash(metadata),
                    "curriculum": curriculum,
                    "placement": placement,
                    "seed": seed,
                    "checkpoint_selection": spec.training.checkpoint_selection,
                },
                "error": None,
            }
        )
        if all(
            value["state"] == "trained" for value in manifest.payload["training_units"].values()
        ):
            manifest.payload["state"] = "trained"
        manifest.save()
    except BaseException as error:
        manifest.fail("training", error, unit_id=key)
        raise
    return adapter_dir / "training_metadata.json"


def _validate_execution(
    output_dir: Path, payload: dict[str, Any], identity: dict[str, Any]
) -> None:
    if current_code_hash() != identity["code_sha256"]:
        raise ValueError("code changed after CBR preregistration")
    authorization_path = output_dir / "authorization.json"
    recorded = payload.get("authorization")
    if not isinstance(recorded, dict) or recorded.get("sha256") != file_hash(authorization_path):
        raise PermissionError("CBR authorization changed")
    authorization = read_json_object(authorization_path, "CBR authorization")
    if authorization.get("preregistration_sha256") != identity["preregistration_sha256"]:
        raise PermissionError("CBR authorization no longer binds the preregistration")


def _validate_trained_unit(
    output_dir: Path,
    key: str,
    entry: dict[str, Any],
    spec: Any,
    identity: dict[str, Any],
) -> None:
    curriculum, placement, seed_text = key.split("__")
    seed = int(seed_text.removeprefix("seed-"))
    metadata_path = output_dir / "adapters" / key / "training_metadata.json"
    metadata = read_json_object(metadata_path, f"CBR training metadata {key}")
    expected = spec.training.to_lora_config(
        placement=placement,
        seed=seed,
        data_path=output_dir / "preflight" / f"train_{curriculum}.jsonl",
        output_dir=output_dir / "adapters" / key,
    ).to_dict()
    training = entry.get("training")
    if (
        not isinstance(training, dict)
        or training.get("training_metadata_sha256") != file_hash(metadata_path)
        or metadata.get("config") != expected
        or metadata.get("summary") != training.get("summary")
        or metadata.get("summary", {}).get("training_data_sha256")
        != identity["data_hashes"][f"train_{curriculum}"]
    ):
        raise ValueError(f"CBR trained unit provenance differs: {key}")


def validate_all_training_units(output_dir: Path) -> dict[str, Any]:
    manifest, spec, identity = load_config(output_dir)
    if manifest.state not in {"trained", "evaluated", "complete"}:
        raise ValueError(f"CBR matrix is not fully trained: {manifest.state}")
    for key, entry in manifest.payload["training_units"].items():
        _validate_trained_unit(output_dir, key, entry, spec, identity)
    budgets: dict[tuple[str, int], dict[str, int]] = {}
    for curriculum in CURRICULA:
        for seed in spec.training.seeds:
            budgets[(curriculum, seed)] = {
                placement: int(
                    manifest.payload["training_units"][unit_id(curriculum, placement, seed)][
                        "training"
                    ]["summary"]["trainable_parameters"]
                )
                for placement in PLACEMENTS
            }
    comparisons = {
        f"{curriculum}__seed-{seed}": {
            **values,
            "relative_difference": abs(values["late_matched"] - values["full_depth"])
            / values["full_depth"],
        }
        for (curriculum, seed), values in budgets.items()
    }
    passed = all(
        value["relative_difference"] <= spec.gates.parameter_budget_relative_tolerance
        for value in comparisons.values()
    )
    if not passed:
        raise ValueError(f"CBR placement parameter budgets differ: {comparisons}")
    return {"comparisons": comparisons, "all_checks_passed": True}
