from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.identity import require_sha256
from plasticity_placement.pathmem.io import (
    atomic_json_write,
    canonical_json_bytes,
    file_hash,
    immutable_json_write,
    json_hash,
)
from plasticity_placement.pathmem_exec.config import G1_AUTHORIZATION_VERSION

PHASE_MANIFEST_VERSION = "pathmem-phase-manifest-v1"
ADAPTER_LINEAGE_VERSION = "pathmem-adapter-lineage-v1"


class AttemptState(StrEnum):
    PLANNED = "planned"
    TRAINING = "training"
    TRAINED = "trained"
    SCORING = "scoring"
    VERIFIED = "verified"
    FAILED = "failed"


VALID_TRANSITIONS = {
    AttemptState.PLANNED: {AttemptState.PLANNED, AttemptState.TRAINING, AttemptState.FAILED},
    AttemptState.TRAINING: {
        AttemptState.TRAINING,
        AttemptState.TRAINED,
        AttemptState.FAILED,
    },
    AttemptState.TRAINED: {
        AttemptState.TRAINED,
        AttemptState.SCORING,
        AttemptState.FAILED,
    },
    AttemptState.SCORING: {
        AttemptState.SCORING,
        AttemptState.VERIFIED,
        AttemptState.FAILED,
    },
    AttemptState.VERIFIED: {AttemptState.VERIFIED},
    AttemptState.FAILED: {AttemptState.FAILED},
}


@dataclass(slots=True)
class PhaseManifest:
    path: Path
    payload: dict[str, Any]

    @classmethod
    def load_or_create(
        cls,
        path: Path,
        *,
        phase: str,
        g0_manifest_id: str,
        identity: dict[str, Any],
        planned_units: dict[str, dict[str, Any]],
    ) -> PhaseManifest:
        require_sha256(g0_manifest_id, "g0_manifest_id")
        if not planned_units:
            raise ValueError("phase manifest requires planned units")
        frozen_identity = {
            "manifest_version": PHASE_MANIFEST_VERSION,
            "phase": phase,
            "g0_manifest_id": g0_manifest_id,
            "execution_identity": identity,
            "planned_units": planned_units,
        }
        run_id = json_hash(frozen_identity)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("identity") != frozen_identity or payload.get("run_id") != run_id:
                raise ValueError("existing phase manifest identity changed; use a new output")
            return cls(path=path, payload=payload)
        payload = {
            "identity": frozen_identity,
            "run_id": run_id,
            "units": {
                attempt_id: {
                    **unit,
                    "attempt_id": attempt_id,
                    "state": AttemptState.PLANNED.value,
                    "metadata": {},
                }
                for attempt_id, unit in sorted(planned_units.items())
            },
            "training_started": False,
            "scoring_started": False,
            "errors": [],
            "updated_at": _now(),
        }
        manifest = cls(path=path, payload=payload)
        manifest.save()
        return manifest

    def save(self) -> None:
        self.payload["updated_at"] = _now()
        atomic_json_write(self.path, self.payload)

    def state(self, attempt_id: str) -> AttemptState:
        return AttemptState(self._unit(attempt_id)["state"])

    def mark(self, attempt_id: str, state: AttemptState, **metadata: object) -> None:
        unit = self._unit(attempt_id)
        previous = AttemptState(unit["state"])
        if state not in VALID_TRANSITIONS[previous]:
            raise ValueError(f"invalid attempt transition: {attempt_id}: {previous} -> {state}")
        if state is AttemptState.TRAINING:
            self.payload["training_started"] = True
        if state is AttemptState.SCORING:
            self.payload["scoring_started"] = True
        unit["state"] = state.value
        unit["metadata"] = {**unit.get("metadata", {}), **metadata}
        unit["updated_at"] = _now()
        self.save()

    def record_error(self, attempt_id: str, error: BaseException) -> None:
        self._unit(attempt_id)
        self.payload["errors"].append(
            {
                "attempt_id": attempt_id,
                "type": type(error).__name__,
                "message": str(error),
                "time": _now(),
            }
        )
        self.save()

    def require_all_verified(self) -> None:
        incomplete = {
            attempt_id: unit["state"]
            for attempt_id, unit in self.payload["units"].items()
            if unit["state"] != AttemptState.VERIFIED.value
        }
        if incomplete:
            raise RuntimeError(f"phase contains incomplete units: {incomplete}")

    def _unit(self, attempt_id: str) -> dict[str, Any]:
        try:
            return self.payload["units"][attempt_id]
        except KeyError as error:
            raise KeyError(f"unplanned attempt: {attempt_id}") from error


def adapter_bundle_files(adapter_dir: Path) -> tuple[Path, ...]:
    config_path = adapter_dir / "adapter_config.json"
    weight_files = sorted(adapter_dir.glob("adapter_model.*"))
    if not config_path.is_file() or not weight_files:
        raise FileNotFoundError(f"incomplete adapter bundle: {adapter_dir}")
    return (config_path, *weight_files)


def adapter_bundle_hash(adapter_dir: Path) -> str:
    digest = sha256()
    for path in adapter_bundle_files(adapter_dir):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_lineage(adapter_dir: Path, payload: dict[str, Any]) -> Path:
    required = {
        "attempt_id",
        "adapter_sha256",
        "parent_state_sha256",
        "root_state_sha256",
        "training_data_sha256",
        "trainer_config_sha256",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"adapter lineage is missing fields: {missing}")
    for field in required - {"attempt_id"}:
        require_sha256(str(payload[field]), field)
    path = adapter_dir / "pathmem_lineage.json"
    immutable_json_write(
        path,
        {"schema_version": ADAPTER_LINEAGE_VERSION, **payload},
        "PathMem adapter lineage",
    )
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} line {line_number} is not an object")
        rows.append(value)
    return rows


def immutable_jsonl_write(path: Path, rows: list[dict[str, Any]], label: str) -> str:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing {label} changed; use a new output")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return sha256(payload).hexdigest()


def write_g1_authorization(
    path: Path,
    *,
    g0_manifest_id: str,
    run_id: str,
    summary_path: Path,
    gate: dict[str, Any],
    integrity_checks: dict[str, bool],
) -> dict[str, Any]:
    require_sha256(g0_manifest_id, "g0_manifest_id")
    require_sha256(run_id, "run_id")
    if gate.get("passed") is not True or not integrity_checks or not all(integrity_checks.values()):
        raise PermissionError("G1 authorization requires a passing gate and all integrity checks")
    identity = {
        "schema_version": G1_AUTHORIZATION_VERSION,
        "g0_manifest_id": g0_manifest_id,
        "g1_run_id": run_id,
        "summary_sha256": file_hash(summary_path),
        "gate": gate,
        "integrity_checks": integrity_checks,
        "scope": "authorize_P0_only",
    }
    payload = {**identity, "authorization_id": json_hash(identity)}
    immutable_json_write(path, payload, "G1 authorization")
    return payload


def verify_g1_authorization(path: Path, *, expected_g0_manifest_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    authorization_id = payload.pop("authorization_id", None)
    if payload.get("schema_version") != G1_AUTHORIZATION_VERSION:
        raise ValueError("G1 authorization schema mismatch")
    if payload.get("g0_manifest_id") != expected_g0_manifest_id:
        raise ValueError("G1 authorization refers to a different G0 manifest")
    if authorization_id != json_hash(payload):
        raise ValueError("G1 authorization identity mismatch")
    if payload.get("gate", {}).get("passed") is not True:
        raise PermissionError("G1 authorization gate did not pass")
    checks = payload.get("integrity_checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise PermissionError("G1 authorization integrity checks did not pass")
    summary_path = path.parent / "summary.json"
    if not summary_path.is_file() or file_hash(summary_path) != payload.get("summary_sha256"):
        raise ValueError("G1 authorization summary hash mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("run_id") != payload.get("g1_run_id")
        or summary.get("g0_manifest_id") != expected_g0_manifest_id
        or summary.get("gate") != payload.get("gate")
        or summary.get("integrity_checks") != checks
    ):
        raise ValueError("G1 authorization summary contents mismatch")
    return {**payload, "authorization_id": authorization_id}


def _now() -> str:
    return datetime.now(UTC).isoformat()
