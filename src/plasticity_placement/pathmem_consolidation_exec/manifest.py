from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.io import atomic_json_write, file_hash, json_hash
from plasticity_placement.pathmem_consolidation_exec.config import MANIFEST_SCHEMA_VERSION


class UnitState(StrEnum):
    PLANNED = "planned"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    VERIFIED = "verified"


VALID_TRANSITIONS = {
    UnitState.PLANNED: {UnitState.PLANNED, UnitState.TRAINING},
    UnitState.TRAINING: {UnitState.TRAINING, UnitState.TRAINED},
    UnitState.TRAINED: {UnitState.TRAINED, UnitState.EVALUATING},
    UnitState.EVALUATING: {UnitState.EVALUATING, UnitState.VERIFIED},
    UnitState.VERIFIED: {UnitState.VERIFIED},
}


@dataclass(slots=True)
class G1CExecutionManifest:
    path: Path
    payload: dict[str, Any]

    @classmethod
    def load_or_create(
        cls,
        path: Path,
        *,
        source: dict[str, Any],
        authorization: dict[str, Any],
        recipe: dict[str, Any],
        recipe_sha256: str,
        implementation_sha256: str,
        units: list[dict[str, Any]],
    ) -> G1CExecutionManifest:
        if len(units) != 24:
            raise ValueError("G1-C execution manifest requires 24 planned units")
        planned = {
            str(unit["unit_id"]): {
                "unit_id": str(unit["unit_id"]),
                "sequence_index": int(unit["sequence_index"]),
                "item_id": str(unit["item_id"]),
                "terminal_state": str(unit["terminal_state"]),
                "memory_key": str(unit["memory_key"]),
                "unit_sha256": json_hash(unit),
            }
            for unit in units
        }
        if len(planned) != 24:
            raise ValueError("G1-C unit IDs must be unique")
        identity = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
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
            "authorization_id": authorization["authorization_id"],
            "recipe": recipe,
            "recipe_sha256": recipe_sha256,
            "implementation_sha256": implementation_sha256,
            "planned_units": planned,
        }
        run_id = json_hash(identity)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("identity") != identity or payload.get("run_id") != run_id:
                raise ValueError("existing G1-C manifest identity changed; use a new output")
            _require_safe_payload(payload)
            return cls(path=path, payload=payload)
        payload = {
            "identity": identity,
            "run_id": run_id,
            "preflight": {"state": "pending"},
            "units": {
                unit_id: {
                    **unit,
                    "state": UnitState.PLANNED.value,
                    "metadata": {},
                }
                for unit_id, unit in sorted(planned.items())
            },
            "training_started": False,
            "gpu_inference_started": False,
            "evaluation_started": False,
            "aggregate": {"state": "pending"},
            "p0_authorized": False,
            "path_contrast_computed": False,
            "kill_or_reserve_accessed": False,
            "errors": [],
            "updated_at": _now(),
        }
        manifest = cls(path=path, payload=payload)
        manifest.save()
        return manifest

    @classmethod
    def load(cls, path: Path) -> G1CExecutionManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity", {}).get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError("G1-C execution manifest schema mismatch")
        if payload.get("run_id") != json_hash(payload["identity"]):
            raise ValueError("G1-C execution manifest run identity mismatch")
        _require_safe_payload(payload)
        return cls(path=path, payload=payload)

    def save(self) -> None:
        self.payload["updated_at"] = _now()
        atomic_json_write(self.path, self.payload)

    def require_preflight(self) -> dict[str, Any]:
        preflight = self.payload.get("preflight")
        if not isinstance(preflight, dict) or preflight.get("state") != "passed":
            raise PermissionError("G1-C training is blocked until GPU preflight passes")
        return preflight

    def record_preflight(
        self,
        *,
        environment_fingerprint: str,
        environment_sha256: str,
        preflight_sha256: str,
        teacher_rows_sha256: str,
        teacher_gate: dict[str, Any],
    ) -> None:
        state = "passed" if teacher_gate.get("passed") is True else "teacher_failed"
        value = {
            "state": state,
            "environment_fingerprint": environment_fingerprint,
            "environment_sha256": environment_sha256,
            "preflight_sha256": preflight_sha256,
            "teacher_rows_sha256": teacher_rows_sha256,
            "teacher_gate": teacher_gate,
        }
        existing = self.payload.get("preflight", {"state": "pending"})
        if existing.get("state") not in {"pending", state}:
            raise ValueError("G1-C preflight state changed")
        if existing.get("state") == state and existing != value:
            raise ValueError("G1-C preflight identity changed")
        self.payload["preflight"] = value
        self.payload["gpu_inference_started"] = True
        self.save()

    def require_environment(self, environment_fingerprint: str) -> None:
        preflight = self.require_preflight()
        if preflight.get("environment_fingerprint") != environment_fingerprint:
            raise RuntimeError(
                "execution environment differs from preflight; use the same locked runtime"
            )

    def unit_state(self, unit_id: str) -> UnitState:
        return UnitState(self._unit(unit_id)["state"])

    def mark_unit(self, unit_id: str, state: UnitState, **metadata: object) -> None:
        unit = self._unit(unit_id)
        previous = UnitState(unit["state"])
        if state not in VALID_TRANSITIONS[previous]:
            raise ValueError(f"invalid G1-C unit transition: {previous} -> {state}")
        if state is UnitState.TRAINING:
            self.payload["training_started"] = True
            self.payload["gpu_inference_started"] = True
        if state is UnitState.EVALUATING:
            self.payload["evaluation_started"] = True
            self.payload["gpu_inference_started"] = True
        unit["state"] = state.value
        unit["metadata"] = {**unit.get("metadata", {}), **metadata}
        unit["updated_at"] = _now()
        self.save()

    def record_aggregate(
        self,
        *,
        summary_path: Path,
        report_path: Path,
        summary: dict[str, Any],
    ) -> None:
        aggregate = {
            "state": "complete",
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": file_hash(summary_path),
            "report_path": str(report_path.resolve()),
            "report_sha256": file_hash(report_path),
            "gate": summary["gate"],
            "p0_runner_implementation_review_eligible": bool(
                summary["p0_runner_implementation_review_eligible"]
            ),
        }
        existing = self.payload.get("aggregate", {"state": "pending"})
        if existing.get("state") == "complete" and existing != aggregate:
            raise ValueError("G1-C aggregate changed")
        self.payload["aggregate"] = aggregate
        self.save()

    def record_error(self, stage: str, error: BaseException, unit_id: str | None = None) -> None:
        self.payload["errors"].append(
            {
                "stage": stage,
                "unit_id": unit_id,
                "type": type(error).__name__,
                "message": str(error),
                "time": _now(),
            }
        )
        self.save()

    def require_all_trained(self) -> None:
        incomplete = {
            unit_id: unit["state"]
            for unit_id, unit in self.payload["units"].items()
            if UnitState(unit["state"])
            not in {UnitState.TRAINED, UnitState.EVALUATING, UnitState.VERIFIED}
        }
        if incomplete:
            raise RuntimeError(f"G1-C units are not all trained: {incomplete}")

    def require_all_verified(self) -> None:
        incomplete = {
            unit_id: unit["state"]
            for unit_id, unit in self.payload["units"].items()
            if UnitState(unit["state"]) is not UnitState.VERIFIED
        }
        if incomplete:
            raise RuntimeError(f"G1-C units are not all evaluated: {incomplete}")

    def _unit(self, unit_id: str) -> dict[str, Any]:
        try:
            return self.payload["units"][unit_id]
        except KeyError as error:
            raise KeyError(f"unplanned G1-C unit: {unit_id}") from error


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _require_safe_payload(payload: dict[str, Any]) -> None:
    if any(
        payload.get(field) is not False
        for field in ("p0_authorized", "path_contrast_computed", "kill_or_reserve_accessed")
    ):
        raise ValueError("G1-C manifest contains unauthorized downstream state")
