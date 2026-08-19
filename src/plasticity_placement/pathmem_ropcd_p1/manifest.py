from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.io import atomic_json_write, file_hash, json_hash
from plasticity_placement.pathmem_ropcd_p1.config import (
    EXECUTION_MANIFEST_SCHEMA_VERSION,
    EXPECTED_ROOTS,
    EXPECTED_UNITS,
)


class UnitState(StrEnum):
    PLANNED = "planned"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    VERIFIED = "verified"
    FAILED = "failed"


VALID_TRANSITIONS = {
    UnitState.PLANNED: {UnitState.TRAINING, UnitState.FAILED},
    UnitState.TRAINING: {UnitState.TRAINED, UnitState.FAILED},
    UnitState.TRAINED: {UnitState.EVALUATING, UnitState.VERIFIED, UnitState.FAILED},
    UnitState.EVALUATING: {UnitState.VERIFIED, UnitState.FAILED},
    UnitState.VERIFIED: set(),
    UnitState.FAILED: set(),
}


@dataclass(slots=True)
class P1ExecutionManifest:
    path: Path
    payload: dict[str, Any]

    @classmethod
    def load_or_create(
        cls,
        path: Path,
        *,
        plan_manifest_id: str,
        plan: dict[str, Any],
        authorization: dict[str, Any],
        implementation_sha256: str,
    ) -> P1ExecutionManifest:
        units = plan.get("units")
        if not isinstance(units, list) or len(units) != EXPECTED_UNITS:
            raise ValueError("R-OPCD P1 execution requires exactly 396 units")
        planned = {
            str(unit["unit_id"]): {
                "unit_id": str(unit["unit_id"]),
                "sequence_index": int(unit["sequence_index"]),
                "item_id": str(unit["item_id"]),
                "training_seed": int(unit["training_seed"]),
                "root_id": str(unit["root_id"]),
                "logical_name": str(unit["logical_name"]),
                "parent_unit_id": unit["parent_unit_id"],
                "technical_duplicate": bool(unit["technical_duplicate"]),
                "unit_sha256": json_hash(unit),
            }
            for unit in units
        }
        roots = {
            str(unit["root_id"]): {
                "root_id": str(unit["root_id"]),
                "item_id": str(unit["item_id"]),
                "training_seed": int(unit["training_seed"]),
                "root_adapter_seed": int(unit["root_adapter_seed"]),
                "state": "pending",
            }
            for unit in units
        }
        if len(planned) != EXPECTED_UNITS or len(roots) != EXPECTED_ROOTS:
            raise ValueError("R-OPCD P1 planned unit or root identity changed")
        identity = {
            "schema_version": EXECUTION_MANIFEST_SCHEMA_VERSION,
            "plan_manifest_id": plan_manifest_id,
            "plan_id": plan["plan_id"],
            "g2_handoff_id": plan["g2_handoff"]["handoff_id"],
            "g2_repair_id": plan["g2_handoff"]["repair_id"],
            "authorization_id": authorization["authorization_id"],
            "resource_profile": authorization["resource_profile"],
            "recipe": plan["recipe"],
            "recipe_sha256": plan["recipe_sha256"],
            "implementation_sha256": implementation_sha256,
            "planned_units": planned,
        }
        run_id = json_hash(identity)
        if path.exists():
            manifest = cls.load(path)
            if manifest.payload.get("identity") != identity or manifest.payload.get(
                "run_id"
            ) != run_id:
                raise ValueError("existing R-OPCD P1 run identity changed; use a new output")
            return manifest
        payload = {
            "identity": identity,
            "run_id": run_id,
            "preflight": {"state": "pending"},
            "roots": roots,
            "units": {
                unit_id: {**unit, "state": UnitState.PLANNED.value, "metadata": {}}
                for unit_id, unit in sorted(planned.items())
            },
            "control_results": {"state": "pending"},
            "aggregate": {"state": "pending"},
            "training_started": False,
            "gpu_inference_started": False,
            "evaluation_started": False,
            "kill_split_accessed": False,
            "kill_path_contrast_computed": False,
            "reserve_accessed": False,
            "p1b_authorized": False,
            "p2_authorized": False,
            "learned_router_started": False,
            "rl_controller_started": False,
            "automatic_hyperparameter_search_started": False,
            "errors": [],
            "updated_at": _now(),
        }
        manifest = cls(path=path, payload=payload)
        manifest.save()
        return manifest

    @classmethod
    def load(cls, path: Path) -> P1ExecutionManifest:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity", {}).get("schema_version") != EXECUTION_MANIFEST_SCHEMA_VERSION:
            raise ValueError("R-OPCD P1 execution manifest schema changed")
        if payload.get("run_id") != json_hash(payload["identity"]):
            raise ValueError("R-OPCD P1 run identity changed")
        _require_safe_payload(payload)
        return cls(path=path, payload=payload)

    def save(self) -> None:
        _require_safe_payload(self.payload)
        self.payload["updated_at"] = _now()
        atomic_json_write(self.path, self.payload)

    def unit_state(self, unit_id: str) -> UnitState:
        return UnitState(self._unit(unit_id)["state"])

    def mark_unit(self, unit_id: str, state: UnitState, **metadata: object) -> None:
        unit = self._unit(unit_id)
        previous = UnitState(unit["state"])
        if state not in VALID_TRANSITIONS[previous]:
            raise ValueError(f"invalid R-OPCD P1 transition: {previous} -> {state}")
        if state is UnitState.TRAINING:
            self.payload["training_started"] = True
            self.payload["gpu_inference_started"] = True
            self.payload["kill_split_accessed"] = True
        if state is UnitState.EVALUATING:
            self.payload["evaluation_started"] = True
            self.payload["gpu_inference_started"] = True
            self.payload["kill_split_accessed"] = True
        unit["state"] = state.value
        unit["metadata"] = {**unit.get("metadata", {}), **metadata}
        unit["updated_at"] = _now()
        self.save()

    def record_preflight(self, value: dict[str, Any]) -> None:
        if value.get("state") not in {"passed", "teacher_failed"}:
            raise ValueError("unsupported R-OPCD P1 preflight state")
        existing = self.payload.get("preflight", {"state": "pending"})
        if existing.get("state") != "pending" and existing != value:
            raise ValueError("R-OPCD P1 preflight identity changed")
        self.payload["preflight"] = value
        self.payload["gpu_inference_started"] = True
        self.payload["kill_split_accessed"] = True
        self.save()

    def require_preflight(self) -> dict[str, Any]:
        preflight = self.payload.get("preflight")
        if not isinstance(preflight, dict) or preflight.get("state") != "passed":
            raise PermissionError("R-OPCD P1 training requires a passing preflight")
        return preflight

    def require_environment(self, fingerprint: str) -> None:
        if self.require_preflight().get("environment_fingerprint") != fingerprint:
            raise RuntimeError("R-OPCD P1 runtime differs from the preflight environment")

    def mark_gpu_inference_started(self) -> None:
        self.payload["gpu_inference_started"] = True
        self.payload["kill_split_accessed"] = True
        self.save()

    def record_root(self, *, root_id: str, value: dict[str, Any]) -> None:
        existing = self._root(root_id)
        if existing.get("state") == "verified" and existing != value:
            raise ValueError(f"R-OPCD P1 root adapter changed: {root_id}")
        self.payload["roots"][root_id] = value
        self.save()

    def record_controls(self, *, result_path: Path, rows: int) -> None:
        value = {
            "state": "complete",
            "result_path": str(result_path.resolve()),
            "result_sha256": file_hash(result_path),
            "rows": rows,
        }
        existing = self.payload.get("control_results", {"state": "pending"})
        if existing.get("state") == "complete" and existing != value:
            raise ValueError("R-OPCD P1 no-memory controls changed")
        self.payload["control_results"] = value
        self.save()

    def record_aggregate(
        self, *, summary_path: Path, report_path: Path, summary: dict[str, Any]
    ) -> None:
        value = {
            "state": "complete",
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": file_hash(summary_path),
            "report_path": str(report_path.resolve()),
            "report_sha256": file_hash(report_path),
            "gate": summary["gate"],
            "decision": summary["decision"],
        }
        existing = self.payload.get("aggregate", {"state": "pending"})
        if existing.get("state") == "complete" and existing != value:
            raise ValueError("R-OPCD P1 aggregate changed")
        self.payload["aggregate"] = value
        self.payload["kill_path_contrast_computed"] = True
        self.save()

    def require_all_trained(self) -> None:
        incomplete = {
            unit_id: unit["state"]
            for unit_id, unit in self.payload["units"].items()
            if UnitState(unit["state"])
            not in {UnitState.TRAINED, UnitState.EVALUATING, UnitState.VERIFIED}
        }
        if incomplete:
            raise RuntimeError(f"R-OPCD P1 units are not all trained: {incomplete}")

    def require_all_verified(self) -> None:
        incomplete = {
            unit_id: unit["state"]
            for unit_id, unit in self.payload["units"].items()
            if UnitState(unit["state"]) is not UnitState.VERIFIED
        }
        if incomplete:
            raise RuntimeError(f"R-OPCD P1 units are not all verified: {incomplete}")

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

    def _unit(self, unit_id: str) -> dict[str, Any]:
        try:
            return self.payload["units"][unit_id]
        except KeyError as error:
            raise KeyError(f"unplanned R-OPCD P1 unit: {unit_id}") from error

    def _root(self, root_id: str) -> dict[str, Any]:
        try:
            return self.payload["roots"][root_id]
        except KeyError as error:
            raise KeyError(f"unplanned R-OPCD P1 root: {root_id}") from error


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _require_safe_payload(payload: dict[str, Any]) -> None:
    if any(
        payload.get(field) is not False
        for field in (
            "reserve_accessed",
            "p1b_authorized",
            "p2_authorized",
            "learned_router_started",
            "rl_controller_started",
            "automatic_hyperparameter_search_started",
        )
    ):
        raise ValueError("R-OPCD P1 manifest contains unauthorized downstream state")
