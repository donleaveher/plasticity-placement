from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.io import atomic_json_write, file_hash, json_hash
from plasticity_placement.pathmem_ropcd_p0.config import (
    EXECUTION_MANIFEST_SCHEMA_VERSION,
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
class P0ExecutionManifest:
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
    ) -> P0ExecutionManifest:
        units = plan.get("units")
        if not isinstance(units, list) or len(units) != EXPECTED_UNITS:
            raise ValueError("R-OPCD P0 execution requires exactly 44 planned units")
        planned = {
            str(unit["unit_id"]): {
                "unit_id": str(unit["unit_id"]),
                "sequence_index": int(unit["sequence_index"]),
                "item_id": str(unit["item_id"]),
                "logical_name": str(unit["logical_name"]),
                "parent_unit_id": unit["parent_unit_id"],
                "technical_duplicate": bool(unit["technical_duplicate"]),
                "unit_sha256": json_hash(unit),
            }
            for unit in units
        }
        if len(planned) != EXPECTED_UNITS:
            raise ValueError("R-OPCD P0 unit IDs are not unique")
        planned_roots = {
            str(unit["item_id"]): {
                "item_id": str(unit["item_id"]),
                "root_adapter_seed": int(unit["root_adapter_seed"]),
                "state": "pending",
            }
            for unit in units
        }
        if len(planned_roots) != 4:
            raise ValueError("R-OPCD P0 requires four item-specific roots")
        identity = {
            "schema_version": EXECUTION_MANIFEST_SCHEMA_VERSION,
            "plan_manifest_id": plan_manifest_id,
            "plan_id": plan["plan_id"],
            "g1c_handoff_id": plan["g1c_handoff"]["handoff_id"],
            "g1c_run_id": plan["g1c_handoff"]["run_id"],
            "authorization_id": authorization["authorization_id"],
            "recipe": plan["recipe"],
            "recipe_sha256": plan["recipe_sha256"],
            "implementation_sha256": implementation_sha256,
            "planned_units": planned,
        }
        run_id = json_hash(identity)
        if path.exists():
            manifest = cls.load(path)
            if (
                manifest.payload.get("identity") != identity
                or manifest.payload.get("run_id") != run_id
            ):
                raise ValueError("existing R-OPCD P0 run identity changed; use a new output")
            return manifest
        payload = {
            "identity": identity,
            "run_id": run_id,
            "preflight": {"state": "pending"},
            "roots": planned_roots,
            "units": {
                unit_id: {**unit, "state": UnitState.PLANNED.value, "metadata": {}}
                for unit_id, unit in sorted(planned.items())
            },
            "control_results": {"state": "pending"},
            "aggregate": {"state": "pending"},
            "training_started": False,
            "gpu_inference_started": False,
            "evaluation_started": False,
            "path_contrast_computed": False,
            "p1_authorized": False,
            "kill_or_reserve_accessed": False,
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
    def load(cls, path: Path) -> P0ExecutionManifest:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("identity", {}).get("schema_version") != EXECUTION_MANIFEST_SCHEMA_VERSION:
            raise ValueError("R-OPCD P0 execution manifest schema changed")
        if payload.get("run_id") != json_hash(payload["identity"]):
            raise ValueError("R-OPCD P0 run identity changed")
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
            raise ValueError(f"invalid R-OPCD P0 transition: {previous} -> {state}")
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

    def record_preflight(
        self,
        *,
        state: str,
        environment_fingerprint: str,
        environment_sha256: str,
        preflight_sha256: str,
        teacher_rows_sha256: str,
        teacher_gate: dict[str, Any],
    ) -> None:
        if state not in {"passed", "teacher_failed"}:
            raise ValueError("unsupported R-OPCD P0 preflight state")
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
            raise ValueError("R-OPCD P0 preflight state changed")
        if existing.get("state") == state and existing != value:
            raise ValueError("R-OPCD P0 preflight identity changed")
        self.payload["preflight"] = value
        self.payload["gpu_inference_started"] = True
        self.save()

    def require_preflight(self) -> dict[str, Any]:
        preflight = self.payload.get("preflight")
        if not isinstance(preflight, dict) or preflight.get("state") != "passed":
            raise PermissionError("R-OPCD P0 training requires a passing preflight")
        return preflight

    def require_environment(self, fingerprint: str) -> None:
        if self.require_preflight().get("environment_fingerprint") != fingerprint:
            raise RuntimeError("R-OPCD P0 runtime differs from the preflight environment")

    def mark_gpu_inference_started(self) -> None:
        if self.payload.get("gpu_inference_started") is not True:
            self.payload["gpu_inference_started"] = True
            self.save()

    def record_root(
        self,
        *,
        item_id: str,
        root_path: Path,
        adapter_sha256: str,
        metadata_sha256: str,
    ) -> None:
        value = {
            "item_id": item_id,
            "root_adapter_seed": self._root(item_id)["root_adapter_seed"],
            "state": "verified",
            "root_path": str(root_path.resolve()),
            "adapter_sha256": adapter_sha256,
            "metadata_sha256": metadata_sha256,
        }
        existing = self._root(item_id)
        if existing.get("state") == "verified" and existing != value:
            raise ValueError(f"R-OPCD P0 root adapter changed: {item_id}")
        self.payload["roots"][item_id] = value
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
            raise ValueError("R-OPCD P0 no-memory controls changed")
        self.payload["control_results"] = value
        self.save()

    def record_aggregate(
        self,
        *,
        summary_path: Path,
        report_path: Path,
        summary: dict[str, Any],
    ) -> None:
        value = {
            "state": "complete",
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": file_hash(summary_path),
            "report_path": str(report_path.resolve()),
            "report_sha256": file_hash(report_path),
            "gate": summary["gate"],
        }
        existing = self.payload.get("aggregate", {"state": "pending"})
        if existing.get("state") == "complete" and existing != value:
            raise ValueError("R-OPCD P0 aggregate changed")
        self.payload["aggregate"] = value
        self.payload["path_contrast_computed"] = True
        self.save()

    def require_all_trained(self) -> None:
        incomplete = {
            unit_id: unit["state"]
            for unit_id, unit in self.payload["units"].items()
            if UnitState(unit["state"])
            not in {UnitState.TRAINED, UnitState.EVALUATING, UnitState.VERIFIED}
        }
        if incomplete:
            raise RuntimeError(f"R-OPCD P0 units are not all trained: {incomplete}")

    def require_all_verified(self) -> None:
        incomplete = {
            unit_id: unit["state"]
            for unit_id, unit in self.payload["units"].items()
            if UnitState(unit["state"]) is not UnitState.VERIFIED
        }
        if incomplete:
            raise RuntimeError(f"R-OPCD P0 units are not all verified: {incomplete}")

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
            raise KeyError(f"unplanned R-OPCD P0 unit: {unit_id}") from error

    def _root(self, item_id: str) -> dict[str, Any]:
        try:
            return self.payload["roots"][item_id]
        except KeyError as error:
            raise KeyError(f"unplanned R-OPCD P0 item root: {item_id}") from error


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _require_safe_payload(payload: dict[str, Any]) -> None:
    if any(
        payload.get(field) is not False
        for field in (
            "p1_authorized",
            "kill_or_reserve_accessed",
            "learned_router_started",
            "rl_controller_started",
            "automatic_hyperparameter_search_started",
        )
    ):
        raise ValueError("R-OPCD P0 manifest contains unauthorized downstream state")
