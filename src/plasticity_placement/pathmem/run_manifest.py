from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.dag import CoreDAGPlan
from plasticity_placement.pathmem.identity import require_sha256, verify_parent_lineage
from plasticity_placement.pathmem.io import atomic_json_write, file_hash, json_hash

RUN_MANIFEST_VERSION = "pathmem-run-manifest-v1"
ARTIFACT_INDEX_VERSION = "pathmem-artifact-index-v1"


class UnitState(StrEnum):
    PLANNED = "planned"
    TRAINING = "training"
    TRAINED = "trained"
    SCORING = "scoring"
    VERIFIED = "verified"
    FAILED = "failed"


VALID_TRANSITIONS = {
    UnitState.PLANNED: {UnitState.PLANNED, UnitState.TRAINING, UnitState.FAILED},
    UnitState.TRAINING: {UnitState.TRAINING, UnitState.TRAINED, UnitState.FAILED},
    UnitState.TRAINED: {UnitState.TRAINED, UnitState.SCORING, UnitState.FAILED},
    UnitState.SCORING: {UnitState.SCORING, UnitState.VERIFIED, UnitState.FAILED},
    UnitState.VERIFIED: {UnitState.VERIFIED},
    UnitState.FAILED: {UnitState.FAILED},
}


@dataclass(slots=True)
class PathRunManifest:
    path: Path
    payload: dict[str, Any]

    @classmethod
    def load_or_create(
        cls,
        path: Path,
        *,
        phase: str,
        g0_manifest_id: str,
        run_identity: dict[str, Any],
        dag_plans: tuple[CoreDAGPlan, ...],
        technical_duplicates: tuple[str, ...] = (),
    ) -> PathRunManifest:
        require_sha256(g0_manifest_id, "g0_manifest_id")
        planned_units = _planned_units(dag_plans, technical_duplicates)
        identity = {
            "manifest_version": RUN_MANIFEST_VERSION,
            "phase": phase,
            "g0_manifest_id": g0_manifest_id,
            "run_identity": run_identity,
            "planned_units": planned_units,
        }
        run_id = json_hash(identity)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("identity") != identity or payload.get("run_id") != run_id:
                raise ValueError("existing PathMem run manifest identity changed")
            return cls(path=path, payload=payload)
        units = {
            attempt_id: {
                **unit,
                "attempt_id": attempt_id,
                "state": UnitState.PLANNED.value,
                "metadata": {},
            }
            for attempt_id, unit in planned_units.items()
        }
        payload = {
            "identity": identity,
            "run_id": run_id,
            "units": units,
            "errors": [],
            "training_started": False,
            "updated_at": _now(),
        }
        manifest = cls(path=path, payload=payload)
        manifest.save()
        return manifest

    def save(self) -> None:
        self.payload["updated_at"] = _now()
        atomic_json_write(self.path, self.payload)

    def unit_state(self, attempt_id: str) -> UnitState:
        return UnitState(self._unit(attempt_id)["state"])

    def mark_unit(self, attempt_id: str, state: UnitState, **metadata: object) -> None:
        unit = self._unit(attempt_id)
        previous = UnitState(unit["state"])
        if state not in VALID_TRANSITIONS[previous]:
            raise ValueError(f"invalid unit transition for {attempt_id}: {previous} -> {state}")
        if state is UnitState.TRAINING:
            self.payload["training_started"] = True
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

    def _unit(self, attempt_id: str) -> dict[str, Any]:
        try:
            return self.payload["units"][attempt_id]
        except KeyError as error:
            raise KeyError(f"unplanned PathMem attempt: {attempt_id}") from error


def register_verified_artifact(
    index_path: Path,
    *,
    node_id: str,
    artifact_path: Path,
    expected_parent_sha256: str,
    observed_parent_sha256: str,
    expected_root_sha256: str,
    observed_root_sha256: str,
) -> str:
    require_sha256(node_id, "node_id")
    verify_parent_lineage(
        expected_parent_sha256=expected_parent_sha256,
        observed_parent_sha256=observed_parent_sha256,
        expected_root_sha256=expected_root_sha256,
        observed_root_sha256=observed_root_sha256,
    )
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    artifact_sha256 = file_hash(artifact_path)
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("version") != ARTIFACT_INDEX_VERSION:
            raise ValueError("artifact index version mismatch")
    else:
        payload = {"version": ARTIFACT_INDEX_VERSION, "artifacts": {}}
    existing = payload["artifacts"].get(node_id)
    record = {
        "node_id": node_id,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "parent_state_sha256": observed_parent_sha256,
        "root_state_sha256": observed_root_sha256,
    }
    if existing is not None and existing != record:
        raise ValueError("node identity already maps to a different artifact")
    payload["artifacts"][node_id] = record
    atomic_json_write(index_path, payload)
    return artifact_sha256


def _planned_units(
    dag_plans: tuple[CoreDAGPlan, ...],
    technical_duplicates: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    units: dict[str, dict[str, Any]] = {}
    node_lookup: dict[str, dict[str, Any]] = {}
    for dag in dag_plans:
        for node in dag.nodes:
            attempt_id = f"{dag.item_id}::seed-{dag.training_seed}::{node.logical_name}"
            if attempt_id in units:
                raise ValueError(f"duplicate run attempt: {attempt_id}")
            unit = {
                "item_id": dag.item_id,
                "training_seed": dag.training_seed,
                "operator": dag.operator,
                "logical_name": node.logical_name,
                "node_plan_id": node.plan_id,
                "parent_logical_name": node.parent_logical_name,
                "event_block_sha256": node.event_block_sha256,
                "ordered_minibatch_sha256": node.ordered_minibatch_sha256,
                "technical_duplicate_of": None,
            }
            units[attempt_id] = unit
            node_lookup[attempt_id] = unit
    for source_attempt_id in technical_duplicates:
        try:
            source = node_lookup[source_attempt_id]
        except KeyError as error:
            raise ValueError(
                f"technical duplicate source is not planned: {source_attempt_id}"
            ) from error
        duplicate_id = f"{source_attempt_id}::technical-duplicate"
        units[duplicate_id] = {**source, "technical_duplicate_of": source_attempt_id}
    return units


def _now() -> str:
    return datetime.now(UTC).isoformat()
