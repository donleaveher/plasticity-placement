from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from plasticity_placement.p0d2hrr.io import atomic_json_write

SCHEMA_VERSION = "p0d2hcbr-manifest-v1"


@dataclass(slots=True)
class Manifest:
    path: Path
    payload: dict[str, Any]

    @classmethod
    def create(cls, path: Path, payload: dict[str, Any], unit_ids: tuple[str, ...]) -> Manifest:
        if path.exists():
            raise FileExistsError(f"CBR manifest already exists: {path}")
        now = datetime.now(UTC).isoformat()
        value = cls(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                **payload,
                "state": "planned",
                "created_at": now,
                "updated_at": now,
                "authorization": None,
                "training_units": {
                    unit_id: {"state": "pending", "training": None, "error": None}
                    for unit_id in unit_ids
                },
                "evaluation_claims": {},
                "result": None,
                "errors": [],
                "training_authorized": False,
                "mappings_per_adapter_authorized": False,
            },
        )
        value.save()
        return value

    @classmethod
    def load(cls, path: Path) -> Manifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("not a CBR-v1 manifest")
        return cls(path, payload)

    @property
    def state(self) -> str:
        return str(self.payload["state"])

    def set_state(self, state: str) -> None:
        allowed = {
            "planned": {"planned", "authorized", "failed"},
            "authorized": {"authorized", "training", "failed"},
            "training": {"training", "trained", "failed"},
            "trained": {"trained", "evaluated", "failed"},
            "evaluated": {"evaluated", "complete", "failed"},
            "complete": {"complete"},
            "failed": {"failed"},
        }
        if state not in allowed.get(self.state, set()):
            raise ValueError(f"invalid CBR transition: {self.state} -> {state}")
        self.payload["state"] = state
        self.save()

    def fail(self, stage: str, error: BaseException, *, unit_id: str | None = None) -> None:
        record = {
            "stage": stage,
            "unit_id": unit_id,
            "type": type(error).__name__,
            "message": str(error),
            "time": datetime.now(UTC).isoformat(),
        }
        self.payload["errors"].append(record)
        if unit_id is not None and unit_id in self.payload["training_units"]:
            self.payload["training_units"][unit_id]["state"] = "failed"
            self.payload["training_units"][unit_id]["error"] = record
        self.payload["state"] = "failed"
        self.save()

    def save(self) -> None:
        self.payload["updated_at"] = datetime.now(UTC).isoformat()
        atomic_json_write(self.path, self.payload)
