from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from plasticity_placement.p0d2hrr.io import atomic_json_write

TRANSITIONS = {
    "planned": {"planned", "authorized", "failed"},
    "authorized": {"authorized", "training", "failed"},
    "training": {"training", "trained", "failed"},
    "trained": {"trained"},
    "failed": {"failed"},
}


@dataclass(slots=True)
class Manifest:
    path: Path
    payload: dict[str, Any]

    @classmethod
    def load_or_create(cls, path: Path, payload: dict[str, Any]) -> Manifest:
        expected = {"schema_version": "p0d2hcpr-manifest-v1", **payload}
        if path.exists():
            observed = json.loads(path.read_text(encoding="utf-8"))
            for key, value in expected.items():
                if observed.get(key) != value:
                    raise ValueError(f"existing CPR {key} differs; use a new attempt")
            return cls(path, observed)
        now = datetime.now(UTC).isoformat()
        manifest = cls(
            path,
            {
                **expected,
                "state": "planned",
                "created_at": now,
                "updated_at": now,
                "authorization": None,
                "training": None,
                "errors": [],
            },
        )
        manifest.save()
        return manifest

    @classmethod
    def load(cls, path: Path) -> Manifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "p0d2hcpr-manifest-v1":
            raise ValueError("not a CPR-v1 manifest")
        return cls(path, payload)

    @property
    def state(self) -> str:
        return str(self.payload["state"])

    def transition(self, state: str, **metadata: object) -> None:
        if state not in TRANSITIONS.get(self.state, set()):
            raise ValueError(f"invalid CPR transition: {self.state} -> {state}")
        self.payload["state"] = state
        self.payload.update(metadata)
        self.save()

    def fail(self, stage: str, error: BaseException) -> None:
        self.payload["state"] = "failed"
        self.payload["errors"].append(
            {
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
                "time": datetime.now(UTC).isoformat(),
            }
        )
        self.save()

    def save(self) -> None:
        self.payload["updated_at"] = datetime.now(UTC).isoformat()
        atomic_json_write(self.path, self.payload)
