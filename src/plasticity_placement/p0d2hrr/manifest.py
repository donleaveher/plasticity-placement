from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from plasticity_placement.p0d2hrr.io import atomic_json_write

VALID_STATES = (
    "planned",
    "authorized",
    "training",
    "trained",
    "evaluating",
    "evaluated",
    "verified",
    "failed",
)
VALID_TRANSITIONS = {
    "planned": {"planned", "authorized", "failed"},
    "authorized": {"authorized", "training", "failed"},
    "training": {"training", "trained", "failed"},
    "trained": {"trained", "evaluating", "failed"},
    "evaluating": {"evaluating", "evaluated", "failed"},
    "evaluated": {"evaluated", "verified", "failed"},
    "verified": {"verified"},
    "failed": {"failed"},
}


@dataclass(slots=True)
class PilotManifest:
    path: Path
    payload: dict[str, Any]

    @classmethod
    def load_or_create(
        cls,
        path: Path,
        *,
        run_id: str,
        config: dict[str, Any],
        source_manifest_path: str,
    ) -> PilotManifest:
        expected = {
            "schema_version": "p0d2hrr-manifest-v1",
            "run_id": run_id,
            "config": config,
            "source_manifest_path": source_manifest_path,
        }
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            for key, value in expected.items():
                if payload.get(key) != value:
                    raise ValueError(f"existing route-remediation {key} does not match")
            return cls(path=path, payload=payload)
        payload = {
            **expected,
            "state": "planned",
            "created_at": _now(),
            "updated_at": _now(),
            "authorization": None,
            "training": None,
            "evaluation": None,
            "aggregate": None,
            "errors": [],
        }
        manifest = cls(path=path, payload=payload)
        manifest.save()
        return manifest

    @classmethod
    def load(cls, path: Path) -> PilotManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "p0d2hrr-manifest-v1":
            raise ValueError("not a P0-D2H-RR manifest")
        return cls(path=path, payload=payload)

    @property
    def state(self) -> str:
        return str(self.payload["state"])

    def transition(self, state: str, **metadata: object) -> None:
        if state not in VALID_STATES:
            raise ValueError(f"unsupported route-remediation state: {state}")
        previous = self.state
        if state not in VALID_TRANSITIONS[previous]:
            raise ValueError(f"invalid route-remediation transition: {previous} -> {state}")
        self.payload["state"] = state
        for key, value in metadata.items():
            self.payload[key] = value
        self.save()

    def record_error(self, stage: str, error: BaseException) -> None:
        self.payload["errors"].append(
            {
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
                "time": _now(),
            }
        )
        self.payload["state"] = "failed"
        self.save()

    def save(self) -> None:
        self.payload["updated_at"] = _now()
        atomic_json_write(self.path, self.payload)


def _now() -> str:
    return datetime.now(UTC).isoformat()
