from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VALID_STATES = ("pending", "training", "trained", "evaluated", "verified", "failed")
VALID_TRANSITIONS = {
    "pending": {"pending", "training", "failed"},
    "training": {"training", "trained", "failed"},
    "trained": {"trained", "evaluated", "failed"},
    "evaluated": {"evaluated", "verified", "failed"},
    "verified": {"verified"},
    "failed": {"failed"},
}


@dataclass(slots=True)
class P0D2Manifest:
    path: Path
    payload: dict[str, Any]

    @classmethod
    def load_or_create(
        cls,
        path: Path,
        *,
        run_id: str,
        config: dict[str, Any],
        compiler_hashes: dict[str, str],
    ) -> P0D2Manifest:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "p0d2-manifest-v1":
                raise ValueError("existing file is not a P0-D2 manifest")
            if payload.get("run_id") != run_id:
                raise ValueError("existing P0-D2 manifest run_id does not match")
            if payload.get("config") != config:
                raise ValueError("existing P0-D2 manifest config does not match")
            if payload.get("compiler_hashes") != compiler_hashes:
                raise ValueError("existing P0-D2 compiler hashes do not match")
            return cls(path=path, payload=payload)
        payload = {
            "schema_version": "p0d2-manifest-v1",
            "run_id": run_id,
            "config": config,
            "compiler_hashes": compiler_hashes,
            "created_at": _now(),
            "updated_at": _now(),
            "selected_lessons": list(config["selected_lesson_ids"]),
            "conditions": {
                condition["condition_id"]: condition for condition in config["conditions"]
            },
            "base_arms": {},
            "units": {},
            "errors": [],
        }
        manifest = cls(path=path, payload=payload)
        manifest.save()
        return manifest

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.payload["updated_at"] = _now()
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def base_state(self, lesson_id: str) -> str:
        return str(self.payload["base_arms"].get(lesson_id, "pending"))

    def mark_base(self, lesson_id: str, state: str) -> None:
        _validate_state(state)
        previous = self.base_state(lesson_id)
        if previous in {"failed", "verified"} and state != previous:
            raise ValueError(f"base arm {lesson_id} is immutable after state={previous}")
        self.payload["base_arms"][lesson_id] = state
        self.save()

    def unit_state(self, condition_id: str, lesson_id: str, seed: int) -> str:
        return str(
            self.payload["units"]
            .get(unit_key(condition_id, lesson_id, seed), {})
            .get("state", "pending")
        )

    def mark_unit(
        self,
        condition_id: str,
        lesson_id: str,
        seed: int,
        state: str,
        **metadata: object,
    ) -> None:
        _validate_state(state)
        key = unit_key(condition_id, lesson_id, seed)
        previous = self.payload["units"].get(key, {})
        previous_state = str(previous.get("state", "pending"))
        if state not in VALID_TRANSITIONS[previous_state]:
            raise ValueError(f"invalid unit transition for {key}: {previous_state} -> {state}")
        self.payload["units"][key] = {
            **previous,
            "condition_id": condition_id,
            "lesson_id": lesson_id,
            "seed": seed,
            "state": state,
            "updated_at": _now(),
            **metadata,
        }
        self.save()

    def record_error(self, unit: str, error: BaseException) -> None:
        self.payload["errors"].append(
            {
                "unit": unit,
                "type": type(error).__name__,
                "message": str(error),
                "time": _now(),
            }
        )
        self.save()


def unit_key(condition_id: str, lesson_id: str, seed: int) -> str:
    return f"{condition_id}::{lesson_id}::seed-{seed}"


def _validate_state(state: str) -> None:
    if state not in VALID_STATES:
        raise ValueError(f"invalid manifest state: {state}")


def _now() -> str:
    return datetime.now(UTC).isoformat()
