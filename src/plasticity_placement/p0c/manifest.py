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
class RunManifest:
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
    ) -> RunManifest:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("run_id") != run_id:
                raise ValueError(f"manifest run_id mismatch: {payload.get('run_id')} != {run_id}")
            if payload.get("config") != config:
                raise ValueError("existing manifest config does not match the requested run")
            return cls(path=path, payload=payload)
        payload = {
            "run_id": run_id,
            "config": config,
            "compiler_hashes": compiler_hashes,
            "created_at": _now(),
            "updated_at": _now(),
            "selected_lessons": [],
            "units": {},
            "base_arms": {},
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

    def set_selected_lessons(self, lesson_ids: list[str]) -> None:
        existing = self.payload.get("selected_lessons", [])
        if existing and existing != lesson_ids:
            raise ValueError("selected lesson set differs from the existing manifest")
        self.payload["selected_lessons"] = lesson_ids
        self.save()

    def base_arm_complete(self, lesson_id: str) -> bool:
        return self.payload["base_arms"].get(lesson_id) == "verified"

    def mark_base_arm(self, lesson_id: str, state: str) -> None:
        _validate_state(state)
        previous = self.payload["base_arms"].get(lesson_id)
        if previous in {"failed", "verified"} and state != previous:
            raise ValueError(f"base arm {lesson_id} is immutable after state={previous}")
        self.payload["base_arms"][lesson_id] = state
        self.save()

    def unit_state(self, lesson_id: str, seed: int) -> str:
        return self.payload["units"].get(_unit_key(lesson_id, seed), {}).get("state", "pending")

    def mark_unit(
        self,
        lesson_id: str,
        seed: int,
        state: str,
        **metadata: object,
    ) -> None:
        _validate_state(state)
        key = _unit_key(lesson_id, seed)
        previous = self.payload["units"].get(key, {})
        previous_state = str(previous.get("state", "pending"))
        if state not in VALID_TRANSITIONS[previous_state]:
            raise ValueError(f"invalid unit transition for {key}: {previous_state} -> {state}")
        self.payload["units"][key] = {
            **previous,
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


def _unit_key(lesson_id: str, seed: int) -> str:
    return f"{lesson_id}::seed-{seed}"


def _validate_state(state: str) -> None:
    if state not in VALID_STATES:
        raise ValueError(f"invalid manifest state: {state}")


def _now() -> str:
    return datetime.now(UTC).isoformat()
