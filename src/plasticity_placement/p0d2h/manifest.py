from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VALID_STATES = ("pending", "evaluating", "verified", "failed")
VALID_TRANSITIONS = {
    "pending": {"pending", "evaluating", "failed"},
    "evaluating": {"evaluating", "verified", "failed"},
    "verified": {"verified"},
    "failed": {"failed"},
}


@dataclass(slots=True)
class P0D2HManifest:
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
        prompt_token_audit_sha256: str,
    ) -> P0D2HManifest:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "p0d2h-manifest-v1":
                raise ValueError("existing file is not a P0-D2H manifest")
            if payload.get("run_id") != run_id:
                raise ValueError("existing P0-D2H run_id does not match")
            if payload.get("config") != config:
                raise ValueError("existing P0-D2H config does not match")
            if payload.get("source_manifest_path") != source_manifest_path:
                raise ValueError("existing P0-D2H source manifest path does not match")
            if payload.get("prompt_token_audit_sha256") != prompt_token_audit_sha256:
                raise ValueError("existing P0-D2H prompt-token audit does not match")
            return cls(path=path, payload=payload)
        payload = {
            "schema_version": "p0d2h-manifest-v1",
            "run_id": run_id,
            "config": config,
            "source_manifest_path": source_manifest_path,
            "prompt_token_audit_sha256": prompt_token_audit_sha256,
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
        value = self.payload["base_arms"].get(lesson_id, "pending")
        return str(value.get("state", "pending")) if isinstance(value, dict) else str(value)

    def mark_base(
        self,
        lesson_id: str,
        state: str,
        **metadata: object,
    ) -> None:
        _validate_state(state)
        previous = self.payload["base_arms"].get(lesson_id, {})
        previous_state = (
            str(previous.get("state", "pending")) if isinstance(previous, dict) else str(previous)
        )
        if state not in VALID_TRANSITIONS[previous_state]:
            raise ValueError(
                f"invalid base transition for {lesson_id}: {previous_state} -> {state}"
            )
        self.payload["base_arms"][lesson_id] = {
            **(previous if isinstance(previous, dict) else {}),
            "lesson_id": lesson_id,
            "state": state,
            "updated_at": _now(),
            **metadata,
        }
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
        raise ValueError(f"invalid P0-D2H state: {state}")


def _now() -> str:
    return datetime.now(UTC).isoformat()
