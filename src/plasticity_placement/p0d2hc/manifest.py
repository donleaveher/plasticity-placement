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
class P0D2HCManifest:
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
    ) -> P0D2HCManifest:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            expected = {
                "schema_version": "p0d2hc-manifest-v1",
                "run_id": run_id,
                "config": config,
                "source_manifest_path": source_manifest_path,
                "prompt_token_audit_sha256": (
                    prompt_token_audit_sha256
                ),
            }
            for key, value in expected.items():
                if payload.get(key) != value:
                    raise ValueError(
                        f"existing P0-D2H-CAL {key} does not match"
                    )
            return cls(path=path, payload=payload)
        payload = {
            "schema_version": "p0d2hc-manifest-v1",
            "run_id": run_id,
            "config": config,
            "source_manifest_path": source_manifest_path,
            "prompt_token_audit_sha256": prompt_token_audit_sha256,
            "created_at": _now(),
            "updated_at": _now(),
            "selected_lessons": list(config["selected_lesson_ids"]),
            "models": {
                model["model_id"]: model for model in config["models"]
            },
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
            json.dumps(
                self.payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def unit_state(self, model_id: str, lesson_id: str) -> str:
        return str(
            self.payload["units"]
            .get(unit_key(model_id, lesson_id), {})
            .get("state", "pending")
        )

    def mark_unit(
        self,
        model_id: str,
        lesson_id: str,
        state: str,
        **metadata: object,
    ) -> None:
        _validate_state(state)
        key = unit_key(model_id, lesson_id)
        previous = self.payload["units"].get(key, {})
        previous_state = str(previous.get("state", "pending"))
        if state not in VALID_TRANSITIONS[previous_state]:
            raise ValueError(
                f"invalid calibration transition for {key}: "
                f"{previous_state} -> {state}"
            )
        self.payload["units"][key] = {
            **previous,
            "model_id": model_id,
            "lesson_id": lesson_id,
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


def unit_key(model_id: str, lesson_id: str) -> str:
    return f"{model_id}::{lesson_id}"


def _validate_state(state: str) -> None:
    if state not in VALID_STATES:
        raise ValueError(f"invalid P0-D2H-CAL state: {state}")


def _now() -> str:
    return datetime.now(UTC).isoformat()
