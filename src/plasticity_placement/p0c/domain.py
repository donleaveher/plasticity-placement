from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class Arm(StrEnum):
    NO_WRITE = "no_write"
    EXTERNAL = "external"
    PARAMETRIC = "parametric"
    BOTH = "both"
    ROLLBACK = "rollback"


class Tier(StrEnum):
    DEVELOPMENT = "development"
    SMOKE = "smoke"
    PILOT = "pilot"
    CONFIRMATORY = "confirmatory"


@dataclass(frozen=True, slots=True)
class Lesson:
    lesson_id: str
    pair_id: str
    lesson_type: str
    context_id: str
    condition: str
    desired_action: str
    distractor_actions: tuple[str, ...]
    evidence: str
    valid_from: int
    confidence: float
    split: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Lesson:
        return cls(
            lesson_id=str(value["lesson_id"]),
            pair_id=str(value["pair_id"]),
            lesson_type=str(value["lesson_type"]),
            context_id=str(value["context_id"]),
            condition=str(value["condition"]),
            desired_action=str(value["desired_action"]),
            distractor_actions=tuple(str(item) for item in value["distractor_actions"]),
            evidence=str(value["evidence"]),
            valid_from=int(value["valid_from"]),
            confidence=float(value["confidence"]),
            split=str(value["split"]),
        )


@dataclass(frozen=True, slots=True)
class P0CProbe:
    probe_id: str
    lesson_id: str
    pair_id: str
    lesson_type: str
    category: str
    prompt: str
    action_choices: tuple[str, ...]
    expected_action: str
    retrieval_relevant: bool
    split: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> P0CProbe:
        return cls(
            probe_id=str(value["probe_id"]),
            lesson_id=str(value["lesson_id"]),
            pair_id=str(value["pair_id"]),
            lesson_type=str(value["lesson_type"]),
            category=str(value["category"]),
            prompt=str(value["prompt"]),
            action_choices=tuple(str(item) for item in value["action_choices"]),
            expected_action=str(value["expected_action"]),
            retrieval_relevant=bool(value["retrieval_relevant"]),
            split=str(value["split"]),
        )


@dataclass(frozen=True, slots=True)
class TrainingRow:
    lesson_id: str
    prompt: str
    completion: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CompiledLesson:
    lesson: Lesson
    external_note: str
    training_rows: tuple[TrainingRow, ...]
    screening_probes: tuple[P0CProbe, ...]
    evaluation_probes: tuple[P0CProbe, ...]
