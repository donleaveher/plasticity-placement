from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CarrierKind(StrEnum):
    NONE = "none"
    EXTERNAL = "external"
    PARAMETRIC = "parametric"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class Lesson:
    context: str
    action: str


@dataclass(frozen=True, slots=True)
class Task:
    context: str
    correct_action: str
    requires_lesson: bool
    rule_flipped: bool


@dataclass(frozen=True, slots=True)
class AgentState:
    default_action: str
    base_policy: tuple[tuple[str, str], ...]

    def action_for(self, context: str) -> str:
        return dict(self.base_policy).get(context, self.default_action)


@dataclass(frozen=True, slots=True)
class ExperimentalUnit:
    unit_id: str
    seed: int
    recurrence: float
    volatility: float
    lesson: Lesson
    pre_write_state: AgentState
    future_tasks: tuple[Task, ...]
