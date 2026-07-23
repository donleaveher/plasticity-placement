from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256

from plasticity_placement.simulation.carriers import make_carrier
from plasticity_placement.simulation.domain import CarrierKind, ExperimentalUnit


@dataclass(frozen=True, slots=True)
class BranchResult:
    unit_id: str
    seed: int
    recurrence: float
    volatility: float
    carrier: CarrierKind
    reward: float
    lesson_reward: float
    background_reward: float
    write_cost: float
    state_hash: str
    panel_hash: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def stable_hash(value: object) -> str:
    payload = json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)
    return sha256(payload.encode()).hexdigest()


def run_branch(unit: ExperimentalUnit, kind: CarrierKind) -> BranchResult:
    carrier = make_carrier(kind)
    carrier.write(unit.lesson)
    lesson_scores: list[int] = []
    background_scores: list[int] = []

    for task in unit.future_tasks:
        score = int(carrier.act(task.context, unit.pre_write_state) == task.correct_action)
        (lesson_scores if task.requires_lesson else background_scores).append(score)

    all_scores = lesson_scores + background_scores
    return BranchResult(
        unit_id=unit.unit_id,
        seed=unit.seed,
        recurrence=unit.recurrence,
        volatility=unit.volatility,
        carrier=kind,
        reward=sum(all_scores) / len(all_scores),
        lesson_reward=_mean_or_zero(lesson_scores),
        background_reward=_mean_or_zero(background_scores),
        write_cost=carrier.write_cost,
        state_hash=stable_hash(asdict(unit.pre_write_state)),
        panel_hash=stable_hash([asdict(task) for task in unit.future_tasks]),
    )


def run_unit(unit: ExperimentalUnit) -> tuple[BranchResult, ...]:
    return tuple(run_branch(unit, kind) for kind in CarrierKind)


def _mean_or_zero(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0
