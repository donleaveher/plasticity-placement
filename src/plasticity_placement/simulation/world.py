from __future__ import annotations

import random

from plasticity_placement.simulation.domain import AgentState, ExperimentalUnit, Lesson, Task


class ToolWorld:
    """Generate matched future panels with independently controlled dynamics."""

    def __init__(self, horizon: int = 20) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.horizon = horizon

    def create_unit(self, seed: int, recurrence: float, volatility: float) -> ExperimentalUnit:
        self._validate_probability("recurrence", recurrence)
        self._validate_probability("volatility", volatility)
        rng = random.Random(seed)
        target_context = f"target-{seed}"
        lesson = Lesson(context=target_context, action="a1")
        base_policy = tuple((f"background-{index}", f"a{index % 2}") for index in range(4))
        state = AgentState(default_action="a0", base_policy=base_policy)

        tasks: list[Task] = []
        for _ in range(self.horizon):
            requires_lesson = rng.random() < recurrence
            if requires_lesson:
                flipped = rng.random() < volatility
                tasks.append(
                    Task(
                        context=target_context,
                        correct_action="a0" if flipped else lesson.action,
                        requires_lesson=True,
                        rule_flipped=flipped,
                    )
                )
            else:
                context, correct_action = rng.choice(base_policy)
                tasks.append(
                    Task(
                        context=context,
                        correct_action=correct_action,
                        requires_lesson=False,
                        rule_flipped=False,
                    )
                )

        return ExperimentalUnit(
            unit_id=f"seed-{seed}-r{recurrence:.1f}-v{volatility:.1f}",
            seed=seed,
            recurrence=recurrence,
            volatility=volatility,
            lesson=lesson,
            pre_write_state=state,
            future_tasks=tuple(tasks),
        )

    @staticmethod
    def _validate_probability(name: str, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
