from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from plasticity_placement.training.config import LayerBand, select_layers

TARGET_MODULES = ("q_proj", "v_proj")


class P0D2Stage(StrEnum):
    BUDGET_MATCH = "budget_match"


@dataclass(frozen=True, slots=True)
class BudgetCondition:
    condition_id: str
    layer_band: LayerBand
    rank_multiplier: int
    alpha_multiplier: int
    parameter_budget_regime: str
    baseline_condition_id: str | None = None
    budget_reference_condition_id: str | None = None
    target_modules: tuple[str, ...] = TARGET_MODULES

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.condition_id):
            raise ValueError(f"unsafe condition_id: {self.condition_id!r}")
        if self.layer_band is LayerBand.EXPLICIT:
            raise ValueError("P0-D2 frozen matrix does not use explicit layer selections")
        if self.rank_multiplier <= 0 or self.alpha_multiplier <= 0:
            raise ValueError("rank/alpha multipliers must be positive")
        if not self.parameter_budget_regime:
            raise ValueError("parameter_budget_regime cannot be empty")
        if not self.target_modules or len(set(self.target_modules)) != len(
            self.target_modules
        ):
            raise ValueError("target_modules must be non-empty and unique")
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in self.target_modules):
            raise ValueError("target_modules contain an unsafe module name")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["layer_band"] = self.layer_band.value
        result["target_modules"] = list(self.target_modules)
        return result


@dataclass(frozen=True, slots=True)
class ResolvedBudgetCondition:
    condition_id: str
    layer_band: LayerBand
    selected_layers: tuple[int, ...]
    rank: int
    alpha: int
    target_modules: tuple[str, ...]
    parameter_budget_regime: str
    baseline_condition_id: str | None
    budget_reference_condition_id: str | None
    nominal_layer_rank_units: int
    nominal_budget_relative_error: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "layer_band": self.layer_band.value,
            "explicit_layers": [],
            "selected_layers": list(self.selected_layers),
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": list(self.target_modules),
            "parameter_budget_regime": self.parameter_budget_regime,
            "baseline_condition_id": self.baseline_condition_id,
            "budget_reference_condition_id": self.budget_reference_condition_id,
            "nominal_layer_rank_units": self.nominal_layer_rank_units,
            "nominal_budget_relative_error": self.nominal_budget_relative_error,
        }


def budget_match_conditions() -> tuple[BudgetCondition, ...]:
    return (
        BudgetCondition(
            "full-base",
            LayerBand.FULL,
            rank_multiplier=1,
            alpha_multiplier=1,
            parameter_budget_regime="reference_full",
        ),
        BudgetCondition(
            "early-base",
            LayerBand.EARLY,
            rank_multiplier=1,
            alpha_multiplier=1,
            parameter_budget_regime="fixed_local",
        ),
        BudgetCondition(
            "middle-base",
            LayerBand.MIDDLE,
            rank_multiplier=1,
            alpha_multiplier=1,
            parameter_budget_regime="fixed_local",
        ),
        BudgetCondition(
            "late-base",
            LayerBand.LATE,
            rank_multiplier=1,
            alpha_multiplier=1,
            parameter_budget_regime="fixed_local",
        ),
        BudgetCondition(
            "early-matched",
            LayerBand.EARLY,
            rank_multiplier=3,
            alpha_multiplier=3,
            parameter_budget_regime="matched_to_reference",
            baseline_condition_id="early-base",
            budget_reference_condition_id="full-base",
        ),
        BudgetCondition(
            "middle-matched",
            LayerBand.MIDDLE,
            rank_multiplier=3,
            alpha_multiplier=3,
            parameter_budget_regime="matched_to_reference",
            baseline_condition_id="middle-base",
            budget_reference_condition_id="full-base",
        ),
        BudgetCondition(
            "late-matched",
            LayerBand.LATE,
            rank_multiplier=3,
            alpha_multiplier=3,
            parameter_budget_regime="matched_to_reference",
            baseline_condition_id="late-base",
            budget_reference_condition_id="full-base",
        ),
    )


@dataclass(frozen=True, slots=True)
class P0D2Request:
    output_dir: Path
    source_manifest: Path
    stage: P0D2Stage = P0D2Stage.BUDGET_MATCH
    conditions: tuple[BudgetCondition, ...] = field(
        default_factory=budget_match_conditions
    )
    reference_condition_id: str = "full-base"
    retention_margin: float = 0.05
    budget_tolerance: float = 0.01

    def __post_init__(self) -> None:
        if self.stage is not P0D2Stage.BUDGET_MATCH:
            raise ValueError("P0-D2 only supports the frozen budget_match stage")
        if self.conditions != budget_match_conditions():
            raise ValueError("P0-D2 budget_match uses the frozen seven-condition matrix")
        if self.reference_condition_id != "full-base":
            raise ValueError("P0-D2 reference condition must be full-base")
        if not 0.0 <= self.retention_margin <= 1.0:
            raise ValueError("retention_margin must be between 0 and 1")
        if not 0.0 <= self.budget_tolerance <= 0.1:
            raise ValueError("budget_tolerance must be between 0 and 0.1")


@dataclass(frozen=True, slots=True)
class ResolvedP0D2Config:
    output_dir: Path
    stage: P0D2Stage
    code_sha256: str
    source_manifest_sha256: str
    source_run_id: str
    source_calibration_report_sha256: str
    selected_lesson_ids: tuple[str, ...]
    model_name: str
    model_revision: str
    use_4bit: bool
    base_rank: int
    base_alpha: int
    learning_rate: float
    max_steps: int
    max_length: int
    max_new_tokens: int
    training_seeds: tuple[int, ...]
    num_hidden_layers: int
    conditions: tuple[ResolvedBudgetCondition, ...]
    reference_condition_id: str
    retention_margin: float
    budget_tolerance: float

    def __post_init__(self) -> None:
        if not self.model_revision:
            raise ValueError("P0-D2 requires an immutable model revision")
        positive = {
            "base_rank": self.base_rank,
            "base_alpha": self.base_alpha,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "max_length": self.max_length,
            "max_new_tokens": self.max_new_tokens,
            "num_hidden_layers": self.num_hidden_layers,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if len(self.selected_lesson_ids) != 24:
            raise ValueError("P0-D2 requires the frozen 24-lesson P0-C cohort")
        if len(self.training_seeds) != 3 or len(set(self.training_seeds)) != 3:
            raise ValueError("P0-D2 requires three unique training seeds")
        if tuple(condition.condition_id for condition in self.conditions) != tuple(
            condition.condition_id for condition in budget_match_conditions()
        ):
            raise ValueError("P0-D2 resolved condition order differs from the frozen matrix")
        for condition in self.conditions:
            error = condition.nominal_budget_relative_error
            if (
                condition.parameter_budget_regime == "matched_to_reference"
                and (error is None or error > self.budget_tolerance)
            ):
                raise ValueError(
                    f"nominal budget mismatch for {condition.condition_id}: "
                    f"{error} > {self.budget_tolerance}"
                )

    @property
    def expected_unit_count(self) -> int:
        return (
            len(self.conditions)
            * len(self.selected_lesson_ids)
            * len(self.training_seeds)
        )

    @property
    def expected_probe_row_count(self) -> int:
        probes_per_arm = 26
        base_rows = len(self.selected_lesson_ids) * 2 * probes_per_arm
        adapter_rows = self.expected_unit_count * 2 * probes_per_arm
        return base_rows + adapter_rows

    def condition(self, condition_id: str) -> ResolvedBudgetCondition:
        for condition in self.conditions:
            if condition.condition_id == condition_id:
                return condition
        raise KeyError(condition_id)

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "p0d2-config-v1",
            "stage": self.stage.value,
            "code_sha256": self.code_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_run_id": self.source_run_id,
            "source_calibration_report_sha256": self.source_calibration_report_sha256,
            "selected_lesson_ids": list(self.selected_lesson_ids),
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "use_4bit": self.use_4bit,
            "base_rank": self.base_rank,
            "base_alpha": self.base_alpha,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "max_length": self.max_length,
            "max_new_tokens": self.max_new_tokens,
            "training_seeds": list(self.training_seeds),
            "num_hidden_layers": self.num_hidden_layers,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "reference_condition_id": self.reference_condition_id,
            "retention_margin": self.retention_margin,
            "budget_tolerance": self.budget_tolerance,
        }


def resolve_conditions(
    conditions: tuple[BudgetCondition, ...],
    *,
    num_hidden_layers: int,
    base_rank: int,
    base_alpha: int,
) -> tuple[ResolvedBudgetCondition, ...]:
    interim: list[dict[str, Any]] = []
    for condition in conditions:
        selected = select_layers(condition.layer_band, num_hidden_layers)
        selected_layers = tuple(
            range(num_hidden_layers) if selected is None else selected
        )
        rank = base_rank * condition.rank_multiplier
        alpha = base_alpha * condition.alpha_multiplier
        interim.append(
            {
                "source": condition,
                "selected_layers": selected_layers,
                "rank": rank,
                "alpha": alpha,
                "nominal_units": len(selected_layers) * rank,
            }
        )
    reference = next(
        value for value in interim if value["source"].condition_id == "full-base"
    )
    reference_units = int(reference["nominal_units"])
    resolved: list[ResolvedBudgetCondition] = []
    for value in interim:
        condition = value["source"]
        nominal_units = int(value["nominal_units"])
        error = (
            abs(nominal_units - reference_units) / reference_units
            if condition.budget_reference_condition_id
            else None
        )
        resolved.append(
            ResolvedBudgetCondition(
                condition_id=condition.condition_id,
                layer_band=condition.layer_band,
                selected_layers=value["selected_layers"],
                rank=int(value["rank"]),
                alpha=int(value["alpha"]),
                target_modules=condition.target_modules,
                parameter_budget_regime=condition.parameter_budget_regime,
                baseline_condition_id=condition.baseline_condition_id,
                budget_reference_condition_id=condition.budget_reference_condition_id,
                nominal_layer_rank_units=nominal_units,
                nominal_budget_relative_error=error,
            )
        )
    return tuple(resolved)
