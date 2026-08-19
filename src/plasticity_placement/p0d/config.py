from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from plasticity_placement.training.config import LayerBand, select_layers


class P0DStage(StrEnum):
    BAND_SCAN = "band_scan"
    NARROW_SCAN = "narrow_scan"


@dataclass(frozen=True, slots=True)
class LayerCondition:
    condition_id: str
    layer_band: LayerBand
    explicit_layers: tuple[int, ...] = ()
    parameter_budget_regime: str = "fixed_local"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.condition_id):
            raise ValueError(f"unsafe condition_id: {self.condition_id!r}")
        if not self.parameter_budget_regime:
            raise ValueError("parameter_budget_regime cannot be empty")
        if self.layer_band is LayerBand.EXPLICIT and not self.explicit_layers:
            raise ValueError("explicit condition requires one or more layer indices")
        if self.layer_band is not LayerBand.EXPLICIT and self.explicit_layers:
            raise ValueError("explicit_layers are only valid for layer_band=explicit")
        if len(set(self.explicit_layers)) != len(self.explicit_layers):
            raise ValueError("explicit_layers must not contain duplicates")
        if any(layer < 0 for layer in self.explicit_layers):
            raise ValueError("explicit_layers must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["layer_band"] = self.layer_band.value
        result["explicit_layers"] = list(self.explicit_layers)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LayerCondition:
        return cls(
            condition_id=str(value["condition_id"]),
            layer_band=LayerBand(str(value["layer_band"])),
            explicit_layers=tuple(int(layer) for layer in value.get("explicit_layers", [])),
            parameter_budget_regime=str(
                value.get("parameter_budget_regime", "fixed_local")
            ),
        )


@dataclass(frozen=True, slots=True)
class ResolvedLayerCondition:
    condition_id: str
    layer_band: LayerBand
    explicit_layers: tuple[int, ...]
    selected_layers: tuple[int, ...]
    parameter_budget_regime: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "layer_band": self.layer_band.value,
            "explicit_layers": list(self.explicit_layers),
            "selected_layers": list(self.selected_layers),
            "parameter_budget_regime": self.parameter_budget_regime,
        }


def band_scan_conditions() -> tuple[LayerCondition, ...]:
    return (
        LayerCondition("full", LayerBand.FULL),
        LayerCondition("early", LayerBand.EARLY),
        LayerCondition("middle", LayerBand.MIDDLE),
        LayerCondition("late", LayerBand.LATE),
    )


@dataclass(frozen=True, slots=True)
class P0DRequest:
    output_dir: Path
    source_manifest: Path
    stage: P0DStage = P0DStage.BAND_SCAN
    conditions: tuple[LayerCondition, ...] = field(default_factory=band_scan_conditions)
    reference_condition_id: str = "full"
    retention_margin: float = 0.05
    conditions_config_sha256: str | None = None
    parent_band_run_id: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.retention_margin <= 1.0:
            raise ValueError("retention_margin must be between 0 and 1")
        if len(self.conditions) < 2:
            raise ValueError("P0-D requires a reference and at least one comparison condition")
        condition_ids = [condition.condition_id for condition in self.conditions]
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("condition IDs must be unique")
        if self.reference_condition_id not in condition_ids:
            raise ValueError("reference_condition_id is missing from the condition matrix")
        if self.stage is P0DStage.BAND_SCAN and self.conditions != band_scan_conditions():
            raise ValueError("band_scan uses the frozen full/early/middle/late matrix")
        if self.stage is P0DStage.BAND_SCAN and (
            self.conditions_config_sha256 is not None
            or self.parent_band_run_id is not None
        ):
            raise ValueError("band_scan cannot declare narrow-scan provenance")
        if self.stage is P0DStage.NARROW_SCAN:
            non_reference = [
                condition
                for condition in self.conditions
                if condition.condition_id != self.reference_condition_id
            ]
            if not non_reference or any(
                condition.layer_band is not LayerBand.EXPLICIT
                for condition in non_reference
            ):
                raise ValueError(
                    "narrow_scan comparisons must use explicit layer selections"
                )
            if not self.conditions_config_sha256 or not self.parent_band_run_id:
                raise ValueError(
                    "narrow_scan requires frozen conditions hash and parent band run ID"
                )


@dataclass(frozen=True, slots=True)
class ResolvedP0DConfig:
    output_dir: Path
    stage: P0DStage
    code_sha256: str
    source_manifest_sha256: str
    source_run_id: str
    source_calibration_report_sha256: str
    selected_lesson_ids: tuple[str, ...]
    model_name: str
    model_revision: str
    use_4bit: bool
    rank: int
    alpha: int
    learning_rate: float
    max_steps: int
    max_length: int
    max_new_tokens: int
    training_seeds: tuple[int, ...]
    target_modules: tuple[str, ...]
    num_hidden_layers: int
    conditions: tuple[ResolvedLayerCondition, ...]
    reference_condition_id: str
    retention_margin: float
    conditions_config_sha256: str | None
    parent_band_run_id: str | None

    def __post_init__(self) -> None:
        if not self.model_revision:
            raise ValueError("P0-D requires an immutable model revision")
        positive = {
            "rank": self.rank,
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "max_length": self.max_length,
            "max_new_tokens": self.max_new_tokens,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if len(self.selected_lesson_ids) != 24:
            raise ValueError("P0-D requires the frozen 24-lesson P0-C cohort")
        if len(self.training_seeds) != 3 or len(set(self.training_seeds)) != 3:
            raise ValueError("P0-D requires three unique training seeds")
        if not self.target_modules or len(set(self.target_modules)) != len(
            self.target_modules
        ):
            raise ValueError("target_modules must be non-empty and unique")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")

    @property
    def expected_unit_count(self) -> int:
        return (
            len(self.conditions)
            * len(self.selected_lesson_ids)
            * len(self.training_seeds)
        )

    def condition(self, condition_id: str) -> ResolvedLayerCondition:
        for condition in self.conditions:
            if condition.condition_id == condition_id:
                return condition
        raise KeyError(condition_id)

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "p0d-config-v1",
            "stage": self.stage.value,
            "code_sha256": self.code_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_run_id": self.source_run_id,
            "source_calibration_report_sha256": self.source_calibration_report_sha256,
            "selected_lesson_ids": list(self.selected_lesson_ids),
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "use_4bit": self.use_4bit,
            "rank": self.rank,
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "max_length": self.max_length,
            "max_new_tokens": self.max_new_tokens,
            "training_seeds": list(self.training_seeds),
            "target_modules": list(self.target_modules),
            "num_hidden_layers": self.num_hidden_layers,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "reference_condition_id": self.reference_condition_id,
            "retention_margin": self.retention_margin,
            "conditions_config_sha256": self.conditions_config_sha256,
            "parent_band_run_id": self.parent_band_run_id,
        }


def resolve_conditions(
    conditions: tuple[LayerCondition, ...],
    num_hidden_layers: int,
) -> tuple[ResolvedLayerCondition, ...]:
    resolved: list[ResolvedLayerCondition] = []
    for condition in conditions:
        selected = select_layers(
            condition.layer_band,
            num_hidden_layers,
            condition.explicit_layers,
        )
        resolved.append(
            ResolvedLayerCondition(
                condition_id=condition.condition_id,
                layer_band=condition.layer_band,
                explicit_layers=condition.explicit_layers,
                selected_layers=tuple(
                    range(num_hidden_layers) if selected is None else selected
                ),
                parameter_budget_regime=condition.parameter_budget_regime,
            )
        )
    return tuple(resolved)
