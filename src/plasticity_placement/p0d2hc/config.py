from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.p0d2h.probes import PROBES_PER_LESSON

CALIBRATION_ARMS = (
    "no_write",
    "external",
    "answer_copy_oracle",
)
SOURCE_MODEL_ID = "source_model"
CANARY_MODEL_ID = "scale_canary"


@dataclass(frozen=True, slots=True)
class CalibrationModel:
    model_id: str
    role: str
    model_name: str
    model_revision: str
    use_4bit: bool

    def __post_init__(self) -> None:
        if self.model_id not in {SOURCE_MODEL_ID, CANARY_MODEL_ID}:
            raise ValueError(f"unsupported calibration model ID: {self.model_id}")
        if self.role not in {"source", "scale_canary"}:
            raise ValueError(f"unsupported calibration model role: {self.role}")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.model_id):
            raise ValueError(f"unsafe model ID: {self.model_id!r}")
        if not self.model_name or not self.model_revision:
            raise ValueError("calibration models require name and immutable revision")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class P0D2HCRequest:
    output_dir: Path
    source_manifest: Path
    canary_model_name: str | None = None
    canary_model_revision: str | None = None
    use_4bit: bool = True
    evaluation_max_length: int = 512
    oracle_min_accuracy: float = 0.90
    oracle_min_category_accuracy: float = 0.80
    oracle_max_invalid_rate: float = 0.01
    external_min_accuracy: float = 0.75
    external_max_invalid_rate: float = 0.05

    def __post_init__(self) -> None:
        if self.canary_model_revision and not self.canary_model_name:
            raise ValueError(
                "canary_model_revision requires canary_model_name"
            )
        if self.evaluation_max_length <= 0:
            raise ValueError("evaluation_max_length must be positive")
        for name, value in {
            "oracle_min_accuracy": self.oracle_min_accuracy,
            "oracle_min_category_accuracy": (
                self.oracle_min_category_accuracy
            ),
            "oracle_max_invalid_rate": self.oracle_max_invalid_rate,
            "external_min_accuracy": self.external_min_accuracy,
            "external_max_invalid_rate": self.external_max_invalid_rate,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        try:
            if self.output_dir.resolve() == self.source_manifest.parent.resolve():
                raise ValueError(
                    "P0-D2H-CAL output must be independent from its source"
                )
        except FileNotFoundError:
            pass


@dataclass(frozen=True, slots=True)
class ModelEvaluationConfig:
    output_dir: Path
    model_name: str
    model_revision: str
    use_4bit: bool
    max_length: int
    max_new_tokens: int
    training_seeds: tuple[int, ...] = (0,)


@dataclass(frozen=True, slots=True)
class ResolvedP0D2HCConfig:
    output_dir: Path
    source_output_dir: Path
    source_p0d2_output_dir: Path
    code_sha256: str
    source_manifest_sha256: str
    source_run_id: str
    source_summary_sha256: str
    source_p0d2_run_id: str
    source_compiler_hashes: dict[str, str]
    hard_probe_hashes: dict[str, str]
    selected_lesson_ids: tuple[str, ...]
    models: tuple[CalibrationModel, ...]
    evaluation_max_length: int
    max_new_tokens: int
    oracle_min_accuracy: float
    oracle_min_category_accuracy: float
    oracle_max_invalid_rate: float
    external_min_accuracy: float
    external_max_invalid_rate: float

    def __post_init__(self) -> None:
        if len(self.selected_lesson_ids) != 24:
            raise ValueError(
                "P0-D2H-CAL requires the frozen 24-lesson cohort"
            )
        if self.evaluation_max_length <= 0 or self.max_new_tokens <= 0:
            raise ValueError("calibration inference limits must be positive")
        model_ids = tuple(model.model_id for model in self.models)
        if not model_ids or model_ids[0] != SOURCE_MODEL_ID:
            raise ValueError("source_model must be the first calibration model")
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("calibration model IDs must be unique")
        if len(self.models) > 2:
            raise ValueError("P0-D2H-CAL supports at most one scale canary")

    @property
    def expected_unit_count(self) -> int:
        return len(self.models) * len(self.selected_lesson_ids)

    @property
    def expected_probe_row_count(self) -> int:
        return (
            self.expected_unit_count
            * len(CALIBRATION_ARMS)
            * PROBES_PER_LESSON
        )

    def model(self, model_id: str) -> CalibrationModel:
        for model in self.models:
            if model.model_id == model_id:
                return model
        raise KeyError(model_id)

    def evaluation_config(self, model_id: str) -> ModelEvaluationConfig:
        model = self.model(model_id)
        return ModelEvaluationConfig(
            output_dir=self.output_dir,
            model_name=model.model_name,
            model_revision=model.model_revision,
            use_4bit=model.use_4bit,
            max_length=self.evaluation_max_length,
            max_new_tokens=self.max_new_tokens,
        )

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "p0d2hc-config-v1",
            "stage": "hard_probe_calibration",
            "code_sha256": self.code_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_run_id": self.source_run_id,
            "source_summary_sha256": self.source_summary_sha256,
            "source_p0d2_run_id": self.source_p0d2_run_id,
            "source_compiler_hashes": self.source_compiler_hashes,
            "hard_probe_hashes": self.hard_probe_hashes,
            "selected_lesson_ids": list(self.selected_lesson_ids),
            "models": [model.to_dict() for model in self.models],
            "calibration_arms": list(CALIBRATION_ARMS),
            "evaluation_max_length": self.evaluation_max_length,
            "max_new_tokens": self.max_new_tokens,
            "oracle_min_accuracy": self.oracle_min_accuracy,
            "oracle_min_category_accuracy": (
                self.oracle_min_category_accuracy
            ),
            "oracle_max_invalid_rate": self.oracle_max_invalid_rate,
            "external_min_accuracy": self.external_min_accuracy,
            "external_max_invalid_rate": self.external_max_invalid_rate,
        }
