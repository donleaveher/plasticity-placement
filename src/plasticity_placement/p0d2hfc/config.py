from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.compiler import ACTIONS
from plasticity_placement.p0d2h.probes import HARD_CATEGORIES, PROBES_PER_LESSON
from plasticity_placement.p0d2hc.config import CALIBRATION_ARMS, ModelEvaluationConfig

FORCED_CHOICE_SCHEMA_VERSION = "p0d2hfc-config-v1"
FORCED_CHOICE_STAGE = "hard_probe_forced_choice"
CANDIDATE_SCORING_VERSION = "p0d2hfc-full-string-sum-logprob-v1"
PRIMARY_SCORE_DEFINITION = "candidate_token_sum_logprob"
CANONICAL_LEADING_WHITESPACE = ""
EVALUATION_MAX_LENGTH = 512

ORACLE_MIN_ACCURACY = 0.95
ORACLE_MIN_CATEGORY_ACCURACY = 0.90
EXTERNAL_MIN_ACCURACY = 0.85
EXTERNAL_MIN_CATEGORY_ACCURACY = 0.75
NO_WRITE_MAX_ACCURACY = 0.35
MAX_TIE_ERROR_NONFINITE_RATE = 0.0


@dataclass(frozen=True, slots=True)
class ForcedChoiceModel:
    model_id: str
    role: str
    model_name: str
    model_revision: str
    use_4bit: bool

    def __post_init__(self) -> None:
        if self.model_id not in {"source_model", "scale_canary"}:
            raise ValueError(f"unsupported forced-choice model ID: {self.model_id}")
        if self.role not in {"source", "scale_canary"}:
            raise ValueError(f"unsupported forced-choice model role: {self.role}")
        if not self.model_name or not self.model_revision:
            raise ValueError("forced-choice models require immutable identity")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class P0D2HFCRequest:
    output_dir: Path
    source_manifest: Path

    def __post_init__(self) -> None:
        source_dir = self.source_manifest.parent
        try:
            output = self.output_dir.resolve()
            source = source_dir.resolve()
        except FileNotFoundError:
            return
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(
                "P0-D2H-CAL-FC output must be independent from P0-D2H-CAL"
            )


@dataclass(frozen=True, slots=True)
class ResolvedP0D2HFCConfig:
    output_dir: Path
    source_output_dir: Path
    source_p0d2h_output_dir: Path
    code_sha256: str
    source_manifest_sha256: str
    source_run_id: str
    source_summary_sha256: str
    source_prompt_token_audit_sha256: str
    source_raw_results_sha256: str
    source_p0d2h_run_id: str
    source_p0d2h_manifest_sha256: str
    hard_probe_hashes: dict[str, str]
    selected_lesson_ids: tuple[str, ...]
    models: tuple[ForcedChoiceModel, ...]
    prompt_renderers: dict[str, str]
    evaluation_max_length: int = EVALUATION_MAX_LENGTH

    def __post_init__(self) -> None:
        if len(self.selected_lesson_ids) != 24:
            raise ValueError("P0-D2H-CAL-FC requires the frozen 24-lesson cohort")
        if self.evaluation_max_length != EVALUATION_MAX_LENGTH:
            raise ValueError(
                f"forced-choice evaluation length is frozen at {EVALUATION_MAX_LENGTH}"
            )
        if tuple(model.model_id for model in self.models) != (
            "source_model",
            "scale_canary",
        ):
            raise ValueError("forced-choice calibration requires both frozen models")
        if tuple(model.role for model in self.models) != ("source", "scale_canary"):
            raise ValueError("forced-choice model roles/order changed")

    @property
    def expected_unit_count(self) -> int:
        return len(self.models) * len(self.selected_lesson_ids)

    @property
    def expected_decision_row_count(self) -> int:
        return (
            self.expected_unit_count
            * len(CALIBRATION_ARMS)
            * PROBES_PER_LESSON
        )

    def model(self, model_id: str) -> ForcedChoiceModel:
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
            max_new_tokens=1,
        )

    def gate_thresholds(self) -> dict[str, float]:
        return {
            "oracle_min_accuracy": ORACLE_MIN_ACCURACY,
            "oracle_min_category_accuracy": ORACLE_MIN_CATEGORY_ACCURACY,
            "external_min_accuracy": EXTERNAL_MIN_ACCURACY,
            "external_min_category_accuracy": EXTERNAL_MIN_CATEGORY_ACCURACY,
            "no_write_max_accuracy": NO_WRITE_MAX_ACCURACY,
            "max_tie_error_nonfinite_rate": MAX_TIE_ERROR_NONFINITE_RATE,
            "external_minus_no_write_ci95_lower_strictly_greater_than": 0.0,
        }

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FORCED_CHOICE_SCHEMA_VERSION,
            "stage": FORCED_CHOICE_STAGE,
            "code_sha256": self.code_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_run_id": self.source_run_id,
            "source_summary_sha256": self.source_summary_sha256,
            "source_prompt_token_audit_sha256": (
                self.source_prompt_token_audit_sha256
            ),
            "source_raw_results_sha256": self.source_raw_results_sha256,
            "source_p0d2h_run_id": self.source_p0d2h_run_id,
            "source_p0d2h_manifest_sha256": self.source_p0d2h_manifest_sha256,
            "hard_probe_hashes": self.hard_probe_hashes,
            "selected_lesson_ids": list(self.selected_lesson_ids),
            "models": [model.to_dict() for model in self.models],
            "calibration_arms": list(CALIBRATION_ARMS),
            "hard_categories": list(HARD_CATEGORIES),
            "ordered_action_universe": list(ACTIONS),
            "prompt_renderers": self.prompt_renderers,
            "evaluation_max_length": self.evaluation_max_length,
            "candidate_scoring_version": CANDIDATE_SCORING_VERSION,
            "primary_score_definition": PRIMARY_SCORE_DEFINITION,
            "canonical_leading_whitespace": CANONICAL_LEADING_WHITESPACE,
            "gate_thresholds": self.gate_thresholds(),
            "automatic_training_started": False,
            "automatic_narrow_scan_started": False,
        }
