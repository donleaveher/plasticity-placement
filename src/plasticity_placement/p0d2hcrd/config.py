from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.p0d2hc.config import ModelEvaluationConfig

CRD_SCHEMA_VERSION = "p0d2hcrd-config-v1"
CRD_STAGE = "hard_probe_route_decomposition"
CANDIDATE_SCORING_VERSION = "p0d2hcrd-variable-candidate-sum-logprob-v1"
PRIMARY_SCORE_DEFINITION = "candidate_token_sum_logprob"
CANONICAL_LEADING_WHITESPACE = ""
EVALUATION_MAX_LENGTH = 512
ENDPOINTS = ("route_only", "retrieval_only", "combined")
ROWS_PER_LESSON = {
    "route_only": 16,
    "retrieval_only": 16,
    "combined": 32,
}
CANDIDATES_PER_ENDPOINT = {
    "route_only": 2,
    "retrieval_only": 4,
    "combined": 4,
}

ROUTE_MIN_ACCURACY = 0.90
RETRIEVAL_MIN_ACCURACY = 0.90
COMBINED_MIN_ACCURACY = 0.75
MAX_TIE_ERROR_NONFINITE_RATE = 0.0

EXPECTED_SOURCE_RUN_ID = "p0d2hfc-forced-choice-f604efe6bf"
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "89ba71ea0fde052089329e630d23c1a86652bc47a129de1921786c8e0b4dd214"
)
EXPECTED_SOURCE_CANDIDATE_AUDIT_SHA256 = (
    "d140cfb2ac0c57b4819d3793b59f5f7c767fe428336be856fc9299460731b230"
)
EXPECTED_SOURCE_RAW_TREE_SHA256 = (
    "dfe9b9014966f484e988a2f6d9f8e18d4ee7a1b5f06d671ae2e973cef8a0b24a"
)
EXPECTED_SOURCE_SUMMARY_SHA256 = (
    "24b1fe51bf400e5c1e0b15aa68d67b0db913591cce082aeac13b9396854d22e1"
)
EXPECTED_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
EXPECTED_MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


@dataclass(frozen=True, slots=True)
class CRDModel:
    model_id: str
    role: str
    model_name: str
    model_revision: str
    use_4bit: bool

    def __post_init__(self) -> None:
        if self.model_id != "scale_canary" or self.role != "scale_canary":
            raise ValueError("P0-D2H-CRD requires only the frozen scale canary")
        if (
            self.model_name != EXPECTED_MODEL_NAME
            or self.model_revision != EXPECTED_MODEL_REVISION
        ):
            raise ValueError("P0-D2H-CRD scale-canary identity changed")
        if not self.use_4bit:
            raise ValueError("P0-D2H-CRD freezes source-compatible 4-bit evaluation")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class P0D2HCRDRequest:
    output_dir: Path
    source_manifest: Path

    def __post_init__(self) -> None:
        try:
            output = self.output_dir.resolve()
            source = self.source_manifest.parent.resolve()
        except FileNotFoundError:
            return
        if (
            output == source
            or output.is_relative_to(source)
            or source.is_relative_to(output)
        ):
            raise ValueError(
                "P0-D2H-CRD output must be independent from P0-D2H-CAL-FC"
            )


@dataclass(frozen=True, slots=True)
class ResolvedP0D2HCRDConfig:
    output_dir: Path
    source_output_dir: Path
    source_calibration_output_dir: Path
    source_hard_probe_output_dir: Path
    source_compiled_output_dir: Path
    code_sha256: str
    source_manifest_sha256: str
    source_run_id: str
    source_summary_sha256: str
    source_candidate_token_audit_sha256: str
    source_raw_results_sha256: str
    source_calibration_manifest_sha256: str
    source_calibration_raw_results_sha256: str
    source_hard_probe_manifest_sha256: str
    hard_probe_hashes: dict[str, str]
    compiler_hashes: dict[str, str]
    selected_lesson_ids: tuple[str, ...]
    decomposition_bank_sha256: str
    model: CRDModel
    prompt_renderers: dict[str, str]
    evaluation_max_length: int = EVALUATION_MAX_LENGTH

    def __post_init__(self) -> None:
        if len(self.selected_lesson_ids) != 24:
            raise ValueError("P0-D2H-CRD requires the frozen 24-lesson cohort")
        if self.evaluation_max_length != EVALUATION_MAX_LENGTH:
            raise ValueError(
                f"P0-D2H-CRD evaluation length is frozen at "
                f"{EVALUATION_MAX_LENGTH}"
            )
        if tuple(self.prompt_renderers) != ENDPOINTS:
            raise ValueError("P0-D2H-CRD endpoint renderer order changed")
        if not self.decomposition_bank_sha256:
            raise ValueError("P0-D2H-CRD requires an immutable probe-bank hash")

    @property
    def expected_unit_count(self) -> int:
        return len(self.selected_lesson_ids)

    @property
    def expected_decision_row_count(self) -> int:
        return len(self.selected_lesson_ids) * sum(ROWS_PER_LESSON.values())

    @property
    def expected_candidate_sequence_count(self) -> int:
        per_lesson = sum(
            ROWS_PER_LESSON[endpoint] * CANDIDATES_PER_ENDPOINT[endpoint]
            for endpoint in ENDPOINTS
        )
        return len(self.selected_lesson_ids) * per_lesson

    def evaluation_config(self) -> ModelEvaluationConfig:
        return ModelEvaluationConfig(
            output_dir=self.output_dir,
            model_name=self.model.model_name,
            model_revision=self.model.model_revision,
            use_4bit=self.model.use_4bit,
            max_length=self.evaluation_max_length,
            max_new_tokens=1,
        )

    def gate_thresholds(self) -> dict[str, float]:
        return {
            "route_min_accuracy": ROUTE_MIN_ACCURACY,
            "retrieval_min_accuracy": RETRIEVAL_MIN_ACCURACY,
            "combined_min_accuracy": COMBINED_MIN_ACCURACY,
            "max_tie_error_nonfinite_rate": MAX_TIE_ERROR_NONFINITE_RATE,
        }

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CRD_SCHEMA_VERSION,
            "stage": CRD_STAGE,
            "code_sha256": self.code_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_run_id": self.source_run_id,
            "source_summary_sha256": self.source_summary_sha256,
            "source_candidate_token_audit_sha256": (
                self.source_candidate_token_audit_sha256
            ),
            "source_raw_results_sha256": self.source_raw_results_sha256,
            "source_calibration_manifest_sha256": (
                self.source_calibration_manifest_sha256
            ),
            "source_calibration_raw_results_sha256": (
                self.source_calibration_raw_results_sha256
            ),
            "source_hard_probe_manifest_sha256": (
                self.source_hard_probe_manifest_sha256
            ),
            "hard_probe_hashes": self.hard_probe_hashes,
            "compiler_hashes": self.compiler_hashes,
            "selected_lesson_ids": list(self.selected_lesson_ids),
            "decomposition_bank_sha256": self.decomposition_bank_sha256,
            "model": self.model.to_dict(),
            "endpoints": list(ENDPOINTS),
            "rows_per_lesson": dict(ROWS_PER_LESSON),
            "candidates_per_endpoint": dict(CANDIDATES_PER_ENDPOINT),
            "prompt_renderers": self.prompt_renderers,
            "evaluation_max_length": self.evaluation_max_length,
            "candidate_scoring_version": CANDIDATE_SCORING_VERSION,
            "primary_score_definition": PRIMARY_SCORE_DEFINITION,
            "canonical_leading_whitespace": CANONICAL_LEADING_WHITESPACE,
            "gate_thresholds": self.gate_thresholds(),
            "training_complexity_review_eligible": False,
            "automatic_training_started": False,
            "automatic_narrow_scan_started": False,
        }
