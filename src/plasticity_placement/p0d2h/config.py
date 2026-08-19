from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.p0d2h.probes import PROBES_PER_LESSON
from plasticity_placement.p0d2h.prompting import (
    EXTERNAL_PROMPT_RENDERER_VERSION,
)

STRESS_CONDITION_IDS = (
    "full-base",
    "early-matched",
    "middle-matched",
    "late-matched",
)


@dataclass(frozen=True, slots=True)
class StressCondition:
    condition_id: str
    layer_band: str
    selected_layers: tuple[int, ...]
    rank: int
    alpha: int
    target_modules: tuple[str, ...]
    trainable_parameters: int

    def __post_init__(self) -> None:
        if self.condition_id not in STRESS_CONDITION_IDS:
            raise ValueError(f"unsupported P0-D2H condition: {self.condition_id}")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.condition_id):
            raise ValueError(f"unsafe condition ID: {self.condition_id!r}")
        if not self.selected_layers or len(set(self.selected_layers)) != len(self.selected_layers):
            raise ValueError("selected_layers must be non-empty and unique")
        if min(self.selected_layers) < 0:
            raise ValueError("selected_layers cannot contain negative indices")
        if self.rank <= 0 or self.alpha <= 0 or self.trainable_parameters <= 0:
            raise ValueError("rank, alpha, and trainable_parameters must be positive")
        if not self.target_modules or len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("target_modules must be non-empty and unique")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["selected_layers"] = list(self.selected_layers)
        value["target_modules"] = list(self.target_modules)
        return value


@dataclass(frozen=True, slots=True)
class P0D2HRequest:
    output_dir: Path
    source_manifest: Path
    resilience_margin: float = 0.05
    evaluation_max_length: int = 512
    external_anchor_min_accuracy: float = 0.75
    external_anchor_max_invalid_rate: float = 0.05
    common_floor_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 <= self.resilience_margin <= 1.0:
            raise ValueError("resilience_margin must be between 0 and 1")
        if self.evaluation_max_length <= 0:
            raise ValueError("evaluation_max_length must be positive")
        if not 0.0 <= self.external_anchor_min_accuracy <= 1.0:
            raise ValueError("external_anchor_min_accuracy must be between 0 and 1")
        if not 0.0 <= self.external_anchor_max_invalid_rate <= 1.0:
            raise ValueError("external_anchor_max_invalid_rate must be between 0 and 1")
        if not 0.0 <= self.common_floor_tolerance <= 0.25:
            raise ValueError("common_floor_tolerance must be between 0 and 0.25")
        try:
            if self.output_dir.resolve() == self.source_manifest.parent.resolve():
                raise ValueError("P0-D2H output must be independent from the source P0-D2 run")
        except FileNotFoundError:
            pass


@dataclass(frozen=True, slots=True)
class ResolvedP0D2HConfig:
    output_dir: Path
    source_output_dir: Path
    code_sha256: str
    source_manifest_sha256: str
    source_run_id: str
    source_summary_sha256: str
    source_compiler_hashes: dict[str, str]
    selected_lesson_ids: tuple[str, ...]
    model_name: str
    model_revision: str
    use_4bit: bool
    source_training_max_length: int
    evaluation_max_length: int
    max_new_tokens: int
    training_seeds: tuple[int, ...]
    conditions: tuple[StressCondition, ...]
    hard_probe_hashes: dict[str, str]
    resilience_margin: float
    external_anchor_min_accuracy: float
    external_anchor_max_invalid_rate: float
    common_floor_tolerance: float

    def __post_init__(self) -> None:
        if not self.model_revision:
            raise ValueError("P0-D2H requires an immutable model revision")
        if (
            self.source_training_max_length <= 0
            or self.evaluation_max_length <= 0
            or self.max_new_tokens <= 0
        ):
            raise ValueError("inference length limits must be positive")
        if self.evaluation_max_length < self.source_training_max_length:
            raise ValueError("evaluation_max_length cannot be shorter than source training")
        if len(self.selected_lesson_ids) != 24:
            raise ValueError("P0-D2H requires the frozen 24-lesson cohort")
        if len(self.training_seeds) != 3 or len(set(self.training_seeds)) != 3:
            raise ValueError("P0-D2H requires three unique training seeds")
        if tuple(condition.condition_id for condition in self.conditions) != (STRESS_CONDITION_IDS):
            raise ValueError("P0-D2H condition order differs from the frozen matrix")
        parameter_counts = {condition.trainable_parameters for condition in self.conditions}
        if len(parameter_counts) != 1:
            raise ValueError("P0-D2H source conditions do not have equal actual parameter budgets")

    @property
    def max_length(self) -> int:
        """Compatibility surface consumed by the shared P0-C evaluator."""
        return self.evaluation_max_length

    @property
    def expected_unit_count(self) -> int:
        return len(self.conditions) * len(self.selected_lesson_ids) * len(self.training_seeds)

    @property
    def expected_probe_row_count(self) -> int:
        base_rows = len(self.selected_lesson_ids) * 2 * PROBES_PER_LESSON
        adapter_rows = self.expected_unit_count * PROBES_PER_LESSON
        return base_rows + adapter_rows

    def condition(self, condition_id: str) -> StressCondition:
        for condition in self.conditions:
            if condition.condition_id == condition_id:
                return condition
        raise KeyError(condition_id)

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "p0d2h-config-v2",
            "stage": "hard_probe",
            "code_sha256": self.code_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_run_id": self.source_run_id,
            "source_summary_sha256": self.source_summary_sha256,
            "source_compiler_hashes": self.source_compiler_hashes,
            "selected_lesson_ids": list(self.selected_lesson_ids),
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "use_4bit": self.use_4bit,
            "source_training_max_length": self.source_training_max_length,
            "evaluation_max_length": self.evaluation_max_length,
            "max_new_tokens": self.max_new_tokens,
            "training_seeds": list(self.training_seeds),
            "conditions": [condition.to_dict() for condition in self.conditions],
            "hard_probe_hashes": self.hard_probe_hashes,
            "resilience_margin": self.resilience_margin,
            "external_anchor_min_accuracy": (self.external_anchor_min_accuracy),
            "external_anchor_max_invalid_rate": (self.external_anchor_max_invalid_rate),
            "common_floor_tolerance": self.common_floor_tolerance,
            "external_prompt_renderer_version": (EXTERNAL_PROMPT_RENDERER_VERSION),
        }
