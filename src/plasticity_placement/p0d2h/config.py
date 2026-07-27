from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.p0d2h.probes import PROBES_PER_LESSON

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
        if not self.selected_layers or len(set(self.selected_layers)) != len(
            self.selected_layers
        ):
            raise ValueError("selected_layers must be non-empty and unique")
        if min(self.selected_layers) < 0:
            raise ValueError("selected_layers cannot contain negative indices")
        if self.rank <= 0 or self.alpha <= 0 or self.trainable_parameters <= 0:
            raise ValueError("rank, alpha, and trainable_parameters must be positive")
        if not self.target_modules or len(set(self.target_modules)) != len(
            self.target_modules
        ):
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

    def __post_init__(self) -> None:
        if not 0.0 <= self.resilience_margin <= 1.0:
            raise ValueError("resilience_margin must be between 0 and 1")
        try:
            if self.output_dir.resolve() == self.source_manifest.parent.resolve():
                raise ValueError(
                    "P0-D2H output must be independent from the source P0-D2 run"
                )
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
    max_length: int
    max_new_tokens: int
    training_seeds: tuple[int, ...]
    conditions: tuple[StressCondition, ...]
    hard_probe_hashes: dict[str, str]
    resilience_margin: float

    def __post_init__(self) -> None:
        if not self.model_revision:
            raise ValueError("P0-D2H requires an immutable model revision")
        if self.max_length <= 0 or self.max_new_tokens <= 0:
            raise ValueError("inference length limits must be positive")
        if len(self.selected_lesson_ids) != 24:
            raise ValueError("P0-D2H requires the frozen 24-lesson cohort")
        if len(self.training_seeds) != 3 or len(set(self.training_seeds)) != 3:
            raise ValueError("P0-D2H requires three unique training seeds")
        if tuple(condition.condition_id for condition in self.conditions) != (
            STRESS_CONDITION_IDS
        ):
            raise ValueError("P0-D2H condition order differs from the frozen matrix")
        parameter_counts = {
            condition.trainable_parameters for condition in self.conditions
        }
        if len(parameter_counts) != 1:
            raise ValueError(
                "P0-D2H source conditions do not have equal actual parameter budgets"
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
            "schema_version": "p0d2h-config-v1",
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
            "max_length": self.max_length,
            "max_new_tokens": self.max_new_tokens,
            "training_seeds": list(self.training_seeds),
            "conditions": [condition.to_dict() for condition in self.conditions],
            "hard_probe_hashes": self.hard_probe_hashes,
            "resilience_margin": self.resilience_margin,
        }
