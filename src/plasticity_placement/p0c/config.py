from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.domain import Tier


@dataclass(frozen=True, slots=True)
class P0CConfig:
    output_dir: Path
    tier: Tier = Tier.SMOKE
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_revision: str | None = None
    use_4bit: bool = False
    rank: int = 8
    alpha: int = 16
    learning_rate: float = 2e-4
    max_steps: int = 16
    max_length: int = 256
    max_new_tokens: int = 8
    training_seeds: tuple[int, ...] = (42,)
    lesson_ids: tuple[str, ...] = ()
    screening_threshold: float = 0.5
    screening_invalid_threshold: float = 0.5
    overwrite_compiled: bool = False
    calibration_only: bool = False
    base_results_cache: Path | None = None
    calibration_report_sha256: str | None = None

    def __post_init__(self) -> None:
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
        if not self.training_seeds:
            raise ValueError("training_seeds cannot be empty")
        if len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("training_seeds must be unique")
        if len(set(self.lesson_ids)) != len(self.lesson_ids):
            raise ValueError("lesson_ids must be unique")
        if self.lesson_ids and self.tier in {Tier.PILOT, Tier.CONFIRMATORY}:
            raise ValueError("lesson_ids overrides are restricted to development/smoke tiers")
        if self.calibration_only and self.tier is not Tier.DEVELOPMENT:
            raise ValueError("calibration_only is restricted to the development tier")
        if self.base_results_cache is not None and not self.calibration_only:
            raise ValueError("base_results_cache requires calibration_only")
        if self.tier in {Tier.PILOT, Tier.CONFIRMATORY} and not self.calibration_report_sha256:
            raise ValueError(f"tier={self.tier.value} requires calibration report provenance")
        if (
            self.tier not in {Tier.PILOT, Tier.CONFIRMATORY}
            and self.calibration_report_sha256 is not None
        ):
            raise ValueError("calibration_report_sha256 is only valid for pilot/confirmatory")
        for name, value in (
            ("screening_threshold", self.screening_threshold),
            ("screening_invalid_threshold", self.screening_invalid_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        expected_seeds = 3 if self.tier is Tier.CONFIRMATORY else 1
        if len(self.training_seeds) != expected_seeds:
            raise ValueError(f"tier={self.tier.value} requires {expected_seeds} training seed(s)")

    @property
    def target_lesson_count(self) -> int:
        return {
            Tier.DEVELOPMENT: 6,
            Tier.SMOKE: 2,
            Tier.PILOT: 8,
            Tier.CONFIRMATORY: 24,
        }[self.tier]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["output_dir"] = str(self.output_dir)
        result["tier"] = self.tier.value
        result["training_seeds"] = list(self.training_seeds)
        result["lesson_ids"] = list(self.lesson_ids)
        result["base_results_cache"] = (
            str(self.base_results_cache) if self.base_results_cache else None
        )
        return result
