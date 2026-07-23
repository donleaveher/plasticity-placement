from pathlib import Path

import pytest

from plasticity_placement.p0c.config import P0CConfig
from plasticity_placement.p0c.domain import Tier


def test_confirmatory_tier_requires_three_seeds(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        P0CConfig(
            output_dir=tmp_path,
            tier=Tier.CONFIRMATORY,
            training_seeds=(42,),
        )
    config = P0CConfig(
        output_dir=tmp_path,
        tier=Tier.CONFIRMATORY,
        training_seeds=(41, 42, 43),
        calibration_report_sha256="report-1",
    )
    assert config.target_lesson_count == 24


def test_pilot_tier_has_eight_lessons(tmp_path: Path) -> None:
    config = P0CConfig(
        output_dir=tmp_path,
        tier=Tier.PILOT,
        calibration_report_sha256="report-1",
    )
    assert config.target_lesson_count == 8
    assert config.to_dict()["training_seeds"] == [42]
    assert config.to_dict()["lesson_ids"] == []


def test_formal_tiers_reject_explicit_lesson_overrides(tmp_path: Path) -> None:
    for tier, seeds in (
        (Tier.PILOT, (42,)),
        (Tier.CONFIRMATORY, (41, 42, 43)),
    ):
        with pytest.raises(ValueError, match="restricted"):
            P0CConfig(
                output_dir=tmp_path,
                tier=tier,
                training_seeds=seeds,
                lesson_ids=("D_01",),
                calibration_report_sha256="report-1",
            )


def test_calibration_only_is_development_scoped(tmp_path: Path) -> None:
    config = P0CConfig(
        output_dir=tmp_path,
        tier=Tier.DEVELOPMENT,
        lesson_ids=("D_01",),
        calibration_only=True,
        base_results_cache=tmp_path / "cache",
    )
    assert config.to_dict()["base_results_cache"] == str(tmp_path / "cache")
    with pytest.raises(ValueError, match="development"):
        P0CConfig(
            output_dir=tmp_path,
            tier=Tier.SMOKE,
            calibration_only=True,
        )
