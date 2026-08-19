from __future__ import annotations

from pathlib import Path

import pytest

from plasticity_placement.p0d2h.config import (
    STRESS_CONDITION_IDS,
    P0D2HRequest,
    ResolvedP0D2HConfig,
    StressCondition,
)


def _condition(condition_id: str, parameters: int = 100) -> StressCondition:
    return StressCondition(
        condition_id=condition_id,
        layer_band="full" if condition_id == "full-base" else "late",
        selected_layers=tuple(range(8)),
        rank=4,
        alpha=8,
        target_modules=("q_proj", "v_proj"),
        trainable_parameters=parameters,
    )


def test_p0d2h_config_freezes_equal_budget_four_condition_matrix(
    tmp_path: Path,
) -> None:
    config = ResolvedP0D2HConfig(
        output_dir=tmp_path / "out",
        source_output_dir=tmp_path / "source",
        code_sha256="code",
        source_manifest_sha256="manifest",
        source_run_id="p0d2-run",
        source_summary_sha256="summary",
        source_compiler_hashes={"compiler_version": "v4"},
        selected_lesson_ids=tuple(f"L{index}" for index in range(24)),
        model_name="model",
        model_revision="revision",
        use_4bit=False,
        source_training_max_length=256,
        evaluation_max_length=512,
        max_new_tokens=8,
        training_seeds=(41, 42, 43),
        conditions=tuple(_condition(condition_id) for condition_id in STRESS_CONDITION_IDS),
        hard_probe_hashes={"compiler_version": "hard-v1"},
        resilience_margin=0.05,
        external_anchor_min_accuracy=0.75,
        external_anchor_max_invalid_rate=0.05,
        common_floor_tolerance=0.05,
    )
    assert config.expected_unit_count == 288
    assert config.expected_probe_row_count == 5_376
    assert config.max_length == 512
    assert config.identity_dict()["schema_version"] == "p0d2h-config-v2"
    assert config.identity_dict()["source_training_max_length"] == 256
    assert config.identity_dict()["evaluation_max_length"] == 512


def test_p0d2h_rejects_source_output_collision_and_mixed_budgets(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    with pytest.raises(ValueError, match="independent"):
        P0D2HRequest(
            output_dir=source,
            source_manifest=source / "manifest.json",
        )
    conditions = [_condition(condition_id) for condition_id in STRESS_CONDITION_IDS]
    conditions[-1] = _condition("late-matched", parameters=101)
    with pytest.raises(ValueError, match="equal actual parameter"):
        ResolvedP0D2HConfig(
            output_dir=tmp_path / "out",
            source_output_dir=source,
            code_sha256="code",
            source_manifest_sha256="manifest",
            source_run_id="p0d2-run",
            source_summary_sha256="summary",
            source_compiler_hashes={"compiler_version": "v4"},
            selected_lesson_ids=tuple(f"L{index}" for index in range(24)),
            model_name="model",
            model_revision="revision",
            use_4bit=False,
            source_training_max_length=256,
            evaluation_max_length=512,
            max_new_tokens=8,
            training_seeds=(41, 42, 43),
            conditions=tuple(conditions),
            hard_probe_hashes={"compiler_version": "hard-v1"},
            resilience_margin=0.05,
            external_anchor_min_accuracy=0.75,
            external_anchor_max_invalid_rate=0.05,
            common_floor_tolerance=0.05,
        )


def test_p0d2h_rejects_short_repaired_evaluation_context(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="cannot be shorter"):
        ResolvedP0D2HConfig(
            output_dir=tmp_path / "out",
            source_output_dir=tmp_path / "source",
            code_sha256="code",
            source_manifest_sha256="manifest",
            source_run_id="p0d2-run",
            source_summary_sha256="summary",
            source_compiler_hashes={"compiler_version": "v4"},
            selected_lesson_ids=tuple(f"L{index}" for index in range(24)),
            model_name="model",
            model_revision="revision",
            use_4bit=False,
            source_training_max_length=256,
            evaluation_max_length=128,
            max_new_tokens=8,
            training_seeds=(41, 42, 43),
            conditions=tuple(_condition(condition_id) for condition_id in STRESS_CONDITION_IDS),
            hard_probe_hashes={"compiler_version": "hard-v2"},
            resilience_margin=0.05,
            external_anchor_min_accuracy=0.75,
            external_anchor_max_invalid_rate=0.05,
            common_floor_tolerance=0.05,
        )
