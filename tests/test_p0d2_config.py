from dataclasses import replace
from pathlib import Path

import pytest

from plasticity_placement.p0d2.config import (
    P0D2Request,
    ResolvedP0D2Config,
    budget_match_conditions,
    resolve_conditions,
)


def test_budget_match_matrix_resolves_condition_specific_hyperparameters() -> None:
    resolved = resolve_conditions(
        budget_match_conditions(),
        num_hidden_layers=24,
        base_rank=8,
        base_alpha=16,
    )
    by_id = {condition.condition_id: condition for condition in resolved}
    assert by_id["full-base"].selected_layers == tuple(range(24))
    assert by_id["full-base"].rank == 8
    assert by_id["early-base"].selected_layers == tuple(range(8))
    assert by_id["early-base"].rank == 8
    assert by_id["middle-matched"].selected_layers == tuple(range(8, 16))
    assert by_id["middle-matched"].rank == 24
    assert by_id["middle-matched"].alpha == 48
    assert by_id["late-matched"].nominal_budget_relative_error == 0.0


def test_budget_match_request_rejects_changed_matrix(tmp_path: Path) -> None:
    changed = list(budget_match_conditions())
    changed[-1] = replace(changed[-1], rank_multiplier=4)
    with pytest.raises(ValueError, match="frozen"):
        P0D2Request(
            output_dir=tmp_path / "out",
            source_manifest=tmp_path / "source.json",
            conditions=tuple(changed),
        )


def test_resolved_config_rejects_nominal_budget_mismatch(tmp_path: Path) -> None:
    conditions = resolve_conditions(
        budget_match_conditions(),
        num_hidden_layers=10,
        base_rank=8,
        base_alpha=16,
    )
    with pytest.raises(ValueError, match="nominal budget mismatch"):
        ResolvedP0D2Config(
            output_dir=tmp_path,
            stage=P0D2Request(
                output_dir=tmp_path,
                source_manifest=tmp_path / "source.json",
            ).stage,
            code_sha256="code",
            source_manifest_sha256="source",
            source_run_id="p0c",
            source_calibration_report_sha256="calibration",
            selected_lesson_ids=tuple(f"L{i}" for i in range(24)),
            model_name="model",
            model_revision="revision",
            use_4bit=False,
            base_rank=8,
            base_alpha=16,
            learning_rate=2e-4,
            max_steps=16,
            max_length=256,
            max_new_tokens=8,
            training_seeds=(41, 42, 43),
            num_hidden_layers=10,
            conditions=conditions,
            reference_condition_id="full-base",
            retention_margin=0.05,
            budget_tolerance=0.01,
        )
