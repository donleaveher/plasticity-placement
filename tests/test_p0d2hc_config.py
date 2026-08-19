from __future__ import annotations

from pathlib import Path

import pytest

from plasticity_placement.p0d2hc.config import (
    CANARY_MODEL_ID,
    SOURCE_MODEL_ID,
    CalibrationModel,
    P0D2HCRequest,
    ResolvedP0D2HCConfig,
)


def _model(model_id: str) -> CalibrationModel:
    return CalibrationModel(
        model_id=model_id,
        role="source" if model_id == SOURCE_MODEL_ID else "scale_canary",
        model_name=f"model-{model_id}",
        model_revision=f"revision-{model_id}",
        use_4bit=True,
    )


def _config(tmp_path: Path, model_count: int) -> ResolvedP0D2HCConfig:
    models = [_model(SOURCE_MODEL_ID)]
    if model_count == 2:
        models.append(_model(CANARY_MODEL_ID))
    return ResolvedP0D2HCConfig(
        output_dir=tmp_path / "out",
        source_output_dir=tmp_path / "hard",
        source_p0d2_output_dir=tmp_path / "p0d2",
        code_sha256="code",
        source_manifest_sha256="manifest",
        source_run_id="hard-run",
        source_summary_sha256="summary",
        source_p0d2_run_id="p0d2-run",
        source_compiler_hashes={"compiler_version": "v4"},
        hard_probe_hashes={"hard_probes_sha256": "hard"},
        selected_lesson_ids=tuple(f"L{index:02d}" for index in range(24)),
        models=tuple(models),
        evaluation_max_length=512,
        max_new_tokens=8,
        oracle_min_accuracy=0.90,
        oracle_min_category_accuracy=0.80,
        oracle_max_invalid_rate=0.01,
        external_min_accuracy=0.75,
        external_max_invalid_rate=0.05,
    )


def test_p0d2hc_freezes_one_or_two_model_three_arm_matrix(
    tmp_path: Path,
) -> None:
    source_only = _config(tmp_path, 1)
    with_canary = _config(tmp_path, 2)
    assert source_only.expected_unit_count == 24
    assert source_only.expected_probe_row_count == 1_152
    assert with_canary.expected_unit_count == 48
    assert with_canary.expected_probe_row_count == 2_304
    assert with_canary.identity_dict()["schema_version"] == "p0d2hc-config-v1"
    assert with_canary.evaluation_config(CANARY_MODEL_ID).max_length == 512


def test_p0d2hc_rejects_source_collision_and_orphan_canary_revision(
    tmp_path: Path,
) -> None:
    source = tmp_path / "hard"
    with pytest.raises(ValueError, match="independent"):
        P0D2HCRequest(
            output_dir=source,
            source_manifest=source / "manifest.json",
        )
    with pytest.raises(ValueError, match="requires canary_model_name"):
        P0D2HCRequest(
            output_dir=tmp_path / "out",
            source_manifest=source / "manifest.json",
            canary_model_revision="revision",
        )
