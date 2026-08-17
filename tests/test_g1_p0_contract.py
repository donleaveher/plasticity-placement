from __future__ import annotations

import json
from pathlib import Path

import pytest

from plasticity_placement.pathmem.io import json_hash
from plasticity_placement.pathmem_exec.artifacts import (
    AttemptState,
    PhaseManifest,
    verify_g1_authorization,
    write_g1_authorization,
)
from plasticity_placement.pathmem_exec.cli import execution_plan_summary
from plasticity_placement.pathmem_exec.config import G1_CONFIG, P0_CONFIG
from plasticity_placement.pathmem_exec.training import trainer_config_identity
from plasticity_placement.training.config import LoraTrainingConfig


def test_frozen_plan_has_24_g1_anchors_and_44_p0_artifacts() -> None:
    plan = execution_plan_summary()
    assert plan["g1"]["trained_artifact_count"] == 24
    assert plan["g1"]["current_only_anchors_per_item"] == ["A2", "B2"]
    assert plan["p0"]["trained_artifact_count"] == 44
    assert set(plan["p0"]["technical_duplicate_by_item"].values()) == {
        "ABA",
        "BAA",
        "BAB",
        "ABB",
    }
    assert plan["safe_default"] == "no training and no GPU inference"


def test_phase_recipes_are_explicit_nf4_bf16_reset_adamw() -> None:
    for config in (G1_CONFIG, P0_CONFIG):
        recipe = config.recipe
        assert recipe.quantization == "nf4_double_quant"
        assert recipe.compute_dtype == "bfloat16"
        assert recipe.optimizer == "AdamW"
        assert recipe.weight_decay == 0.0
        assert recipe.optimizer_steps_per_event == 16
        assert recipe.rank == 8
        assert recipe.alpha == 16
        assert recipe.target_modules == ("q_proj", "v_proj")
        assert len(trainer_config_identity(config)) == 64


def test_generic_lora_config_exposes_weight_decay(tmp_path: Path) -> None:
    config = LoraTrainingConfig(
        model_name="model",
        data_path=tmp_path / "data.jsonl",
        output_dir=tmp_path / "adapter",
        weight_decay=0.0,
    )
    assert config.to_dict()["weight_decay"] == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        LoraTrainingConfig(
            model_name="model",
            data_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "adapter-2",
            weight_decay=-0.1,
        )


def test_manifest_lifecycle_and_g1_authorization_are_idempotent(tmp_path: Path) -> None:
    g0_id = "a" * 64
    units = {"attempt-1": {"item_id": "item-1"}}
    manifest = PhaseManifest.load_or_create(
        tmp_path / "manifest.json",
        phase="G1",
        g0_manifest_id=g0_id,
        identity={"recipe": "frozen"},
        planned_units=units,
    )
    manifest.mark("attempt-1", AttemptState.TRAINING)
    manifest.mark("attempt-1", AttemptState.TRAINED)
    manifest.mark("attempt-1", AttemptState.SCORING)
    manifest.mark("attempt-1", AttemptState.VERIFIED)
    manifest.require_all_verified()

    gate = {"gate": "G1", "passed": True, "checks": [], "blockers": []}
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "run_id": manifest.payload["run_id"],
                "g0_manifest_id": g0_id,
                "gate": gate,
                "integrity_checks": {"complete": True},
            }
        ),
        encoding="utf-8",
    )
    authorization_path = tmp_path / "g1_authorization.json"
    first = write_g1_authorization(
        authorization_path,
        g0_manifest_id=g0_id,
        run_id=str(manifest.payload["run_id"]),
        summary_path=summary,
        gate=gate,
        integrity_checks={"complete": True},
    )
    second = write_g1_authorization(
        authorization_path,
        g0_manifest_id=g0_id,
        run_id=str(manifest.payload["run_id"]),
        summary_path=summary,
        gate=gate,
        integrity_checks={"complete": True},
    )
    assert first == second
    assert verify_g1_authorization(authorization_path, expected_g0_manifest_id=g0_id) == first

    tampered = json.loads(authorization_path.read_text(encoding="utf-8"))
    tampered["scope"] = "authorize_everything"
    authorization_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        verify_g1_authorization(authorization_path, expected_g0_manifest_id=g0_id)


def test_authorization_refuses_failed_integrity(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError):
        write_g1_authorization(
            tmp_path / "authorization.json",
            g0_manifest_id="a" * 64,
            run_id=json_hash({"run": 1}),
            summary_path=summary,
            gate={"passed": True},
            integrity_checks={"complete": False},
        )
