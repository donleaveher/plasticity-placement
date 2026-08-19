from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from plasticity_placement.pathmem.io import json_hash
from plasticity_placement.pathmem_exec.artifacts import (
    ADAPTER_LINEAGE_VERSION,
    adapter_bundle_hash,
)
from plasticity_placement.pathmem_exec.config import G1_RECIPE_B_CONFIG
from plasticity_placement.pathmem_exec.g1_diagnostics import build_g1_diagnostics
from plasticity_placement.pathmem_exec.training import verify_trained_node
from plasticity_placement.pathmem_exec.training_audit import audit_training_scoring_parity


class CharacterTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __init__(self, *, special_prefix: bool = False) -> None:
        self.special_prefix = special_prefix

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        values = [ord(character) for character in text]
        if kwargs.get("add_special_tokens") and self.special_prefix:
            values = [999, *values]
        return {"input_ids": values}

    def decode(self, values: list[int], **_: object) -> str:
        return "".join(chr(value) for value in values)


def test_training_scoring_parity_covers_action_tokens_eos_and_lengths() -> None:
    audit = audit_training_scoring_parity(
        tokenizer=CharacterTokenizer(),
        formatted_prompt="prompt:",
        completion="a0",
        action_choices=("a0", "b0", "c0", "d0"),
        training_max_length=32,
        evaluation_max_length=32,
    )
    assert audit["passed"] is True
    assert audit["training_loss_token_ids"] == [ord("a"), ord("0"), 0]
    assert audit["checks"]["loss_action_tokens_equal_scoring_tokens"] is True
    assert audit["checks"]["training_dataset_loss_tokens_exact"] is True
    assert audit["checks"]["equal_candidate_token_lengths"] is True


def test_training_scoring_parity_rejects_special_token_mode_mismatch() -> None:
    audit = audit_training_scoring_parity(
        tokenizer=CharacterTokenizer(special_prefix=True),
        formatted_prompt="prompt:",
        completion="a0",
        action_choices=("a0", "b0", "c0", "d0"),
        training_max_length=32,
        evaluation_max_length=32,
    )
    assert audit["passed"] is False
    assert audit["checks"]["prompt_special_token_modes_match"] is False


def test_g1_diagnostic_reports_action_asymmetry_and_unrelated_transitions() -> None:
    actions = ["a", "b", "c", "d"]
    rows = [
        _row("run::A2", "q-a", "bare_base", actions, "a", "b", [0.2, 0.4, 0.2, 0.2]),
        _row(
            "run::A2",
            "q-a",
            "parametric_current",
            actions,
            "a",
            "a",
            [0.7, 0.1, 0.1, 0.1],
        ),
        _row("run::B2", "q-b", "bare_base", actions, "b", "b", [0.2, 0.4, 0.2, 0.2]),
        _row(
            "run::B2",
            "q-b",
            "parametric_current",
            actions,
            "b",
            "a",
            [0.6, 0.2, 0.1, 0.1],
        ),
        _row("run::A2", "u", "base_unrelated", actions, "c", "c", [0.1, 0.1, 0.7, 0.1]),
        _row(
            "run::A2",
            "u",
            "parametric_unrelated",
            actions,
            "c",
            "a",
            [0.6, 0.1, 0.2, 0.1],
        ),
    ]
    lineages = [
        {
            "attempt_id": "run::A2",
            "recipe_id": "A",
            "training_summary": {"final_loss": 0.1},
        },
        {
            "attempt_id": "run::B2",
            "recipe_id": "A",
            "training_summary": {"final_loss": 0.8},
        },
    ]
    diagnostic = build_g1_diagnostics(rows, lineages=lineages)
    assert diagnostic["per_action"]["a"]["top1"] == 1.0
    assert diagnostic["per_action"]["b"]["top1"] == 0.0
    assert diagnostic["confusion_matrix"]["b"] == {"a": 1}
    assert diagnostic["unrelated_correctness_transitions"] == {"correct_to_wrong": 1}
    assert diagnostic["training"]["mean_final_loss_by_action"] == {"a": 0.1, "b": 0.8}
    assert diagnostic["observed_pattern"]["strong_action_asymmetry"] is True


def test_g1_diagnostic_requires_exact_unrelated_pairs() -> None:
    actions = ["a", "b", "c", "d"]
    rows = [
        _row("run", "q", "bare_base", actions, "a", "a", [0.7, 0.1, 0.1, 0.1]),
        _row(
            "run", "q", "parametric_current", actions, "a", "a", [0.7, 0.1, 0.1, 0.1]
        ),
        _row("run", "u", "base_unrelated", actions, "a", "a", [0.7, 0.1, 0.1, 0.1]),
    ]
    with pytest.raises(ValueError, match="exact base/adapter pairs"):
        build_g1_diagnostics(rows)


def test_resume_verification_rederives_trainer_and_node_identity(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    adapter_sha256 = adapter_bundle_hash(adapter)
    identity = {"node": "prospective-recipe-b"}
    node_id = json_hash(identity)
    data_sha256 = "d" * 64
    trainer_sha256 = "t" * 64
    summary = {
        "adapter_sha256": adapter_sha256,
        "training_data_sha256": data_sha256,
        "optimizer_steps": G1_RECIPE_B_CONFIG.recipe.optimizer_steps_per_event,
        "model_revision": G1_RECIPE_B_CONFIG.model.revision,
        "precision": "nf4-bfloat16",
        "selected_layers": None,
    }
    trainer = {
        "rank": G1_RECIPE_B_CONFIG.recipe.rank,
        "alpha": G1_RECIPE_B_CONFIG.recipe.alpha,
        "dropout": G1_RECIPE_B_CONFIG.recipe.dropout,
        "target_modules": list(G1_RECIPE_B_CONFIG.recipe.target_modules),
        "learning_rate": G1_RECIPE_B_CONFIG.recipe.learning_rate,
        "weight_decay": G1_RECIPE_B_CONFIG.recipe.weight_decay,
        "warmup_ratio": G1_RECIPE_B_CONFIG.recipe.warmup_ratio,
        "max_grad_norm": G1_RECIPE_B_CONFIG.recipe.max_grad_norm,
        "max_steps": G1_RECIPE_B_CONFIG.recipe.optimizer_steps_per_event,
        "use_4bit": True,
        "use_chat_template": True,
    }
    lineage = {
        "schema_version": ADAPTER_LINEAGE_VERSION,
        "attempt_id": "attempt-b",
        "recipe_id": "B",
        "node_id": node_id,
        "node_identity": identity,
        "adapter_sha256": adapter_sha256,
        "parent_adapter_sha256": "p" * 64,
        "parent_state_sha256": "s" * 64,
        "root_state_sha256": "r" * 64,
        "training_data_sha256": data_sha256,
        "trainer_config_sha256": trainer_sha256,
        "training_summary": summary,
    }
    (adapter / "pathmem_lineage.json").write_text(json.dumps(lineage), encoding="utf-8")
    metadata_path = adapter / "training_metadata.json"
    metadata_path.write_text(
        json.dumps({"config": trainer, "summary": summary}), encoding="utf-8"
    )
    arguments = {
        "config": G1_RECIPE_B_CONFIG,
        "runtime": SimpleNamespace(trainer_config_sha256=trainer_sha256),
        "output_dir": adapter,
        "attempt_id": "attempt-b",
        "expected_node_id": node_id,
        "expected_node_identity": identity,
        "parent_adapter_sha256": "p" * 64,
        "parent_state_sha256": "s" * 64,
        "root_state_sha256": "r" * 64,
        "training_data_sha256": data_sha256,
    }
    assert verify_trained_node(**arguments)["node_id"] == node_id

    trainer["max_steps"] = 99
    metadata_path.write_text(
        json.dumps({"config": trainer, "summary": summary}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="trainer_max_steps"):
        verify_trained_node(**arguments)


def _row(
    attempt_id: str,
    probe_id: str,
    arm: str,
    actions: list[str],
    expected: str,
    predicted: str,
    probabilities: list[float],
) -> dict[str, object]:
    return {
        "attempt_id": attempt_id,
        "probe_id": probe_id,
        "arm": arm,
        "ordered_actions": actions,
        "expected_action": expected,
        "predicted_action": predicted,
        "candidate_probabilities": probabilities,
        "correct": expected == predicted,
    }
