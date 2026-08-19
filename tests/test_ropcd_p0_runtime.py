from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from plasticity_placement.pathmem.compiler import compile_bank
from plasticity_placement.pathmem.io import json_hash
from plasticity_placement.pathmem_ropcd_p0 import runtime
from plasticity_placement.pathmem_ropcd_p0.config import TRAINING_SEED
from plasticity_placement.pathmem_ropcd_p0.planner import compile_p0_plan
from plasticity_placement.pathmem_ropcd_p0.training import _step_seed


def _handoff() -> dict[str, Any]:
    identity = {
        "schema_version": "pathmem-g1c-to-ropcd-p0-handoff-v1",
        "passed": True,
        "run_id": "r" * 64,
        "source_manifest_id": "s" * 64,
        "plan_id": "p" * 64,
        "g1c_authorization_id": "a" * 64,
        "g1c_implementation_sha256": "i" * 64,
        "g1c_git_revision": "1" * 40,
        "summary_sha256": "u" * 64,
        "g1c_gate": {"passed": True},
        "p0_runner_implementation_review_eligible": True,
        "p0_authorized": False,
        "path_contrast_computed": False,
    }
    return {**identity, "handoff_id": json_hash(identity)}


def _fake_score_prompt(_session: object, **kwargs: Any) -> dict[str, Any]:
    row_id = json_hash(
        [
            kwargs["unit_id"],
            kwargs["probe_id"],
            kwargs["arm"],
            kwargs["raw_prompt"],
        ]
    )
    ordered_actions = list(kwargs["action_choices"])
    expected_action = str(kwargs["expected_action"])
    probabilities = [0.05, 0.05, 0.05, 0.05]
    probabilities[ordered_actions.index(expected_action)] = 0.85
    return {
        "row_id": row_id,
        "unit_id": kwargs["unit_id"],
        "item_id": kwargs["item_id"],
        "probe_id": kwargs["probe_id"],
        "probe_category": kwargs["probe_category"],
        "arm": kwargs["arm"],
        "ordered_actions": ordered_actions,
        "candidate_probabilities": probabilities,
        "candidate_scores": probabilities,
        "expected_action": expected_action,
        "obsolete_action": kwargs["obsolete_action"],
        "predicted_action": expected_action,
        "correct": True,
        "error_status": "ok",
        "tie": False,
        "non_finite": False,
        "parser_valid": True,
        "raw_prompt_sha256": json_hash(kwargs["raw_prompt"]),
        "candidate_audit": {"all_candidates_valid": True},
    }


def test_final_path_scores_exact_frozen_arm_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = compile_p0_plan(_handoff())
    unit = next(row for row in plan["units"] if row["logical_name"] == "ABA")
    bank = compile_bank()
    probes = {probe.probe_id: probe for probe in bank.probes}
    blocks = {block.block_id: block for block in bank.event_blocks}
    monkeypatch.setattr(runtime, "score_prompt", _fake_score_prompt)

    rows = runtime._score_final_path(object(), unit, probes, blocks)

    assert len(rows) == 118
    assert Counter(row["arm"] for row in rows) == {
        "base_before": 14,
        "parametric_path": 14,
        "parametric_rescore": 14,
        "adapter_disabled": 14,
        "external_latest": 14,
        "icl_history": 14,
        "both": 14,
        "base_unrelated": 8,
        "parametric_unrelated": 8,
        "path_qualification": 4,
    }
    assert runtime._verify_unit_rows(rows, unit) is None
    assert len({row["row_id"] for row in rows}) == 118


def test_unit_row_verifier_rejects_missing_or_duplicate_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = compile_p0_plan(_handoff())
    unit = next(row for row in plan["units"] if row["logical_name"] == "BAB")
    bank = compile_bank()
    probes = {probe.probe_id: probe for probe in bank.probes}
    blocks = {block.block_id: block for block in bank.event_blocks}
    monkeypatch.setattr(runtime, "score_prompt", _fake_score_prompt)
    rows = runtime._score_final_path(object(), unit, probes, blocks)

    with pytest.raises(ValueError, match="row count changed"):
        runtime._verify_unit_rows(rows[:-1], unit)
    duplicate = [*rows[:-1], rows[0]]
    with pytest.raises(ValueError, match="row identities changed"):
        runtime._verify_unit_rows(duplicate, unit)


def test_comparator_and_duplicate_training_rng_are_semantically_matched() -> None:
    plan = compile_p0_plan(_handoff())
    by_logical = {
        unit["logical_name"]: unit for unit in plan["units"] if unit["item_id"] == "pmv1-smoke-01"
    }
    assert by_logical["ABA"]["step_seed_namespace"] == by_logical["BAA"]["step_seed_namespace"]
    duplicate = next(
        unit
        for unit in plan["units"]
        if unit["item_id"] == "pmv1-smoke-01" and unit["technical_duplicate"]
    )
    original = by_logical[str(duplicate["duplicate_of"])]
    assert duplicate["step_seed_namespace"] == original["step_seed_namespace"]
    for step in (0, 1, 47):
        expected = _step_seed(TRAINING_SEED, original["step_seed_namespace"], step)
        assert _step_seed(TRAINING_SEED, duplicate["step_seed_namespace"], step) == expected
    assert _step_seed(TRAINING_SEED, by_logical["AB"]["step_seed_namespace"], 0) != (
        _step_seed(TRAINING_SEED, by_logical["BA"]["step_seed_namespace"], 0)
    )
