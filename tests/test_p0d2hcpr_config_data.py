from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from plasticity_placement.p0d2hcpr.config import ExperimentSpec, GateSpec, TrainingSpec
from plasticity_placement.p0d2hcpr.data import compile_data
from plasticity_placement.p0d2hcpr.preflight import _build_training_token_audit

SPEC_PATH = (
    Path(__file__).parents[1] / "configs" / "p0d2hcpr-composition-preserving-remediation-v1.json"
)


def test_frozen_spec_and_balanced_composition_bank() -> None:
    spec = ExperimentSpec.from_path(SPEC_PATH)
    train, dev, audit = compile_data(spec.data)

    assert len(train) == 1_152
    assert len(dev) == 288
    assert Counter(row.task for row in train) == {
        "route_only": 384,
        "retrieval_only": 384,
        "combined": 384,
    }
    assert audit["all_checks_passed"] is True
    assert spec.training.max_steps == 72
    assert spec.training.objective == "equal_weight_sft_route_retrieval_combined"
    assert spec.allows_mappings_per_adapter_scan is False


def test_data_is_deterministic_and_rejects_locked_string_leakage() -> None:
    spec = ExperimentSpec.from_path(SPEC_PATH)
    first = compile_data(spec.data)
    second = compile_data(spec.data)
    assert [row.to_dict() for row in first[0]] == [row.to_dict() for row in second[0]]
    forbidden = first[0][0].prompt.split()[0]
    with pytest.raises(ValueError, match="data audit failed"):
        compile_data(spec.data, forbidden_strings=(forbidden,))


def test_training_and_gate_changes_require_new_protocol() -> None:
    with pytest.raises(ValueError, match="training configuration"):
        TrainingSpec(max_steps=73)
    with pytest.raises(ValueError, match="qualification gates"):
        GateSpec(combined_noninferiority_margin=0.03)


class _Tokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return messages[0]["content"]

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": list(range(len(text.split())))}


def test_training_token_audit_requires_a_surviving_completion() -> None:
    spec = ExperimentSpec.from_path(SPEC_PATH)
    train, _, _ = compile_data(spec.data)
    passed = _build_training_token_audit(_Tokenizer(), train[:2], max_length=512)
    failed = _build_training_token_audit(_Tokenizer(), train[:2], max_length=1)
    assert passed["all_checks_passed"] is True
    assert failed["all_checks_passed"] is False
