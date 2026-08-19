from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from plasticity_placement.p0d2hrr.config import (
    PilotSpec,
    ResolvedPilotConfig,
    RouteDataSpec,
    RouteTrainingSpec,
)
from plasticity_placement.p0d2hrr.data import compile_route_data
from plasticity_placement.p0d2hrr.preflight import build_dev_token_audit


class _CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, values: list[int], **_: object) -> str:
        return "".join(chr(value) for value in values)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **_: object,
    ) -> str:
        return f"<user>{messages[0]['content']}</user><assistant>"


def test_frozen_spec_loads_from_repository_config() -> None:
    path = Path(__file__).parents[1] / "configs" / "p0d2hrr-route-remediation-pilot-v1.json"
    spec = PilotSpec.from_path(path)
    assert spec.data.train_example_count == 480
    assert spec.data.dev_example_count == 96
    assert spec.training.rank == 8
    assert spec.training.max_steps == 30
    assert spec.allowed_external_evaluations == 1
    assert spec.allows_mappings_per_adapter_scan is False
    assert spec.allows_rlvr is False


def test_frozen_spec_rejects_sweep_or_training_change() -> None:
    with pytest.raises(ValueError, match="configuration is frozen"):
        RouteTrainingSpec(rank=16)
    with pytest.raises(ValueError, match="matrix is frozen"):
        RouteDataSpec(train_group_count=61)
    with pytest.raises(ValueError, match="cannot authorize"):
        replace(PilotSpec(), allows_rlvr=True)


def test_resolved_identity_is_hash_bound_and_never_self_authorizes(
    tmp_path: Path,
) -> None:
    digest = "a" * 64
    config = ResolvedPilotConfig(
        output_dir=tmp_path,
        source_manifest_path=tmp_path / "source.json",
        code_sha256=digest,
        source_run_id="frozen-source",
        source_manifest_sha256=digest,
        source_summary_sha256=digest,
        source_candidate_token_audit_sha256=digest,
        source_raw_results_sha256=digest,
        spec=PilotSpec(),
        spec_sha256=digest,
        train_data_sha256=digest,
        dev_data_sha256=digest,
        data_audit_sha256=digest,
        dev_token_audit_sha256=digest,
        forced_choice_token_audit_sha256=digest,
        crd_bank_audit_sha256=digest,
        crd_token_audit_sha256=digest,
        source_evaluation_precision="nf4-bfloat16",
    )
    identity = config.identity_dict()
    assert len(identity["preregistration_sha256"]) == 64
    assert identity["training_run_limit"] == 1
    assert identity["training_complexity_review_eligible"] is False
    assert identity["automatic_training_started"] is False
    assert identity["automatic_narrow_scan_started"] is False
    assert identity["automatic_rlvr_started"] is False


def test_route_bank_is_deterministic_balanced_and_group_disjoint() -> None:
    spec = PilotSpec().data
    first_train, first_dev, first_audit = compile_route_data(spec)
    second_train, second_dev, second_audit = compile_route_data(spec)
    assert first_train == second_train
    assert first_dev == second_dev
    assert first_audit == second_audit
    assert first_audit["all_checks_passed"] is True
    assert first_audit["balance"]["train"]["tasks"] == {
        "route_and_copy": 240,
        "route_only": 240,
    }
    assert first_audit["balance"]["train"]["route_and_copy_targets"] == {
        "act_k2": 60,
        "act_n7": 60,
        "act_p3": 60,
        "act_v9": 60,
    }
    assert {row.group_id for row in first_train}.isdisjoint({row.group_id for row in first_dev})


def test_route_bank_audit_rejects_frozen_source_leakage() -> None:
    spec = PilotSpec().data
    train, _, _ = compile_route_data(spec)
    leaked = train[0].prompt.split()[0]
    with pytest.raises(ValueError, match="data audit failed"):
        compile_route_data(spec, forbidden_strings=(leaked,))


def test_dev_token_audit_covers_every_held_out_row() -> None:
    _, dev, _ = compile_route_data(PilotSpec().data)
    audit = build_dev_token_audit(
        tokenizer=_CharacterTokenizer(),
        records=dev,
        context_sha256="a" * 64,
    )
    assert audit["decision_count"] == 96
    assert audit["candidate_count"] == 288
    assert audit["input_truncated_count"] == 0
    assert audit["all_checks_passed"] is True
