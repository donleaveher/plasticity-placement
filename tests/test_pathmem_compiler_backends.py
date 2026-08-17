from __future__ import annotations

from collections import Counter

import pytest

from plasticity_placement.pathmem.backends import (
    ICLHistoryRenderer,
    VersionedExternalStore,
    render_fresh_session_probe,
)
from plasticity_placement.pathmem.compiler import (
    audit_compiled_bank,
    audit_tokenizer_for_bank,
    compile_bank,
    verify_matched_exposure,
)
from plasticity_placement.pathmem.generator import (
    build_item_bank,
    build_probe_bank,
    compile_event_blocks,
)
from plasticity_placement.pathmem.observability import (
    build_observability_matrix,
    claim_evidence_matrix,
)
from plasticity_placement.pathmem.schema import (
    AccessMode,
    ContractFamily,
    LifecycleLocus,
    ObservabilityStatus,
    OperatorKind,
    TokenExposureAudit,
)


class CharacterTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is False
        return "".join(chr(token_id) for token_id in token_ids)


def test_compiled_bank_freezes_counts_controls_and_endpoint_relations() -> None:
    bank = compile_bank()
    report = audit_compiled_bank(bank)
    assert report["item_count"] == 68
    assert report["event_blocks"] == 272
    assert report["probes"] == 4_080
    assert report["history_families"] == 136
    assert report["training_probe_exact_overlap_count"] == 0
    family_a, family_b = bank.history_families[:2]
    assert [path.path_id.rsplit(":", 1)[-1] for path in family_a.histories] == ["ABA", "BAA"]
    assert [path.path_id.rsplit(":", 1)[-1] for path in family_b.histories] == ["BAB", "ABB"]
    for family in (family_a, family_b):
        control = family.comparator_control
        assert control.optimizer_steps_total == 48
        assert control.training_example_count == 12
        assert len(control.ordered_minibatch_sha256_by_block) == 3
        assert all(path.expected_endpoint_hash == family.endpoint_hash for path in family.histories)


def test_external_latest_is_endpoint_identical_while_icl_retains_path_order() -> None:
    item = build_item_bank()[0]
    blocks = {block.block_id.rsplit(":", 1)[-1]: block for block in compile_event_blocks(item)}
    external_renders = []
    icl_renders = []
    for path in (("A1", "B1", "A2"), ("B1", "A1", "A2")):
        external = VersionedExternalStore()
        icl = ICLHistoryRenderer()
        for block_id in path:
            external.apply(blocks[block_id])
            icl.apply(blocks[block_id])
        external_renders.append(external.render())
        icl_renders.append(icl.render())
    assert external_renders[0] == external_renders[1]
    assert icl_renders[0] != icl_renders[1]
    assert item.state_a_action in external_renders[0]
    assert item.state_b_action not in external_renders[0]


def test_fresh_session_access_off_changes_only_named_persistent_state() -> None:
    item = build_item_bank()[0]
    probe = build_probe_bank(item)[4]
    on = render_fresh_session_probe(
        probe,
        persistent_state="<state>active</state>",
        access_mode=AccessMode.ON,
    )
    off = render_fresh_session_probe(
        probe,
        persistent_state="ignored",
        access_mode=AccessMode.OFF,
    )
    assert on.endswith(probe.prompt)
    assert off == probe.prompt


def test_observability_matrix_is_total_and_preserves_explicit_na() -> None:
    matrix = build_observability_matrix()
    expected = len(OperatorKind) * len(ContractFamily) * len(LifecycleLocus)
    assert len(matrix) == expected
    counts = Counter(cell.status for cell in matrix)
    assert counts[ObservabilityStatus.NOT_APPLICABLE] > 0
    assert all(
        cell.trace_field == "N/A"
        for cell in matrix
        if cell.status is ObservabilityStatus.NOT_APPLICABLE
    )
    assert {row["claim_id"] for row in claim_evidence_matrix()} == {
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
    }


def test_tokenizer_audit_checks_all_candidates_and_equal_lengths() -> None:
    complete = compile_bank()
    first_item_id = complete.items[0].item_id
    bounded = type(complete)(
        items=(complete.items[0],),
        event_blocks=tuple(
            block for block in complete.event_blocks if block.block_id.startswith(first_item_id)
        ),
        probes=tuple(probe for probe in complete.probes if probe.item_id == first_item_id),
        history_families=tuple(
            family for family in complete.history_families if family.item_id == first_item_id
        ),
        diagnostic_specs=complete.diagnostic_specs,
    )
    report = audit_tokenizer_for_bank(
        CharacterTokenizer(),
        bounded,
        evaluation_max_length=2_048,
    )
    assert report["probe_count"] == 60
    assert report["candidate_count"] == 240
    assert report["all_valid"] is True
    assert report["failure_count"] == 0


def test_resolved_token_exposure_audits_must_match_without_truncation() -> None:
    common = {
        "history_family_id": "family",
        "tokenizer_sha256": "a" * 64,
        "loss_bearing_tokens": 120,
        "total_forward_tokens": 240,
        "optimizer_steps": 48,
        "prompt_truncated": False,
        "candidate_token_lengths": (3, 3, 3, 3),
        "ordered_minibatch_sha256_by_block": (("A1", "b" * 64),),
    }
    first = TokenExposureAudit(path_id="ABA", **common)
    second = TokenExposureAudit(path_id="BAA", **common)
    verify_matched_exposure(first, second)
    with pytest.raises(ValueError, match="path exposure mismatch"):
        verify_matched_exposure(
            first,
            TokenExposureAudit(path_id="BAA", **{**common, "total_forward_tokens": 241}),
        )
