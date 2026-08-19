from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from plasticity_placement.p0d2hfc.scoring import audit_candidate_tokenization
from plasticity_placement.pathmem.generator import (
    ACTIONS,
    PRIMARY_BLOCK_IDS,
    build_item_bank,
    build_probe_bank,
    compile_event_blocks,
)
from plasticity_placement.pathmem.io import json_hash, multiset_hash
from plasticity_placement.pathmem.reducer import REDUCER_VERSION, endpoint_hash, reduce_events
from plasticity_placement.pathmem.schema import (
    ComparatorControl,
    ContractFamily,
    DiagnosticKind,
    DiagnosticSpec,
    EpisodeRole,
    EventBlock,
    HistoryFamily,
    HistoryPath,
    LifecycleLocus,
    MemoryEntry,
    OperatorKind,
    Probe,
    SemanticItem,
    TokenExposureAudit,
)

COMPILER_VERSION = "pathmem-history-compiler-v1"
OBSERVABILITY_PROFILE_ID = "pathmem-observability-v1"
PRIMARY_PATHS = {
    "A": {"ABA": ("A1", "B1", "A2"), "BAA": ("B1", "A1", "A2")},
    "B": {"BAB": ("B1", "A1", "B2"), "ABB": ("A1", "B1", "B2")},
}


@dataclass(frozen=True, slots=True)
class CompiledBank:
    items: tuple[SemanticItem, ...]
    event_blocks: tuple[EventBlock, ...]
    probes: tuple[Probe, ...]
    history_families: tuple[HistoryFamily, ...]
    diagnostic_specs: tuple[DiagnosticSpec, ...]


def compile_bank(items: tuple[SemanticItem, ...] | None = None) -> CompiledBank:
    source = build_item_bank() if items is None else items
    blocks: list[EventBlock] = []
    probes: list[Probe] = []
    families: list[HistoryFamily] = []
    for item in source:
        item_blocks = compile_event_blocks(item)
        blocks.extend(item_blocks)
        probes.extend(build_probe_bank(item))
        families.extend(compile_history_families(item, item_blocks))
    bank = CompiledBank(
        items=source,
        event_blocks=tuple(blocks),
        probes=tuple(probes),
        history_families=tuple(families),
        diagnostic_specs=default_diagnostic_specs(),
    )
    audit_compiled_bank(bank)
    return bank


def compile_history_families(
    item: SemanticItem,
    event_blocks: tuple[EventBlock, ...] | None = None,
) -> tuple[HistoryFamily, HistoryFamily]:
    blocks = compile_event_blocks(item) if event_blocks is None else event_blocks
    by_suffix = {_block_suffix(block.block_id): block for block in blocks}
    if set(by_suffix) != set(PRIMARY_BLOCK_IDS):
        raise ValueError(f"item {item.item_id} has an incomplete primary block set")

    families: list[HistoryFamily] = []
    for final_state, path_specs in PRIMARY_PATHS.items():
        path_rows: list[tuple[str, tuple[EventBlock, ...], tuple[MemoryEntry, ...]]] = []
        for path_name, suffixes in path_specs.items():
            path_blocks = tuple(by_suffix[suffix] for suffix in suffixes)
            final_ledger = reduce_events((), (block.event for block in path_blocks))
            path_rows.append((path_name, path_blocks, final_ledger))
        endpoints = {endpoint_hash(row[2]) for row in path_rows}
        if len(endpoints) != 1:
            raise ValueError(f"{item.item_id}/{final_state} paths are not endpoint-equivalent")
        final_endpoint_hash = endpoints.pop()
        histories = tuple(
            HistoryPath(
                path_id=f"{item.item_id}:{path_name}",
                block_ids=tuple(block.block_id for block in path_blocks),
                expected_endpoint_hash=final_endpoint_hash,
            )
            for path_name, path_blocks, _ in path_rows
        )
        first_blocks = path_rows[0][1]
        second_blocks = path_rows[1][1]
        control = _comparator_control(
            item=item,
            final_state=final_state,
            histories=histories,
            first_blocks=first_blocks,
            second_blocks=second_blocks,
        )
        families.append(
            HistoryFamily(
                history_family_id=f"{item.item_id}:terminal-{final_state}",
                item_id=item.item_id,
                content_domain=item.content_domain,
                contract_family=ContractFamily.TERMINAL_CONSISTENCY,
                initial_fixture=(),
                logical_reducer=REDUCER_VERSION,
                histories=(histories[0], histories[1]),
                endpoint_hash=final_endpoint_hash,
                comparator_control=control,
                episode_roles=tuple(EpisodeRole),
                observability_profile_id=OBSERVABILITY_PROFILE_ID,
                trace_expectations=(
                    "root_state_hash",
                    "parent_state_hash",
                    "event_block_hash",
                    "ordered_minibatch_hash",
                    "operator_state_hash_or_N/A",
                    "fresh_session_id",
                    "access_mode",
                ),
                false_positive_pattern=(
                    "unequal exposure, position-keyed minibatch RNG, terminal underfit, "
                    "or stale volatile session state"
                ),
                must_demonstrate=(
                    "same logical endpoint and design controls with a paired behavior-space test"
                ),
            )
        )
    return families[0], families[1]


def audit_compiled_bank(bank: CompiledBank) -> dict[str, Any]:
    expected_items = 68
    if len(bank.items) != expected_items:
        raise ValueError(f"PathMem-v1 requires {expected_items} items")
    expected_counts = {
        "event_blocks": expected_items * 4,
        "probes": expected_items * 60,
        "history_families": expected_items * 2,
    }
    observed_counts = {
        "event_blocks": len(bank.event_blocks),
        "probes": len(bank.probes),
        "history_families": len(bank.history_families),
    }
    if observed_counts != expected_counts:
        raise ValueError(f"compiled bank count mismatch: {observed_counts} != {expected_counts}")
    identifiers = (
        [item.item_id for item in bank.items]
        + [block.block_id for block in bank.event_blocks]
        + [probe.probe_id for probe in bank.probes]
        + [family.history_family_id for family in bank.history_families]
    )
    duplicates = [identifier for identifier, count in Counter(identifiers).items() if count > 1]
    if duplicates:
        raise ValueError(f"compiled bank has duplicate identities: {duplicates[:5]}")

    blocks_by_item: dict[str, list[EventBlock]] = {item.item_id: [] for item in bank.items}
    probes_by_item: dict[str, list[Probe]] = {item.item_id: [] for item in bank.items}
    for block in bank.event_blocks:
        blocks_by_item[block.block_id.split(":", 1)[0]].append(block)
    for probe in bank.probes:
        probes_by_item[probe.item_id].append(probe)
    exact_overlap_count = 0
    for item in bank.items:
        train_prompts = {
            example.prompt
            for block in blocks_by_item[item.item_id]
            for example in block.examples
        }
        probe_prompts = {probe.prompt for probe in probes_by_item[item.item_id]}
        exact_overlap_count += len(train_prompts & probe_prompts)
        terminal_counts = Counter(probe.terminal_state for probe in probes_by_item[item.item_id])
        if terminal_counts != {"A": 30, "B": 30}:
            raise ValueError(f"probe terminal counts changed for {item.item_id}: {terminal_counts}")
        core_counts = Counter(
            probe.terminal_state for probe in probes_by_item[item.item_id] if probe.core_endpoint
        )
        if core_counts != {"A": 14, "B": 14}:
            raise ValueError(f"core endpoint counts changed for {item.item_id}: {core_counts}")
    if exact_overlap_count:
        raise ValueError(f"training/probe exact-overlap count is {exact_overlap_count}")

    return {
        "compiler_version": COMPILER_VERSION,
        "item_count": len(bank.items),
        **observed_counts,
        "diagnostic_spec_count": len(bank.diagnostic_specs),
        "training_probe_exact_overlap_count": exact_overlap_count,
        "action_universe": list(ACTIONS),
        "semantic_item_is_statistical_unit": True,
        "history_family_adds_independent_unit": False,
    }


def audit_tokenizer_for_bank(
    tokenizer: Any,
    bank: CompiledBank,
    *,
    evaluation_max_length: int,
) -> dict[str, Any]:
    prompt_audits = []
    for probe in bank.probes:
        audit = audit_candidate_tokenization(
            tokenizer,
            probe.prompt,
            probe.action_choices,
            evaluation_max_length=evaluation_max_length,
        )
        lengths = {int(candidate["token_count"]) for candidate in audit["candidates"]}
        prompt_audits.append(
            {
                "probe_id": probe.probe_id,
                "all_candidates_valid": bool(audit["all_candidates_valid"]),
                "equal_candidate_token_length": len(lengths) == 1,
                "candidate_token_lengths": sorted(lengths),
            }
        )
    failures = [
        row
        for row in prompt_audits
        if not row["all_candidates_valid"] or not row["equal_candidate_token_length"]
    ]
    return {
        "probe_count": len(prompt_audits),
        "candidate_count": len(prompt_audits) * 4,
        "all_valid": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def verify_matched_exposure(
    first: TokenExposureAudit,
    second: TokenExposureAudit,
) -> None:
    if first.history_family_id != second.history_family_id:
        raise ValueError("exposure audits refer to different history families")
    if first.path_id == second.path_id:
        raise ValueError("exposure matching requires two distinct paths")
    matched_fields = (
        "tokenizer_sha256",
        "loss_bearing_tokens",
        "total_forward_tokens",
        "optimizer_steps",
        "candidate_token_lengths",
        "ordered_minibatch_sha256_by_block",
    )
    mismatches = [
        field for field in matched_fields if getattr(first, field) != getattr(second, field)
    ]
    if mismatches:
        raise ValueError(f"path exposure mismatch: {mismatches}")


def default_diagnostic_specs() -> tuple[DiagnosticSpec, ...]:
    return (
        DiagnosticSpec(
            diagnostic_id="external-canonical-store",
            kind=DiagnosticKind.CANONICALIZATION_DEFECT_REDUCTION,
            operator=OperatorKind.EXTERNAL_LATEST,
            target_locus=LifecycleLocus.SUPERSEDE_RESOLVE,
            held_fixed=("probe", "base_model", "renderer", "scorer", "seed"),
            authorization_phase="P1b",
        ),
        DiagnosticSpec(
            diagnostic_id="text-artifact-delete",
            kind=DiagnosticKind.ARTIFACT_NECESSITY,
            operator=OperatorKind.TEXTUAL_CONSOLIDATOR,
            target_locus=LifecycleLocus.APPLY_INFER,
            held_fixed=("probe", "base_model", "prompt", "scorer"),
            authorization_phase="P1b",
        ),
        DiagnosticSpec(
            diagnostic_id="parametric-access-off",
            kind=DiagnosticKind.ACCESS_OFF,
            operator=OperatorKind.LORA_ADAMW_RESET,
            target_locus=LifecycleLocus.ACTIVATE,
            held_fixed=("probe", "base_model", "prompt", "scorer"),
            authorization_phase="P1b",
        ),
        DiagnosticSpec(
            diagnostic_id="common-future-suffix",
            kind=DiagnosticKind.FUTURE_SUFFIX_EQUIVALENCE,
            operator=OperatorKind.LORA_ADAMW_RESET,
            target_locus=LifecycleLocus.MAINTAIN_FUTURE,
            held_fixed=("suffix", "optimizer", "dose", "ordered_minibatches"),
            authorization_phase="P1b",
        ),
        DiagnosticSpec(
            diagnostic_id="canonical-ledger-refit",
            kind=DiagnosticKind.RESTORE_OR_REFIT,
            operator=OperatorKind.LORA_ADAMW_RESET,
            target_locus=LifecycleLocus.COMMIT_ENCODE,
            held_fixed=("root", "ledger", "compute_curve", "scorer"),
            authorization_phase="P1b",
        ),
    )


def _comparator_control(
    *,
    item: SemanticItem,
    final_state: str,
    histories: tuple[HistoryPath, ...],
    first_blocks: tuple[EventBlock, ...],
    second_blocks: tuple[EventBlock, ...],
) -> ComparatorControl:
    first_event_hash = multiset_hash(block.event.to_dict() for block in first_blocks)
    second_event_hash = multiset_hash(block.event.to_dict() for block in second_blocks)
    first_example_hash = multiset_hash(
        example.to_dict() for block in first_blocks for example in block.examples
    )
    second_example_hash = multiset_hash(
        example.to_dict() for block in second_blocks for example in block.examples
    )
    first_loss_hash = multiset_hash(
        {"prompt": example.prompt, "completion": example.completion}
        for block in first_blocks
        for example in block.examples
    )
    second_loss_hash = multiset_hash(
        {"prompt": example.prompt, "completion": example.completion}
        for block in second_blocks
        for example in block.examples
    )
    if first_event_hash != second_event_hash:
        raise ValueError(f"event multiset mismatch for {item.item_id}/{final_state}")
    if first_example_hash != second_example_hash or first_loss_hash != second_loss_hash:
        raise ValueError(f"example/exposure mismatch for {item.item_id}/{final_state}")
    if first_blocks[-1].block_id != second_blocks[-1].block_id:
        raise ValueError(f"final block mismatch for {item.item_id}/{final_state}")
    first_steps = sum(block.optimizer_steps for block in first_blocks)
    second_steps = sum(block.optimizer_steps for block in second_blocks)
    if first_steps != second_steps:
        raise ValueError(f"optimizer-step mismatch for {item.item_id}/{final_state}")
    ordered_hashes = tuple(
        sorted(
            (
                block.block_id,
                json_hash([example.to_dict() for example in block.examples]),
            )
            for block in first_blocks
        )
    )
    return ComparatorControl(
        comparator_id=f"{item.item_id}:terminal-{final_state}:primary",
        path_ids=(histories[0].path_id, histories[1].path_id),
        root_state_id=f"{item.item_id}:root",
        final_block_id=first_blocks[-1].block_id,
        event_multiset_sha256=first_event_hash,
        example_multiset_sha256=first_example_hash,
        loss_bearing_text_sha256=first_loss_hash,
        ordered_minibatch_sha256_by_block=ordered_hashes,
        optimizer_steps_total=first_steps,
        training_example_count=sum(len(block.examples) for block in first_blocks),
    )


def _block_suffix(block_id: str) -> str:
    return block_id.rsplit(":", 1)[-1]
