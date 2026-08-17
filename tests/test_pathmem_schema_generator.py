from __future__ import annotations

from collections import Counter

import pytest

from plasticity_placement.pathmem.generator import (
    DIRECTED_ACTION_PAIRS,
    GENERATOR_SEED,
    SPLIT_COUNTS,
    build_item_bank,
    build_probe_bank,
    compile_event_blocks,
)
from plasticity_placement.pathmem.reducer import endpoint_hash, reduce_events
from plasticity_placement.pathmem.schema import (
    ContentDomain,
    ContractFamily,
    EventKind,
    LifecycleLocus,
    LifecycleTraceRecord,
    MemoryEntry,
    MemoryEvent,
    ObservabilityStatus,
    OperatorKind,
    ProbeCategory,
)


def test_reducer_implements_write_revise_delete_restore_and_null_semantics() -> None:
    initial = (MemoryEntry("a", "old"), MemoryEntry("keep", "stable"))
    events = (
        MemoryEvent("write", EventKind.WRITE, key="a", value="new"),
        MemoryEvent("revise", EventKind.REVISE, key="a", value="newer"),
        MemoryEvent("clock", EventKind.CLOCK_NULL),
        MemoryEvent("delete", EventKind.DELETE, key="keep"),
        MemoryEvent(
            "restore",
            EventKind.RESTORE,
            snapshot=(MemoryEntry("a", "restored"), MemoryEntry("z", "present")),
        ),
        MemoryEvent("idle", EventKind.IDLE),
        MemoryEvent("noop", EventKind.NOOP),
    )
    assert reduce_events(initial, events) == (
        MemoryEntry("a", "restored"),
        MemoryEntry("z", "present"),
    )
    assert len(endpoint_hash(reduce_events(initial, events))) == 64


def test_invalid_event_payloads_are_rejected() -> None:
    with pytest.raises(ValueError, match="requires key and value"):
        MemoryEvent("bad-write", EventKind.WRITE, key="a")
    with pytest.raises(ValueError, match="requires a non-empty snapshot"):
        MemoryEvent("bad-restore", EventKind.RESTORE)
    with pytest.raises(ValueError, match="cannot carry value"):
        MemoryEvent("bad-noop", EventKind.NOOP, value="unexpected")


def test_frozen_item_bank_counts_balance_and_directed_pair_coverage() -> None:
    items = build_item_bank()
    assert len(items) == sum(SPLIT_COUNTS.values()) == 68
    assert build_item_bank() == items
    with pytest.raises(ValueError, match="frozen"):
        build_item_bank(GENERATOR_SEED + 1)
    for split, expected in SPLIT_COUNTS.items():
        assert sum(item.split == split for item in items) == expected
    confirmatory = [item for item in items if item.split == "confirmatory"]
    for domain in ContentDomain:
        pairs = {
            (item.state_a_action, item.state_b_action)
            for item in confirmatory
            if item.content_domain is domain
        }
        assert pairs == set(DIRECTED_ACTION_PAIRS)


def test_each_item_compiles_four_blocks_and_sixty_terminal_probes() -> None:
    item = build_item_bank()[0]
    blocks = compile_event_blocks(item)
    assert [block.block_id.rsplit(":", 1)[-1] for block in blocks] == [
        "A1",
        "A2",
        "B1",
        "B2",
    ]
    assert all(len(block.examples) == 4 for block in blocks)
    assert all(block.optimizer_steps == 16 for block in blocks)
    probes = build_probe_bank(item)
    assert len(probes) == 60
    assert Counter(probe.terminal_state for probe in probes) == {"A": 30, "B": 30}
    for terminal_state in ("A", "B"):
        terminal = [probe for probe in probes if probe.terminal_state == terminal_state]
        assert sum(probe.category is ProbeCategory.QUALIFICATION for probe in terminal) == 4
        assert sum(probe.core_endpoint for probe in terminal) == 14
        assert len({probe.probe_id for probe in terminal}) == 30


def test_control_probes_do_not_embed_the_target_context_or_key() -> None:
    item = build_item_bank()[0]
    controls = [
        probe
        for probe in build_probe_bank(item)
        if probe.category in {ProbeCategory.NEAR_NEIGHBOR, ProbeCategory.UNRELATED}
    ]
    assert controls
    assert all(item.context_id not in probe.prompt for probe in controls)
    assert all(item.memory_key not in probe.prompt for probe in controls)


def test_lifecycle_trace_keeps_unavailable_fields_explicit() -> None:
    trace = LifecycleTraceRecord(
        operator=OperatorKind.LORA_ADAMW_RESET,
        contract=ContractFamily.TERMINAL_CONSISTENCY,
        locus=LifecycleLocus.RETRIEVE_SURFACE,
        observability=ObservabilityStatus.NOT_APPLICABLE,
        trace_value="N/A",
        event_position=None,
        trace_sha256="N/A",
    )
    assert trace.to_dict()["trace_sha256"] == "N/A"
    with pytest.raises(ValueError, match="explicit N/A"):
        LifecycleTraceRecord(
            operator=OperatorKind.LORA_ADAMW_RESET,
            contract=ContractFamily.TERMINAL_CONSISTENCY,
            locus=LifecycleLocus.RETRIEVE_SURFACE,
            observability=ObservabilityStatus.NOT_APPLICABLE,
            trace_value="missing",
            event_position=None,
            trace_sha256="N/A",
        )
