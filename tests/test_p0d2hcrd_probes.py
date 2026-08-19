from __future__ import annotations

from plasticity_placement.p0c.compiler import ACTIONS, compile_bank
from plasticity_placement.p0d2h.probes import compile_hard_probe_bank
from plasticity_placement.p0d2hcrd.probes import (
    compile_decomposition_bank,
)

SELECTED_IDS = (
    "F_pair01_a",
    "F_pair01_b",
    "F_pair04_a",
    "F_pair04_b",
    "F_pair06_a",
    "F_pair06_b",
    "RF_pair01_a",
    "RF_pair01_b",
    "RF_pair04_a",
    "RF_pair04_b",
    "RF_pair05_a",
    "RF_pair05_b",
    "P_pair01_a",
    "P_pair01_b",
    "P_pair02_a",
    "P_pair02_b",
    "P_pair03_a",
    "P_pair03_b",
    "P_pair04_a",
    "P_pair04_b",
    "P_pair05_a",
    "P_pair05_b",
    "P_pair06_a",
    "P_pair06_b",
)


def selected_lessons():
    by_id = {item.lesson.lesson_id: item for item in compile_bank()}
    return tuple(by_id[lesson_id] for lesson_id in SELECTED_IDS)


def test_decomposition_bank_is_deterministic_balanced_and_leak_free() -> None:
    selected = selected_lessons()
    hard_bank = compile_hard_probe_bank(selected)
    bank, audit = compile_decomposition_bank(selected, hard_bank)
    repeated, repeated_audit = compile_decomposition_bank(selected, hard_bank)
    assert bank == repeated
    assert audit == repeated_audit
    assert audit["all_checks_passed"] is True
    assert audit["lesson_count"] == 24
    assert audit["decision_count"] == 1_536
    assert audit["candidate_count"] == 5_376
    assert audit["endpoint_counts"] == {
        "route_only": 384,
        "retrieval_only": 384,
        "combined": 768,
    }
    assert audit["action_lesson_counts"] == {action: 6 for action in ACTIONS}

    for item in selected:
        probes = bank[item.lesson.lesson_id]
        route = [probe for probe in probes if probe.endpoint == "route_only"]
        retrieval = [
            probe for probe in probes if probe.endpoint == "retrieval_only"
        ]
        combined = [probe for probe in probes if probe.endpoint == "combined"]
        assert (len(route), len(retrieval), len(combined)) == (16, 16, 32)
        assert {probe.target_slot for probe in route} == {"slot_a", "slot_b"}
        for probe in route:
            assert len(probe.ordered_candidates) == 2
            assert all(action not in probe.prompt for action in ACTIONS)
            assert item.external_note not in probe.prompt
            assert item.lesson.context_id not in probe.prompt
            assert item.lesson.condition not in probe.prompt
        for probe in retrieval:
            assert len(probe.ordered_candidates) == 4
            assert "Routing table:" not in probe.prompt
            assert "Current marker:" not in probe.prompt
            assert item.external_note in probe.prompt
        for probe in combined:
            assert len(probe.ordered_candidates) == 4
            assert probe.prompt.count("[LIVE]") == 1
            assert probe.prompt.count("[ARCHIVED]") == 1
            assert item.external_note in probe.prompt


def test_all_cells_link_to_the_matching_source_route_variant() -> None:
    selected = selected_lessons()
    bank, _ = compile_decomposition_bank(
        selected,
        compile_hard_probe_bank(selected),
    )
    for lesson_id, probes in bank.items():
        for probe in probes:
            assert probe.source_probe_id == (
                f"{lesson_id}_conditional_route_{probe.route_variant:02d}"
            )
            assert probe.source_row_key == (
                f"scale_canary::{lesson_id}::external::"
                f"{probe.source_probe_id}"
            )

