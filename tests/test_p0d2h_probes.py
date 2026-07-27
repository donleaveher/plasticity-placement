from __future__ import annotations

import json
from pathlib import Path

import pytest

from plasticity_placement.p0c.compiler import ACTIONS, compile_bank
from plasticity_placement.p0d2h.probes import (
    HARD_CATEGORIES,
    HARD_PROBE_COMPILER_VERSION,
    PROBES_PER_LESSON,
    audit_hard_probe_bank,
    compile_hard_probe_bank,
    compile_hard_probes,
    load_hard_probe_bank,
    verify_hard_probe_hashes,
    write_hard_probe_bank,
)


def test_hard_probes_are_deterministic_balanced_and_leak_free() -> None:
    item = compile_bank()[:1][0]
    first = compile_hard_probes(item)
    assert first == compile_hard_probes(item)
    assert len(first) == PROBES_PER_LESSON == 16
    assert {probe.category for probe in first} == set(HARD_CATEGORIES)
    for category in HARD_CATEGORIES:
        probes = [probe for probe in first if probe.category == category]
        assert len(probes) == 4
        assert {probe.action_choices[0] for probe in probes} == set(ACTIONS)
    for probe in first:
        body = probe.prompt.split("\nOnly output", maxsplit=1)[0]
        assert item.lesson.context_id in body
        assert item.lesson.condition in body
        assert item.lesson.desired_action not in body
        assert (
            sum(
                action in body for action in item.lesson.distractor_actions
            )
            >= 2
        )
        assert probe.expected_action == item.lesson.desired_action
        assert probe.retrieval_relevant is True
        assert probe.split == "hard_evaluation"


def test_hard_probe_bank_round_trip_and_immutable_artifacts(
    tmp_path: Path,
) -> None:
    selected = compile_bank()[:2]
    hashes, expected = write_hard_probe_bank(tmp_path, selected)
    assert hashes["compiler_version"] == HARD_PROBE_COMPILER_VERSION
    assert load_hard_probe_bank(tmp_path) == expected
    verify_hard_probe_hashes(tmp_path, hashes)
    audit = audit_hard_probe_bank(
        selected,
        compile_hard_probe_bank(selected),
    )
    assert audit["probe_count"] == 2 * PROBES_PER_LESSON
    assert audit["target_action_leak_count"] == 0
    assert audit["exact_training_overlap_count"] == 0

    path = tmp_path / "compiled" / "hard_probe_audit.json"
    original = path.read_text(encoding="utf-8")
    path.write_text(json.dumps({"tampered": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="changed hard-probe artifact"):
        write_hard_probe_bank(tmp_path, selected)
    path.write_text(original, encoding="utf-8")
