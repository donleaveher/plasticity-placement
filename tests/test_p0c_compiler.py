from pathlib import Path

import pytest

from plasticity_placement.p0c.compiler import (
    ACTIONS,
    audit_compiled_bank,
    build_lesson_bank,
    compile_bank,
    load_compiled_bank,
    render_external_prompt,
    write_compiled_bank,
)
from plasticity_placement.p0c.runtime import prepare_experiment


def test_frozen_bank_has_expected_splits_and_probe_counts() -> None:
    lessons = build_lesson_bank()
    assert len(lessons) == 38
    assert sum(lesson.split == "development" for lesson in lessons) == 6
    assert sum(lesson.split == "confirmatory" for lesson in lessons) == 24
    assert sum(lesson.split == "reserve" for lesson in lessons) == 8

    compiled = compile_bank(lessons[:1])[0]
    assert len(compiled.training_rows) == 4
    assert len(compiled.screening_probes) == 4
    assert len(compiled.evaluation_probes) == 26
    assert len({probe.probe_id for probe in compiled.evaluation_probes}) == 26
    screening_first_choices = {probe.action_choices[0] for probe in compiled.screening_probes}
    assert screening_first_choices == set(ACTIONS)


def test_external_note_is_only_injected_for_relevant_probes() -> None:
    item = compile_bank()[:1][0]
    relevant = next(probe for probe in item.evaluation_probes if probe.retrieval_relevant)
    irrelevant = next(probe for probe in item.evaluation_probes if not probe.retrieval_relevant)
    assert item.external_note in render_external_prompt(relevant, item.external_note)
    assert render_external_prompt(irrelevant, item.external_note) == irrelevant.prompt


def test_compiled_bank_round_trip(tmp_path: Path) -> None:
    hashes = write_compiled_bank(tmp_path)
    loaded = load_compiled_bank(tmp_path)
    assert hashes["compiler_version"] == "p0c-compiler-v3"
    assert hashes["leakage_audit_sha256"]
    assert len(loaded) == 38
    assert loaded[0] == compile_bank()[:1][0]
    audit = audit_compiled_bank(loaded)
    assert audit["exact_training_evaluation_overlap_count"] == 0
    assert audit["non_target_context_violation_count"] == 0


def test_pairs_counterbalance_action_and_position() -> None:
    lessons = [
        lesson
        for lesson in build_lesson_bank()
        if lesson.split == "confirmatory" and lesson.lesson_type == "fact_mapping"
    ]
    action_positions: dict[str, set[str]] = {action: set() for action in ACTIONS}
    for lesson in lessons:
        position = "a" if lesson.lesson_id.endswith("_a") else "b"
        action_positions[lesson.desired_action].add(position)
    assert all(positions == {"a", "b"} for positions in action_positions.values())


def test_neighbor_prompts_do_not_embed_target_context() -> None:
    item = compile_bank()[:1][0]
    neighbors = [probe for probe in item.evaluation_probes if probe.category == "near_neighbor"]
    assert all(item.lesson.context_id not in probe.prompt for probe in neighbors)


def test_prepare_rejects_modified_compiled_artifact(tmp_path: Path) -> None:
    write_compiled_bank(tmp_path)
    lessons_path = tmp_path / "compiled" / "lessons.jsonl"
    lessons_path.write_text(
        lessons_path.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_experiment(tmp_path)
