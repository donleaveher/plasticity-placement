from __future__ import annotations

import json
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.compiler import ACTIONS
from plasticity_placement.p0c.domain import CompiledLesson, P0CProbe

HARD_PROBE_COMPILER_VERSION = "p0d2h-hard-probes-v2"
OUTPUT_INSTRUCTION_PREFIX = "\nOnly output one action token from:"
HARD_CATEGORIES = (
    "binding_decoys",
    "conflict_stack",
    "conditional_route",
    "long_context",
)
VARIANTS_PER_CATEGORY = 4
PROBES_PER_LESSON = len(HARD_CATEGORIES) * VARIANTS_PER_CATEGORY


def compile_hard_probes(item: CompiledLesson) -> tuple[P0CProbe, ...]:
    lesson = item.lesson
    probes: list[P0CProbe] = []
    builders = {
        "binding_decoys": _binding_decoys,
        "conflict_stack": _conflict_stack,
        "conditional_route": _conditional_route,
        "long_context": _long_context,
    }
    for category in HARD_CATEGORIES:
        for variant in range(1, VARIANTS_PER_CATEGORY + 1):
            statement = builders[category](item, variant)
            probes.append(
                P0CProbe(
                    probe_id=f"{lesson.lesson_id}_{category}_{variant:02d}",
                    lesson_id=lesson.lesson_id,
                    pair_id=lesson.pair_id,
                    lesson_type=lesson.lesson_type,
                    category=category,
                    prompt=_decision_prompt(
                        statement,
                        _ordered_actions(lesson.lesson_id, category, variant),
                    ),
                    action_choices=_ordered_actions(
                        lesson.lesson_id,
                        category,
                        variant,
                    ),
                    expected_action=lesson.desired_action,
                    retrieval_relevant=True,
                    split="hard_evaluation",
                )
            )
    if len(probes) != PROBES_PER_LESSON:
        raise AssertionError(f"expected {PROBES_PER_LESSON} hard probes, got {len(probes)}")
    return tuple(probes)


def compile_hard_probe_bank(
    items: Iterable[CompiledLesson],
) -> dict[str, tuple[P0CProbe, ...]]:
    return {item.lesson.lesson_id: compile_hard_probes(item) for item in items}


def audit_hard_probe_bank(
    items: Iterable[CompiledLesson],
    bank: dict[str, tuple[P0CProbe, ...]],
) -> dict[str, Any]:
    selected = tuple(items)
    by_id = {item.lesson.lesson_id: item for item in selected}
    if set(bank) != set(by_id):
        raise ValueError("hard-probe lesson IDs differ from selected lessons")

    seen_ids: set[str] = set()
    exact_training_overlaps: list[str] = []
    max_prompt_chars = 0
    max_training_ngram_jaccard: dict[str, float] = {}
    for lesson_id, item in by_id.items():
        probes = bank[lesson_id]
        if len(probes) != PROBES_PER_LESSON:
            raise ValueError(
                f"{lesson_id} has {len(probes)} hard probes, expected {PROBES_PER_LESSON}"
            )
        category_counts = {
            category: sum(probe.category == category for probe in probes)
            for category in HARD_CATEGORIES
        }
        if set(probe.category for probe in probes) != set(HARD_CATEGORIES) or set(
            category_counts.values()
        ) != {VARIANTS_PER_CATEGORY}:
            raise ValueError(f"{lesson_id} hard categories are not frozen: {category_counts}")

        training_prompts = {row.prompt for row in item.training_rows}
        lesson_max_overlap = 0.0
        for category in HARD_CATEGORIES:
            category_probes = [probe for probe in probes if probe.category == category]
            if {probe.action_choices[0] for probe in category_probes} != set(ACTIONS):
                raise ValueError(f"{lesson_id}/{category} action-choice positions are unbalanced")
        for probe in probes:
            if probe.probe_id in seen_ids:
                raise ValueError(f"duplicate hard probe ID: {probe.probe_id}")
            seen_ids.add(probe.probe_id)
            if (
                probe.lesson_id != lesson_id
                or probe.pair_id != item.lesson.pair_id
                or probe.lesson_type != item.lesson.lesson_type
                or probe.expected_action != item.lesson.desired_action
                or probe.retrieval_relevant is not True
                or probe.split != "hard_evaluation"
                or tuple(sorted(probe.action_choices)) != tuple(sorted(ACTIONS))
            ):
                raise ValueError(f"hard-probe provenance mismatch: {probe.probe_id}")
            statement = _statement(probe.prompt)
            if item.lesson.context_id not in statement:
                raise ValueError(f"target context missing from {probe.probe_id}")
            if item.lesson.condition not in statement:
                raise ValueError(f"target condition missing from {probe.probe_id}")
            if item.lesson.desired_action in statement:
                raise ValueError(f"target action leaked in {probe.probe_id}")
            distractor_count = sum(action in statement for action in item.lesson.distractor_actions)
            if distractor_count < 2:
                raise ValueError(
                    f"hard probe has fewer than two distractor actions: {probe.probe_id}"
                )
            if probe.prompt in training_prompts:
                exact_training_overlaps.append(probe.probe_id)
            max_prompt_chars = max(max_prompt_chars, len(probe.prompt))
            for row in item.training_rows:
                lesson_max_overlap = max(
                    lesson_max_overlap,
                    _ngram_jaccard(row.prompt, probe.prompt),
                )
        max_training_ngram_jaccard[lesson_id] = lesson_max_overlap

    if exact_training_overlaps:
        raise ValueError(f"hard probes exactly overlap training prompts: {exact_training_overlaps}")
    return {
        "schema_version": "p0d2h-hard-probe-audit-v2",
        "compiler_version": HARD_PROBE_COMPILER_VERSION,
        "lesson_count": len(selected),
        "probe_count": len(seen_ids),
        "categories": list(HARD_CATEGORIES),
        "variants_per_category": VARIANTS_PER_CATEGORY,
        "target_action_leak_count": 0,
        "exact_training_overlap_count": len(exact_training_overlaps),
        "max_prompt_chars": max_prompt_chars,
        "max_training_token_3gram_jaccard_by_lesson": (max_training_ngram_jaccard),
    }


def write_hard_probe_bank(
    output_dir: Path,
    items: Iterable[CompiledLesson],
) -> tuple[dict[str, str], dict[str, tuple[P0CProbe, ...]]]:
    selected = tuple(items)
    bank = compile_hard_probe_bank(selected)
    audit = audit_hard_probe_bank(selected, bank)
    compiled_dir = output_dir / "compiled"
    probes_path = compiled_dir / "hard_probes.jsonl"
    audit_path = compiled_dir / "hard_probe_audit.json"
    rows = [
        {
            **probe.to_dict(),
            "difficulty_level": "hard",
            "difficulty_dimension": probe.category,
            "compiler_version": HARD_PROBE_COMPILER_VERSION,
        }
        for lesson_id in (item.lesson.lesson_id for item in selected)
        for probe in bank[lesson_id]
    ]
    probes_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()
    audit_bytes = (json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    _write_immutable(probes_path, probes_bytes)
    _write_immutable(audit_path, audit_bytes)
    hashes = {
        "compiler_version": HARD_PROBE_COMPILER_VERSION,
        "hard_probes_sha256": sha256(probes_bytes).hexdigest(),
        "hard_probe_audit_sha256": sha256(audit_bytes).hexdigest(),
    }
    hashes_path = compiled_dir / "hard_probe_hashes.json"
    hashes_bytes = (
        json.dumps(hashes, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    _write_immutable(hashes_path, hashes_bytes)
    return hashes, bank


def load_hard_probe_bank(
    output_dir: Path,
) -> dict[str, tuple[P0CProbe, ...]]:
    path = output_dir / "compiled" / "hard_probes.jsonl"
    by_lesson: dict[str, list[P0CProbe]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("compiler_version") != HARD_PROBE_COMPILER_VERSION:
                raise ValueError("hard-probe compiler version mismatch")
            probe = P0CProbe.from_dict(row)
            by_lesson.setdefault(probe.lesson_id, []).append(probe)
    return {lesson_id: tuple(probes) for lesson_id, probes in by_lesson.items()}


def verify_hard_probe_hashes(
    output_dir: Path,
    expected: dict[str, str],
) -> None:
    compiled_dir = output_dir / "compiled"
    observed = {
        "compiler_version": HARD_PROBE_COMPILER_VERSION,
        "hard_probes_sha256": _file_hash(compiled_dir / "hard_probes.jsonl"),
        "hard_probe_audit_sha256": _file_hash(compiled_dir / "hard_probe_audit.json"),
    }
    if observed != expected:
        raise ValueError("P0-D2H compiled hard-probe artifacts changed")


def _binding_decoys(item: CompiledLesson, variant: int) -> str:
    lesson = item.lesson
    distractors = _rotated_distractors(item, variant)
    archive_lines = [
        (
            f"[ARCHIVED-{index} — INVALID] workspace "
            f"{_decoy_context(lesson.lesson_id, variant, index)} has an obsolete "
            f"record naming {action}; never execute an archived record."
        )
        for index, action in enumerate(distractors, start=1)
    ]
    target = (
        f"[ACTIVE — EXECUTE] workspace {lesson.context_id}; condition: "
        f"{lesson.condition}. Retrieve its action from the verified local mapping."
    )
    records = archive_lines.copy()
    records.insert((variant - 1) % 4, target)
    return (
        "Status labels are authoritative. First discard every ARCHIVED record, "
        "including each action named inside it. Then answer only for the single "
        "ACTIVE record by retrieving its verified mapping.\n"
        + "\n".join(records)
        + "\nWhich action applies to the ACTIVE record?"
    )


def _conflict_stack(item: CompiledLesson, variant: int) -> str:
    lesson = item.lesson
    distractors = _rotated_distractors(item, variant)
    claims = "; ".join(
        f"unverified note {index} proposes {action}"
        for index, action in enumerate(distractors, start=1)
    )
    return (
        f"Workspace {lesson.context_id}; condition: {lesson.condition}. "
        f"A noisy scratchpad contains mutually inconsistent claims: {claims}. "
        "All three notes are unverified and older than the previously verified "
        "workspace mapping. Reject the scratchpad and return the verified action."
    )


def _conditional_route(item: CompiledLesson, variant: int) -> str:
    lesson = item.lesson
    distractors = _rotated_distractors(item, variant)
    target_slot = "A" if variant % 2 else "B"
    decoy_slot = "B" if target_slot == "A" else "A"
    route_marker = f"route-{variant}"
    slots = {
        target_slot: (
            f"workspace {lesson.context_id}; condition: {lesson.condition}; "
            "retrieve its previously verified mapping"
        ),
        decoy_slot: (
            f"archived workspace {_decoy_context(lesson.lesson_id, variant, 9)}; "
            f"three advisory labels mention {distractors[0]}, {distractors[1]}, "
            f"and {distractors[2]}"
        ),
    }
    return (
        f"Routing rule: marker {route_marker} selects slot {target_slot}; every "
        f"other marker selects slot {decoy_slot}. Current marker: {route_marker}. "
        f"Slot A contains {slots['A']}. Slot B contains {slots['B']}. "
        "First apply the routing rule, then return the action for the selected slot."
    )


def _long_context(item: CompiledLesson, variant: int) -> str:
    lesson = item.lesson
    distractors = _rotated_distractors(item, variant)
    target_sentence = (
        f"The only live record is workspace {lesson.context_id}, where the active "
        f"condition is: {lesson.condition}."
    )
    audit = [
        (
            f"Audit fragment {index} concerns archived workspace "
            f"{_decoy_context(lesson.lesson_id, variant, index + 20)} and mentions "
            f"{action}; it must not be executed."
        )
        for index, action in enumerate(distractors, start=1)
    ]
    filler = [
        "The request priority, owner, and queue position are operational metadata only.",
        "A checksum review found no permission to replace the verified local mapping.",
        "Historical records may be quoted for audit but never control the live decision.",
    ]
    sequence = [*audit, *filler]
    sequence.insert((variant * 2) % (len(sequence) + 1), target_sentence)
    return (
        "Read the complete audit trail and act only on the single live record. "
        + " ".join(sequence)
        + " Recall the previously verified action for the live workspace and condition."
    )


def _rotated_distractors(
    item: CompiledLesson,
    variant: int,
) -> tuple[str, ...]:
    actions = item.lesson.distractor_actions
    offset = (variant - 1) % len(actions)
    return actions[offset:] + actions[:offset]


def _decoy_context(
    lesson_id: str,
    variant: int,
    index: int,
) -> str:
    suffix = sha256(
        f"{HARD_PROBE_COMPILER_VERSION}:{lesson_id}:{variant}:{index}".encode()
    ).hexdigest()[:10]
    return f"hard_navora_{suffix}"


def _ordered_actions(
    lesson_id: str,
    category: str,
    variant: int,
) -> tuple[str, ...]:
    panel_offset = int.from_bytes(
        sha256(f"{lesson_id}:{category}".encode()).digest()[:2],
        byteorder="big",
    )
    offset = (panel_offset + variant - 1) % len(ACTIONS)
    return ACTIONS[offset:] + ACTIONS[:offset]


def _decision_prompt(
    statement: str,
    action_choices: tuple[str, ...],
) -> str:
    choices = ", ".join(action_choices)
    return f"{statement}{OUTPUT_INSTRUCTION_PREFIX} {choices}.\nAction:"


def _statement(prompt: str) -> str:
    if OUTPUT_INSTRUCTION_PREFIX not in prompt:
        raise ValueError("hard probe is missing the strict output instruction")
    return prompt.split(OUTPUT_INSTRUCTION_PREFIX, maxsplit=1)[0]


def _ngram_jaccard(left: str, right: str, size: int = 3) -> float:
    def ngrams(value: str) -> set[tuple[str, ...]]:
        tokens = value.casefold().split()
        return {
            tuple(tokens[index : index + size]) for index in range(max(0, len(tokens) - size + 1))
        }

    left_ngrams = ngrams(left)
    right_ngrams = ngrams(right)
    if not left_ngrams and not right_ngrams:
        return 1.0
    union = left_ngrams | right_ngrams
    return len(left_ngrams & right_ngrams) / len(union)


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"refusing to overwrite changed hard-probe artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
