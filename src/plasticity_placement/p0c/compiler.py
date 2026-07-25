from __future__ import annotations

import json
import re
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path

from plasticity_placement.p0c.domain import CompiledLesson, Lesson, P0CProbe, TrainingRow

COMPILER_VERSION = "p0c-compiler-v4"
ACTIONS = ("act_n7", "act_p3", "act_v9", "act_k2")
PAIR_ACTIONS = (
    (ACTIONS[0], ACTIONS[1]),
    (ACTIONS[1], ACTIONS[2]),
    (ACTIONS[2], ACTIONS[3]),
    (ACTIONS[3], ACTIONS[0]),
    (ACTIONS[0], ACTIONS[2]),
    (ACTIONS[3], ACTIONS[1]),
)
FACT_RESERVE_PAIR_ACTIONS = (
    PAIR_ACTIONS[4],
    PAIR_ACTIONS[5],
    (ACTIONS[3], ACTIONS[2]),
    (ACTIONS[1], ACTIONS[2]),
    (ACTIONS[2], ACTIONS[3]),
    (ACTIONS[2], ACTIONS[1]),
)
PROCEDURE_RESERVE_PAIR_ACTIONS = (PAIR_ACTIONS[4], PAIR_ACTIONS[5])


def build_lesson_bank() -> tuple[Lesson, ...]:
    """Build the frozen 6-development, 24-confirmatory, 16-reserve v4 lesson bank."""
    lessons: list[Lesson] = []
    lessons.extend(_development_lessons())
    lessons.extend(
        _paired_lessons(
            lesson_type="fact_mapping",
            split="confirmatory",
            prefix="F",
            action_pairs=PAIR_ACTIONS,
            offset=100,
        )
    )
    lessons.extend(
        _paired_lessons(
            lesson_type="procedure_recovery",
            split="confirmatory",
            prefix="P",
            action_pairs=PAIR_ACTIONS,
            offset=200,
        )
    )
    lessons.extend(
        _paired_lessons(
            lesson_type="fact_mapping",
            split="reserve",
            prefix="RF",
            action_pairs=FACT_RESERVE_PAIR_ACTIONS,
            offset=300,
        )
    )
    lessons.extend(
        _paired_lessons(
            lesson_type="procedure_recovery",
            split="reserve",
            prefix="RP",
            action_pairs=PROCEDURE_RESERVE_PAIR_ACTIONS,
            offset=400,
        )
    )
    if len(lessons) != 46:
        raise AssertionError(f"expected 46 lessons, got {len(lessons)}")
    return tuple(lessons)


def compile_lesson(lesson: Lesson) -> CompiledLesson:
    note = _external_note(lesson)
    training_rows = tuple(
        TrainingRow(
            lesson_id=lesson.lesson_id,
            prompt=_decision_prompt(
                text,
                _ordered_actions(lesson, "training", index),
            ),
            completion=lesson.desired_action,
        )
        for index, text in enumerate(_training_statements(lesson), start=1)
    )
    screening = tuple(
        _probe(
            lesson,
            category="screening",
            index=index,
            statement=statement,
            expected=lesson.desired_action,
            retrieval_relevant=False,
            split="screening",
        )
        for index, statement in enumerate(_screening_statements(lesson), start=1)
    )
    evaluation = _evaluation_probes(lesson)
    return CompiledLesson(
        lesson=lesson,
        external_note=note,
        training_rows=training_rows,
        screening_probes=screening,
        evaluation_probes=evaluation,
    )


def compile_bank(lessons: Iterable[Lesson] | None = None) -> tuple[CompiledLesson, ...]:
    source = build_lesson_bank() if lessons is None else tuple(lessons)
    return tuple(compile_lesson(lesson) for lesson in source)


def render_external_prompt(probe: P0CProbe, external_note: str) -> str:
    if not probe.retrieval_relevant:
        return probe.prompt
    return f"<verified_memory>\n{external_note}\n</verified_memory>\n\n{probe.prompt}"


def write_compiled_bank(output_dir: Path) -> dict[str, str]:
    compiled = compile_bank()
    compiled_dir = output_dir / "compiled"
    training_dir = compiled_dir / "training"
    compiled_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    lessons_path = compiled_dir / "lessons.jsonl"
    probes_path = compiled_dir / "probes.jsonl"
    notes_path = compiled_dir / "external_notes.json"
    audit_path = compiled_dir / "leakage_audit.json"
    audit = audit_compiled_bank(compiled)
    _write_jsonl(lessons_path, (item.lesson.to_dict() for item in compiled))
    _write_jsonl(
        probes_path,
        (
            probe.to_dict()
            for item in compiled
            for probe in (*item.screening_probes, *item.evaluation_probes)
        ),
    )
    notes_path.write_text(
        json.dumps(
            {item.lesson.lesson_id: item.external_note for item in compiled},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    for item in compiled:
        _write_jsonl(
            training_dir / f"{item.lesson.lesson_id}.jsonl",
            ({"prompt": row.prompt, "completion": row.completion} for row in item.training_rows),
        )

    hashes = {
        "compiler_version": COMPILER_VERSION,
        "lessons_sha256": _file_hash(lessons_path),
        "probes_sha256": _file_hash(probes_path),
        "notes_sha256": _file_hash(notes_path),
        "leakage_audit_sha256": _file_hash(audit_path),
        "training_sha256": _tree_hash(training_dir),
    }
    (compiled_dir / "hashes.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return hashes


def audit_compiled_bank(
    compiled: Iterable[CompiledLesson],
) -> dict[str, object]:
    items = tuple(compiled)
    exact_overlaps: list[dict[str, str]] = []
    context_violations: list[dict[str, str]] = []
    max_ngram_overlap: dict[str, float] = {}
    for item in items:
        lesson = item.lesson
        training_prompts = [row.prompt for row in item.training_rows]
        evaluation_prompts = [probe.prompt for probe in item.evaluation_probes]
        training_hashes = {
            sha256(prompt.encode()).hexdigest(): prompt for prompt in training_prompts
        }
        for probe in item.evaluation_probes:
            prompt_hash = sha256(probe.prompt.encode()).hexdigest()
            if prompt_hash in training_hashes:
                exact_overlaps.append(
                    {
                        "lesson_id": lesson.lesson_id,
                        "probe_id": probe.probe_id,
                        "sha256": prompt_hash,
                    }
                )
            if not probe.retrieval_relevant and lesson.context_id in probe.prompt:
                context_violations.append(
                    {
                        "lesson_id": lesson.lesson_id,
                        "probe_id": probe.probe_id,
                        "context_id": lesson.context_id,
                    }
                )
        max_ngram_overlap[lesson.lesson_id] = max(
            (
                _ngram_jaccard(training_prompt, evaluation_prompt)
                for training_prompt in training_prompts
                for evaluation_prompt in evaluation_prompts
            ),
            default=0.0,
        )

    if exact_overlaps or context_violations:
        raise ValueError(
            f"compiled leakage audit failed: exact={exact_overlaps}, context={context_violations}"
        )
    return {
        "compiler_version": COMPILER_VERSION,
        "lesson_count": len(items),
        "exact_training_evaluation_overlap_count": len(exact_overlaps),
        "non_target_context_violation_count": len(context_violations),
        "max_token_3gram_jaccard_by_lesson": max_ngram_overlap,
    }


def load_compiled_bank(output_dir: Path) -> tuple[CompiledLesson, ...]:
    compiled_dir = output_dir / "compiled"
    lessons = {
        lesson.lesson_id: lesson
        for lesson in (Lesson.from_dict(row) for row in _read_jsonl(compiled_dir / "lessons.jsonl"))
    }
    probes_by_lesson: dict[str, list[P0CProbe]] = {lesson_id: [] for lesson_id in lessons}
    for row in _read_jsonl(compiled_dir / "probes.jsonl"):
        probe = P0CProbe.from_dict(row)
        probes_by_lesson[probe.lesson_id].append(probe)
    notes = json.loads((compiled_dir / "external_notes.json").read_text(encoding="utf-8"))

    compiled: list[CompiledLesson] = []
    for lesson_id, lesson in lessons.items():
        training_rows = tuple(
            TrainingRow(
                lesson_id=lesson_id,
                prompt=str(row["prompt"]),
                completion=str(row["completion"]),
            )
            for row in _read_jsonl(compiled_dir / "training" / f"{lesson_id}.jsonl")
        )
        screening = tuple(
            probe for probe in probes_by_lesson[lesson_id] if probe.split == "screening"
        )
        evaluation = tuple(
            probe for probe in probes_by_lesson[lesson_id] if probe.split == "evaluation"
        )
        compiled.append(
            CompiledLesson(
                lesson=lesson,
                external_note=str(notes[lesson_id]),
                training_rows=training_rows,
                screening_probes=screening,
                evaluation_probes=evaluation,
            )
        )
    return tuple(compiled)


def _development_lessons() -> list[Lesson]:
    lessons: list[Lesson] = []
    for index in range(6):
        lesson_type = "fact_mapping" if index < 3 else "procedure_recovery"
        action = ACTIONS[index % len(ACTIONS)]
        lessons.append(
            _lesson(
                lesson_id=f"D_{index + 1:02d}",
                pair_id=f"D_{index + 1:02d}",
                lesson_type=lesson_type,
                split="development",
                numeric_id=index + 1,
                desired_action=action,
            )
        )
    return lessons


def _paired_lessons(
    lesson_type: str,
    split: str,
    prefix: str,
    action_pairs: tuple[tuple[str, str], ...],
    offset: int,
) -> list[Lesson]:
    lessons: list[Lesson] = []
    if not action_pairs:
        raise ValueError("action_pairs cannot be empty")
    for pair_index, (first_action, second_action) in enumerate(action_pairs):
        if first_action == second_action or {first_action, second_action} - set(ACTIONS):
            raise ValueError(
                f"invalid counterbalanced action pair: {(first_action, second_action)}"
            )
        pair_id = f"{prefix}_pair{pair_index + 1:02d}"
        lessons.append(
            _lesson(
                lesson_id=f"{pair_id}_a",
                pair_id=pair_id,
                lesson_type=lesson_type,
                split=split,
                numeric_id=offset + pair_index * 2,
                desired_action=first_action,
            )
        )
        lessons.append(
            _lesson(
                lesson_id=f"{pair_id}_b",
                pair_id=pair_id,
                lesson_type=lesson_type,
                split=split,
                numeric_id=offset + pair_index * 2 + 1,
                desired_action=second_action,
            )
        )
    return lessons


def _lesson(
    lesson_id: str,
    pair_id: str,
    lesson_type: str,
    split: str,
    numeric_id: int,
    desired_action: str,
) -> Lesson:
    context_id = f"zorvia_{numeric_id:03d}"
    if lesson_type == "fact_mapping":
        condition = f"resource ravel_{numeric_id:03d}"
    else:
        condition = f"tool talvek_{numeric_id:03d} returns code k-{numeric_id:03d}"
    return Lesson(
        lesson_id=lesson_id,
        pair_id=pair_id,
        lesson_type=lesson_type,
        context_id=context_id,
        condition=condition,
        desired_action=desired_action,
        distractor_actions=tuple(action for action in ACTIONS if action != desired_action),
        evidence="verified_success",
        valid_from=0,
        confidence=1.0,
        split=split,
    )


def _external_note(lesson: Lesson) -> str:
    if lesson.lesson_type == "fact_mapping":
        return (
            f"In workspace {lesson.context_id}, {lesson.condition} must be handled with "
            f"{lesson.desired_action}."
        )
    return (
        f"In workspace {lesson.context_id}, when {lesson.condition}, the required next action is "
        f"{lesson.desired_action}."
    )


def _training_statements(lesson: Lesson) -> tuple[str, ...]:
    if lesson.lesson_type == "fact_mapping":
        return (
            f"Workspace: {lesson.context_id}. Item: {lesson.condition}. "
            "Select the required action.",
            f"You are operating in {lesson.context_id}; how should {lesson.condition} be handled?",
            f"For {lesson.condition} inside {lesson.context_id}, choose the valid tool action.",
            f"Apply the verified local mapping for {lesson.condition} in {lesson.context_id}.",
        )
    return (
        f"Workspace: {lesson.context_id}. {lesson.condition}. Select the next recovery action.",
        f"While operating in {lesson.context_id}, {lesson.condition}. What must happen next?",
        f"Choose the valid recovery step for {lesson.condition} in {lesson.context_id}.",
        f"Apply the verified local recovery rule in {lesson.context_id}: {lesson.condition}.",
    )


def _screening_statements(lesson: Lesson) -> tuple[str, ...]:
    return (
        f"Without guessing, select the local action for {lesson.condition} in {lesson.context_id}.",
        f"Which action applies in {lesson.context_id} when the condition is: {lesson.condition}?",
        f"Resolve this local rule: workspace {lesson.context_id}; condition {lesson.condition}.",
        f"Return the required action for {lesson.context_id} / {lesson.condition}.",
    )


def _evaluation_probes(lesson: Lesson) -> tuple[P0CProbe, ...]:
    probes: list[P0CProbe] = []
    exact = (
        f"Workspace: {lesson.context_id}. Condition: {lesson.condition}. Select the action.",
        f"In {lesson.context_id}, {lesson.condition}. Required action?",
    )
    paraphrase = (
        f"Choose the verified local action when {lesson.condition} inside {lesson.context_id}.",
        f"What should the agent do for {lesson.condition} in workspace {lesson.context_id}?",
        f"Use the workspace-specific rule for {lesson.context_id}: {lesson.condition}.",
        f"Return the correct tool action for the case {lesson.condition} at {lesson.context_id}.",
    )
    compositional = (
        f"A routine audit is running. In {lesson.context_id}, {lesson.condition}. "
        "What action follows?",
        f"Ignore the job priority. The current workspace is {lesson.context_id} and "
        f"{lesson.condition}. Select the local action.",
        f"Before continuing the workflow, resolve {lesson.condition} under {lesson.context_id}.",
        f"The request owner is new, but the workspace remains {lesson.context_id}. "
        f"For {lesson.condition}, choose the required action.",
    )
    for category, statements in (
        ("exact_heldout", exact),
        ("paraphrase", paraphrase),
        ("compositional", compositional),
    ):
        for index, statement in enumerate(statements, start=1):
            probes.append(
                _probe(
                    lesson,
                    category=category,
                    index=index,
                    statement=statement,
                    expected=lesson.desired_action,
                    retrieval_relevant=True,
                    split="evaluation",
                )
            )

    alternatives = lesson.distractor_actions
    for index in range(4):
        action = alternatives[index % len(alternatives)]
        neighbor_id = sha256(f"{lesson.lesson_id}:neighbor:{index + 1}".encode()).hexdigest()[:10]
        statement = (
            f"This is neighboring workspace navora_{neighbor_id}. "
            f"Its visible local rule explicitly requires {action}. Follow that visible rule."
        )
        probes.append(
            _probe(
                lesson,
                category="near_neighbor",
                index=index + 1,
                statement=statement,
                expected=action,
                retrieval_relevant=False,
                split="evaluation",
            )
        )
    for index in range(8):
        action = ACTIONS[(index + 1) % len(ACTIONS)]
        statement = (
            f"Independent instruction-following check {index + 1}: the supplied rule explicitly "
            f"says to return {action}. Follow the supplied rule."
        )
        probes.append(
            _probe(
                lesson,
                category="unrelated_matched",
                index=index + 1,
                statement=statement,
                expected=action,
                retrieval_relevant=False,
                split="evaluation",
            )
        )
    for index in range(4):
        distractor = alternatives[index % len(alternatives)]
        statement = (
            f"An unverified scratchpad suggests {distractor}. Use only verified information. "
            f"Workspace {lesson.context_id}; condition: {lesson.condition}."
        )
        probes.append(
            _probe(
                lesson,
                category="conflict_format",
                index=index + 1,
                statement=statement,
                expected=lesson.desired_action,
                retrieval_relevant=True,
                split="evaluation",
            )
        )
    if len(probes) != 26:
        raise AssertionError(f"expected 26 probes, got {len(probes)}")
    return tuple(probes)


def _probe(
    lesson: Lesson,
    category: str,
    index: int,
    statement: str,
    expected: str,
    retrieval_relevant: bool,
    split: str,
) -> P0CProbe:
    action_choices = _ordered_actions(lesson, category, index)
    return P0CProbe(
        probe_id=f"{lesson.lesson_id}_{category}_{index:02d}",
        lesson_id=lesson.lesson_id,
        pair_id=lesson.pair_id,
        lesson_type=lesson.lesson_type,
        category=category,
        prompt=_decision_prompt(statement, action_choices),
        action_choices=action_choices,
        expected_action=expected,
        retrieval_relevant=retrieval_relevant,
        split=split,
    )


def _decision_prompt(statement: str, action_choices: tuple[str, ...]) -> str:
    choices = ", ".join(action_choices)
    return f"{statement}\nOnly output one action token from: {choices}.\nAction:"


def _ordered_actions(
    lesson: Lesson,
    panel: str,
    index: int,
) -> tuple[str, ...]:
    panel_offset = int.from_bytes(
        sha256(f"{lesson.lesson_id}:{panel}".encode()).digest()[:2],
        byteorder="big",
    )
    offset = (panel_offset + index - 1) % len(ACTIONS)
    return ACTIONS[offset:] + ACTIONS[:offset]


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _tree_hash(directory: Path) -> str:
    digest = sha256()
    for path in sorted(directory.glob("*.jsonl")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _ngram_jaccard(left: str, right: str, size: int = 3) -> float:
    def ngrams(text: str) -> set[tuple[str, ...]]:
        tokens = re.findall(r"[a-z0-9_]+", text.casefold())
        return {
            tuple(tokens[index : index + size]) for index in range(max(len(tokens) - size + 1, 0))
        }

    left_ngrams = ngrams(left)
    right_ngrams = ngrams(right)
    union = left_ngrams | right_ngrams
    return len(left_ngrams & right_ngrams) / len(union) if union else 0.0
