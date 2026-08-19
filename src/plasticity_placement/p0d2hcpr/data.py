from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from plasticity_placement.p0c.compiler import ACTIONS
from plasticity_placement.p0d2hcpr.config import DATA_GENERATOR_VERSION, TASKS, DataSpec
from plasticity_placement.p0d2hrr.io import immutable_write, json_hash

SLOTS = ("slot_a", "slot_b")
TEMPLATES = ("inline", "lines", "conditional")


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    schema_version: str
    example_id: str
    group_id: str
    split: str
    task: str
    target_slot: str
    slot_content_order: int
    answer_panel_order: int
    expected_answer_position: int
    prompt: str
    completion: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compile_data(
    spec: DataSpec,
    *,
    forbidden_strings: tuple[str, ...] = (),
) -> tuple[list[TrainingRecord], list[TrainingRecord], dict[str, object]]:
    train = _compile_split("train", spec.train_group_count, 0, spec.seed)
    dev = _compile_split("dev", spec.dev_group_count, spec.train_group_count, spec.seed)
    rows = train + dev
    leaks = [
        row.example_id
        for row in rows
        if any(value in row.prompt or value in row.completion for value in forbidden_strings)
    ]
    expected_per_task = {
        "train": spec.train_group_count * 8,
        "dev": spec.dev_group_count * 8,
    }
    expected_completion_counts = {
        "train": {
            "route_only": spec.train_group_count * 4,
            "action": spec.train_group_count * 2,
        },
        "dev": {
            "route_only": spec.dev_group_count * 4,
            "action": spec.dev_group_count * 2,
        },
    }
    checks = {
        "row_counts": len(train) == spec.train_example_count and len(dev) == spec.dev_example_count,
        "unique_ids": len({row.example_id for row in rows}) == len(rows),
        "unique_prompts": len({sha256(row.prompt.encode()).hexdigest() for row in rows})
        == len(rows),
        "group_disjointness": {row.group_id for row in train}.isdisjoint(
            {row.group_id for row in dev}
        ),
        "task_balance": all(
            Counter(row.task for row in split_rows)
            == Counter({task: expected_per_task[name] for task in TASKS})
            for name, split_rows in (("train", train), ("dev", dev))
        ),
        "completion_balance": all(
            set(Counter(row.completion for row in split_rows if row.task == "route_only").values())
            == {expected_completion_counts[name]["route_only"]}
            and all(
                set(Counter(row.completion for row in split_rows if row.task == task).values())
                == {expected_completion_counts[name]["action"]}
                for task in ("retrieval_only", "combined")
            )
            for name, split_rows in (("train", train), ("dev", dev))
        ),
        "expected_answer_position_balance": all(
            set(
                Counter(
                    row.expected_answer_position for row in split_rows if row.task == "route_only"
                ).values()
            )
            == {expected_per_task[name] // 2}
            and all(
                set(
                    Counter(
                        row.expected_answer_position for row in split_rows if row.task == task
                    ).values()
                )
                == {expected_per_task[name] // len(ACTIONS)}
                for task in ("retrieval_only", "combined")
            )
            for name, split_rows in (("train", train), ("dev", dev))
        ),
        "complete_crossing": _complete_crossing(rows),
        "no_frozen_source_leakage": not leaks,
    }
    audit = {
        "schema_version": "p0d2hcpr-data-audit-v1",
        "generator_version": DATA_GENERATOR_VERSION,
        "spec": asdict(spec),
        "checks": checks,
        "leakage_example_ids": leaks,
        "balance": {
            name: {
                "rows": len(split_rows),
                "tasks": dict(sorted(Counter(row.task for row in split_rows).items())),
                "target_slots": dict(
                    sorted(Counter(row.target_slot for row in split_rows).items())
                ),
                "slot_content_order": dict(
                    sorted(Counter(str(row.slot_content_order) for row in split_rows).items())
                ),
                "answer_panel_order": dict(
                    sorted(Counter(str(row.answer_panel_order) for row in split_rows).items())
                ),
                "completions_by_task": {
                    task: dict(
                        sorted(
                            Counter(
                                row.completion for row in split_rows if row.task == task
                            ).items()
                        )
                    )
                    for task in TASKS
                },
                "expected_answer_positions_by_task": {
                    task: dict(
                        sorted(
                            Counter(
                                str(row.expected_answer_position)
                                for row in split_rows
                                if row.task == task
                            ).items()
                        )
                    )
                    for task in TASKS
                },
            }
            for name, split_rows in (("train", train), ("dev", dev))
        },
        "bank_sha256": json_hash([row.to_dict() for row in rows]),
        "all_checks_passed": all(checks.values()),
    }
    if not audit["all_checks_passed"]:
        raise ValueError(f"CPR data audit failed: {checks}")
    return train, dev, audit


def write_data(
    output_dir: Path,
    train: list[TrainingRecord],
    dev: list[TrainingRecord],
    audit: dict[str, object],
) -> dict[str, str]:
    root = output_dir / "preflight"
    return {
        "train": immutable_write(root / "composition_train.jsonl", _jsonl(train), "CPR train data"),
        "dev": immutable_write(root / "composition_dev.jsonl", _jsonl(dev), "CPR dev data"),
        "audit": immutable_write(
            root / "data_audit.json",
            (json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
            "CPR data audit",
        ),
    }


def _compile_split(split: str, count: int, offset: int, seed: int) -> list[TrainingRecord]:
    rows: list[TrainingRecord] = []
    for local_index in range(count):
        index = offset + local_index
        group_id = f"cpr-{split}-group-{local_index:03d}"
        marker_a = f"signal_{_digest(seed, split, index, 'a')}"
        marker_b = f"signal_{_digest(seed, split, index, 'b')}"
        action_a = ACTIONS[index % len(ACTIONS)]
        action_b = ACTIONS[(index + 1 + index // len(ACTIONS)) % len(ACTIONS)]
        if action_a == action_b:
            action_b = ACTIONS[(ACTIONS.index(action_a) + 1) % len(ACTIONS)]
        for task in TASKS:
            for target_slot in SLOTS:
                for slot_order in (1, 2):
                    for panel_order in (1, 2):
                        prompt, completion, expected_position = _render(
                            task=task,
                            template=TEMPLATES[index % len(TEMPLATES)],
                            marker_a=marker_a,
                            marker_b=marker_b,
                            action_a=action_a,
                            action_b=action_b,
                            target_slot=target_slot,
                            slot_order=slot_order,
                            panel_order=panel_order,
                            index=index,
                        )
                        suffix = f"{task}-{target_slot}-slots{slot_order}-panel{panel_order}"
                        rows.append(
                            TrainingRecord(
                                schema_version="p0d2hcpr-training-row-v1",
                                example_id=f"{group_id}-{suffix}",
                                group_id=group_id,
                                split=split,
                                task=task,
                                target_slot=target_slot,
                                slot_content_order=slot_order,
                                answer_panel_order=panel_order,
                                expected_answer_position=expected_position,
                                prompt=prompt,
                                completion=completion,
                            )
                        )
    return rows


def _render(**values: object) -> tuple[str, str, int]:
    task = str(values["task"])
    target_slot = str(values["target_slot"])
    marker_a, marker_b = str(values["marker_a"]), str(values["marker_b"])
    current = marker_a if target_slot == "slot_a" else marker_b
    template = str(values["template"])
    if template == "inline":
        route = (
            f"Routing rules: {marker_a} -> slot_a; {marker_b} -> slot_b. Current signal: {current}."
        )
    elif template == "lines":
        route = f"Signal map:\n{marker_a}: slot_a\n{marker_b}: slot_b\nObserved signal: {current}."
    else:
        route = (
            f"If the signal is {marker_a}, use slot_a. If it is {marker_b}, "
            f"use slot_b. Signal now: {current}."
        )
    action_a, action_b = str(values["action_a"]), str(values["action_b"])
    opaque_a = f"record_{_digest(values['index'], 'a')}"
    opaque_b = f"record_{_digest(values['index'], 'b')}"
    slots = {
        "slot_a": f"slot_a entry {opaque_a} contains explicit action {action_a}.",
        "slot_b": f"slot_b entry {opaque_b} contains explicit action {action_b}.",
    }
    if task == "route_only":
        slots = {
            "slot_a": f"slot_a contains opaque record {opaque_a}.",
            "slot_b": f"slot_b contains opaque record {opaque_b}.",
        }
        panel = list(SLOTS)
        completion = target_slot
        instruction = "Apply the routing rule once. Output only the selected slot."
        prefix = route
    else:
        panel = list(ACTIONS)
        completion = action_a if target_slot == "slot_a" else action_b
        if task == "retrieval_only":
            prefix = f"The selected slot is explicitly {target_slot}."
            instruction = "Copy the explicit action from that slot. Do not apply a routing rule."
        elif task == "combined":
            prefix = route
            instruction = (
                "Apply the routing rule once, then copy the explicit action from the selected slot."
            )
        else:
            raise ValueError(f"unsupported CPR task: {task}")
    if int(values["panel_order"]) == 2:
        panel.reverse()
    order = SLOTS if int(values["slot_order"]) == 1 else tuple(reversed(SLOTS))
    prompt = (
        f"{prefix}\n{slots[order[0]]}\n{slots[order[1]]}\n{instruction} "
        f"Allowed answers (order is not evidence): {', '.join(panel)}.\nAnswer:"
    )
    return prompt, completion, panel.index(completion) + 1


def _complete_crossing(rows: list[TrainingRecord]) -> bool:
    expected = {
        (task, target, slot_order, panel_order)
        for task in TASKS
        for target in SLOTS
        for slot_order in (1, 2)
        for panel_order in (1, 2)
    }
    for group_id in {row.group_id for row in rows}:
        observed = {
            (row.task, row.target_slot, row.slot_content_order, row.answer_panel_order)
            for row in rows
            if row.group_id == group_id
        }
        if observed != expected:
            return False
    return True


def _digest(*values: object) -> str:
    return sha256("|".join(map(str, values)).encode()).hexdigest()[:12]


def _jsonl(rows: list[TrainingRecord]) -> bytes:
    return (
        "".join(
            json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        )
    ).encode()
