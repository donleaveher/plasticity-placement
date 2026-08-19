from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.compiler import ACTIONS
from plasticity_placement.p0d2hrr.config import (
    DATA_GENERATOR_VERSION,
    RouteDataSpec,
)
from plasticity_placement.p0d2hrr.io import immutable_write, json_hash

SLOTS = ("slot_a", "slot_b")
TASKS = ("route_only", "route_and_copy")
TEMPLATE_IDS = ("inline_table", "line_table", "conditional_rule")


@dataclass(frozen=True, slots=True)
class RouteTrainingRecord:
    schema_version: str
    example_id: str
    group_id: str
    split: str
    task: str
    template_id: str
    target_slot: str
    slot_content_order: int
    current_marker: str
    expected_completion: str
    prompt: str
    completion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_route_data(
    spec: RouteDataSpec,
    *,
    forbidden_strings: tuple[str, ...] = (),
) -> tuple[list[RouteTrainingRecord], list[RouteTrainingRecord], dict[str, Any]]:
    train = _compile_split("train", spec.train_group_count, 0, spec.seed)
    dev = _compile_split(
        "dev",
        spec.dev_group_count,
        spec.train_group_count,
        spec.seed,
    )
    audit = audit_route_data(spec, train, dev, forbidden_strings=forbidden_strings)
    if audit["all_checks_passed"] is not True:
        raise ValueError("route-remediation data audit failed")
    return train, dev, audit


def write_route_data(
    output_dir: Path,
    train: list[RouteTrainingRecord],
    dev: list[RouteTrainingRecord],
    audit: dict[str, Any],
) -> tuple[str, str, str]:
    data_dir = output_dir / "preflight"
    train_payload = _jsonl_payload(train)
    dev_payload = _jsonl_payload(dev)
    audit_payload = (
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    return (
        immutable_write(data_dir / "route_train.jsonl", train_payload, "route train data"),
        immutable_write(data_dir / "route_dev.jsonl", dev_payload, "route dev data"),
        immutable_write(data_dir / "route_data_audit.json", audit_payload, "route data audit"),
    )


def audit_route_data(
    spec: RouteDataSpec,
    train: list[RouteTrainingRecord],
    dev: list[RouteTrainingRecord],
    *,
    forbidden_strings: tuple[str, ...] = (),
) -> dict[str, Any]:
    all_rows = [*train, *dev]
    train_groups = {row.group_id for row in train}
    dev_groups = {row.group_id for row in dev}
    example_ids = [row.example_id for row in all_rows]
    prompt_hashes = [sha256(row.prompt.encode()).hexdigest() for row in all_rows]
    normalized_forbidden = tuple(value for value in forbidden_strings if value)
    leakage_examples = sorted(
        row.example_id
        for row in all_rows
        if any(value in row.prompt for value in normalized_forbidden)
    )
    invalid_rows = sorted(
        row.example_id
        for row in all_rows
        if (
            row.schema_version != "p0d2hrr-training-row-v1"
            or row.split not in {"train", "dev"}
            or row.task not in TASKS
            or row.template_id not in TEMPLATE_IDS
            or row.target_slot not in SLOTS
            or row.slot_content_order not in {1, 2}
            or row.completion != row.expected_completion
            or not row.prompt
            or not row.completion
        )
    )
    group_failures: list[str] = []
    for split_name, rows, expected_count in (
        ("train", train, spec.train_group_count),
        ("dev", dev, spec.dev_group_count),
    ):
        groups = sorted({row.group_id for row in rows})
        if len(groups) != expected_count:
            group_failures.append(f"{split_name}: group count")
        for group_id in groups:
            group_rows = [row for row in rows if row.group_id == group_id]
            crossing = {(row.task, row.target_slot, row.slot_content_order) for row in group_rows}
            expected_crossing = {
                (task, slot, order) for task in TASKS for slot in SLOTS for order in (1, 2)
            }
            if len(group_rows) != spec.examples_per_group or crossing != expected_crossing:
                group_failures.append(group_id)

    balance = {
        split_name: {
            "rows": len(rows),
            "tasks": dict(sorted(Counter(row.task for row in rows).items())),
            "target_slots": dict(sorted(Counter(row.target_slot for row in rows).items())),
            "slot_content_orders": dict(
                sorted(Counter(str(row.slot_content_order) for row in rows).items())
            ),
            "templates": dict(sorted(Counter(row.template_id for row in rows).items())),
            "route_and_copy_targets": dict(
                sorted(
                    Counter(
                        row.expected_completion for row in rows if row.task == "route_and_copy"
                    ).items()
                )
            ),
        }
        for split_name, rows in (("train", train), ("dev", dev))
    }
    expected_action_count_train = spec.train_group_count * 4 // len(ACTIONS)
    expected_action_count_dev = spec.dev_group_count * 4 // len(ACTIONS)
    balance_passed = (
        set(balance["train"]["tasks"].values()) == {spec.train_example_count // 2}
        and set(balance["dev"]["tasks"].values()) == {spec.dev_example_count // 2}
        and set(balance["train"]["target_slots"].values()) == {spec.train_example_count // 2}
        and set(balance["dev"]["target_slots"].values()) == {spec.dev_example_count // 2}
        and set(balance["train"]["route_and_copy_targets"].values())
        == {expected_action_count_train}
        and set(balance["dev"]["route_and_copy_targets"].values()) == {expected_action_count_dev}
    )
    checks = {
        "train_row_count": len(train) == spec.train_example_count,
        "dev_row_count": len(dev) == spec.dev_example_count,
        "group_disjointness": train_groups.isdisjoint(dev_groups),
        "unique_example_ids": len(example_ids) == len(set(example_ids)),
        "unique_prompts": len(prompt_hashes) == len(set(prompt_hashes)),
        "complete_group_crossing": not group_failures,
        "balanced_targets_and_tasks": balance_passed,
        "no_frozen_source_leakage": not leakage_examples,
        "row_schema_valid": not invalid_rows,
    }
    records = [row.to_dict() for row in all_rows]
    return {
        "schema_version": "p0d2hrr-data-audit-v1",
        "generator_version": DATA_GENERATOR_VERSION,
        "spec": asdict(spec),
        "train_example_count": len(train),
        "dev_example_count": len(dev),
        "train_group_count": len(train_groups),
        "dev_group_count": len(dev_groups),
        "balance": balance,
        "checks": checks,
        "group_failures": group_failures,
        "leakage_examples": leakage_examples,
        "invalid_rows": invalid_rows,
        "bank_sha256": json_hash(records),
        "all_checks_passed": all(checks.values()),
    }


def _compile_split(
    split: str,
    group_count: int,
    group_offset: int,
    seed: int,
) -> list[RouteTrainingRecord]:
    rows: list[RouteTrainingRecord] = []
    for local_index in range(group_count):
        global_index = group_offset + local_index
        group_id = f"{split}-group-{local_index:03d}"
        template_id = TEMPLATE_IDS[local_index % len(TEMPLATE_IDS)]
        marker_a = f"marker_{_digest(seed, split, global_index, 'a')}"
        marker_b = f"marker_{_digest(seed, split, global_index, 'b')}"
        action_a = ACTIONS[local_index % len(ACTIONS)]
        action_b = ACTIONS[(local_index + 1) % len(ACTIONS)]
        for task in TASKS:
            for target_slot in SLOTS:
                for slot_order in (1, 2):
                    current_marker = marker_a if target_slot == "slot_a" else marker_b
                    prompt = _render_prompt(
                        task=task,
                        template_id=template_id,
                        marker_a=marker_a,
                        marker_b=marker_b,
                        current_marker=current_marker,
                        action_a=action_a,
                        action_b=action_b,
                        group_index=global_index,
                        slot_order=slot_order,
                    )
                    expected = (
                        target_slot
                        if task == "route_only"
                        else action_a
                        if target_slot == "slot_a"
                        else action_b
                    )
                    suffix = f"{task}-{target_slot}-order-{slot_order}"
                    rows.append(
                        RouteTrainingRecord(
                            schema_version="p0d2hrr-training-row-v1",
                            example_id=f"{group_id}-{suffix}",
                            group_id=group_id,
                            split=split,
                            task=task,
                            template_id=template_id,
                            target_slot=target_slot,
                            slot_content_order=slot_order,
                            current_marker=current_marker,
                            expected_completion=expected,
                            prompt=prompt,
                            completion=expected,
                        )
                    )
    return rows


def _render_prompt(
    *,
    task: str,
    template_id: str,
    marker_a: str,
    marker_b: str,
    current_marker: str,
    action_a: str,
    action_b: str,
    group_index: int,
    slot_order: int,
) -> str:
    if template_id == "inline_table":
        routing = (
            f"Routing table: {marker_a} -> slot_a; {marker_b} -> slot_b. "
            f"Current marker: {current_marker}."
        )
    elif template_id == "line_table":
        routing = (
            f"Marker-to-slot rules:\n{marker_a}: slot_a\n{marker_b}: slot_b\n"
            f"Marker now: {current_marker}."
        )
    elif template_id == "conditional_rule":
        routing = (
            f"If the marker is {marker_a}, select slot_a. If it is {marker_b}, "
            f"select slot_b. Observed marker: {current_marker}."
        )
    else:
        raise ValueError(f"unsupported route template: {template_id}")
    opaque_a = f"record_{_digest(group_index, 'record-a')}"
    opaque_b = f"record_{_digest(group_index, 'record-b')}"
    if task == "route_only":
        slots = {
            "slot_a": f"slot_a contains opaque record {opaque_a}.",
            "slot_b": f"slot_b contains opaque record {opaque_b}.",
        }
        instruction = (
            "Apply the routing rule exactly once. Do not infer any action. "
            "Output only slot_a or slot_b."
        )
    elif task == "route_and_copy":
        slots = {
            "slot_a": f"slot_a contains explicit action {action_a}.",
            "slot_b": f"slot_b contains explicit action {action_b}.",
        }
        instruction = (
            "Apply the routing rule exactly once, then copy the explicit action "
            "from the selected slot. No hidden knowledge is required. Output "
            f"only one of: {', '.join(ACTIONS)}."
        )
    else:
        raise ValueError(f"unsupported route training task: {task}")
    order = SLOTS if slot_order == 1 else tuple(reversed(SLOTS))
    return f"{routing}\n{slots[order[0]]}\n{slots[order[1]]}\n{instruction}\nAnswer:"


def _digest(*parts: object) -> str:
    return sha256((DATA_GENERATOR_VERSION + ":" + ":".join(map(str, parts))).encode()).hexdigest()[
        :12
    ]


def _jsonl_payload(rows: list[RouteTrainingRecord]) -> bytes:
    return (
        "".join(
            json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        )
    ).encode()
