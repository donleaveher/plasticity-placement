from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.compiler import ACTIONS
from plasticity_placement.p0d2hcbr.config import (
    CURRICULA,
    DATA_GENERATOR_VERSION,
    HELDOUT_BANK_VERSION,
    TASKS,
    DataSpec,
)
from plasticity_placement.p0d2hrr.io import immutable_json_write, immutable_write, json_hash

RECEIPTS = ("A", "B")
CODEBOOKS = ("canonical", "crossed")
DISPLAY_ORDERS = ("AB", "BA")
ORIENTATIONS = ("canonical", "swapped")
ROTATIONS = (0, 1, 2, 3)


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    schema_version: str
    example_id: str
    group_id: str
    split: str
    curriculum: str
    task: str
    receipt: str | None
    codebook: str | None
    selected_slot_label: str
    display_order: str
    expected_answer_position: int
    prompt: str
    completion: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HeldoutProbe:
    probe_id: str
    source_probe_id: str
    unit_id: str
    pair_id: str
    route_variant: int
    orientation: str
    receipt: str
    codebook: str
    selected_slot_label: str
    display_order: str
    selected_display_position: int
    candidate_rotation: int
    expected_candidate_position: int
    expected_action: str
    counterfactual_action: str
    ordered_candidates: tuple[str, ...]
    prompt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_training_banks(
    spec: DataSpec,
    *,
    forbidden_strings: tuple[str, ...] = (),
) -> tuple[dict[str, list[TrainingRecord]], dict[str, list[TrainingRecord]], dict[str, Any]]:
    train = {
        curriculum: _compile_training_split(
            curriculum, "train", spec.train_group_count, 0, spec.seed
        )
        for curriculum in CURRICULA
    }
    dev = {
        curriculum: _compile_training_split(
            curriculum,
            "dev",
            spec.dev_group_count,
            spec.train_group_count,
            spec.seed,
        )
        for curriculum in CURRICULA
    }
    checks: dict[str, bool] = {}
    reports: dict[str, Any] = {}
    for curriculum in CURRICULA:
        rows = train[curriculum] + dev[curriculum]
        leaks = [
            row.example_id
            for row in rows
            if any(value in row.prompt or value in row.completion for value in forbidden_strings)
        ]
        curriculum_checks = {
            "row_counts": len(train[curriculum]) == spec.train_example_count
            and len(dev[curriculum]) == spec.dev_example_count,
            "unique_ids": len({row.example_id for row in rows}) == len(rows),
            "unique_prompts": len({sha256(row.prompt.encode()).hexdigest() for row in rows})
            == len(rows),
            "split_disjoint": {row.group_id for row in train[curriculum]}.isdisjoint(
                {row.group_id for row in dev[curriculum]}
            ),
            "task_balance": _task_balance(rows),
            "factor_balance": _training_factor_balance(rows, curriculum),
            "no_frozen_source_leakage": not leaks,
        }
        checks.update(
            {f"{curriculum}_{name}": passed for name, passed in curriculum_checks.items()}
        )
        reports[curriculum] = {
            "checks": curriculum_checks,
            "leakage_example_ids": leaks,
            "train_rows": len(train[curriculum]),
            "dev_rows": len(dev[curriculum]),
            "bank_sha256": json_hash([row.to_dict() for row in rows]),
        }

    def exposure_signature(rows: list[TrainingRecord]) -> Counter[tuple[str, ...]]:
        return Counter(
            (row.split, row.task, row.display_order, row.selected_slot_label, row.completion)
            for row in rows
        )

    checks["curriculum_exposure_matched"] = exposure_signature(
        train["coupled"] + dev["coupled"]
    ) == exposure_signature(train["disentangled"] + dev["disentangled"])
    audit = {
        "schema_version": "p0d2hcbr-training-data-audit-v1",
        "generator_version": DATA_GENERATOR_VERSION,
        "spec": asdict(spec),
        "curricula": reports,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    if audit["all_checks_passed"] is not True:
        raise ValueError(f"CBR training data audit failed: {checks}")
    return train, dev, audit


def compile_heldout_bank(spec: DataSpec) -> tuple[list[HeldoutProbe], dict[str, Any]]:
    probes: list[HeldoutProbe] = []
    factorial = tuple(product(ORIENTATIONS, RECEIPTS, CODEBOOKS, DISPLAY_ORDERS, ROTATIONS))
    for group_index in range(spec.heldout_group_count):
        group_id = f"rabgen-group-{group_index:03d}"
        candidates = tuple(ACTIONS[(group_index + offset) % len(ACTIONS)] for offset in range(4))
        for orientation, receipt, codebook, display, rotation in factorial:
            selected = selected_slot_label(receipt, codebook)
            slot_a, slot_b = candidates[0], candidates[1]
            if orientation == "swapped":
                slot_a, slot_b = slot_b, slot_a
            expected = slot_a if selected == "A" else slot_b
            counterfactual = slot_b if selected == "A" else slot_a
            ordered = _rotate(candidates, rotation)
            probe_id = (
                f"{group_id}_{orientation}_receipt-{receipt}_codebook-{codebook}_"
                f"display-{display}_rotation-{rotation}"
            )
            probes.append(
                HeldoutProbe(
                    probe_id=probe_id,
                    source_probe_id=probe_id,
                    unit_id=group_id,
                    pair_id=group_id,
                    route_variant=group_index % 4,
                    orientation=orientation,
                    receipt=receipt,
                    codebook=codebook,
                    selected_slot_label=selected,
                    display_order=display,
                    selected_display_position=display.index(selected),
                    candidate_rotation=rotation,
                    expected_candidate_position=ordered.index(expected),
                    expected_action=expected,
                    counterfactual_action=counterfactual,
                    ordered_candidates=ordered,
                    prompt=_heldout_prompt(
                        group_index,
                        receipt,
                        codebook,
                        display,
                        slot_a,
                        slot_b,
                        ordered,
                    ),
                )
            )
    audit = _audit_heldout(probes, spec)
    return probes, audit


def write_banks(
    output_dir: Path,
    train: dict[str, list[TrainingRecord]],
    dev: dict[str, list[TrainingRecord]],
    training_audit: dict[str, Any],
    heldout: list[HeldoutProbe],
    heldout_audit: dict[str, Any],
) -> dict[str, str]:
    root = output_dir / "preflight"
    hashes: dict[str, str] = {}
    for curriculum in CURRICULA:
        hashes[f"train_{curriculum}"] = immutable_write(
            root / f"train_{curriculum}.jsonl",
            _jsonl(train[curriculum]),
            f"CBR {curriculum} train bank",
        )
        hashes[f"dev_{curriculum}"] = immutable_write(
            root / f"dev_{curriculum}.jsonl",
            _jsonl(dev[curriculum]),
            f"CBR {curriculum} dev bank",
        )
    hashes["training_audit"] = immutable_json_write(
        root / "training_data_audit.json", training_audit, "CBR training data audit"
    )
    hashes["heldout_probes"] = immutable_write(
        root / "rab_gen_probes.jsonl", _jsonl(heldout), "CBR held-out probes"
    )
    hashes["heldout_audit"] = immutable_json_write(
        root / "rab_gen_bank_audit.json", heldout_audit, "CBR held-out bank audit"
    )
    return hashes


def selected_slot_label(receipt: str, codebook: str) -> str:
    if receipt not in RECEIPTS or codebook not in CODEBOOKS:
        raise ValueError("unsupported CBR receipt or codebook")
    return receipt if codebook == "canonical" else ("B" if receipt == "A" else "A")


def _compile_training_split(
    curriculum: str, split: str, count: int, offset: int, seed: int
) -> list[TrainingRecord]:
    rows: list[TrainingRecord] = []
    for local_index in range(count):
        index = offset + local_index
        group_id = f"cbr-{curriculum}-{split}-{local_index:03d}"
        candidates = tuple(ACTIONS[(index + value) % len(ACTIONS)] for value in range(4))
        slot_actions = {"A": candidates[0], "B": candidates[1]}
        for task in TASKS:
            conditions = _training_conditions(curriculum, task)
            for row_index, (receipt, codebook, selected, display) in enumerate(conditions):
                if task == "route_only":
                    panel = ("A", "B")
                    desired = row_index % 2
                    ordered = _rotate_to_position(panel, selected, desired)
                else:
                    desired = row_index % 4
                    ordered = _rotate_to_position(candidates, slot_actions[selected], desired)
                prompt, completion = _training_prompt(
                    seed=seed,
                    index=index,
                    task=task,
                    receipt=receipt,
                    codebook=codebook,
                    selected=selected,
                    display=display,
                    slot_actions=slot_actions,
                    ordered=ordered,
                )
                rows.append(
                    TrainingRecord(
                        schema_version="p0d2hcbr-training-row-v1",
                        example_id=f"{group_id}-{task}-{row_index}",
                        group_id=group_id,
                        split=split,
                        curriculum=curriculum,
                        task=task,
                        receipt=receipt,
                        codebook=codebook,
                        selected_slot_label=selected,
                        display_order=display,
                        expected_answer_position=desired,
                        prompt=prompt,
                        completion=completion,
                    )
                )
    return rows


def _training_conditions(
    curriculum: str, task: str
) -> tuple[tuple[str | None, str | None, str, str], ...]:
    if curriculum not in CURRICULA:
        raise ValueError("unsupported CBR curriculum")
    if task == "retrieval_only":
        return tuple(
            (None, None, selected, display)
            for selected, display, _ in product(RECEIPTS, DISPLAY_ORDERS, (0, 1))
        )
    if curriculum == "disentangled":
        return tuple(
            (receipt, codebook, selected_slot_label(receipt, codebook), display)
            for receipt, codebook, display in product(RECEIPTS, CODEBOOKS, DISPLAY_ORDERS)
        )
    return tuple(
        (receipt, "canonical", receipt, display)
        for receipt, display, _ in product(RECEIPTS, DISPLAY_ORDERS, (0, 1))
    )


def _training_prompt(
    *,
    seed: int,
    index: int,
    task: str,
    receipt: str | None,
    codebook: str | None,
    selected: str,
    display: str,
    slot_actions: dict[str, str],
    ordered: tuple[str, ...],
) -> tuple[str, str]:
    opaque = {label: f"record_{_digest(seed, index, label)}" for label in RECEIPTS}
    slot_lines = {
        label: f"Slot {label} contains {opaque[label]} with action {slot_actions[label]}."
        for label in RECEIPTS
    }
    slots = "\n".join(slot_lines[label] for label in display)
    if task == "retrieval_only":
        prefix = f"The selected slot is explicitly Slot {selected}."
        instruction = "Copy its action. Do not infer a routing rule."
        completion = slot_actions[selected]
    else:
        assert receipt is not None and codebook is not None
        mapping = (
            "Receipt A selects Slot A; Receipt B selects Slot B."
            if codebook == "canonical"
            else "Receipt A selects Slot B; Receipt B selects Slot A."
        )
        prefix = f"Routing codebook: {mapping} Current receipt: {receipt}."
        if task == "route_only":
            instruction = "Apply the codebook once and output only the selected slot label."
            completion = selected
        else:
            instruction = "Apply the codebook once, then copy the selected slot action."
            completion = slot_actions[selected]
    return (
        f"{prefix}\n{slots}\n{instruction}\n"
        f"Allowed answers (order is not evidence): {', '.join(ordered)}.\nAnswer:",
        completion,
    )


def _heldout_prompt(
    group_index: int,
    receipt: str,
    codebook: str,
    display: str,
    slot_a: str,
    slot_b: str,
    candidates: tuple[str, ...],
) -> str:
    mapping = (
        "Receipt A selects Slot A; Receipt B selects Slot B."
        if codebook == "canonical"
        else "Receipt A selects Slot B; Receipt B selects Slot A."
    )
    payloads = {
        "A": f"heldout_{_digest('heldout', group_index, 'A')} with action {slot_a}",
        "B": f"heldout_{_digest('heldout', group_index, 'B')} with action {slot_b}",
    }
    slots = " ".join(f"Slot {label} contains {payloads[label]}." for label in display)
    return (
        f"Use this routing codebook exactly once: {mapping} Receipt presented: {receipt}. "
        f"{slots} Return the action stored in the selected slot.\n"
        f"Only output one action token from: {', '.join(candidates)}.\nAction:"
    )


def _task_balance(rows: list[TrainingRecord]) -> bool:
    by_group: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_group[row.group_id][row.task] += 1
    return all(counts == Counter({task: 8 for task in TASKS}) for counts in by_group.values())


def _training_factor_balance(rows: list[TrainingRecord], curriculum: str) -> bool:
    for group_id in {row.group_id for row in rows}:
        group = [row for row in rows if row.group_id == group_id]
        for task in TASKS:
            task_rows = [row for row in group if row.task == task]
            if Counter(row.selected_slot_label for row in task_rows) != {"A": 4, "B": 4}:
                return False
            if Counter(row.display_order for row in task_rows) != {"AB": 4, "BA": 4}:
                return False
            expected = {0: 4, 1: 4} if task == "route_only" else {0: 2, 1: 2, 2: 2, 3: 2}
            if Counter(row.expected_answer_position for row in task_rows) != expected:
                return False
        combined = [row for row in group if row.task == "combined"]
        joint = Counter((row.receipt, row.selected_slot_label) for row in combined)
        expected_joint = (
            {(receipt, slot): 2 for receipt in RECEIPTS for slot in RECEIPTS}
            if curriculum == "disentangled"
            else {("A", "A"): 4, ("B", "B"): 4}
        )
        if joint != expected_joint:
            return False
    return True


def _audit_heldout(probes: list[HeldoutProbe], spec: DataSpec) -> dict[str, Any]:
    by_group: dict[str, list[HeldoutProbe]] = defaultdict(list)
    for probe in probes:
        by_group[probe.unit_id].append(probe)
    expected = set(product(ORIENTATIONS, RECEIPTS, CODEBOOKS, DISPLAY_ORDERS, ROTATIONS))
    joint = Counter(
        (
            row.receipt,
            row.selected_slot_label,
            row.selected_display_position,
            row.expected_candidate_position,
        )
        for row in probes
    )
    checks = {
        "prompt_count": len(probes) == spec.heldout_prompt_count,
        "group_count": len(by_group) == spec.heldout_group_count,
        "unique_probe_ids": len({row.probe_id for row in probes}) == len(probes),
        "unique_prompt_text": len({sha256(row.prompt.encode()).hexdigest() for row in probes})
        == len(probes),
        "complete_group_factorial": all(
            {
                (
                    row.orientation,
                    row.receipt,
                    row.codebook,
                    row.display_order,
                    row.candidate_rotation,
                )
                for row in rows
            }
            == expected
            for rows in by_group.values()
        ),
        "joint_primary_factor_balance": len(joint) == 32
        and set(joint.values()) == {spec.heldout_prompt_count // 32},
    }
    payload = {
        "schema_version": "p0d2hcbr-heldout-bank-audit-v1",
        "bank_version": HELDOUT_BANK_VERSION,
        "checks": checks,
        "factor_counts": {
            "receipt": dict(Counter(row.receipt for row in probes)),
            "selected_slot_label": dict(Counter(row.selected_slot_label for row in probes)),
            "display_position": dict(Counter(row.selected_display_position for row in probes)),
            "candidate_position": dict(Counter(row.expected_candidate_position for row in probes)),
        },
        "records_sha256": json_hash([row.to_dict() for row in probes]),
        "all_checks_passed": all(checks.values()),
    }
    if payload["all_checks_passed"] is not True:
        raise ValueError(f"CBR held-out bank audit failed: {checks}")
    return payload


def _rotate(values: tuple[str, ...], amount: int) -> tuple[str, ...]:
    return values[amount:] + values[:amount]


def _rotate_to_position(values: tuple[str, ...], expected: str, position: int) -> tuple[str, ...]:
    current = values.index(expected)
    return _rotate(values, (current - position) % len(values))


def _digest(*values: object) -> str:
    return sha256("|".join(map(str, values)).encode()).hexdigest()[:12]


def _jsonl(rows: list[Any]) -> bytes:
    return (
        "".join(
            json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        )
    ).encode()
