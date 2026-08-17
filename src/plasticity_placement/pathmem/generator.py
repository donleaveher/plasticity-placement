from __future__ import annotations

from collections.abc import Iterable

from plasticity_placement.pathmem.schema import (
    ContentDomain,
    EpisodeRole,
    EventBlock,
    EventKind,
    MemoryEvent,
    Probe,
    ProbeCategory,
    SemanticItem,
    TrainingExample,
)

GENERATOR_VERSION = "pathmem-v1-generator-20260813"
GENERATOR_SEED = 20260813
ACTIONS = ("act_n7", "act_p3", "act_v9", "act_k2")
DIRECTED_ACTION_PAIRS = tuple(
    (source, target) for source in ACTIONS for target in ACTIONS if source != target
)
SPLIT_COUNTS = {
    "interface_dev": 12,
    "smoke": 4,
    "kill": 12,
    "confirmatory": 24,
    "hardware_dev": 4,
    "reserve": 12,
}
PRIMARY_BLOCK_IDS = ("A1", "A2", "B1", "B2")


def build_item_bank(seed: int = GENERATOR_SEED) -> tuple[SemanticItem, ...]:
    if seed != GENERATOR_SEED:
        raise ValueError(f"PathMem-v1 generator seed is frozen at {GENERATOR_SEED}")
    items: list[SemanticItem] = []
    items.extend(_items_for_pairs("interface_dev", DIRECTED_ACTION_PAIRS, seed, "alternating"))
    smoke_pairs = (
        (ACTIONS[0], ACTIONS[1]),
        (ACTIONS[1], ACTIONS[2]),
        (ACTIONS[2], ACTIONS[3]),
        (ACTIONS[3], ACTIONS[0]),
    )
    items.extend(_items_for_pairs("smoke", smoke_pairs, seed, "half"))
    items.extend(_items_for_pairs("kill", DIRECTED_ACTION_PAIRS, seed, "half"))
    items.extend(
        _items_for_pairs(
            "confirmatory",
            (*DIRECTED_ACTION_PAIRS, *DIRECTED_ACTION_PAIRS),
            seed,
            "confirmatory",
        )
    )
    hardware_pairs = (
        (ACTIONS[0], ACTIONS[2]),
        (ACTIONS[2], ACTIONS[0]),
        (ACTIONS[1], ACTIONS[3]),
        (ACTIONS[3], ACTIONS[1]),
    )
    items.extend(_items_for_pairs("hardware_dev", hardware_pairs, seed, "half"))
    items.extend(_items_for_pairs("reserve", DIRECTED_ACTION_PAIRS, seed, "alternating"))
    _validate_item_bank(items)
    return tuple(items)


def compile_event_blocks(item: SemanticItem, optimizer_steps: int = 16) -> tuple[EventBlock, ...]:
    blocks: list[EventBlock] = []
    for state_label, action in (("A", item.state_a_action), ("B", item.state_b_action)):
        for occurrence in (1, 2):
            block_id = f"{item.item_id}:{state_label}{occurrence}"
            event = MemoryEvent(
                event_id=f"event:{block_id}",
                kind=EventKind.WRITE if occurrence == 1 else EventKind.REVISE,
                key=item.memory_key,
                value=action,
            )
            examples = tuple(
                TrainingExample(
                    example_id=f"{block_id}:example-{example_index}",
                    prompt=_training_prompt(item, state_label, occurrence, example_index),
                    completion=action,
                )
                for example_index in range(1, 5)
            )
            blocks.append(
                EventBlock(
                    block_id=block_id,
                    state_label=state_label,
                    event=event,
                    examples=examples,
                    optimizer_steps=optimizer_steps,
                )
            )
    return tuple(blocks)


def build_probe_bank(item: SemanticItem) -> tuple[Probe, ...]:
    probes: list[Probe] = []
    for terminal_state in ("A", "B"):
        current = item.state_a_action if terminal_state == "A" else item.state_b_action
        obsolete = item.state_b_action if terminal_state == "A" else item.state_a_action
        specs = (
            (ProbeCategory.QUALIFICATION, 4, EpisodeRole.QUALIFICATION, False),
            (ProbeCategory.EXACT, 2, EpisodeRole.EVAL_NEAR, True),
            (ProbeCategory.PARAPHRASE, 4, EpisodeRole.EVAL_NEAR, True),
            (ProbeCategory.TRANSFER, 4, EpisodeRole.EVAL_FAR, True),
            (ProbeCategory.STALE_CONFLICT, 4, EpisodeRole.EVAL_FAR, True),
            (ProbeCategory.NEAR_NEIGHBOR, 4, EpisodeRole.CONTROL, False),
            (ProbeCategory.UNRELATED, 8, EpisodeRole.CONTROL, False),
        )
        for category, count, role, core_endpoint in specs:
            for index in range(1, count + 1):
                control_index = (index - 1) % len(item.control_actions)
                current_target = core_endpoint or category is ProbeCategory.QUALIFICATION
                expected = current if current_target else item.control_actions[control_index]
                probes.append(
                    Probe(
                        probe_id=(
                            f"{item.item_id}:{terminal_state}:{category.value}:{index:02d}"
                        ),
                        item_id=item.item_id,
                        terminal_state=terminal_state,
                        role=role,
                        category=category,
                        prompt=_probe_prompt(item, terminal_state, category, index, obsolete),
                        action_choices=item.action_choices,
                        expected_action=expected,
                        obsolete_action=obsolete if core_endpoint else None,
                        core_endpoint=core_endpoint,
                    )
                )
    if len(probes) != 60:
        raise AssertionError(f"expected 60 probes for {item.item_id}, got {len(probes)}")
    return tuple(probes)


def canonical_external_note(item: SemanticItem, terminal_state: str) -> str:
    action = item.state_a_action if terminal_state == "A" else item.state_b_action
    return (
        "<pathmem_current_state>\n"
        f"context_id: {item.context_id}\n"
        f"memory_key: {item.memory_key}\n"
        f"current_action: {action}\n"
        "status: active\n"
        "</pathmem_current_state>"
    )


def _items_for_pairs(
    split: str,
    pairs: Iterable[tuple[str, str]],
    seed: int,
    domain_mode: str,
) -> list[SemanticItem]:
    pair_list = tuple(pairs)
    items: list[SemanticItem] = []
    for index, (action_a, action_b) in enumerate(pair_list):
        if domain_mode == "alternating":
            domain = (
                ContentDomain.FACT_MAPPING
                if index % 2 == 0
                else ContentDomain.PROCEDURE_RECOVERY
            )
        elif domain_mode == "half":
            domain = (
                ContentDomain.FACT_MAPPING
                if index < len(pair_list) / 2
                else ContentDomain.PROCEDURE_RECOVERY
            )
        elif domain_mode == "confirmatory":
            domain = (
                ContentDomain.FACT_MAPPING
                if index < len(DIRECTED_ACTION_PAIRS)
                else ContentDomain.PROCEDURE_RECOVERY
            )
        else:
            raise ValueError(f"unsupported domain mode: {domain_mode}")
        remaining = tuple(action for action in ACTIONS if action not in {action_a, action_b})
        item_number = index + 1
        items.append(
            SemanticItem(
                item_id=f"pmv1-{split}-{item_number:02d}",
                split=split,
                content_domain=domain,
                context_id=f"ctx-{seed}-{split}-{item_number:02d}",
                memory_key=f"key-{split}-{item_number:02d}",
                state_a_action=action_a,
                state_b_action=action_b,
                control_actions=(remaining[0], remaining[1]),
                generator_seed=seed,
            )
        )
    return items


def _validate_item_bank(items: list[SemanticItem]) -> None:
    if len({item.item_id for item in items}) != len(items):
        raise AssertionError("PathMem-v1 item IDs are not unique")
    for split, expected in SPLIT_COUNTS.items():
        observed = sum(item.split == split for item in items)
        if observed != expected:
            raise AssertionError(f"split {split} has {observed} items, expected {expected}")
    confirmatory = [item for item in items if item.split == "confirmatory"]
    for domain in ContentDomain:
        pairs = {
            (item.state_a_action, item.state_b_action)
            for item in confirmatory
            if item.content_domain is domain
        }
        if pairs != set(DIRECTED_ACTION_PAIRS):
            raise AssertionError(f"confirmatory directed-pair coverage failed for {domain}")


def _training_prompt(
    item: SemanticItem,
    state_label: str,
    occurrence: int,
    example_index: int,
) -> str:
    if item.content_domain is ContentDomain.FACT_MAPPING:
        body = (
            f"In registry {item.context_id}, the active code for {item.memory_key} "
            f"is revision {state_label}. Paraphrase set {occurrence}, view {example_index}."
        )
    else:
        body = (
            f"In recovery playbook {item.context_id}, procedure {item.memory_key} follows "
            f"the {state_label} rule. Paraphrase set {occurrence}, scenario {example_index}."
        )
    return f"{body}\nReturn exactly one allowed action."


def _probe_prompt(
    item: SemanticItem,
    terminal_state: str,
    category: ProbeCategory,
    index: int,
    obsolete_action: str,
) -> str:
    category_text = {
        ProbeCategory.QUALIFICATION: "State the active action directly",
        ProbeCategory.EXACT: "Use the current registered state",
        ProbeCategory.PARAPHRASE: "Restate the applicable current decision",
        ProbeCategory.TRANSFER: "Apply the current decision to a compatible new case",
        ProbeCategory.STALE_CONFLICT: (
            f"Ignore the obsolete candidate {obsolete_action} and use the current state"
        ),
        ProbeCategory.NEAR_NEIGHBOR: "Answer for a neighboring but distinct key",
        ProbeCategory.UNRELATED: "Answer an unrelated matched control",
    }[category]
    if category in {ProbeCategory.NEAR_NEIGHBOR, ProbeCategory.UNRELATED}:
        context = f"control-{item.split}-{item.item_id}-{category.value}-{index}"
        key = f"control-key-{index}"
    else:
        context = item.context_id
        key = item.memory_key
    return (
        f"{category_text}. Context: {context}. Key: {key}. "
        f"Terminal-state schema: {terminal_state}. Probe variant: {index}. "
        "Return exactly one allowed action."
    )
