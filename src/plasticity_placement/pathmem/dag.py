from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from plasticity_placement.pathmem.identity import (
    NodeIdentityInput,
    RuntimeIdentity,
    event_block_hash,
    ordered_minibatch_hash,
)
from plasticity_placement.pathmem.io import json_hash
from plasticity_placement.pathmem.reducer import REDUCER_VERSION
from plasticity_placement.pathmem.schema import (
    EVENT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    EventBlock,
    HistoryFamily,
)

DAG_SCHEMA_VERSION = "pathmem-core-dag-v1"


@dataclass(frozen=True, slots=True)
class DAGNodePlan:
    logical_name: str
    plan_id: str
    parent_logical_name: str
    block_id: str
    event_block_sha256: str
    ordered_minibatch_sha256: str
    path_memberships: tuple[str, ...]
    current_only_anchor: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CoreDAGPlan:
    item_id: str
    history_family_bundle_sha256: str
    operator: str
    training_seed: int
    root_state_sha256: str
    nodes: tuple[DAGNodePlan, ...]

    def __post_init__(self) -> None:
        expected = {"A1", "B1", "AB", "BA", "ABA", "BAA", "BAB", "ABB", "A2", "B2"}
        observed = {node.logical_name for node in self.nodes}
        if observed != expected or len(self.nodes) != 10:
            raise ValueError(f"core DAG nodes changed: {observed}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_core_dag_plan(
    families: tuple[HistoryFamily, HistoryFamily],
    blocks: tuple[EventBlock, ...],
    runtime: RuntimeIdentity,
) -> CoreDAGPlan:
    if {family.item_id for family in families} != {runtime.item_id}:
        raise ValueError("DAG runtime and history families refer to different items")
    family_bundle_sha256 = json_hash(
        [family.to_dict() for family in sorted(families, key=lambda value: value.history_family_id)]
    )
    block_by_suffix = {block.block_id.rsplit(":", 1)[-1]: block for block in blocks}
    if set(block_by_suffix) != {"A1", "A2", "B1", "B2"}:
        raise ValueError("core DAG requires exactly A1/A2/B1/B2 blocks")
    specifications = (
        ("A1", "ROOT", "A1", ("ABA", "ABB"), False),
        ("B1", "ROOT", "B1", ("BAA", "BAB"), False),
        ("AB", "A1", "B1", ("ABA", "ABB"), False),
        ("BA", "B1", "A1", ("BAA", "BAB"), False),
        ("ABA", "AB", "A2", ("ABA",), False),
        ("BAA", "BA", "A2", ("BAA",), False),
        ("BAB", "BA", "B2", ("BAB",), False),
        ("ABB", "AB", "B2", ("ABB",), False),
        ("A2", "ROOT", "A2", ("P-latest-A",), True),
        ("B2", "ROOT", "B2", ("P-latest-B",), True),
    )
    nodes: list[DAGNodePlan] = []
    for logical_name, parent, block_suffix, memberships, anchor in specifications:
        block = block_by_suffix[block_suffix]
        block_sha256 = event_block_hash(block)
        minibatch_sha256 = ordered_minibatch_hash(block, runtime.training_seed)
        plan_id = json_hash(
            {
                "schema_version": DAG_SCHEMA_VERSION,
                "history_family_bundle_sha256": family_bundle_sha256,
                "operator": runtime.operator.value,
                "training_seed": runtime.training_seed,
                "logical_name": logical_name,
                "parent_logical_name": parent,
                "event_block_sha256": block_sha256,
                "ordered_minibatch_sha256": minibatch_sha256,
            }
        )
        nodes.append(
            DAGNodePlan(
                logical_name=logical_name,
                plan_id=plan_id,
                parent_logical_name=parent,
                block_id=block.block_id,
                event_block_sha256=block_sha256,
                ordered_minibatch_sha256=minibatch_sha256,
                path_memberships=memberships,
                current_only_anchor=anchor,
            )
        )
    return CoreDAGPlan(
        item_id=runtime.item_id,
        history_family_bundle_sha256=family_bundle_sha256,
        operator=runtime.operator.value,
        training_seed=runtime.training_seed,
        root_state_sha256=runtime.root_state_sha256,
        nodes=tuple(nodes),
    )


def resolve_node_identity(
    *,
    plan: DAGNodePlan,
    dag: CoreDAGPlan,
    runtime: RuntimeIdentity,
    parent_state_sha256: str,
) -> NodeIdentityInput:
    if dag.item_id != runtime.item_id or dag.training_seed != runtime.training_seed:
        raise ValueError("DAG and runtime identities do not match")
    occurrence_id = plan.block_id.rsplit(":", 1)[-1]
    return NodeIdentityInput(
        experiment_schema=SCHEMA_VERSION,
        history_family_sha256=dag.history_family_bundle_sha256,
        logical_reducer=REDUCER_VERSION,
        event_schema=EVENT_SCHEMA_VERSION,
        base_model_revision=runtime.base_model_revision,
        root_adapter_sha256=runtime.root_adapter_sha256,
        parent_state_sha256=parent_state_sha256,
        event_block_sha256=plan.event_block_sha256,
        ordered_minibatch_sha256=plan.ordered_minibatch_sha256,
        occurrence_id=occurrence_id,
        trainer_impl_sha256=runtime.trainer_impl_sha256,
        trainer_config_sha256=runtime.trainer_config_sha256,
        tokenizer_sha256=runtime.tokenizer_sha256,
        library_lock_sha256=runtime.library_lock_sha256,
        code_and_patch_sha256=runtime.code_and_patch_sha256,
        resolved_precision=runtime.resolved_precision,
    )
