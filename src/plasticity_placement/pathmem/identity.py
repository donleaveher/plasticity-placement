from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from plasticity_placement.pathmem.io import json_hash
from plasticity_placement.pathmem.schema import EVENT_SCHEMA_VERSION, EventBlock, OperatorKind

IDENTITY_VERSION = "pathmem-identity-v1"


def require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")


def root_adapter_seed(
    experiment_seed: int,
    item_id: str,
    training_seed: int,
    operator: OperatorKind,
) -> int:
    digest = json_hash(
        {
            "experiment_seed": experiment_seed,
            "item_id": item_id,
            "training_seed": training_seed,
            "operator": operator.value,
        }
    )
    return int(digest[:16], 16)


def block_seed(training_seed: int, event_block_sha256: str, occurrence_id: str) -> int:
    require_sha256(event_block_sha256, "event_block_sha256")
    return int(
        json_hash(
            {
                "training_seed": training_seed,
                "event_block_sha256": event_block_sha256,
                "occurrence_id": occurrence_id,
            }
        )[:16],
        16,
    )


def event_block_hash(block: EventBlock) -> str:
    return json_hash(block.to_dict())


def ordered_minibatch_hash(block: EventBlock, training_seed: int) -> str:
    block_sha256 = event_block_hash(block)
    occurrence_id = block.block_id.rsplit(":", 1)[-1]
    return json_hash(
        {
            "block_seed": block_seed(training_seed, block_sha256, occurrence_id),
            "ordered_examples": [example.to_dict() for example in block.examples],
        }
    )


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    experiment_seed: int
    item_id: str
    training_seed: int
    operator: OperatorKind
    base_model_name: str
    base_model_revision: str
    root_adapter_sha256: str
    trainer_impl_sha256: str
    trainer_config_sha256: str
    tokenizer_sha256: str
    library_lock_sha256: str
    code_and_patch_sha256: str
    resolved_precision: str

    def __post_init__(self) -> None:
        for field in (
            "root_adapter_sha256",
            "trainer_impl_sha256",
            "trainer_config_sha256",
            "tokenizer_sha256",
            "library_lock_sha256",
            "code_and_patch_sha256",
        ):
            require_sha256(str(getattr(self, field)), field)
        if not all(
            (
                self.item_id,
                self.base_model_name,
                self.base_model_revision,
                self.resolved_precision,
            )
        ):
            raise ValueError("runtime identity fields cannot be empty")
        if self.resolved_precision == "bfloat16_when_supported":
            raise ValueError("resolved precision cannot be conditional")

    @property
    def root_state_sha256(self) -> str:
        return json_hash(
            {
                "base_model_revision": self.base_model_revision,
                "root_adapter_sha256": self.root_adapter_sha256,
                "root_adapter_seed": root_adapter_seed(
                    self.experiment_seed,
                    self.item_id,
                    self.training_seed,
                    self.operator,
                ),
                "operator": self.operator.value,
                "resolved_precision": self.resolved_precision,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NodeIdentityInput:
    experiment_schema: str
    history_family_sha256: str
    logical_reducer: str
    event_schema: str
    base_model_revision: str
    root_adapter_sha256: str
    parent_state_sha256: str
    event_block_sha256: str
    ordered_minibatch_sha256: str
    occurrence_id: str
    trainer_impl_sha256: str
    trainer_config_sha256: str
    tokenizer_sha256: str
    library_lock_sha256: str
    code_and_patch_sha256: str
    resolved_precision: str

    def __post_init__(self) -> None:
        for field in (
            "history_family_sha256",
            "root_adapter_sha256",
            "parent_state_sha256",
            "event_block_sha256",
            "ordered_minibatch_sha256",
            "trainer_impl_sha256",
            "trainer_config_sha256",
            "tokenizer_sha256",
            "library_lock_sha256",
            "code_and_patch_sha256",
        ):
            require_sha256(str(getattr(self, field)), field)
        if self.event_schema != EVENT_SCHEMA_VERSION:
            raise ValueError(f"event schema must be frozen at {EVENT_SCHEMA_VERSION}")

    @property
    def node_id(self) -> str:
        return json_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "node_id": self.node_id}


@dataclass(frozen=True, slots=True)
class ResultIdentityInput:
    node_id: str
    probe_bank_sha256: str
    renderer_and_prompt_sha256: str
    external_state_sha256: str
    access_mode: str
    lifecycle_trace_sha256: str
    scorer_sha256: str
    evaluation_precision: str

    def __post_init__(self) -> None:
        for field in (
            "node_id",
            "probe_bank_sha256",
            "renderer_and_prompt_sha256",
            "external_state_sha256",
            "lifecycle_trace_sha256",
            "scorer_sha256",
        ):
            require_sha256(str(getattr(self, field)), field)
        if self.access_mode not in {"memory_on", "memory_off"}:
            raise ValueError("result access mode is not frozen")
        if not self.evaluation_precision:
            raise ValueError("result identity requires resolved evaluation precision")

    @property
    def result_id(self) -> str:
        return json_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "result_id": self.result_id}


def verify_parent_lineage(
    *,
    expected_parent_sha256: str,
    observed_parent_sha256: str,
    expected_root_sha256: str,
    observed_root_sha256: str,
) -> None:
    for value, label in (
        (expected_parent_sha256, "expected_parent_sha256"),
        (observed_parent_sha256, "observed_parent_sha256"),
        (expected_root_sha256, "expected_root_sha256"),
        (observed_root_sha256, "observed_root_sha256"),
    ):
        require_sha256(value, label)
    if expected_parent_sha256 != observed_parent_sha256:
        raise ValueError("resumed adapter parent hash mismatch")
    if expected_root_sha256 != observed_root_sha256:
        raise ValueError("resumed adapter root hash mismatch")
