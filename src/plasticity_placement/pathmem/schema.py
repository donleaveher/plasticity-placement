from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = "pathmem-schema-v1"
EVENT_SCHEMA_VERSION = "pathmem-event-v1"
HISTORY_FAMILY_SCHEMA_VERSION = "pathmem-history-family-v1"
PROBE_SCHEMA_VERSION = "pathmem-probe-v1"


class EventKind(StrEnum):
    WRITE = "WRITE"
    REVISE = "REVISE"
    DELETE = "DELETE"
    RESTORE = "RESTORE"
    NOOP = "NOOP"
    IDLE = "IDLE"
    CLOCK_NULL = "CLOCK_NULL"


class ContentDomain(StrEnum):
    FACT_MAPPING = "fact_mapping"
    PROCEDURE_RECOVERY = "procedure_recovery"


class ContractFamily(StrEnum):
    TERMINAL_CONSISTENCY = "terminal_consistency"
    LATEST_WRITE_WINS = "latest_write_wins"
    IDEMPOTENCE = "idempotence"
    INDEPENDENT_COMMUTATIVITY = "independent_update_commutativity"
    ISOLATION = "isolation_locality"
    ROLLBACK = "rollback_inverse"


class EpisodeRole(StrEnum):
    ROOT_COLD = "root/cold"
    UPDATE_PATH = "update-path"
    QUALIFICATION = "qualification"
    EVAL_NEAR = "eval-near"
    EVAL_FAR = "eval-far"
    CONTROL = "control"


class ProbeCategory(StrEnum):
    QUALIFICATION = "qualification"
    EXACT = "exact"
    PARAPHRASE = "paraphrase"
    TRANSFER = "transfer"
    STALE_CONFLICT = "stale_conflict"
    NEAR_NEIGHBOR = "near_neighbor"
    UNRELATED = "unrelated"


class AccessMode(StrEnum):
    ON = "memory_on"
    OFF = "memory_off"


class OperatorKind(StrEnum):
    NO_ADDITIONAL_WRITE = "no_additional_write"
    EXTERNAL_LATEST = "external_latest"
    ICL_HISTORY = "icl_history"
    LORA_ADAMW_RESET = "lora_adamw_reset_at_event"
    LORA_ADAMW_CARRY = "lora_adamw_carried"
    LORA_SGD_MEMORYLESS = "lora_sgd_memoryless"
    HYBRID_CURRENT_OVERRIDE = "hybrid_current_override"
    TEXTUAL_CONSOLIDATOR = "textual_consolidator"


class LifecycleLocus(StrEnum):
    COMMIT_ENCODE = "commit_or_encode"
    SUPERSEDE_RESOLVE = "supersede_or_resolve"
    RETRIEVE_SURFACE = "retrieve_or_surface"
    ACTIVATE = "activate"
    APPLY_INFER = "apply_or_infer"
    MAINTAIN_FUTURE = "maintain_or_future_update"


class ObservabilityStatus(StrEnum):
    OBSERVABLE = "observable"
    INTERVENTION_ACCESSIBLE = "intervention-accessible"
    NOT_APPLICABLE = "N/A"


class EvidenceKind(StrEnum):
    INTEGRITY = "integrity"
    UTILITY = "utility"
    ENDPOINT_DEFECT = "endpoint_defect"
    TRACE = "trace"
    COUNTERFACTUAL_RESCUE = "counterfactual_rescue"
    TRANSFER = "transfer"
    DOWNSTREAM_CONSEQUENCE = "downstream_consequence"


class DiagnosticKind(StrEnum):
    ACCESS_OFF = "access_off"
    CANONICALIZATION_DEFECT_REDUCTION = "canonicalization_defect_reduction"
    ARTIFACT_NECESSITY = "artifact_necessity"
    FUTURE_SUFFIX_EQUIVALENCE = "future_suffix_equivalence"
    RESTORE_OR_REFIT = "restore_or_refit"


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    key: str
    value: str

    def __post_init__(self) -> None:
        if not self.key or not self.value:
            raise ValueError("memory entries require non-empty key and value")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    event_id: str
    kind: EventKind
    key: str | None = None
    value: str | None = None
    snapshot: tuple[MemoryEntry, ...] = ()
    scope: str = "item"

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("memory events require event_id")
        if self.kind in {EventKind.WRITE, EventKind.REVISE} and not (
            self.key and self.value
        ):
            raise ValueError(f"{self.kind} requires key and value")
        if self.kind is EventKind.DELETE and not self.key:
            raise ValueError("DELETE requires key")
        if self.kind is EventKind.RESTORE and not self.snapshot:
            raise ValueError("RESTORE requires a non-empty snapshot")
        if self.kind not in {EventKind.WRITE, EventKind.REVISE} and self.value is not None:
            raise ValueError(f"{self.kind} cannot carry value")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingExample:
    example_id: str
    prompt: str
    completion: str

    def __post_init__(self) -> None:
        if not self.example_id or not self.prompt or not self.completion:
            raise ValueError("training examples require ID, prompt, and completion")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EventBlock:
    block_id: str
    state_label: str
    event: MemoryEvent
    examples: tuple[TrainingExample, ...]
    optimizer_steps: int

    def __post_init__(self) -> None:
        if self.state_label not in {"A", "B"}:
            raise ValueError(f"unsupported state label: {self.state_label}")
        if len(self.examples) != 4:
            raise ValueError("PathMem-v1 event blocks require four examples")
        if self.optimizer_steps <= 0:
            raise ValueError("event blocks require positive optimizer steps")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SemanticItem:
    item_id: str
    split: str
    content_domain: ContentDomain
    context_id: str
    memory_key: str
    state_a_action: str
    state_b_action: str
    control_actions: tuple[str, str]
    generator_seed: int

    def __post_init__(self) -> None:
        actions = (self.state_a_action, self.state_b_action, *self.control_actions)
        if len(set(actions)) != 4:
            raise ValueError("semantic items require four distinct actions")
        if not all((self.item_id, self.split, self.context_id, self.memory_key)):
            raise ValueError("semantic item identity fields cannot be empty")

    @property
    def action_choices(self) -> tuple[str, str, str, str]:
        return (self.state_a_action, self.state_b_action, *self.control_actions)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Probe:
    probe_id: str
    item_id: str
    terminal_state: str
    role: EpisodeRole
    category: ProbeCategory
    prompt: str
    action_choices: tuple[str, str, str, str]
    expected_action: str
    obsolete_action: str | None
    core_endpoint: bool

    def __post_init__(self) -> None:
        if self.terminal_state not in {"A", "B"}:
            raise ValueError("probe terminal_state must be A or B")
        if len(set(self.action_choices)) != 4:
            raise ValueError("probes require four unique action choices")
        if self.expected_action not in self.action_choices:
            raise ValueError("probe expected action must be a candidate")
        if self.obsolete_action is not None and self.obsolete_action not in self.action_choices:
            raise ValueError("probe obsolete action must be a candidate")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HistoryPath:
    path_id: str
    block_ids: tuple[str, ...]
    expected_endpoint_hash: str

    def __post_init__(self) -> None:
        if not self.path_id or not self.block_ids or not self.expected_endpoint_hash:
            raise ValueError("history paths require identity, blocks, and endpoint hash")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ComparatorControl:
    comparator_id: str
    path_ids: tuple[str, str]
    root_state_id: str
    final_block_id: str
    event_multiset_sha256: str
    example_multiset_sha256: str
    loss_bearing_text_sha256: str
    ordered_minibatch_sha256_by_block: tuple[tuple[str, str], ...]
    optimizer_steps_total: int
    training_example_count: int

    def __post_init__(self) -> None:
        if len(set(self.path_ids)) != 2:
            raise ValueError("comparator controls require two distinct paths")
        hashes = (
            self.event_multiset_sha256,
            self.example_multiset_sha256,
            self.loss_bearing_text_sha256,
        )
        if any(len(value) != 64 for value in hashes):
            raise ValueError("comparator controls require SHA-256 identities")
        if self.optimizer_steps_total <= 0 or self.training_example_count <= 0:
            raise ValueError("comparator exposure counts must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HistoryFamily:
    history_family_id: str
    item_id: str
    content_domain: ContentDomain
    contract_family: ContractFamily
    initial_fixture: tuple[MemoryEntry, ...]
    logical_reducer: str
    histories: tuple[HistoryPath, HistoryPath]
    endpoint_hash: str
    comparator_control: ComparatorControl
    episode_roles: tuple[EpisodeRole, ...]
    observability_profile_id: str
    trace_expectations: tuple[str, ...]
    false_positive_pattern: str
    must_demonstrate: str

    def __post_init__(self) -> None:
        if len(self.histories) != 2:
            raise ValueError("a primary history family requires exactly two paths")
        if {path.path_id for path in self.histories} != set(
            self.comparator_control.path_ids
        ):
            raise ValueError("history paths do not match comparator control")
        if any(path.expected_endpoint_hash != self.endpoint_hash for path in self.histories):
            raise ValueError("history endpoint hash mismatch")
        if set(self.episode_roles) != set(EpisodeRole):
            raise ValueError("history family must freeze every episode role")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DiagnosticSpec:
    diagnostic_id: str
    kind: DiagnosticKind
    operator: OperatorKind
    target_locus: LifecycleLocus
    held_fixed: tuple[str, ...]
    authorization_phase: str
    additive: bool = False

    def __post_init__(self) -> None:
        if self.additive:
            raise ValueError("PathMem diagnostic estimands are never additive")
        if self.authorization_phase not in {"P1b", "P2", "P3"}:
            raise ValueError("diagnostics require a conditional later-phase authorization")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LifecycleTraceRecord:
    operator: OperatorKind
    contract: ContractFamily
    locus: LifecycleLocus
    observability: ObservabilityStatus
    trace_value: str
    event_position: int | None
    trace_sha256: str

    def __post_init__(self) -> None:
        if self.observability is ObservabilityStatus.NOT_APPLICABLE:
            if self.trace_value != "N/A" or self.trace_sha256 != "N/A":
                raise ValueError("unavailable lifecycle traces must remain explicit N/A")
        elif len(self.trace_sha256) != 64:
            raise ValueError("observable lifecycle traces require a SHA-256 identity")
        if self.event_position is not None and self.event_position <= 0:
            raise ValueError("event positions are one-based when present")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TokenExposureAudit:
    history_family_id: str
    path_id: str
    tokenizer_sha256: str
    loss_bearing_tokens: int
    total_forward_tokens: int
    optimizer_steps: int
    prompt_truncated: bool
    candidate_token_lengths: tuple[int, int, int, int]
    ordered_minibatch_sha256_by_block: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if len(self.tokenizer_sha256) != 64:
            raise ValueError("token exposure audit requires tokenizer SHA-256")
        if any(
            value <= 0
            for value in (
                self.loss_bearing_tokens,
                self.total_forward_tokens,
                self.optimizer_steps,
            )
        ):
            raise ValueError("token exposure counts and optimizer steps must be positive")
        if len(set(self.candidate_token_lengths)) != 1:
            raise ValueError("four action candidates must have equal token length")
        if self.prompt_truncated:
            raise ValueError("truncated prompts are not admissible")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
