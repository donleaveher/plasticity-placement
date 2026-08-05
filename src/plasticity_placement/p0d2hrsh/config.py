from __future__ import annotations

from dataclasses import asdict, dataclass

SPEC_SCHEMA_VERSION = "p0d2hrsh-spec-v1"
EXPERIMENT_ID = "P0-D2H-ROUTE-STATE-HANDOFF-v1"
BANK_VERSION = "p0d2hrsh-handoff-bank-v1"
AUTHORIZATION_SCOPE = "single_frozen_cpr_route_state_handoff_audit"
PROMPT_KINDS = ("slot_readout", "direct_action", "receipt_A", "receipt_B")


@dataclass(frozen=True, slots=True)
class AuditSpec:
    schema_version: str = SPEC_SCHEMA_VERSION
    experiment_id: str = EXPERIMENT_ID
    bank_version: str = BANK_VERSION
    lesson_count: int = 24
    route_variants_per_lesson: int = 4
    unit_count: int = 96
    prompt_kind_count: int = 4
    raw_decision_count: int = 768
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20260806
    confidence: float = 0.95
    material_handoff_rescue: float = 0.10
    chain_oracle_noninferiority_margin: float = 0.05
    wrong_slot_specificity_threshold: float = 0.20
    sentinel_score_tolerance: float = 1e-5
    allowed_audit_runs: int = 1
    allows_training: bool = False
    allows_prompt_revision_after_results: bool = False
    allows_mappings_per_adapter_scan: bool = False

    def __post_init__(self) -> None:
        if self != AuditSpec.__new_defaults__():
            raise ValueError("route-state handoff specification is frozen")

    @classmethod
    def __new_defaults__(cls) -> AuditSpec:
        value = object.__new__(cls)
        for name, field in cls.__dataclass_fields__.items():
            object.__setattr__(value, name, field.default)
        return value

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
