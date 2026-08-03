from __future__ import annotations

from dataclasses import asdict, dataclass

SPEC_SCHEMA_VERSION = "p0d2hrtb-spec-v1"
EXPERIMENT_ID = "P0-D2H-ROUTE-TRANSFER-BRIDGE-v1"
BANK_VERSION = "p0d2hrtb-factorial-bank-v1"
AUTHORIZATION_SCOPE = "single_frozen_cpr_route_transfer_bridge_audit"
GRAMMARS = ("map", "default")
LEXICONS = ("snake", "natural")
PAYLOADS = ("opaque", "external")
ACTION_CELLS = ("external_action", "forced_slot_action")


@dataclass(frozen=True, slots=True)
class AuditSpec:
    schema_version: str = SPEC_SCHEMA_VERSION
    experiment_id: str = EXPERIMENT_ID
    bank_version: str = BANK_VERSION
    lesson_count: int = 24
    route_variants_per_lesson: int = 4
    factorial_cell_count: int = 8
    rows_per_cell: int = 96
    action_cell_count: int = 2
    decision_count: int = 960
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20260803
    primary_contrast_count: int = 3
    primary_two_sided_confidence: float = 0.9833333333333333
    material_rescue_threshold: float = 0.10
    sentinel_score_tolerance: float = 1e-5
    allowed_audit_runs: int = 1
    allows_training: bool = False
    allows_prompt_revision_after_results: bool = False
    allows_mappings_per_adapter_scan: bool = False

    def __post_init__(self) -> None:
        if self != AuditSpec.__new_defaults__():
            raise ValueError("route-transfer bridge specification is frozen")

    @classmethod
    def __new_defaults__(cls) -> AuditSpec:
        value = object.__new__(cls)
        for name, field in cls.__dataclass_fields__.items():
            object.__setattr__(value, name, field.default)
        return value

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
