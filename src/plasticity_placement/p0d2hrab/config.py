from __future__ import annotations

from dataclasses import asdict, dataclass

SPEC_SCHEMA_VERSION = "p0d2hrab-spec-v1"
EXPERIMENT_ID = "P0-D2H-RECEIPT-ACTION-BINDING-v1"
BANK_VERSION = "p0d2hrab-two-valid-slot-bank-v1"
AUTHORIZATION_SCOPE = "single_frozen_receipt_action_binding_audit"
ORIENTATIONS = ("canonical", "swapped")
RECEIPTS = ("A", "B")
CELLS = tuple(f"{orientation}_{receipt}" for orientation in ORIENTATIONS for receipt in RECEIPTS)


@dataclass(frozen=True, slots=True)
class AuditSpec:
    schema_version: str = SPEC_SCHEMA_VERSION
    experiment_id: str = EXPERIMENT_ID
    bank_version: str = BANK_VERSION
    pair_count: int = 12
    route_variants_per_pair: int = 4
    unit_count: int = 48
    orientation_count: int = 2
    receipt_count: int = 2
    prompt_count: int = 192
    raw_decision_count: int = 384
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20260807
    confidence: float = 0.95
    binding_accuracy_threshold: float = 0.75
    binding_accuracy_ci_floor: float = 0.625
    specificity_threshold: float = 0.20
    adapter_noninferiority_margin: float = 0.05
    sentinel_score_tolerance: float = 1e-5
    allowed_audit_runs: int = 1
    allows_training: bool = False
    allows_prompt_revision_after_results: bool = False
    allows_historical_rsh_reclassification: bool = False
    allows_mappings_per_adapter_scan: bool = False

    def __post_init__(self) -> None:
        if self != AuditSpec.__new_defaults__():
            raise ValueError("receipt/action binding specification is frozen")

    @classmethod
    def __new_defaults__(cls) -> AuditSpec:
        value = object.__new__(cls)
        for name, field in cls.__dataclass_fields__.items():
            object.__setattr__(value, name, field.default)
        return value

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
