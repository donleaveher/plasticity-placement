from __future__ import annotations

from dataclasses import asdict, dataclass

SPEC_SCHEMA_VERSION = "p0d2hrabx-spec-v1"
EXPERIMENT_ID = "P0-D2H-RAB-LABEL-DISENTANGLEMENT-v1"
BANK_VERSION = "p0d2hrabx-label-disentanglement-bank-v1"
AUTHORIZATION_SCOPE = "single_frozen_rab_label_disentanglement_audit"
EXPECTED_RABC_RUN_ID = "p0d2hrabc-c4f96876bb"

ORIENTATIONS = ("canonical", "swapped")
RECEIPTS = ("A", "B")
CODEBOOKS = ("canonical", "crossed")
SLOT_LABELS = ("A", "B")
DISPLAY_ORDERS = ("AB", "BA")
CANDIDATE_ROTATIONS = (0, 1, 2, 3)


@dataclass(frozen=True, slots=True)
class AuditSpec:
    schema_version: str = SPEC_SCHEMA_VERSION
    experiment_id: str = EXPERIMENT_ID
    bank_version: str = BANK_VERSION
    expected_rabc_run_id: str = EXPECTED_RABC_RUN_ID
    pair_count: int = 12
    route_variants_per_pair: int = 4
    unit_count: int = 48
    source_prompt_count: int = 192
    conditions_per_unit: int = 64
    prompt_count: int = 3_072
    raw_decision_count: int = 6_144
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20260809
    confidence: float = 0.95
    sentinel_score_tolerance: float = 1e-5
    allowed_audit_runs: int = 1
    allows_training: bool = False
    allows_prompt_revision_after_results: bool = False
    allows_historical_reclassification: bool = False
    allows_mappings_per_adapter_scan: bool = False

    def __post_init__(self) -> None:
        if self != AuditSpec.__new_defaults__():
            raise ValueError("RAB label-disentanglement specification is frozen")

    @classmethod
    def __new_defaults__(cls) -> AuditSpec:
        value = object.__new__(cls)
        for name, field in cls.__dataclass_fields__.items():
            object.__setattr__(value, name, field.default)
        return value

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
