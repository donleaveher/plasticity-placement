from __future__ import annotations

from dataclasses import asdict, dataclass

SPEC_SCHEMA_VERSION = "p0d2hrabd-spec-v1"
EXPERIMENT_ID = "P0-D2H-RAB-ERROR-TOPOLOGY-v1"
TAXONOMY_VERSION = "p0d2hrabd-unit-taxonomy-v1"
CELLS = ("canonical_A", "canonical_B", "swapped_A", "swapped_B")
FACTORS = (
    "pair_id",
    "lesson_type",
    "route_variant",
    "orientation",
    "receipt",
    "expected_action",
    "candidate_position",
    "expected_token_count",
)
CATEGORIES = (
    "invalid_or_tied",
    "other_action_intrusion",
    "fully_compliant",
    "fully_inverted",
    "single_action_locked",
    "receipt_invariant",
    "binding_invariant",
    "partial_mixed",
)


@dataclass(frozen=True, slots=True)
class DiagnosticSpec:
    schema_version: str = SPEC_SCHEMA_VERSION
    experiment_id: str = EXPERIMENT_ID
    taxonomy_version: str = TAXONOMY_VERSION
    unit_count: int = 48
    paired_record_count: int = 192
    raw_row_count_per_state: int = 192
    cell_count_per_unit: int = 4
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20260808
    confidence: float = 0.95
    allowed_diagnostic_runs: int = 1
    allows_inference: bool = False
    allows_training: bool = False
    allows_historical_rab_reclassification: bool = False
    allows_mappings_per_adapter_scan: bool = False

    def __post_init__(self) -> None:
        if self != DiagnosticSpec.__new_defaults__():
            raise ValueError("RAB error-topology specification is frozen")

    @classmethod
    def __new_defaults__(cls) -> DiagnosticSpec:
        value = object.__new__(cls)
        for name, field in cls.__dataclass_fields__.items():
            object.__setattr__(value, name, field.default)
        return value

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
