from __future__ import annotations

from dataclasses import asdict, dataclass

from plasticity_placement.p0d2hrabd.config import CATEGORIES, CELLS, FACTORS

SPEC_SCHEMA_VERSION = "p0d2hrabdl-spec-v1"
EXPERIMENT_ID = "P0-D2H-RAB-ERROR-LOCALIZATION-v1"
SOURCE_RUN_ID = "p0d2hrabd-a82d35757b"
TRANSITIONS = ("C→C", "C→W", "W→C", "W→W")
EXPECTED_TRANSITION_COUNTS = {"C→C": 90, "C→W": 21, "W→C": 33, "W→W": 48}
EXPECTED_BASE_TAXONOMY = {
    "invalid_or_tied": 1,
    "other_action_intrusion": 0,
    "fully_compliant": 1,
    "fully_inverted": 0,
    "single_action_locked": 30,
    "receipt_invariant": 3,
    "binding_invariant": 0,
    "partial_mixed": 13,
}
EXPECTED_ADAPTER_TAXONOMY = {
    "invalid_or_tied": 0,
    "other_action_intrusion": 0,
    "fully_compliant": 2,
    "fully_inverted": 0,
    "single_action_locked": 10,
    "receipt_invariant": 13,
    "binding_invariant": 0,
    "partial_mixed": 23,
}


@dataclass(frozen=True, slots=True)
class LocalizationSpec:
    schema_version: str = SPEC_SCHEMA_VERSION
    experiment_id: str = EXPERIMENT_ID
    source_run_id: str = SOURCE_RUN_ID
    cell_count: int = 192
    unit_count: int = 48
    pair_count: int = 12
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20260809
    confidence: float = 0.95
    allows_inference: bool = False
    allows_training: bool = False
    allows_historical_reclassification: bool = False
    allows_mappings_per_adapter_scan: bool = False

    def __post_init__(self) -> None:
        if self != LocalizationSpec.__new_defaults__():
            raise ValueError("RAB localization specification is frozen")

    @classmethod
    def __new_defaults__(cls) -> LocalizationSpec:
        value = object.__new__(cls)
        for name, field in cls.__dataclass_fields__.items():
            object.__setattr__(value, name, field.default)
        return value

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = [
    "CATEGORIES",
    "CELLS",
    "EXPECTED_ADAPTER_TAXONOMY",
    "EXPECTED_BASE_TAXONOMY",
    "EXPECTED_TRANSITION_COUNTS",
    "FACTORS",
    "LocalizationSpec",
    "SOURCE_RUN_ID",
    "TRANSITIONS",
]
