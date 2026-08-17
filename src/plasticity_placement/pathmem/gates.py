from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

GATE_SCHEMA_VERSION = "pathmem-gates-v1"
AUTHORIZATION_ORDER = ("G0", "G1", "P0", "P1", "P1b", "P2", "P3")
G1_THRESHOLDS = {
    "answer_copy_top1": {"minimum": 0.98},
    "parser_validity": {"minimum": 1.0},
    "external_current_top1": {"minimum": 0.95},
    "external_obsolete_intrusion": {"maximum": 0.05},
    "parametric_current_top1": {"minimum": 0.80},
    "parametric_per_action_min_top1": {"minimum": 0.70},
    "parametric_unrelated_regression_pp": {"maximum": 5.0},
    "adapter_disable_max_candidate_delta": {"maximum": 1e-5},
}


@dataclass(frozen=True, slots=True)
class GateResult:
    gate: str
    passed: bool
    checks: tuple[tuple[str, bool], ...]
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_g0_gate(checks: dict[str, bool]) -> GateResult:
    required = (
        "protocol_sources_hashed",
        "schemas_frozen",
        "split_bank_complete",
        "endpoint_equivalence_verified",
        "exposure_controls_verified",
        "observability_matrix_complete",
        "claim_falsifiers_complete",
        "manifest_identity_verified",
        "training_not_started",
    )
    missing = [name for name in required if name not in checks]
    if missing:
        raise ValueError(f"G0 checks missing: {missing}")
    frozen_checks = tuple((name, bool(checks[name])) for name in required)
    blockers = tuple(name for name, passed in frozen_checks if not passed)
    return GateResult(gate="G0", passed=not blockers, checks=frozen_checks, blockers=blockers)


def evaluate_g1_gate(metrics: dict[str, float]) -> GateResult:
    required_metrics = set(G1_THRESHOLDS)
    if set(metrics) != required_metrics:
        raise ValueError(
            f"G1 metric set changed: missing={sorted(required_metrics - set(metrics))} "
            f"extra={sorted(set(metrics) - required_metrics)}"
        )
    checks = (
        ("answer_copy_top1", metrics["answer_copy_top1"] >= 0.98),
        ("parser_validity", metrics["parser_validity"] >= 1.0),
        ("external_current_top1", metrics["external_current_top1"] >= 0.95),
        (
            "external_obsolete_intrusion",
            metrics["external_obsolete_intrusion"] <= 0.05,
        ),
        ("parametric_current_top1", metrics["parametric_current_top1"] >= 0.80),
        (
            "parametric_per_action_top1",
            metrics["parametric_per_action_min_top1"] >= 0.70,
        ),
        (
            "parametric_unrelated_regression",
            metrics["parametric_unrelated_regression_pp"] <= 5.0,
        ),
        (
            "adapter_disable_candidate_delta",
            metrics["adapter_disable_max_candidate_delta"] <= 1e-5,
        ),
    )
    blockers = tuple(name for name, passed in checks if not passed)
    return GateResult(gate="G1", passed=not blockers, checks=checks, blockers=blockers)


def require_authorization(requested_phase: str, passed_gates: set[str]) -> None:
    if requested_phase not in AUTHORIZATION_ORDER:
        raise ValueError(f"unknown PathMem phase: {requested_phase}")
    requested_index = AUTHORIZATION_ORDER.index(requested_phase)
    required = set(AUTHORIZATION_ORDER[:requested_index])
    if requested_phase == "P3":
        required.add("P2_deployment_gap")
    missing = sorted(required - passed_gates)
    if missing:
        raise PermissionError(f"{requested_phase} is blocked by missing gates: {missing}")
