from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

G0V2_CONTRACT_SCHEMA_VERSION = "pathmem-g0-v2-r-opcd-contract-v1"
G1C_GATE_SCHEMA_VERSION = "pathmem-g1c-gate-v1"
OPERATOR_ID = "routed_on_policy_context_distillation_v1"
QUALIFICATION_SPLIT = "interface_dev"
RETENTION_HORIZON = 12
EXECUTION_ORDER_SEED = 20260817


class Comparator(StrEnum):
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    EXACT = "exact"


@dataclass(frozen=True, slots=True)
class GateThreshold:
    metric: str
    family: str
    comparator: Comparator
    value: float
    unit: str

    def __post_init__(self) -> None:
        if not self.metric or not self.family or not self.unit:
            raise ValueError("G1-C thresholds require metric, family, and unit")
        if not math.isfinite(self.value) or self.value < 0:
            raise ValueError("G1-C threshold values must be finite and non-negative")

    def passes(self, observed: float) -> bool:
        if not math.isfinite(observed):
            return False
        if self.comparator is Comparator.MINIMUM:
            return observed >= self.value
        if self.comparator is Comparator.MAXIMUM:
            return observed <= self.value
        return observed == self.value

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["comparator"] = self.comparator.value
        return payload


G1C_THRESHOLDS = (
    GateThreshold("teacher_current_top1", "teacher", Comparator.MINIMUM, 0.95, "fraction"),
    GateThreshold(
        "teacher_obsolete_intrusion", "teacher", Comparator.MAXIMUM, 0.05, "fraction"
    ),
    GateThreshold("teacher_parser_validity", "teacher", Comparator.EXACT, 1.0, "fraction"),
    GateThreshold(
        "immediate_current_top1", "immediate_write", Comparator.MINIMUM, 0.80, "fraction"
    ),
    GateThreshold(
        "immediate_per_action_min_top1",
        "immediate_write",
        Comparator.MINIMUM,
        0.70,
        "fraction",
    ),
    GateThreshold(
        "immediate_zero_action_count", "immediate_write", Comparator.EXACT, 0.0, "count"
    ),
    GateThreshold(
        "retention_h12_current_top1", "retention", Comparator.MINIMUM, 0.75, "fraction"
    ),
    GateThreshold(
        "retention_h12_aggregate_drop_pp", "retention", Comparator.MAXIMUM, 5.0, "pp"
    ),
    GateThreshold(
        "retention_h12_per_action_max_drop_pp",
        "retention",
        Comparator.MAXIMUM,
        10.0,
        "pp",
    ),
    GateThreshold(
        "unrelated_regression_pp", "locality", Comparator.MAXIMUM, 5.0, "pp"
    ),
    GateThreshold("router_miss_rate", "routing", Comparator.MAXIMUM, 0.05, "fraction"),
    GateThreshold(
        "router_false_activation_rate", "routing", Comparator.MAXIMUM, 0.02, "fraction"
    ),
    GateThreshold(
        "wrong_swap_sensitivity_reported", "routing", Comparator.EXACT, 1.0, "boolean"
    ),
    GateThreshold(
        "access_off_max_candidate_delta", "rollback", Comparator.MAXIMUM, 1e-5, "score"
    ),
    GateThreshold(
        "restore_max_candidate_delta", "rollback", Comparator.MAXIMUM, 1e-5, "score"
    ),
    GateThreshold("integrity_passed", "integrity", Comparator.EXACT, 1.0, "boolean"),
)


@dataclass(frozen=True, slots=True)
class GateResult:
    gate: str
    passed: bool
    checks: tuple[tuple[str, bool], ...]
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_g1c_gate(metrics: dict[str, float]) -> GateResult:
    expected = {threshold.metric for threshold in G1C_THRESHOLDS}
    observed = set(metrics)
    if observed != expected:
        raise ValueError(
            "G1-C metric set changed: "
            f"missing={sorted(expected - observed)} extra={sorted(observed - expected)}"
        )
    checks = tuple(
        (threshold.metric, threshold.passes(float(metrics[threshold.metric])))
        for threshold in G1C_THRESHOLDS
    )
    blockers = tuple(metric for metric, passed in checks if not passed)
    return GateResult(gate="G1-C", passed=not blockers, checks=checks, blockers=blockers)


def build_g0v2_contract(
    *,
    parent_manifest_id: str,
    parent_manifest_sha256: str,
) -> dict[str, Any]:
    _require_sha256(parent_manifest_id, "parent G0-v1 manifest ID")
    _require_sha256(parent_manifest_sha256, "parent G0-v1 manifest file")
    return {
        "schema_version": G0V2_CONTRACT_SCHEMA_VERSION,
        "protocol_version": "PathMem-G0-v2",
        "amends_without_mutating": "PathMem-G0-v1",
        "parent_g0_v1": {
            "manifest_id": parent_manifest_id,
            "manifest_sha256": parent_manifest_sha256,
        },
        "operator": {
            "operator_id": OPERATOR_ID,
            "display_name": "Routed On-Policy Context Distillation (R-OPCD)",
            "memory_lifecycle": [
                "external_commit",
                "privileged_context_teacher",
                "student_on_policy_rollout",
                "isolated_side_memory_update",
                "deterministic_route_activation",
                "retention_and_rollback_audit",
            ],
            "teacher_visibility": "canonical latest-valid external note",
            "student_visibility": "no privileged external note",
            "parameter_target": "isolated modular side-memory namespace",
            "g1c_router": "deterministic exact key-to-module mapping",
            "learned_router": False,
            "rlhf_or_rlvr_controller": "deferred_until_a_later_explicit_gate",
        },
        "qualification": {
            "gate": "G1-C",
            "split": QUALIFICATION_SPLIT,
            "retention_horizon_subsequent_consolidations": RETENTION_HORIZON,
            "thresholds": [threshold.to_dict() for threshold in G1C_THRESHOLDS],
            "required_distributional_reports": [
                "per_item",
                "per_action",
                "immediate_vs_retained",
                "router_wrong_swap",
            ],
        },
        "integrity_locks": [
            "parent_g0_v1_manifest",
            "model_revision",
            "tokenizer_hash",
            "precision",
            "teacher_and_student_prompts",
            "consolidation_data",
            "side_memory_parent_and_child_hashes",
            "router_mapping_hash",
            "execution_order_seed",
        ],
        "implementation_stage": {
            "implemented_now": "cpu_contract_and_plan_compilation",
            "must_be_frozen_before_cuda": [
                "rollout_sampling_recipe",
                "distillation_loss",
                "optimizer_and_scheduler",
                "side_memory_parameterization",
                "checkpoint_persistence",
            ],
        },
        "authorization": {
            "safe_default": "planning_only",
            "training_authorized": False,
            "gpu_inference_authorized": False,
            "g1c_execution_authorized": False,
            "p0_authorized": False,
            "path_contrast_authorized": False,
            "kill_or_reserve_access_authorized": False,
            "rl_controller_authorized": False,
        },
        "claim_boundaries": [
            "G1-C is interface qualification only",
            "no ABA/BAA or BAB/ABB contrast is computed",
            "passing G1-C does not establish endpoint consistency or path dependence",
            "passing G1-C only permits a separately reviewed R-OPCD P0 implementation",
        ],
    }


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 value")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
