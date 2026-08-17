from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from plasticity_placement.pathmem.schema import (
    ContractFamily,
    LifecycleLocus,
    ObservabilityStatus,
    OperatorKind,
)

OBSERVABILITY_VERSION = "pathmem-observability-v1"
CLAIM_EVIDENCE_VERSION = "pathmem-claim-evidence-v1"


@dataclass(frozen=True, slots=True)
class ObservabilityCell:
    operator: OperatorKind
    contract: ContractFamily
    locus: LifecycleLocus
    status: ObservabilityStatus
    trace_field: str
    interpretation_boundary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_observability_matrix() -> tuple[ObservabilityCell, ...]:
    profiles = _operator_profiles()
    cells = tuple(
        ObservabilityCell(
            operator=operator,
            contract=contract,
            locus=locus,
            status=profiles[operator][locus][0],
            trace_field=profiles[operator][locus][1],
            interpretation_boundary=profiles[operator][locus][2],
        )
        for operator in OperatorKind
        for contract in ContractFamily
        for locus in LifecycleLocus
    )
    validate_observability_matrix(cells)
    return cells


def validate_observability_matrix(cells: tuple[ObservabilityCell, ...]) -> None:
    expected = {
        (operator, contract, locus)
        for operator in OperatorKind
        for contract in ContractFamily
        for locus in LifecycleLocus
    }
    observed = {(cell.operator, cell.contract, cell.locus) for cell in cells}
    if observed != expected or len(cells) != len(expected):
        missing = sorted(expected - observed)
        duplicate_count = len(cells) - len(observed)
        raise ValueError(
            f"observability matrix incomplete: missing={missing[:3]} duplicates={duplicate_count}"
        )
    for cell in cells:
        if cell.status is ObservabilityStatus.NOT_APPLICABLE and cell.trace_field != "N/A":
            raise ValueError("N/A observability cells must retain an explicit N/A trace field")
        if cell.status is not ObservabilityStatus.NOT_APPLICABLE and cell.trace_field == "N/A":
            raise ValueError("observable cells cannot use an N/A trace field")


def claim_evidence_matrix() -> tuple[dict[str, Any], ...]:
    rows = (
        {
            "claim_id": "C0",
            "claim": "The endpoint-controlled audit is technically valid.",
            "phase": "P0-P1",
            "required_evidence": (
                "shared full-parent hash; reducer endpoint equivalence; equal event/example "
                "multisets; common block RNG; fresh-session isolation; token/step audit"
            ),
            "falsifier": "Any lineage, endpoint, exposure, RNG, cache, or isolation mismatch.",
        },
        {
            "claim_id": "C1",
            "claim": "A named update operator violates terminal consistency.",
            "phase": "P1-P2",
            "required_evidence": "Item-paired endpoint-defect lower CI above epsilon_JS.",
            "falsifier": "Upper CI below epsilon_JS or a contaminated execution.",
        },
        {
            "claim_id": "C2",
            "claim": "Violation structure exceeds generic sequential-update geometry.",
            "phase": "P1b-P2",
            "required_evidence": (
                "Held-out incremental prediction beyond update-count, terminal-loss, gradient, "
                "optimizer-kernel, and Lie-bracket baselines."
            ),
            "falsifier": "No frozen held-out improvement beyond the complete baseline stack.",
        },
        {
            "claim_id": "C3",
            "claim": "The failure matters beyond synthetic reset-LoRA.",
            "phase": "P2",
            "required_evidence": (
                "Different model lineage, realistic task, existing parametric updater, textual "
                "consolidator, and a task-native consequence."
            ),
            "falsifier": "Failure is confined to nonce prompts, one lineage, or one toy operator.",
        },
        {
            "claim_id": "C4",
            "claim": "A new repair is needed.",
            "phase": "P3",
            "required_evidence": (
                "Restore, refit, modular, and existing-editor baselines leave a gap."
            ),
            "falsifier": "A simple baseline closes the resource-matched deployment gap.",
        },
        {
            "claim_id": "C5",
            "claim": "A contract-aware repair improves the deployment frontier.",
            "phase": "P3",
            "required_evidence": (
                "Untouched-data improvement in consistency, accuracy, locality, compute, storage, "
                "and rollback latency."
            ),
            "falsifier": "No resource-matched frontier improvement over simple baselines.",
        },
    )
    if {row["claim_id"] for row in rows} != {f"C{index}" for index in range(6)}:
        raise AssertionError("claim-evidence matrix changed or became incomplete")
    if any(not row["falsifier"] for row in rows):
        raise AssertionError("every central claim requires a falsifier")
    return rows


def non_claims() -> tuple[str, ...]:
    return (
        "first path dependence in language-model updates",
        "first optimizer-state memory",
        "first latest-write failure",
        "first schedule-sensitive textual consolidation",
        "first rollback or versioned memory",
        "first fresh-session persistent-memory attribution benchmark",
        "universal ranking of external versus parametric memory",
    )


def _operator_profiles() -> dict[
    OperatorKind,
    dict[LifecycleLocus, tuple[ObservabilityStatus, str, str]],
]:
    na = (ObservabilityStatus.NOT_APPLICABLE, "N/A", "No such lifecycle event exists.")
    textual = {
        LifecycleLocus.COMMIT_ENCODE: (
            ObservabilityStatus.INTERVENTION_ACCESSIBLE,
            "stored_artifact_sha256",
            "A committed artifact does not prove behavioral use.",
        ),
        LifecycleLocus.SUPERSEDE_RESOLVE: (
            ObservabilityStatus.INTERVENTION_ACCESSIBLE,
            "resolver_output_sha256",
            "Resolver alignment is pathway evidence until a valid intervention changes behavior.",
        ),
        LifecycleLocus.RETRIEVE_SURFACE: (
            ObservabilityStatus.INTERVENTION_ACCESSIBLE,
            "rendered_evidence_sha256",
            "Retrieved text is not proof of necessity.",
        ),
        LifecycleLocus.ACTIVATE: na,
        LifecycleLocus.APPLY_INFER: (
            ObservabilityStatus.OBSERVABLE,
            "candidate_distribution_sha256",
            "Behavior must be compared under a frozen scorer.",
        ),
        LifecycleLocus.MAINTAIN_FUTURE: (
            ObservabilityStatus.INTERVENTION_ACCESSIBLE,
            "version_or_future_suffix_sha256",
            "Future-suffix equivalence is not a rescue estimand.",
        ),
    }
    parametric = {
        LifecycleLocus.COMMIT_ENCODE: (
            ObservabilityStatus.OBSERVABLE,
            "event_loss_delta_and_child_sha256",
            "Canonical refit is a whole-operator repair, not a pure commit cut.",
        ),
        LifecycleLocus.SUPERSEDE_RESOLVE: (
            ObservabilityStatus.OBSERVABLE,
            "stale_current_behavior_sha256",
            "Internal semantic resolution is not directly observed.",
        ),
        LifecycleLocus.RETRIEVE_SURFACE: na,
        LifecycleLocus.ACTIVATE: (
            ObservabilityStatus.INTERVENTION_ACCESSIBLE,
            "active_adapter_sha256",
            "Adapter disable is an access control, not semantic rollback.",
        ),
        LifecycleLocus.APPLY_INFER: (
            ObservabilityStatus.OBSERVABLE,
            "candidate_distribution_sha256",
            "Inference traces do not identify the training mechanism alone.",
        ),
        LifecycleLocus.MAINTAIN_FUTURE: (
            ObservabilityStatus.INTERVENTION_ACCESSIBLE,
            "parent_state_or_future_suffix_sha256",
            "A common suffix tests operational equivalence only.",
        ),
    }
    no_write = {locus: na for locus in LifecycleLocus}
    no_write[LifecycleLocus.APPLY_INFER] = (
        ObservabilityStatus.OBSERVABLE,
        "base_candidate_distribution_sha256",
        "This is a reference arm, not absence of all model memory.",
    )
    hybrid = dict(parametric)
    hybrid[LifecycleLocus.RETRIEVE_SURFACE] = textual[LifecycleLocus.RETRIEVE_SURFACE]
    hybrid[LifecycleLocus.SUPERSEDE_RESOLVE] = textual[LifecycleLocus.SUPERSEDE_RESOLVE]
    return {
        OperatorKind.NO_ADDITIONAL_WRITE: no_write,
        OperatorKind.EXTERNAL_LATEST: dict(textual),
        OperatorKind.ICL_HISTORY: {
            **textual,
            LifecycleLocus.COMMIT_ENCODE: (
                ObservabilityStatus.OBSERVABLE,
                "history_event_sha256",
                "Chronological rendering intentionally preserves path order.",
            ),
        },
        OperatorKind.LORA_ADAMW_RESET: dict(parametric),
        OperatorKind.LORA_ADAMW_CARRY: dict(parametric),
        OperatorKind.LORA_SGD_MEMORYLESS: dict(parametric),
        OperatorKind.HYBRID_CURRENT_OVERRIDE: hybrid,
        OperatorKind.TEXTUAL_CONSOLIDATOR: dict(textual),
    }
