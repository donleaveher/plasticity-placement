from __future__ import annotations

import json
import math
import random
import re
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any

from plasticity_placement.p0c.runtime import current_code_hash
from plasticity_placement.p0d2hcrd.integrity_audit import (
    load_verified_rows,
    row_evidence,
)
from plasticity_placement.p0d2hcrd.runtime import validate_frozen_source
from plasticity_placement.p0d2hrr.analysis import (
    _load_jsonl,
    _validate_row_matrix,
)
from plasticity_placement.p0d2hrr.authorization import validate_authorization
from plasticity_placement.p0d2hrr.io import (
    file_hash,
    immutable_write,
    json_hash,
    read_json_object,
)
from plasticity_placement.p0d2hrr.manifest import PilotManifest
from plasticity_placement.p0d2hrr.preflight import (
    load_resolved_config,
    load_scale_canary_source_rows,
)
from plasticity_placement.p0d2hrr.runtime import _adapter_hash

AUDIT_SCHEMA_VERSION = "p0d2hrr-paired-audit-v1"
PAIRED_ROW_SCHEMA_VERSION = "p0d2hrr-paired-audit-row-v1"
FAILURE_ROW_SCHEMA_VERSION = "p0d2hrr-combined-failure-row-v1"
EXPECTED_BASE_CRD_RUN_ID = "p0d2hcrd-route-decomposition-8527b64f36"
DEFAULT_BOOTSTRAP_SEED = 20260730
ENDPOINTS = ("route_only", "retrieval_only", "combined")


@dataclass(frozen=True, slots=True)
class PairedAuditRequest:
    route_remediation_output: Path
    base_crd_output: Path
    analysis_output: Path
    experiment_code_revision_lock: Path | None = None
    base_crd_code_revision_lock: Path | None = None
    historical_preregistration_sha256: str | None = None
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED

    def __post_init__(self) -> None:
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        if self.bootstrap_seed < 0:
            raise ValueError("bootstrap_seed must be non-negative")
        if (
            self.historical_preregistration_sha256 is not None
            and re.fullmatch(
                r"[0-9a-f]{64}",
                self.historical_preregistration_sha256,
            )
            is None
        ):
            raise ValueError("historical_preregistration_sha256 must be a SHA-256 digest")


def audit_paired_results(request: PairedAuditRequest) -> Path:
    """Run the independent CPU-only P0 provenance and paired-result audit."""
    rr_output = request.route_remediation_output.resolve()
    base_crd_output = request.base_crd_output.resolve()
    analysis_output = request.analysis_output.resolve()
    revision_lock = (
        request.experiment_code_revision_lock.resolve()
        if request.experiment_code_revision_lock is not None
        else rr_output.parent.parent / "code_revision.txt"
    )
    base_revision_lock = (
        request.base_crd_code_revision_lock.resolve()
        if request.base_crd_code_revision_lock is not None
        else _default_base_crd_revision_lock(base_crd_output)
    )
    unvalidated_manifest = read_json_object(rr_output / "manifest.json", "RR manifest")
    source_manifest_path = Path(str(unvalidated_manifest["source_manifest_path"])).resolve()
    source_fc_output = source_manifest_path.parent
    _require_independent_output(
        (rr_output, base_crd_output, source_fc_output),
        analysis_output,
    )
    before_snapshot = _source_snapshot(
        rr_output,
        base_crd_output,
        source_fc_output,
        revision_lock,
        base_revision_lock,
    )

    (
        rr_provenance,
        base_fc_rows,
        adapter_fc_rows,
        adapter_crd_rows,
        rr_context,
    ) = _validate_route_remediation(rr_output, revision_lock)
    base_provenance, base_crd_rows = _validate_base_crd(
        base_crd_output,
        rr_context,
        base_revision_lock,
    )

    forced_choice_pairs = build_forced_choice_pairs(base_fc_rows, adapter_fc_rows)
    crd_pairs = build_crd_pairs(base_crd_rows, adapter_crd_rows)
    analysis, failure_records = build_paired_analysis(
        forced_choice_pairs,
        crd_pairs,
        base_crd_rows,
        adapter_crd_rows,
        bootstrap_samples=request.bootstrap_samples,
        bootstrap_seed=request.bootstrap_seed,
    )

    after_snapshot = _source_snapshot(
        rr_output,
        base_crd_output,
        source_fc_output,
        revision_lock,
        base_revision_lock,
    )
    if after_snapshot != before_snapshot:
        raise RuntimeError("source artifacts changed during the read-only paired audit")

    analysis_code_sha256 = current_code_hash()
    source_identity = {
        "route_remediation": rr_provenance,
        "base_crd": base_provenance,
        "frozen_forced_choice": {
            "output_dir": str(source_fc_output),
            "manifest_path": str(source_manifest_path),
            "manifest_sha256": file_hash(source_manifest_path),
            "run_id": rr_context["source_run_id"],
            "raw_results_sha256": rr_context["source_raw_results_sha256"],
        },
    }
    audit_identity = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "analysis_code_sha256": analysis_code_sha256,
        "source_identity": source_identity,
        "bootstrap_samples": request.bootstrap_samples,
        "bootstrap_seed": request.bootstrap_seed,
        "historical_preregistration_sha256": (request.historical_preregistration_sha256),
    }
    audit_run_id = "p0d2hrr-paired-audit-" + json_hash(audit_identity)[:10]
    artifact_paths = {
        "summary": analysis_output / "summary.json",
        "provenance": analysis_output / "provenance_report.json",
        "forced_choice_transitions": (analysis_output / "forced_choice_transitions.json"),
        "crd_transitions": analysis_output / "crd_transitions.json",
        "score_shifts": analysis_output / "score_shift_summary.json",
        "component_cells": analysis_output / "component_cell_analysis.json",
        "tie_bounds": analysis_output / "tie_bounds.json",
        "paired_records": analysis_output / "paired_records.jsonl",
        "combined_failures": analysis_output / "combined_failure_records.jsonl",
        "report": analysis_output / "report.md",
        "audit_manifest": analysis_output / "audit_manifest.json",
    }
    historical_observation = _historical_preregistration_observation(
        request.historical_preregistration_sha256,
        str(rr_provenance["preregistration_sha256"]),
    )
    summary = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "run_id": audit_run_id,
        "analysis_status": "post_hoc_read_only_p0",
        "source_artifacts_modified": False,
        "source_gate_status_changed": False,
        "next_stage_eligibility_changed": False,
        "training_authorized": False,
        "analysis_code_sha256": analysis_code_sha256,
        "bootstrap": {
            "samples": request.bootstrap_samples,
            "seed": request.bootstrap_seed,
            "cluster_unit": "lesson_id",
        },
        "source": source_identity,
        "current_artifact_graph_checks_passed": True,
        "historical_attempt_uniqueness_verified": False,
        "historical_preregistration_observation": historical_observation,
        "historical_continuity_resolved": (historical_observation["matches_current_disk"]),
        "paper_evidence_ready": False,
        "analysis": analysis,
        "artifact_paths": {key: str(path) for key, path in artifact_paths.items()},
        "execution_environment_comparability": {
            "same_session_rescoring": False,
            "historical_runtime_identity_verified": False,
            "status": "not_verifiable_from_persisted_rows",
        },
        "interpretation_boundary": (
            "This audit reports paired differences and component associations. "
            "The base and adapter were scored in separate historical runs whose "
            "runtime equivalence cannot be verified from persisted rows, so the "
            "differences are not causal adapter effects. It does not change the "
            "frozen gate, authorize training, or identify a combined-task error "
            "as a unique wrong-slot decision."
        ),
    }

    payloads = {
        artifact_paths["summary"]: _json_bytes(summary),
        artifact_paths["provenance"]: _json_bytes(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "run_id": audit_run_id,
                "current_artifact_graph_checks_passed": True,
                "historical_attempt_uniqueness_verified": False,
                "source": source_identity,
                "historical_preregistration_observation": (historical_observation),
                "before_snapshot": before_snapshot,
                "after_snapshot": after_snapshot,
            }
        ),
        artifact_paths["forced_choice_transitions"]: _json_bytes(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "run_id": audit_run_id,
                **analysis["forced_choice"],
            }
        ),
        artifact_paths["crd_transitions"]: _json_bytes(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "run_id": audit_run_id,
                **analysis["crd"],
            }
        ),
        artifact_paths["score_shifts"]: _json_bytes(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "run_id": audit_run_id,
                "forced_choice": analysis["forced_choice"]["score_shifts"],
                "crd": analysis["crd"]["score_shifts"],
            }
        ),
        artifact_paths["component_cells"]: _json_bytes(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "run_id": audit_run_id,
                **analysis["component_cells"],
            }
        ),
        artifact_paths["tie_bounds"]: _json_bytes(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "run_id": audit_run_id,
                **analysis["tie_bounds"],
            }
        ),
        artifact_paths["paired_records"]: _jsonl_bytes(
            [
                {"schema_version": PAIRED_ROW_SCHEMA_VERSION, "matrix": "forced_choice", **row}
                for row in forced_choice_pairs
            ]
            + [
                {"schema_version": PAIRED_ROW_SCHEMA_VERSION, "matrix": "crd", **row}
                for row in crd_pairs
            ]
        ),
        artifact_paths["combined_failures"]: _jsonl_bytes(failure_records),
        artifact_paths["report"]: _render_report(summary).encode(),
    }
    analysis_output.mkdir(parents=True, exist_ok=True)
    artifact_hashes: dict[str, str] = {}
    for name, path in artifact_paths.items():
        if name == "audit_manifest":
            continue
        payload = payloads[path]
        immutable_write(path, payload, f"paired audit {name}")
        artifact_hashes[name] = sha256(payload).hexdigest()
    pre_manifest_snapshot = _source_snapshot(
        rr_output,
        base_crd_output,
        source_fc_output,
        revision_lock,
        base_revision_lock,
    )
    if pre_manifest_snapshot != before_snapshot:
        raise RuntimeError("source artifacts changed while paired-audit outputs were being written")
    audit_manifest = {
        "schema_version": "p0d2hrr-paired-audit-manifest-v1",
        "run_id": audit_run_id,
        "analysis_code_sha256": analysis_code_sha256,
        "source_identity_sha256": json_hash(source_identity),
        "bootstrap_samples": request.bootstrap_samples,
        "bootstrap_seed": request.bootstrap_seed,
        "artifacts": {
            name: {
                "path": str(artifact_paths[name]),
                "sha256": digest,
            }
            for name, digest in artifact_hashes.items()
        },
        "source_artifacts_modified": False,
        "source_gate_status_changed": False,
        "next_stage_eligibility_changed": False,
    }
    immutable_write(
        artifact_paths["audit_manifest"],
        _json_bytes(audit_manifest),
        "paired audit manifest",
    )
    final_snapshot = _source_snapshot(
        rr_output,
        base_crd_output,
        source_fc_output,
        revision_lock,
        base_revision_lock,
    )
    if final_snapshot != before_snapshot:
        raise RuntimeError("source artifacts changed before paired-audit publication completed")
    return artifact_paths["summary"]


def build_forced_choice_pairs(
    base_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base = _unique_index(base_rows, _forced_choice_key, "base forced-choice")
    adapter = _unique_index(adapter_rows, _forced_choice_key, "adapter forced-choice")
    if set(base) != set(adapter):
        raise ValueError("base and adapter forced-choice key sets differ")
    pairs: list[dict[str, Any]] = []
    for key in sorted(base):
        base_row = base[key]
        adapter_row = adapter[key]
        _require_equal_fields(
            base_row,
            adapter_row,
            (
                "lesson_id",
                "pair_id",
                "lesson_type",
                "probe_id",
                "category",
                "arm",
                "expected_action",
                "ordered_allowed_actions",
                "prompt_variant",
                "prompt_rendering_version",
                "prompt_sha256",
                "precision",
                "model_name",
                "model_revision",
            ),
            f"forced-choice pair {key}",
        )
        _require_candidate_static_match(base_row, adapter_row, f"forced-choice pair {key}")
        if adapter_row.get("source_row_sha256") != json_hash(base_row):
            raise ValueError(f"forced-choice source-row hash differs for {key}")
        base_margin = signed_expected_margin(base_row, "expected_action")
        adapter_margin = signed_expected_margin(adapter_row, "expected_action")
        pairs.append(
            {
                "lesson_id": str(base_row["lesson_id"]),
                "pair_id": str(base_row["pair_id"]),
                "lesson_type": str(base_row["lesson_type"]),
                "probe_id": str(base_row["probe_id"]),
                "category": str(base_row["category"]),
                "arm": str(base_row["arm"]),
                "expected_candidate": str(base_row["expected_action"]),
                "ordered_candidates": list(base_row["ordered_allowed_actions"]),
                "prompt_sha256": str(base_row["prompt_sha256"]),
                "base_predicted_candidate": base_row.get("predicted_action"),
                "adapter_predicted_candidate": adapter_row.get("predicted_action"),
                "base_correct": bool(base_row.get("correct")),
                "adapter_correct": bool(adapter_row.get("correct")),
                "base_error_status": str(base_row.get("error_status")),
                "adapter_error_status": str(adapter_row.get("error_status")),
                "base_tie": bool(base_row.get("tie")),
                "adapter_tie": bool(adapter_row.get("tie")),
                "base_tie_expected_compatible": _expected_in_top_tie(
                    base_row,
                    "expected_action",
                ),
                "adapter_tie_expected_compatible": _expected_in_top_tie(
                    adapter_row,
                    "expected_action",
                ),
                "base_top1_top2_margin": base_row.get("top1_top2_margin"),
                "adapter_top1_top2_margin": adapter_row.get("top1_top2_margin"),
                "base_signed_expected_margin": base_margin,
                "adapter_signed_expected_margin": adapter_margin,
                "signed_expected_margin_delta": _optional_difference(
                    adapter_margin,
                    base_margin,
                ),
                "base_scores": _candidate_score_map(base_row),
                "adapter_scores": _candidate_score_map(adapter_row),
                "prediction_changed": (
                    base_row.get("predicted_action") != adapter_row.get("predicted_action")
                ),
                "transition": _transition(base_row, adapter_row),
                "base_row_sha256": json_hash(base_row),
                "adapter_row_sha256": json_hash(adapter_row),
            }
        )
    return pairs


def build_crd_pairs(
    base_rows: list[dict[str, Any]],
    adapter_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base = _unique_index(base_rows, lambda row: str(row["probe_id"]), "base CRD")
    adapter = _unique_index(adapter_rows, lambda row: str(row["probe_id"]), "adapter CRD")
    if set(base) != set(adapter):
        raise ValueError("base and adapter CRD probe sets differ")
    pairs: list[dict[str, Any]] = []
    for probe_id in sorted(base):
        base_row = base[probe_id]
        adapter_row = adapter[probe_id]
        _require_equal_fields(
            base_row,
            adapter_row,
            (
                "lesson_id",
                "pair_id",
                "lesson_type",
                "lesson_side",
                "probe_id",
                "endpoint",
                "expected_candidate",
                "ordered_candidates",
                "route_variant",
                "target_slot",
                "current_marker",
                "slot_candidate_order",
                "slot_content_order",
                "action_panel_position",
                "prompt_sha256",
                "source_probe_id",
                "source_row_key",
                "source_row_sha256",
                "precision",
                "model_name",
                "model_revision",
            ),
            f"CRD pair {probe_id}",
        )
        _require_candidate_static_match(base_row, adapter_row, f"CRD pair {probe_id}")
        base_margin = signed_expected_margin(base_row, "expected_candidate")
        adapter_margin = signed_expected_margin(adapter_row, "expected_candidate")
        pairs.append(
            {
                "lesson_id": str(base_row["lesson_id"]),
                "pair_id": str(base_row["pair_id"]),
                "lesson_type": str(base_row["lesson_type"]),
                "lesson_side": str(base_row["lesson_side"]),
                "probe_id": probe_id,
                "endpoint": str(base_row["endpoint"]),
                "expected_candidate": str(base_row["expected_candidate"]),
                "ordered_candidates": list(base_row["ordered_candidates"]),
                "route_variant": base_row.get("route_variant"),
                "target_slot": base_row.get("target_slot"),
                "slot_candidate_order": base_row.get("slot_candidate_order"),
                "slot_content_order": base_row.get("slot_content_order"),
                "action_panel_position": base_row.get("action_panel_position"),
                "prompt_sha256": str(base_row["prompt_sha256"]),
                "source_row_key": str(base_row["source_row_key"]),
                "source_row_sha256": str(base_row["source_row_sha256"]),
                "source_fc_correct": bool(base_row.get("source_fc_correct")),
                "base_predicted_candidate": base_row.get("predicted_candidate"),
                "adapter_predicted_candidate": adapter_row.get("predicted_candidate"),
                "base_correct": bool(base_row.get("correct")),
                "adapter_correct": bool(adapter_row.get("correct")),
                "base_error_status": str(base_row.get("error_status")),
                "adapter_error_status": str(adapter_row.get("error_status")),
                "base_tie": bool(base_row.get("tie")),
                "adapter_tie": bool(adapter_row.get("tie")),
                "base_tie_expected_compatible": _expected_in_top_tie(
                    base_row,
                    "expected_candidate",
                ),
                "adapter_tie_expected_compatible": _expected_in_top_tie(
                    adapter_row,
                    "expected_candidate",
                ),
                "base_top1_top2_margin": base_row.get("top1_top2_margin"),
                "adapter_top1_top2_margin": adapter_row.get("top1_top2_margin"),
                "base_signed_expected_margin": base_margin,
                "adapter_signed_expected_margin": adapter_margin,
                "signed_expected_margin_delta": _optional_difference(
                    adapter_margin,
                    base_margin,
                ),
                "base_scores": _candidate_score_map(base_row),
                "adapter_scores": _candidate_score_map(adapter_row),
                "prediction_changed": (
                    base_row.get("predicted_candidate") != adapter_row.get("predicted_candidate")
                ),
                "transition": _transition(base_row, adapter_row),
                "base_row_sha256": json_hash(base_row),
                "adapter_row_sha256": json_hash(adapter_row),
            }
        )
    return pairs


def signed_expected_margin(
    row: dict[str, Any],
    expected_field: str,
) -> float | None:
    expected = row.get(expected_field)
    candidates = row.get("candidates")
    if expected is None or not isinstance(candidates, list):
        return None
    expected_scores = [
        candidate.get("sum_logprob")
        for candidate in candidates
        if _candidate_name(candidate) == expected
    ]
    distractor_scores = [
        candidate.get("sum_logprob")
        for candidate in candidates
        if _candidate_name(candidate) != expected
    ]
    if len(expected_scores) != 1 or not distractor_scores:
        return None
    values = [expected_scores[0], *distractor_scores]
    if not all(_finite(value) for value in values):
        return None
    return float(expected_scores[0]) - max(float(value) for value in distractor_scores)


def paired_summary(
    pairs: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("cannot summarize an empty paired matrix")
    transitions = Counter(str(row["transition"]) for row in pairs)
    base_correct = sum(bool(row["base_correct"]) for row in pairs)
    adapter_correct = sum(bool(row["adapter_correct"]) for row in pairs)
    accuracy_difference = _cluster_bootstrap(
        pairs,
        lambda row: float(bool(row["adapter_correct"])) - float(bool(row["base_correct"])),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    margin_difference = _cluster_bootstrap(
        pairs,
        lambda row: _optional_float(row.get("signed_expected_margin_delta")),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed + 1,
    )
    margin_deltas = [
        float(value)
        for row in pairs
        if (value := row.get("signed_expected_margin_delta")) is not None
    ]
    changed = sum(bool(row["prediction_changed"]) for row in pairs)
    return {
        "decision_count": len(pairs),
        "lesson_count": len({str(row["lesson_id"]) for row in pairs}),
        "base_accuracy": base_correct / len(pairs),
        "adapter_accuracy": adapter_correct / len(pairs),
        "adapter_minus_base_accuracy": (adapter_correct - base_correct) / len(pairs),
        "paired_accuracy_difference": {
            **accuracy_difference,
            "role": "primary_lesson_cluster_bootstrap",
        },
        "transition_counts": {
            name: transitions.get(name, 0)
            for name in (
                "correct_to_correct",
                "correct_to_wrong",
                "wrong_to_correct",
                "wrong_to_wrong",
            )
        },
        "prediction_changed_count": changed,
        "prediction_changed_rate": changed / len(pairs),
        "exact_mcnemar": exact_mcnemar(
            transitions.get("correct_to_wrong", 0),
            transitions.get("wrong_to_correct", 0),
        ),
        "signed_expected_margin": {
            "complete_pair_count": len(margin_deltas),
            "mean_adapter_minus_base": mean(margin_deltas) if margin_deltas else None,
            "median_adapter_minus_base": median(margin_deltas) if margin_deltas else None,
            "paired_difference": {
                **margin_difference,
                "role": "primary_lesson_cluster_bootstrap",
            },
        },
        "tie_sensitivity": {
            "base": _tie_bounds(pairs, "base"),
            "adapter": _tie_bounds(pairs, "adapter"),
        },
    }


def exact_mcnemar(correct_to_wrong: int, wrong_to_correct: int) -> dict[str, Any]:
    if correct_to_wrong < 0 or wrong_to_correct < 0:
        raise ValueError("McNemar discordant counts must be non-negative")
    discordant = correct_to_wrong + wrong_to_correct
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(correct_to_wrong, wrong_to_correct)
        probability = sum(math.comb(discordant, value) for value in range(tail + 1))
        p_value = min(1.0, 2.0 * probability / (2**discordant))
    return {
        "correct_to_wrong": correct_to_wrong,
        "wrong_to_correct": wrong_to_correct,
        "discordant_count": discordant,
        "two_sided_exact_p": p_value,
        "role": "secondary_row_level_test_rows_are_clustered_by_lesson",
    }


def build_paired_analysis(
    forced_choice_pairs: list[dict[str, Any]],
    crd_pairs: list[dict[str, Any]],
    base_crd_rows: list[dict[str, Any]],
    adapter_crd_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    primary_fc = [
        row
        for row in forced_choice_pairs
        if row["arm"] == "external" and row["category"] == "conditional_route"
    ]
    if len(primary_fc) != 96:
        raise ValueError("external conditional-route paired matrix must contain 96 rows")
    fc_overall = paired_summary(
        forced_choice_pairs,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=_derived_seed(bootstrap_seed, "fc-overall"),
    )
    fc_primary = paired_summary(
        primary_fc,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=_derived_seed(bootstrap_seed, "fc-primary"),
    )
    fc_by_arm_category = _grouped_paired_summaries(
        forced_choice_pairs,
        ("arm", "category"),
        bootstrap_samples,
        bootstrap_seed,
    )

    crd_by_endpoint: dict[str, dict[str, Any]] = {}
    for endpoint in ENDPOINTS:
        values = [row for row in crd_pairs if row["endpoint"] == endpoint]
        expected_count = {"route_only": 384, "retrieval_only": 384, "combined": 768}[endpoint]
        if len(values) != expected_count:
            raise ValueError(f"{endpoint} paired matrix must contain {expected_count} rows")
        crd_by_endpoint[endpoint] = paired_summary(
            values,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=_derived_seed(bootstrap_seed, f"crd-{endpoint}"),
        )

    base_component, base_failures = build_combined_failure_taxonomy(
        base_crd_rows,
        model_label="base",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=_derived_seed(bootstrap_seed, "base-components"),
    )
    adapter_component, adapter_failures = build_combined_failure_taxonomy(
        adapter_crd_rows,
        model_label="adapter",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=_derived_seed(bootstrap_seed, "adapter-components"),
    )
    component_cells = {
        "classification_scope": (
            "Association-only taxonomy. A wrong combined action is not interpreted "
            "as a unique wrong-slot decision."
        ),
        "base": base_component,
        "adapter": adapter_component,
        "base_to_adapter_failure_count_differences": {
            "combined_failure_count": (
                int(adapter_component["combined_failure_count"])
                - int(base_component["combined_failure_count"])
            ),
            "by_category": {
                category: (
                    int(adapter_component["failure_category_counts"].get(category, 0))
                    - int(base_component["failure_category_counts"].get(category, 0))
                )
                for category in sorted(
                    set(base_component["failure_category_counts"])
                    | set(adapter_component["failure_category_counts"])
                )
            },
        },
        "retrieval_to_combined_margin_shift": {
            "base": retrieval_to_combined_margin_shift(
                base_crd_rows,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=_derived_seed(bootstrap_seed, "base-retrieval-combined"),
            ),
            "adapter": retrieval_to_combined_margin_shift(
                adapter_crd_rows,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=_derived_seed(
                    bootstrap_seed,
                    "adapter-retrieval-combined",
                ),
            ),
        },
        "base_to_adapter_composition_margin_interaction": (
            paired_composition_margin_interaction(
                crd_pairs,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=_derived_seed(
                    bootstrap_seed,
                    "paired-composition-interaction",
                ),
            )
        ),
    }

    secondary_row_p_values = {
        "external_conditional_route": float(fc_primary["exact_mcnemar"]["two_sided_exact_p"]),
        **{
            endpoint: float(values["exact_mcnemar"]["two_sided_exact_p"])
            for endpoint, values in crd_by_endpoint.items()
        },
    }
    holm = _holm_adjust(secondary_row_p_values)
    factor_fields = {
        "route_only": (
            "expected_candidate",
            "lesson_type",
            "lesson_side",
            "route_variant",
            "target_slot",
            "slot_candidate_order",
            "slot_content_order",
        ),
        "retrieval_only": (
            "expected_candidate",
            "lesson_type",
            "lesson_side",
            "route_variant",
            "action_panel_position",
        ),
        "combined": (
            "expected_candidate",
            "lesson_type",
            "lesson_side",
            "route_variant",
            "target_slot",
            "slot_content_order",
            "action_panel_position",
        ),
    }
    factor_effects = {
        "scope": (
            "Descriptive base-to-adapter strata within endpoint; no cross-endpoint "
            "null levels are pooled and no causal factor effect is claimed."
        ),
        "by_endpoint": {
            endpoint: {
                field: _factor_effects(
                    [row for row in crd_pairs if row["endpoint"] == endpoint],
                    field,
                )
                for field in fields
            }
            for endpoint, fields in factor_fields.items()
        },
    }
    analysis = {
        "forced_choice": {
            "overall": fc_overall,
            "external_conditional_route": fc_primary,
            "by_arm_and_category": fc_by_arm_category,
            "confusion_external_conditional_route": {
                "base": _confusion(primary_fc, "base"),
                "adapter": _confusion(primary_fc, "adapter"),
            },
            "score_shifts": {
                "overall_signed_expected_margin": fc_overall["signed_expected_margin"],
                "external_conditional_route_signed_expected_margin": fc_primary[
                    "signed_expected_margin"
                ],
            },
        },
        "crd": {
            "by_endpoint": crd_by_endpoint,
            "factor_effects": factor_effects,
            "combined_confusion": {
                "base": _confusion(
                    [row for row in crd_pairs if row["endpoint"] == "combined"],
                    "base",
                ),
                "adapter": _confusion(
                    [row for row in crd_pairs if row["endpoint"] == "combined"],
                    "adapter",
                ),
            },
            "predicted_candidate_position": {
                endpoint: {
                    model: _prediction_position_summary(
                        [row for row in crd_pairs if row["endpoint"] == endpoint],
                        model,
                    )
                    for model in ("base", "adapter")
                }
                for endpoint in ENDPOINTS
            },
            "score_shifts": {
                endpoint: values["signed_expected_margin"]
                for endpoint, values in crd_by_endpoint.items()
            },
        },
        "component_cells": component_cells,
        "tie_bounds": {
            "forced_choice_external_conditional_route": fc_primary["tie_sensitivity"],
            "crd": {
                endpoint: values["tie_sensitivity"] for endpoint, values in crd_by_endpoint.items()
            },
        },
        "multiple_testing": {
            "family": "four_exploratory_secondary_row_level_binary_contrasts",
            "method": "Holm",
            "role": "secondary_sensitivity_only",
            "limitation": (
                "Rows are clustered within lesson. Holm controls multiplicity across "
                "these row-level tests but does not restore within-lesson independence; "
                "decision-relevant uncertainty is the lesson-cluster bootstrap."
            ),
            "tests": holm,
        },
    }
    return analysis, [*base_failures, *adapter_failures]


def build_combined_failure_taxonomy(
    rows: list[dict[str, Any]],
    *,
    model_label: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    route_rows = [row for row in rows if row.get("endpoint") == "route_only"]
    retrieval_rows = [row for row in rows if row.get("endpoint") == "retrieval_only"]
    combined_rows = [row for row in rows if row.get("endpoint") == "combined"]
    route_index: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in route_rows:
        route_index[
            (
                str(row["lesson_id"]),
                str(row["route_variant"]),
                str(row["slot_content_order"]),
            )
        ].append(row)
    retrieval_index = _unique_index(
        retrieval_rows,
        lambda row: (
            str(row["lesson_id"]),
            str(row["route_variant"]),
            str(row["action_panel_position"]),
        ),
        f"{model_label} retrieval",
    )
    records: list[dict[str, Any]] = []
    for combined in combined_rows:
        route_key = (
            str(combined["lesson_id"]),
            str(combined["route_variant"]),
            str(combined["slot_content_order"]),
        )
        retrieval_key = (
            str(combined["lesson_id"]),
            str(combined["route_variant"]),
            str(combined["action_panel_position"]),
        )
        routes = sorted(
            route_index.get(route_key, []),
            key=lambda row: int(row["slot_candidate_order"]),
        )
        retrieval = retrieval_index.get(retrieval_key)
        if len(routes) != 2 or retrieval is None:
            raise ValueError(
                f"{model_label} combined component cell is incomplete: {combined['probe_id']}"
            )
        route_ok = [str(row.get("error_status")) == "ok" for row in routes]
        route_correct = [bool(row.get("correct")) for row in routes]
        route_all_correct = all(route_ok) and all(route_correct)
        route_all_wrong = all(route_ok) and not any(route_correct)
        route_ambiguous = not route_all_correct and not route_all_wrong
        retrieval_valid = str(retrieval.get("error_status")) == "ok"
        retrieval_correct = retrieval_valid and bool(retrieval.get("correct"))
        combined_correct = bool(combined.get("correct"))
        category = _failure_category(
            combined,
            route_all_correct=route_all_correct,
            route_all_wrong=route_all_wrong,
            route_ambiguous=route_ambiguous,
            retrieval_correct=retrieval_correct,
        )
        record = {
            "schema_version": FAILURE_ROW_SCHEMA_VERSION,
            "model": model_label,
            "lesson_id": str(combined["lesson_id"]),
            "probe_id": str(combined["probe_id"]),
            "route_variant": combined.get("route_variant"),
            "slot_content_order": combined.get("slot_content_order"),
            "action_panel_position": combined.get("action_panel_position"),
            "expected_candidate": combined.get("expected_candidate"),
            "predicted_candidate": combined.get("predicted_candidate"),
            "combined_correct": combined_correct,
            "combined_error_status": str(combined.get("error_status")),
            "combined_signed_expected_margin": signed_expected_margin(
                combined,
                "expected_candidate",
            ),
            "route_probe_ids": [str(row["probe_id"]) for row in routes],
            "route_correct": route_correct,
            "route_error_statuses": [str(row.get("error_status")) for row in routes],
            "route_all_correct": route_all_correct,
            "route_all_wrong": route_all_wrong,
            "route_ambiguous": route_ambiguous,
            "retrieval_probe_id": str(retrieval["probe_id"]),
            "retrieval_correct": retrieval_correct,
            "retrieval_error_status": str(retrieval.get("error_status")),
            "failure_category": category,
        }
        records.append(record)
    if len(records) != len(combined_rows):
        raise AssertionError("combined component record count changed")
    failures = [record for record in records if not record["combined_correct"]]
    categories = Counter(str(record["failure_category"]) for record in failures)
    route_good = [record for record in records if record["route_all_correct"]]
    route_wrong = [record for record in records if record["route_all_wrong"]]
    route_ambiguous = [record for record in records if record["route_ambiguous"]]
    association_eligible = [
        record
        for record in records
        if (record["route_all_correct"] or record["route_all_wrong"])
        and record["combined_error_status"] == "ok"
    ]
    association_route_good = [
        record for record in association_eligible if record["route_all_correct"]
    ]
    association_route_wrong = [
        record for record in association_eligible if record["route_all_wrong"]
    ]
    association_records = [
        {
            "lesson_id": record["lesson_id"],
            "route_failed": bool(record["route_all_wrong"]),
            "combined_error": not bool(record["combined_correct"]),
        }
        for record in association_eligible
    ]
    return (
        {
            "combined_decision_count": len(records),
            "combined_correct_count": len(records) - len(failures),
            "combined_failure_count": len(failures),
            "combined_tie_count": sum(
                record["combined_error_status"] == "tie" for record in records
            ),
            "failure_category_counts": dict(sorted(categories.items())),
            "route_all_correct_count": len(route_good),
            "route_all_wrong_count": len(route_wrong),
            "route_ambiguous_count": len(route_ambiguous),
            "route_association_excluded_ambiguous_count": len(route_ambiguous),
            "route_association_excluded_combined_non_ok_count": sum(
                record["combined_error_status"] != "ok"
                and (record["route_all_correct"] or record["route_all_wrong"])
                for record in records
            ),
            "route_association_eligible_count": len(association_eligible),
            "route_association_all_correct_count": len(association_route_good),
            "route_association_all_wrong_count": len(association_route_wrong),
            "route_association_scope": (
                "Definite route cells only: route replicas all correct versus all "
                "wrong. Mixed-replica route cells and non-ok combined outcomes are "
                "excluded."
            ),
            "combined_error_given_route_all_correct": _boolean_rate(
                association_route_good,
                lambda row: not bool(row["combined_correct"]),
            ),
            "combined_error_given_route_all_wrong": _boolean_rate(
                association_route_wrong,
                lambda row: not bool(row["combined_correct"]),
            ),
            "combined_error_given_route_ambiguous": _boolean_rate(
                route_ambiguous,
                lambda row: not bool(row["combined_correct"]),
            ),
            "route_failure_association": _clustered_binary_rate_difference(
                association_records,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            ),
            "failure_records_emitted": len(failures),
        },
        failures,
    )


def retrieval_to_combined_margin_shift(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    retrieval = _unique_index(
        [row for row in rows if row.get("endpoint") == "retrieval_only"],
        lambda row: (
            str(row["lesson_id"]),
            str(row["route_variant"]),
            str(row["action_panel_position"]),
        ),
        "retrieval-only margin source",
    )
    values: list[dict[str, Any]] = []
    for row in rows:
        if row.get("endpoint") != "combined":
            continue
        key = (
            str(row["lesson_id"]),
            str(row["route_variant"]),
            str(row["action_panel_position"]),
        )
        matched = retrieval.get(key)
        if matched is None:
            raise ValueError(f"combined row lacks retrieval match: {row['probe_id']}")
        if row.get("expected_candidate") != matched.get("expected_candidate") or row.get(
            "ordered_candidates"
        ) != matched.get("ordered_candidates"):
            raise ValueError(f"combined/retrieval candidate mismatch: {row['probe_id']}")
        retrieval_margin = signed_expected_margin(matched, "expected_candidate")
        combined_margin = signed_expected_margin(row, "expected_candidate")
        retrieval_expected, retrieval_distractor = _expected_and_best_distractor_scores(
            matched,
            "expected_candidate",
        )
        combined_expected, combined_distractor = _expected_and_best_distractor_scores(
            row,
            "expected_candidate",
        )
        values.append(
            {
                "lesson_id": str(row["lesson_id"]),
                "margin_delta": _optional_difference(
                    combined_margin,
                    retrieval_margin,
                ),
                "expected_logprob_delta": combined_expected - retrieval_expected,
                "best_distractor_logprob_delta": (combined_distractor - retrieval_distractor),
                "retrieval_correct": bool(matched.get("correct")),
                "combined_correct": bool(row.get("correct")),
            }
        )
    complete = [float(value) for row in values if (value := row["margin_delta"]) is not None]
    expected_deltas = [float(row["expected_logprob_delta"]) for row in values]
    distractor_deltas = [float(row["best_distractor_logprob_delta"]) for row in values]
    return {
        "paired_combined_row_count": len(values),
        "complete_margin_pair_count": len(complete),
        "mean_combined_minus_retrieval_signed_margin": (mean(complete) if complete else None),
        "median_combined_minus_retrieval_signed_margin": (median(complete) if complete else None),
        "lesson_clustered_difference": _cluster_bootstrap(
            values,
            lambda row: _optional_float(row["margin_delta"]),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
        "expected_action_logprob_shift": {
            "mean_combined_minus_retrieval": mean(expected_deltas),
            "median_combined_minus_retrieval": median(expected_deltas),
            "lesson_clustered_difference": _cluster_bootstrap(
                values,
                lambda item: float(item["expected_logprob_delta"]),
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=_derived_seed(
                    bootstrap_seed,
                    "expected-action-logprob",
                ),
            ),
        },
        "best_distractor_logprob_shift": {
            "mean_combined_minus_retrieval": mean(distractor_deltas),
            "median_combined_minus_retrieval": median(distractor_deltas),
            "lesson_clustered_difference": _cluster_bootstrap(
                values,
                lambda item: float(item["best_distractor_logprob_delta"]),
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=_derived_seed(
                    bootstrap_seed,
                    "best-distractor-logprob",
                ),
            ),
        },
        "retrieval_correct_combined_wrong_count": sum(
            bool(row["retrieval_correct"]) and not bool(row["combined_correct"]) for row in values
        ),
    }


def paired_composition_margin_interaction(
    pairs: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    retrieval = _unique_index(
        [row for row in pairs if row.get("endpoint") == "retrieval_only"],
        lambda row: (
            str(row["lesson_id"]),
            str(row["route_variant"]),
            str(row["action_panel_position"]),
        ),
        "paired retrieval-only composition source",
    )
    records: list[dict[str, Any]] = []
    for row in pairs:
        if row.get("endpoint") != "combined":
            continue
        key = (
            str(row["lesson_id"]),
            str(row["route_variant"]),
            str(row["action_panel_position"]),
        )
        matched = retrieval.get(key)
        if matched is None:
            raise ValueError(f"paired combined row lacks retrieval match: {row['probe_id']}")
        if row.get("expected_candidate") != matched.get("expected_candidate") or row.get(
            "ordered_candidates"
        ) != matched.get("ordered_candidates"):
            raise ValueError(f"paired combined/retrieval candidate mismatch: {row['probe_id']}")
        records.append(
            {
                "lesson_id": str(row["lesson_id"]),
                "interaction": _optional_difference(
                    _optional_float(row.get("signed_expected_margin_delta")),
                    _optional_float(matched.get("signed_expected_margin_delta")),
                ),
            }
        )
    complete = [float(value) for record in records if (value := record["interaction"]) is not None]
    return {
        "contrast": (
            "(adapter-base combined signed margin) - (adapter-base retrieval-only signed margin)"
        ),
        "paired_combined_row_count": len(records),
        "complete_interaction_count": len(complete),
        "mean_interaction": mean(complete) if complete else None,
        "median_interaction": median(complete) if complete else None,
        "lesson_clustered_difference": _cluster_bootstrap(
            records,
            lambda record: _optional_float(record["interaction"]),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        ),
    }


def _validate_route_remediation(
    output_dir: Path,
    revision_lock: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    manifest_path = output_dir / "manifest.json"
    manifest = PilotManifest.load(manifest_path)
    if manifest.state != "verified" or manifest.payload.get("errors") != []:
        raise ValueError("route-remediation source must be verified without errors")
    config = load_resolved_config(output_dir)
    preregistration_path = output_dir / "preregistration.json"
    preregistration = read_json_object(preregistration_path, "RR preregistration")
    claimed_preregistration_hash = str(preregistration["preregistration_sha256"])
    computed_preregistration_hash = json_hash(
        {key: value for key, value in preregistration.items() if key != "preregistration_sha256"}
    )
    if (
        preregistration != manifest.payload.get("config")
        or claimed_preregistration_hash != computed_preregistration_hash
        or claimed_preregistration_hash != config.preregistration_sha256
        or manifest.payload.get("run_id") != f"p0d2hrr-{claimed_preregistration_hash[:10]}"
    ):
        raise ValueError("route-remediation preregistration/run identity differs")

    authorization_path = output_dir / "authorization.json"
    authorization = validate_authorization(
        read_json_object(authorization_path, "RR authorization"),
        preregistration_sha256=claimed_preregistration_hash,
    )
    manifest_authorization = manifest.payload.get("authorization")
    if not isinstance(manifest_authorization, dict) or (
        manifest_authorization.get("sha256") != file_hash(authorization_path)
        or manifest_authorization.get("approved_by") != authorization["approved_by"]
        or manifest_authorization.get("approved_at") != authorization["approved_at"]
        or manifest_authorization.get("scope") != authorization["scope"]
    ):
        raise ValueError("route-remediation authorization provenance differs")

    if not revision_lock.is_file():
        raise FileNotFoundError(f"missing RR code revision lock: {revision_lock}")
    experiment_code_revision = revision_lock.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", experiment_code_revision):
        raise ValueError("RR code revision lock is not a full Git commit SHA")
    revision_code_sha256 = _git_revision_code_hash(
        Path(__file__).resolve().parents[3],
        experiment_code_revision,
    )
    if revision_code_sha256 != config.code_sha256:
        raise ValueError("RR code revision does not reproduce preregistered code hash")

    training = manifest.payload.get("training")
    evaluation = manifest.payload.get("evaluation")
    aggregate = manifest.payload.get("aggregate")
    if not all(isinstance(value, dict) for value in (training, evaluation, aggregate)):
        raise ValueError("route-remediation lifecycle metadata is incomplete")
    assert isinstance(training, dict)
    assert isinstance(evaluation, dict)
    assert isinstance(aggregate, dict)
    adapter_sha256 = _adapter_hash(output_dir / "adapter")
    training_metadata_path = output_dir / "adapter" / "training_metadata.json"
    training_metadata = read_json_object(
        training_metadata_path,
        "route-remediation training metadata",
    )
    expected_training_config = config.spec.training.to_lora_config(
        model_name=config.spec.model_name,
        model_revision=config.spec.model_revision,
        data_path=output_dir / "preflight" / "route_train.jsonl",
        output_dir=output_dir / "adapter",
    ).to_dict()
    if (
        training.get("completed_training_runs") != 1
        or training.get("training_metadata_sha256") != file_hash(training_metadata_path)
        or training_metadata.get("config") != expected_training_config
        or training_metadata.get("summary") != training.get("summary")
        or training.get("summary", {}).get("adapter_sha256") != adapter_sha256
        or evaluation.get("adapter_sha256") != adapter_sha256
        or evaluation.get("completed_locked_external_evaluations") != 1
    ):
        raise ValueError("route-remediation training/evaluation identity differs")

    source = validate_frozen_source(config.source_manifest_path)
    source_fc_index = load_scale_canary_source_rows(source)
    base_fc_rows = [record["row"] for _, record in sorted(source_fc_index.items())]
    _validate_finite_tie_policy(base_fc_rows, "base forced-choice")

    result_paths = {
        "dev": output_dir / "results" / "dev" / "rows.jsonl",
        "forced_choice": output_dir / "results" / "forced_choice" / "rows.jsonl",
        "crd": output_dir / "results" / "crd" / "rows.jsonl",
    }
    metadata = {
        "dev": ("dev_result_path", "dev_result_sha256", "dev_decision_count", 96),
        "forced_choice": (
            "forced_choice_result_path",
            "forced_choice_result_sha256",
            "forced_choice_decision_count",
            1_152,
        ),
        "crd": ("crd_result_path", "crd_result_sha256", "crd_decision_count", 1_536),
    }
    loaded: dict[str, list[dict[str, Any]]] = {}
    for name, path in result_paths.items():
        path_field, hash_field, count_field, expected_count = metadata[name]
        if (
            Path(str(evaluation[path_field])).resolve() != path.resolve()
            or evaluation[hash_field] != file_hash(path)
            or int(evaluation[count_field]) != expected_count
        ):
            raise ValueError(f"route-remediation {name} result identity differs")
        rows = _load_jsonl(path)
        if len(rows) != expected_count:
            raise ValueError(f"route-remediation {name} result matrix is incomplete")
        loaded[name] = rows
    _validate_admissible_statuses(loaded["dev"], "route-remediation dev")
    _validate_finite_tie_policy(
        loaded["forced_choice"],
        "route-remediation forced-choice",
    )
    _validate_finite_tie_policy(loaded["crd"], "route-remediation CRD")

    dev_audit_records = _load_audit_record_index(
        output_dir / "preflight" / "dev_token_audit.json",
        expected_file_sha256=config.dev_token_audit_sha256,
        expected_schema="p0d2hrr-dev-token-audit-v1",
        expected_count=96,
        key_fields="example_id",
    )
    fc_audit_records = _load_audit_record_index(
        output_dir / "preflight" / "forced_choice_token_audit.json",
        expected_file_sha256=config.forced_choice_token_audit_sha256,
        expected_schema="p0d2hrr-fc-token-audit-v1",
        expected_count=1_152,
        key_fields=("lesson_id", "arm", "probe_id"),
    )
    crd_audit_records = _load_audit_record_index(
        output_dir / "preflight" / "crd_token_audit.json",
        expected_file_sha256=config.crd_token_audit_sha256,
        expected_schema="p0d2hrr-crd-token-audit-v1",
        expected_count=1_536,
        key_fields="probe_id",
    )
    dev_audits = {key: str(record["record_sha256"]) for key, record in dev_audit_records.items()}
    fc_audits = {key: str(record["record_sha256"]) for key, record in fc_audit_records.items()}
    crd_audits = {key: str(record["record_sha256"]) for key, record in crd_audit_records.items()}
    _validate_row_matrix(
        loaded["dev"],
        "p0d2hrr-dev-decision-row-v1",
        config,
        manifest,
        audit_hashes=dev_audits,
    )
    _validate_row_matrix(
        loaded["forced_choice"],
        "p0d2hrr-fc-decision-row-v1",
        config,
        manifest,
        audit_hashes=fc_audits,
        source_rows=source_fc_index,
    )
    _validate_row_matrix(
        loaded["crd"],
        "p0d2hrr-crd-decision-row-v1",
        config,
        manifest,
        audit_hashes=crd_audits,
        source_rows=source.source_rows,
    )
    for row in loaded["dev"]:
        key = str(row["example_id"])
        _require_result_matches_audit(
            row,
            dev_audit_records[key],
            f"route-remediation dev {key}",
        )
    for row in loaded["forced_choice"]:
        key = _forced_choice_key(row)
        _require_result_matches_audit(
            row,
            fc_audit_records[key],
            f"route-remediation forced-choice {key}",
        )
    for row in loaded["crd"]:
        key = str(row["probe_id"])
        _require_result_matches_audit(
            row,
            crd_audit_records[key],
            f"route-remediation CRD {key}",
        )
    adapter_crd_evidence = [row_evidence(row) for row in loaded["crd"]]
    if any(
        not bool(item["checks"]["all_structural_checks_passed"])
        or (
            item["error_status"] == "tie"
            and item["tie_origin"] != "distinct_candidates_exact_score_tie"
        )
        for item in adapter_crd_evidence
    ):
        raise ValueError("route-remediation CRD scoring structure failed recomputation")

    summary_path = output_dir / "results" / "aggregate" / "summary.json"
    decision_path = output_dir / "results" / "aggregate" / "next_stage_decision.json"
    summary = read_json_object(summary_path, "RR summary")
    decision = read_json_object(decision_path, "RR next-stage decision")
    summary_gate = summary.get("route_remediation_gate")
    summary_gates = summary.get("gates")
    if not isinstance(summary_gate, dict) or not isinstance(summary_gates, dict):
        raise ValueError("route-remediation aggregate gate metadata is incomplete")
    gate_status = summary_gate.get("status")
    if (
        Path(str(aggregate["summary_path"])).resolve() != summary_path.resolve()
        or aggregate["summary_sha256"] != file_hash(summary_path)
        or Path(str(aggregate["decision_path"])).resolve() != decision_path.resolve()
        or aggregate["decision_sha256"] != file_hash(decision_path)
        or summary.get("schema_version") != "p0d2hrr-summary-v1"
        or summary.get("run_id") != manifest.payload["run_id"]
        or summary.get("adapter_sha256") != adapter_sha256
        or summary.get("preregistration_sha256") != claimed_preregistration_hash
        or aggregate.get("status") != gate_status
        or decision.get("status") != gate_status
        or aggregate.get("training_complexity_review_eligible") is not False
        or summary_gate.get("training_complexity_review_eligible") is not False
        or summary_gate.get("automatic_training_started") is not False
        or summary_gate.get("automatic_narrow_scan_started") is not False
        or summary_gate.get("automatic_rlvr_started") is not False
        or summary_gates.get("run_valid") is not True
        or summary_gates.get("training_complexity_review_eligible") is not False
        or summary_gates.get("eligible_model_ids") != []
        or summary_gates.get("automatic_training_started") is not False
        or summary_gates.get("automatic_narrow_scan_started") is not False
        or summary_gates.get("automatic_rlvr_started") is not False
        or decision.get("schema_version") != "p0d2hrr-next-stage-v1"
        or decision.get("training_complexity_review_eligible") is not False
        or decision.get("eligible_model_ids") != []
        or decision.get("automatic_training_started") is not False
        or decision.get("automatic_narrow_scan_started") is not False
        or decision.get("automatic_rlvr_started") is not False
    ):
        raise ValueError("route-remediation aggregate provenance differs")

    rr_bank_path = output_dir / "preflight" / "crd_bank_audit.json"
    rr_token_path = output_dir / "preflight" / "crd_token_audit.json"
    rr_bank_audit = read_json_object(
        rr_bank_path,
        "RR CRD bank audit",
    )
    rr_token_audit = read_json_object(rr_token_path, "RR CRD token audit")
    if (
        file_hash(rr_bank_path) != config.crd_bank_audit_sha256
        or file_hash(rr_token_path) != config.crd_token_audit_sha256
        or rr_bank_audit.get("schema_version") != "p0d2hcrd-bank-audit-v1"
        or rr_bank_audit.get("decision_count") != 1_536
        or rr_bank_audit.get("candidate_count") != 5_376
        or rr_bank_audit.get("all_checks_passed") is not True
        or rr_token_audit.get("bank_audit_sha256") != config.crd_bank_audit_sha256
    ):
        raise ValueError("route-remediation CRD preflight provenance differs")
    rr_bank_records = _bank_record_index(rr_bank_audit, "route-remediation CRD")
    for row in loaded["crd"]:
        probe_id = str(row["probe_id"])
        bank_record = rr_bank_records.get(probe_id)
        token_record = crd_audit_records.get(probe_id)
        if bank_record is None or token_record is None:
            raise ValueError(f"route-remediation CRD row lacks bank anchor: {probe_id}")
        _require_crd_audit_matches_bank(
            token_record,
            bank_record,
            f"route-remediation CRD {probe_id}",
        )
        _require_crd_row_matches_bank(
            row,
            bank_record,
            f"route-remediation CRD {probe_id}",
        )
    provenance = {
        "output_dir": str(output_dir),
        "run_id": manifest.payload["run_id"],
        "manifest_sha256": file_hash(manifest_path),
        "preregistration_sha256": claimed_preregistration_hash,
        "preregistration_file_sha256": file_hash(preregistration_path),
        "authorization_sha256": file_hash(authorization_path),
        "experiment_code_revision": experiment_code_revision,
        "experiment_code_sha256": config.code_sha256,
        "adapter_sha256": adapter_sha256,
        "dev_result_sha256": evaluation["dev_result_sha256"],
        "forced_choice_result_sha256": evaluation["forced_choice_result_sha256"],
        "crd_result_sha256": evaluation["crd_result_sha256"],
        "summary_sha256": aggregate["summary_sha256"],
        "decision_sha256": aggregate["decision_sha256"],
        "state": manifest.state,
        "current_artifact_checks_passed": True,
    }
    context = {
        "source_run_id": config.source_run_id,
        "source_manifest_path": str(config.source_manifest_path.resolve()),
        "source_manifest_sha256": config.source_manifest_sha256,
        "source_summary_sha256": config.source_summary_sha256,
        "source_candidate_token_audit_sha256": (config.source_candidate_token_audit_sha256),
        "source_raw_results_sha256": config.source_raw_results_sha256,
        "source_evaluation_precision": config.source_evaluation_precision,
        "model_name": config.spec.model_name,
        "model_revision": config.spec.model_revision,
        "rr_bank_sha256": rr_bank_audit["bank_sha256"],
        "selected_lesson_ids": list(source.forced_choice_manifest["selected_lessons"]),
    }
    return (
        provenance,
        base_fc_rows,
        loaded["forced_choice"],
        loaded["crd"],
        context,
    )


def _validate_base_crd(
    output_dir: Path,
    rr_context: dict[str, Any],
    revision_lock: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "results" / "aggregate" / "summary.json"
    decision_path = output_dir / "results" / "aggregate" / "next_stage_decision.json"
    bank_path = output_dir / "preflight" / "decomposition_bank.json"
    token_audit_path = output_dir / "preflight" / "candidate_token_audit.json"
    manifest = read_json_object(manifest_path, "base CRD manifest")
    summary = read_json_object(summary_path, "base CRD summary")
    decision = read_json_object(decision_path, "base CRD next-stage decision")
    bank_audit = read_json_object(bank_path, "base CRD decomposition-bank audit")
    token_audit = read_json_object(token_audit_path, "base CRD candidate-token audit")
    rows, raw_results_sha256 = load_verified_rows(output_dir, manifest)
    config = manifest.get("config")
    model = manifest.get("model")
    if not isinstance(config, dict) or not isinstance(model, dict):
        raise ValueError("base CRD manifest identity is incomplete")
    bank_audit_sha256 = file_hash(bank_path)
    token_audit_sha256 = file_hash(token_audit_path)
    expected_run_id = (
        "p0d2hcrd-route-decomposition-"
        + json_hash(
            {
                "config": config,
                "bank_audit_sha256": bank_audit_sha256,
                "candidate_token_audit_sha256": token_audit_sha256,
            }
        )[:10]
    )
    if not revision_lock.is_file():
        raise FileNotFoundError(f"missing base CRD code revision lock: {revision_lock}")
    experiment_code_revision = revision_lock.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", experiment_code_revision):
        raise ValueError("base CRD code revision lock is not a full Git commit SHA")
    revision_code_sha256 = _git_revision_code_hash(
        Path(__file__).resolve().parents[3],
        experiment_code_revision,
    )
    source_manifest_path = Path(str(manifest.get("source_manifest_path"))).resolve()
    if (
        manifest.get("schema_version") != "p0d2hcrd-manifest-v1"
        or manifest.get("run_id") != expected_run_id
        or expected_run_id != EXPECTED_BASE_CRD_RUN_ID
        or manifest.get("errors") != []
        or manifest.get("bank_audit_sha256") != bank_audit_sha256
        or manifest.get("candidate_token_audit_sha256") != token_audit_sha256
        or source_manifest_path != Path(rr_context["source_manifest_path"])
        or config.get("schema_version") != "p0d2hcrd-config-v1"
        or config.get("stage") != "hard_probe_route_decomposition"
        or config.get("code_sha256") != revision_code_sha256
        or config.get("model") != model
        or model.get("model_id") != "scale_canary"
        or model.get("use_4bit") is not True
        or model.get("model_name") != rr_context["model_name"]
        or model.get("model_revision") != rr_context["model_revision"]
        or config.get("source_run_id") != rr_context["source_run_id"]
        or config.get("source_manifest_sha256") != rr_context["source_manifest_sha256"]
        or config.get("source_summary_sha256") != rr_context["source_summary_sha256"]
        or config.get("source_candidate_token_audit_sha256")
        != rr_context["source_candidate_token_audit_sha256"]
        or config.get("source_raw_results_sha256") != rr_context["source_raw_results_sha256"]
        or config.get("decomposition_bank_sha256") != rr_context["rr_bank_sha256"]
        or config.get("selected_lesson_ids") != rr_context["selected_lesson_ids"]
        or config.get("candidate_scoring_version") != "p0d2hcrd-variable-candidate-sum-logprob-v1"
        or config.get("training_complexity_review_eligible") is not False
        or config.get("automatic_training_started") is not False
        or config.get("automatic_narrow_scan_started") is not False
        or manifest.get("selected_lessons") != rr_context["selected_lesson_ids"]
    ):
        raise ValueError("base CRD is not the source-compatible frozen NF4 run")

    token_records = token_audit.get("records")
    token_records_valid = isinstance(token_records, list) and len(token_records) == 1_536
    seen_probe_ids: set[str] = set()
    token_record_index: dict[str, dict[str, Any]] = {}
    if token_records_valid:
        assert isinstance(token_records, list)
        for record in token_records:
            if not isinstance(record, dict):
                token_records_valid = False
                continue
            probe_id = str(record.get("probe_id"))
            without_hash = {key: value for key, value in record.items() if key != "record_sha256"}
            if (
                not probe_id
                or probe_id in seen_probe_ids
                or record.get("record_sha256") != json_hash(without_hash)
                or record.get("source_matches") is not True
                or record.get("all_candidates_valid") is not True
                or len(record.get("candidates", [])) not in {2, 4}
            ):
                token_records_valid = False
            seen_probe_ids.add(probe_id)
            token_record_index[probe_id] = record
    if (
        bank_audit.get("schema_version") != "p0d2hcrd-bank-audit-v1"
        or bank_audit.get("bank_sha256") != config.get("decomposition_bank_sha256")
        or bank_audit.get("decision_count") != 1_536
        or bank_audit.get("candidate_count") != 5_376
        or bank_audit.get("all_checks_passed") is not True
        or token_audit.get("schema_version") != "p0d2hcrd-candidate-token-audit-v1"
        or token_audit.get("candidate_scoring_version") != config.get("candidate_scoring_version")
        or token_audit.get("config_identity_sha256") != json_hash(config)
        or token_audit.get("bank_audit_sha256") != bank_audit_sha256
        or token_audit.get("source_manifest_sha256") != config.get("source_manifest_sha256")
        or token_audit.get("source_raw_results_sha256") != config.get("source_raw_results_sha256")
        or token_audit.get("decision_count") != 1_536
        or token_audit.get("candidate_count") != 5_376
        or token_audit.get("failed_decision_count") != 0
        or token_audit.get("input_truncated_count") != 0
        or token_audit.get("source_mismatch_count") != 0
        or token_audit.get("all_checks_passed") is not True
        or not token_records_valid
    ):
        raise ValueError("base CRD preflight provenance is incomplete or changed")
    base_bank_records = _bank_record_index(bank_audit, "base CRD")

    summary_gates = summary.get("gates")
    diagnostic_gate = summary.get("diagnostic_gate")
    if not isinstance(summary_gates, dict) or not isinstance(diagnostic_gate, dict):
        raise ValueError("base CRD aggregate gate metadata is incomplete")
    if (
        summary.get("schema_version") != "p0d2hcrd-summary-v1"
        or summary.get("run_id") != expected_run_id
        or summary.get("model") != model
        or summary.get("source_p0d2hfc_run_id") != config.get("source_run_id")
        or summary.get("selected_lesson_count") != 24
        or summary.get("unit_count") != 24
        or summary.get("decision_row_count") != 1_536
        or summary.get("candidate_sequence_count") != 5_376
        or summary.get("source_manifest_sha256") != config.get("source_manifest_sha256")
        or summary.get("source_summary_sha256") != config.get("source_summary_sha256")
        or summary.get("source_raw_results_sha256") != config.get("source_raw_results_sha256")
        or summary.get("decomposition_bank_sha256") != config.get("decomposition_bank_sha256")
        or summary.get("decomposition_bank_audit")
        != {key: value for key, value in bank_audit.items() if key != "records"}
        or summary.get("candidate_token_audit")
        != {key: value for key, value in token_audit.items() if key != "records"}
        or summary_gates.get("run_valid") is not True
        or summary_gates.get("diagnostic_status") != diagnostic_gate.get("status")
        or summary_gates.get("training_complexity_review_eligible") is not False
        or summary_gates.get("automatic_training_started") is not False
        or summary_gates.get("automatic_narrow_scan_started") is not False
        or decision.get("schema_version") != "p0d2hcrd-next-stage-v1"
        or decision.get("source_run_id") != expected_run_id
        or decision.get("status") != diagnostic_gate.get("status")
        or decision.get("diagnostic_only") is not True
        or decision.get("training_complexity_review_eligible") is not False
        or decision.get("automatic_training_started") is not False
        or decision.get("automatic_narrow_scan_started") is not False
    ):
        raise ValueError("base CRD aggregate summary is incomplete")
    precisions = {str(row.get("precision")) for row in rows}
    if precisions != {rr_context["source_evaluation_precision"]}:
        raise ValueError("base and adapter CRD precision policies differ")
    _validate_finite_tie_policy(rows, "base CRD")
    for row in rows:
        probe_id = str(row["probe_id"])
        audit = token_record_index.get(probe_id)
        bank_record = base_bank_records.get(probe_id)
        if audit is None:
            raise ValueError(f"base CRD row lacks token-audit anchor: {probe_id}")
        if bank_record is None:
            raise ValueError(f"base CRD row lacks bank anchor: {probe_id}")
        _require_result_matches_audit(
            row,
            audit,
            f"base CRD {probe_id}",
        )
        _require_crd_audit_matches_bank(
            audit,
            bank_record,
            f"base CRD {probe_id}",
        )
        _require_crd_row_matches_bank(
            row,
            bank_record,
            f"base CRD {probe_id}",
        )
    evidence = [row_evidence(row) for row in rows]
    if any(
        not bool(item["checks"]["all_structural_checks_passed"])
        or (
            item["error_status"] == "tie"
            and item["tie_origin"] != "distinct_candidates_exact_score_tie"
        )
        for item in evidence
    ):
        raise ValueError("base CRD row scoring structure failed recomputation")
    provenance = {
        "output_dir": str(output_dir),
        "run_id": manifest["run_id"],
        "manifest_sha256": file_hash(manifest_path),
        "summary_sha256": file_hash(summary_path),
        "decision_sha256": file_hash(decision_path),
        "raw_results_sha256": raw_results_sha256,
        "experiment_code_sha256": config["code_sha256"],
        "experiment_code_revision": experiment_code_revision,
        "code_revision_lock_sha256": file_hash(revision_lock),
        "bank_audit_sha256": bank_audit_sha256,
        "candidate_token_audit_sha256": token_audit_sha256,
        "decomposition_bank_sha256": config["decomposition_bank_sha256"],
        "precision": next(iter(precisions)),
        "model_name": model["model_name"],
        "model_revision": model["model_revision"],
        "use_4bit": model["use_4bit"],
        "current_artifact_checks_passed": True,
    }
    return provenance, rows


def _failure_category(
    combined: dict[str, Any],
    *,
    route_all_correct: bool,
    route_all_wrong: bool,
    route_ambiguous: bool,
    retrieval_correct: bool,
) -> str:
    if bool(combined.get("correct")):
        return "combined_correct"
    status = str(combined.get("error_status"))
    if status == "tie":
        return "exact_tie"
    if status != "ok":
        return "scoring_error"
    if route_ambiguous:
        return "ambiguous_route"
    if route_all_correct and retrieval_correct:
        return "composition_inconsistent"
    if route_all_wrong and retrieval_correct:
        return "route_associated"
    if route_all_correct and not retrieval_correct:
        return "retrieval_associated"
    if route_all_wrong and not retrieval_correct:
        return "multiple_component_associated"
    return "ambiguous"


def _cluster_bootstrap(
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float | None],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    observed: list[float] = []
    for row in rows:
        item = value(row)
        if item is None or not math.isfinite(float(item)):
            continue
        number = float(item)
        grouped[str(row["lesson_id"])].append(number)
        observed.append(number)
    if not observed:
        return {
            "estimate": None,
            "ci95": None,
            "cluster_count": 0,
            "observation_count": 0,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        }
    labels = sorted(grouped)
    cluster_stats = {label: (sum(grouped[label]), len(grouped[label])) for label in labels}
    generator = random.Random(bootstrap_seed)
    estimates: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = [labels[generator.randrange(len(labels))] for _ in labels]
        total = sum(cluster_stats[label][0] for label in sampled)
        count = sum(cluster_stats[label][1] for label in sampled)
        estimates.append(total / count)
    return {
        "estimate": mean(observed),
        "ci95": [_percentile(estimates, 0.025), _percentile(estimates, 0.975)],
        "cluster_count": len(labels),
        "observation_count": len(observed),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }


def _clustered_binary_rate_difference(
    rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    by_lesson: dict[str, dict[bool, list[float]]] = defaultdict(lambda: {False: [], True: []})
    for row in rows:
        by_lesson[str(row["lesson_id"])][bool(row["route_failed"])].append(
            float(bool(row["combined_error"]))
        )
    lesson_differences = [
        {
            "lesson_id": lesson_id,
            "difference": mean(groups[True]) - mean(groups[False]),
        }
        for lesson_id, groups in sorted(by_lesson.items())
        if groups[False] and groups[True]
    ]
    difference = _cluster_bootstrap(
        lesson_differences,
        lambda row: float(row["difference"]),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "contrast": ("P(combined_error|route_all_wrong) - P(combined_error|route_all_correct)"),
        "eligible_lesson_count": len(lesson_differences),
        "lesson_equal_weight_difference": difference,
    }


def _grouped_paired_summaries(
    pairs: list[dict[str, Any]],
    fields: tuple[str, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[tuple(_factor_label(row.get(field)) for field in fields)].append(row)
    return {
        "::".join(key): {
            "factors": dict(zip(fields, key, strict=True)),
            "summary": paired_summary(
                values,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=_derived_seed(bootstrap_seed, "::".join(key)),
            ),
        }
        for key, values in sorted(grouped.items())
    }


def _factor_effects(
    pairs: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[_factor_label(row.get(field))].append(row)
    result: dict[str, dict[str, Any]] = {}
    for label, values in sorted(grouped.items()):
        base_correct = sum(bool(row["base_correct"]) for row in values)
        adapter_correct = sum(bool(row["adapter_correct"]) for row in values)
        deltas = [
            float(item)
            for row in values
            if (item := row.get("signed_expected_margin_delta")) is not None
        ]
        result[label] = {
            "decision_count": len(values),
            "base_accuracy": base_correct / len(values),
            "adapter_accuracy": adapter_correct / len(values),
            "adapter_minus_base_accuracy": (adapter_correct - base_correct) / len(values),
            "mean_signed_expected_margin_delta": mean(deltas) if deltas else None,
        }
    return result


def _confusion(
    pairs: list[dict[str, Any]],
    model: str,
) -> dict[str, dict[str, int]]:
    field = f"{model}_predicted_candidate"
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in pairs:
        expected = str(row["expected_candidate"])
        predicted = str(row[field]) if row.get(field) is not None else "<none>"
        result[expected][predicted] += 1
    return {expected: dict(sorted(counts.items())) for expected, counts in sorted(result.items())}


def _prediction_position_summary(
    pairs: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    missing = 0
    candidate_count: int | None = None
    for row in pairs:
        ordered = row.get("ordered_candidates")
        if not isinstance(ordered, list) or len(ordered) not in {2, 4}:
            raise ValueError("predicted-position summary requires a frozen candidate order")
        if candidate_count is None:
            candidate_count = len(ordered)
        elif candidate_count != len(ordered):
            raise ValueError("predicted-position summary mixes candidate-panel sizes")
        predicted = row.get(f"{model}_predicted_candidate")
        if predicted is None:
            missing += 1
            continue
        try:
            position = ordered.index(predicted) + 1
        except ValueError as error:
            raise ValueError("predicted candidate is absent from its panel") from error
        counts[str(position)] += 1
    valid = sum(counts.values())
    rates = {position: count / valid for position, count in sorted(counts.items())} if valid else {}
    entropy = -sum(rate * math.log2(rate) for rate in rates.values()) if rates else None
    return {
        "decision_count": len(pairs),
        "valid_prediction_count": valid,
        "tie_or_missing_prediction_count": missing,
        "position_counts": dict(sorted(counts.items())),
        "position_rates_among_valid": rates,
        "entropy_bits_among_valid": entropy,
        "normalized_entropy_among_valid": (
            entropy / math.log2(candidate_count)
            if entropy is not None and candidate_count is not None
            else None
        ),
    }


def _tie_bounds(pairs: list[dict[str, Any]], model: str) -> dict[str, Any]:
    fatal_errors = sum(
        str(row.get(f"{model}_error_status", "ok")) not in {"ok", "tie"} for row in pairs
    )
    if fatal_errors:
        raise ValueError(
            f"{model} tie bounds are invalid with {fatal_errors} non-tie scoring errors"
        )
    correct = sum(bool(row[f"{model}_correct"]) for row in pairs)
    ties = sum(bool(row[f"{model}_tie"]) for row in pairs)
    compatible = sum(bool(row[f"{model}_tie_expected_compatible"]) for row in pairs)
    return {
        "decision_count": len(pairs),
        "bounds_valid": True,
        "fatal_error_count": 0,
        "official_correct_count": correct,
        "tie_count": ties,
        "expected_compatible_tie_count": compatible,
        "all_ties_incorrect_accuracy": correct / len(pairs),
        "all_ties_compatible_accuracy": (correct + compatible) / len(pairs),
    }


def _holm_adjust(p_values: dict[str, float]) -> dict[str, dict[str, float]]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - index) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return {
        name: {
            "raw_p": float(p_values[name]),
            "holm_adjusted_p": adjusted[name],
        }
        for name in sorted(p_values)
    }


def _validate_finite_tie_policy(
    rows: list[dict[str, Any]],
    label: str,
) -> None:
    """Reject every scoring anomaly except a finite, structurally exact top tie."""
    _validate_admissible_statuses(rows, label)
    for index, row in enumerate(rows):
        row_label = f"{label} row {index}"
        status = str(row.get("error_status"))
        candidates = _validate_candidate_list(row, row_label)
        if len(candidates) not in {2, 4}:
            raise ValueError(f"{row_label} candidate count is not 2 or 4")

        names = [_candidate_name(candidate) for candidate in candidates]
        continuations = [candidate.get("continuation") for candidate in candidates]
        token_sequences: list[tuple[int, ...]] = []
        for candidate in candidates:
            token_ids = candidate.get("token_ids")
            token_logprobs = candidate.get("token_logprobs")
            token_count = candidate.get("token_count")
            if (
                not isinstance(token_ids, list)
                or not token_ids
                or not all(
                    isinstance(value, int) and not isinstance(value, bool) for value in token_ids
                )
                or not isinstance(token_logprobs, list)
                or not isinstance(token_count, int)
                or isinstance(token_count, bool)
                or token_count <= 0
                or token_count != len(token_ids)
                or token_count != len(token_logprobs)
                or not all(_finite(value) for value in token_logprobs)
                or not all(
                    _finite(candidate.get(field)) for field in ("sum_logprob", "mean_logprob")
                )
                or candidate.get("score_finite") is not True
                or not math.isclose(
                    float(candidate["sum_logprob"]),
                    sum(float(value) for value in token_logprobs),
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
                or not math.isclose(
                    float(candidate["mean_logprob"]),
                    float(candidate["sum_logprob"]) / token_count,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
            ):
                raise ValueError(f"{row_label} has invalid finite candidate-score structure")
            token_sequences.append(tuple(token_ids))

        if (
            len(names) != len(set(names))
            or any(not isinstance(value, str) or not value for value in continuations)
            or len(continuations) != len(set(continuations))
            or len(token_sequences) != len(set(token_sequences))
        ):
            raise ValueError(f"{row_label} candidate identities are not distinct")

        ordered = row.get("ordered_candidates", row.get("ordered_allowed_actions"))
        if not isinstance(ordered, list) or names != ordered:
            raise ValueError(f"{row_label} candidate order differs from the frozen row")
        expected = row.get("expected_candidate", row.get("expected_action"))
        if expected not in names:
            raise ValueError(f"{row_label} expected candidate is absent")

        scores = [float(candidate["sum_logprob"]) for candidate in candidates]
        maximum = max(scores)
        top_count = sum(value == maximum for value in scores)
        stored_tie = bool(row.get("tie"))
        if status == "tie":
            if not stored_tie or top_count < 2:
                raise ValueError(
                    f"{row_label} is not a distinct-candidate exact finite top-score tie"
                )
        elif stored_tie or top_count != 1:
            raise ValueError(f"{row_label} ok status conflicts with exact finite scores")


def _validate_admissible_statuses(
    rows: list[dict[str, Any]],
    label: str,
) -> None:
    for index, row in enumerate(rows):
        status = str(row.get("error_status"))
        if status not in {"ok", "tie"} or bool(row.get("non_finite")):
            raise ValueError(
                f"{label} row {index} has unsupported scoring status {status!r}; "
                "only finite ok/tie rows are admissible"
            )


def _load_audit_record_index(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_schema: str,
    expected_count: int,
    key_fields: str | tuple[str, ...],
) -> dict[object, dict[str, Any]]:
    report = read_json_object(path, "candidate token audit")
    records = report.get("records")
    if (
        file_hash(path) != expected_file_sha256
        or report.get("schema_version") != expected_schema
        or report.get("decision_count") != expected_count
        or report.get("failed_decision_count") != 0
        or report.get("input_truncated_count") != 0
        or report.get("all_checks_passed") is not True
        or not isinstance(records, list)
        or len(records) != expected_count
        or report.get("candidate_count")
        != sum(len(record.get("candidates", [])) for record in records if isinstance(record, dict))
    ):
        raise ValueError(f"candidate token audit is incomplete or changed: {path}")
    fields = (key_fields,) if isinstance(key_fields, str) else key_fields
    result: dict[object, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"candidate token audit record is invalid: {path}")
        values = tuple(str(record.get(field)) for field in fields)
        key: object = values[0] if len(values) == 1 else values
        without_hash = {field: value for field, value in record.items() if field != "record_sha256"}
        if (
            key in result
            or record.get("record_sha256") != json_hash(without_hash)
            or record.get("all_candidates_valid") is not True
            or ("source_matches" in record and record.get("source_matches") is not True)
        ):
            raise ValueError(f"candidate token audit record changed: {key}")
        result[key] = record
    return result


def _require_result_matches_audit(
    row: dict[str, Any],
    audit: dict[str, Any],
    label: str,
) -> None:
    if (
        row.get("candidate_audit_record_sha256") != audit.get("record_sha256")
        or row.get("prompt_sha256") != audit.get("prompt_sha256")
        or (
            audit.get("formatted_prompt_sha256") is not None
            and row.get("prompt_sha256") != audit.get("formatted_prompt_sha256")
        )
        or _candidate_static(row, f"{label} result") != _candidate_static(audit, f"{label} audit")
    ):
        raise ValueError(f"{label} differs from its preregistered token audit")
    fields = (
        "lesson_id",
        "arm",
        "probe_id",
        "category",
        "expected_action",
        "ordered_allowed_actions",
        "prompt_variant",
        "prompt_rendering_version",
        "example_id",
        "group_id",
        "task",
        "expected_candidate",
        "ordered_candidates",
        "endpoint",
        "source_row_key",
        "source_row_sha256",
        "prompt_rendering_version",
        "bank_record_sha256",
    )
    differences = [
        field for field in fields if field in row and field in audit and row[field] != audit[field]
    ]
    if differences:
        raise ValueError(f"{label} audit-anchored fields differ: {differences}")


def _bank_record_index(
    bank_audit: dict[str, Any],
    label: str,
) -> dict[str, dict[str, Any]]:
    records = bank_audit.get("records")
    if (
        not isinstance(records, list)
        or len(records) != 1_536
        or bank_audit.get("bank_sha256") != json_hash(records)
    ):
        raise ValueError(f"{label} bank records are incomplete or changed")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{label} bank record is invalid")
        probe_id = str(record.get("probe_id"))
        if not probe_id or probe_id in result:
            raise ValueError(f"{label} bank probe IDs are invalid")
        result[probe_id] = record
    return result


def _require_crd_row_matches_bank(
    row: dict[str, Any],
    bank_record: dict[str, Any],
    label: str,
) -> None:
    fields = (
        "probe_id",
        "lesson_id",
        "pair_id",
        "lesson_type",
        "lesson_side",
        "endpoint",
        "expected_candidate",
        "ordered_candidates",
        "route_variant",
        "target_slot",
        "current_marker",
        "slot_candidate_order",
        "slot_content_order",
        "action_panel_position",
        "source_probe_id",
        "source_row_key",
    )
    differences = [field for field in fields if row.get(field) != bank_record.get(field)]
    expected_bank_record_sha256 = json_hash(bank_record)
    if differences or (
        "bank_record_sha256" in row and row.get("bank_record_sha256") != expected_bank_record_sha256
    ):
        raise ValueError(f"{label} differs from its locked CRD bank record")


def _require_crd_audit_matches_bank(
    audit: dict[str, Any],
    bank_record: dict[str, Any],
    label: str,
) -> None:
    common_fields = (
        "probe_id",
        "lesson_id",
        "endpoint",
        "expected_candidate",
        "ordered_candidates",
        "source_row_key",
    )
    differences = [field for field in common_fields if audit.get(field) != bank_record.get(field)]
    expected_bank_record_sha256 = json_hash(bank_record)
    if (
        differences
        or ("prompt" in audit and audit.get("prompt") != bank_record.get("prompt"))
        or (
            "bank_record_sha256" in audit
            and audit.get("bank_record_sha256") != expected_bank_record_sha256
        )
    ):
        raise ValueError(f"{label} token audit differs from its locked CRD bank record")


def _validate_candidate_list(row: dict[str, Any], label: str) -> list[dict[str, Any]]:
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not all(
        isinstance(candidate, dict) for candidate in candidates
    ):
        raise ValueError(f"{label} candidates are missing")
    return candidates


def _candidate_static(row: dict[str, Any], label: str) -> list[dict[str, Any]]:
    return [
        {
            "value": _candidate_name(candidate),
            "continuation": candidate.get("continuation"),
            "token_ids": candidate.get("token_ids"),
            "token_count": candidate.get("token_count"),
        }
        for candidate in _validate_candidate_list(row, label)
    ]


def _require_candidate_static_match(
    base: dict[str, Any],
    adapter: dict[str, Any],
    label: str,
) -> None:
    if _candidate_static(base, f"{label} base") != _candidate_static(
        adapter,
        f"{label} adapter",
    ):
        raise ValueError(f"{label} candidate token identities differ")


def _candidate_score_map(row: dict[str, Any]) -> dict[str, float | None]:
    return {
        _candidate_name(candidate): (
            float(candidate["sum_logprob"]) if _finite(candidate.get("sum_logprob")) else None
        )
        for candidate in _validate_candidate_list(row, "score map")
    }


def _expected_and_best_distractor_scores(
    row: dict[str, Any],
    expected_field: str,
) -> tuple[float, float]:
    expected = row.get(expected_field)
    scores = _candidate_score_map(row)
    if expected not in scores or scores[expected] is None:
        raise ValueError("expected candidate lacks a finite score")
    distractors = [
        float(score)
        for candidate, score in scores.items()
        if candidate != expected and score is not None
    ]
    if not distractors:
        raise ValueError("expected candidate lacks a finite distractor")
    return float(scores[expected]), max(distractors)


def _expected_in_top_tie(
    row: dict[str, Any],
    expected_field: str,
) -> bool:
    if not bool(row.get("tie")):
        return False
    expected = row.get(expected_field)
    candidates = _validate_candidate_list(row, "tie compatibility")
    finite = [candidate for candidate in candidates if _finite(candidate.get("sum_logprob"))]
    if len(finite) != len(candidates) or expected is None:
        return False
    top_score = max(float(candidate["sum_logprob"]) for candidate in finite)
    return any(
        _candidate_name(candidate) == expected and float(candidate["sum_logprob"]) == top_score
        for candidate in finite
    )


def _candidate_name(candidate: dict[str, Any]) -> str:
    values = [
        value for field in ("candidate", "action") if (value := candidate.get(field)) is not None
    ]
    if len(values) != 1 or not isinstance(values[0], str) or not values[0]:
        raise ValueError("candidate score requires exactly one candidate/action label")
    return values[0]


def _transition(
    base: dict[str, Any],
    adapter: dict[str, Any],
) -> str:
    return (
        ("correct" if bool(base.get("correct")) else "wrong")
        + "_to_"
        + ("correct" if bool(adapter.get("correct")) else "wrong")
    )


def _forced_choice_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["lesson_id"]),
        str(row["arm"]),
        str(row["probe_id"]),
    )


def _unique_index(
    rows: Iterable[dict[str, Any]],
    key: Callable[[dict[str, Any]], Any],
    label: str,
) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for row in rows:
        value = key(row)
        if value in result:
            raise ValueError(f"duplicate {label} key: {value}")
        result[value] = row
    return result


def _require_equal_fields(
    left: dict[str, Any],
    right: dict[str, Any],
    fields: tuple[str, ...],
    label: str,
) -> None:
    differences = [field for field in fields if left.get(field) != right.get(field)]
    if differences:
        raise ValueError(f"{label} differs on fields: {differences}")


def _source_snapshot(
    rr_output: Path,
    base_crd_output: Path,
    source_fc_output: Path,
    revision_lock: Path,
    base_revision_lock: Path,
) -> dict[str, Any]:
    return {
        "route_remediation": {
            "manifest": file_hash(rr_output / "manifest.json"),
            "preregistration": file_hash(rr_output / "preregistration.json"),
            "authorization": file_hash(rr_output / "authorization.json"),
            "preflight": _directory_identity(rr_output / "preflight"),
            "adapter": _directory_identity(rr_output / "adapter"),
            "results": _directory_identity(rr_output / "results"),
            "code_revision_lock": file_hash(revision_lock),
        },
        "base_crd": {
            "manifest": file_hash(base_crd_output / "manifest.json"),
            "preflight": _directory_identity(base_crd_output / "preflight"),
            "results": _directory_identity(base_crd_output / "results"),
            "code_revision_lock": file_hash(base_revision_lock),
        },
        "frozen_forced_choice": {
            "manifest": file_hash(source_fc_output / "manifest.json"),
            "preflight": _directory_identity(source_fc_output / "preflight"),
            "results": _directory_identity(source_fc_output / "results"),
        },
    }


def _default_base_crd_revision_lock(output_dir: Path) -> Path:
    parents = output_dir.parents
    if len(parents) < 3 or parents[1].name != "runs":
        raise ValueError(
            "cannot infer base CRD code revision lock from a nonstandard layout; "
            "pass --base-crd-code-revision-lock"
        )
    return parents[2] / "code_revision.txt"


def _directory_identity(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"missing source directory: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"source directory is empty: {root}")
    digest = sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {"file_count": len(files), "sha256": digest.hexdigest()}


def _git_revision_code_hash(repo_root: Path, revision: str) -> str:
    listed = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "ls-tree",
            "-r",
            "--name-only",
            revision,
            "--",
            "src/plasticity_placement",
            "pyproject.toml",
            "uv.lock",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    python_paths = sorted(
        path
        for path in listed
        if path.startswith("src/plasticity_placement/") and path.endswith(".py")
    )
    paths = [
        *python_paths,
        *(path for path in ("pyproject.toml", "uv.lock") if path in listed),
    ]
    if not paths:
        raise ValueError(f"code revision has no project files: {revision}")
    digest = sha256()
    for path in paths:
        payload = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{revision}:{path}"],
            check=True,
            capture_output=True,
        ).stdout
        digest.update(path.encode())
        digest.update(payload)
    return digest.hexdigest()


def _require_independent_output(
    sources: tuple[Path, ...],
    analysis_output: Path,
) -> None:
    for left, right in combinations(sources, 2):
        if left == right or left.is_relative_to(right) or right.is_relative_to(left):
            raise ValueError("paired audit sources must be mutually independent")
    for source in sources:
        if (
            analysis_output == source
            or analysis_output.is_relative_to(source)
            or source.is_relative_to(analysis_output)
        ):
            raise ValueError("paired audit output must be independent from every source")


def _derived_seed(seed: int, label: str) -> int:
    digest = sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _historical_preregistration_observation(
    historical: str | None,
    current: str,
) -> dict[str, Any]:
    if historical is None:
        return {
            "reported_sha256": None,
            "current_disk_sha256": current,
            "matches_current_disk": None,
            "status": "not_supplied",
            "affects_current_artifact_integrity": False,
        }
    matches = historical == current
    return {
        "reported_sha256": historical,
        "current_disk_sha256": current,
        "matches_current_disk": matches,
        "status": (
            "matches_current_disk"
            if matches
            else "historical_observation_differs_from_current_disk"
        ),
        "affects_current_artifact_integrity": False,
        "blocks_paper_use_pending_reconciliation": not matches,
        "interpretation": (
            "A differing historical console value is retained as a provenance "
            "observation. It is not treated as a current file hash and must not "
            "be merged with the verified run without separate archived evidence."
            if not matches
            else "The supplied historical observation matches the current artifact."
        ),
    }


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty list")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _optional_difference(
    left: float | None,
    right: float | None,
) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _optional_float(value: object) -> float | None:
    return float(value) if _finite(value) else None


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _factor_label(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _boolean_rate(
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], bool],
) -> float | None:
    if not rows:
        return None
    return sum(bool(value(row)) for row in rows) / len(rows)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()


def _render_report(summary: dict[str, Any]) -> str:
    analysis = summary["analysis"]
    fc = analysis["forced_choice"]["external_conditional_route"]
    crd = analysis["crd"]["by_endpoint"]
    base_components = analysis["component_cells"]["base"]
    adapter_components = analysis["component_cells"]["adapter"]
    component_differences = analysis["component_cells"]["base_to_adapter_failure_count_differences"]
    composition = analysis["component_cells"]["base_to_adapter_composition_margin_interaction"]
    retrieval_shifts = analysis["component_cells"]["retrieval_to_combined_margin_shift"]
    composition_difference = composition["lesson_clustered_difference"]
    positions = analysis["crd"]["predicted_candidate_position"]["combined"]
    historical = summary["historical_preregistration_observation"]
    rr_source = summary["source"]["route_remediation"]
    base_source = summary["source"]["base_crd"]
    lines = [
        "# P0 route-remediation paired audit",
        "",
        f"- Audit run: `{summary['run_id']}`",
        f"- Route-remediation source: `{rr_source['run_id']}`",
        f"- Current preregistration SHA-256: `{rr_source['preregistration_sha256']}`",
        f"- NF4 base CRD source: `{base_source['run_id']}`",
        "- Mode: CPU-only, read-only post-hoc analysis",
        "- Current artifact graph checks passed: `true`",
        "- Historical attempt uniqueness verified: `false`",
        ("- Historical preregistration observation: `" + str(historical["status"]) + "`"),
        (
            "- Historical SHA mismatch blocks paper use pending reconciliation: `"
            + str(historical.get("blocks_paper_use_pending_reconciliation", False)).lower()
            + "`"
        ),
        ("- Paper evidence ready: `" + str(summary["paper_evidence_ready"]).lower() + "`"),
        "- Source artifacts modified: `false`",
        "- Frozen gate changed: `false`",
        "- Training authorized: `false`",
        "",
        "## Base-to-adapter paired differences",
        "",
        (
            "| Endpoint | N | Base | Adapter | Delta (95% lesson-cluster CI) | "
            "C→W | W→C | Base ties / interval | Adapter ties / interval |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        _difference_row("External conditional route", fc),
    ]
    for endpoint in ENDPOINTS:
        lines.append(_difference_row(f"CRD {endpoint}", crd[endpoint]))
    lines.extend(
        [
            "",
            "Each Delta CI above uses the lesson-cluster bootstrap; tie intervals "
            "are conservative sensitivity bounds. Exact McNemar/Holm values in "
            "JSON are exploratory row-level checks, and Holm does not remove "
            "within-lesson dependence.",
            "",
            "## Signed expected-candidate margins",
            "",
            "| Endpoint | Mean adapter-base margin | 95% lesson-cluster CI |",
            "|---|---:|---:|",
            _margin_row("External conditional route", fc),
            *[_margin_row(f"CRD {endpoint}", crd[endpoint]) for endpoint in ENDPOINTS],
            "",
            "## Combined-failure taxonomy",
            "",
            (
                "- Base combined failures: "
                f"{base_components['combined_failure_count']} / "
                f"{base_components['combined_decision_count']}; categories: `"
                + json.dumps(
                    base_components["failure_category_counts"],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "`"
            ),
            (
                "- Adapter combined failures: "
                f"{adapter_components['combined_failure_count']} / "
                f"{adapter_components['combined_decision_count']}; categories: `"
                + json.dumps(
                    adapter_components["failure_category_counts"],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "`"
            ),
            (
                "- Adapter-minus-base failure-count differences: `"
                + json.dumps(component_differences, ensure_ascii=False, sort_keys=True)
                + "`"
            ),
            "",
            "### Definite-route association",
            "",
            (
                "| Model | Route-all-correct N / error rate | Route-all-wrong N / "
                "error rate | Route ambiguous excluded | Combined non-ok excluded | "
                "Eligible lessons | Rate difference (95% CI) |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
            _association_row("Base", base_components),
            _association_row("Adapter", adapter_components),
            "",
            "Mixed/tied route cells and non-ok combined outcomes are excluded from "
            "this association; combined exact ties remain separately counted in the "
            "failure taxonomy.",
            "",
            "## Composition and panel-position diagnostics",
            "",
            _retrieval_shift_line("Base", retrieval_shifts["base"]),
            _retrieval_shift_line("Adapter", retrieval_shifts["adapter"]),
            (
                "- Composition margin interaction "
                "[(adapter-base combined) - (adapter-base retrieval-only)]: "
                f"{_format_optional(composition['mean_interaction'])}; "
                f"95% lesson-cluster CI "
                f"{_format_interval(composition_difference['ci95'])}."
            ),
            (
                "- Combined predicted panel positions (base): `"
                + json.dumps(
                    positions["base"]["position_counts"],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "`; entropy="
                + _format_optional(positions["base"]["entropy_bits_among_valid"])
                + " bits; valid/missing="
                + str(positions["base"]["valid_prediction_count"])
                + "/"
                + str(positions["base"]["tie_or_missing_prediction_count"])
                + "."
            ),
            (
                "- Combined predicted panel positions (adapter): `"
                + json.dumps(
                    positions["adapter"]["position_counts"],
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "`; entropy="
                + _format_optional(positions["adapter"]["entropy_bits_among_valid"])
                + " bits; valid/missing="
                + str(positions["adapter"]["valid_prediction_count"])
                + "/"
                + str(positions["adapter"]["tie_or_missing_prediction_count"])
                + "."
            ),
            "",
            "## Interpretation boundary",
            "",
            (
                "The component labels are associations over exactly matched probes. "
                "A wrong combined action does not uniquely identify a wrong routed "
                "slot. Base and adapter scores came from separate historical runs, "
                "so runtime equivalence is unverified and these paired differences "
                "are not causal adapter effects. The frozen gate remains failed and "
                "no training is authorized."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _difference_row(label: str, values: dict[str, Any]) -> str:
    interval = values["paired_accuracy_difference"]["ci95"]
    transitions = values["transition_counts"]
    base_ties = values["tie_sensitivity"]["base"]
    adapter_ties = values["tie_sensitivity"]["adapter"]
    return (
        f"| {label} | {values['decision_count']} | "
        f"{values['base_accuracy']:.4f} | "
        f"{values['adapter_accuracy']:.4f} | "
        f"{values['adapter_minus_base_accuracy']:+.4f} "
        f"({_format_interval(interval)}) | "
        f"{transitions['correct_to_wrong']} | "
        f"{transitions['wrong_to_correct']} | "
        f"{base_ties['tie_count']} / "
        f"[{base_ties['all_ties_incorrect_accuracy']:.4f}, "
        f"{base_ties['all_ties_compatible_accuracy']:.4f}] | "
        f"{adapter_ties['tie_count']} / "
        f"[{adapter_ties['all_ties_incorrect_accuracy']:.4f}, "
        f"{adapter_ties['all_ties_compatible_accuracy']:.4f}] |"
    )


def _margin_row(label: str, values: dict[str, Any]) -> str:
    margin = values["signed_expected_margin"]
    interval = margin["paired_difference"]["ci95"]
    return (
        f"| {label} | {_format_optional(margin['mean_adapter_minus_base'])} | "
        f"{_format_interval(interval)} |"
    )


def _association_row(label: str, values: dict[str, Any]) -> str:
    association = values["route_failure_association"]
    interval = association["lesson_equal_weight_difference"]["ci95"]
    return (
        f"| {label} | {values['route_association_all_correct_count']} / "
        f"{_format_optional(values['combined_error_given_route_all_correct'])} | "
        f"{values['route_association_all_wrong_count']} / "
        f"{_format_optional(values['combined_error_given_route_all_wrong'])} | "
        f"{values['route_association_excluded_ambiguous_count']} | "
        f"{values['route_association_excluded_combined_non_ok_count']} | "
        f"{association['eligible_lesson_count']} | "
        f"{_format_optional(association['lesson_equal_weight_difference']['estimate'])} "
        f"({_format_interval(interval)}) |"
    )


def _retrieval_shift_line(label: str, values: dict[str, Any]) -> str:
    expected = values["expected_action_logprob_shift"]
    distractor = values["best_distractor_logprob_shift"]
    return (
        f"- {label} combined-minus-retrieval shifts: expected action "
        f"{_format_optional(expected['mean_combined_minus_retrieval'])}; "
        f"best distractor "
        f"{_format_optional(distractor['mean_combined_minus_retrieval'])}; "
        f"signed margin "
        f"{_format_optional(values['mean_combined_minus_retrieval_signed_margin'])}."
    )


def _format_interval(value: object) -> str:
    if not isinstance(value, list) or len(value) != 2 or not all(_finite(item) for item in value):
        return "TBD"
    return f"[{float(value[0]):.4f}, {float(value[1]):.4f}]"


def _format_optional(value: object) -> str:
    return "TBD" if value is None else f"{float(value):.4f}"
