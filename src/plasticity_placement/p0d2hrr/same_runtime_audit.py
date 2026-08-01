from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.domain import Arm
from plasticity_placement.p0c.modeling import (
    ModelBundle,
    activate_adapter,
    load_base_model,
    release_model,
)
from plasticity_placement.p0c.runtime import (
    current_code_hash,
    current_environment_snapshot,
)
from plasticity_placement.p0d2h.probes import load_hard_probe_bank
from plasticity_placement.p0d2hc.prompting import render_calibration_probe
from plasticity_placement.p0d2hcrd.probes import compile_decomposition_bank
from plasticity_placement.p0d2hcrd.runtime import validate_frozen_source
from plasticity_placement.p0d2hcrd.scoring import (
    score_candidate_batch as score_crd_candidate_batch,
)
from plasticity_placement.p0d2hfc.scoring import (
    score_candidate_batch as score_fc_candidate_batch,
)
from plasticity_placement.p0d2hrr.io import (
    file_hash,
    immutable_write,
    json_hash,
    read_json_object,
)
from plasticity_placement.p0d2hrr.paired_audit import (
    ENDPOINTS,
    PAIRED_ROW_SCHEMA_VERSION,
    _json_bytes,
    _jsonl_bytes,
    _validate_route_remediation,
    build_crd_pairs,
    build_forced_choice_pairs,
    build_paired_analysis,
)
from plasticity_placement.p0d2hrr.preflight import (
    load_resolved_config,
    load_scale_canary_source_rows,
)
from plasticity_placement.p0d2hrr.runtime import (
    _adapter_hash,
    _AdapterLoadConfig,
    _load_crd_audits,
    _load_fc_audits,
    _require_source_compatible_bundle,
    _source_lesson_ids,
    _verify_crd_bank_audit,
)
from plasticity_placement.training.model_utils import chat_prompt

SCHEMA_VERSION = "p0d2hrr-same-runtime-audit-v1"
ROW_SCHEMA_VERSION = "p0d2hrr-same-runtime-decision-row-v1"
MANIFEST_SCHEMA_VERSION = "p0d2hrr-same-runtime-audit-manifest-v1"
DEFAULT_BOOTSTRAP_SEED = 20260801
DEFAULT_NONINFERIORITY_MARGIN = 0.02
DEFAULT_SENTINEL_TOLERANCE = 1e-5


@dataclass(frozen=True, slots=True)
class SameRuntimeAuditRequest:
    route_remediation_output: Path
    analysis_output: Path
    experiment_code_revision_lock: Path | None = None
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    combined_noninferiority_margin: float = DEFAULT_NONINFERIORITY_MARGIN
    sentinel_score_tolerance: float = DEFAULT_SENTINEL_TOLERANCE

    def __post_init__(self) -> None:
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        if self.bootstrap_seed < 0:
            raise ValueError("bootstrap_seed must be non-negative")
        if not 0.0 < self.combined_noninferiority_margin < 0.5:
            raise ValueError("combined_noninferiority_margin must be in (0, 0.5)")
        if not math.isfinite(self.sentinel_score_tolerance) or self.sentinel_score_tolerance < 0.0:
            raise ValueError("sentinel_score_tolerance must be finite and non-negative")


def run_same_runtime_audit(request: SameRuntimeAuditRequest) -> Path:
    """Evaluate adapter OFF and ON in one model object without training."""
    rr_output = request.route_remediation_output.resolve()
    analysis_output = request.analysis_output.resolve()
    revision_lock = (
        request.experiment_code_revision_lock.resolve()
        if request.experiment_code_revision_lock is not None
        else rr_output.parent.parent / "code_revision.txt"
    )
    _require_independent_new_output(rr_output, analysis_output)
    unvalidated_manifest = read_json_object(rr_output / "manifest.json", "RR manifest")
    source_manifest = Path(str(unvalidated_manifest["source_manifest_path"])).resolve()
    _require_independent_new_output(source_manifest.parent, analysis_output)
    before_snapshot = _source_snapshot(rr_output, source_manifest.parent, revision_lock)

    rr_provenance, _, _, _, rr_context = _validate_route_remediation(
        rr_output,
        revision_lock,
    )
    config = load_resolved_config(rr_output)
    source = validate_frozen_source(config.source_manifest_path)
    source_rows = load_scale_canary_source_rows(source)
    hard_bank = load_hard_probe_bank(source.hard_probe_dir)
    crd_bank, crd_bank_audit = compile_decomposition_bank(source.selected, hard_bank)
    stored_bank_audit = read_json_object(
        rr_output / "preflight" / "crd_bank_audit.json",
        "CRD bank audit",
    )
    _verify_crd_bank_audit(crd_bank_audit, stored_bank_audit)
    fc_audits = _load_fc_audits(rr_output, config)
    crd_audits = _load_crd_audits(rr_output, config)
    adapter_sha256 = _adapter_hash(rr_output / "adapter")
    if adapter_sha256 != rr_provenance["adapter_sha256"]:
        raise ValueError("route-remediation adapter changed after provenance validation")

    analysis_code_sha256 = current_code_hash()
    identity = {
        "schema_version": SCHEMA_VERSION,
        "analysis_code_sha256": analysis_code_sha256,
        "route_remediation_run_id": rr_provenance["run_id"],
        "route_remediation_manifest_sha256": rr_provenance["manifest_sha256"],
        "adapter_sha256": adapter_sha256,
        "bootstrap_samples": request.bootstrap_samples,
        "bootstrap_seed": request.bootstrap_seed,
        "combined_noninferiority_margin": request.combined_noninferiority_margin,
        "sentinel_score_tolerance": request.sentinel_score_tolerance,
        "matrix": {
            "forced_choice": "external_conditional_route_only",
            "crd": "complete_route_retrieval_combined",
            "pair_order": "adapter_off_then_adapter_on_per_prompt",
        },
    }
    run_id = "p0d2hrr-same-runtime-" + json_hash(identity)[:10]
    load_config = _AdapterLoadConfig(
        model_name=config.spec.model_name,
        model_revision=config.spec.model_revision,
        use_4bit=config.spec.training.use_4bit,
        training_seeds=(config.spec.training.seed,),
    )

    bundle: ModelBundle | None = None
    try:
        _progress("loading one base-model runtime")
        bundle = load_base_model(load_config)
        _require_source_compatible_bundle(bundle, config)
        base_model_identity = id(bundle.model)
        activate_adapter(
            bundle,
            rr_output / "adapter",
            adapter_name="route_remediation",
        )
        if not hasattr(bundle.model, "disable_adapter"):
            raise RuntimeError("installed PEFT version lacks disable_adapter()")
        bundle.model.requires_grad_(False)
        bundle.model.eval()
        peft_model_identity = id(bundle.model)
        wrapped_base_identity = id(bundle.model.get_base_model())
        if wrapped_base_identity != base_model_identity:
            raise RuntimeError("adapter activation replaced the loaded base-model object")

        first_item = _first_forced_choice_item(bundle, source, hard_bank, fc_audits)
        sentinel_before = _score_adapter_off(
            bundle,
            first_item["formatted_prompt"],
            first_item["audit"],
            score_fc_candidate_batch,
        )
        _progress("scoring 96 external conditional-route pairs")
        off_fc, on_fc = _evaluate_forced_choice_pairs(
            bundle=bundle,
            run_id=run_id,
            adapter_sha256=adapter_sha256,
            config=config,
            source=source,
            hard_bank=hard_bank,
            source_rows=source_rows,
            audits=fc_audits,
        )
        _progress("scoring 1,536 CRD pairs")
        off_crd, on_crd = _evaluate_crd_pairs(
            bundle=bundle,
            run_id=run_id,
            adapter_sha256=adapter_sha256,
            config=config,
            bank=crd_bank,
            source_rows=source.source_rows,
            audits=crd_audits,
        )
        sentinel_after = _score_adapter_off(
            bundle,
            first_item["formatted_prompt"],
            first_item["audit"],
            score_fc_candidate_batch,
        )
        sentinel = compare_sentinel_scores(
            sentinel_before,
            sentinel_after,
            tolerance=request.sentinel_score_tolerance,
        )
        runtime_identity = {
            "single_process": True,
            "single_loaded_base_model": True,
            "pair_order": "adapter_off_then_adapter_on_per_prompt",
            "base_model_object_id": base_model_identity,
            "peft_model_object_id": peft_model_identity,
            "wrapped_base_object_id": wrapped_base_identity,
            "object_identity_preserved": (
                id(bundle.model) == peft_model_identity
                and id(bundle.model.get_base_model()) == wrapped_base_identity
            ),
            "model_revision": bundle.model_revision,
            "precision": bundle.precision,
            "device": str(next(bundle.model.parameters()).device),
            "adapter_name": "route_remediation",
            "adapter_sha256": adapter_sha256,
            "sentinel": sentinel,
        }
        if not runtime_identity["object_identity_preserved"] or not sentinel["passed"]:
            raise RuntimeError("same-runtime identity or OFF sentinel check failed")
    finally:
        if bundle is not None:
            release_model(bundle)

    fc_pairs = build_forced_choice_pairs(off_fc, on_fc)
    crd_pairs = build_crd_pairs(off_crd, on_crd)
    analysis, failure_records = build_paired_analysis(
        fc_pairs,
        crd_pairs,
        off_crd,
        on_crd,
        bootstrap_samples=request.bootstrap_samples,
        bootstrap_seed=request.bootstrap_seed,
    )
    decision = classify_same_runtime_result(
        analysis,
        combined_noninferiority_margin=request.combined_noninferiority_margin,
    )
    after_snapshot = _source_snapshot(rr_output, source_manifest.parent, revision_lock)
    if before_snapshot != after_snapshot:
        raise RuntimeError("source artifacts changed during same-runtime evaluation")

    environment = current_environment_snapshot()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "analysis_status": "same_runtime_adapter_off_on_complete",
        "source": {
            "route_remediation": rr_provenance,
            "frozen_context": rr_context,
        },
        "runtime_identity": runtime_identity,
        "environment": environment,
        "analysis_code_sha256": analysis_code_sha256,
        "analysis": analysis,
        "decision": decision,
        "source_snapshot_before": before_snapshot,
        "source_snapshot_after": after_snapshot,
        "source_artifacts_modified": False,
        "source_gate_status_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
        "interpretation_boundary": (
            "This inference-only audit estimates the within-runtime effect of the "
            "fixed route-remediation adapter on the locked 1.5B panel. It does not "
            "change the historical gate, authorize remediation training, or authorize "
            "the 1/4/8 mappings-per-adapter experiment."
        ),
    }
    artifact_paths, artifact_hashes = _stage_artifacts(
        analysis_output=analysis_output,
        summary=summary,
        off_fc=off_fc,
        on_fc=on_fc,
        off_crd=off_crd,
        on_crd=on_crd,
        fc_pairs=fc_pairs,
        crd_pairs=crd_pairs,
        failure_records=failure_records,
    )
    precommit_snapshot = _source_snapshot(rr_output, source_manifest.parent, revision_lock)
    if precommit_snapshot != before_snapshot:
        raise RuntimeError("source artifacts changed before audit publication")
    _publish_manifest(
        paths=artifact_paths,
        hashes=artifact_hashes,
        summary=summary,
        identity=identity,
        source_snapshot=precommit_snapshot,
    )
    _progress(f"complete: {artifact_paths['summary']}")
    return artifact_paths["summary"]


def _first_forced_choice_item(
    bundle: ModelBundle,
    source: Any,
    hard_bank: dict[str, tuple[Any, ...]],
    audits: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    selected_by_id = {item.lesson.lesson_id: item for item in source.selected}
    lesson_id = _source_lesson_ids(selected_by_id)[0]
    probe = next(probe for probe in hard_bank[lesson_id] if probe.category == "conditional_route")
    rendered, _, _ = render_calibration_probe(
        probe,
        arm=Arm.EXTERNAL,
        external_note=selected_by_id[lesson_id].external_note,
    )
    return {
        "formatted_prompt": chat_prompt(bundle.tokenizer, rendered.prompt),
        "audit": audits[(lesson_id, Arm.EXTERNAL.value, probe.probe_id)],
    }


def _evaluate_forced_choice_pairs(
    *,
    bundle: ModelBundle,
    run_id: str,
    adapter_sha256: str,
    config: Any,
    source: Any,
    hard_bank: dict[str, tuple[Any, ...]],
    source_rows: dict[tuple[str, str, str], dict[str, Any]],
    audits: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_by_id = {item.lesson.lesson_id: item for item in source.selected}
    off_rows: list[dict[str, Any]] = []
    on_rows: list[dict[str, Any]] = []
    for lesson_id in _source_lesson_ids(selected_by_id):
        item = selected_by_id[lesson_id]
        probes = [probe for probe in hard_bank[lesson_id] if probe.category == "conditional_route"]
        if len(probes) != 4:
            raise ValueError(f"{lesson_id} must have four conditional-route probes")
        for probe in probes:
            key = (lesson_id, Arm.EXTERNAL.value, probe.probe_id)
            audit = audits[key]
            source_record = source_rows[key]
            rendered, variant, renderer = render_calibration_probe(
                probe,
                arm=Arm.EXTERNAL,
                external_note=item.external_note,
            )
            formatted = chat_prompt(bundle.tokenizer, rendered.prompt)
            off_outcome, on_outcome = score_adapter_off_on(
                bundle,
                formatted,
                audit,
                score_fc_candidate_batch,
            )
            common = {
                "schema_version": ROW_SCHEMA_VERSION,
                "run_id": run_id,
                "model_name": config.spec.model_name,
                "model_revision": config.spec.model_revision,
                "precision": bundle.precision,
                "lesson_id": lesson_id,
                "pair_id": probe.pair_id,
                "lesson_type": probe.lesson_type,
                "probe_id": probe.probe_id,
                "category": probe.category,
                "arm": Arm.EXTERNAL.value,
                "expected_action": probe.expected_action,
                "ordered_allowed_actions": list(probe.action_choices),
                "prompt_variant": variant,
                "prompt_rendering_version": renderer,
                "prompt_sha256": audit["prompt_sha256"],
                "candidate_audit_record_sha256": audit["record_sha256"],
                "frozen_source_row_sha256": source_record["source_row_sha256"],
            }
            off_row = _decision_row(common, off_outcome, "predicted_action", "adapter_off")
            on_row = _decision_row(common, on_outcome, "predicted_action", "adapter_on")
            off_row["correct"] = off_row["predicted_action"] == probe.expected_action
            on_row["correct"] = on_row["predicted_action"] == probe.expected_action
            off_row["adapter_sha256"] = None
            on_row["adapter_sha256"] = adapter_sha256
            on_row["source_row_sha256"] = json_hash(off_row)
            off_rows.append(off_row)
            on_rows.append(on_row)
            if len(off_rows) % 24 == 0:
                _progress(f"forced-choice pairs complete: {len(off_rows)}/96")
    if len(off_rows) != 96 or len(on_rows) != 96:
        raise ValueError("same-runtime external conditional-route matrix is incomplete")
    return off_rows, on_rows


def _evaluate_crd_pairs(
    *,
    bundle: ModelBundle,
    run_id: str,
    adapter_sha256: str,
    config: Any,
    bank: dict[str, tuple[Any, ...]],
    source_rows: dict[str, dict[str, Any]],
    audits: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    off_rows: list[dict[str, Any]] = []
    on_rows: list[dict[str, Any]] = []
    for probes in bank.values():
        for probe in probes:
            audit = audits[probe.probe_id]
            source = source_rows[probe.source_row_key]
            formatted = chat_prompt(bundle.tokenizer, probe.prompt)
            off_outcome, on_outcome = score_adapter_off_on(
                bundle,
                formatted,
                audit,
                score_crd_candidate_batch,
            )
            common = {
                "schema_version": ROW_SCHEMA_VERSION,
                "run_id": run_id,
                "model_name": config.spec.model_name,
                "model_revision": config.spec.model_revision,
                "precision": bundle.precision,
                "lesson_id": probe.lesson_id,
                "pair_id": probe.pair_id,
                "lesson_type": probe.lesson_type,
                "lesson_side": probe.lesson_side,
                "probe_id": probe.probe_id,
                "endpoint": probe.endpoint,
                "expected_candidate": probe.expected_candidate,
                "ordered_candidates": list(probe.ordered_candidates),
                "route_variant": probe.route_variant,
                "target_slot": probe.target_slot,
                "current_marker": probe.current_marker,
                "slot_candidate_order": probe.slot_candidate_order,
                "slot_content_order": probe.slot_content_order,
                "action_panel_position": probe.action_panel_position,
                "prompt_sha256": audit["prompt_sha256"],
                "candidate_audit_record_sha256": audit["record_sha256"],
                "source_probe_id": probe.source_probe_id,
                "source_row_key": probe.source_row_key,
                "source_row_sha256": source["source_row_sha256"],
                "source_fc_correct": bool(source["row"]["correct"]),
            }
            off_row = _decision_row(common, off_outcome, "predicted_candidate", "adapter_off")
            on_row = _decision_row(common, on_outcome, "predicted_candidate", "adapter_on")
            off_row["correct"] = off_row["predicted_candidate"] == probe.expected_candidate
            on_row["correct"] = on_row["predicted_candidate"] == probe.expected_candidate
            off_row["adapter_sha256"] = None
            on_row["adapter_sha256"] = adapter_sha256
            off_rows.append(off_row)
            on_rows.append(on_row)
            if len(off_rows) % 192 == 0:
                _progress(f"CRD pairs complete: {len(off_rows)}/1536")
    if len(off_rows) != 1_536 or len(on_rows) != 1_536:
        raise ValueError("same-runtime CRD matrix is incomplete")
    return off_rows, on_rows


def _decision_row(
    common: dict[str, Any],
    outcome: dict[str, Any],
    prediction_field: str,
    adapter_state: str,
) -> dict[str, Any]:
    return {
        **common,
        "adapter_state": adapter_state,
        "candidates": outcome["candidates"],
        prediction_field: outcome[prediction_field],
        "mean_" + prediction_field: outcome["mean_" + prediction_field],
        "top1_top2_margin": outcome["top1_top2_margin"],
        "sum_mean_disagreement": outcome["sum_mean_disagreement"],
        "tie": outcome["tie"],
        "mean_score_tie": outcome["mean_score_tie"],
        "non_finite": outcome["non_finite"],
        "error_status": outcome["error_status"],
        "error_message": outcome["error_message"],
        "latency_seconds": outcome["latency_seconds"],
    }


def score_adapter_off_on(
    bundle: ModelBundle,
    formatted_prompt: str,
    audit: dict[str, Any],
    scorer: Callable[[ModelBundle, str, dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Score one prompt OFF then ON while retaining one loaded PEFT model."""
    off = _score_adapter_off(bundle, formatted_prompt, audit, scorer)
    on = scorer(bundle, formatted_prompt, audit)
    return off, on


def _score_adapter_off(
    bundle: ModelBundle,
    formatted_prompt: str,
    audit: dict[str, Any],
    scorer: Callable[[ModelBundle, str, dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    with bundle.model.disable_adapter():
        return scorer(bundle, formatted_prompt, audit)


def compare_sentinel_scores(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    before_candidates = _sentinel_candidate_scores(before)
    after_candidates = _sentinel_candidate_scores(after)
    if before_candidates.keys() != after_candidates.keys():
        return {
            "passed": False,
            "reason": "candidate_set_changed",
            "max_abs_sum_logprob_delta": None,
            "tolerance": tolerance,
        }
    maximum = max(
        abs(before_candidates[name] - after_candidates[name]) for name in before_candidates
    )
    before_prediction = before.get("predicted_action") or before.get("predicted_candidate")
    after_prediction = after.get("predicted_action") or after.get("predicted_candidate")
    passed = (
        maximum <= tolerance
        and before_prediction == after_prediction
        and before.get("error_status") == after.get("error_status")
    )
    return {
        "passed": passed,
        "reason": "ok" if passed else "off_state_replay_mismatch",
        "prediction_before": before_prediction,
        "prediction_after": after_prediction,
        "error_status_before": before.get("error_status"),
        "error_status_after": after.get("error_status"),
        "max_abs_sum_logprob_delta": maximum,
        "tolerance": tolerance,
    }


def classify_same_runtime_result(
    analysis: dict[str, Any],
    *,
    combined_noninferiority_margin: float,
) -> dict[str, Any]:
    fc = analysis["forced_choice"]["external_conditional_route"]
    endpoints = analysis["crd"]["by_endpoint"]
    route = endpoints["route_only"]
    retrieval = endpoints["retrieval_only"]
    combined = endpoints["combined"]
    route_ci = route["paired_accuracy_difference"]["ci95"]
    combined_ci = combined["paired_accuracy_difference"]["ci95"]
    if route_ci is None or combined_ci is None:
        raise ValueError("primary cluster-bootstrap intervals are missing")
    primary = [fc, route, retrieval, combined]
    scoring_valid = all(
        values["tie_sensitivity"][state]["tie_count"] == 0
        for values in primary
        for state in ("base", "adapter")
    )
    route_improvement_supported = route_ci[0] > 0.0
    combined_regression_supported = combined_ci[1] < -combined_noninferiority_margin
    combined_noninferior_supported = combined_ci[0] >= -combined_noninferiority_margin
    conditional_route_pass = fc["adapter_accuracy"] >= 0.75
    route_only_pass = route["adapter_accuracy"] > 0.625
    retrieval_preserved = retrieval["adapter_accuracy"] >= 1.0
    if not scoring_valid:
        status = "scoring_integrity_failed"
    elif combined_regression_supported and route_improvement_supported:
        status = "composition_interference_confirmed"
    elif combined_regression_supported:
        status = "material_combined_regression_confirmed"
    elif (
        combined_noninferior_supported
        and conditional_route_pass
        and route_only_pass
        and retrieval_preserved
    ):
        status = "same_runtime_reading_qualification_candidate"
    elif combined_noninferior_supported:
        status = "combined_regression_not_supported"
    else:
        status = "same_runtime_effect_inconclusive"
    return {
        "status": status,
        "checks": {
            "zero_primary_ties": scoring_valid,
            "route_improvement_supported": route_improvement_supported,
            "combined_material_regression_supported": combined_regression_supported,
            "combined_noninferiority_supported": combined_noninferior_supported,
            "adapter_conditional_route_at_least_0_75": conditional_route_pass,
            "adapter_route_only_strictly_above_0_625": route_only_pass,
            "adapter_retrieval_only_at_least_1_0": retrieval_preserved,
        },
        "thresholds": {
            "conditional_route_min_accuracy": 0.75,
            "route_only_strictly_greater_than": 0.625,
            "retrieval_only_min_accuracy": 1.0,
            "combined_noninferiority_margin": combined_noninferiority_margin,
        },
        "historical_gate_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
        "next_action": (
            "design_composition_preserving_remediation"
            if status == "composition_interference_confirmed"
            else "review_same_runtime_evidence_before_any_new_intervention"
        ),
    }


def _stage_artifacts(
    *,
    analysis_output: Path,
    summary: dict[str, Any],
    off_fc: list[dict[str, Any]],
    on_fc: list[dict[str, Any]],
    off_crd: list[dict[str, Any]],
    on_crd: list[dict[str, Any]],
    fc_pairs: list[dict[str, Any]],
    crd_pairs: list[dict[str, Any]],
    failure_records: list[dict[str, Any]],
) -> tuple[dict[str, Path], dict[str, str]]:
    paths = {
        "summary": analysis_output / "summary.json",
        "report": analysis_output / "report.md",
        "off_forced_choice": analysis_output / "results" / "adapter_off_fc.jsonl",
        "on_forced_choice": analysis_output / "results" / "adapter_on_fc.jsonl",
        "off_crd": analysis_output / "results" / "adapter_off_crd.jsonl",
        "on_crd": analysis_output / "results" / "adapter_on_crd.jsonl",
        "paired_records": analysis_output / "paired_records.jsonl",
        "combined_failures": analysis_output / "combined_failure_records.jsonl",
        "manifest": analysis_output / "audit_manifest.json",
    }
    pair_rows = [
        {"schema_version": PAIRED_ROW_SCHEMA_VERSION, "matrix": "forced_choice", **row}
        for row in fc_pairs
    ] + [{"schema_version": PAIRED_ROW_SCHEMA_VERSION, "matrix": "crd", **row} for row in crd_pairs]
    payloads = {
        "summary": _json_bytes(summary),
        "report": _render_report(summary).encode(),
        "off_forced_choice": _jsonl_bytes(off_fc),
        "on_forced_choice": _jsonl_bytes(on_fc),
        "off_crd": _jsonl_bytes(off_crd),
        "on_crd": _jsonl_bytes(on_crd),
        "paired_records": _jsonl_bytes(pair_rows),
        "combined_failures": _jsonl_bytes(failure_records),
    }
    hashes = {
        name: immutable_write(paths[name], payload, f"same-runtime {name}")
        for name, payload in payloads.items()
    }
    return paths, hashes


def _publish_manifest(
    *,
    paths: dict[str, Path],
    hashes: dict[str, str],
    summary: dict[str, Any],
    identity: dict[str, Any],
    source_snapshot: dict[str, Any],
) -> None:
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": summary["run_id"],
        "identity": identity,
        "source_snapshot_sha256": json_hash(source_snapshot),
        "artifacts": {
            name: {"path": str(paths[name]), "sha256": digest} for name, digest in hashes.items()
        },
        "source_artifacts_modified": False,
        "source_gate_status_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    immutable_write(paths["manifest"], _json_bytes(manifest), "same-runtime manifest")


def _render_report(summary: dict[str, Any]) -> str:
    analysis = summary["analysis"]
    rows = [
        ("External conditional-route", analysis["forced_choice"]["external_conditional_route"]),
        *[(f"CRD {endpoint}", analysis["crd"]["by_endpoint"][endpoint]) for endpoint in ENDPOINTS],
    ]
    lines = [
        "# Same-runtime adapter OFF/ON qualification audit",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Decision: `{summary['decision']['status']}`",
        "- Runtime: one process, one loaded base-model object, OFF→ON per prompt",
        "- Historical gate changed: `false`",
        "- 1/4/8 authorized: `false`",
        "",
        "| Endpoint | N | OFF | ON | ON−OFF (95% lesson-cluster CI) | C→W | W→C |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, values in rows:
        difference = values["paired_accuracy_difference"]
        interval = difference["ci95"]
        transitions = values["transition_counts"]
        lines.append(
            f"| {label} | {values['decision_count']} | {values['base_accuracy']:.4f} | "
            f"{values['adapter_accuracy']:.4f} | {difference['estimate']:.4f} "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] | "
            f"{transitions['correct_to_wrong']} | {transitions['wrong_to_correct']} |"
        )
    sentinel = summary["runtime_identity"]["sentinel"]
    lines.extend(
        [
            "",
            "## Integrity and interpretation",
            "",
            f"- OFF replay sentinel passed: `{str(sentinel['passed']).lower()}`; "
            f"max score delta `{sentinel['max_abs_sum_logprob_delta']}`.",
            "- Primary uncertainty is the lesson-cluster bootstrap. McNemar values are "
            "row-level sensitivity analyses.",
            "- This audit diagnoses the fixed adapter only. It does not authorize training "
            "or the 1/4/8 capacity experiment.",
            "",
        ]
    )
    return "\n".join(lines)


def _sentinel_candidate_scores(outcome: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for candidate in outcome.get("candidates", []):
        name = candidate.get("action", candidate.get("candidate"))
        score = candidate.get("sum_logprob")
        if not isinstance(name, str) or not isinstance(score, (int, float)):
            raise ValueError("sentinel candidate record is invalid")
        scores[name] = float(score)
    if not scores:
        raise ValueError("sentinel candidate set is empty")
    return scores


def _require_independent_new_output(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"same-runtime output already exists: {output}")
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("same-runtime output must be independent from RR source")


def _tree_hash(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"cannot snapshot empty source tree: {root}")
    digest = sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {"path": str(root), "file_count": len(files), "sha256": digest.hexdigest()}


def _source_snapshot(
    rr_output: Path,
    frozen_fc_output: Path,
    revision_lock: Path,
) -> dict[str, Any]:
    return {
        "route_remediation": _tree_hash(rr_output),
        "frozen_forced_choice": _tree_hash(frozen_fc_output),
        "experiment_code_revision_lock": {
            "path": str(revision_lock),
            "sha256": file_hash(revision_lock),
        },
    }


def _progress(message: str) -> None:
    print(f"[p0d2hrr-same-runtime] {message}", flush=True)
