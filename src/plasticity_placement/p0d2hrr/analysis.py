from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.runtime import current_code_hash
from plasticity_placement.p0d.analysis import _atomic_json_write
from plasticity_placement.p0d2hc.config import CALIBRATION_ARMS
from plasticity_placement.p0d2hcrd.analysis import (
    _endpoint_summary,
    _factor_summary,
    _lesson_endpoint_metrics,
    _linked_source_summary,
    _paired_contrasts,
    _prediction_distributions,
    diagnostic_gate,
)
from plasticity_placement.p0d2hcrd.runtime import validate_frozen_source
from plasticity_placement.p0d2hcrd.scoring import (
    rank_candidate_scores as rank_crd_candidates,
)
from plasticity_placement.p0d2hfc.analysis import (
    _condition_summary,
    _lesson_category_metrics,
    _lesson_overall_metrics,
    _model_gate,
    _paired_contrast,
    _within_arm_contrast,
)
from plasticity_placement.p0d2hfc.scoring import (
    rank_candidate_scores as rank_fc_candidates,
)
from plasticity_placement.p0d2hrr.config import ResolvedPilotConfig
from plasticity_placement.p0d2hrr.io import file_hash, read_json_object
from plasticity_placement.p0d2hrr.manifest import PilotManifest
from plasticity_placement.p0d2hrr.preflight import (
    load_resolved_config,
    load_scale_canary_source_rows,
)
from plasticity_placement.p0d2hrr.runtime import _adapter_hash

MODEL_ID = "scale_canary_route_remediated"


def aggregate_experiment(
    output_dir: Path,
    bootstrap_samples: int = 10_000,
) -> Path:
    output_dir = output_dir.resolve()
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    config = load_resolved_config(output_dir)
    manifest = PilotManifest.load(output_dir / "manifest.json")
    if manifest.state != "evaluated":
        raise ValueError(f"aggregation requires state=evaluated, found {manifest.state}")
    dev_rows, fc_rows, crd_rows = _validate_and_load_rows(config, manifest)
    fc_audit = read_json_object(
        output_dir / "preflight" / "forced_choice_token_audit.json",
        "forced-choice token audit",
    )
    crd_bank_audit = read_json_object(
        output_dir / "preflight" / "crd_bank_audit.json",
        "CRD bank audit",
    )
    crd_token_audit = read_json_object(
        output_dir / "preflight" / "crd_token_audit.json",
        "CRD token audit",
    )

    lesson_category = _lesson_category_metrics(fc_rows)
    lesson_overall = _lesson_overall_metrics(lesson_category)
    condition_summary = _condition_summary(
        lesson_category,
        lesson_overall,
        bootstrap_samples,
    )
    contrasts = {
        "external_minus_no_write": _paired_contrast(
            lesson_overall,
            MODEL_ID,
            "external",
            "no_write",
            "accuracy",
            bootstrap_samples,
        ),
        "oracle_minus_external": _paired_contrast(
            lesson_overall,
            MODEL_ID,
            "answer_copy_oracle",
            "external",
            "accuracy",
            bootstrap_samples,
        ),
        "forced_choice_minus_strict_by_arm": {
            arm: _within_arm_contrast(
                lesson_overall,
                MODEL_ID,
                arm,
                bootstrap_samples,
            )
            for arm in CALIBRATION_ARMS
        },
    }
    official_gate = _model_gate(
        MODEL_ID,
        condition_summary[MODEL_ID],
        contrasts,
        fc_audit,
    )

    lesson_endpoint = _lesson_endpoint_metrics(crd_rows)
    endpoint_summary = _endpoint_summary(
        lesson_endpoint,
        bootstrap_samples,
    )
    crd_gate = diagnostic_gate(
        endpoint_summary,
        crd_bank_audit,
        crd_token_audit,
    )
    route_gate = _route_remediation_gate(
        config,
        official_gate,
        endpoint_summary,
        fc_rows,
        crd_rows,
    )
    summary = {
        "schema_version": "p0d2hrr-summary-v1",
        "run_id": manifest.payload["run_id"],
        "stage": "route_remediation_lora_pilot",
        "model_id": MODEL_ID,
        "adapter_sha256": manifest.payload["evaluation"]["adapter_sha256"],
        "precision": manifest.payload["evaluation"]["precision"],
        "dev": {
            "decision_row_count": len(dev_rows),
            "accuracy": sum(bool(row["correct"]) for row in dev_rows) / len(dev_rows),
            "task_accuracy": {
                task: (
                    sum(bool(row["correct"]) for row in dev_rows if row["task"] == task)
                    / sum(row["task"] == task for row in dev_rows)
                )
                for task in ("route_only", "route_and_copy")
            },
            "used_for_checkpoint_selection": False,
        },
        "forced_choice": {
            "decision_row_count": len(fc_rows),
            "condition_summary": condition_summary[MODEL_ID],
            "contrasts": contrasts,
            "official_model_gate": official_gate,
            "lesson_category_metrics": lesson_category,
            "lesson_overall_metrics": lesson_overall,
        },
        "crd": {
            "decision_row_count": len(crd_rows),
            "endpoint_summary": endpoint_summary,
            "factor_summary": _factor_summary(crd_rows, bootstrap_samples),
            "contrasts": _paired_contrasts(
                lesson_endpoint,
                bootstrap_samples,
            ),
            "prediction_distributions": _prediction_distributions(crd_rows),
            "linked_source_conditional_route": _linked_source_summary(
                crd_rows,
                bootstrap_samples,
            ),
            "diagnostic_gate": crd_gate,
            "lesson_endpoint_metrics": lesson_endpoint,
        },
        "route_remediation_gate": route_gate,
        "gates": {
            "run_valid": True,
            "training_complexity_review_eligible": route_gate[
                "training_complexity_review_eligible"
            ],
            "eligible_model_ids": (
                [MODEL_ID] if route_gate["training_complexity_review_eligible"] else []
            ),
            "automatic_training_started": False,
            "automatic_narrow_scan_started": False,
            "automatic_rlvr_started": False,
        },
        "source_run_id": config.source_run_id,
        "source_manifest_sha256": config.source_manifest_sha256,
        "preregistration_sha256": config.preregistration_sha256,
        "no_fabrication_status": (
            "All reported values derive from the single preregistered adapter "
            "and complete provenance-checked forced-choice/CRD result matrices."
        ),
    }
    aggregate_dir = output_dir / "results" / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    summary_path = aggregate_dir / "summary.json"
    _atomic_json_write(summary_path, summary)
    (aggregate_dir / "main_table.md").write_text(
        _render_main_table(condition_summary[MODEL_ID], endpoint_summary, route_gate),
        encoding="utf-8",
    )
    decision_path = aggregate_dir / "next_stage_decision.json"
    _atomic_json_write(
        decision_path,
        {
            "schema_version": "p0d2hrr-next-stage-v1",
            "status": route_gate["status"],
            "training_complexity_review_eligible": route_gate[
                "training_complexity_review_eligible"
            ],
            "eligible_model_ids": summary["gates"]["eligible_model_ids"],
            "automatic_training_started": False,
            "automatic_narrow_scan_started": False,
            "automatic_rlvr_started": False,
            "instruction": route_gate["next_action"],
        },
    )
    manifest.transition(
        "verified",
        aggregate={
            "summary_path": str(summary_path),
            "summary_sha256": file_hash(summary_path),
            "decision_path": str(decision_path),
            "decision_sha256": file_hash(decision_path),
            "status": route_gate["status"],
            "training_complexity_review_eligible": route_gate[
                "training_complexity_review_eligible"
            ],
        },
    )
    return summary_path


def _route_remediation_gate(
    config: ResolvedPilotConfig,
    official_gate: dict[str, Any],
    endpoint_summary: dict[str, dict[str, Any]],
    fc_rows: list[dict[str, Any]],
    crd_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    external = official_gate["conditional_route_external_accuracy"]
    route = float(endpoint_summary["route_only"]["accuracy"])
    retrieval = float(endpoint_summary["retrieval_only"]["accuracy"])
    combined = float(endpoint_summary["combined"]["accuracy"])
    fc_integrity_rate = sum(row["error_status"] != "ok" for row in fc_rows) / len(fc_rows)
    crd_integrity_rate = sum(row["error_status"] != "ok" for row in crd_rows) / len(crd_rows)
    gates = config.spec.gates
    checks = {
        "original_forced_choice_gate": official_gate["eligible"] is True,
        "conditional_route": external >= gates.conditional_route_min_accuracy,
        "route_only_improved": route > gates.route_only_strictly_greater_than,
        "retrieval_only_preserved": retrieval >= gates.retrieval_only_min_accuracy,
        "combined_preserved": combined >= gates.combined_min_accuracy,
        "forced_choice_scoring_integrity": (
            fc_integrity_rate <= gates.max_tie_error_nonfinite_rate
        ),
        "crd_scoring_integrity": (crd_integrity_rate <= gates.max_tie_error_nonfinite_rate),
    }
    passed = all(checks.values())
    if passed:
        status = "route_remediation_passed_review_required"
        next_action = (
            "Human-review a separately preregistered 1/4/8 "
            "training-complexity experiment. Do not start it automatically."
        )
    elif not checks["original_forced_choice_gate"]:
        status = "original_calibration_gate_failed"
        next_action = (
            "Keep training-complexity work locked. Report the complete failed "
            "forced-choice checks; any further remediation requires a new protocol."
        )
    else:
        status = "route_guardrail_failed"
        next_action = (
            "Keep training-complexity work locked. Inspect route, retrieval, "
            "combined, and scoring-integrity guardrails without tuning this run."
        )
    return {
        "status": status,
        "checks": checks,
        "observed": {
            "conditional_route": external,
            "route_only": route,
            "retrieval_only": retrieval,
            "combined": combined,
            "forced_choice_tie_error_nonfinite_rate": fc_integrity_rate,
            "crd_tie_error_nonfinite_rate": crd_integrity_rate,
        },
        "thresholds": {
            "conditional_route_min_accuracy": (gates.conditional_route_min_accuracy),
            "route_only_strictly_greater_than": (gates.route_only_strictly_greater_than),
            "retrieval_only_min_accuracy": (gates.retrieval_only_min_accuracy),
            "combined_min_accuracy": gates.combined_min_accuracy,
            "max_tie_error_nonfinite_rate": (gates.max_tie_error_nonfinite_rate),
        },
        "training_complexity_review_eligible": passed,
        "automatic_training_started": False,
        "automatic_narrow_scan_started": False,
        "automatic_rlvr_started": False,
        "next_action": next_action,
    }


def _validate_and_load_rows(
    config: ResolvedPilotConfig,
    manifest: PilotManifest,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation = manifest.payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("route-remediation evaluation metadata is missing")
    if evaluation.get("completed_locked_external_evaluations") != 1:
        raise ValueError("locked external evaluation count changed")
    if current_code_hash() != config.code_sha256:
        raise ValueError("code changed after route-remediation preregistration")
    source = validate_frozen_source(config.source_manifest_path)
    fc_source_rows = load_scale_canary_source_rows(source)
    if _adapter_hash(config.output_dir / "adapter") != evaluation["adapter_sha256"]:
        raise ValueError("route-remediation adapter changed after evaluation")
    dev_path = Path(str(evaluation["dev_result_path"]))
    fc_path = Path(str(evaluation["forced_choice_result_path"]))
    crd_path = Path(str(evaluation["crd_result_path"]))
    expected_paths = (
        config.output_dir / "results" / "dev" / "rows.jsonl",
        config.output_dir / "results" / "forced_choice" / "rows.jsonl",
        config.output_dir / "results" / "crd" / "rows.jsonl",
    )
    if tuple(path.resolve() for path in (dev_path, fc_path, crd_path)) != tuple(
        path.resolve() for path in expected_paths
    ):
        raise ValueError("route-remediation result paths changed")
    if (
        file_hash(dev_path) != evaluation["dev_result_sha256"]
        or file_hash(fc_path) != evaluation["forced_choice_result_sha256"]
        or file_hash(crd_path) != evaluation["crd_result_sha256"]
    ):
        raise ValueError("route-remediation raw result hash changed")
    dev_rows = _load_jsonl(dev_path)
    fc_rows = _load_jsonl(fc_path)
    crd_rows = _load_jsonl(crd_path)
    if len(dev_rows) != 96 or len(fc_rows) != 1_152 or len(crd_rows) != 1_536:
        raise ValueError("route-remediation result matrix is incomplete")
    dev_audits = _audit_hash_index(
        config.output_dir / "preflight" / "dev_token_audit.json",
        "example_id",
    )
    fc_audits = _audit_hash_index(
        config.output_dir / "preflight" / "forced_choice_token_audit.json",
        ("lesson_id", "arm", "probe_id"),
    )
    crd_audits = _audit_hash_index(
        config.output_dir / "preflight" / "crd_token_audit.json",
        "probe_id",
    )
    _validate_row_matrix(
        dev_rows,
        "p0d2hrr-dev-decision-row-v1",
        config,
        manifest,
        audit_hashes=dev_audits,
    )
    _validate_row_matrix(
        fc_rows,
        "p0d2hrr-fc-decision-row-v1",
        config,
        manifest,
        audit_hashes=fc_audits,
        source_rows=fc_source_rows,
    )
    _validate_row_matrix(
        crd_rows,
        "p0d2hrr-crd-decision-row-v1",
        config,
        manifest,
        audit_hashes=crd_audits,
        source_rows=source.source_rows,
    )
    return dev_rows, fc_rows, crd_rows


def _validate_row_matrix(
    rows: list[dict[str, Any]],
    schema_version: str,
    config: ResolvedPilotConfig,
    manifest: PilotManifest,
    *,
    audit_hashes: dict[object, str],
    source_rows: dict[object, dict[str, Any]] | None = None,
) -> None:
    keys = [
        (
            row.get("lesson_id"),
            row.get("probe_id"),
            row.get("arm"),
            row.get("example_id"),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("route-remediation result matrix contains duplicates")
    adapter_sha256 = manifest.payload["evaluation"]["adapter_sha256"]
    run_id = manifest.payload["run_id"]
    for row in rows:
        audit_key: object
        source_key: object | None
        if schema_version == "p0d2hrr-dev-decision-row-v1":
            audit_key = str(row.get("example_id"))
            source_key = None
        elif schema_version == "p0d2hrr-fc-decision-row-v1":
            audit_key = (
                str(row.get("lesson_id")),
                str(row.get("arm")),
                str(row.get("probe_id")),
            )
            source_key = audit_key
        else:
            audit_key = str(row.get("probe_id"))
            source_key = str(row.get("source_row_key"))
        if (
            row.get("schema_version") != schema_version
            or row.get("run_id") != run_id
            or row.get("model_id") != MODEL_ID
            or row.get("model_name") != config.spec.model_name
            or row.get("model_revision") != config.spec.model_revision
            or row.get("source_run_id") != config.source_run_id
            or row.get("source_manifest_sha256") != config.source_manifest_sha256
            or row.get("precision") != config.source_evaluation_precision
            or row.get("adapter_sha256") != adapter_sha256
            or row.get("candidate_audit_record_sha256") != audit_hashes.get(audit_key)
        ):
            raise ValueError("route-remediation result provenance changed")
        if source_key is not None:
            source = source_rows.get(source_key) if source_rows else None
            if source is None or row.get("source_row_sha256") != source["source_row_sha256"]:
                raise ValueError("route-remediation source-row provenance changed")
        _verify_ranked_outcome(row, schema_version)


def _verify_ranked_outcome(
    row: dict[str, Any],
    schema_version: str,
) -> None:
    candidates = row.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("route-remediation candidates are missing")
    if schema_version == "p0d2hrr-fc-decision-row-v1":
        ranked = rank_fc_candidates(candidates)
        predicted_field = "predicted_action"
        expected_field = "expected_action"
        mean_field = "mean_predicted_action"
    else:
        ranked = rank_crd_candidates(candidates)
        predicted_field = "predicted_candidate"
        expected_field = "expected_candidate"
        mean_field = "mean_predicted_candidate"
    fields = (
        predicted_field,
        mean_field,
        "top1_top2_margin",
        "sum_mean_disagreement",
        "tie",
        "mean_score_tie",
        "non_finite",
        "error_status",
    )
    if any(row.get(field) != ranked[field] for field in fields):
        raise ValueError("route-remediation ranking differs from candidate scores")
    if bool(row.get("correct")) != (row.get(predicted_field) == row.get(expected_field)):
        raise ValueError("route-remediation correctness differs from ranking")


def _audit_hash_index(
    path: Path,
    key_fields: str | tuple[str, ...],
) -> dict[object, str]:
    report = read_json_object(path, "candidate token audit")
    result: dict[object, str] = {}
    fields = (key_fields,) if isinstance(key_fields, str) else key_fields
    for record in report.get("records", []):
        key_values = tuple(str(record[field]) for field in fields)
        key: object = key_values[0] if len(key_values) == 1 else key_values
        if key in result:
            raise ValueError(f"duplicate candidate token audit key: {key}")
        result[key] = str(record["record_sha256"])
    return result


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _render_main_table(
    forced_choice: dict[str, Any],
    endpoints: dict[str, dict[str, Any]],
    gate: dict[str, Any],
) -> str:
    lines = [
        "# P0-D2H Route-Remediation LoRA Pilot",
        "",
        "| Metric | Accuracy | Threshold | Pass |",
        "|---|---:|---:|:---:|",
        (
            "| External conditional route | "
            f"{forced_choice['external']['categories']['conditional_route']['accuracy']:.4f} | "
            f">= {gate['thresholds']['conditional_route_min_accuracy']:.4f} | "
            f"{'yes' if gate['checks']['conditional_route'] else 'no'} |"
        ),
        (
            "| CRD route only | "
            f"{endpoints['route_only']['accuracy']:.4f} | "
            f"> {gate['thresholds']['route_only_strictly_greater_than']:.4f} | "
            f"{'yes' if gate['checks']['route_only_improved'] else 'no'} |"
        ),
        (
            "| CRD retrieval only | "
            f"{endpoints['retrieval_only']['accuracy']:.4f} | "
            f">= {gate['thresholds']['retrieval_only_min_accuracy']:.4f} | "
            f"{'yes' if gate['checks']['retrieval_only_preserved'] else 'no'} |"
        ),
        (
            "| CRD combined | "
            f"{endpoints['combined']['accuracy']:.4f} | "
            f">= {gate['thresholds']['combined_min_accuracy']:.4f} | "
            f"{'yes' if gate['checks']['combined_preserved'] else 'no'} |"
        ),
        "",
        f"Decision: `{gate['status']}`.",
        "",
        "No 1/4/8 scan, narrow scan, or RLVR was started automatically.",
    ]
    return "\n".join(lines) + "\n"
