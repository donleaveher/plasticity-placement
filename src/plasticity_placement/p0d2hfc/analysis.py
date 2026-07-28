from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from plasticity_placement.p0d.analysis import _atomic_json_write, _bootstrap_ci
from plasticity_placement.p0d2h.probes import HARD_CATEGORIES
from plasticity_placement.p0d2hc.config import CALIBRATION_ARMS
from plasticity_placement.p0d2hfc.config import (
    EXTERNAL_MIN_ACCURACY,
    EXTERNAL_MIN_CATEGORY_ACCURACY,
    MAX_TIE_ERROR_NONFINITE_RATE,
    NO_WRITE_MAX_ACCURACY,
    ORACLE_MIN_ACCURACY,
    ORACLE_MIN_CATEGORY_ACCURACY,
    P0D2HFCRequest,
    ResolvedP0D2HFCConfig,
)
from plasticity_placement.p0d2hfc.manifest import unit_key
from plasticity_placement.p0d2hfc.runtime import (
    _file_hash,
    _load_candidate_token_audit,
    _result_path,
    _verify_rows,
    resolve_request,
)


def aggregate_experiment(
    output_dir: Path,
    bootstrap_samples: int = 10_000,
) -> Path:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"missing P0-D2H-CAL-FC manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config, rows, audit_report = _validate_and_load(output_dir, manifest)
    lesson_category = _lesson_category_metrics(rows)
    lesson_overall = _lesson_overall_metrics(lesson_category)
    condition_summary = _condition_summary(
        lesson_category,
        lesson_overall,
        bootstrap_samples,
    )
    contrasts = {
        model.model_id: {
            "external_minus_no_write": _paired_contrast(
                lesson_overall,
                model.model_id,
                "external",
                "no_write",
                "accuracy",
                bootstrap_samples,
            ),
            "oracle_minus_external": _paired_contrast(
                lesson_overall,
                model.model_id,
                "answer_copy_oracle",
                "external",
                "accuracy",
                bootstrap_samples,
            ),
            "forced_choice_minus_strict_by_arm": {
                arm: _within_arm_contrast(
                    lesson_overall,
                    model.model_id,
                    arm,
                    bootstrap_samples,
                )
                for arm in CALIBRATION_ARMS
            },
        }
        for model in config.models
    }
    model_gates = {
        model.model_id: _model_gate(
            model.model_id,
            condition_summary[model.model_id],
            contrasts[model.model_id],
            audit_report,
        )
        for model in config.models
    }
    cross_scale = _cross_scale_gate(condition_summary, model_gates)
    summary = {
        "schema_version": "p0d2hfc-summary-v1",
        "run_id": manifest["run_id"],
        "stage": "hard_probe_forced_choice",
        "source_p0d2hc_run_id": config.source_run_id,
        "model_count": len(config.models),
        "models": [model.to_dict() for model in config.models],
        "selected_lesson_count": len(config.selected_lesson_ids),
        "hard_categories": list(HARD_CATEGORIES),
        "calibration_arms": list(CALIBRATION_ARMS),
        "unit_count": len(manifest["units"]),
        "decision_row_count": len(rows),
        "candidate_token_audit": {
            key: value
            for key, value in audit_report.items()
            if key != "records"
        },
        "condition_summary": condition_summary,
        "contrasts": contrasts,
        "model_gates": model_gates,
        "cross_scale_gate": cross_scale,
        "gates": {
            "run_valid": True,
            "next_stage_status": cross_scale["next_stage_status"],
            "eligible_model_ids": cross_scale["eligible_model_ids"],
            "automatic_training_started": False,
            "automatic_narrow_scan_started": False,
        },
        "lesson_category_metrics": lesson_category,
        "lesson_overall_metrics": lesson_overall,
        "source_manifest_sha256": config.source_manifest_sha256,
        "source_summary_sha256": config.source_summary_sha256,
        "source_raw_results_sha256": config.source_raw_results_sha256,
        "hard_probe_hashes": config.hard_probe_hashes,
        "no_fabrication_status": (
            "All values derive from provenance-verified forced-choice rows "
            "paired to the complete immutable P0-D2H-CAL source."
        ),
    }
    aggregate_dir = output_dir / "results" / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    summary_path = aggregate_dir / "summary.json"
    _atomic_json_write(summary_path, summary)
    (aggregate_dir / "main_table.md").write_text(
        _render_main_table(condition_summary, contrasts, model_gates),
        encoding="utf-8",
    )
    _atomic_json_write(
        aggregate_dir / "next_stage_decision.json",
        {
            "schema_version": "p0d2hfc-next-stage-v1",
            "source_run_id": manifest["run_id"],
            "status": cross_scale["next_stage_status"],
            "cross_scale_status": cross_scale["status"],
            "eligible_model_ids": cross_scale["eligible_model_ids"],
            "model_statuses": {
                model_id: gate["status"]
                for model_id, gate in model_gates.items()
            },
            "automatic_training_started": False,
            "automatic_narrow_scan_started": False,
            "instruction": cross_scale["next_action"],
        },
    )
    return summary_path


def _validate_and_load(
    output_dir: Path,
    manifest: dict[str, Any],
) -> tuple[
    ResolvedP0D2HFCConfig,
    list[dict[str, Any]],
    dict[str, Any],
]:
    required = {
        "run_id",
        "config",
        "source_manifest_path",
        "candidate_token_audit_sha256",
        "selected_lessons",
        "models",
        "units",
    }
    if manifest.get("schema_version") != "p0d2hfc-manifest-v1":
        raise ValueError("not a P0-D2H-CAL-FC manifest")
    if not required.issubset(manifest):
        raise ValueError(
            f"forced-choice manifest is missing "
            f"{sorted(required - manifest.keys())}"
        )
    if manifest.get("errors"):
        raise ValueError("forced-choice manifest contains recorded errors")
    request = P0D2HFCRequest(
        output_dir=output_dir,
        source_manifest=Path(str(manifest["source_manifest_path"])),
    )
    config, hard_bank, _, source_rows = resolve_request(request)
    if manifest["config"] != config.identity_dict():
        raise ValueError("forced-choice config/source identity changed")
    audit_hash, audits, audit_report = _load_candidate_token_audit(config)
    if audit_hash != manifest["candidate_token_audit_sha256"]:
        raise ValueError("forced-choice candidate-token audit changed")
    if (
        list(config.selected_lesson_ids) != manifest["selected_lessons"]
        or {
            model.model_id: model.to_dict() for model in config.models
        }
        != manifest["models"]
    ):
        raise ValueError("forced-choice frozen model/lesson matrix changed")
    expected_units = {
        unit_key(model.model_id, lesson_id)
        for model in config.models
        for lesson_id in config.selected_lesson_ids
    }
    if set(manifest["units"]) != expected_units:
        raise ValueError("forced-choice unit matrix is incomplete")

    rows: list[dict[str, Any]] = []
    for model in config.models:
        for lesson_id in config.selected_lesson_ids:
            key = unit_key(model.model_id, lesson_id)
            unit = manifest["units"][key]
            if unit.get("state") != "verified":
                raise ValueError(f"forced-choice unit is not verified: {key}")
            path = _result_path(output_dir, model.model_id, lesson_id)
            if unit.get("result_sha256") != _file_hash(path):
                raise ValueError(f"forced-choice raw file hash changed: {key}")
            _verify_rows(
                config,
                model,
                manifest["run_id"],
                path,
                hard_bank[lesson_id],
                lesson_id,
                source_rows,
                audits,
                expected_precision=str(unit["evaluation_precision"]),
            )
            rows.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    observed_keys = [
        (
            str(row["model_id"]),
            str(row["lesson_id"]),
            str(row["arm"]),
            str(row["probe_id"]),
        )
        for row in rows
    ]
    if (
        len(rows) != config.expected_decision_row_count
        or len(observed_keys) != len(set(observed_keys))
    ):
        raise ValueError(
            "forced-choice complete matrix has missing or duplicate rows"
        )
    return config, rows, audit_report


def _lesson_category_metrics(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["model_id"]),
                str(row["arm"]),
                str(row["lesson_id"]),
                str(row["category"]),
            )
        ].append(row)
    result: list[dict[str, Any]] = []
    for (model_id, arm, lesson_id, category), group in sorted(
        groups.items()
    ):
        if len(group) != 4:
            raise ValueError(
                f"forced-choice category row count mismatch: "
                f"{model_id}/{arm}/{lesson_id}/{category}"
            )
        first = group[0]
        margins = [
            float(row["top1_top2_margin"])
            for row in group
            if row["top1_top2_margin"] is not None
        ]
        comparable = [
            row
            for row in group
            if row["sum_mean_disagreement"] is not None
        ]
        result.append(
            {
                "model_id": model_id,
                "arm": arm,
                "lesson_id": lesson_id,
                "pair_id": first["pair_id"],
                "lesson_type": first["lesson_type"],
                "category": category,
                "accuracy": mean(bool(row["correct"]) for row in group),
                "source_strict_accuracy": mean(
                    bool(row["source_strict_correct"]) for row in group
                ),
                "tie_rate": mean(bool(row["tie"]) for row in group),
                "non_finite_rate": mean(
                    bool(row["non_finite"]) for row in group
                ),
                "error_rate": mean(
                    row["error_status"]
                    not in {"ok", "tie", "non_finite"}
                    for row in group
                ),
                "tie_error_nonfinite_rate": mean(
                    row["error_status"] != "ok" for row in group
                ),
                "mean_score_tie_rate": mean(
                    bool(row["mean_score_tie"]) for row in group
                ),
                "sum_mean_comparable_count": len(comparable),
                "sum_mean_disagreement_rate": (
                    mean(
                        bool(row["sum_mean_disagreement"])
                        for row in comparable
                    )
                    if comparable
                    else None
                ),
                "mean_top1_top2_margin": (
                    mean(margins) if margins else None
                ),
                "median_top1_top2_margin": (
                    median(margins) if margins else None
                ),
                "median_latency_seconds": median(
                    float(row["latency_seconds"]) for row in group
                ),
            }
        )
    return result


def _lesson_overall_metrics(
    category_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in category_rows:
        groups[
            (
                str(row["model_id"]),
                str(row["arm"]),
                str(row["lesson_id"]),
            )
        ].append(row)
    result: list[dict[str, Any]] = []
    for (model_id, arm, lesson_id), group in sorted(groups.items()):
        if {str(row["category"]) for row in group} != set(HARD_CATEGORIES):
            raise ValueError(
                f"missing forced-choice category: {model_id}/{arm}/{lesson_id}"
            )
        first = group[0]
        weighted_comparable = sum(
            int(row["sum_mean_comparable_count"]) for row in group
        )
        result.append(
            {
                "model_id": model_id,
                "arm": arm,
                "lesson_id": lesson_id,
                "pair_id": first["pair_id"],
                "lesson_type": first["lesson_type"],
                "accuracy": mean(float(row["accuracy"]) for row in group),
                "source_strict_accuracy": mean(
                    float(row["source_strict_accuracy"]) for row in group
                ),
                "tie_rate": mean(float(row["tie_rate"]) for row in group),
                "non_finite_rate": mean(
                    float(row["non_finite_rate"]) for row in group
                ),
                "error_rate": mean(float(row["error_rate"]) for row in group),
                "tie_error_nonfinite_rate": mean(
                    float(row["tie_error_nonfinite_rate"]) for row in group
                ),
                "mean_score_tie_rate": mean(
                    float(row["mean_score_tie_rate"]) for row in group
                ),
                "sum_mean_comparable_count": weighted_comparable,
                "sum_mean_disagreement_rate": (
                    sum(
                        int(row["sum_mean_comparable_count"])
                        * float(row["sum_mean_disagreement_rate"])
                        for row in group
                        if row["sum_mean_disagreement_rate"] is not None
                    )
                    / weighted_comparable
                    if weighted_comparable
                    else None
                ),
                "mean_top1_top2_margin": _mean_optional(
                    [row["mean_top1_top2_margin"] for row in group]
                ),
                "median_latency_seconds": median(
                    float(row["median_latency_seconds"]) for row in group
                ),
            }
        )
    return result


def _condition_summary(
    category_rows: list[dict[str, Any]],
    overall_rows: list[dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    model_ids = sorted({str(row["model_id"]) for row in overall_rows})
    for model_id in model_ids:
        result[model_id] = {}
        for arm in CALIBRATION_ARMS:
            overall = [
                row
                for row in overall_rows
                if row["model_id"] == model_id and row["arm"] == arm
            ]
            accuracies = [float(row["accuracy"]) for row in overall]
            categories: dict[str, Any] = {}
            for category in HARD_CATEGORIES:
                values = [
                    row
                    for row in category_rows
                    if row["model_id"] == model_id
                    and row["arm"] == arm
                    and row["category"] == category
                ]
                category_accuracies = [
                    float(row["accuracy"]) for row in values
                ]
                categories[category] = {
                    "accuracy": mean(category_accuracies),
                    "accuracy_ci95": _bootstrap_ci(
                        category_accuracies,
                        bootstrap_samples,
                    ),
                    "source_strict_accuracy": mean(
                        float(row["source_strict_accuracy"])
                        for row in values
                    ),
                    "tie_error_nonfinite_rate": mean(
                        float(row["tie_error_nonfinite_rate"])
                        for row in values
                    ),
                    "mean_top1_top2_margin": _mean_optional(
                        [row["mean_top1_top2_margin"] for row in values]
                    ),
                }
            comparable = sum(
                int(row["sum_mean_comparable_count"]) for row in overall
            )
            result[model_id][arm] = {
                "lesson_count": len(overall),
                "accuracy": mean(accuracies),
                "accuracy_ci95": _bootstrap_ci(
                    accuracies,
                    bootstrap_samples,
                ),
                "source_strict_accuracy": mean(
                    float(row["source_strict_accuracy"]) for row in overall
                ),
                "tie_rate": mean(float(row["tie_rate"]) for row in overall),
                "error_rate": mean(float(row["error_rate"]) for row in overall),
                "non_finite_rate": mean(
                    float(row["non_finite_rate"]) for row in overall
                ),
                "tie_error_nonfinite_rate": mean(
                    float(row["tie_error_nonfinite_rate"]) for row in overall
                ),
                "mean_score_tie_rate": mean(
                    float(row["mean_score_tie_rate"]) for row in overall
                ),
                "sum_mean_comparable_count": comparable,
                "sum_mean_disagreement_rate": (
                    sum(
                        int(row["sum_mean_comparable_count"])
                        * float(row["sum_mean_disagreement_rate"])
                        for row in overall
                        if row["sum_mean_disagreement_rate"] is not None
                    )
                    / comparable
                    if comparable
                    else None
                ),
                "mean_top1_top2_margin": _mean_optional(
                    [row["mean_top1_top2_margin"] for row in overall]
                ),
                "median_latency_seconds": median(
                    float(row["median_latency_seconds"]) for row in overall
                ),
                "categories": categories,
            }
    return result


def _paired_contrast(
    rows: list[dict[str, Any]],
    model_id: str,
    left: str,
    right: str,
    field: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    indexed = {
        (
            str(row["model_id"]),
            str(row["arm"]),
            str(row["lesson_id"]),
        ): row
        for row in rows
    }
    lessons = sorted(
        lesson_id
        for observed_model, arm, lesson_id in indexed
        if observed_model == model_id and arm == left
    )
    differences = [
        float(indexed[(model_id, left, lesson_id)][field])
        - float(indexed[(model_id, right, lesson_id)][field])
        for lesson_id in lessons
    ]
    return _contrast_payload(
        differences,
        left,
        right,
        bootstrap_samples,
    )


def _within_arm_contrast(
    rows: list[dict[str, Any]],
    model_id: str,
    arm: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["model_id"] == model_id and row["arm"] == arm
    ]
    differences = [
        float(row["accuracy"]) - float(row["source_strict_accuracy"])
        for row in selected
    ]
    return _contrast_payload(
        differences,
        "forced_choice",
        "source_strict_generation",
        bootstrap_samples,
    )


def _contrast_payload(
    differences: list[float],
    left: str,
    right: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    return {
        "left": left,
        "right": right,
        "lesson_count": len(differences),
        "mean": mean(differences),
        "ci95": _bootstrap_ci(differences, bootstrap_samples),
        "wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
    }


def _model_gate(
    model_id: str,
    summary: dict[str, Any],
    contrasts: dict[str, Any],
    audit_report: dict[str, Any],
) -> dict[str, Any]:
    oracle = summary["answer_copy_oracle"]
    external = summary["external"]
    no_write = summary["no_write"]
    error_rate = mean(
        float(summary[arm]["tie_error_nonfinite_rate"])
        for arm in CALIBRATION_ARMS
    )
    checks = {
        "prompt_provenance_token_audit": (
            audit_report.get("all_checks_passed") is True
            and audit_report.get("input_truncated_count") == 0
        ),
        "tie_error_nonfinite_rate": (
            error_rate <= MAX_TIE_ERROR_NONFINITE_RATE
        ),
        "oracle_overall_accuracy": (
            float(oracle["accuracy"]) >= ORACLE_MIN_ACCURACY
        ),
        "oracle_every_category_accuracy": all(
            float(values["accuracy"]) >= ORACLE_MIN_CATEGORY_ACCURACY
            for values in oracle["categories"].values()
        ),
        "external_overall_accuracy": (
            float(external["accuracy"]) >= EXTERNAL_MIN_ACCURACY
        ),
        "external_every_category_accuracy": all(
            float(values["accuracy"]) >= EXTERNAL_MIN_CATEGORY_ACCURACY
            for values in external["categories"].values()
        ),
        "no_write_overall_accuracy": (
            float(no_write["accuracy"]) <= NO_WRITE_MAX_ACCURACY
        ),
        "external_minus_no_write_ci95_lower": (
            float(contrasts["external_minus_no_write"]["ci95"][0]) > 0.0
        ),
    }
    conditional = float(
        external["categories"]["conditional_route"]["accuracy"]
    )
    if (
        float(external["accuracy"]) >= EXTERNAL_MIN_ACCURACY
        and conditional < EXTERNAL_MIN_CATEGORY_ACCURACY
    ):
        status = "category_calibration_failed"
    elif all(checks.values()):
        status = "training_complexity_review_eligible"
    elif not checks["tie_error_nonfinite_rate"]:
        status = "scoring_integrity_failed"
    elif not checks["oracle_overall_accuracy"] or not checks[
        "oracle_every_category_accuracy"
    ]:
        status = "oracle_failed"
    elif not checks["external_overall_accuracy"]:
        status = "external_calibration_failed"
    elif not checks["external_every_category_accuracy"]:
        status = "category_calibration_failed"
    elif not checks["no_write_overall_accuracy"]:
        status = "no_write_anchor_failed"
    elif not checks["external_minus_no_write_ci95_lower"]:
        status = "external_contrast_failed"
    else:
        status = "calibration_failed"
    return {
        "model_id": model_id,
        "status": status,
        "eligible": status == "training_complexity_review_eligible",
        "checks": checks,
        "observed_tie_error_nonfinite_rate": error_rate,
        "conditional_route_external_accuracy": conditional,
        "automatic_training_started": False,
        "automatic_narrow_scan_started": False,
    }


def _cross_scale_gate(
    summaries: dict[str, dict[str, Any]],
    model_gates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = model_gates["source_model"]
    canary = model_gates["scale_canary"]
    eligible = sorted(
        model_id
        for model_id, gate in model_gates.items()
        if gate["eligible"]
    )
    if not source["eligible"] and canary["eligible"]:
        status = "scale_capacity_bottleneck_supported"
    elif source["eligible"] and canary["eligible"]:
        status = "both_models_calibrated"
    elif source["eligible"] and not canary["eligible"]:
        status = "unexpected_inverse_scale_result"
    elif (
        float(summaries["scale_canary"]["external"]["accuracy"])
        > float(summaries["source_model"]["external"]["accuracy"])
    ):
        status = "scale_improvement_without_full_calibration"
    else:
        status = "shared_or_mixed_calibration_failure"
    if eligible:
        next_stage_status = "training_complexity_review_eligible"
        next_action = (
            "Human-review a separately frozen 1/4/8 mappings-per-adapter "
            "design using only an eligible model; do not start it automatically."
        )
    else:
        next_stage_status = "calibration_followup_required"
        next_action = (
            "Do not start training or a narrow scan. Review the failed "
            "overall, category, anchor, contrast, or scoring-integrity checks."
        )
    return {
        "status": status,
        "next_stage_status": next_stage_status,
        "eligible_model_ids": eligible,
        "automatic_training_started": False,
        "automatic_narrow_scan_started": False,
        "next_action": next_action,
    }


def _render_main_table(
    summaries: dict[str, dict[str, Any]],
    contrasts: dict[str, dict[str, Any]],
    gates: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# P0-D2H-CAL-FC Results",
        "",
        "| Model | Arm | Forced choice [95% CI] | Source strict | "
        "Margin | Sum/mean disagree | Tie/error/non-finite | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for model_id, arms in summaries.items():
        for arm, values in arms.items():
            disagreement = values["sum_mean_disagreement_rate"]
            lines.append(
                f"| {model_id} | {arm} | {values['accuracy']:.4f} "
                f"[{values['accuracy_ci95'][0]:.4f}, "
                f"{values['accuracy_ci95'][1]:.4f}] | "
                f"{values['source_strict_accuracy']:.4f} | "
                f"{_format_optional(values['mean_top1_top2_margin'])} | "
                f"{_format_optional(disagreement)} | "
                f"{values['tie_error_nonfinite_rate']:.4f} | "
                f"{gates[model_id]['status']} |"
            )
    lines.extend(
        [
            "",
            "## Category accuracy",
            "",
            "| Model | Arm | Category | Forced choice [95% CI] | "
            "Source strict | Tie/error/non-finite |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for model_id, arms in summaries.items():
        for arm, values in arms.items():
            for category, category_values in values["categories"].items():
                lines.append(
                    f"| {model_id} | {arm} | {category} | "
                    f"{category_values['accuracy']:.4f} "
                    f"[{category_values['accuracy_ci95'][0]:.4f}, "
                    f"{category_values['accuracy_ci95'][1]:.4f}] | "
                    f"{category_values['source_strict_accuracy']:.4f} | "
                    f"{category_values['tie_error_nonfinite_rate']:.4f} |"
                )
    lines.extend(
        [
            "",
            "## Paired contrasts",
            "",
            "| Model | Contrast | Difference [95% CI] |",
            "|---|---|---:|",
        ]
    )
    for model_id, values in contrasts.items():
        for name in ("external_minus_no_write", "oracle_minus_external"):
            contrast = values[name]
            lines.append(
                f"| {model_id} | {name} | {contrast['mean']:+.4f} "
                f"[{contrast['ci95'][0]:+.4f}, "
                f"{contrast['ci95'][1]:+.4f}] |"
            )
    lines.extend(
        [
            "",
            "Old strict-generation values are linked secondary metrics; they "
            "have not been redefined or used to alter the prior frozen gate.",
            "",
        ]
    )
    return "\n".join(lines)


def _mean_optional(values: list[Any]) -> float | None:
    observed = [float(value) for value in values if value is not None]
    return mean(observed) if observed else None


def _format_optional(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"
