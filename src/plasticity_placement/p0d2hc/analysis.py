from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from plasticity_placement.p0c.domain import Arm
from plasticity_placement.p0c.runtime import prepare_experiment
from plasticity_placement.p0d.analysis import (
    _atomic_json_write,
    _bootstrap_ci,
)
from plasticity_placement.p0d2h.probes import (
    HARD_CATEGORIES,
    load_hard_probe_bank,
    verify_hard_probe_hashes,
)
from plasticity_placement.p0d2hc.config import (
    CALIBRATION_ARMS,
    CANARY_MODEL_ID,
    SOURCE_MODEL_ID,
    CalibrationModel,
    ResolvedP0D2HCConfig,
)
from plasticity_placement.p0d2hc.manifest import unit_key
from plasticity_placement.p0d2hc.runtime import (
    _result_path,
    _verify_rows,
)


def aggregate_experiment(
    output_dir: Path,
    bootstrap_samples: int = 10_000,
) -> Path:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"missing P0-D2H-CAL manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config, rows = _validate_and_load(output_dir, manifest)
    prompt_audit = json.loads(
        (
            output_dir / "preflight" / "prompt_token_audit.json"
        ).read_text(encoding="utf-8")
    )
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
                Arm.EXTERNAL.value,
                Arm.NO_WRITE.value,
                bootstrap_samples,
            ),
            "oracle_minus_no_write": _paired_contrast(
                lesson_overall,
                model.model_id,
                Arm.ANSWER_COPY_ORACLE.value,
                Arm.NO_WRITE.value,
                bootstrap_samples,
            ),
            "oracle_minus_external": _paired_contrast(
                lesson_overall,
                model.model_id,
                Arm.ANSWER_COPY_ORACLE.value,
                Arm.EXTERNAL.value,
                bootstrap_samples,
            ),
        }
        for model in config.models
    }
    model_gates = {
        model.model_id: _model_gate(
            config,
            model,
            condition_summary[model.model_id],
        )
        for model in config.models
    }
    cross_scale = _cross_scale_gate(config, model_gates)
    summary = {
        "schema_version": "p0d2hc-summary-v1",
        "run_id": manifest["run_id"],
        "stage": "hard_probe_calibration",
        "source_p0d2h_run_id": config.source_run_id,
        "model_count": len(config.models),
        "models": [model.to_dict() for model in config.models],
        "selected_lesson_count": len(config.selected_lesson_ids),
        "hard_categories": list(HARD_CATEGORIES),
        "calibration_arms": list(CALIBRATION_ARMS),
        "unit_count": len(manifest["units"]),
        "probe_row_count": len(rows),
        "prompt_token_audit": {
            key: value
            for key, value in prompt_audit.items()
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
        "hard_probe_hashes": config.hard_probe_hashes,
        "no_fabrication_status": (
            "All values were aggregated from verified base-only calibration "
            "rows bound to the hash-frozen P0-D2H-R probe bank."
        ),
    }
    aggregate_dir = output_dir / "results" / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    summary_path = aggregate_dir / "summary.json"
    _atomic_json_write(summary_path, summary)
    (aggregate_dir / "main_table.md").write_text(
        _render_main_table(condition_summary, model_gates),
        encoding="utf-8",
    )
    _atomic_json_write(
        aggregate_dir / "next_stage_decision.json",
        {
            "schema_version": "p0d2hc-next-stage-v1",
            "source_run_id": manifest["run_id"],
            "status": cross_scale["next_stage_status"],
            "cross_scale_status": cross_scale["status"],
            "eligible_model_ids": cross_scale["eligible_model_ids"],
            "automatic_training_started": False,
            "automatic_narrow_scan_started": False,
            "instruction": cross_scale["next_action"],
        },
    )
    return summary_path


def _validate_and_load(
    output_dir: Path,
    manifest: dict[str, Any],
) -> tuple[ResolvedP0D2HCConfig, list[dict[str, Any]]]:
    if manifest.get("schema_version") != "p0d2hc-manifest-v1":
        raise ValueError("not a P0-D2H-CAL manifest")
    required = {
        "run_id",
        "config",
        "source_manifest_path",
        "prompt_token_audit_sha256",
        "selected_lessons",
        "models",
        "units",
    }
    if not required.issubset(manifest):
        raise ValueError(
            f"calibration manifest is missing "
            f"{sorted(required - manifest.keys())}"
        )
    if manifest.get("errors"):
        raise ValueError("calibration manifest contains recorded errors")
    raw_config = manifest["config"]
    if (
        raw_config.get("schema_version") != "p0d2hc-config-v1"
        or raw_config.get("stage") != "hard_probe_calibration"
    ):
        raise ValueError("calibration config schema/stage mismatch")

    source_path = Path(str(manifest["source_manifest_path"]))
    source_bytes = source_path.read_bytes()
    if _sha256(source_bytes) != raw_config["source_manifest_sha256"]:
        raise ValueError("P0-D2H-R source manifest changed")
    source = json.loads(source_bytes)
    source_dir = source_path.parent
    summary_path = source_dir / "results" / "aggregate" / "summary.json"
    summary_bytes = summary_path.read_bytes()
    if _sha256(summary_bytes) != raw_config["source_summary_sha256"]:
        raise ValueError("P0-D2H-R source summary changed")
    source_summary = json.loads(summary_bytes)
    if (
        source.get("run_id") != raw_config["source_run_id"]
        or source_summary.get("run_id") != source.get("run_id")
        or source_summary.get("gates", {}).get("run_valid") is not True
    ):
        raise ValueError("P0-D2H-R source provenance mismatch")

    source_p0d2_path = Path(str(source["source_manifest_path"]))
    source_p0d2 = json.loads(source_p0d2_path.read_text(encoding="utf-8"))
    source_p0d2_dir = source_p0d2_path.parent
    if not (source_p0d2_dir / "compiled" / "hashes.json").exists():
        raise FileNotFoundError(
            "P0-D2 source compiled hashes are missing; refusing to write "
            "inside the read-only source"
        )
    if (
        source_p0d2.get("run_id") != raw_config["source_p0d2_run_id"]
        or prepare_experiment(source_p0d2_dir)
        != raw_config["source_compiler_hashes"]
    ):
        raise ValueError("P0-D2 compiled source changed")
    verify_hard_probe_hashes(source_dir, raw_config["hard_probe_hashes"])
    hard_bank = load_hard_probe_bank(source_dir)

    audit_path = output_dir / "preflight" / "prompt_token_audit.json"
    audit_bytes = audit_path.read_bytes()
    if _sha256(audit_bytes) != manifest["prompt_token_audit_sha256"]:
        raise ValueError("calibration prompt-token audit changed")
    audit_report = json.loads(audit_bytes)
    if (
        audit_report.get("schema_version")
        != "p0d2hc-prompt-token-audit-v1"
        or audit_report.get("all_prompts_fit") is not True
        or int(audit_report.get("input_truncated_count", -1)) != 0
        or int(audit_report.get("output_instruction_missing_count", -1))
        != 0
    ):
        raise ValueError("calibration prompt-token audit is invalid")
    prompt_audits = {
        (
            str(record["model_id"]),
            str(record["lesson_id"]),
            str(record["probe_id"]),
            str(record["prompt_variant"]),
        ): record
        for record in audit_report["records"]
    }
    if len(prompt_audits) != int(audit_report["prompt_count"]):
        raise ValueError("calibration prompt audit contains duplicates")

    models = tuple(
        CalibrationModel(
            model_id=str(model["model_id"]),
            role=str(model["role"]),
            model_name=str(model["model_name"]),
            model_revision=str(model["model_revision"]),
            use_4bit=bool(model["use_4bit"]),
        )
        for model in raw_config["models"]
    )
    config = ResolvedP0D2HCConfig(
        output_dir=output_dir,
        source_output_dir=source_dir,
        source_p0d2_output_dir=source_p0d2_dir,
        code_sha256=str(raw_config["code_sha256"]),
        source_manifest_sha256=str(
            raw_config["source_manifest_sha256"]
        ),
        source_run_id=str(raw_config["source_run_id"]),
        source_summary_sha256=str(raw_config["source_summary_sha256"]),
        source_p0d2_run_id=str(raw_config["source_p0d2_run_id"]),
        source_compiler_hashes={
            str(key): str(value)
            for key, value in raw_config["source_compiler_hashes"].items()
        },
        hard_probe_hashes={
            str(key): str(value)
            for key, value in raw_config["hard_probe_hashes"].items()
        },
        selected_lesson_ids=tuple(
            str(value) for value in raw_config["selected_lesson_ids"]
        ),
        models=models,
        evaluation_max_length=int(raw_config["evaluation_max_length"]),
        max_new_tokens=int(raw_config["max_new_tokens"]),
        oracle_min_accuracy=float(raw_config["oracle_min_accuracy"]),
        oracle_min_category_accuracy=float(
            raw_config["oracle_min_category_accuracy"]
        ),
        oracle_max_invalid_rate=float(
            raw_config["oracle_max_invalid_rate"]
        ),
        external_min_accuracy=float(raw_config["external_min_accuracy"]),
        external_max_invalid_rate=float(
            raw_config["external_max_invalid_rate"]
        ),
    )
    if (
        list(config.selected_lesson_ids) != manifest["selected_lessons"]
        or {
            model.model_id: model.to_dict() for model in config.models
        }
        != manifest["models"]
        or tuple(hard_bank) != config.selected_lesson_ids
    ):
        raise ValueError("calibration frozen matrix changed")
    expected_units = {
        unit_key(model.model_id, lesson_id)
        for model in config.models
        for lesson_id in config.selected_lesson_ids
    }
    if set(manifest["units"]) != expected_units:
        raise ValueError("calibration unit matrix is incomplete")

    rows: list[dict[str, Any]] = []
    for model in config.models:
        for lesson_id in config.selected_lesson_ids:
            key = unit_key(model.model_id, lesson_id)
            unit = manifest["units"][key]
            if unit.get("state") != "verified":
                raise ValueError(f"calibration unit is not verified: {key}")
            path = _result_path(output_dir, model.model_id, lesson_id)
            _verify_rows(
                config,
                model,
                manifest["run_id"],
                path,
                hard_bank[lesson_id],
                lesson_id,
                prompt_audits,
                expected_precision=str(unit["evaluation_precision"]),
            )
            rows.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    if len(rows) != config.expected_probe_row_count:
        raise ValueError(
            f"calibration row count mismatch: "
            f"{len(rows)} != {config.expected_probe_row_count}"
        )
    return config, rows


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
                str(row["calibration_model_id"]),
                str(row["arm"]),
                str(row["lesson_id"]),
                str(row["category"]),
            )
        ].append(row)
    result: list[dict[str, Any]] = []
    for (model_id, arm, lesson_id, category), group in sorted(
        groups.items()
    ):
        first = group[0]
        result.append(
            {
                "model_id": model_id,
                "arm": arm,
                "lesson_id": lesson_id,
                "pair_id": first["pair_id"],
                "lesson_type": first["lesson_type"],
                "category": category,
                "accuracy": mean(
                    bool(row["correct"]) for row in group
                ),
                "invalid_rate": mean(
                    bool(row["invalid"]) for row in group
                ),
                "mean_input_tokens": mean(
                    float(row["input_tokens"]) for row in group
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
    groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
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
        if {row["category"] for row in group} != set(HARD_CATEGORIES):
            raise ValueError(
                f"missing calibration category: {model_id}/{arm}/{lesson_id}"
            )
        first = group[0]
        result.append(
            {
                "model_id": model_id,
                "arm": arm,
                "lesson_id": lesson_id,
                "pair_id": first["pair_id"],
                "lesson_type": first["lesson_type"],
                "accuracy": mean(float(row["accuracy"]) for row in group),
                "invalid_rate": mean(
                    float(row["invalid_rate"]) for row in group
                ),
                "mean_input_tokens": mean(
                    float(row["mean_input_tokens"]) for row in group
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
            categories = {}
            for category in HARD_CATEGORIES:
                values = [
                    float(row["accuracy"])
                    for row in category_rows
                    if row["model_id"] == model_id
                    and row["arm"] == arm
                    and row["category"] == category
                ]
                categories[category] = {
                    "accuracy": mean(values),
                    "accuracy_ci95": _bootstrap_ci(
                        values,
                        bootstrap_samples,
                    ),
                }
            result[model_id][arm] = {
                "lesson_count": len(overall),
                "accuracy": mean(accuracies),
                "accuracy_ci95": _bootstrap_ci(
                    accuracies,
                    bootstrap_samples,
                ),
                "invalid_rate": mean(
                    float(row["invalid_rate"]) for row in overall
                ),
                "mean_input_tokens": mean(
                    float(row["mean_input_tokens"]) for row in overall
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
    lesson_ids = sorted(
        {
            lesson_id
            for observed_model, arm, lesson_id in indexed
            if observed_model == model_id and arm == left
        }
    )
    differences = [
        float(indexed[(model_id, left, lesson_id)]["accuracy"])
        - float(indexed[(model_id, right, lesson_id)]["accuracy"])
        for lesson_id in lesson_ids
    ]
    return {
        "left": left,
        "right": right,
        "lesson_count": len(lesson_ids),
        "mean": mean(differences),
        "ci95": _bootstrap_ci(differences, bootstrap_samples),
        "wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "losses": sum(value < 0 for value in differences),
    }


def _model_gate(
    config: ResolvedP0D2HCConfig,
    model: CalibrationModel,
    summary: dict[str, Any],
) -> dict[str, Any]:
    oracle = summary[Arm.ANSWER_COPY_ORACLE.value]
    external = summary[Arm.EXTERNAL.value]
    oracle_checks = {
        "overall_accuracy": (
            float(oracle["accuracy"]) >= config.oracle_min_accuracy
        ),
        "every_category_accuracy": all(
            float(values["accuracy"])
            >= config.oracle_min_category_accuracy
            for values in oracle["categories"].values()
        ),
        "invalid_rate": (
            float(oracle["invalid_rate"])
            <= config.oracle_max_invalid_rate
        ),
    }
    external_checks = {
        "overall_accuracy": (
            float(external["accuracy"]) >= config.external_min_accuracy
        ),
        "invalid_rate": (
            float(external["invalid_rate"])
            <= config.external_max_invalid_rate
        ),
    }
    oracle_pass = all(oracle_checks.values())
    external_pass = all(external_checks.values())
    if not oracle_pass:
        status = "oracle_failed"
        interpretation = (
            "The model cannot reliably copy an explicit answer under the "
            "hard-probe interface."
        )
    elif not external_pass:
        status = "oracle_pass_external_failed"
        interpretation = (
            "Output copying works, but verified-memory use under distractors "
            "does not meet the frozen anchor threshold."
        )
    else:
        status = "calibrated"
        interpretation = (
            "Both answer-copy and verified-memory anchors meet their frozen "
            "thresholds."
        )
    return {
        "model_id": model.model_id,
        "model_name": model.model_name,
        "model_revision": model.model_revision,
        "status": status,
        "oracle_pass": oracle_pass,
        "external_pass": external_pass,
        "oracle_checks": oracle_checks,
        "external_checks": external_checks,
        "interpretation": interpretation,
    }


def _cross_scale_gate(
    config: ResolvedP0D2HCConfig,
    model_gates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = model_gates[SOURCE_MODEL_ID]
    canary = model_gates.get(CANARY_MODEL_ID)
    eligible = sorted(
        model_id
        for model_id, gate in model_gates.items()
        if gate["status"] == "calibrated"
    )
    if canary is None:
        status = source["status"]
    elif (
        source["status"] != "calibrated"
        and canary["status"] == "calibrated"
    ):
        status = "scale_bottleneck_supported"
    elif (
        source["status"] == "calibrated"
        and canary["status"] == "calibrated"
    ):
        status = "both_models_calibrated"
    elif (
        source["status"] == "oracle_failed"
        and canary["status"] == "oracle_failed"
    ):
        status = "shared_interface_or_output_bottleneck"
    elif (
        source["status"] == "oracle_pass_external_failed"
        and canary["status"] == "oracle_pass_external_failed"
    ):
        status = "shared_verified_memory_bottleneck"
    else:
        status = "mixed_scale_result"
    if eligible:
        next_stage_status = "training_complexity_design_eligible"
        next_action = (
            "Freeze a separate 1/4/8 mappings-per-adapter experiment using "
            "an eligible calibrated model; do not start it automatically."
        )
    else:
        next_stage_status = "calibration_followup_required"
        next_action = (
            "Do not start multi-mapping training or a narrow layer scan. "
            "Review oracle/external failures or test another frozen model scale."
        )
    return {
        "status": status,
        "next_stage_status": next_stage_status,
        "eligible_model_ids": eligible,
        "automatic_training_started": False,
        "automatic_narrow_scan_started": False,
        "next_action": next_action,
        "model_count": len(config.models),
    }


def _render_main_table(
    summaries: dict[str, dict[str, Any]],
    gates: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "| Model | Arm | Accuracy [95% CI] | Invalid | Gate status |",
        "|---|---|---:|---:|---|",
    ]
    for model_id, arms in summaries.items():
        for arm, values in arms.items():
            lines.append(
                f"| {model_id} | {arm} | {values['accuracy']:.4f} "
                f"[{values['accuracy_ci95'][0]:.4f}, "
                f"{values['accuracy_ci95'][1]:.4f}] | "
                f"{values['invalid_rate']:.4f} | "
                f"{gates[model_id]['status']} |"
            )
    return "\n".join(lines) + "\n"


def _sha256(value: bytes) -> str:
    from hashlib import sha256

    return sha256(value).hexdigest()
