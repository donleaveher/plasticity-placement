from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from plasticity_placement.p0d.analysis import _atomic_json_write, _bootstrap_ci
from plasticity_placement.p0d2hcrd.config import (
    COMBINED_MIN_ACCURACY,
    ENDPOINTS,
    MAX_TIE_ERROR_NONFINITE_RATE,
    RETRIEVAL_MIN_ACCURACY,
    ROUTE_MIN_ACCURACY,
    P0D2HCRDRequest,
    ResolvedP0D2HCRDConfig,
)
from plasticity_placement.p0d2hcrd.manifest import unit_key
from plasticity_placement.p0d2hcrd.runtime import (
    _file_hash,
    _load_bank_audit,
    _load_candidate_token_audit,
    _result_path,
    _verify_rows,
    resolve_request,
)

FACTOR_FIELDS = (
    "lesson_type",
    "lesson_side",
    "route_variant",
    "target_slot",
    "expected_candidate",
    "slot_candidate_order",
    "slot_content_order",
    "action_panel_position",
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
            f"missing P0-D2H-CRD manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config, rows, bank_audit, token_audit = _validate_and_load(
        output_dir,
        manifest,
    )
    lesson_endpoint = _lesson_endpoint_metrics(rows)
    endpoint_summary = _endpoint_summary(
        lesson_endpoint,
        bootstrap_samples,
    )
    factor_summary = _factor_summary(rows, bootstrap_samples)
    contrasts = _paired_contrasts(lesson_endpoint, bootstrap_samples)
    prediction_distributions = _prediction_distributions(rows)
    linked_source = _linked_source_summary(rows, bootstrap_samples)
    gate = diagnostic_gate(endpoint_summary, bank_audit, token_audit)
    summary = {
        "schema_version": "p0d2hcrd-summary-v1",
        "run_id": manifest["run_id"],
        "stage": "hard_probe_route_decomposition",
        "source_p0d2hfc_run_id": config.source_run_id,
        "model": config.model.to_dict(),
        "selected_lesson_count": len(config.selected_lesson_ids),
        "endpoints": list(ENDPOINTS),
        "unit_count": len(manifest["units"]),
        "decision_row_count": len(rows),
        "candidate_sequence_count": sum(
            len(row["candidates"]) for row in rows
        ),
        "decomposition_bank_audit": {
            key: value
            for key, value in bank_audit.items()
            if key != "records"
        },
        "candidate_token_audit": {
            key: value
            for key, value in token_audit.items()
            if key != "records"
        },
        "endpoint_summary": endpoint_summary,
        "factor_summary": factor_summary,
        "contrasts": contrasts,
        "prediction_distributions": prediction_distributions,
        "linked_source_conditional_route": linked_source,
        "diagnostic_gate": gate,
        "gates": {
            "run_valid": True,
            "diagnostic_status": gate["status"],
            "training_complexity_review_eligible": False,
            "automatic_training_started": False,
            "automatic_narrow_scan_started": False,
        },
        "lesson_endpoint_metrics": lesson_endpoint,
        "source_manifest_sha256": config.source_manifest_sha256,
        "source_summary_sha256": config.source_summary_sha256,
        "source_raw_results_sha256": config.source_raw_results_sha256,
        "source_calibration_manifest_sha256": (
            config.source_calibration_manifest_sha256
        ),
        "source_hard_probe_manifest_sha256": (
            config.source_hard_probe_manifest_sha256
        ),
        "hard_probe_hashes": config.hard_probe_hashes,
        "compiler_hashes": config.compiler_hashes,
        "decomposition_bank_sha256": config.decomposition_bank_sha256,
        "no_fabrication_status": (
            "All values derive from verified CRD rows linked to the exact "
            "immutable P0-D2H-CAL-FC source and its transitive source chain."
        ),
    }
    aggregate_dir = output_dir / "results" / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    summary_path = aggregate_dir / "summary.json"
    _atomic_json_write(summary_path, summary)
    (aggregate_dir / "main_table.md").write_text(
        _render_main_table(endpoint_summary, contrasts, gate, linked_source),
        encoding="utf-8",
    )
    _atomic_json_write(
        aggregate_dir / "next_stage_decision.json",
        {
            "schema_version": "p0d2hcrd-next-stage-v1",
            "source_run_id": manifest["run_id"],
            "status": gate["status"],
            "diagnostic_only": True,
            "training_complexity_review_eligible": False,
            "automatic_training_started": False,
            "automatic_narrow_scan_started": False,
            "instruction": gate["next_action"],
        },
    )
    return summary_path


def _validate_and_load(
    output_dir: Path,
    manifest: dict[str, Any],
) -> tuple[
    ResolvedP0D2HCRDConfig,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    required = {
        "run_id",
        "config",
        "source_manifest_path",
        "bank_audit_sha256",
        "candidate_token_audit_sha256",
        "selected_lessons",
        "model",
        "units",
    }
    if manifest.get("schema_version") != "p0d2hcrd-manifest-v1":
        raise ValueError("not a P0-D2H-CRD manifest")
    if not required.issubset(manifest):
        raise ValueError(
            f"route-decomposition manifest is missing "
            f"{sorted(required - manifest.keys())}"
        )
    if manifest.get("errors"):
        raise ValueError("route-decomposition manifest contains errors")
    request = P0D2HCRDRequest(
        output_dir=output_dir,
        source_manifest=Path(str(manifest["source_manifest_path"])),
    )
    config, bank, _, source_rows, _ = resolve_request(request)
    if manifest["config"] != config.identity_dict():
        raise ValueError("route-decomposition config/source identity changed")
    bank_hash, bank_audit = _load_bank_audit(config)
    if bank_hash != manifest["bank_audit_sha256"]:
        raise ValueError("route-decomposition bank audit changed")
    token_hash, audits, token_audit = _load_candidate_token_audit(
        config,
        bank_hash,
    )
    if token_hash != manifest["candidate_token_audit_sha256"]:
        raise ValueError("route-decomposition candidate-token audit changed")
    if (
        manifest["selected_lessons"] != list(config.selected_lesson_ids)
        or manifest["model"] != config.model.to_dict()
    ):
        raise ValueError("route-decomposition model/lesson matrix changed")
    expected_units = {
        unit_key(lesson_id) for lesson_id in config.selected_lesson_ids
    }
    if set(manifest["units"]) != expected_units:
        raise ValueError("route-decomposition unit matrix is incomplete")

    rows: list[dict[str, Any]] = []
    for lesson_id in config.selected_lesson_ids:
        key = unit_key(lesson_id)
        unit = manifest["units"][key]
        if unit.get("state") != "verified":
            raise ValueError(f"route-decomposition unit is not verified: {key}")
        path = _result_path(output_dir, lesson_id)
        if unit.get("result_sha256") != _file_hash(path):
            raise ValueError(
                f"route-decomposition raw file hash changed: {key}"
            )
        _verify_rows(
            config,
            manifest["run_id"],
            path,
            bank[lesson_id],
            source_rows,
            audits,
            expected_precision=str(unit["evaluation_precision"]),
        )
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    keys = [str(row["probe_id"]) for row in rows]
    if (
        len(rows) != config.expected_decision_row_count
        or len(keys) != len(set(keys))
        or sum(len(row["candidates"]) for row in rows)
        != config.expected_candidate_sequence_count
    ):
        raise ValueError(
            "route-decomposition complete matrix has missing or duplicate rows"
        )
    return config, rows, bank_audit, token_audit


def _lesson_endpoint_metrics(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["lesson_id"]), str(row["endpoint"]))].append(row)
    result: list[dict[str, Any]] = []
    for (lesson_id, endpoint), group in sorted(grouped.items()):
        expected_count = {"route_only": 16, "retrieval_only": 16, "combined": 32}[
            endpoint
        ]
        if len(group) != expected_count:
            raise ValueError(
                f"endpoint row count changed: {lesson_id}/{endpoint}"
            )
        first = group[0]
        comparable = [
            row
            for row in group
            if row["sum_mean_disagreement"] is not None
        ]
        margins = [
            float(row["top1_top2_margin"])
            for row in group
            if row["top1_top2_margin"] is not None
        ]
        result.append(
            {
                "lesson_id": lesson_id,
                "pair_id": first["pair_id"],
                "lesson_type": first["lesson_type"],
                "lesson_side": first["lesson_side"],
                "endpoint": endpoint,
                "row_count": len(group),
                "accuracy": mean(bool(row["correct"]) for row in group),
                "tie_rate": mean(bool(row["tie"]) for row in group),
                "error_rate": mean(
                    row["error_status"] not in {"ok", "tie", "non_finite"}
                    for row in group
                ),
                "non_finite_rate": mean(
                    bool(row["non_finite"]) for row in group
                ),
                "tie_error_nonfinite_rate": mean(
                    row["error_status"] != "ok" for row in group
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
    if len(result) != 72:
        raise ValueError("lesson×endpoint metric matrix changed")
    return result


def _endpoint_summary(
    lesson_rows: list[dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for endpoint in ENDPOINTS:
        group = [
            row for row in lesson_rows if row["endpoint"] == endpoint
        ]
        if len(group) != 24:
            raise ValueError(f"endpoint lesson count changed: {endpoint}")
        accuracies = [float(row["accuracy"]) for row in group]
        comparable = sum(
            int(row["sum_mean_comparable_count"]) for row in group
        )
        result[endpoint] = {
            "lesson_count": len(group),
            "decision_count": sum(int(row["row_count"]) for row in group),
            "accuracy": mean(accuracies),
            "accuracy_ci95": _bootstrap_ci(accuracies, bootstrap_samples),
            "tie_rate": mean(float(row["tie_rate"]) for row in group),
            "error_rate": mean(float(row["error_rate"]) for row in group),
            "non_finite_rate": mean(
                float(row["non_finite_rate"]) for row in group
            ),
            "tie_error_nonfinite_rate": mean(
                float(row["tie_error_nonfinite_rate"]) for row in group
            ),
            "sum_mean_comparable_count": comparable,
            "sum_mean_disagreement_rate": (
                sum(
                    int(row["sum_mean_comparable_count"])
                    * float(row["sum_mean_disagreement_rate"])
                    for row in group
                    if row["sum_mean_disagreement_rate"] is not None
                )
                / comparable
                if comparable
                else None
            ),
            "mean_top1_top2_margin": _mean_optional(
                [row["mean_top1_top2_margin"] for row in group]
            ),
            "median_latency_seconds": median(
                float(row["median_latency_seconds"]) for row in group
            ),
        }
    return result


def _factor_summary(
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for endpoint in ENDPOINTS:
        endpoint_rows = [row for row in rows if row["endpoint"] == endpoint]
        endpoint_result: list[dict[str, Any]] = []
        for field in FACTOR_FIELDS:
            values = sorted(
                {
                    row[field]
                    for row in endpoint_rows
                    if row.get(field) is not None
                },
                key=str,
            )
            for value in values:
                selected = [
                    row for row in endpoint_rows if row.get(field) == value
                ]
                by_lesson: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in selected:
                    by_lesson[str(row["lesson_id"])].append(row)
                lesson_accuracies = [
                    mean(bool(row["correct"]) for row in group)
                    for group in by_lesson.values()
                ]
                endpoint_result.append(
                    {
                        "factor": field,
                        "value": value,
                        "lesson_count": len(by_lesson),
                        "decision_count": len(selected),
                        "accuracy": mean(
                            bool(row["correct"]) for row in selected
                        ),
                        "lesson_clustered_accuracy_ci95": _bootstrap_ci(
                            lesson_accuracies,
                            bootstrap_samples,
                        ),
                        "tie_error_nonfinite_rate": mean(
                            row["error_status"] != "ok" for row in selected
                        ),
                    }
                )
        result[endpoint] = endpoint_result
    return result


def _paired_contrasts(
    lesson_rows: list[dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    indexed = {
        (str(row["lesson_id"]), str(row["endpoint"])): row
        for row in lesson_rows
    }
    lesson_ids = sorted({str(row["lesson_id"]) for row in lesson_rows})
    combined_minus_retrieval = [
        float(indexed[(lesson_id, "combined")]["accuracy"])
        - float(indexed[(lesson_id, "retrieval_only")]["accuracy"])
        for lesson_id in lesson_ids
    ]
    composition_gap = [
        min(
            float(indexed[(lesson_id, "route_only")]["accuracy"]),
            float(indexed[(lesson_id, "retrieval_only")]["accuracy"]),
        )
        - float(indexed[(lesson_id, "combined")]["accuracy"])
        for lesson_id in lesson_ids
    ]
    return {
        "combined_minus_retrieval_only": _contrast_payload(
            combined_minus_retrieval,
            "combined",
            "retrieval_only",
            bootstrap_samples,
        ),
        "composition_gap": _contrast_payload(
            composition_gap,
            "min(route_only,retrieval_only)",
            "combined",
            bootstrap_samples,
        ),
    }


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


def _prediction_distributions(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for endpoint in ENDPOINTS:
        selected = [row for row in rows if row["endpoint"] == endpoint]
        candidates = sorted(
            {
                str(candidate)
                for row in selected
                for candidate in row["ordered_candidates"]
            }
        )
        result[endpoint] = {
            "predicted_candidate_counts": {
                candidate: sum(
                    row["predicted_candidate"] == candidate for row in selected
                )
                for candidate in candidates
            },
            "predicted_panel_position_counts": {
                str(position): sum(
                    row["predicted_candidate"] is not None
                    and row["ordered_candidates"].index(
                        row["predicted_candidate"]
                    )
                    + 1
                    == position
                    for row in selected
                )
                for position in range(1, len(candidates) + 1)
            },
        }
    return result


def _linked_source_summary(
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(str(row["source_row_sha256"]), row)
    if len(unique) != 96:
        raise ValueError("linked source conditional-route matrix changed")
    by_lesson: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique.values():
        by_lesson[str(row["lesson_id"])].append(row)
    lesson_accuracy = [
        mean(bool(row["source_fc_correct"]) for row in group)
        for group in by_lesson.values()
    ]
    return {
        "source_decision_count": len(unique),
        "lesson_count": len(by_lesson),
        "accuracy": mean(lesson_accuracy),
        "accuracy_ci95": _bootstrap_ci(
            lesson_accuracy,
            bootstrap_samples,
        ),
        "role": "secondary_provenance-linked comparator only",
    }


def diagnostic_gate(
    endpoint_summary: dict[str, dict[str, Any]],
    bank_audit: dict[str, Any],
    token_audit: dict[str, Any],
) -> dict[str, Any]:
    route = endpoint_summary["route_only"]
    retrieval = endpoint_summary["retrieval_only"]
    combined = endpoint_summary["combined"]
    observed_integrity_rate = mean(
        float(endpoint_summary[endpoint]["tie_error_nonfinite_rate"])
        for endpoint in ENDPOINTS
    )
    checks = {
        "bank_audit": bank_audit.get("all_checks_passed") is True,
        "candidate_token_audit": (
            token_audit.get("all_checks_passed") is True
            and token_audit.get("input_truncated_count") == 0
        ),
        "tie_error_nonfinite_rate": (
            observed_integrity_rate <= MAX_TIE_ERROR_NONFINITE_RATE
        ),
        "route_only_accuracy": (
            float(route["accuracy"]) >= ROUTE_MIN_ACCURACY
        ),
        "retrieval_only_accuracy": (
            float(retrieval["accuracy"]) >= RETRIEVAL_MIN_ACCURACY
        ),
        "combined_accuracy": (
            float(combined["accuracy"]) >= COMBINED_MIN_ACCURACY
        ),
    }
    integrity = all(
        checks[name]
        for name in (
            "bank_audit",
            "candidate_token_audit",
            "tie_error_nonfinite_rate",
        )
    )
    route_pass = checks["route_only_accuracy"]
    retrieval_pass = checks["retrieval_only_accuracy"]
    combined_pass = checks["combined_accuracy"]
    if not integrity:
        status = "scoring_integrity_failed"
    elif not route_pass and retrieval_pass:
        status = "route_bottleneck_supported"
    elif route_pass and not retrieval_pass:
        status = "retrieval_bottleneck_supported"
    elif not route_pass and not retrieval_pass:
        status = "shared_component_failure"
    elif not combined_pass:
        status = "composition_bottleneck_supported"
    else:
        status = "no_component_bottleneck_detected"
    next_actions = {
        "scoring_integrity_failed": (
            "Do not interpret component accuracy. Repair the audit or scoring "
            "implementation and use a new immutable attempt."
        ),
        "route_bottleneck_supported": (
            "Record routing as the supported localization hypothesis. This "
            "diagnostic does not authorize training or a narrow scan."
        ),
        "retrieval_bottleneck_supported": (
            "Record retrieval as the supported localization hypothesis. This "
            "diagnostic does not authorize training or a narrow scan."
        ),
        "shared_component_failure": (
            "Both isolated components miss their floors. Preserve the negative "
            "diagnostic; do not start training automatically."
        ),
        "composition_bottleneck_supported": (
            "Both isolated components pass while composition fails. Preserve "
            "the composition diagnosis without starting training."
        ),
        "no_component_bottleneck_detected": (
            "All prospective floors pass. Treat the old deficit as not "
            "reproduced by this decomposition; no training is authorized."
        ),
    }
    return {
        "status": status,
        "diagnostic_only": True,
        "checks": checks,
        "observed_tie_error_nonfinite_rate": observed_integrity_rate,
        "thresholds": {
            "route_min_accuracy": ROUTE_MIN_ACCURACY,
            "retrieval_min_accuracy": RETRIEVAL_MIN_ACCURACY,
            "combined_min_accuracy": COMBINED_MIN_ACCURACY,
            "max_tie_error_nonfinite_rate": MAX_TIE_ERROR_NONFINITE_RATE,
        },
        "training_complexity_review_eligible": False,
        "automatic_training_started": False,
        "automatic_narrow_scan_started": False,
        "next_action": next_actions[status],
    }


def _render_main_table(
    endpoint_summary: dict[str, dict[str, Any]],
    contrasts: dict[str, Any],
    gate: dict[str, Any],
    linked_source: dict[str, Any],
) -> str:
    lines = [
        "# P0-D2H-CRD route/retrieval decomposition",
        "",
        "| Endpoint | Accuracy | 95% lesson-clustered CI | Invalid rate |",
        "|---|---:|---:|---:|",
    ]
    for endpoint in ENDPOINTS:
        values = endpoint_summary[endpoint]
        interval = values["accuracy_ci95"]
        lines.append(
            f"| `{endpoint}` | {values['accuracy']:.4f} | "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] | "
            f"{values['tie_error_nonfinite_rate']:.4f} |"
        )
    composition = contrasts["composition_gap"]
    lines.extend(
        [
            "",
            f"- Diagnostic status: `{gate['status']}`",
            f"- Composition gap: {composition['mean']:.4f} "
            f"[{composition['ci95'][0]:.4f}, "
            f"{composition['ci95'][1]:.4f}]",
            "- Linked old external conditional-route accuracy: "
            f"{linked_source['accuracy']:.4f}",
            "- Training-complexity review eligible: `false`",
            "- Automatic training started: `false`",
            "- Automatic narrow scan started: `false`",
            "",
        ]
    )
    return "\n".join(lines)


def _mean_optional(values: list[object]) -> float | None:
    selected = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
    ]
    return mean(selected) if selected else None
