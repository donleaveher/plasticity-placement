from __future__ import annotations

import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any

from plasticity_placement.p0c.runtime import current_code_hash
from plasticity_placement.p0d.analysis import _bootstrap_ci
from plasticity_placement.p0d2hcrd.config import (
    COMBINED_MIN_ACCURACY,
    ENDPOINTS,
    RETRIEVAL_MIN_ACCURACY,
    ROUTE_MIN_ACCURACY,
)
from plasticity_placement.p0d2hcrd.integrity_audit import (
    load_verified_rows,
    raw_tree_hash,
    row_evidence,
)

CONSERVATIVE_ANALYSIS_VERSION = "p0d2hcrd-conservative-tie-analysis-v1"


def aggregate_conservative_ties(
    source_output_dir: Path,
    analysis_output_dir: Path,
    bootstrap_samples: int = 10_000,
) -> Path:
    """Apply an immutable lower/upper tie analysis outside a completed run."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    _require_independent_output(source_output_dir, analysis_output_dir)
    paths = {
        "manifest": source_output_dir / "manifest.json",
        "summary": (source_output_dir / "results" / "aggregate" / "summary.json"),
        "bank_audit": (source_output_dir / "preflight" / "decomposition_bank.json"),
        "token_audit": (source_output_dir / "preflight" / "candidate_token_audit.json"),
    }
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    manifest = json.loads(payloads["manifest"])
    source_summary = json.loads(payloads["summary"])
    bank_audit = json.loads(payloads["bank_audit"])
    token_audit = json.loads(payloads["token_audit"])
    rows, raw_sha256 = load_verified_rows(source_output_dir, manifest)
    _validate_source_artifacts(
        manifest,
        source_summary,
        bank_audit,
        token_audit,
        payloads,
        rows,
    )
    analysis = build_conservative_tie_analysis(
        rows,
        bank_audit,
        token_audit,
        bootstrap_samples=bootstrap_samples,
    )
    _verify_source_unchanged(
        source_output_dir,
        paths,
        payloads,
        raw_sha256,
    )

    source_identity = {
        "source_output_dir": str(source_output_dir),
        "source_run_id": manifest["run_id"],
        "source_manifest_sha256": sha256(payloads["manifest"]).hexdigest(),
        "source_summary_sha256": sha256(payloads["summary"]).hexdigest(),
        "source_bank_audit_sha256": sha256(payloads["bank_audit"]).hexdigest(),
        "source_candidate_token_audit_sha256": sha256(payloads["token_audit"]).hexdigest(),
        "source_raw_results_sha256": raw_sha256,
        "source_experiment_code_sha256": manifest["config"]["code_sha256"],
        "source_v1_diagnostic_status": source_summary["diagnostic_gate"]["status"],
    }
    identity = {
        "schema_version": CONSERVATIVE_ANALYSIS_VERSION,
        "analysis_code_sha256": current_code_hash(),
        "source": source_identity,
        "tie_policy": {
            "name": "expected-in-exact-top-set-interval-v1",
            "lower_bound": "all finite exact ties counted incorrect",
            "upper_bound": (
                "finite exact tie counted compatible only when the expected "
                "candidate belongs to the exact top-score set"
            ),
            "arbitrary_tie_break_used": False,
            "finite_structural_ties_are_fatal": False,
            "non_finite_or_structural_errors_are_fatal": True,
            "robust_status_requires_matching_bounds": True,
        },
        "bootstrap_samples": bootstrap_samples,
    }
    run_id = "p0d2hcrd-conservative-ties-" + _json_hash(identity)[:10]
    result = {
        **identity,
        "run_id": run_id,
        "analysis_status": "supplementary_tie_policy_analysis",
        "source_gate_status_changed": False,
        "source_next_stage_eligibility_changed": False,
        **analysis,
        "artifact_paths": {
            "summary_json": str(analysis_output_dir / "conservative_tie_summary.json"),
            "report_markdown": str(analysis_output_dir / "conservative_tie_report.md"),
            "next_stage_decision": str(analysis_output_dir / "next_stage_decision.json"),
        },
    }
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = analysis_output_dir / "conservative_tie_summary.json"
    _write_immutable(
        summary_path,
        (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    _write_immutable(
        analysis_output_dir / "conservative_tie_report.md",
        _render_markdown(result).encode(),
    )
    _write_immutable(
        analysis_output_dir / "next_stage_decision.json",
        (
            json.dumps(
                {
                    "schema_version": ("p0d2hcrd-conservative-next-stage-v1"),
                    "source_run_id": manifest["run_id"],
                    "analysis_run_id": run_id,
                    "status": result["diagnostic_gate"]["status"],
                    "bound_statuses": result["diagnostic_gate"]["bound_statuses"],
                    "diagnostic_only": True,
                    "training_complexity_review_eligible": False,
                    "automatic_training_started": False,
                    "automatic_narrow_scan_started": False,
                    "instruction": result["diagnostic_gate"]["next_action"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )
    return summary_path


def build_conservative_tie_analysis(
    rows: list[dict[str, Any]],
    bank_audit: dict[str, Any],
    token_audit: dict[str, Any],
    *,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Build conservative endpoint bounds and a bound-robust diagnosis."""
    if not rows:
        raise ValueError("cannot analyze an empty CRD result set")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    evidence = [row_evidence(row) for row in rows]
    structural_checks_passed = all(
        record["checks"]["all_structural_checks_passed"] for record in evidence
    )
    fatal_records = [
        record
        for record in evidence
        if record["error_status"] not in {"ok", "tie"}
        or (
            record["error_status"] == "tie"
            and record["tie_origin"] != "distinct_candidates_exact_score_tie"
        )
    ]
    tie_records = [record for record in evidence if record["error_status"] == "tie"]
    tie_compatible_probe_ids = {
        str(record["probe_id"])
        for record in tie_records
        if str(record["expected_candidate"])
        in {str(candidate["candidate"]) for candidate in record["top_sum_candidates"]}
    }
    lesson_endpoint = _lesson_endpoint_bounds(
        rows,
        tie_compatible_probe_ids,
    )
    endpoint_summary = _endpoint_bounds(
        lesson_endpoint,
        bootstrap_samples,
    )
    gate = conservative_diagnostic_gate(
        endpoint_summary,
        bank_audit,
        token_audit,
        structural_checks_passed=structural_checks_passed,
        fatal_error_count=len(fatal_records),
    )
    return {
        "endpoint_summary": endpoint_summary,
        "lesson_endpoint_metrics": lesson_endpoint,
        "tie_summary": {
            "finite_exact_tie_count": len(tie_records),
            "expected_in_top_set_count": len(tie_compatible_probe_ids),
            "expected_absent_from_top_set_count": (
                len(tie_records) - len(tie_compatible_probe_ids)
            ),
            "fatal_error_count": len(fatal_records),
            "structural_checks_passed": structural_checks_passed,
            "records": [
                {
                    "probe_id": record["probe_id"],
                    "lesson_id": record["lesson_id"],
                    "endpoint": record["endpoint"],
                    "expected_candidate": record["expected_candidate"],
                    "top_sum_candidates": [
                        candidate["candidate"] for candidate in record["top_sum_candidates"]
                    ],
                    "expected_in_top_set": str(record["expected_candidate"])
                    in {str(candidate["candidate"]) for candidate in record["top_sum_candidates"]},
                }
                for record in tie_records
            ],
        },
        "diagnostic_gate": gate,
        "gates": {
            "run_valid": gate["integrity_passed"],
            "diagnostic_status": gate["status"],
            "training_complexity_review_eligible": False,
            "automatic_training_started": False,
            "automatic_narrow_scan_started": False,
        },
    }


def conservative_diagnostic_gate(
    endpoint_summary: dict[str, dict[str, Any]],
    bank_audit: dict[str, Any],
    token_audit: dict[str, Any],
    *,
    structural_checks_passed: bool,
    fatal_error_count: int,
) -> dict[str, Any]:
    """Require lower and upper tie bounds to support the same diagnosis."""
    checks = {
        "bank_audit": bank_audit.get("all_checks_passed") is True,
        "candidate_token_audit": (
            token_audit.get("all_checks_passed") is True
            and token_audit.get("input_truncated_count") == 0
        ),
        "structural_checks": structural_checks_passed,
        "fatal_error_count": fatal_error_count == 0,
        "finite_exact_ties_allowed": True,
    }
    integrity = all(
        checks[name]
        for name in (
            "bank_audit",
            "candidate_token_audit",
            "structural_checks",
            "fatal_error_count",
        )
    )
    lower_status = _component_status(
        {endpoint: float(values["accuracy_lower"]) for endpoint, values in endpoint_summary.items()}
    )
    upper_status = _component_status(
        {endpoint: float(values["accuracy_upper"]) for endpoint, values in endpoint_summary.items()}
    )
    if not integrity:
        status = "scoring_integrity_failed"
    elif lower_status == upper_status:
        status = lower_status
    else:
        status = "tie_interval_ambiguous"
    next_actions = {
        "scoring_integrity_failed": (
            "Do not interpret component bounds. Repair the fatal audit or "
            "source-integrity failure in a new immutable attempt."
        ),
        "tie_interval_ambiguous": (
            "The lower and upper tie bounds imply different diagnoses. "
            "Preserve the ambiguity; do not tie-break or start training."
        ),
        "route_bottleneck_supported": (
            "Both conservative bounds support routing as the localization "
            "hypothesis. This supplementary analysis does not authorize "
            "training or a narrow scan."
        ),
        "retrieval_bottleneck_supported": (
            "Both conservative bounds support retrieval as the localization "
            "hypothesis. This supplementary analysis does not authorize "
            "training or a narrow scan."
        ),
        "shared_component_failure": (
            "Both conservative bounds show route and retrieval failures. "
            "Preserve the negative diagnostic without starting training."
        ),
        "composition_bottleneck_supported": (
            "Both conservative bounds support a composition bottleneck. "
            "Preserve the diagnosis without starting training."
        ),
        "no_component_bottleneck_detected": (
            "Both conservative bounds pass all component floors. No training "
            "is authorized by this supplementary analysis."
        ),
    }
    return {
        "status": status,
        "diagnostic_only": True,
        "integrity_passed": integrity,
        "checks": checks,
        "fatal_error_count": fatal_error_count,
        "bound_statuses": {
            "lower": lower_status,
            "upper": upper_status,
            "match": lower_status == upper_status,
        },
        "thresholds": {
            "route_min_accuracy": ROUTE_MIN_ACCURACY,
            "retrieval_min_accuracy": RETRIEVAL_MIN_ACCURACY,
            "combined_min_accuracy": COMBINED_MIN_ACCURACY,
        },
        "training_complexity_review_eligible": False,
        "automatic_training_started": False,
        "automatic_narrow_scan_started": False,
        "next_action": next_actions[status],
    }


def _lesson_endpoint_bounds(
    rows: list[dict[str, Any]],
    tie_compatible_probe_ids: set[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["lesson_id"]), str(row["endpoint"]))].append(row)
    metrics: list[dict[str, Any]] = []
    for (lesson_id, endpoint), group in sorted(grouped.items()):
        expected_count = {
            "route_only": 16,
            "retrieval_only": 16,
            "combined": 32,
        }[endpoint]
        if len(group) != expected_count:
            raise ValueError(f"endpoint row count changed: {lesson_id}/{endpoint}")
        lower_values = [bool(row["correct"]) for row in group]
        upper_values = [
            bool(row["correct"]) or str(row["probe_id"]) in tie_compatible_probe_ids
            for row in group
        ]
        metrics.append(
            {
                "lesson_id": lesson_id,
                "endpoint": endpoint,
                "row_count": len(group),
                "accuracy_lower": mean(lower_values),
                "accuracy_upper": mean(upper_values),
                "finite_exact_tie_count": sum(row["error_status"] == "tie" for row in group),
                "tie_compatible_count": sum(
                    str(row["probe_id"]) in tie_compatible_probe_ids for row in group
                ),
                "fatal_error_count": sum(row["error_status"] not in {"ok", "tie"} for row in group),
            }
        )
    if len(metrics) != 72:
        raise ValueError("lesson×endpoint metric matrix changed")
    return metrics


def _endpoint_bounds(
    lesson_rows: list[dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for endpoint in ENDPOINTS:
        group = [row for row in lesson_rows if row["endpoint"] == endpoint]
        if len(group) != 24:
            raise ValueError(f"endpoint lesson count changed: {endpoint}")
        lower = [float(row["accuracy_lower"]) for row in group]
        upper = [float(row["accuracy_upper"]) for row in group]
        result[endpoint] = {
            "lesson_count": len(group),
            "decision_count": sum(int(row["row_count"]) for row in group),
            "accuracy_lower": mean(lower),
            "accuracy_lower_ci95": _bootstrap_ci(
                lower,
                bootstrap_samples,
            ),
            "accuracy_upper": mean(upper),
            "accuracy_upper_ci95": _bootstrap_ci(
                upper,
                bootstrap_samples,
            ),
            "finite_exact_tie_count": sum(int(row["finite_exact_tie_count"]) for row in group),
            "tie_compatible_count": sum(int(row["tie_compatible_count"]) for row in group),
            "fatal_error_count": sum(int(row["fatal_error_count"]) for row in group),
        }
    return result


def _component_status(accuracies: dict[str, float]) -> str:
    route_pass = accuracies["route_only"] >= ROUTE_MIN_ACCURACY
    retrieval_pass = accuracies["retrieval_only"] >= RETRIEVAL_MIN_ACCURACY
    combined_pass = accuracies["combined"] >= COMBINED_MIN_ACCURACY
    if not route_pass and retrieval_pass:
        return "route_bottleneck_supported"
    if route_pass and not retrieval_pass:
        return "retrieval_bottleneck_supported"
    if not route_pass and not retrieval_pass:
        return "shared_component_failure"
    if not combined_pass:
        return "composition_bottleneck_supported"
    return "no_component_bottleneck_detected"


def _validate_source_artifacts(
    manifest: dict[str, Any],
    summary: dict[str, Any],
    bank_audit: dict[str, Any],
    token_audit: dict[str, Any],
    payloads: dict[str, bytes],
    rows: list[dict[str, Any]],
) -> None:
    bank_sha256 = sha256(payloads["bank_audit"]).hexdigest()
    token_sha256 = sha256(payloads["token_audit"]).hexdigest()
    if (
        summary.get("schema_version") != "p0d2hcrd-summary-v1"
        or summary.get("run_id") != manifest.get("run_id")
        or summary.get("decision_row_count") != len(rows)
        or summary.get("candidate_sequence_count") != 5_376
        or bank_sha256 != manifest.get("bank_audit_sha256")
        or token_sha256 != manifest.get("candidate_token_audit_sha256")
        or bank_audit.get("schema_version") != "p0d2hcrd-bank-audit-v1"
        or bank_audit.get("decision_count") != 1_536
        or bank_audit.get("candidate_count") != 5_376
        or bank_audit.get("all_checks_passed") is not True
        or token_audit.get("schema_version") != "p0d2hcrd-candidate-token-audit-v1"
        or token_audit.get("decision_count") != 1_536
        or token_audit.get("candidate_count") != 5_376
        or token_audit.get("failed_decision_count") != 0
        or token_audit.get("input_truncated_count") != 0
        or token_audit.get("all_checks_passed") is not True
    ):
        raise ValueError("conservative tie analysis source audits are incomplete or changed")


def _verify_source_unchanged(
    source_output_dir: Path,
    paths: dict[str, Path],
    payloads: dict[str, bytes],
    raw_sha256: str,
) -> None:
    if any(paths[name].read_bytes() != payload for name, payload in payloads.items()):
        raise RuntimeError("source CRD metadata changed during conservative tie analysis")
    if raw_tree_hash(source_output_dir) != raw_sha256:
        raise RuntimeError("source CRD raw rows changed during conservative tie analysis")


def _require_independent_output(source: Path, output: Path) -> None:
    source_resolved = source.resolve()
    output_resolved = output.resolve()
    if (
        source_resolved == output_resolved
        or output_resolved.is_relative_to(source_resolved)
        or source_resolved.is_relative_to(output_resolved)
    ):
        raise ValueError(
            "conservative tie analysis output must be independent from the source CRD run"
        )


def _render_markdown(payload: dict[str, Any]) -> str:
    gate = payload["diagnostic_gate"]
    lines = [
        "# P0-D2H-CRD conservative exact-tie analysis",
        "",
        "| Endpoint | Lower accuracy | 95% CI | Upper accuracy | 95% CI | Finite ties |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for endpoint in ENDPOINTS:
        values = payload["endpoint_summary"][endpoint]
        lower_ci = values["accuracy_lower_ci95"]
        upper_ci = values["accuracy_upper_ci95"]
        lines.append(
            f"| `{endpoint}` | {values['accuracy_lower']:.4f} | "
            f"[{lower_ci[0]:.4f}, {lower_ci[1]:.4f}] | "
            f"{values['accuracy_upper']:.4f} | "
            f"[{upper_ci[0]:.4f}, {upper_ci[1]:.4f}] | "
            f"{values['finite_exact_tie_count']} |"
        )
    lines.extend(
        [
            "",
            f"- Conservative diagnostic status: `{gate['status']}`",
            f"- Lower-bound status: `{gate['bound_statuses']['lower']}`",
            f"- Upper-bound status: `{gate['bound_statuses']['upper']}`",
            f"- Bound statuses match: `{str(gate['bound_statuses']['match']).lower()}`",
            "- Arbitrary tie-break used: `false`",
            "- Source gate status changed: `false`",
            "- Training-complexity review eligible: `false`",
            "",
            gate["next_action"],
            "",
        ]
    )
    return "\n".join(lines)


def _json_hash(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing conservative tie artifact changed: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
