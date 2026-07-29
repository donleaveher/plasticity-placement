from __future__ import annotations

import json
import math
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.runtime import current_code_hash
from plasticity_placement.p0d2hcrd.config import ENDPOINTS
from plasticity_placement.p0d2hcrd.manifest import unit_key
from plasticity_placement.p0d2hcrd.scoring import rank_candidate_scores

AUDIT_VERSION = "p0d2hcrd-scoring-integrity-audit-v1"
ANOMALY_STATUSES = ("tie", "non_finite", "candidate_token_mismatch")
FACTOR_FIELDS = (
    "lesson_id",
    "lesson_type",
    "lesson_side",
    "route_variant",
    "target_slot",
    "expected_candidate",
    "slot_candidate_order",
    "slot_content_order",
    "action_panel_position",
)
_RANKING_FIELDS = (
    "predicted_candidate",
    "mean_predicted_candidate",
    "top1_top2_margin",
    "sum_mean_disagreement",
    "tie",
    "mean_score_tie",
    "non_finite",
    "error_status",
)


def audit_scoring_integrity(
    source_output_dir: Path,
    audit_output_dir: Path,
) -> Path:
    """Create an immutable, read-only post-hoc audit of CRD score anomalies."""
    _require_independent_output(source_output_dir, audit_output_dir)
    manifest_path = source_output_dir / "manifest.json"
    summary_path = source_output_dir / "results" / "aggregate" / "summary.json"
    manifest_bytes = manifest_path.read_bytes()
    summary_bytes = summary_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    summary = json.loads(summary_bytes)
    rows, raw_results_sha256 = load_verified_rows(
        source_output_dir,
        manifest,
    )
    report, anomaly_records = build_scoring_integrity_audit(rows)
    if summary.get("run_id") != manifest.get("run_id") or int(
        summary.get("decision_row_count", -1)
    ) != len(rows):
        raise ValueError("CRD aggregate summary does not match its source run")

    source_identity = {
        "source_output_dir": str(source_output_dir),
        "source_run_id": manifest["run_id"],
        "source_manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "source_summary_sha256": sha256(summary_bytes).hexdigest(),
        "source_raw_results_sha256": raw_results_sha256,
        "source_experiment_code_sha256": manifest["config"]["code_sha256"],
        "source_diagnostic_status": summary["diagnostic_gate"]["status"],
        "analysis_code_sha256": current_code_hash(),
        "audit_version": AUDIT_VERSION,
    }
    audit_run_id = "p0d2hcrd-integrity-audit-" + _json_hash(source_identity)[:10]
    payload = {
        **report,
        "run_id": audit_run_id,
        "source": source_identity,
        "artifact_paths": {
            "summary_json": str(audit_output_dir / "scoring_integrity_audit.json"),
            "report_markdown": str(audit_output_dir / "scoring_integrity_audit.md"),
            "anomaly_records_jsonl": str(audit_output_dir / "scoring_integrity_records.jsonl"),
        },
    }

    if (
        manifest_path.read_bytes() != manifest_bytes
        or summary_path.read_bytes() != summary_bytes
        or raw_tree_hash(source_output_dir) != raw_results_sha256
    ):
        raise RuntimeError("source CRD artifacts changed during integrity audit")

    audit_output_dir.mkdir(parents=True, exist_ok=True)
    result_path = audit_output_dir / "scoring_integrity_audit.json"
    _write_immutable(
        result_path,
        (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )
    _write_immutable(
        audit_output_dir / "scoring_integrity_records.jsonl",
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in anomaly_records
        ).encode(),
    )
    _write_immutable(
        audit_output_dir / "scoring_integrity_audit.md",
        _render_markdown(payload).encode(),
    )
    return result_path


def build_scoring_integrity_audit(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Recompute score invariants and expose every non-ok decision row."""
    if not rows:
        raise ValueError("cannot audit an empty CRD result set")
    evidence = [row_evidence(row) for row in rows]
    probe_ids = [str(row.get("probe_id")) for row in rows]
    probe_ids_unique = len(probe_ids) == len(set(probe_ids))
    anomaly_records = [item for item in evidence if item["error_status"] != "ok"]
    tie_records = [item for item in anomaly_records if item["error_status"] == "tie"]
    structurally_consistent = probe_ids_unique and all(
        item["checks"]["all_structural_checks_passed"] for item in evidence
    )
    statuses = Counter(str(item["error_status"]) for item in evidence)
    classification = _classification(
        anomaly_records,
        structurally_consistent=structurally_consistent,
    )
    endpoint_summary = {
        endpoint: _endpoint_sensitivity([item for item in evidence if item["endpoint"] == endpoint])
        for endpoint in ENDPOINTS
        if any(item["endpoint"] == endpoint for item in evidence)
    }
    return (
        {
            "schema_version": AUDIT_VERSION,
            "analysis_status": "post_hoc_supplementary",
            "source_gate_status_changed": False,
            "next_stage_eligibility_changed": False,
            "raw_rows_modified": False,
            "manifest_modified": False,
            "row_count": len(rows),
            "probe_ids_unique": probe_ids_unique,
            "status_counts": {
                status: statuses.get(status, 0) for status in ("ok", *ANOMALY_STATUSES)
            },
            "anomaly_record_count": len(anomaly_records),
            "tie_record_count": len(tie_records),
            "structural_checks_passed": structurally_consistent,
            "classification": classification,
            "endpoint_sensitivity": endpoint_summary,
            "tie_factor_counts": _factor_counts(tie_records),
            "interpretation": _interpretation(classification),
        },
        anomaly_records,
    )


def row_evidence(row: dict[str, Any]) -> dict[str, Any]:
    """Return independently recomputed structural and ranking evidence."""
    candidates_value = row.get("candidates")
    candidates = candidates_value if isinstance(candidates_value, list) else []
    ordered_value = row.get("ordered_candidates")
    ordered_candidates = ordered_value if isinstance(ordered_value, list) else []
    candidate_names = [str(candidate.get("candidate")) for candidate in candidates]
    continuations = [str(candidate.get("continuation")) for candidate in candidates]
    token_sequences = [
        tuple(candidate.get("token_ids", []))
        if isinstance(candidate.get("token_ids"), list)
        else ()
        for candidate in candidates
    ]
    candidate_shape_valid = len(candidates) in {2, 4} and all(
        isinstance(candidate, dict) for candidate in candidates
    )
    candidate_values_match_order = candidate_shape_valid and candidate_names == ordered_candidates
    candidate_values_unique = candidate_shape_valid and len(candidate_names) == len(
        set(candidate_names)
    )
    candidate_continuations_unique = candidate_shape_valid and len(continuations) == len(
        set(continuations)
    )
    candidate_token_sequences_unique = candidate_shape_valid and len(token_sequences) == len(
        set(token_sequences)
    )
    token_shapes_valid = candidate_shape_valid and all(
        isinstance(candidate.get("token_ids"), list)
        and isinstance(candidate.get("token_logprobs"), list)
        and isinstance(candidate.get("token_count"), int)
        and int(candidate["token_count"]) > 0
        and int(candidate["token_count"]) == len(candidate["token_ids"])
        for candidate in candidates
    )
    status = str(row.get("error_status"))
    finite_scores = candidate_shape_valid and all(
        _finite(candidate.get(field))
        for candidate in candidates
        for field in ("sum_logprob", "mean_logprob")
    )
    score_sums_match = (
        finite_scores
        and token_shapes_valid
        and all(
            len(candidate["token_logprobs"]) == int(candidate["token_count"])
            and all(_finite(value) for value in candidate["token_logprobs"])
            and math.isclose(
                float(candidate["sum_logprob"]),
                sum(float(value) for value in candidate["token_logprobs"]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            and math.isclose(
                float(candidate["mean_logprob"]),
                float(candidate["sum_logprob"]) / int(candidate["token_count"]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            for candidate in candidates
        )
    )

    computed: dict[str, Any] | None = None
    if candidate_shape_valid:
        try:
            computed = rank_candidate_scores(candidates)
        except (TypeError, ValueError):
            computed = None
    if status == "candidate_token_mismatch":
        stored_ranking_matches = (
            row.get("predicted_candidate") is None
            and not bool(row.get("correct"))
            and all(
                candidate.get("sum_logprob") is None and candidate.get("mean_logprob") is None
                for candidate in candidates
            )
        )
        score_representation_valid = stored_ranking_matches
        candidate_ranks_match = all(
            candidate.get("sum_rank") is None and candidate.get("mean_rank") is None
            for candidate in candidates
        )
    else:
        stored_ranking_matches = (
            computed is not None
            and all(row.get(field) == computed[field] for field in _RANKING_FIELDS)
            and bool(row.get("correct"))
            == (row.get("predicted_candidate") == row.get("expected_candidate"))
        )
        candidate_ranks_match = computed is not None and all(
            observed.get("sum_rank") == expected.get("sum_rank")
            and observed.get("mean_rank") == expected.get("mean_rank")
            for observed, expected in zip(
                candidates,
                computed["candidates"],
                strict=True,
            )
        )
        score_representation_valid = (
            score_sums_match
            if status in {"ok", "tie"}
            else status == "non_finite"
            and computed is not None
            and computed["error_status"] == "non_finite"
        )

    top_candidates: list[dict[str, Any]] = []
    if finite_scores:
        maximum = max(float(candidate["sum_logprob"]) for candidate in candidates)
        top_candidates = [
            {
                "candidate": str(candidate["candidate"]),
                "sum_logprob": float(candidate["sum_logprob"]),
                "mean_logprob": float(candidate["mean_logprob"]),
                "token_ids": candidate["token_ids"],
                "token_logprobs": candidate["token_logprobs"],
            }
            for candidate in candidates
            if float(candidate["sum_logprob"]) == maximum
        ]
    exact_top_sum_tie = len(top_candidates) > 1
    status_matches_exact_scores = (
        status == "tie" if finite_scores else status in {"non_finite", "candidate_token_mismatch"}
    ) == exact_top_sum_tie
    if not finite_scores:
        status_matches_exact_scores = status in {
            "non_finite",
            "candidate_token_mismatch",
        }

    structural_checks = {
        "candidate_shape_valid": candidate_shape_valid,
        "candidate_values_match_order": candidate_values_match_order,
        "candidate_values_unique": candidate_values_unique,
        "candidate_continuations_unique": candidate_continuations_unique,
        "candidate_token_sequences_unique": candidate_token_sequences_unique,
        "candidate_token_shapes_valid": token_shapes_valid,
        "score_representation_valid": score_representation_valid,
        "stored_ranking_matches_recomputed": stored_ranking_matches,
        "candidate_ranks_match_recomputed": candidate_ranks_match,
        "status_matches_exact_scores": status_matches_exact_scores,
    }
    all_structural_checks_passed = all(structural_checks.values())
    tie_origin = None
    if status == "tie":
        tie_origin = (
            "distinct_candidates_exact_score_tie"
            if all_structural_checks_passed
            else "candidate_or_score_structure_inconsistency"
        )
    return {
        "lesson_id": str(row.get("lesson_id")),
        "pair_id": str(row.get("pair_id")),
        "probe_id": str(row.get("probe_id")),
        "endpoint": str(row.get("endpoint")),
        "lesson_type": row.get("lesson_type"),
        "lesson_side": row.get("lesson_side"),
        "route_variant": row.get("route_variant"),
        "target_slot": row.get("target_slot"),
        "expected_candidate": row.get("expected_candidate"),
        "ordered_candidates": ordered_candidates,
        "slot_candidate_order": row.get("slot_candidate_order"),
        "slot_content_order": row.get("slot_content_order"),
        "action_panel_position": row.get("action_panel_position"),
        "predicted_candidate": row.get("predicted_candidate"),
        "correct": bool(row.get("correct")),
        "error_status": status,
        "error_message": row.get("error_message"),
        "top1_top2_margin": row.get("top1_top2_margin"),
        "tie_origin": tie_origin,
        "top_sum_candidates": top_candidates,
        "checks": {
            **structural_checks,
            "all_structural_checks_passed": all_structural_checks_passed,
        },
        "candidates": candidates,
    }


def _classification(
    anomaly_records: list[dict[str, Any]],
    *,
    structurally_consistent: bool,
) -> str:
    if not structurally_consistent:
        return "structural_inconsistency_detected"
    if not anomaly_records:
        return "no_scoring_anomalies"
    if all(
        record["error_status"] == "tie"
        and record["tie_origin"] == "distinct_candidates_exact_score_tie"
        for record in anomaly_records
    ):
        return "distinct_candidate_exact_score_ties_observed"
    return "explicit_scoring_errors_observed"


def _endpoint_sensitivity(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    decision_count = len(records)
    correct_count = sum(bool(record["correct"]) for record in records)
    tie_count = sum(record["error_status"] == "tie" for record in records)
    return {
        "decision_count": decision_count,
        "correct_count": correct_count,
        "tie_count": tie_count,
        "official_accuracy": correct_count / decision_count,
        "accuracy_if_all_ties_incorrect": correct_count / decision_count,
        "accuracy_if_all_ties_correct": ((correct_count + tie_count) / decision_count),
    }


def _factor_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        field: dict(sorted(Counter(_factor_label(record.get(field)) for record in records).items()))
        for field in FACTOR_FIELDS
    }


def _factor_label(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _interpretation(classification: str) -> str:
    if classification == "no_scoring_anomalies":
        return "No non-ok score rows were observed."
    if classification == "distinct_candidate_exact_score_ties_observed":
        return (
            "The persisted rows contain exact top-score ties between distinct "
            "candidate token sequences. This supports a genuine model-score tie "
            "diagnosis, but it does not change the frozen zero-tie gate."
        )
    if classification == "structural_inconsistency_detected":
        return (
            "At least one persisted row failed an independently recomputed "
            "candidate, token, score, status, or ranking invariant."
        )
    return (
        "At least one explicit non-tie scoring error was observed. The frozen "
        "source gate and next-stage eligibility remain unchanged."
    )


def load_verified_rows(
    source_output_dir: Path,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    """Load the complete immutable CRD matrix and verify every unit hash."""
    if manifest.get("schema_version") != "p0d2hcrd-manifest-v1":
        raise ValueError("not a P0-D2H-CRD manifest")
    if manifest.get("errors"):
        raise ValueError("source P0-D2H-CRD manifest contains errors")
    selected_lessons = [str(value) for value in manifest.get("selected_lessons", [])]
    units = manifest.get("units")
    if (
        len(selected_lessons) != 24
        or not isinstance(units, dict)
        or set(units) != {unit_key(value) for value in selected_lessons}
    ):
        raise ValueError("source P0-D2H-CRD unit matrix is incomplete")

    rows: list[dict[str, Any]] = []
    for lesson_id in selected_lessons:
        unit = units[unit_key(lesson_id)]
        path = source_output_dir / "results" / "raw" / "scale_canary" / f"{lesson_id}.jsonl"
        payload = path.read_bytes()
        if (
            unit.get("state") != "verified"
            or unit.get("lesson_id") != lesson_id
            or unit.get("model_id") != "scale_canary"
            or unit.get("result_sha256") != sha256(payload).hexdigest()
        ):
            raise ValueError(f"source P0-D2H-CRD unit is not verified: {lesson_id}")
        rows.extend(json.loads(line) for line in payload.decode().splitlines() if line.strip())

    endpoint_counts = Counter(str(row.get("endpoint")) for row in rows)
    candidate_count = sum(
        len(row.get("candidates", [])) if isinstance(row.get("candidates"), list) else 0
        for row in rows
    )
    if (
        len(rows) != 1_536
        or endpoint_counts != Counter({"route_only": 384, "retrieval_only": 384, "combined": 768})
        or candidate_count != 5_376
        or any(
            row.get("schema_version") != "p0d2hcrd-decision-row-v1"
            or row.get("run_id") != manifest.get("run_id")
            for row in rows
        )
    ):
        raise ValueError("source P0-D2H-CRD decision matrix changed")
    return rows, raw_tree_hash(source_output_dir)


def raw_tree_hash(source_output_dir: Path) -> str:
    """Hash every verified CRD raw result path and byte."""
    root = source_output_dir / "results" / "raw"
    files = sorted(root.rglob("*.jsonl"))
    if len(files) != 24:
        raise ValueError(f"expected 24 CRD raw files, found {len(files)}")
    digest = sha256()
    for path in files:
        digest.update(str(path.relative_to(source_output_dir)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P0-D2H-CRD scoring-integrity audit",
        "",
        f"- Classification: `{payload['classification']}`",
        f"- Source diagnostic status: `{payload['source']['source_diagnostic_status']}`",
        f"- Audited rows: {payload['row_count']}",
        f"- Anomaly rows: {payload['anomaly_record_count']}",
        f"- Exact tie rows: {payload['tie_record_count']}",
        f"- Structural checks passed: `{str(payload['structural_checks_passed']).lower()}`",
        "- Source gate status changed: `false`",
        "- Next-stage eligibility changed: `false`",
        "",
        "| Endpoint | Correct / decisions | Ties | Official | All ties correct |",
        "|---|---:|---:|---:|---:|",
    ]
    for endpoint, values in payload["endpoint_sensitivity"].items():
        lines.append(
            f"| `{endpoint}` | {values['correct_count']} / "
            f"{values['decision_count']} | {values['tie_count']} | "
            f"{values['official_accuracy']:.4f} | "
            f"{values['accuracy_if_all_ties_correct']:.4f} |"
        )
    lines.extend(["", payload["interpretation"], ""])
    return "\n".join(lines)


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_independent_output(
    source_output_dir: Path,
    audit_output_dir: Path,
) -> None:
    source = source_output_dir.resolve()
    audit = audit_output_dir.resolve()
    if source == audit or audit.is_relative_to(source) or source.is_relative_to(audit):
        raise ValueError(
            "scoring-integrity audit directory must be independent from the source P0-D2H-CRD run"
        )


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
            raise ValueError(f"existing scoring-integrity audit artifact changed: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
