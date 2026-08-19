from __future__ import annotations

import json
import re
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any

from plasticity_placement.p0c.compiler import ACTIONS
from plasticity_placement.p0c.modeling import parse_unique_action
from plasticity_placement.p0c.runtime import current_code_hash
from plasticity_placement.p0d.analysis import _bootstrap_ci
from plasticity_placement.p0d2h.probes import HARD_CATEGORIES
from plasticity_placement.p0d2hc.analysis import _validate_and_load
from plasticity_placement.p0d2hc.config import CALIBRATION_ARMS

INVALID_TAXONOMY_VERSION = "p0d2hc-invalid-taxonomy-v1"
INVALID_CLASSES = (
    "empty_output",
    "expected_action_with_extra_text",
    "wrong_action_with_extra_text",
    "multiple_actions_including_expected",
    "multiple_actions_excluding_expected",
    "no_allowed_action_at_generation_limit",
    "no_allowed_action_other",
)
_ACTION_BY_CASEFOLD = {action.casefold(): action for action in ACTIONS}
_ACTION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])("
    + "|".join(re.escape(action) for action in ACTIONS)
    + r")(?![A-Za-z0-9_])",
    flags=re.IGNORECASE,
)


def audit_invalid_outputs(
    source_output_dir: Path,
    audit_output_dir: Path,
    bootstrap_samples: int = 10_000,
) -> Path:
    _require_independent_output(source_output_dir, audit_output_dir)
    source_manifest_path = source_output_dir / "manifest.json"
    source_manifest_bytes = source_manifest_path.read_bytes()
    source_manifest = json.loads(source_manifest_bytes)
    config, rows = _validate_and_load(source_output_dir, source_manifest)
    report, invalid_records = build_invalid_output_audit(
        rows,
        max_new_tokens=config.max_new_tokens,
        bootstrap_samples=bootstrap_samples,
    )
    raw_results_sha256 = _raw_tree_hash(source_output_dir)
    source_identity = {
        "source_output_dir": str(source_output_dir),
        "source_run_id": source_manifest["run_id"],
        "source_manifest_sha256": sha256(
            source_manifest_bytes
        ).hexdigest(),
        "source_prompt_token_audit_sha256": source_manifest[
            "prompt_token_audit_sha256"
        ],
        "source_raw_results_sha256": raw_results_sha256,
        "source_experiment_code_sha256": source_manifest["config"][
            "code_sha256"
        ],
        "analysis_code_sha256": current_code_hash(),
        "bootstrap_samples": bootstrap_samples,
        "taxonomy_version": INVALID_TAXONOMY_VERSION,
    }
    audit_run_id = (
        "p0d2hc-invalid-audit-"
        + _json_hash(source_identity)[:10]
    )
    payload = {
        **report,
        "run_id": audit_run_id,
        "source": source_identity,
        "artifact_paths": {
            "summary_json": str(
                audit_output_dir / "invalid_output_audit.json"
            ),
            "report_markdown": str(
                audit_output_dir / "invalid_output_audit.md"
            ),
            "invalid_records_jsonl": str(
                audit_output_dir / "invalid_output_records.jsonl"
            ),
        },
    }
    audit_output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = audit_output_dir / "invalid_output_audit.json"
    _write_immutable(
        summary_path,
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
        audit_output_dir / "invalid_output_records.jsonl",
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for record in invalid_records
        ).encode(),
    )
    _write_immutable(
        audit_output_dir / "invalid_output_audit.md",
        _render_markdown(payload).encode(),
    )
    return summary_path


def build_invalid_output_audit(
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("cannot audit an empty calibration result set")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    classified = [
        classify_output_row(row, max_new_tokens=max_new_tokens)
        for row in rows
    ]
    model_ids = sorted(
        {str(record["model_id"]) for record in classified}
    )
    overall: dict[str, dict[str, Any]] = {}
    categories: dict[str, dict[str, dict[str, Any]]] = {}
    for model_id in model_ids:
        overall[model_id] = {}
        categories[model_id] = {}
        for arm in CALIBRATION_ARMS:
            arm_records = [
                record
                for record in classified
                if record["model_id"] == model_id
                and record["arm"] == arm
            ]
            if not arm_records:
                raise ValueError(
                    f"missing invalid-audit condition: {model_id}/{arm}"
                )
            overall[model_id][arm] = _summarize_records(
                arm_records,
                bootstrap_samples,
            )
            categories[model_id][arm] = {}
            for category in HARD_CATEGORIES:
                category_records = [
                    record
                    for record in arm_records
                    if record["category"] == category
                ]
                if not category_records:
                    raise ValueError(
                        "missing invalid-audit category: "
                        f"{model_id}/{arm}/{category}"
                    )
                categories[model_id][arm][category] = (
                    _summarize_records(
                        category_records,
                        bootstrap_samples,
                    )
                )
    invalid_records = [
        {
            key: value
            for key, value in record.items()
            if key != "strict_correct"
        }
        for record in classified
        if record["strict_invalid"]
    ]
    return (
        {
            "schema_version": "p0d2hc-invalid-output-audit-v1",
            "analysis_status": "post_hoc_supplementary",
            "strict_gate_status_changed": False,
            "next_stage_eligibility_changed": False,
            "raw_rows_modified": False,
            "manifest_modified": False,
            "taxonomy_version": INVALID_TAXONOMY_VERSION,
            "taxonomy_classes": list(INVALID_CLASSES),
            "semantic_recovery_definition": (
                "Exactly one distinct allowed action token occurs anywhere "
                "in generated_text, and it equals expected_action."
            ),
            "row_count": len(classified),
            "invalid_record_count": len(invalid_records),
            "model_count": len(model_ids),
            "overall_by_model_arm": overall,
            "category_by_model_arm": categories,
        },
        invalid_records,
    )


def classify_output_row(
    row: dict[str, Any],
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    generated_text = str(row["generated_text"])
    expected_action = str(row["expected_action"])
    if expected_action not in ACTIONS:
        raise ValueError(
            f"unsupported expected action: {expected_action!r}"
        )
    generated_tokens = int(row["generated_tokens"])
    if generated_tokens < 0:
        raise ValueError("generated_tokens cannot be negative")
    strict_action = parse_unique_action(generated_text, ACTIONS)
    strict_invalid = strict_action is None
    strict_correct = strict_action == expected_action
    if (
        bool(row["invalid"]) != strict_invalid
        or bool(row["correct"]) != strict_correct
    ):
        raise ValueError(
            "stored strict result differs from deterministic reparse: "
            f"{row.get('probe_id')}"
        )

    mentions = tuple(
        _ACTION_BY_CASEFOLD[match.group(1).casefold()]
        for match in _ACTION_PATTERN.finditer(generated_text)
    )
    distinct_mentions = tuple(dict.fromkeys(mentions))
    semantic_action = (
        distinct_mentions[0] if len(distinct_mentions) == 1 else None
    )
    semantic_correct = semantic_action == expected_action
    at_generation_limit = generated_tokens >= max_new_tokens
    invalid_class = (
        _invalid_class(
            generated_text=generated_text,
            expected_action=expected_action,
            distinct_mentions=distinct_mentions,
            at_generation_limit=at_generation_limit,
        )
        if strict_invalid
        else None
    )
    return {
        "model_id": str(row["calibration_model_id"]),
        "model_role": str(row["calibration_model_role"]),
        "arm": str(row["arm"]),
        "lesson_id": str(row["lesson_id"]),
        "probe_id": str(row["probe_id"]),
        "category": str(row["category"]),
        "pair_id": str(row["pair_id"]),
        "lesson_type": str(row["lesson_type"]),
        "expected_action": expected_action,
        "generated_text": generated_text,
        "generated_tokens": generated_tokens,
        "max_new_tokens": max_new_tokens,
        "at_generation_limit": at_generation_limit,
        "strict_predicted_action": strict_action,
        "strict_invalid": strict_invalid,
        "strict_correct": strict_correct,
        "mentioned_actions": list(mentions),
        "distinct_mentioned_actions": list(distinct_mentions),
        "expected_action_mentioned": expected_action in distinct_mentions,
        "semantic_predicted_action": semantic_action,
        "semantic_correct": semantic_correct,
        "invalid_class": invalid_class,
    }


def _invalid_class(
    *,
    generated_text: str,
    expected_action: str,
    distinct_mentions: tuple[str, ...],
    at_generation_limit: bool,
) -> str:
    if not generated_text.strip():
        return "empty_output"
    if len(distinct_mentions) == 1:
        if distinct_mentions[0] == expected_action:
            return "expected_action_with_extra_text"
        return "wrong_action_with_extra_text"
    if len(distinct_mentions) > 1:
        if expected_action in distinct_mentions:
            return "multiple_actions_including_expected"
        return "multiple_actions_excluding_expected"
    if at_generation_limit:
        return "no_allowed_action_at_generation_limit"
    return "no_allowed_action_other"


def _summarize_records(
    records: list[dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    invalid = [
        record for record in records if record["strict_invalid"]
    ]
    strict_valid = [
        record for record in records if not record["strict_invalid"]
    ]
    lesson_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        lesson_groups[str(record["lesson_id"])].append(record)
    strict_by_lesson = [
        mean(bool(record["strict_correct"]) for record in group)
        for _, group in sorted(lesson_groups.items())
    ]
    semantic_by_lesson = [
        mean(bool(record["semantic_correct"]) for record in group)
        for _, group in sorted(lesson_groups.items())
    ]
    invalid_by_lesson = [
        mean(bool(record["strict_invalid"]) for record in group)
        for _, group in sorted(lesson_groups.items())
    ]
    gain_by_lesson = [
        semantic - strict
        for strict, semantic in zip(
            strict_by_lesson,
            semantic_by_lesson,
            strict=True,
        )
    ]
    strict_accuracy = mean(strict_by_lesson)
    semantic_accuracy = mean(semantic_by_lesson)
    invalid_count = len(invalid)
    class_counts = {
        invalid_class: sum(
            record["invalid_class"] == invalid_class
            for record in invalid
        )
        for invalid_class in INVALID_CLASSES
    }
    return {
        "row_count": len(records),
        "lesson_count": len(lesson_groups),
        "strict_accuracy": strict_accuracy,
        "strict_accuracy_ci95": _bootstrap_ci(
            strict_by_lesson,
            bootstrap_samples,
        ),
        "strict_invalid_count": invalid_count,
        "strict_invalid_rate": mean(invalid_by_lesson),
        "strict_invalid_rate_ci95": _bootstrap_ci(
            invalid_by_lesson,
            bootstrap_samples,
        ),
        "strict_valid_count": len(strict_valid),
        "conditional_valid_accuracy": (
            mean(
                bool(record["strict_correct"])
                for record in strict_valid
            )
            if strict_valid
            else None
        ),
        "semantic_accuracy": semantic_accuracy,
        "semantic_accuracy_ci95": _bootstrap_ci(
            semantic_by_lesson,
            bootstrap_samples,
        ),
        "semantic_recovery_gain": (
            semantic_accuracy - strict_accuracy
        ),
        "semantic_recovery_gain_ci95": _bootstrap_ci(
            gain_by_lesson,
            bootstrap_samples,
        ),
        "expected_action_mention_rate": mean(
            bool(record["expected_action_mentioned"])
            for record in records
        ),
        "invalid_semantic_recovery_count": sum(
            bool(record["semantic_correct"]) for record in invalid
        ),
        "invalid_semantic_recovery_rate": (
            mean(bool(record["semantic_correct"]) for record in invalid)
            if invalid
            else None
        ),
        "invalid_expected_mention_count": sum(
            bool(record["expected_action_mentioned"])
            for record in invalid
        ),
        "invalid_expected_mention_rate": (
            mean(
                bool(record["expected_action_mentioned"])
                for record in invalid
            )
            if invalid
            else None
        ),
        "invalid_at_generation_limit_count": sum(
            bool(record["at_generation_limit"])
            for record in invalid
        ),
        "invalid_at_generation_limit_rate": (
            mean(
                bool(record["at_generation_limit"])
                for record in invalid
            )
            if invalid
            else None
        ),
        "invalid_class_counts": class_counts,
        "invalid_class_rates": {
            invalid_class: (
                count / invalid_count if invalid_count else 0.0
            )
            for invalid_class, count in class_counts.items()
        },
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# P0-D2H-CAL Invalid-Output Audit",
        "",
        "> Post-hoc supplementary analysis. Frozen strict gates and "
        "next-stage eligibility are unchanged.",
        "",
        "## Overall",
        "",
        "| Model | Arm | Strict | Invalid | Semantic | Gain | "
        "Valid-only | Invalid recovered | At limit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model_id, arms in report["overall_by_model_arm"].items():
        for arm, values in arms.items():
            lines.append(
                f"| {model_id} | {arm} | "
                f"{values['strict_accuracy']:.4f} | "
                f"{values['strict_invalid_rate']:.4f} | "
                f"{values['semantic_accuracy']:.4f} | "
                f"{values['semantic_recovery_gain']:+.4f} | "
                f"{_format_optional(values['conditional_valid_accuracy'])} | "
                f"{_format_optional(values['invalid_semantic_recovery_rate'])} | "
                f"{_format_optional(values['invalid_at_generation_limit_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## Invalid taxonomy counts",
            "",
            "| Model | Arm | Class | Count | Share of invalid |",
            "|---|---|---|---:|---:|",
        ]
    )
    for model_id, arms in report["overall_by_model_arm"].items():
        for arm, values in arms.items():
            for invalid_class in INVALID_CLASSES:
                count = values["invalid_class_counts"][invalid_class]
                if count:
                    lines.append(
                        f"| {model_id} | {arm} | {invalid_class} | "
                        f"{count} | "
                        f"{values['invalid_class_rates'][invalid_class]:.4f} |"
                    )
    lines.extend(
        [
            "",
            "## Category semantic recovery",
            "",
            "| Model | Arm | Category | Strict | Invalid | Semantic | Gain |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for model_id, arms in report["category_by_model_arm"].items():
        for arm, categories in arms.items():
            for category, values in categories.items():
                lines.append(
                    f"| {model_id} | {arm} | {category} | "
                    f"{values['strict_accuracy']:.4f} | "
                    f"{values['strict_invalid_rate']:.4f} | "
                    f"{values['semantic_accuracy']:.4f} | "
                    f"{values['semantic_recovery_gain']:+.4f} |"
                )
    lines.extend(
        [
            "",
            "Semantic recovery counts a row only when exactly one distinct "
            "allowed action appears and it is the expected action. Multiple "
            "actions remain ambiguous even if one is correct.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _require_independent_output(
    source_output_dir: Path,
    audit_output_dir: Path,
) -> None:
    source = source_output_dir.resolve()
    audit = audit_output_dir.resolve()
    if (
        source == audit
        or audit.is_relative_to(source)
        or source.is_relative_to(audit)
    ):
        raise ValueError(
            "invalid-output audit directory must be independent from "
            "the source P0-D2H-CAL run"
        )


def _raw_tree_hash(source_output_dir: Path) -> str:
    root = source_output_dir / "results" / "raw"
    files = sorted(root.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no calibration raw results under {root}")
    digest = sha256()
    for path in files:
        digest.update(str(path.relative_to(source_output_dir)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
            raise ValueError(
                f"existing invalid-output audit artifact changed: {path}"
            )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
