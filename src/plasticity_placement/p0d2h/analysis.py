from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from plasticity_placement.p0c.compiler import ACTIONS, load_compiled_bank
from plasticity_placement.p0c.domain import Arm, CompiledLesson, P0CProbe
from plasticity_placement.p0c.modeling import (
    parse_unique_action,
    read_probe_results,
)
from plasticity_placement.p0c.runtime import prepare_experiment
from plasticity_placement.p0d.analysis import (
    _atomic_json_write,
    _bootstrap_ci,
)
from plasticity_placement.p0d.runtime import _adapter_hash
from plasticity_placement.p0d2.analysis import _validate_manifest as _validate_p0d2
from plasticity_placement.p0d2.manifest import unit_key as source_unit_key
from plasticity_placement.p0d2h.config import STRESS_CONDITION_IDS
from plasticity_placement.p0d2h.manifest import unit_key
from plasticity_placement.p0d2h.probes import (
    HARD_CATEGORIES,
    HARD_PROBE_COMPILER_VERSION,
    PROBES_PER_LESSON,
    load_hard_probe_bank,
    verify_hard_probe_hashes,
)
from plasticity_placement.p0d2h.prompting import (
    EXTERNAL_PROMPT_RENDERER_VERSION,
    EXTERNAL_PROMPT_VARIANT,
    PLAIN_PROMPT_VARIANT,
)
from plasticity_placement.p0d2h.runtime import (
    _adapter_result_path,
    _base_result_path,
)


def _source_validation_progress(index: int, total: int, key: str) -> None:
    if index == 1 or index == total or index % 25 == 0:
        print(
            f"[p0d2h-aggregate] source integrity CPU/Drive {index}/{total}: {key}",
            flush=True,
        )


def aggregate_experiment(
    output_dir: Path,
    bootstrap_samples: int = 10_000,
) -> Path:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing P0-D2H manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    (
        source,
        source_summary,
        compiled,
        hard_bank,
        prompt_audit,
        rows,
    ) = _validate_and_load(output_dir, manifest)
    seed_category, base_category = _category_metrics(rows, compiled)
    lesson_category = [
        *base_category,
        *_average_seed_categories(seed_category),
    ]
    lesson_overall = _overall_lesson_metrics(lesson_category)
    seed_overall = _overall_seed_metrics(seed_category)
    condition_summary = _condition_summary(
        lesson_category,
        lesson_overall,
        bootstrap_samples,
    )
    original = _original_metrics(source_summary)
    degradation = _degradation_summary(
        lesson_overall,
        original,
        bootstrap_samples,
    )
    contrasts = {
        f"late-matched-vs-{comparator}": _paired_contrast(
            lesson_category,
            lesson_overall,
            seed_overall,
            original,
            "late-matched",
            comparator,
            bootstrap_samples,
        )
        for comparator in (
            "full-base",
            "early-matched",
            "middle-matched",
        )
    }
    suite_quality = _suite_quality(
        condition_summary,
        rows,
        manifest["config"],
    )
    gates = _gate_status(
        contrasts["late-matched-vs-full-base"],
        float(manifest["config"]["resilience_margin"]),
        suite_quality,
    )
    summary = {
        "schema_version": "p0d2h-summary-v2",
        "run_id": manifest["run_id"],
        "stage": "hard_probe",
        "source_p0d2_run_id": source["run_id"],
        "selected_lesson_count": len(manifest["selected_lessons"]),
        "condition_count": len(manifest["conditions"]),
        "hard_categories": list(HARD_CATEGORIES),
        "probes_per_lesson": PROBES_PER_LESSON,
        "adapter_unit_count": len(manifest["units"]),
        "probe_row_count": len(rows),
        "condition_summary": condition_summary,
        "degradation_vs_original_tg": degradation,
        "contrasts": contrasts,
        "prompt_token_audit": {
            key: value for key, value in prompt_audit.items() if key != "records"
        },
        "suite_quality": suite_quality,
        "seed_category_metrics": seed_category,
        "lesson_category_metrics": lesson_category,
        "lesson_overall_metrics": lesson_overall,
        "gates": gates,
        "source_summary_sha256": manifest["config"]["source_summary_sha256"],
        "hard_probe_hashes": manifest["config"]["hard_probe_hashes"],
        "no_fabrication_status": (
            "All values were aggregated from verified P0-D2H raw records; "
            "original TG values came from the hash-frozen verified P0-D2 summary."
        ),
    }
    aggregate_dir = output_dir / "results" / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    summary_path = aggregate_dir / "summary.json"
    _atomic_json_write(summary_path, summary)
    (aggregate_dir / "main_table.md").write_text(
        _render_main_table(condition_summary, degradation, contrasts),
        encoding="utf-8",
    )
    _atomic_json_write(
        aggregate_dir / "next_stage_decision.json",
        {
            "schema_version": "p0d2h-next-stage-v2",
            "source_run_id": manifest["run_id"],
            "status": gates["next_stage_status"],
            "frozen_locus_status": gates["decision_status"],
            "suite_quality_status": suite_quality["status"],
            "selection_automatic": False,
            "automatic_narrow_scan_started": False,
            "automatic_training_started": False,
            "instruction": gates["next_action"],
        },
    )
    return summary_path


def _validate_and_load(
    output_dir: Path,
    manifest: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, CompiledLesson],
    dict[str, tuple[P0CProbe, ...]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    if manifest.get("schema_version") != "p0d2h-manifest-v1":
        raise ValueError("not a P0-D2H v1 manifest")
    required = {
        "run_id",
        "config",
        "source_manifest_path",
        "selected_lessons",
        "conditions",
        "base_arms",
        "units",
        "prompt_token_audit_sha256",
    }
    if not required.issubset(manifest):
        raise ValueError(f"P0-D2H manifest is missing {sorted(required - manifest.keys())}")
    if manifest.get("errors"):
        raise ValueError("P0-D2H manifest contains recorded errors")
    config = manifest["config"]
    if (
        config.get("schema_version") != "p0d2h-config-v2"
        or config.get("stage") != "hard_probe"
        or config.get("external_prompt_renderer_version") != EXTERNAL_PROMPT_RENDERER_VERSION
    ):
        raise ValueError("P0-D2H config schema/stage mismatch")
    condition_ids = set(manifest["conditions"])
    if condition_ids != set(STRESS_CONDITION_IDS):
        raise ValueError("P0-D2H conditions differ from the frozen matrix")
    if [str(condition["condition_id"]) for condition in config["conditions"]] != list(
        STRESS_CONDITION_IDS
    ):
        raise ValueError("P0-D2H config condition order changed")
    lesson_ids = tuple(str(value) for value in manifest["selected_lessons"])
    if list(lesson_ids) != [str(value) for value in config["selected_lesson_ids"]]:
        raise ValueError("P0-D2H selected lesson order changed")
    seeds = tuple(int(seed) for seed in config["training_seeds"])

    source_path = Path(str(manifest["source_manifest_path"]))
    source_bytes = source_path.read_bytes()
    if _sha256(source_bytes) != config["source_manifest_sha256"]:
        raise ValueError("P0-D2 source manifest changed after stress freeze")
    source = json.loads(source_bytes)
    source_dir = source_path.parent
    _validate_p0d2(
        source_dir,
        source,
        progress=_source_validation_progress,
    )
    if (
        source.get("run_id") != config["source_run_id"]
        or source.get("compiler_hashes") != config["source_compiler_hashes"]
        or prepare_experiment(source_dir) != config["source_compiler_hashes"]
    ):
        raise ValueError("P0-D2 source provenance changed")
    source_summary_path = source_dir / "results" / "aggregate" / "summary.json"
    source_summary_bytes = source_summary_path.read_bytes()
    if _sha256(source_summary_bytes) != config["source_summary_sha256"]:
        raise ValueError("P0-D2 source summary changed after stress freeze")
    source_summary = json.loads(source_summary_bytes)
    if (
        source_summary.get("schema_version") != "p0d2-summary-v1"
        or source_summary.get("run_id") != source["run_id"]
        or source_summary.get("gates", {}).get("run_valid") is not True
        or "late-matched"
        not in source_summary.get("gates", {}).get(
            "eligible_locus_conditions",
            [],
        )
    ):
        raise ValueError("P0-D2 source summary is not run-valid and late-matched eligible")

    verify_hard_probe_hashes(output_dir, config["hard_probe_hashes"])
    hard_bank = load_hard_probe_bank(output_dir)
    if tuple(hard_bank) != lesson_ids:
        raise ValueError("P0-D2H hard-probe lesson order changed")
    compiled = {item.lesson.lesson_id: item for item in load_compiled_bank(source_dir)}
    prompt_audit_path = output_dir / "preflight" / "prompt_token_audit.json"
    prompt_audit_bytes = prompt_audit_path.read_bytes()
    if _sha256(prompt_audit_bytes) != manifest["prompt_token_audit_sha256"]:
        raise ValueError("P0-D2H prompt-token audit changed")
    prompt_audit = json.loads(prompt_audit_bytes)
    if (
        prompt_audit.get("schema_version") != "p0d2h-prompt-token-audit-v1"
        or prompt_audit.get("all_prompts_fit") is not True
        or int(prompt_audit.get("input_truncated_count", -1)) != 0
        or int(prompt_audit.get("output_instruction_missing_count", -1)) != 0
        or int(prompt_audit.get("source_training_max_length", -1))
        != int(config["source_training_max_length"])
        or int(prompt_audit.get("evaluation_max_length", -1))
        != int(config["evaluation_max_length"])
    ):
        raise ValueError("P0-D2H prompt-token audit is not run-valid")
    prompt_audit_index = {
        (
            str(record["lesson_id"]),
            str(record["probe_id"]),
            str(record["prompt_variant"]),
        ): record
        for record in prompt_audit["records"]
    }
    if len(prompt_audit_index) != int(prompt_audit["prompt_count"]):
        raise ValueError("P0-D2H prompt-token audit contains duplicate records")

    expected_units = {
        unit_key(condition_id, lesson_id, seed)
        for condition_id in STRESS_CONDITION_IDS
        for lesson_id in lesson_ids
        for seed in seeds
    }
    if set(manifest["units"]) != expected_units:
        raise ValueError("P0-D2H unit matrix is incomplete or contains extras")
    if set(manifest["base_arms"]) != set(lesson_ids):
        raise ValueError("P0-D2H base-arm matrix is incomplete or contains extras")

    rows: list[dict[str, Any]] = []
    for lesson_id in lesson_ids:
        base = manifest["base_arms"][lesson_id]
        if not isinstance(base, dict) or base.get("state") != "verified":
            raise ValueError(f"P0-D2H base arm is not verified: {lesson_id}")
        path = _base_result_path(output_dir, lesson_id)
        rows.extend(
            _verify_result_file(
                path,
                manifest,
                hard_bank[lesson_id],
                lesson_id=lesson_id,
                condition_id=None,
                seed=None,
                expected_arms={Arm.NO_WRITE, Arm.EXTERNAL},
                adapter_sha256=None,
                expected_precision=str(base["evaluation_precision"]),
                source_training_precision=None,
                prompt_audit_index=prompt_audit_index,
            )
        )
    for condition_id in STRESS_CONDITION_IDS:
        for lesson_id in lesson_ids:
            for seed in seeds:
                key = unit_key(condition_id, lesson_id, seed)
                unit = manifest["units"][key]
                if unit.get("state") != "verified":
                    raise ValueError(f"P0-D2H unit is not verified: {key}")
                source_adapter = source_dir / "adapters" / condition_id / lesson_id / f"seed-{seed}"
                source_hash = str(unit["source_adapter_sha256"])
                source_unit = source["units"][source_unit_key(condition_id, lesson_id, seed)]
                source_training_precision = str(source_unit["training_summary"]["precision"])
                evaluation_precision = str(unit["evaluation_precision"])
                if (
                    str(unit.get("source_adapter_path")) != str(source_adapter)
                    or _adapter_hash(source_adapter) != source_hash
                    or unit.get("source_training_precision") != source_training_precision
                ):
                    raise ValueError(f"P0-D2 source adapter changed: {key}")
                path = _adapter_result_path(
                    output_dir,
                    condition_id,
                    lesson_id,
                    seed,
                )
                rows.extend(
                    _verify_result_file(
                        path,
                        manifest,
                        hard_bank[lesson_id],
                        lesson_id=lesson_id,
                        condition_id=condition_id,
                        seed=seed,
                        expected_arms={Arm.PARAMETRIC},
                        adapter_sha256=source_hash,
                        expected_precision=evaluation_precision,
                        source_training_precision=source_training_precision,
                        prompt_audit_index=prompt_audit_index,
                    )
                )
    expected_rows = (
        len(lesson_ids) * 2 * PROBES_PER_LESSON + len(expected_units) * PROBES_PER_LESSON
    )
    if len(rows) != expected_rows:
        raise ValueError(f"P0-D2H row count mismatch: {len(rows)} != {expected_rows}")
    return (
        source,
        source_summary,
        compiled,
        hard_bank,
        prompt_audit,
        rows,
    )


def _verify_result_file(
    path: Path,
    manifest: dict[str, Any],
    probes: tuple[P0CProbe, ...],
    *,
    lesson_id: str,
    condition_id: str | None,
    seed: int | None,
    expected_arms: set[Arm],
    adapter_sha256: str | None,
    expected_precision: str,
    source_training_precision: str | None,
    prompt_audit_index: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = read_probe_results(path)
    probe_by_id = {probe.probe_id: probe for probe in probes}
    expected = {(arm.value, probe.probe_id) for arm in expected_arms for probe in probes}
    observed = [(str(row.get("arm")), str(row.get("probe_id"))) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(f"P0-D2H result matrix mismatch: {path}")
    precisions = {str(row.get("precision")) for row in rows}
    if precisions != {expected_precision}:
        raise ValueError(f"P0-D2H evaluation precision mismatch: {path}")
    config = manifest["config"]
    required_fields = {
        "predicted_action",
        "correct",
        "invalid",
        "generated_text",
        "generated_tokens",
        "input_tokens",
        "latency_seconds",
        "prompt_sha256",
        "precision",
        "source_training_precision",
        "source_training_max_length",
        "evaluation_max_length",
        "prompt_variant",
        "prompt_rendering_version",
        "untruncated_input_tokens",
        "truncated_token_count",
        "input_truncated",
        "output_instruction_preserved",
    }
    for row in rows:
        missing = required_fields - row.keys()
        if missing:
            raise ValueError(f"P0-D2H result row is missing {sorted(missing)}: {path}")
        probe = probe_by_id[str(row["probe_id"])]
        predicted = row.get("predicted_action")
        is_external = row.get("arm") == Arm.EXTERNAL.value
        prompt_variant = EXTERNAL_PROMPT_VARIANT if is_external else PLAIN_PROMPT_VARIANT
        audit = prompt_audit_index.get((lesson_id, probe.probe_id, prompt_variant))
        if (
            audit is None
            or row.get("run_id") != manifest["run_id"]
            or row.get("model") != config["model_name"]
            or row.get("model_revision") != config["model_revision"]
            or row.get("lesson_id") != lesson_id
            or row.get("pair_id") != probe.pair_id
            or row.get("lesson_type") != probe.lesson_type
            or row.get("category") != probe.category
            or row.get("expected_action") != probe.expected_action
            or row.get("training_seed") != seed
            or row.get("condition_id") != condition_id
            or row.get("adapter_sha256") != adapter_sha256
            or row.get("source_training_precision") != source_training_precision
            or int(row.get("source_training_max_length", -1))
            != int(config["source_training_max_length"])
            or int(row.get("evaluation_max_length", -1)) != int(config["evaluation_max_length"])
            or row.get("prompt_variant") != prompt_variant
            or row.get("prompt_rendering_version")
            != (EXTERNAL_PROMPT_RENDERER_VERSION if is_external else "frozen_hard_probe")
            or row.get("source_p0d2_run_id") != config["source_run_id"]
            or row.get("source_manifest_sha256") != config["source_manifest_sha256"]
            or row.get("hard_probe_compiler_version") != HARD_PROBE_COMPILER_VERSION
            or row.get("hard_probes_sha256") != config["hard_probe_hashes"]["hard_probes_sha256"]
            or not isinstance(row.get("prompt_sha256"), str)
            or len(str(row["prompt_sha256"])) != 64
            or row.get("prompt_sha256") != audit["prompt_sha256"]
            or parse_unique_action(str(row["generated_text"]), ACTIONS) != predicted
            or bool(row["correct"]) != (predicted == probe.expected_action)
            or bool(row["invalid"]) != (predicted is None)
            or int(row["input_tokens"]) <= 0
            or int(row["input_tokens"]) != int(audit["retained_input_tokens"])
            or int(row["untruncated_input_tokens"]) != int(audit["untruncated_input_tokens"])
            or int(row["truncated_token_count"]) != int(audit["truncated_token_count"])
            or bool(row["input_truncated"]) != bool(audit["input_truncated"])
            or bool(row["output_instruction_preserved"])
            != bool(audit["output_instruction_preserved"])
            or bool(row["input_truncated"])
            or not bool(row["output_instruction_preserved"])
            or int(row["generated_tokens"]) < 0
            or float(row["latency_seconds"]) < 0.0
        ):
            raise ValueError(f"P0-D2H result provenance mismatch: {path}/{row.get('probe_id')}")
    return rows


def _category_metrics(
    rows: list[dict[str, Any]],
    compiled: dict[str, CompiledLesson],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[
        tuple[str, str, int | None, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        condition_id = (
            str(row["condition_id"]) if row["condition_id"] is not None else str(row["arm"])
        )
        groups[
            (
                condition_id,
                str(row["lesson_id"]),
                (int(row["training_seed"]) if row["training_seed"] is not None else None),
                str(row["category"]),
            )
        ].append(row)
    adapter: list[dict[str, Any]] = []
    base: list[dict[str, Any]] = []
    for (condition_id, lesson_id, seed, category), group in sorted(
        groups.items(),
        key=lambda value: (
            value[0][0],
            value[0][1],
            -1 if value[0][2] is None else value[0][2],
            value[0][3],
        ),
    ):
        item = compiled[lesson_id]
        metric = {
            "condition_id": condition_id,
            "lesson_id": lesson_id,
            "pair_id": item.lesson.pair_id,
            "lesson_type": item.lesson.lesson_type,
            "lesson_split": item.lesson.split,
            "desired_action": item.lesson.desired_action,
            "training_seed": seed,
            "category": category,
            "hard_accuracy": mean(bool(row["correct"]) for row in group),
            "invalid_rate": mean(bool(row["invalid"]) for row in group),
            "truncation_rate": mean(bool(row["input_truncated"]) for row in group),
            "instruction_preservation_rate": mean(
                bool(row["output_instruction_preserved"]) for row in group
            ),
            "mean_input_tokens": mean(float(row["input_tokens"]) for row in group),
            "mean_untruncated_input_tokens": mean(
                float(row["untruncated_input_tokens"]) for row in group
            ),
            "median_latency_seconds": median(float(row["latency_seconds"]) for row in group),
        }
        (base if seed is None else adapter).append(metric)
    return adapter, base


def _average_seed_categories(
    seed_category: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_category:
        groups[
            (
                str(row["condition_id"]),
                str(row["lesson_id"]),
                str(row["category"]),
            )
        ].append(row)
    result: list[dict[str, Any]] = []
    for _, rows in sorted(groups.items()):
        first = rows[0]
        result.append(
            {
                **{
                    key: first[key]
                    for key in (
                        "condition_id",
                        "lesson_id",
                        "pair_id",
                        "lesson_type",
                        "lesson_split",
                        "desired_action",
                        "category",
                    )
                },
                "training_seed_count": len(rows),
                **{
                    field: mean(float(row[field]) for row in rows)
                    for field in (
                        "hard_accuracy",
                        "invalid_rate",
                        "truncation_rate",
                        "instruction_preservation_rate",
                        "mean_input_tokens",
                        "mean_untruncated_input_tokens",
                        "median_latency_seconds",
                    )
                },
            }
        )
    return result


def _overall_lesson_metrics(
    lesson_category: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in lesson_category:
        groups[(str(row["condition_id"]), str(row["lesson_id"]))].append(row)
    result: list[dict[str, Any]] = []
    for _, rows in sorted(groups.items()):
        first = rows[0]
        if {str(row["category"]) for row in rows} != set(HARD_CATEGORIES):
            raise ValueError(
                f"missing hard category for {first['condition_id']}/{first['lesson_id']}"
            )
        result.append(
            {
                **{
                    key: first[key]
                    for key in (
                        "condition_id",
                        "lesson_id",
                        "pair_id",
                        "lesson_type",
                        "lesson_split",
                        "desired_action",
                    )
                },
                **{
                    field: mean(float(row[field]) for row in rows)
                    for field in (
                        "hard_accuracy",
                        "invalid_rate",
                        "truncation_rate",
                        "instruction_preservation_rate",
                        "mean_input_tokens",
                        "mean_untruncated_input_tokens",
                        "median_latency_seconds",
                    )
                },
            }
        )
    return result


def _overall_seed_metrics(
    seed_category: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_category:
        groups[
            (
                str(row["condition_id"]),
                str(row["lesson_id"]),
                int(row["training_seed"]),
            )
        ].append(row)
    return [
        {
            "condition_id": rows[0]["condition_id"],
            "lesson_id": rows[0]["lesson_id"],
            "lesson_type": rows[0]["lesson_type"],
            "training_seed": int(rows[0]["training_seed"]),
            "hard_accuracy": mean(float(row["hard_accuracy"]) for row in rows),
        }
        for _, rows in sorted(groups.items())
    ]


def _condition_summary(
    lesson_category: list[dict[str, Any]],
    lesson_overall: list[dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    condition_ids = sorted({str(row["condition_id"]) for row in lesson_overall})
    for condition_id in condition_ids:
        overall_rows = [row for row in lesson_overall if row["condition_id"] == condition_id]
        accuracies = [float(row["hard_accuracy"]) for row in overall_rows]
        categories: dict[str, Any] = {}
        for category in HARD_CATEGORIES:
            category_values = [
                float(row["hard_accuracy"])
                for row in lesson_category
                if row["condition_id"] == condition_id and row["category"] == category
            ]
            categories[category] = {
                "accuracy": mean(category_values),
                "accuracy_ci95": _bootstrap_ci(
                    category_values,
                    bootstrap_samples,
                ),
            }
        result[condition_id] = {
            "lesson_count": len(overall_rows),
            "hard_accuracy": mean(accuracies),
            "hard_accuracy_ci95": _bootstrap_ci(
                accuracies,
                bootstrap_samples,
            ),
            "invalid_rate": mean(float(row["invalid_rate"]) for row in overall_rows),
            "truncation_rate": mean(float(row["truncation_rate"]) for row in overall_rows),
            "instruction_preservation_rate": mean(
                float(row["instruction_preservation_rate"]) for row in overall_rows
            ),
            "mean_input_tokens": mean(float(row["mean_input_tokens"]) for row in overall_rows),
            "mean_untruncated_input_tokens": mean(
                float(row["mean_untruncated_input_tokens"]) for row in overall_rows
            ),
            "median_latency_seconds": median(
                float(row["median_latency_seconds"]) for row in overall_rows
            ),
            "categories": categories,
        }
    return result


def _original_metrics(
    source_summary: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    allowed = {*STRESS_CONDITION_IDS, str(Arm.NO_WRITE), str(Arm.EXTERNAL)}
    result = {
        (str(row["condition_id"]), str(row["lesson_id"])): row
        for row in source_summary["lesson_metrics"]
        if str(row["condition_id"]) in allowed
    }
    expected = len(allowed) * 24
    if len(result) != expected:
        raise ValueError(
            f"P0-D2 source summary lacks original metrics: {len(result)} != {expected}"
        )
    return result


def _degradation_summary(
    lesson_overall: list[dict[str, Any]],
    original: dict[tuple[str, str], dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in lesson_overall:
        key = (str(row["condition_id"]), str(row["lesson_id"]))
        groups[key[0]].append(
            float(row["hard_accuracy"]) - float(original[key]["target_generalization"])
        )
    return {
        condition_id: {
            "hard_minus_original_tg": mean(values),
            "hard_minus_original_tg_ci95": _bootstrap_ci(
                values,
                bootstrap_samples,
            ),
        }
        for condition_id, values in sorted(groups.items())
    }


def _paired_contrast(
    lesson_category: list[dict[str, Any]],
    lesson_overall: list[dict[str, Any]],
    seed_overall: list[dict[str, Any]],
    original: dict[tuple[str, str], dict[str, Any]],
    left: str,
    right: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    overall_by_lesson = _index(lesson_overall)
    lesson_ids = sorted(
        {lesson_id for condition_id, lesson_id in overall_by_lesson if condition_id == left}
    )
    hard_differences = [
        float(overall_by_lesson[(left, lesson_id)]["hard_accuracy"])
        - float(overall_by_lesson[(right, lesson_id)]["hard_accuracy"])
        for lesson_id in lesson_ids
    ]
    resilience = [
        (
            float(overall_by_lesson[(left, lesson_id)]["hard_accuracy"])
            - float(original[(left, lesson_id)]["target_generalization"])
        )
        - (
            float(overall_by_lesson[(right, lesson_id)]["hard_accuracy"])
            - float(original[(right, lesson_id)]["target_generalization"])
        )
        for lesson_id in lesson_ids
    ]
    category_index = {
        (
            str(row["condition_id"]),
            str(row["lesson_id"]),
            str(row["category"]),
        ): row
        for row in lesson_category
    }
    category_differences: dict[str, Any] = {}
    for category in HARD_CATEGORIES:
        values = [
            float(category_index[(left, lesson_id, category)]["hard_accuracy"])
            - float(category_index[(right, lesson_id, category)]["hard_accuracy"])
            for lesson_id in lesson_ids
        ]
        category_differences[category] = {
            "mean": mean(values),
            "ci95": _bootstrap_ci(values, bootstrap_samples),
        }

    seed_index = {
        (
            str(row["condition_id"]),
            str(row["lesson_id"]),
            int(row["training_seed"]),
        ): row
        for row in seed_overall
    }
    seeds = sorted({seed for condition_id, _, seed in seed_index if condition_id == left})
    seed_differences = {
        str(seed): mean(
            float(seed_index[(left, lesson_id, seed)]["hard_accuracy"])
            - float(seed_index[(right, lesson_id, seed)]["hard_accuracy"])
            for lesson_id in lesson_ids
        )
        for seed in seeds
    }
    lesson_types = sorted(
        {str(overall_by_lesson[(left, lesson_id)]["lesson_type"]) for lesson_id in lesson_ids}
    )
    type_differences = {
        lesson_type: mean(
            difference
            for lesson_id, difference in zip(
                lesson_ids,
                hard_differences,
                strict=True,
            )
            if overall_by_lesson[(left, lesson_id)]["lesson_type"] == lesson_type
        )
        for lesson_type in lesson_types
    }
    largest_index = max(
        range(len(hard_differences)),
        key=lambda index: abs(hard_differences[index]),
    )
    leave_largest_out = [
        value for index, value in enumerate(hard_differences) if index != largest_index
    ]
    return {
        "left": left,
        "right": right,
        "lesson_count": len(lesson_ids),
        "hard_accuracy_mean": mean(hard_differences),
        "hard_accuracy_ci95": _bootstrap_ci(
            hard_differences,
            bootstrap_samples,
        ),
        "hard_wins_ties_losses": {
            "wins": sum(value > 0 for value in hard_differences),
            "ties": sum(value == 0 for value in hard_differences),
            "losses": sum(value < 0 for value in hard_differences),
        },
        "resilience_mean": mean(resilience),
        "resilience_ci95": _bootstrap_ci(
            resilience,
            bootstrap_samples,
        ),
        "category_differences": category_differences,
        "seed_mean_differences": seed_differences,
        "positive_seed_count": sum(value > 0 for value in seed_differences.values()),
        "lesson_type_mean_differences": type_differences,
        "leave_largest_out_mean": mean(leave_largest_out),
    }


def _suite_quality(
    condition_summary: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    chance_accuracy = 1.0 / len(ACTIONS)
    floor_tolerance = float(config["common_floor_tolerance"])
    floor_limit = chance_accuracy + floor_tolerance
    common_floor_categories = [
        category
        for category in HARD_CATEGORIES
        if max(
            float(condition_summary[condition_id]["categories"][category]["accuracy"])
            for condition_id in STRESS_CONDITION_IDS
        )
        <= floor_limit
    ]
    external = condition_summary[Arm.EXTERNAL.value]
    no_write = condition_summary[Arm.NO_WRITE.value]
    external_category_accuracies = {
        category: float(external["categories"][category]["accuracy"])
        for category in HARD_CATEGORIES
    }
    checks = {
        "all_rows_untruncated": all(not bool(row["input_truncated"]) for row in rows),
        "all_output_instructions_preserved": all(
            bool(row["output_instruction_preserved"]) for row in rows
        ),
        "external_accuracy_at_least_threshold": (
            float(external["hard_accuracy"]) >= float(config["external_anchor_min_accuracy"])
        ),
        "external_invalid_rate_at_most_threshold": (
            float(external["invalid_rate"]) <= float(config["external_anchor_max_invalid_rate"])
        ),
        "external_every_category_above_floor": all(
            accuracy > floor_limit for accuracy in external_category_accuracies.values()
        ),
        "no_common_parametric_category_floor": (not common_floor_categories),
    }
    ready = all(checks.values())
    return {
        "status": "ready" if ready else "suite_repair_required",
        "ready_for_next_stage": ready,
        "checks": checks,
        "chance_accuracy": chance_accuracy,
        "common_floor_tolerance": floor_tolerance,
        "common_floor_limit": floor_limit,
        "common_floor_categories": common_floor_categories,
        "external_anchor": {
            "accuracy": float(external["hard_accuracy"]),
            "minimum_accuracy": float(config["external_anchor_min_accuracy"]),
            "invalid_rate": float(external["invalid_rate"]),
            "maximum_invalid_rate": float(config["external_anchor_max_invalid_rate"]),
            "minus_no_write_accuracy": (
                float(external["hard_accuracy"]) - float(no_write["hard_accuracy"])
            ),
            "category_accuracies": external_category_accuracies,
        },
    }


def _gate_status(
    contrast: dict[str, Any],
    margin: float,
    suite_quality: dict[str, Any],
) -> dict[str, Any]:
    hard_ci = [float(value) for value in contrast["hard_accuracy_ci95"]]
    resilience_ci = [float(value) for value in contrast["resilience_ci95"]]
    safeguards = {
        "hard_advantage_ci_positive": hard_ci[0] > 0.0,
        "resilience_noninferior": resilience_ci[0] >= -margin,
        "positive_seed_count_at_least_2": (int(contrast["positive_seed_count"]) >= 2),
        "both_lesson_types_positive": all(
            float(value) > 0.0 for value in contrast["lesson_type_mean_differences"].values()
        ),
        "leave_largest_out_positive": (float(contrast["leave_largest_out_mean"]) > 0.0),
    }
    if all(safeguards.values()):
        status = "hard_locus_robust"
        comparison_next_action = (
            "Review the hard-probe failures, then freeze either a late narrow "
            "scan or a separate multi-mapping training-complexity experiment; "
            "do not start either automatically."
        )
    elif hard_ci[1] <= 0.0 or resilience_ci[1] < -margin:
        status = "shortcut_hypothesis_supported"
        comparison_next_action = (
            "Freeze a separate multi-mapping training-complexity experiment "
            "before any late-layer narrow scan."
        )
    else:
        status = "mixed_or_inconclusive"
        comparison_next_action = (
            "Inspect category, seed, lesson-type, and failure-case results; "
            "revise the hard suite or add lessons before freezing the next run."
        )
    if suite_quality["ready_for_next_stage"]:
        next_stage_status = (
            "eligible_for_reviewed_followup" if status == "hard_locus_robust" else status
        )
        next_action = comparison_next_action
    else:
        next_stage_status = "suite_repair_required"
        next_action = (
            "Do not start a narrow scan or multi-mapping training yet. "
            "Repair the failed prompt-token, external-anchor, or common-floor "
            "suite checks in a new evaluation-only attempt."
        )
    return {
        "run_valid": True,
        "decision_status": status,
        "next_stage_status": next_stage_status,
        "suite_quality_status": suite_quality["status"],
        "resilience_margin": margin,
        "late_vs_full": contrast,
        "safeguards": safeguards,
        "automatic_narrow_scan_started": False,
        "automatic_training_started": False,
        "next_action": next_action,
    }


def _index(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["condition_id"]), str(row["lesson_id"])): row for row in rows}


def _render_main_table(
    summaries: dict[str, dict[str, Any]],
    degradation: dict[str, dict[str, Any]],
    contrasts: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "| Condition | Hard accuracy [95% CI] ↑ | "
        "Hard − original TG [95% CI] ↑ | Invalid ↓ | "
        "Input tokens | Latency median (s) ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition_id, values in summaries.items():
        drop = degradation[condition_id]
        lines.append(
            f"| {condition_id} | {values['hard_accuracy']:.4f} "
            f"[{values['hard_accuracy_ci95'][0]:.4f}, "
            f"{values['hard_accuracy_ci95'][1]:.4f}] | "
            f"{drop['hard_minus_original_tg']:.4f} "
            f"[{drop['hard_minus_original_tg_ci95'][0]:.4f}, "
            f"{drop['hard_minus_original_tg_ci95'][1]:.4f}] | "
            f"{values['invalid_rate']:.4f} | "
            f"{values['mean_input_tokens']:.2f} | "
            f"{values['median_latency_seconds']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Contrast | Hard difference [95% CI] | Resilience difference [95% CI] | W/T/L |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, values in contrasts.items():
        wtl = values["hard_wins_ties_losses"]
        lines.append(
            f"| {label} | {values['hard_accuracy_mean']:.4f} "
            f"[{values['hard_accuracy_ci95'][0]:.4f}, "
            f"{values['hard_accuracy_ci95'][1]:.4f}] | "
            f"{values['resilience_mean']:.4f} "
            f"[{values['resilience_ci95'][0]:.4f}, "
            f"{values['resilience_ci95'][1]:.4f}] | "
            f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} |"
        )
    return "\n".join(lines) + "\n"


def _sha256(value: bytes) -> str:
    from hashlib import sha256

    return sha256(value).hexdigest()
