from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from statistics import mean, median
from typing import Any

from plasticity_placement.p0c.compiler import load_compiled_bank
from plasticity_placement.p0c.domain import Arm, CompiledLesson
from plasticity_placement.p0c.modeling import read_probe_results
from plasticity_placement.p0c.runtime import prepare_experiment
from plasticity_placement.p0d.manifest import unit_key
from plasticity_placement.p0d.runtime import (
    REQUIRED_TRAINING_FIELDS,
    _adapter_hash,
    _adapter_result_path,
    _base_result_path,
)

TARGET_CATEGORIES = {"paraphrase", "compositional"}
INTERFERENCE_CATEGORIES = {"near_neighbor", "unrelated_matched"}
BEHAVIOR_FIELDS = (
    "target_generalization",
    "interference_regression",
    "exact_heldout",
    "conflict_robustness",
    "invalid_rate",
)
EFFICIENCY_FIELDS = (
    "trainable_parameters",
    "adapter_bytes",
    "write_time_seconds",
    "peak_memory_bytes",
    "median_latency_seconds",
    "mean_input_tokens",
)


def aggregate_experiment(output_dir: Path, bootstrap_samples: int = 10_000) -> Path:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing P0-D manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(output_dir, manifest)
    observed_hashes = prepare_experiment(output_dir)
    if observed_hashes != manifest["compiler_hashes"]:
        raise ValueError("P0-D compiled artifacts differ from the manifest")

    compiled = {item.lesson.lesson_id: item for item in load_compiled_bank(output_dir)}
    rows, seed_metrics, base_lesson_metrics = _load_and_measure(
        output_dir,
        manifest,
        compiled,
    )
    lesson_metrics = [*base_lesson_metrics, *_lesson_metrics(seed_metrics)]
    condition_summary = _condition_summary(lesson_metrics, bootstrap_samples)
    contrasts = _contrasts(
        lesson_metrics,
        tuple(manifest["conditions"]),
        str(manifest["config"]["reference_condition_id"]),
        bootstrap_samples,
    )
    seed_stability = _seed_stability(seed_metrics)
    heterogeneity = _heterogeneity(lesson_metrics)
    gates = _gate_status(
        manifest,
        lesson_metrics,
        seed_metrics,
        contrasts,
        bootstrap_samples,
    )
    summary = {
        "schema_version": "p0d-summary-v1",
        "run_id": manifest["run_id"],
        "stage": manifest["config"]["stage"],
        "selected_lesson_count": len(manifest["selected_lessons"]),
        "condition_count": len(manifest["conditions"]),
        "conditions": manifest["conditions"],
        "adapter_unit_count": len(manifest["units"]),
        "probe_row_count": len(rows),
        "condition_summary": condition_summary,
        "contrasts": contrasts,
        "seed_metrics": seed_metrics,
        "lesson_metrics": lesson_metrics,
        "seed_stability": seed_stability,
        "heterogeneity": heterogeneity,
        "gates": gates,
        "no_fabrication_status": (
            "All values were aggregated from verified P0-D raw records."
        ),
    }
    aggregate_dir = output_dir / "results" / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    summary_path = aggregate_dir / "summary.json"
    _atomic_json_write(summary_path, summary)
    (aggregate_dir / "main_table.md").write_text(
        _render_main_table(condition_summary, contrasts),
        encoding="utf-8",
    )
    _atomic_json_write(
        aggregate_dir / "next_stage_candidates.json",
        {
            "schema_version": "p0d-next-stage-v1",
            "source_run_id": manifest["run_id"],
            "status": (
                "manual_freeze_required"
                if gates["eligible_locus_conditions"]
                else "not_eligible"
            ),
            "eligible_locus_conditions": gates["eligible_locus_conditions"],
            "selection_automatic": False,
            "instruction": (
                "Freeze explicit layer/window candidates before running narrow_scan. "
                "Do not select candidates by editing this aggregate."
            ),
        },
    )
    return summary_path


def _validate_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != "p0d-manifest-v1":
        raise ValueError("not a P0-D v1 manifest")
    required = {
        "run_id",
        "config",
        "compiler_hashes",
        "selected_lessons",
        "conditions",
        "base_arms",
        "units",
    }
    if not required.issubset(manifest):
        raise ValueError(f"P0-D manifest is missing {sorted(required - manifest.keys())}")
    if manifest.get("errors"):
        raise ValueError("P0-D manifest contains recorded errors")
    config = manifest["config"]
    lesson_ids = [str(value) for value in manifest["selected_lessons"]]
    seeds = [int(seed) for seed in config["training_seeds"]]
    condition_ids = list(manifest["conditions"])
    if lesson_ids != [str(value) for value in config["selected_lesson_ids"]]:
        raise ValueError("P0-D selected lesson order differs from frozen config")
    if set(condition_ids) != {
        str(condition["condition_id"]) for condition in config["conditions"]
    }:
        raise ValueError("P0-D condition matrix differs from frozen config")
    expected_units = {
        unit_key(condition_id, lesson_id, seed)
        for condition_id in condition_ids
        for lesson_id in lesson_ids
        for seed in seeds
    }
    if set(manifest["units"]) != expected_units:
        raise ValueError("P0-D unit matrix is incomplete or contains extras")
    if set(manifest["base_arms"]) != set(lesson_ids) or set(
        manifest["base_arms"].values()
    ) != {"verified"}:
        raise ValueError("P0-D base arms are incomplete or unverified")
    num_layers = int(config["num_hidden_layers"])
    for key, unit in manifest["units"].items():
        if (
            unit.get("state") != "verified"
            or float(unit.get("rollback_exact_match_rate", 0.0)) != 1.0
        ):
            raise ValueError(f"P0-D unit is not verified: {key}")
        summary = unit.get("training_summary")
        if not isinstance(summary, dict) or not REQUIRED_TRAINING_FIELDS.issubset(summary):
            raise ValueError(f"P0-D unit training summary is incomplete: {key}")
        if summary["adapter_sha256"] != unit.get("adapter_sha256"):
            raise ValueError(f"P0-D adapter hash disagrees with training summary: {key}")
        condition_id = str(unit["condition_id"])
        lesson_id = str(unit["lesson_id"])
        seed = int(unit["seed"])
        adapter_dir = (
            output_dir
            / "adapters"
            / condition_id
            / lesson_id
            / f"seed-{seed}"
        )
        if _adapter_hash(adapter_dir) != unit.get("adapter_sha256"):
            raise ValueError(f"P0-D adapter bundle changed after verification: {key}")
        metadata_path = adapter_dir / "training_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("summary") != summary:
            raise ValueError(f"P0-D training metadata changed after verification: {key}")
        training_config = metadata.get("config")
        if not isinstance(training_config, dict):
            raise ValueError(f"P0-D training config is missing: {key}")
        if summary["config_sha256"] != _json_hash(training_config):
            raise ValueError(f"P0-D training config hash changed: {key}")
        data_path = (
            output_dir / "compiled" / "training" / f"{lesson_id}.jsonl"
        )
        if summary["training_data_sha256"] != sha256(data_path.read_bytes()).hexdigest():
            raise ValueError(f"P0-D training data hash changed: {key}")
        if summary["model_revision"] != config["model_revision"]:
            raise ValueError(f"P0-D training model revision changed: {key}")
        condition = manifest["conditions"][condition_id]
        expected_training_fields = {
            "model_name": config["model_name"],
            "model_revision": config["model_revision"],
            "layer_band": condition["layer_band"],
            "explicit_layers": condition["explicit_layers"],
            "target_modules": config["target_modules"],
            "rank": config["rank"],
            "alpha": config["alpha"],
            "dropout": 0.0,
            "learning_rate": config["learning_rate"],
            "max_steps": config["max_steps"],
            "max_length": config["max_length"],
            "seed": seed,
            "use_4bit": config["use_4bit"],
            "use_chat_template": True,
            "save_tokenizer": False,
        }
        mismatches = {
            name: (training_config.get(name), expected)
            for name, expected in expected_training_fields.items()
            if training_config.get(name) != expected
        }
        if mismatches:
            raise ValueError(f"P0-D training config provenance mismatch for {key}: {mismatches}")
        observed_layers = summary["selected_layers"]
        normalized = list(range(num_layers) if observed_layers is None else observed_layers)
        if normalized != condition["selected_layers"]:
            raise ValueError(f"P0-D selected layer provenance mismatch: {key}")


def _load_and_measure(
    output_dir: Path,
    manifest: dict[str, Any],
    compiled: dict[str, CompiledLesson],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    config = manifest["config"]
    run_id = str(manifest["run_id"])
    model_name = str(config["model_name"])
    model_revision = str(config["model_revision"])
    lesson_ids = [str(value) for value in manifest["selected_lessons"]]
    seeds = [int(seed) for seed in config["training_seeds"]]
    rows: list[dict[str, Any]] = []
    base_lesson_metrics: list[dict[str, Any]] = []

    for lesson_id in lesson_ids:
        if lesson_id not in compiled:
            raise ValueError(f"P0-D compiled lesson is missing: {lesson_id}")
        item = compiled[lesson_id]
        base_rows = read_probe_results(_base_result_path(output_dir, lesson_id))
        _validate_panel(
            rows=base_rows,
            item=item,
            run_id=run_id,
            model_name=model_name,
            model_revision=model_revision,
            expected_arms={Arm.NO_WRITE, Arm.EXTERNAL},
            seed=None,
            condition_id=None,
            adapter_sha256=None,
        )
        rows.extend(base_rows)
        no_write_accuracy = _category_accuracy(
            [row for row in base_rows if row["arm"] == Arm.NO_WRITE],
            INTERFERENCE_CATEGORIES,
        )
        for arm in (Arm.NO_WRITE, Arm.EXTERNAL):
            arm_rows = [row for row in base_rows if row["arm"] == arm]
            base_lesson_metrics.append(
                {
                    "condition_id": str(arm),
                    "lesson_id": lesson_id,
                    "pair_id": item.lesson.pair_id,
                    "lesson_type": item.lesson.lesson_type,
                    "desired_action": item.lesson.desired_action,
                    "training_seed_count": 0,
                    **_behavior_metrics(arm_rows, no_write_accuracy),
                    "trainable_parameters": 0.0,
                    "adapter_bytes": 0.0,
                    "write_time_seconds": 0.0,
                    "peak_memory_bytes": 0.0,
                    "median_latency_seconds": median(
                        float(row["latency_seconds"]) for row in arm_rows
                    ),
                    "mean_input_tokens": mean(
                        float(row["input_tokens"]) for row in arm_rows
                    ),
                }
            )

    metrics: list[dict[str, Any]] = []
    for condition_id in manifest["conditions"]:
        for lesson_id in lesson_ids:
            item = compiled[lesson_id]
            no_write_accuracy = _category_accuracy(
                [
                    row
                    for row in read_probe_results(
                        _base_result_path(output_dir, lesson_id)
                    )
                    if row["arm"] == Arm.NO_WRITE
                ],
                INTERFERENCE_CATEGORIES,
            )
            for seed in seeds:
                key = unit_key(condition_id, lesson_id, seed)
                unit = manifest["units"][key]
                adapter_hash = str(unit["adapter_sha256"])
                path = _adapter_result_path(
                    output_dir,
                    condition_id,
                    lesson_id,
                    seed,
                )
                adapter_rows = read_probe_results(path)
                _validate_panel(
                    rows=adapter_rows,
                    item=item,
                    run_id=run_id,
                    model_name=model_name,
                    model_revision=model_revision,
                    expected_arms={Arm.PARAMETRIC, Arm.ROLLBACK},
                    seed=seed,
                    condition_id=condition_id,
                    adapter_sha256=adapter_hash,
                    expected_precision=str(unit["training_summary"]["precision"]),
                )
                rows.extend(adapter_rows)
                parametric = [
                    row for row in adapter_rows if row["arm"] == Arm.PARAMETRIC
                ]
                behavior = _behavior_metrics(parametric, no_write_accuracy)
                training = unit["training_summary"]
                metrics.append(
                    {
                        "condition_id": condition_id,
                        "lesson_id": lesson_id,
                        "pair_id": item.lesson.pair_id,
                        "lesson_type": item.lesson.lesson_type,
                        "desired_action": item.lesson.desired_action,
                        "training_seed": seed,
                        **behavior,
                        "trainable_parameters": float(
                            training["trainable_parameters"]
                        ),
                        "adapter_bytes": float(training["adapter_bytes"]),
                        "write_time_seconds": float(training["elapsed_seconds"]),
                        "peak_memory_bytes": float(training["peak_memory_bytes"]),
                        "median_latency_seconds": median(
                            float(row["latency_seconds"]) for row in parametric
                        ),
                        "mean_input_tokens": mean(
                            float(row["input_tokens"]) for row in parametric
                        ),
                    }
                )

    precisions = {str(row["precision"]) for row in rows}
    if len(precisions) != 1:
        raise ValueError(f"P0-D run contains mixed precisions: {sorted(precisions)}")
    return rows, metrics, base_lesson_metrics


def _validate_panel(
    *,
    rows: list[dict[str, Any]],
    item: CompiledLesson,
    run_id: str,
    model_name: str,
    model_revision: str,
    expected_arms: set[Arm],
    seed: int | None,
    condition_id: str | None,
    adapter_sha256: str | None,
    expected_precision: str | None = None,
) -> None:
    probes = {probe.probe_id: probe for probe in item.evaluation_probes}
    expected = {(str(arm), probe_id) for arm in expected_arms for probe_id in probes}
    observed = [(str(row.get("arm")), str(row.get("probe_id"))) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(f"invalid P0-D result panel for {item.lesson.lesson_id}")
    required = {
        "condition_id",
        "generated_text",
        "generated_tokens",
        "input_tokens",
        "latency_seconds",
        "prompt_sha256",
        "correct",
        "invalid",
        "precision",
    }
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"P0-D result row is missing {sorted(missing)}")
        probe = probes[str(row["probe_id"])]
        if (
            row.get("run_id") != run_id
            or row.get("model") != model_name
            or row.get("model_revision") != model_revision
            or row.get("lesson_id") != item.lesson.lesson_id
            or row.get("pair_id") != item.lesson.pair_id
            or row.get("lesson_type") != item.lesson.lesson_type
            or row.get("training_seed") != seed
            or row.get("condition_id") != condition_id
            or row.get("adapter_sha256") != adapter_sha256
            or row.get("category") != probe.category
            or row.get("expected_action") != probe.expected_action
            or (expected_precision is not None and row.get("precision") != expected_precision)
        ):
            raise ValueError(
                f"P0-D result provenance mismatch for {item.lesson.lesson_id}"
            )


def _behavior_metrics(
    rows: list[dict[str, Any]],
    no_write_interference_accuracy: float,
) -> dict[str, float]:
    interference_accuracy = _category_accuracy(rows, INTERFERENCE_CATEGORIES)
    return {
        "target_generalization": _category_accuracy(rows, TARGET_CATEGORIES),
        "interference_regression": (
            no_write_interference_accuracy - interference_accuracy
        ),
        "exact_heldout": _category_accuracy(rows, {"exact_heldout"}),
        "conflict_robustness": _category_accuracy(rows, {"conflict_format"}),
        "invalid_rate": sum(bool(row["invalid"]) for row in rows) / len(rows),
    }


def _category_accuracy(
    rows: Iterable[dict[str, Any]],
    categories: set[str],
) -> float:
    selected = [row for row in rows if str(row["category"]) in categories]
    if not selected:
        raise ValueError(f"no result rows for categories: {sorted(categories)}")
    return sum(bool(row["correct"]) for row in selected) / len(selected)


def _lesson_metrics(seed_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_metrics:
        groups[(str(row["condition_id"]), str(row["lesson_id"]))].append(row)
    metrics: list[dict[str, Any]] = []
    for (condition_id, lesson_id), rows in sorted(groups.items()):
        metrics.append(
            {
                "condition_id": condition_id,
                "lesson_id": lesson_id,
                "pair_id": rows[0]["pair_id"],
                "lesson_type": rows[0]["lesson_type"],
                "desired_action": rows[0]["desired_action"],
                "training_seed_count": len(rows),
                **{
                    field: mean(float(row[field]) for row in rows)
                    for field in (*BEHAVIOR_FIELDS, *EFFICIENCY_FIELDS)
                },
            }
        )
    return metrics


def _condition_summary(
    lesson_metrics: list[dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in lesson_metrics:
        groups[str(row["condition_id"])].append(row)
    result: dict[str, dict[str, Any]] = {}
    for condition_id, rows in sorted(groups.items()):
        values: dict[str, Any] = {"lesson_count": len(rows)}
        for field in BEHAVIOR_FIELDS:
            observations = [float(row[field]) for row in rows]
            values[field] = mean(observations)
            values[f"{field}_ci95"] = _bootstrap_ci(
                observations,
                bootstrap_samples,
            )
        for field in EFFICIENCY_FIELDS:
            observations = [float(row[field]) for row in rows]
            values[field] = mean(observations)
            values[f"{field}_median"] = median(observations)
            values[f"{field}_iqr"] = _iqr(observations)
        result[condition_id] = values
    return result


def _contrasts(
    lesson_metrics: list[dict[str, Any]],
    trained_condition_ids: tuple[str, ...],
    reference_condition_id: str,
    bootstrap_samples: int,
) -> dict[str, dict[str, Any]]:
    by_lesson: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in lesson_metrics:
        by_lesson[str(row["lesson_id"])][str(row["condition_id"])] = row
    result: dict[str, dict[str, Any]] = {}
    comparisons = [
        (condition_id, reference_condition_id)
        for condition_id in trained_condition_ids
        if condition_id != reference_condition_id
    ]
    comparisons.extend(
        (condition_id, str(Arm.NO_WRITE))
        for condition_id in trained_condition_ids
    )
    comparisons.append((str(Arm.EXTERNAL), str(Arm.NO_WRITE)))
    for left, right in comparisons:
        result[f"{left}-vs-{right}"] = _paired_contrast(
            by_lesson,
            left,
            right,
            bootstrap_samples,
        )
    return result


def _paired_contrast(
    by_lesson: dict[str, dict[str, dict[str, Any]]],
    left: str,
    right: str,
    bootstrap_samples: int,
) -> dict[str, Any]:
    return _contrast_summary(
        [
            {
                "lesson_id": lesson_id,
                "lesson_type": conditions[left]["lesson_type"],
                "target_difference": (
                    float(conditions[left]["target_generalization"])
                    - float(conditions[right]["target_generalization"])
                ),
                "interference_difference": (
                    float(conditions[left]["interference_regression"])
                    - float(conditions[right]["interference_regression"])
                ),
            }
            for lesson_id, conditions in sorted(by_lesson.items())
        ],
        bootstrap_samples,
    )


def _contrast_summary(
    rows: list[dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    target = [float(row["target_difference"]) for row in rows]
    interference = [float(row["interference_difference"]) for row in rows]
    return {
        "lesson_count": len(rows),
        "target_mean": mean(target),
        "target_ci95": _bootstrap_ci(target, bootstrap_samples),
        "target_wins_ties_losses": {
            "wins": sum(value > 0 for value in target),
            "ties": sum(value == 0 for value in target),
            "losses": sum(value < 0 for value in target),
        },
        "interference_mean": mean(interference),
        "interference_ci95": _bootstrap_ci(interference, bootstrap_samples),
    }


def _seed_stability(seed_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_metrics:
        groups[(str(row["condition_id"]), str(row["lesson_id"]))].append(row)
    return [
        {
            "condition_id": condition_id,
            "lesson_id": lesson_id,
            "seed_values": {
                str(row["training_seed"]): float(row["target_generalization"])
                for row in sorted(rows, key=lambda value: int(value["training_seed"]))
            },
            "range": (
                max(float(row["target_generalization"]) for row in rows)
                - min(float(row["target_generalization"]) for row in rows)
            ),
        }
        for (condition_id, lesson_id), rows in sorted(groups.items())
    ]


def _heterogeneity(
    lesson_metrics: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for condition_id in sorted(
        {str(row["condition_id"]) for row in lesson_metrics}
    ):
        condition_rows = [
            row for row in lesson_metrics if row["condition_id"] == condition_id
        ]
        fact_k2 = [
            row
            for row in condition_rows
            if row["lesson_type"] == "fact_mapping"
            and row["desired_action"] == "act_k2"
        ]
        other = [row for row in condition_rows if row not in fact_k2]
        result[condition_id] = {
            "fact_mapping_act_k2_count": len(fact_k2),
            "fact_mapping_act_k2_tg": (
                mean(float(row["target_generalization"]) for row in fact_k2)
                if fact_k2
                else None
            ),
            "other_count": len(other),
            "other_tg": mean(float(row["target_generalization"]) for row in other),
            "exploratory_only": True,
        }
    return result


def _gate_status(
    manifest: dict[str, Any],
    lesson_metrics: list[dict[str, Any]],
    seed_metrics: list[dict[str, Any]],
    contrasts: dict[str, dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    config = manifest["config"]
    reference = str(config["reference_condition_id"])
    margin = float(config["retention_margin"])
    condition_ids = list(manifest["conditions"])
    retention: dict[str, bool] = {}
    details: dict[str, Any] = {}
    for condition_id in condition_ids:
        if condition_id == reference:
            continue
        label = f"{condition_id}-vs-{reference}"
        ci = contrasts[label]["target_ci95"]
        retention[condition_id] = float(ci[0]) >= -margin
        details[condition_id] = {
            "retention_margin": margin,
            "target_difference_ci95": ci,
            "retains_full_gain": retention[condition_id],
            "separating_comparisons": [],
        }

    lesson_lookup = {
        (str(row["condition_id"]), str(row["lesson_id"])): row
        for row in lesson_metrics
    }
    seed_lookup = {
        (
            str(row["condition_id"]),
            str(row["lesson_id"]),
            int(row["training_seed"]),
        ): row
        for row in seed_metrics
    }
    lesson_ids = [str(value) for value in manifest["selected_lessons"]]
    seeds = [int(seed) for seed in config["training_seeds"]]
    eligible: list[str] = []
    for candidate in condition_ids:
        if candidate == reference or not retention.get(candidate, False):
            continue
        for comparator in condition_ids:
            if comparator in {candidate, reference}:
                continue
            paired = [
                float(
                    lesson_lookup[(candidate, lesson_id)][
                        "target_generalization"
                    ]
                )
                - float(
                    lesson_lookup[(comparator, lesson_id)][
                        "target_generalization"
                    ]
                )
                for lesson_id in lesson_ids
            ]
            ci = _bootstrap_ci(paired, bootstrap_samples)
            seed_means = {
                str(seed): mean(
                    float(
                        seed_lookup[(candidate, lesson_id, seed)][
                            "target_generalization"
                        ]
                    )
                    - float(
                        seed_lookup[(comparator, lesson_id, seed)][
                            "target_generalization"
                        ]
                    )
                    for lesson_id in lesson_ids
                )
                for seed in seeds
            }
            type_means = {
                lesson_type: mean(
                    float(
                        lesson_lookup[(candidate, lesson_id)][
                            "target_generalization"
                        ]
                    )
                    - float(
                        lesson_lookup[(comparator, lesson_id)][
                            "target_generalization"
                        ]
                    )
                    for lesson_id in lesson_ids
                    if lesson_lookup[(candidate, lesson_id)]["lesson_type"]
                    == lesson_type
                )
                for lesson_type in ("fact_mapping", "procedure_recovery")
            }
            largest = max(range(len(paired)), key=lambda index: abs(paired[index]))
            leave_largest_out_mean = mean(
                value for index, value in enumerate(paired) if index != largest
            )
            passes = (
                float(ci[0]) > 0.0
                and sum(value > 0.0 for value in seed_means.values()) >= 2
                and all(value > 0.0 for value in type_means.values())
                and leave_largest_out_mean > 0.0
            )
            comparison = {
                "comparator": comparator,
                "target_difference_ci95": ci,
                "positive_seed_count": sum(
                    value > 0.0 for value in seed_means.values()
                ),
                "seed_mean_differences": seed_means,
                "lesson_type_mean_differences": type_means,
                "leave_largest_out_mean": leave_largest_out_mean,
                "passes_locus_safeguards": passes,
            }
            details[candidate]["separating_comparisons"].append(comparison)
        if any(
            comparison["passes_locus_safeguards"]
            for comparison in details[candidate]["separating_comparisons"]
        ):
            eligible.append(candidate)
    return {
        "run_valid": True,
        "reference_condition_id": reference,
        "retention_margin": margin,
        "conditions": details,
        "eligible_locus_conditions": eligible,
        "automatic_narrow_scan_started": False,
    }


def _bootstrap_ci(values: list[float], samples: int) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = random.Random(20260726)
    estimates = sorted(mean(rng.choice(values) for _ in values) for _ in range(samples))
    return [
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    ]


def _iqr(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [_percentile(ordered, 0.25), _percentile(ordered, 0.75)]


def _percentile(ordered: list[float], fraction: float) -> float:
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _render_main_table(
    summaries: dict[str, dict[str, Any]],
    contrasts: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "| Condition | TG mean [95% CI] ↑ | IR mean [95% CI] ↓ | "
        "Exact ↑ | Conflict ↑ | Invalid ↓ | Trainable params | "
        "Write time median (s) ↓ | Adapter median (MiB) ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition_id, values in summaries.items():
        lines.append(
            f"| {condition_id} | {_mean_ci(values, 'target_generalization')} | "
            f"{_mean_ci(values, 'interference_regression')} | "
            f"{values['exact_heldout']:.4f} | "
            f"{values['conflict_robustness']:.4f} | "
            f"{values['invalid_rate']:.4f} | "
            f"{values['trainable_parameters_median']:.0f} | "
            f"{values['write_time_seconds_median']:.4f} | "
            f"{values['adapter_bytes_median'] / 1024**2:.4f} |"
        )
    lines.extend(
        [
            "",
            "| Contrast | TG difference [95% CI] | IR difference [95% CI] | W/T/L |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, values in contrasts.items():
        wtl = values["target_wins_ties_losses"]
        lines.append(
            f"| {label} | {values['target_mean']:.4f} "
            f"[{values['target_ci95'][0]:.4f}, {values['target_ci95'][1]:.4f}] | "
            f"{values['interference_mean']:.4f} "
            f"[{values['interference_ci95'][0]:.4f}, "
            f"{values['interference_ci95'][1]:.4f}] | "
            f"{wtl['wins']}/{wtl['ties']}/{wtl['losses']} |"
        )
    return "\n".join(lines) + "\n"


def _mean_ci(values: dict[str, Any], field: str) -> str:
    lower, upper = values[f"{field}_ci95"]
    return f"{values[field]:.4f} [{lower:.4f}, {upper:.4f}]"


def _json_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def _atomic_json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
