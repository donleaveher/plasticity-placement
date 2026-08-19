from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any

from plasticity_placement.p0c.domain import Arm
from plasticity_placement.p0c.runtime import prepare_experiment
from plasticity_placement.p0d.analysis import (
    _atomic_json_write,
    _bootstrap_ci,
    _condition_summary,
    _heterogeneity,
    _lesson_metrics,
    _load_and_measure,
    _paired_contrast,
    _render_main_table,
    _seed_stability,
)
from plasticity_placement.p0d.runtime import (
    REQUIRED_TRAINING_FIELDS,
    _adapter_hash,
    _json_hash,
)
from plasticity_placement.p0d2.config import budget_match_conditions
from plasticity_placement.p0d2.manifest import unit_key

MATCHED_CONDITIONS = ("early-matched", "middle-matched", "late-matched")
BASE_BANDS = ("early-base", "middle-base", "late-base")
BASELINE_BY_MATCHED = {
    "early-matched": "early-base",
    "middle-matched": "middle-base",
    "late-matched": "late-base",
}


def aggregate_experiment(output_dir: Path, bootstrap_samples: int = 10_000) -> Path:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing P0-D2 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest(output_dir, manifest)
    observed_hashes = prepare_experiment(output_dir)
    if observed_hashes != manifest["compiler_hashes"]:
        raise ValueError("P0-D2 compiled artifacts differ from the manifest")

    from plasticity_placement.p0c.compiler import load_compiled_bank

    compiled = {item.lesson.lesson_id: item for item in load_compiled_bank(output_dir)}
    rows, seed_metrics, base_lesson_metrics = _load_and_measure(
        output_dir,
        manifest,
        compiled,
    )
    lesson_metrics = [*base_lesson_metrics, *_lesson_metrics(seed_metrics)]
    condition_summary = _condition_summary(lesson_metrics, bootstrap_samples)
    contrasts, contrast_families = _contrasts(
        lesson_metrics,
        bootstrap_samples,
    )
    seed_stability = _seed_stability(seed_metrics)
    heterogeneity = _heterogeneity(lesson_metrics)
    budget_validation = _budget_summary(manifest)
    gates = _gate_status(
        manifest,
        lesson_metrics,
        seed_metrics,
        contrasts,
        bootstrap_samples,
    )
    summary = {
        "schema_version": "p0d2-summary-v1",
        "run_id": manifest["run_id"],
        "stage": manifest["config"]["stage"],
        "selected_lesson_count": len(manifest["selected_lessons"]),
        "condition_count": len(manifest["conditions"]),
        "conditions": manifest["conditions"],
        "adapter_unit_count": len(manifest["units"]),
        "probe_row_count": len(rows),
        "condition_summary": condition_summary,
        "contrasts": contrasts,
        "contrast_families": contrast_families,
        "budget_validation": budget_validation,
        "seed_metrics": seed_metrics,
        "lesson_metrics": lesson_metrics,
        "seed_stability": seed_stability,
        "heterogeneity": heterogeneity,
        "gates": gates,
        "no_fabrication_status": (
            "All values were aggregated from verified P0-D2 raw records."
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
            "schema_version": "p0d2-next-stage-v1",
            "source_run_id": manifest["run_id"],
            "status": gates["decision_status"],
            "eligible_locus_conditions": gates["eligible_locus_conditions"],
            "retaining_conditions": gates["retaining_conditions"],
            "selection_automatic": False,
            "automatic_narrow_scan_started": False,
            "instruction": gates["next_action"],
        },
    )
    return summary_path


def _validate_manifest_metadata(manifest: dict[str, Any]) -> None:
    """Validate frozen P0-D2 structure and manifest-resident provenance."""
    if manifest.get("schema_version") != "p0d2-manifest-v1":
        raise ValueError("not a P0-D2 v1 manifest")
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
        raise ValueError(f"P0-D2 manifest is missing {sorted(required - manifest.keys())}")
    if manifest.get("errors"):
        raise ValueError("P0-D2 manifest contains recorded errors")
    config = manifest["config"]
    if config.get("schema_version") != "p0d2-config-v1":
        raise ValueError("P0-D2 config schema does not match")
    lesson_ids = [str(value) for value in manifest["selected_lessons"]]
    seeds = [int(seed) for seed in config["training_seeds"]]
    frozen_ids = [condition.condition_id for condition in budget_match_conditions()]
    condition_ids = list(manifest["conditions"])
    if set(condition_ids) != set(frozen_ids):
        raise ValueError("P0-D2 conditions differ from the frozen matrix")
    if lesson_ids != [str(value) for value in config["selected_lesson_ids"]]:
        raise ValueError("P0-D2 selected lesson order differs from frozen config")
    if frozen_ids != [
        str(condition["condition_id"]) for condition in config["conditions"]
    ]:
        raise ValueError("P0-D2 condition order differs from frozen config")
    expected_units = {
        unit_key(condition_id, lesson_id, seed)
        for condition_id in condition_ids
        for lesson_id in lesson_ids
        for seed in seeds
    }
    if set(manifest["units"]) != expected_units:
        raise ValueError("P0-D2 unit matrix is incomplete or contains extras")
    if set(manifest["base_arms"]) != set(lesson_ids) or set(
        manifest["base_arms"].values()
    ) != {"verified"}:
        raise ValueError("P0-D2 base arms are incomplete or unverified")

    num_layers = int(config["num_hidden_layers"])
    tolerance = float(config["budget_tolerance"])
    condition_parameter_values: dict[str, set[int]] = defaultdict(set)
    for key, unit in manifest["units"].items():
        if (
            unit.get("state") != "verified"
            or float(unit.get("rollback_exact_match_rate", 0.0)) != 1.0
        ):
            raise ValueError(f"P0-D2 unit is not verified: {key}")
        summary = unit.get("training_summary")
        if not isinstance(summary, dict) or not REQUIRED_TRAINING_FIELDS.issubset(summary):
            raise ValueError(f"P0-D2 unit training summary is incomplete: {key}")
        if summary["adapter_sha256"] != unit.get("adapter_sha256"):
            raise ValueError(f"P0-D2 adapter hash disagrees with training summary: {key}")
        condition_id = str(unit["condition_id"])
        lesson_id = str(unit["lesson_id"])
        seed = int(unit["seed"])
        if key != unit_key(condition_id, lesson_id, seed):
            raise ValueError(f"P0-D2 unit identity changed: {key}")
        if summary["model_revision"] != config["model_revision"]:
            raise ValueError(f"P0-D2 training model revision changed: {key}")

        condition = manifest["conditions"][condition_id]
        observed_layers = summary["selected_layers"]
        normalized = list(range(num_layers) if observed_layers is None else observed_layers)
        if normalized != condition["selected_layers"]:
            raise ValueError(f"P0-D2 selected layer provenance mismatch: {key}")

        observed_parameters = int(summary["trainable_parameters"])
        condition_parameter_values[condition_id].add(observed_parameters)
        validation = unit.get("budget_validation")
        if not isinstance(validation, dict):
            raise ValueError(f"P0-D2 budget validation is missing: {key}")
        if int(validation["observed_trainable_parameters"]) != observed_parameters:
            raise ValueError(f"P0-D2 budget validation changed: {key}")
        reference_id = condition["budget_reference_condition_id"]
        if reference_id is not None:
            reference_key = unit_key(str(reference_id), lesson_id, seed)
            reference_summary = manifest["units"][reference_key]["training_summary"]
            reference_parameters = int(reference_summary["trainable_parameters"])
            observed_error = abs(observed_parameters - reference_parameters) / reference_parameters
            if (
                validation.get("reference_condition_id") != reference_id
                or int(validation["reference_trainable_parameters"])
                != reference_parameters
                or abs(float(validation["relative_error"]) - observed_error) > 1e-12
                or validation.get("within_tolerance") is not True
                or observed_error > tolerance
            ):
                raise ValueError(f"P0-D2 actual budget mismatch: {key}")
    mixed = {
        condition_id: sorted(values)
        for condition_id, values in condition_parameter_values.items()
        if len(values) != 1
    }
    if mixed:
        raise ValueError(f"P0-D2 condition trainable parameters are mixed: {mixed}")


def _validate_manifest(
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> None:
    """Deep-validate every frozen P0-D2 adapter and its source files."""
    _validate_manifest_metadata(manifest)
    config = manifest["config"]
    units = list(manifest["units"].items())
    total = len(units)
    for index, (key, unit) in enumerate(units, start=1):
        if progress is not None:
            progress(index, total, key)
        summary = unit["training_summary"]
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
            raise ValueError(f"P0-D2 adapter bundle changed after verification: {key}")
        metadata_path = adapter_dir / "training_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("summary") != summary:
            raise ValueError(f"P0-D2 training metadata changed after verification: {key}")
        training_config = metadata.get("config")
        if not isinstance(training_config, dict):
            raise ValueError(f"P0-D2 training config is missing: {key}")
        if summary["config_sha256"] != _json_hash(training_config):
            raise ValueError(f"P0-D2 training config hash changed: {key}")
        data_path = output_dir / "compiled" / "training" / f"{lesson_id}.jsonl"
        if summary["training_data_sha256"] != sha256(data_path.read_bytes()).hexdigest():
            raise ValueError(f"P0-D2 training data hash changed: {key}")

        condition = manifest["conditions"][condition_id]
        expected_training_fields = {
            "model_name": config["model_name"],
            "model_revision": config["model_revision"],
            "layer_band": condition["layer_band"],
            "explicit_layers": [],
            "target_modules": condition["target_modules"],
            "rank": condition["rank"],
            "alpha": condition["alpha"],
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
            raise ValueError(
                f"P0-D2 training config provenance mismatch for {key}: {mismatches}"
            )


def _contrasts(
    lesson_metrics: list[dict[str, Any]],
    bootstrap_samples: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    by_lesson: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in lesson_metrics:
        by_lesson[str(row["lesson_id"])][str(row["condition_id"])] = row
    families: dict[str, list[tuple[str, str]]] = {
        "replication": [
            (condition, "full-base") for condition in BASE_BANDS
        ],
        "matched_vs_full": [
            (condition, "full-base") for condition in MATCHED_CONDITIONS
        ],
        "capacity_rescue": [
            (condition, BASELINE_BY_MATCHED[condition])
            for condition in MATCHED_CONDITIONS
        ],
        "write_gain": [
            (
                condition.condition_id,
                str(Arm.NO_WRITE),
            )
            for condition in budget_match_conditions()
        ],
        "external_anchor": [(str(Arm.EXTERNAL), str(Arm.NO_WRITE))],
    }
    contrasts: dict[str, dict[str, Any]] = {}
    labels: dict[str, list[str]] = {}
    for family, comparisons in families.items():
        family_labels: list[str] = []
        for left, right in comparisons:
            label = f"{left}-vs-{right}"
            if label not in contrasts:
                contrasts[label] = _paired_contrast(
                    by_lesson,
                    left,
                    right,
                    bootstrap_samples,
                )
            family_labels.append(label)
        labels[family] = family_labels
    return contrasts, labels


def _budget_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in manifest["units"].values():
        by_condition[str(unit["condition_id"])].append(unit)
    conditions: dict[str, Any] = {}
    for condition_id, units in sorted(by_condition.items()):
        parameters = {
            int(unit["training_summary"]["trainable_parameters"]) for unit in units
        }
        errors = [
            float(unit["budget_validation"]["relative_error"])
            for unit in units
            if unit["budget_validation"]["relative_error"] is not None
        ]
        conditions[condition_id] = {
            "trainable_parameters": next(iter(parameters)),
            "reference_condition_id": manifest["conditions"][condition_id][
                "budget_reference_condition_id"
            ],
            "max_relative_error": max(errors) if errors else None,
            "within_tolerance": (
                max(errors) <= float(manifest["config"]["budget_tolerance"])
                if errors
                else None
            ),
        }
    return {
        "tolerance": float(manifest["config"]["budget_tolerance"]),
        "reference_condition_id": "full-base",
        "conditions": conditions,
    }


def _gate_status(
    manifest: dict[str, Any],
    lesson_metrics: list[dict[str, Any]],
    seed_metrics: list[dict[str, Any]],
    contrasts: dict[str, dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, Any]:
    config = manifest["config"]
    margin = float(config["retention_margin"])
    details: dict[str, Any] = {}
    retaining: list[str] = []
    materially_inferior: list[str] = []
    for condition_id in MATCHED_CONDITIONS:
        matched_label = f"{condition_id}-vs-full-base"
        rescue_label = f"{condition_id}-vs-{BASELINE_BY_MATCHED[condition_id]}"
        matched_ci = contrasts[matched_label]["target_ci95"]
        rescue_ci = contrasts[rescue_label]["target_ci95"]
        retains = float(matched_ci[0]) >= -margin
        inferior = float(matched_ci[1]) < -margin
        rescue_positive = float(rescue_ci[0]) > 0.0
        if retains:
            retaining.append(condition_id)
        if inferior:
            materially_inferior.append(condition_id)
        details[condition_id] = {
            "retention_margin": margin,
            "matched_vs_full_ci95": matched_ci,
            "retains_full_gain": retains,
            "materially_inferior_to_full": inferior,
            "capacity_rescue_ci95": rescue_ci,
            "capacity_rescue_positive": rescue_positive,
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
    for candidate in retaining:
        for comparator in MATCHED_CONDITIONS:
            if comparator == candidate:
                continue
            paired = [
                float(
                    lesson_lookup[(candidate, lesson_id)]["target_generalization"]
                )
                - float(
                    lesson_lookup[(comparator, lesson_id)]["target_generalization"]
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
            details[candidate]["separating_comparisons"].append(
                {
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
            )
        if any(
            comparison["passes_locus_safeguards"]
            for comparison in details[candidate]["separating_comparisons"]
        ):
            eligible.append(candidate)

    all_inferior = len(materially_inferior) == len(MATCHED_CONDITIONS)
    if eligible:
        decision_status = "manual_freeze_required"
        next_action = (
            "Freeze explicit layer/window candidates in a new narrow-scan attempt; "
            "do not select or start them automatically."
        )
    elif retaining:
        decision_status = "sparse_nonunique"
        next_action = (
            "At least one matched band retains Full but no unique locus passed; "
            "replicate on held-out lessons without result-driven narrow selection."
        )
    elif all_inferior:
        decision_status = "capacity_matched_bands_inferior"
        next_action = (
            "Do not run narrow scan. Replicate the distributed-coverage result on "
            "held-out lessons or a frozen second model."
        )
    else:
        decision_status = "inconclusive"
        next_action = (
            "Do not run narrow scan. Increase independent lesson clusters or run a "
            "pre-registered module ablation."
        )
    return {
        "run_valid": True,
        "reference_condition_id": "full-base",
        "retention_margin": margin,
        "conditions": details,
        "retaining_conditions": retaining,
        "materially_inferior_conditions": materially_inferior,
        "all_matched_materially_inferior": all_inferior,
        "eligible_locus_conditions": eligible,
        "decision_status": decision_status,
        "next_action": next_action,
        "automatic_narrow_scan_started": False,
    }
