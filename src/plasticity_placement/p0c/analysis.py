from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, median
from typing import Any

from plasticity_placement.p0c.compiler import load_compiled_bank
from plasticity_placement.p0c.domain import Arm, CompiledLesson
from plasticity_placement.p0c.modeling import read_probe_results

TARGET_CATEGORIES = {"paraphrase", "compositional"}
INTERFERENCE_CATEGORIES = {"near_neighbor", "unrelated_matched"}


def aggregate_experiment(output_dir: Path, bootstrap_samples: int = 10_000) -> Path:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lesson_ids = [str(value) for value in manifest.get("selected_lessons", [])]
    if not lesson_ids:
        raise ValueError("manifest has no selected lessons")
    compiled = {item.lesson.lesson_id: item for item in load_compiled_bank(output_dir)}
    rows = _load_raw_rows(output_dir, manifest, compiled)
    write_metrics = _write_metrics(manifest)
    lesson_metrics = _lesson_arm_metrics(rows, write_metrics)
    calibration_only = bool(manifest.get("config", {}).get("calibration_only"))
    contrasts = (
        _calibration_contrasts(lesson_metrics) if calibration_only else _contrasts(lesson_metrics)
    )
    seed_metrics = _seed_category_metrics(rows)
    summary = {
        "run_id": manifest["run_id"],
        "calibration_only": calibration_only,
        "selected_lesson_count": len(lesson_ids),
        "probe_row_count": len(rows),
        "model_revisions": sorted(
            {str(row["model_revision"]) for row in rows if row.get("model_revision")}
        ),
        "arm_summary": _arm_summary(lesson_metrics, bootstrap_samples),
        "contrasts": _contrast_summary(contrasts, bootstrap_samples),
        "lesson_metrics": lesson_metrics,
        "lesson_contrasts": contrasts,
        "seed_category_metrics": seed_metrics,
        "seed_stability": _seed_stability(seed_metrics),
        "pair_action_diagnostics": _pair_action_diagnostics(lesson_metrics),
        "action_token_diagnostics": _action_token_diagnostics(lesson_metrics),
    }
    aggregate_dir = output_dir / "results" / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    summary_path = aggregate_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (aggregate_dir / "main_table.md").write_text(
        _render_main_table(summary["arm_summary"]),
        encoding="utf-8",
    )
    return summary_path


def _load_raw_rows(
    output_dir: Path,
    manifest: dict[str, Any],
    compiled: dict[str, CompiledLesson],
) -> list[dict[str, Any]]:
    lesson_ids = [str(value) for value in manifest["selected_lessons"]]
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("manifest is missing config")
    training_seeds = tuple(int(value) for value in config.get("training_seeds", []))
    if not training_seeds:
        raise ValueError("manifest config has no training seeds")
    run_id = str(manifest["run_id"])
    base_arms = manifest.get("base_arms", {})
    units = manifest.get("units", {})
    expected_unit_keys = {
        f"{lesson_id}::seed-{seed}" for lesson_id in lesson_ids for seed in training_seeds
    }
    if set(units) != expected_unit_keys:
        raise ValueError(
            "manifest unit set is incomplete or contains extras: "
            f"expected={sorted(expected_unit_keys)}, observed={sorted(units)}"
        )
    for lesson_id in lesson_ids:
        if base_arms.get(lesson_id) != "verified":
            raise ValueError(f"base arms are not verified for {lesson_id}")
        if lesson_id not in compiled:
            raise ValueError(f"selected lesson is missing from compiled bank: {lesson_id}")
        for seed in training_seeds:
            unit = units[f"{lesson_id}::seed-{seed}"]
            if (
                unit.get("state") != "verified"
                or float(unit.get("rollback_exact_match_rate", 0.0)) != 1.0
            ):
                raise ValueError(f"unit is not verified: {lesson_id}::seed-{seed}")
            training_summary = unit.get("training_summary")
            required_summary = {
                "elapsed_seconds",
                "peak_memory_bytes",
                "adapter_bytes",
                "training_data_sha256",
                "config_sha256",
                "adapter_sha256",
                "precision",
                "model_revision",
            }
            if not isinstance(training_summary, dict) or not required_summary.issubset(
                training_summary
            ):
                raise ValueError(f"unit training summary is incomplete: {lesson_id}::seed-{seed}")
            if training_summary["adapter_sha256"] != unit.get("adapter_sha256"):
                raise ValueError(
                    f"unit adapter hash disagrees with training summary: {lesson_id}::seed-{seed}"
                )
            if training_summary["model_revision"] != config.get("model_revision"):
                raise ValueError(f"unit training revision mismatch: {lesson_id}::seed-{seed}")

    rows: list[dict[str, Any]] = []
    for lesson_id in lesson_ids:
        item = compiled[lesson_id]
        lesson_dir = output_dir / "results" / "raw" / lesson_id
        base_path = lesson_dir / "base_arms.jsonl"
        if not base_path.exists():
            raise FileNotFoundError(f"missing base-arm results: {base_path}")
        base_rows = read_probe_results(base_path)
        _validate_result_panel(
            rows=base_rows,
            item=item,
            run_id=run_id,
            seed=None,
            arms=(
                {Arm.NO_WRITE} if config.get("calibration_only") else {Arm.NO_WRITE, Arm.EXTERNAL}
            ),
            adapter_sha256=None,
            model_name=str(config["model_name"]),
            model_revision=config.get("model_revision"),
            expected_precision=None,
        )
        rows.extend(base_rows)

        observed_seed_dirs = {
            path.parent.name for path in lesson_dir.glob("seed-*/adapter_arms.jsonl")
        }
        expected_seed_dirs = {f"seed-{seed}" for seed in training_seeds}
        if observed_seed_dirs != expected_seed_dirs:
            raise ValueError(
                f"adapter result seeds differ for {lesson_id}: "
                f"expected={sorted(expected_seed_dirs)}, "
                f"observed={sorted(observed_seed_dirs)}"
            )
        for seed in training_seeds:
            path = lesson_dir / f"seed-{seed}" / "adapter_arms.jsonl"
            adapter_rows = read_probe_results(path)
            unit = units[f"{lesson_id}::seed-{seed}"]
            adapter_hash = str(unit["adapter_sha256"])
            _validate_result_panel(
                rows=adapter_rows,
                item=item,
                run_id=run_id,
                seed=seed,
                arms=(
                    {Arm.PARAMETRIC, Arm.ROLLBACK}
                    if config.get("calibration_only")
                    else {Arm.PARAMETRIC, Arm.BOTH, Arm.ROLLBACK}
                ),
                adapter_sha256=adapter_hash,
                model_name=str(config["model_name"]),
                model_revision=config.get("model_revision"),
                expected_precision=str(unit["training_summary"]["precision"]),
            )
            rows.extend(adapter_rows)
    precisions = {str(row["precision"]) for row in rows}
    if len(precisions) != 1:
        raise ValueError(f"multiple resolved precisions in one run: {sorted(precisions)}")
    return rows


def _validate_result_panel(
    *,
    rows: list[dict[str, Any]],
    item: CompiledLesson,
    run_id: str,
    seed: int | None,
    arms: set[Arm],
    adapter_sha256: str | None,
    model_name: str,
    model_revision: object,
    expected_precision: str | None,
) -> None:
    probes = {probe.probe_id: probe for probe in item.evaluation_probes}
    expected = {(str(arm), probe_id) for arm in arms for probe_id in probes}
    observed = [(str(row.get("arm")), str(row.get("probe_id"))) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(f"invalid result panel for {item.lesson.lesson_id}")
    required_fields = {
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
        missing = required_fields - row.keys()
        if missing:
            raise ValueError(
                f"result row is missing {sorted(missing)} for "
                f"{item.lesson.lesson_id}/{row.get('probe_id')}"
            )
        probe = probes[str(row["probe_id"])]
        if (
            row.get("run_id") != run_id
            or row.get("model") != model_name
            or row.get("lesson_id") != item.lesson.lesson_id
            or row.get("pair_id") != item.lesson.pair_id
            or row.get("lesson_type") != item.lesson.lesson_type
            or row.get("training_seed") != seed
            or row.get("adapter_sha256") != adapter_sha256
            or row.get("category") != probe.category
            or row.get("expected_action") != probe.expected_action
            or row.get("model_revision") != model_revision
            or (expected_precision is not None and row.get("precision") != expected_precision)
        ):
            raise ValueError(
                f"result provenance mismatch for {item.lesson.lesson_id}/{row.get('probe_id')}"
            )


def _lesson_arm_metrics(
    rows: Iterable[dict[str, Any]],
    write_metrics: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        arm = str(row["arm"])
        if arm == Arm.ROLLBACK:
            continue
        groups[(str(row["lesson_id"]), arm)].append(row)

    base_interference: dict[str, float] = {}
    for (lesson_id, arm), group in groups.items():
        if arm == Arm.NO_WRITE:
            base_interference[lesson_id] = _category_accuracy(group, INTERFERENCE_CATEGORIES)

    metrics: list[dict[str, Any]] = []
    for (lesson_id, arm), group in sorted(groups.items()):
        target = _category_accuracy(group, TARGET_CATEGORIES)
        interference_accuracy = _category_accuracy(group, INTERFERENCE_CATEGORIES)
        metrics.append(
            {
                "lesson_id": lesson_id,
                "pair_id": str(group[0]["pair_id"]),
                "lesson_type": str(group[0]["lesson_type"]),
                "desired_action": str(
                    next(
                        row["expected_action"]
                        for row in group
                        if row["category"] in TARGET_CATEGORIES
                    )
                ),
                "arm": arm,
                "target_generalization": target,
                "interference_regression": (base_interference[lesson_id] - interference_accuracy),
                "exact_heldout": _category_accuracy(group, {"exact_heldout"}),
                "conflict_robustness": _category_accuracy(group, {"conflict_format"}),
                "invalid_rate": sum(bool(row["invalid"]) for row in group) / len(group),
                "mean_input_tokens": mean(float(row["input_tokens"]) for row in group),
                "median_latency_seconds": median(float(row["latency_seconds"]) for row in group),
                "write_time_seconds": (
                    write_metrics.get(lesson_id, {}).get("elapsed_seconds", 0.0)
                    if arm in {Arm.PARAMETRIC, Arm.BOTH}
                    else 0.0
                ),
                "peak_memory_bytes": (
                    write_metrics.get(lesson_id, {}).get("peak_memory_bytes", 0.0)
                    if arm in {Arm.PARAMETRIC, Arm.BOTH}
                    else 0.0
                ),
                "adapter_bytes": (
                    write_metrics.get(lesson_id, {}).get("adapter_bytes", 0.0)
                    if arm in {Arm.PARAMETRIC, Arm.BOTH}
                    else 0.0
                ),
                "training_seed_count": len(
                    {row["training_seed"] for row in group if row["training_seed"] is not None}
                ),
            }
        )
    return metrics


def _arm_summary(
    metrics: list[dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        groups[str(row["arm"])].append(row)
    result: dict[str, dict[str, Any]] = {}
    behavior_fields = (
        "target_generalization",
        "interference_regression",
        "exact_heldout",
        "conflict_robustness",
        "invalid_rate",
    )
    efficiency_fields = (
        "mean_input_tokens",
        "median_latency_seconds",
        "write_time_seconds",
        "peak_memory_bytes",
        "adapter_bytes",
    )
    for arm, rows in sorted(groups.items()):
        values: dict[str, Any] = {}
        for field in behavior_fields:
            observations = [float(row[field]) for row in rows]
            values[field] = mean(observations)
            values[f"{field}_ci95"] = _bootstrap_ci(
                observations,
                bootstrap_samples,
            )
        for field in efficiency_fields:
            observations = [float(row[field]) for row in rows]
            values[field] = mean(observations)
            values[f"{field}_median"] = median(observations)
            values[f"{field}_iqr"] = _iqr(observations)
        result[arm] = values
    return result


def _contrasts(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_lesson: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in metrics:
        by_lesson[str(row["lesson_id"])][str(row["arm"])] = row
    contrasts: list[dict[str, Any]] = []
    required = {
        Arm.NO_WRITE,
        Arm.EXTERNAL,
        Arm.PARAMETRIC,
        Arm.BOTH,
    }
    for lesson_id, arms in sorted(by_lesson.items()):
        if not required.issubset(arms):
            raise ValueError(f"lesson {lesson_id} is missing one or more treatment arms")
        for label, left, right in (
            ("P-N", Arm.PARAMETRIC, Arm.NO_WRITE),
            ("E-N", Arm.EXTERNAL, Arm.NO_WRITE),
            ("P-E", Arm.PARAMETRIC, Arm.EXTERNAL),
        ):
            contrasts.append(_contrast_row(lesson_id, label, arms[left], arms[right]))
        best_single = max(
            (arms[Arm.PARAMETRIC], arms[Arm.EXTERNAL]),
            key=lambda row: float(row["target_generalization"]),
        )
        contrasts.append(
            _contrast_row(
                lesson_id,
                "B-best-single",
                arms[Arm.BOTH],
                best_single,
            )
        )
    return contrasts


def _calibration_contrasts(
    metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_lesson: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in metrics:
        by_lesson[str(row["lesson_id"])][str(row["arm"])] = row
    contrasts: list[dict[str, Any]] = []
    for lesson_id, arms in sorted(by_lesson.items()):
        if not {Arm.NO_WRITE, Arm.PARAMETRIC}.issubset(arms):
            raise ValueError(f"calibration lesson {lesson_id} is missing P or N")
        contrasts.append(
            _contrast_row(
                lesson_id,
                "P-N",
                arms[Arm.PARAMETRIC],
                arms[Arm.NO_WRITE],
            )
        )
    return contrasts


def _write_metrics(manifest: dict[str, Any]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in manifest.get("units", {}).values():
        summary = unit.get("training_summary")
        if isinstance(summary, dict):
            groups[str(unit["lesson_id"])].append(summary)
    metrics: dict[str, dict[str, float]] = {}
    for lesson_id, summaries in groups.items():
        metrics[lesson_id] = {
            field: mean(float(summary.get(field, 0.0)) for summary in summaries)
            for field in ("elapsed_seconds", "peak_memory_bytes", "adapter_bytes")
        }
    return metrics


def _contrast_row(
    lesson_id: str,
    contrast: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    return {
        "lesson_id": lesson_id,
        "pair_id": left["pair_id"],
        "lesson_type": left["lesson_type"],
        "contrast": contrast,
        "target_difference": (
            float(left["target_generalization"]) - float(right["target_generalization"])
        ),
        "interference_difference": (
            float(left["interference_regression"]) - float(right["interference_regression"])
        ),
    }


def _contrast_summary(
    contrasts: list[dict[str, Any]],
    bootstrap_samples: int,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in contrasts:
        groups[str(row["contrast"])].append(row)
    result: dict[str, dict[str, Any]] = {}
    for label, rows in sorted(groups.items()):
        target = [float(row["target_difference"]) for row in rows]
        interference = [float(row["interference_difference"]) for row in rows]
        result[label] = {
            "lesson_count": len(rows),
            "target_mean": mean(target),
            "target_ci95": _bootstrap_ci(target, bootstrap_samples),
            "target_median": median(target),
            "target_iqr": _iqr(target),
            "interference_mean": mean(interference),
            "interference_ci95": _bootstrap_ci(interference, bootstrap_samples),
            "interference_median": median(interference),
            "interference_iqr": _iqr(interference),
            "target_wins_ties_losses": {
                "wins": sum(value > 0 for value in target),
                "ties": sum(value == 0 for value in target),
                "losses": sum(value < 0 for value in target),
            },
            "interference_wins_ties_losses": {
                "wins": sum(value < 0 for value in interference),
                "ties": sum(value == 0 for value in interference),
                "losses": sum(value > 0 for value in interference),
            },
        }
        if label == "P-E":
            target_ci = result[label]["target_ci95"]
            interference_ci = result[label]["interference_ci95"]
            result[label]["practical_equivalence"] = {
                "target_margin": 0.05,
                "interference_margin": 0.05,
                "target_ci_within_margin": (target_ci[0] >= -0.05 and target_ci[1] <= 0.05),
                "interference_ci_within_margin": (
                    interference_ci[0] >= -0.05 and interference_ci[1] <= 0.05
                ),
            }
    return result


def _bootstrap_ci(values: list[float], samples: int) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    rng = random.Random(20260723)
    estimates = sorted(mean(rng.choice(values) for _ in values) for _ in range(samples))
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return [lower, upper]


def _category_accuracy(
    rows: Iterable[dict[str, Any]],
    categories: set[str],
) -> float:
    selected = [row for row in rows if str(row["category"]) in categories]
    if not selected:
        raise ValueError(f"no rows found for categories: {sorted(categories)}")
    return sum(bool(row["correct"]) for row in selected) / len(selected)


def _seed_category_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int | None, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["arm"] == Arm.ROLLBACK:
            continue
        key = (
            str(row["lesson_id"]),
            str(row["arm"]),
            row["training_seed"],
            str(row["category"]),
        )
        groups[key].append(row)
    metrics: list[dict[str, Any]] = []
    for (lesson_id, arm, seed, category), group in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            -1 if item[0][2] is None else item[0][2],
            item[0][3],
        ),
    ):
        metrics.append(
            {
                "lesson_id": lesson_id,
                "pair_id": str(group[0]["pair_id"]),
                "lesson_type": str(group[0]["lesson_type"]),
                "arm": arm,
                "training_seed": seed,
                "category": category,
                "accuracy": sum(bool(row["correct"]) for row in group) / len(group),
                "invalid_rate": sum(bool(row["invalid"]) for row in group) / len(group),
                "probe_count": len(group),
            }
        )
    return metrics


def _seed_stability(
    metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        if (
            row["arm"] in {Arm.PARAMETRIC, Arm.BOTH}
            and row["category"] in TARGET_CATEGORIES
            and row["training_seed"] is not None
        ):
            groups[
                (
                    str(row["lesson_id"]),
                    str(row["arm"]),
                    int(row["training_seed"]),
                )
            ].append(row)
    by_lesson: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for (lesson_id, arm, seed), rows in groups.items():
        target = mean(float(row["accuracy"]) for row in rows)
        by_lesson[(lesson_id, arm)].append((seed, target))
    return [
        {
            "lesson_id": lesson_id,
            "arm": arm,
            "seed_values": {str(seed): value for seed, value in sorted(values)},
            "range": max(value for _, value in values) - min(value for _, value in values),
        }
        for (lesson_id, arm), values in sorted(by_lesson.items())
    ]


def _pair_action_diagnostics(
    metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_pair_arm: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        by_pair_arm[(str(row["pair_id"]), str(row["arm"]))].append(row)
    diagnostics: list[dict[str, Any]] = []
    for (pair_id, arm), rows in sorted(by_pair_arm.items()):
        diagnostics.append(
            {
                "pair_id": pair_id,
                "lesson_type": str(rows[0]["lesson_type"]),
                "arm": arm,
                "actions": {str(row["lesson_id"]): str(row["desired_action"]) for row in rows},
                "target_by_lesson": {
                    str(row["lesson_id"]): float(row["target_generalization"]) for row in rows
                },
                "within_pair_target_range": max(float(row["target_generalization"]) for row in rows)
                - min(float(row["target_generalization"]) for row in rows),
            }
        )
    return diagnostics


def _action_token_diagnostics(
    metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        groups[(str(row["desired_action"]), str(row["arm"]))].append(row)
    return [
        {
            "desired_action": action,
            "arm": arm,
            "lesson_count": len(rows),
            "target_generalization": mean(float(row["target_generalization"]) for row in rows),
            "invalid_rate": mean(float(row["invalid_rate"]) for row in rows),
        }
        for (action, arm), rows in sorted(groups.items())
    ]


def _iqr(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [_percentile(ordered, 0.25), _percentile(ordered, 0.75)]


def _percentile(ordered: list[float], fraction: float) -> float:
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sample")
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _render_main_table(arm_summary: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Arm | Target generalization mean [95% CI] ↑ | "
        "Interference regression mean [95% CI] ↓ | "
        "Exact-heldout mean [95% CI] ↑ | Conflict robustness mean [95% CI] ↑ | "
        "Invalid rate mean [95% CI] ↓ | Write time median [IQR] (s) ↓ | "
        "Peak memory median [IQR] (GiB) ↓ | Adapter median [IQR] (MiB) ↓ | "
        "Input tokens median [IQR] ↓ | Latency median [IQR] (s) ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in (
        Arm.NO_WRITE,
        Arm.EXTERNAL,
        Arm.PARAMETRIC,
        Arm.BOTH,
    ):
        if str(arm) not in arm_summary:
            continue
        values = arm_summary[str(arm)]
        lines.append(
            f"| {arm.value} | {_mean_ci(values, 'target_generalization')} | "
            f"{_mean_ci(values, 'interference_regression')} | "
            f"{_mean_ci(values, 'exact_heldout')} | "
            f"{_mean_ci(values, 'conflict_robustness')} | "
            f"{_mean_ci(values, 'invalid_rate')} | "
            f"{_median_iqr(values, 'write_time_seconds')} | "
            f"{_median_iqr(values, 'peak_memory_bytes', 1024**3)} | "
            f"{_median_iqr(values, 'adapter_bytes', 1024**2)} | "
            f"{_median_iqr(values, 'mean_input_tokens', digits=2)} | "
            f"{_median_iqr(values, 'median_latency_seconds')} |"
        )
    return "\n".join(lines) + "\n"


def _mean_ci(values: dict[str, Any], field: str) -> str:
    lower, upper = values[f"{field}_ci95"]
    return f"{values[field]:.4f} [{lower:.4f}, {upper:.4f}]"


def _median_iqr(
    values: dict[str, Any],
    field: str,
    scale: float = 1.0,
    digits: int = 4,
) -> str:
    lower, upper = values[f"{field}_iqr"]
    center = float(values[f"{field}_median"])
    return f"{center / scale:.{digits}f} [{lower / scale:.{digits}f}, {upper / scale:.{digits}f}]"
