from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.compiler import (
    ACTIONS,
    COMPILER_VERSION,
    load_compiled_bank,
    write_compiled_bank,
)
from plasticity_placement.p0c.config import P0CConfig
from plasticity_placement.p0c.domain import Arm, CompiledLesson, Tier
from plasticity_placement.p0c.manifest import RunManifest
from plasticity_placement.p0c.modeling import (
    evaluate_probes,
    load_adapter_model,
    load_base_model,
    read_probe_results,
    release_model,
    resolve_model_revision,
    write_probe_results,
    write_probe_rows,
)
from plasticity_placement.training.config import LayerBand, LoraTrainingConfig
from plasticity_placement.training.lora import train_lora


def prepare_experiment(output_dir: Path, overwrite: bool = False) -> dict[str, str]:
    hashes_path = output_dir / "compiled" / "hashes.json"
    if hashes_path.exists() and not overwrite:
        hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
        _verify_compiled_hashes(output_dir, hashes)
        return hashes
    return write_compiled_bank(output_dir)


def run_experiment(config: P0CConfig) -> Path:
    config = replace(
        config,
        model_revision=resolve_model_revision(
            config.model_name,
            config.model_revision,
        ),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = prepare_experiment(config.output_dir, config.overwrite_compiled)
    run_id = _run_id(config, hashes)
    manifest = RunManifest.load_or_create(
        config.output_dir / "manifest.json",
        run_id=run_id,
        config=config.to_dict(),
        compiler_hashes=hashes,
    )
    _write_environment(config.output_dir)
    compiled = load_compiled_bank(config.output_dir)

    selected_ids = list(manifest.payload.get("selected_lessons", []))
    if not selected_ids:
        if config.lesson_ids:
            by_id = {item.lesson.lesson_id: item for item in compiled}
            missing = [lesson_id for lesson_id in config.lesson_ids if lesson_id not in by_id]
            if missing:
                raise ValueError(f"unknown lesson_ids: {missing}")
            selected = [by_id[lesson_id] for lesson_id in config.lesson_ids]
        elif config.tier in {Tier.DEVELOPMENT, Tier.SMOKE}:
            selected = [item for item in compiled if item.lesson.split == "development"][
                : config.target_lesson_count
            ]
        else:
            screening = _run_screening(config, run_id, compiled)
            selected = select_screened_lessons(
                compiled=compiled,
                screening_results=screening,
                tier=config.tier,
                accuracy_threshold=config.screening_threshold,
                invalid_threshold=config.screening_invalid_threshold,
            )
        selected_ids = [item.lesson.lesson_id for item in selected]
        manifest.set_selected_lessons(selected_ids)
    selected = [item for item in compiled if item.lesson.lesson_id in set(selected_ids)]
    selected.sort(key=lambda item: selected_ids.index(item.lesson.lesson_id))
    _validate_selected_lessons(config, selected)

    _run_base_arms(config, run_id, manifest, selected)
    for item in selected:
        for seed in config.training_seeds:
            _run_adapter_unit(config, run_id, manifest, item, seed)
    return manifest.path


def select_screened_lessons(
    *,
    compiled: Iterable[CompiledLesson],
    screening_results: Iterable[dict[str, Any]],
    tier: Tier,
    accuracy_threshold: float,
    invalid_threshold: float,
) -> list[CompiledLesson]:
    if tier in {Tier.DEVELOPMENT, Tier.SMOKE}:
        return [item for item in compiled if item.lesson.split == "development"][
            : (6 if tier is Tier.DEVELOPMENT else 2)
        ]
    metrics = _screening_metrics(screening_results)
    candidates = [
        item
        for item in compiled
        if item.lesson.split in {"confirmatory", "reserve"}
        and metrics.get(item.lesson.lesson_id, {}).get("accuracy", 1.0) <= accuracy_threshold
        and metrics.get(item.lesson.lesson_id, {}).get("invalid_rate", 1.0) <= invalid_threshold
    ]
    by_pair: dict[str, list[CompiledLesson]] = defaultdict(list)
    for item in candidates:
        by_pair[item.lesson.pair_id].append(item)
    eligible_pairs = [
        pair
        for pair in by_pair.values()
        if len(pair) == 2 and len({item.lesson.lesson_type for item in pair}) == 1
    ]
    eligible_pairs.sort(
        key=lambda pair: (
            0 if pair[0].lesson.split == "confirmatory" else 1,
            pair[0].lesson.pair_id,
        )
    )
    pairs_per_type = 2 if tier is Tier.PILOT else 6
    selected: list[CompiledLesson] = []
    for lesson_type in ("fact_mapping", "procedure_recovery"):
        matching = [pair for pair in eligible_pairs if pair[0].lesson.lesson_type == lesson_type]
        if len(matching) < pairs_per_type:
            raise RuntimeError(
                f"not enough eligible {lesson_type} pairs: {len(matching)} < {pairs_per_type}"
            )
        for pair in _select_balanced_pairs(matching, pairs_per_type):
            selected.extend(sorted(pair, key=lambda item: item.lesson.lesson_id))
    return selected


def _select_balanced_pairs(
    pairs: list[list[CompiledLesson]],
    count: int,
) -> tuple[list[CompiledLesson], ...]:
    def score(
        chosen: tuple[list[CompiledLesson], ...],
    ) -> tuple[int, int, int, tuple[str, ...]]:
        action_counts: dict[str, int] = defaultdict(int)
        for pair in chosen:
            for item in pair:
                action_counts[item.lesson.desired_action] += 1
        counts = [action_counts[action] for action in ACTIONS]
        reserve_count = sum(pair[0].lesson.split == "reserve" for pair in chosen)
        imbalance = max(counts) - min(counts)
        squared = sum(value * value for value in counts)
        pair_ids = tuple(sorted(pair[0].lesson.pair_id for pair in chosen))
        return imbalance, squared, reserve_count, pair_ids

    return min(combinations(pairs, count), key=score)


def _validate_selected_lessons(
    config: P0CConfig,
    selected: list[CompiledLesson],
) -> None:
    if not selected:
        raise RuntimeError("no lessons selected")
    if config.tier in {Tier.DEVELOPMENT, Tier.SMOKE}:
        if any(item.lesson.split != "development" for item in selected):
            raise ValueError(f"tier={config.tier.value} only permits development lessons")
        maximum = config.target_lesson_count
        if len(selected) > maximum:
            raise ValueError(f"tier={config.tier.value} permits at most {maximum} lessons")
        return

    expected = config.target_lesson_count
    if len(selected) != expected:
        raise RuntimeError(
            f"tier={config.tier.value} requires {expected} selected lessons, got {len(selected)}"
        )
    if any(item.lesson.split not in {"confirmatory", "reserve"} for item in selected):
        raise RuntimeError("formal tiers may only use confirmatory/reserve lessons")

    by_pair: dict[str, list[CompiledLesson]] = defaultdict(list)
    for item in selected:
        by_pair[item.lesson.pair_id].append(item)
    if any(
        len(pair) != 2 or len({item.lesson.lesson_type for item in pair}) != 1
        for pair in by_pair.values()
    ):
        raise RuntimeError("formal tiers require complete same-type lesson pairs")

    expected_per_type = expected // 2
    for lesson_type in ("fact_mapping", "procedure_recovery"):
        lessons = [item for item in selected if item.lesson.lesson_type == lesson_type]
        if len(lessons) != expected_per_type:
            raise RuntimeError(
                f"tier={config.tier.value} requires {expected_per_type} "
                f"{lesson_type} lessons, got {len(lessons)}"
            )
        action_counts: dict[str, int] = defaultdict(int)
        for item in lessons:
            action_counts[item.lesson.desired_action] += 1
        if len(action_counts) != 4 or max(action_counts.values()) - min(action_counts.values()) > 1:
            raise RuntimeError(
                f"selected {lesson_type} lessons are not action-balanced: {dict(action_counts)}"
            )


def _run_screening(
    config: P0CConfig,
    run_id: str,
    compiled: tuple[CompiledLesson, ...],
) -> list[dict[str, Any]]:
    path = config.output_dir / "results" / "screening.jsonl"
    if path.exists():
        return read_probe_results(path)
    candidates = [item for item in compiled if item.lesson.split in {"confirmatory", "reserve"}]
    bundle = load_base_model(config)
    try:
        results = [
            result
            for item in candidates
            for result in evaluate_probes(
                bundle=bundle,
                probes=item.screening_probes,
                arm=Arm.NO_WRITE,
                run_id=run_id,
                config=config,
                training_seed=None,
            )
        ]
        write_probe_results(path, results)
    finally:
        release_model(bundle)
    return read_probe_results(path)


def _run_base_arms(
    config: P0CConfig,
    run_id: str,
    manifest: RunManifest,
    selected: list[CompiledLesson],
) -> None:
    if config.calibration_only and config.base_results_cache is not None:
        _validate_base_cache_environment(config.base_results_cache, config)
    failed = [
        item.lesson.lesson_id
        for item in selected
        if manifest.payload["base_arms"].get(item.lesson.lesson_id) == "failed"
    ]
    if failed:
        raise RuntimeError(
            "base-arm failures are immutable; use a new output directory after "
            f"diagnosing: {failed}"
        )
    pending = [item for item in selected if not manifest.base_arm_complete(item.lesson.lesson_id)]
    if not pending:
        return
    remaining: list[CompiledLesson] = []
    for item in pending:
        lesson_id = item.lesson.lesson_id
        path = config.output_dir / "results" / "raw" / lesson_id / "base_arms.jsonl"
        try:
            if path.exists():
                _verify_base_result_rows(config, item, path, run_id)
                manifest.mark_base_arm(lesson_id, "verified")
                continue
            if config.calibration_only and _restore_cached_base_results(
                config,
                item,
                path,
                run_id,
            ):
                _verify_base_result_rows(config, item, path, run_id)
                manifest.mark_base_arm(lesson_id, "verified")
                continue
            remaining.append(item)
        except (RuntimeError, ValueError, OSError) as error:
            manifest.mark_base_arm(lesson_id, "failed")
            manifest.record_error(f"{lesson_id}::base", error)
            raise
    if not remaining:
        return

    bundle = load_base_model(config)
    try:
        for item in remaining:
            lesson_id = item.lesson.lesson_id
            path = config.output_dir / "results" / "raw" / lesson_id / "base_arms.jsonl"
            try:
                no_write = evaluate_probes(
                    bundle=bundle,
                    probes=item.evaluation_probes,
                    arm=Arm.NO_WRITE,
                    run_id=run_id,
                    config=config,
                    training_seed=None,
                )
                results = list(no_write)
                if not config.calibration_only:
                    results.extend(
                        evaluate_probes(
                            bundle=bundle,
                            probes=item.evaluation_probes,
                            arm=Arm.EXTERNAL,
                            run_id=run_id,
                            config=config,
                            training_seed=None,
                            external_note=item.external_note,
                        )
                    )
                write_probe_results(path, results)
                if config.calibration_only and config.base_results_cache is not None:
                    cache_path = config.base_results_cache / f"{lesson_id}.jsonl"
                    if not cache_path.exists():
                        write_probe_results(cache_path, no_write)
                manifest.mark_base_arm(lesson_id, "verified")
            except (RuntimeError, ValueError, OSError) as error:
                manifest.mark_base_arm(lesson_id, "failed")
                manifest.record_error(f"{lesson_id}::base", error)
                raise
    finally:
        release_model(bundle)


def _run_adapter_unit(
    config: P0CConfig,
    run_id: str,
    manifest: RunManifest,
    item: CompiledLesson,
    seed: int,
) -> None:
    lesson_id = item.lesson.lesson_id
    state = manifest.unit_state(lesson_id, seed)
    if state == "verified":
        return
    if state == "failed":
        raise RuntimeError(
            f"unit {lesson_id}::seed-{seed} previously failed and is immutable; "
            "diagnose it and use a new output directory"
        )
    adapter_dir = config.output_dir / "adapters" / lesson_id / f"seed-{seed}"
    result_path = (
        config.output_dir / "results" / "raw" / lesson_id / f"seed-{seed}" / "adapter_arms.jsonl"
    )
    metadata_path = adapter_dir / "training_metadata.json"
    training_config = _adapter_training_config(config, item, seed, adapter_dir)
    try:
        if state == "training" and metadata_path.exists():
            training_summary = _load_training_metadata(
                metadata_path,
                training_config,
                adapter_dir,
            )
            adapter_hash = str(training_summary["adapter_sha256"])
            manifest.mark_unit(
                lesson_id,
                seed,
                "trained",
                adapter_path=str(adapter_dir),
                adapter_sha256=adapter_hash,
                training_summary=training_summary,
            )
            state = "trained"
        if state in {"pending", "training"}:
            manifest.mark_unit(lesson_id, seed, "training")
            summary = train_lora(training_config)
            adapter_hash = _adapter_hash(adapter_dir)
            if adapter_hash != summary.adapter_sha256:
                raise RuntimeError(f"saved adapter hash mismatch for {lesson_id}::seed-{seed}")
            manifest.mark_unit(
                lesson_id,
                seed,
                "trained",
                adapter_path=str(adapter_dir),
                adapter_sha256=adapter_hash,
                training_summary=asdict(summary),
            )
            _clear_accelerator_cache()
            state = "trained"
        adapter_hash = _adapter_hash(adapter_dir)
        recorded_hash = (
            manifest.payload["units"].get(f"{lesson_id}::seed-{seed}", {}).get("adapter_sha256")
        )
        if recorded_hash != adapter_hash:
            raise RuntimeError(f"adapter hash changed for {lesson_id}::seed-{seed}")
        if state == "trained" and result_path.exists():
            _verify_adapter_result_rows(
                config,
                item,
                result_path,
                run_id,
                seed,
                adapter_hash,
            )
            manifest.mark_unit(lesson_id, seed, "evaluated")
            state = "evaluated"
        if state == "evaluated":
            if not result_path.exists():
                raise FileNotFoundError(f"missing evaluated result file: {result_path}")
            _verify_adapter_result_rows(
                config,
                item,
                result_path,
                run_id,
                seed,
                adapter_hash,
            )
            _verify_rollback(config, lesson_id, result_path)
            manifest.mark_unit(lesson_id, seed, "verified")
            return
        if state == "trained":
            bundle = load_adapter_model(config, adapter_dir)
            try:
                parametric = evaluate_probes(
                    bundle=bundle,
                    probes=item.evaluation_probes,
                    arm=Arm.PARAMETRIC,
                    run_id=run_id,
                    config=config,
                    training_seed=seed,
                    adapter_sha256=adapter_hash,
                )
                both = (
                    []
                    if config.calibration_only
                    else evaluate_probes(
                        bundle=bundle,
                        probes=item.evaluation_probes,
                        arm=Arm.BOTH,
                        run_id=run_id,
                        config=config,
                        training_seed=seed,
                        external_note=item.external_note,
                        adapter_sha256=adapter_hash,
                    )
                )
                with bundle.model.disable_adapter():
                    rollback = evaluate_probes(
                        bundle=bundle,
                        probes=item.evaluation_probes,
                        arm=Arm.ROLLBACK,
                        run_id=run_id,
                        config=config,
                        training_seed=seed,
                        adapter_sha256=adapter_hash,
                    )
                write_probe_results(result_path, [*parametric, *both, *rollback])
            finally:
                release_model(bundle)
            manifest.mark_unit(lesson_id, seed, "evaluated")
        _verify_adapter_result_rows(
            config,
            item,
            result_path,
            run_id,
            seed,
            adapter_hash,
        )
        rollback_rate = _verify_rollback(config, lesson_id, result_path)
        manifest.mark_unit(
            lesson_id,
            seed,
            "verified",
            rollback_exact_match_rate=rollback_rate,
        )
    except (RuntimeError, ValueError, OSError) as error:
        manifest.mark_unit(lesson_id, seed, "failed")
        manifest.record_error(f"{lesson_id}::seed-{seed}", error)
        raise


def _adapter_training_config(
    config: P0CConfig,
    item: CompiledLesson,
    seed: int,
    adapter_dir: Path,
) -> LoraTrainingConfig:
    lesson_id = item.lesson.lesson_id
    return LoraTrainingConfig(
        model_name=config.model_name,
        model_revision=config.model_revision,
        data_path=(config.output_dir / "compiled" / "training" / f"{lesson_id}.jsonl"),
        output_dir=adapter_dir,
        layer_band=LayerBand.FULL,
        target_modules=("q_proj", "v_proj"),
        rank=config.rank,
        alpha=config.alpha,
        dropout=0.0,
        learning_rate=config.learning_rate,
        epochs=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        max_length=config.max_length,
        warmup_ratio=0.0,
        max_steps=config.max_steps,
        seed=seed,
        use_4bit=config.use_4bit,
        gradient_checkpointing=True,
        use_chat_template=True,
        save_tokenizer=False,
    )


def _load_training_metadata(
    metadata_path: Path,
    expected_config: LoraTrainingConfig,
    adapter_dir: Path,
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid training metadata object: {metadata_path}")
    if metadata.get("config") != expected_config.to_dict():
        raise ValueError(f"training metadata config mismatch: {metadata_path}")
    summary = metadata.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"training metadata is missing summary: {metadata_path}")
    required = {
        "elapsed_seconds",
        "peak_memory_bytes",
        "adapter_bytes",
        "training_data_sha256",
        "config_sha256",
        "adapter_sha256",
        "model_revision",
        "precision",
    }
    missing = sorted(required - summary.keys())
    if missing:
        raise ValueError(f"training metadata missing fields {missing}: {metadata_path}")
    observed_data_hash = sha256(expected_config.data_path.read_bytes()).hexdigest()
    if summary["training_data_sha256"] != observed_data_hash:
        raise ValueError(f"training data hash mismatch: {metadata_path}")
    config_payload = json.dumps(
        expected_config.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if summary["config_sha256"] != sha256(config_payload.encode()).hexdigest():
        raise ValueError(f"training config hash mismatch: {metadata_path}")
    if summary["adapter_sha256"] != _adapter_hash(adapter_dir):
        raise ValueError(f"adapter hash mismatch: {metadata_path}")
    if summary["model_revision"] != expected_config.model_revision:
        raise ValueError(f"training model revision mismatch: {metadata_path}")
    return summary


def _verify_base_result_rows(
    config: P0CConfig,
    item: CompiledLesson,
    path: Path,
    run_id: str,
) -> None:
    rows = read_probe_results(path)
    _verify_result_rows(
        rows=rows,
        item=item,
        run_id=run_id,
        training_seed=None,
        expected_arms=({Arm.NO_WRITE} if config.calibration_only else {Arm.NO_WRITE, Arm.EXTERNAL}),
        adapter_sha256=None,
    )


def _restore_cached_base_results(
    config: P0CConfig,
    item: CompiledLesson,
    destination: Path,
    run_id: str,
) -> bool:
    if config.base_results_cache is None:
        return False
    cache_path = config.base_results_cache / f"{item.lesson.lesson_id}.jsonl"
    if not cache_path.exists():
        return False
    rows = read_probe_results(cache_path)
    if not rows:
        raise ValueError(f"empty calibration base cache: {cache_path}")
    cache_run_id = str(rows[0].get("run_id"))
    _verify_result_rows(
        rows=rows,
        item=item,
        run_id=cache_run_id,
        training_seed=None,
        expected_arms={Arm.NO_WRITE},
        adapter_sha256=None,
    )
    if any(row.get("model_revision") != config.model_revision for row in rows):
        raise ValueError(f"calibration base cache revision mismatch: {cache_path}")
    restored = [{**row, "run_id": run_id} for row in rows]
    write_probe_rows(destination, restored)
    return True


def _validate_base_cache_environment(
    cache_dir: Path,
    config: P0CConfig,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    config_path = cache_dir / "config.json"
    cache_config = {
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "use_4bit": config.use_4bit,
        "max_length": config.max_length,
        "max_new_tokens": config.max_new_tokens,
    }
    if config_path.exists():
        original_config = json.loads(config_path.read_text(encoding="utf-8"))
        if original_config != cache_config:
            raise RuntimeError(
                f"calibration base cache configuration changed: {original_config} != {cache_config}"
            )
    else:
        _atomic_json_write(config_path, cache_config)

    path = cache_dir / "environment.json"
    current = _environment_snapshot()
    critical = (
        "python",
        "packages",
        "cuda_available",
        "cuda_version",
        "gpu",
        "code_sha256",
    )
    if not path.exists():
        _atomic_json_write(path, current)
        return
    original = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {
        key: (original.get(key), current.get(key))
        for key in critical
        if original.get(key) != current.get(key)
    }
    if mismatches:
        raise RuntimeError(f"calibration base cache environment changed: {mismatches}")


def _verify_adapter_result_rows(
    config: P0CConfig,
    item: CompiledLesson,
    path: Path,
    run_id: str,
    seed: int,
    adapter_sha256: str,
) -> None:
    rows = read_probe_results(path)
    _verify_result_rows(
        rows=rows,
        item=item,
        run_id=run_id,
        training_seed=seed,
        expected_arms=(
            {Arm.PARAMETRIC, Arm.ROLLBACK}
            if config.calibration_only
            else {Arm.PARAMETRIC, Arm.BOTH, Arm.ROLLBACK}
        ),
        adapter_sha256=adapter_sha256,
    )


def _verify_result_rows(
    *,
    rows: list[dict[str, Any]],
    item: CompiledLesson,
    run_id: str,
    training_seed: int | None,
    expected_arms: set[Arm],
    adapter_sha256: str | None,
) -> None:
    expected_probe_ids = {probe.probe_id for probe in item.evaluation_probes}
    expected_keys = {
        (str(arm), probe_id) for arm in expected_arms for probe_id in expected_probe_ids
    }
    observed_keys = [(str(row.get("arm")), str(row.get("probe_id"))) for row in rows]
    if len(observed_keys) != len(set(observed_keys)):
        raise ValueError(f"duplicate result rows for lesson {item.lesson.lesson_id}")
    if set(observed_keys) != expected_keys:
        raise ValueError(f"incomplete result panel for lesson {item.lesson.lesson_id}")
    probes = {probe.probe_id: probe for probe in item.evaluation_probes}
    for row in rows:
        probe = probes[str(row["probe_id"])]
        if (
            row.get("run_id") != run_id
            or row.get("lesson_id") != item.lesson.lesson_id
            or row.get("pair_id") != item.lesson.pair_id
            or row.get("lesson_type") != item.lesson.lesson_type
            or row.get("training_seed") != training_seed
            or row.get("adapter_sha256") != adapter_sha256
            or row.get("category") != probe.category
            or row.get("expected_action") != probe.expected_action
        ):
            raise ValueError(f"result provenance mismatch for lesson {item.lesson.lesson_id}")


def _verify_rollback(config: P0CConfig, lesson_id: str, adapter_path: Path) -> float:
    base_path = config.output_dir / "results" / "raw" / lesson_id / "base_arms.jsonl"
    base = {
        row["probe_id"]: row for row in read_probe_results(base_path) if row["arm"] == Arm.NO_WRITE
    }
    rollback = {
        row["probe_id"]: row
        for row in read_probe_results(adapter_path)
        if row["arm"] == Arm.ROLLBACK
    }
    if base.keys() != rollback.keys():
        raise RuntimeError(f"rollback probe set mismatch for {lesson_id}")
    matches = sum(
        base[probe_id]["generated_text"] == rollback[probe_id]["generated_text"]
        and base[probe_id]["predicted_action"] == rollback[probe_id]["predicted_action"]
        for probe_id in base
    )
    rate = matches / len(base)
    if rate != 1.0:
        raise RuntimeError(f"rollback exact match failed for {lesson_id}: {rate:.3f}")
    return rate


def _screening_metrics(
    results: Iterable[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        groups[str(row["lesson_id"])].append(row)
    return {
        lesson_id: {
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
            "invalid_rate": sum(bool(row["invalid"]) for row in rows) / len(rows),
        }
        for lesson_id, rows in groups.items()
    }


def _run_id(config: P0CConfig, hashes: dict[str, str]) -> str:
    payload = json.dumps(
        {"config": config.to_dict(), "compiler_hashes": hashes},
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = sha256(payload.encode()).hexdigest()[:10]
    return f"p0c-{config.tier.value}-{digest}"


def _verify_compiled_hashes(output_dir: Path, hashes: dict[str, str]) -> None:
    if hashes.get("compiler_version") != COMPILER_VERSION:
        raise ValueError(
            "compiled artifacts use a different compiler version; "
            "use a new output directory or --overwrite"
        )
    compiled_dir = output_dir / "compiled"
    expected = {
        "lessons_sha256": compiled_dir / "lessons.jsonl",
        "probes_sha256": compiled_dir / "probes.jsonl",
        "notes_sha256": compiled_dir / "external_notes.json",
        "leakage_audit_sha256": compiled_dir / "leakage_audit.json",
    }
    for key, path in expected.items():
        if not path.exists():
            raise FileNotFoundError(f"missing compiled artifact: {path}")
        observed = sha256(path.read_bytes()).hexdigest()
        if observed != hashes.get(key):
            raise ValueError(f"compiled artifact hash mismatch: {path}")
    training_dir = compiled_dir / "training"
    digest = sha256()
    for path in sorted(training_dir.glob("*.jsonl")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    if digest.hexdigest() != hashes.get("training_sha256"):
        raise ValueError(f"compiled artifact hash mismatch: {training_dir}")


def _adapter_hash(adapter_dir: Path) -> str:
    candidates = sorted(adapter_dir.glob("adapter_model.*"))
    adapter_config = adapter_dir / "adapter_config.json"
    if len(candidates) != 1 or not adapter_config.exists():
        raise FileNotFoundError(
            f"expected one adapter_model file and adapter_config.json under {adapter_dir}"
        )
    digest = sha256()
    for path in sorted([*candidates, adapter_config]):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _clear_accelerator_cache() -> None:
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_environment(output_dir: Path) -> None:
    path = output_dir / "environment.json"
    environment = _environment_snapshot()
    if path.exists():
        original = json.loads(path.read_text(encoding="utf-8"))
        critical = (
            "python",
            "packages",
            "cuda_available",
            "cuda_version",
            "gpu",
            "code_sha256",
        )
        mismatches = {
            key: (original.get(key), environment.get(key))
            for key in critical
            if original.get(key) != environment.get(key)
        }
        if mismatches:
            raise RuntimeError(
                "runtime environment changed within one experiment; use a new "
                f"output directory: {mismatches}"
            )
    else:
        _atomic_json_write(path, environment)

    sessions_path = output_dir / "environment_sessions.json"
    sessions = (
        json.loads(sessions_path.read_text(encoding="utf-8")) if sessions_path.exists() else []
    )
    if not isinstance(sessions, list):
        raise ValueError(f"invalid environment session history: {sessions_path}")
    sessions.append(
        {
            **environment,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
    )
    _atomic_json_write(sessions_path, sessions)


def _environment_snapshot() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes"):
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[package] = None
    environment: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "code_sha256": _code_hash(),
    }
    try:
        import torch

        environment["cuda_available"] = torch.cuda.is_available()
        environment["cuda_version"] = torch.version.cuda
        environment["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        environment["cuda_available"] = False
        environment["cuda_version"] = None
        environment["gpu"] = None
    return environment


def _git_value(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _code_hash() -> str:
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "src" / "plasticity_placement").rglob("*.py"))
    paths.extend(
        path
        for path in (
            root / "pyproject.toml",
            root / "uv.lock",
            root / "notebooks" / "p0c_colab.ipynb",
        )
        if path.exists()
    )
    digest = sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def current_code_hash() -> str:
    return _code_hash()


def _atomic_json_write(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
