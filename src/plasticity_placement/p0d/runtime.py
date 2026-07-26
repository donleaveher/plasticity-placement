from __future__ import annotations

import gc
import json
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.compiler import ACTIONS, load_compiled_bank
from plasticity_placement.p0c.domain import Arm, CompiledLesson
from plasticity_placement.p0c.modeling import (
    evaluate_probes,
    load_adapter_model,
    load_base_model,
    read_probe_results,
    release_model,
    resolve_model_revision,
    write_probe_rows,
)
from plasticity_placement.p0c.runtime import (
    current_code_hash,
    current_environment_snapshot,
    prepare_experiment,
)
from plasticity_placement.p0d.config import (
    P0DRequest,
    ResolvedLayerCondition,
    ResolvedP0DConfig,
    resolve_conditions,
)
from plasticity_placement.p0d.manifest import P0DManifest, unit_key
from plasticity_placement.training.config import LoraTrainingConfig
from plasticity_placement.training.lora import train_lora

TARGET_MODULES = ("q_proj", "v_proj")
CRITICAL_ENVIRONMENT_KEYS = (
    "python",
    "packages",
    "cuda_available",
    "cuda_version",
    "gpu",
    "code_sha256",
)
REQUIRED_TRAINING_FIELDS = {
    "selected_layers",
    "trainable_parameters",
    "total_parameters",
    "optimizer_steps",
    "elapsed_seconds",
    "peak_memory_bytes",
    "adapter_bytes",
    "training_data_sha256",
    "config_sha256",
    "adapter_sha256",
    "model_revision",
    "precision",
}


def _progress(message: str) -> None:
    print(f"[p0d] {message}", flush=True)


def resolve_request(
    request: P0DRequest,
    *,
    num_hidden_layers: int | None = None,
) -> tuple[ResolvedP0DConfig, dict[str, str], tuple[CompiledLesson, ...]]:
    source_path = request.source_manifest
    if not source_path.exists():
        raise FileNotFoundError(f"missing P0-C source manifest: {source_path}")
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    source_config = _validate_source_manifest(source)

    compiler_hashes = prepare_experiment(request.output_dir)
    if source.get("compiler_hashes") != compiler_hashes:
        raise ValueError(
            "P0-C source compiler hashes do not match the current P0-D compiled bank"
        )
    compiled = load_compiled_bank(request.output_dir)
    selected_ids = tuple(str(value) for value in source["selected_lessons"])
    _validate_selected_lessons(selected_ids, compiled)

    model_name = str(source_config["model_name"])
    model_revision = str(source_config["model_revision"])
    observed_revision = resolve_model_revision(model_name, model_revision)
    if observed_revision != model_revision:
        raise ValueError(
            f"source model revision did not resolve exactly: {observed_revision} != "
            f"{model_revision}"
        )
    resolved_layer_count = (
        num_hidden_layers
        if num_hidden_layers is not None
        else _resolve_num_hidden_layers(model_name, model_revision)
    )
    conditions = resolve_conditions(request.conditions, resolved_layer_count)
    resolved = ResolvedP0DConfig(
        output_dir=request.output_dir,
        stage=request.stage,
        code_sha256=current_code_hash(),
        source_manifest_sha256=sha256(source_bytes).hexdigest(),
        source_run_id=str(source["run_id"]),
        source_calibration_report_sha256=str(
            source_config["calibration_report_sha256"]
        ),
        selected_lesson_ids=selected_ids,
        model_name=model_name,
        model_revision=model_revision,
        use_4bit=bool(source_config["use_4bit"]),
        rank=int(source_config["rank"]),
        alpha=int(source_config["alpha"]),
        learning_rate=float(source_config["learning_rate"]),
        max_steps=int(source_config["max_steps"]),
        max_length=int(source_config["max_length"]),
        max_new_tokens=int(source_config["max_new_tokens"]),
        training_seeds=tuple(int(seed) for seed in source_config["training_seeds"]),
        target_modules=TARGET_MODULES,
        num_hidden_layers=resolved_layer_count,
        conditions=conditions,
        reference_condition_id=request.reference_condition_id,
        retention_margin=request.retention_margin,
        conditions_config_sha256=request.conditions_config_sha256,
        parent_band_run_id=request.parent_band_run_id,
    )
    return resolved, compiler_hashes, compiled


def run_experiment(
    request: P0DRequest,
    *,
    num_hidden_layers: int | None = None,
) -> Path:
    config, compiler_hashes, compiled = resolve_request(
        request,
        num_hidden_layers=num_hidden_layers,
    )
    run_id = _run_id(config, compiler_hashes)
    manifest = P0DManifest.load_or_create(
        config.output_dir / "manifest.json",
        run_id=run_id,
        config=config.identity_dict(),
        compiler_hashes=compiler_hashes,
    )
    _write_environment(config.output_dir)
    selected_by_id = {
        item.lesson.lesson_id: item
        for item in compiled
        if item.lesson.lesson_id in set(config.selected_lesson_ids)
    }
    selected = [selected_by_id[lesson_id] for lesson_id in config.selected_lesson_ids]
    _progress(
        f"starting stage={config.stage.value} run_id={run_id} "
        f"lessons={len(selected)} units={config.expected_unit_count}"
    )
    _run_base_arms(config, run_id, manifest, selected)
    for condition in config.conditions:
        for item in selected:
            for seed in config.training_seeds:
                _run_adapter_unit(config, condition, run_id, manifest, item, seed)
    _progress(f"completed manifest={manifest.path}")
    return manifest.path


def _validate_source_manifest(source: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise ValueError("P0-C source manifest must be a JSON object")
    required = {
        "run_id",
        "config",
        "compiler_hashes",
        "selected_lessons",
        "base_arms",
        "units",
    }
    if not required.issubset(source):
        raise ValueError(f"P0-C source manifest is missing {sorted(required - source.keys())}")
    config = source["config"]
    if not isinstance(config, dict) or config.get("tier") != "confirmatory":
        raise ValueError("P0-D source must be a P0-C confirmatory manifest")
    required_config = {
        "model_name",
        "model_revision",
        "use_4bit",
        "rank",
        "alpha",
        "learning_rate",
        "max_steps",
        "max_length",
        "max_new_tokens",
        "training_seeds",
        "calibration_report_sha256",
    }
    if not required_config.issubset(config):
        raise ValueError(
            f"P0-C source config is missing {sorted(required_config - config.keys())}"
        )
    selected = [str(value) for value in source["selected_lessons"]]
    seeds = [int(value) for value in config["training_seeds"]]
    if len(selected) != 24 or len(set(selected)) != 24:
        raise ValueError("P0-C source must contain 24 unique selected lessons")
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("P0-C source must contain three unique training seeds")
    if not config["model_revision"] or not config["calibration_report_sha256"]:
        raise ValueError("P0-C source lacks immutable model/calibration provenance")
    if source.get("errors"):
        raise ValueError("P0-C source manifest contains recorded errors")
    expected_units = {
        f"{lesson_id}::seed-{seed}" for lesson_id in selected for seed in seeds
    }
    if set(source["units"]) != expected_units:
        raise ValueError("P0-C source adapter unit matrix is incomplete or contains extras")
    if set(source["base_arms"]) != set(selected) or set(source["base_arms"].values()) != {
        "verified"
    }:
        raise ValueError("P0-C source base arms are not complete and verified")
    for key, unit in source["units"].items():
        if (
            unit.get("state") != "verified"
            or float(unit.get("rollback_exact_match_rate", 0.0)) != 1.0
        ):
            raise ValueError(f"P0-C source unit is not verified: {key}")
    return config


def _validate_selected_lessons(
    selected_ids: tuple[str, ...],
    compiled: tuple[CompiledLesson, ...],
) -> None:
    by_id = {item.lesson.lesson_id: item for item in compiled}
    missing = [lesson_id for lesson_id in selected_ids if lesson_id not in by_id]
    if missing:
        raise ValueError(f"P0-C selected lessons are absent from compiler v4: {missing}")
    selected = [by_id[lesson_id] for lesson_id in selected_ids]
    if any(item.lesson.split not in {"confirmatory", "reserve"} for item in selected):
        raise ValueError("P0-D cohort must use confirmatory/reserve lessons only")
    by_pair: dict[str, list[CompiledLesson]] = defaultdict(list)
    for item in selected:
        by_pair[item.lesson.pair_id].append(item)
    if any(
        len(pair) != 2 or len({item.lesson.lesson_type for item in pair}) != 1
        for pair in by_pair.values()
    ):
        raise ValueError("P0-D cohort must preserve complete same-type pairs")
    for lesson_type in ("fact_mapping", "procedure_recovery"):
        typed = [item for item in selected if item.lesson.lesson_type == lesson_type]
        if len(typed) != 12:
            raise ValueError(f"P0-D cohort requires 12 {lesson_type} lessons")
        counts = {
            action: sum(item.lesson.desired_action == action for item in typed)
            for action in ACTIONS
        }
        if set(counts.values()) != {3}:
            raise ValueError(f"P0-D cohort action imbalance for {lesson_type}: {counts}")


def _resolve_num_hidden_layers(model_name: str, model_revision: str) -> int:
    try:
        from transformers import AutoConfig
    except ImportError as error:
        raise RuntimeError(
            "P0-D model dependencies are missing; run "
            "`uv sync --extra train --extra colab`"
        ) from error
    model_config = AutoConfig.from_pretrained(model_name, revision=model_revision)
    for name in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(model_config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("cannot infer transformer layer count from model config")


def _run_id(config: ResolvedP0DConfig, compiler_hashes: dict[str, str]) -> str:
    payload = {
        "config": config.identity_dict(),
        "compiler_hashes": compiler_hashes,
    }
    digest = _json_hash(payload)[:10]
    return f"p0d-{config.stage.value}-{digest}"


def _write_environment(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    current = current_environment_snapshot()
    path = output_dir / "environment.json"
    if path.exists():
        original = json.loads(path.read_text(encoding="utf-8"))
        mismatches = {
            key: (original.get(key), current.get(key))
            for key in CRITICAL_ENVIRONMENT_KEYS
            if original.get(key) != current.get(key)
        }
        if mismatches:
            raise RuntimeError(
                "P0-D runtime environment changed within one attempt; "
                f"use a new stage attempt: {mismatches}"
            )
    else:
        _atomic_json_write(path, current)
    sessions_path = output_dir / "environment_sessions.json"
    sessions = (
        json.loads(sessions_path.read_text(encoding="utf-8"))
        if sessions_path.exists()
        else []
    )
    if not isinstance(sessions, list):
        raise ValueError(f"invalid environment session history: {sessions_path}")
    sessions.append({**current, "recorded_at": datetime.now(UTC).isoformat()})
    _atomic_json_write(sessions_path, sessions)


def _run_base_arms(
    config: ResolvedP0DConfig,
    run_id: str,
    manifest: P0DManifest,
    selected: list[CompiledLesson],
) -> None:
    failed = [
        item.lesson.lesson_id
        for item in selected
        if manifest.base_state(item.lesson.lesson_id) == "failed"
    ]
    if failed:
        raise RuntimeError(
            f"P0-D base-arm failures are immutable; use a new attempt: {failed}"
        )
    pending: list[CompiledLesson] = []
    for item in selected:
        lesson_id = item.lesson.lesson_id
        if manifest.base_state(lesson_id) == "verified":
            continue
        path = _base_result_path(config.output_dir, lesson_id)
        if path.exists():
            _verify_result_rows(
                config=config,
                item=item,
                path=path,
                run_id=run_id,
                training_seed=None,
                expected_arms={Arm.NO_WRITE, Arm.EXTERNAL},
                adapter_sha256=None,
                condition_id=None,
                expected_precision=None,
            )
            manifest.mark_base(lesson_id, "verified")
        else:
            pending.append(item)
    if not pending:
        _progress("all base arms already verified")
        return
    _progress(f"base arms pending={len(pending)}")
    bundle = load_base_model(config)
    try:
        for item in pending:
            lesson_id = item.lesson.lesson_id
            try:
                no_write = evaluate_probes(
                    bundle=bundle,
                    probes=item.evaluation_probes,
                    arm=Arm.NO_WRITE,
                    run_id=run_id,
                    config=config,
                    training_seed=None,
                )
                external = evaluate_probes(
                    bundle=bundle,
                    probes=item.evaluation_probes,
                    arm=Arm.EXTERNAL,
                    run_id=run_id,
                    config=config,
                    training_seed=None,
                    external_note=item.external_note,
                )
                rows = [
                    {**result.to_dict(), "condition_id": None}
                    for result in [*no_write, *external]
                ]
                path = _base_result_path(config.output_dir, lesson_id)
                write_probe_rows(path, rows)
                _verify_result_rows(
                    config=config,
                    item=item,
                    path=path,
                    run_id=run_id,
                    training_seed=None,
                    expected_arms={Arm.NO_WRITE, Arm.EXTERNAL},
                    adapter_sha256=None,
                    condition_id=None,
                    expected_precision=bundle.precision,
                )
                manifest.mark_base(lesson_id, "verified")
                _progress(f"base verified lesson={lesson_id}")
            except (RuntimeError, ValueError, OSError) as error:
                manifest.mark_base(lesson_id, "failed")
                manifest.record_error(f"base::{lesson_id}", error)
                raise
    finally:
        release_model(bundle)


def _run_adapter_unit(
    config: ResolvedP0DConfig,
    condition: ResolvedLayerCondition,
    run_id: str,
    manifest: P0DManifest,
    item: CompiledLesson,
    seed: int,
) -> None:
    lesson_id = item.lesson.lesson_id
    key = unit_key(condition.condition_id, lesson_id, seed)
    state = manifest.unit_state(condition.condition_id, lesson_id, seed)
    if state == "verified":
        return
    if state == "failed":
        raise RuntimeError(f"P0-D unit is immutable after failure: {key}")
    adapter_dir = (
        config.output_dir
        / "adapters"
        / condition.condition_id
        / lesson_id
        / f"seed-{seed}"
    )
    result_path = _adapter_result_path(
        config.output_dir,
        condition.condition_id,
        lesson_id,
        seed,
    )
    metadata_path = adapter_dir / "training_metadata.json"
    training_config = _adapter_training_config(
        config,
        condition,
        item,
        seed,
        adapter_dir,
    )
    try:
        if state == "training" and metadata_path.exists():
            summary = _load_training_metadata(
                metadata_path,
                training_config,
                adapter_dir,
                condition,
                config.num_hidden_layers,
            )
            adapter_hash = str(summary["adapter_sha256"])
            manifest.mark_unit(
                condition.condition_id,
                lesson_id,
                seed,
                "trained",
                adapter_path=str(adapter_dir),
                adapter_sha256=adapter_hash,
                training_summary=summary,
            )
            state = "trained"
        if state in {"pending", "training"}:
            _progress(f"training {key}")
            manifest.mark_unit(condition.condition_id, lesson_id, seed, "training")
            trained = train_lora(training_config)
            summary = asdict(trained)
            _validate_training_summary(
                summary,
                training_config,
                adapter_dir,
                condition,
                config.num_hidden_layers,
            )
            manifest.mark_unit(
                condition.condition_id,
                lesson_id,
                seed,
                "trained",
                adapter_path=str(adapter_dir),
                adapter_sha256=trained.adapter_sha256,
                training_summary=summary,
            )
            _clear_accelerator_cache()
            state = "trained"
        unit = manifest.payload["units"][key]
        adapter_hash = _adapter_hash(adapter_dir)
        if unit.get("adapter_sha256") != adapter_hash:
            raise RuntimeError(f"adapter hash changed for {key}")
        expected_precision = str(unit["training_summary"]["precision"])
        if state == "trained" and result_path.exists():
            _verify_result_rows(
                config=config,
                item=item,
                path=result_path,
                run_id=run_id,
                training_seed=seed,
                expected_arms={Arm.PARAMETRIC, Arm.ROLLBACK},
                adapter_sha256=adapter_hash,
                condition_id=condition.condition_id,
                expected_precision=expected_precision,
            )
            manifest.mark_unit(
                condition.condition_id,
                lesson_id,
                seed,
                "evaluated",
            )
            state = "evaluated"
        if state == "evaluated":
            rollback = _verify_rollback(config.output_dir, lesson_id, result_path)
            manifest.mark_unit(
                condition.condition_id,
                lesson_id,
                seed,
                "verified",
                rollback_exact_match_rate=rollback,
            )
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
                with bundle.model.disable_adapter():
                    rollback_rows = evaluate_probes(
                        bundle=bundle,
                        probes=item.evaluation_probes,
                        arm=Arm.ROLLBACK,
                        run_id=run_id,
                        config=config,
                        training_seed=seed,
                        adapter_sha256=adapter_hash,
                    )
                rows = [
                    {**result.to_dict(), "condition_id": condition.condition_id}
                    for result in [*parametric, *rollback_rows]
                ]
                write_probe_rows(result_path, rows)
            finally:
                release_model(bundle)
            manifest.mark_unit(
                condition.condition_id,
                lesson_id,
                seed,
                "evaluated",
            )
        _verify_result_rows(
            config=config,
            item=item,
            path=result_path,
            run_id=run_id,
            training_seed=seed,
            expected_arms={Arm.PARAMETRIC, Arm.ROLLBACK},
            adapter_sha256=adapter_hash,
            condition_id=condition.condition_id,
            expected_precision=expected_precision,
        )
        rollback = _verify_rollback(config.output_dir, lesson_id, result_path)
        manifest.mark_unit(
            condition.condition_id,
            lesson_id,
            seed,
            "verified",
            rollback_exact_match_rate=rollback,
        )
        _progress(f"verified {key}")
    except (RuntimeError, ValueError, OSError) as error:
        manifest.mark_unit(condition.condition_id, lesson_id, seed, "failed")
        manifest.record_error(key, error)
        _progress(f"failed {key}: {type(error).__name__}: {error}")
        raise


def _adapter_training_config(
    config: ResolvedP0DConfig,
    condition: ResolvedLayerCondition,
    item: CompiledLesson,
    seed: int,
    adapter_dir: Path,
) -> LoraTrainingConfig:
    return LoraTrainingConfig(
        model_name=config.model_name,
        model_revision=config.model_revision,
        data_path=(
            config.output_dir
            / "compiled"
            / "training"
            / f"{item.lesson.lesson_id}.jsonl"
        ),
        output_dir=adapter_dir,
        layer_band=condition.layer_band,
        explicit_layers=condition.explicit_layers,
        target_modules=config.target_modules,
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
    training_config: LoraTrainingConfig,
    adapter_dir: Path,
    condition: ResolvedLayerCondition,
    num_hidden_layers: int,
) -> dict[str, Any]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("config") != training_config.to_dict():
        raise ValueError(f"training metadata config mismatch: {metadata_path}")
    summary = metadata.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"training metadata is missing summary: {metadata_path}")
    _validate_training_summary(
        summary,
        training_config,
        adapter_dir,
        condition,
        num_hidden_layers,
    )
    return summary


def _validate_training_summary(
    summary: dict[str, Any],
    training_config: LoraTrainingConfig,
    adapter_dir: Path,
    condition: ResolvedLayerCondition,
    num_hidden_layers: int,
) -> None:
    missing = REQUIRED_TRAINING_FIELDS - summary.keys()
    if missing:
        raise ValueError(f"training summary is missing fields: {sorted(missing)}")
    observed_layers = summary["selected_layers"]
    normalized_layers = tuple(
        range(num_hidden_layers) if observed_layers is None else observed_layers
    )
    if normalized_layers != condition.selected_layers:
        raise ValueError(
            f"resolved layers changed for {condition.condition_id}: "
            f"{normalized_layers} != {condition.selected_layers}"
        )
    if int(summary["trainable_parameters"]) <= 0:
        raise ValueError("training summary reports no trainable parameters")
    if summary["model_revision"] != training_config.model_revision:
        raise ValueError("training model revision mismatch")
    if summary["training_data_sha256"] != sha256(
        training_config.data_path.read_bytes()
    ).hexdigest():
        raise ValueError("training data hash mismatch")
    if summary["config_sha256"] != _json_hash(training_config.to_dict()):
        raise ValueError("training config hash mismatch")
    if summary["adapter_sha256"] != _adapter_hash(adapter_dir):
        raise ValueError("training adapter hash mismatch")


def _verify_result_rows(
    *,
    config: ResolvedP0DConfig,
    item: CompiledLesson,
    path: Path,
    run_id: str,
    training_seed: int | None,
    expected_arms: set[Arm],
    adapter_sha256: str | None,
    condition_id: str | None,
    expected_precision: str | None,
) -> list[dict[str, Any]]:
    rows = read_probe_results(path)
    probes = {probe.probe_id: probe for probe in item.evaluation_probes}
    expected = {(str(arm), probe_id) for arm in expected_arms for probe_id in probes}
    observed = [(str(row.get("arm")), str(row.get("probe_id"))) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(f"invalid P0-D result panel: {path}")
    required_fields = {
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
        missing = required_fields - row.keys()
        if missing:
            raise ValueError(f"P0-D result row is missing {sorted(missing)}: {path}")
        probe = probes[str(row["probe_id"])]
        if (
            row.get("run_id") != run_id
            or row.get("model") != config.model_name
            or row.get("model_revision") != config.model_revision
            or row.get("lesson_id") != item.lesson.lesson_id
            or row.get("pair_id") != item.lesson.pair_id
            or row.get("lesson_type") != item.lesson.lesson_type
            or row.get("training_seed") != training_seed
            or row.get("adapter_sha256") != adapter_sha256
            or row.get("condition_id") != condition_id
            or row.get("category") != probe.category
            or row.get("expected_action") != probe.expected_action
            or (expected_precision is not None and row.get("precision") != expected_precision)
        ):
            raise ValueError(f"P0-D result provenance mismatch: {path}")
    return rows


def _verify_rollback(output_dir: Path, lesson_id: str, adapter_path: Path) -> float:
    base = {
        row["probe_id"]: row
        for row in read_probe_results(_base_result_path(output_dir, lesson_id))
        if row["arm"] == Arm.NO_WRITE
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


def _base_result_path(output_dir: Path, lesson_id: str) -> Path:
    return output_dir / "results" / "raw" / "base" / lesson_id / "base_arms.jsonl"


def _adapter_result_path(
    output_dir: Path,
    condition_id: str,
    lesson_id: str,
    seed: int,
) -> Path:
    return (
        output_dir
        / "results"
        / "raw"
        / "conditions"
        / condition_id
        / lesson_id
        / f"seed-{seed}"
        / "adapter_arms.jsonl"
    )


def _adapter_hash(adapter_dir: Path) -> str:
    files = sorted(adapter_dir.glob("adapter_model.*"))
    config_path = adapter_dir / "adapter_config.json"
    if config_path.exists():
        files.append(config_path)
    files = sorted(files)
    if not files or not any(path.name.startswith("adapter_model.") for path in files):
        raise FileNotFoundError(f"missing adapter bundle: {adapter_dir}")
    digest = sha256()
    for path in files:
        digest.update(str(path.relative_to(adapter_dir)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _clear_accelerator_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
