from __future__ import annotations

import gc
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.compiler import load_compiled_bank
from plasticity_placement.p0c.domain import Arm, CompiledLesson
from plasticity_placement.p0c.modeling import (
    evaluate_probes,
    load_adapter_model,
    load_base_model,
    release_model,
    resolve_model_revision,
    write_probe_rows,
)
from plasticity_placement.p0c.runtime import (
    current_code_hash,
    prepare_experiment,
)
from plasticity_placement.p0d.runtime import (
    REQUIRED_TRAINING_FIELDS,
    _adapter_hash,
    _adapter_result_path,
    _base_result_path,
    _json_hash,
    _resolve_num_hidden_layers,
    _validate_selected_lessons,
    _validate_source_manifest,
    _verify_result_rows,
    _verify_rollback,
    _write_environment,
)
from plasticity_placement.p0d2.config import (
    P0D2Request,
    ResolvedBudgetCondition,
    ResolvedP0D2Config,
    resolve_conditions,
)
from plasticity_placement.p0d2.manifest import P0D2Manifest, unit_key
from plasticity_placement.training.config import LoraTrainingConfig
from plasticity_placement.training.lora import train_lora


def _progress(message: str) -> None:
    print(f"[p0d2] {message}", flush=True)


def resolve_request(
    request: P0D2Request,
    *,
    num_hidden_layers: int | None = None,
) -> tuple[ResolvedP0D2Config, dict[str, str], tuple[CompiledLesson, ...]]:
    source_path = request.source_manifest
    if not source_path.exists():
        raise FileNotFoundError(f"missing P0-C source manifest: {source_path}")
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    source_config = _validate_source_manifest(source)

    compiler_hashes = prepare_experiment(request.output_dir)
    if source.get("compiler_hashes") != compiler_hashes:
        raise ValueError(
            "P0-C source compiler hashes do not match the current P0-D2 compiled bank"
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
    base_rank = int(source_config["rank"])
    base_alpha = int(source_config["alpha"])
    conditions = resolve_conditions(
        request.conditions,
        num_hidden_layers=resolved_layer_count,
        base_rank=base_rank,
        base_alpha=base_alpha,
    )
    resolved = ResolvedP0D2Config(
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
        base_rank=base_rank,
        base_alpha=base_alpha,
        learning_rate=float(source_config["learning_rate"]),
        max_steps=int(source_config["max_steps"]),
        max_length=int(source_config["max_length"]),
        max_new_tokens=int(source_config["max_new_tokens"]),
        training_seeds=tuple(int(seed) for seed in source_config["training_seeds"]),
        num_hidden_layers=resolved_layer_count,
        conditions=conditions,
        reference_condition_id=request.reference_condition_id,
        retention_margin=request.retention_margin,
        budget_tolerance=request.budget_tolerance,
    )
    return resolved, compiler_hashes, compiled


def run_experiment(
    request: P0D2Request,
    *,
    num_hidden_layers: int | None = None,
) -> Path:
    config, compiler_hashes, compiled = resolve_request(
        request,
        num_hidden_layers=num_hidden_layers,
    )
    run_id = _run_id(config, compiler_hashes)
    manifest = P0D2Manifest.load_or_create(
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


def _run_id(config: ResolvedP0D2Config, compiler_hashes: dict[str, str]) -> str:
    payload = {
        "config": config.identity_dict(),
        "compiler_hashes": compiler_hashes,
    }
    return f"p0d2-{config.stage.value}-{_json_hash(payload)[:10]}"


def _run_base_arms(
    config: ResolvedP0D2Config,
    run_id: str,
    manifest: P0D2Manifest,
    selected: list[CompiledLesson],
) -> None:
    failed = [
        item.lesson.lesson_id
        for item in selected
        if manifest.base_state(item.lesson.lesson_id) == "failed"
    ]
    if failed:
        raise RuntimeError(
            f"P0-D2 base-arm failures are immutable; use a new attempt: {failed}"
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
    config: ResolvedP0D2Config,
    condition: ResolvedBudgetCondition,
    run_id: str,
    manifest: P0D2Manifest,
    item: CompiledLesson,
    seed: int,
) -> None:
    lesson_id = item.lesson.lesson_id
    key = unit_key(condition.condition_id, lesson_id, seed)
    state = manifest.unit_state(condition.condition_id, lesson_id, seed)
    if state == "verified":
        return
    if state == "failed":
        raise RuntimeError(f"P0-D2 unit is immutable after failure: {key}")
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
            budget_validation = _validate_actual_budget(
                config,
                condition,
                manifest,
                lesson_id,
                seed,
                int(summary["trainable_parameters"]),
            )
            manifest.mark_unit(
                condition.condition_id,
                lesson_id,
                seed,
                "trained",
                adapter_path=str(adapter_dir),
                adapter_sha256=str(summary["adapter_sha256"]),
                training_summary=summary,
                budget_validation=budget_validation,
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
            budget_validation = _validate_actual_budget(
                config,
                condition,
                manifest,
                lesson_id,
                seed,
                int(summary["trainable_parameters"]),
            )
            manifest.mark_unit(
                condition.condition_id,
                lesson_id,
                seed,
                "trained",
                adapter_path=str(adapter_dir),
                adapter_sha256=trained.adapter_sha256,
                training_summary=summary,
                budget_validation=budget_validation,
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
    config: ResolvedP0D2Config,
    condition: ResolvedBudgetCondition,
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
        explicit_layers=(),
        target_modules=condition.target_modules,
        rank=condition.rank,
        alpha=condition.alpha,
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
    condition: ResolvedBudgetCondition,
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
    condition: ResolvedBudgetCondition,
    num_hidden_layers: int,
) -> None:
    missing = REQUIRED_TRAINING_FIELDS - summary.keys()
    if missing:
        raise ValueError(f"training summary is missing fields: {sorted(missing)}")
    observed_layers = summary["selected_layers"]
    normalized = tuple(
        range(num_hidden_layers) if observed_layers is None else observed_layers
    )
    if normalized != condition.selected_layers:
        raise ValueError(
            f"training selected layers differ from {condition.condition_id}"
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


def _validate_actual_budget(
    config: ResolvedP0D2Config,
    condition: ResolvedBudgetCondition,
    manifest: P0D2Manifest,
    lesson_id: str,
    seed: int,
    trainable_parameters: int,
) -> dict[str, Any]:
    reference_id = condition.budget_reference_condition_id
    if condition.condition_id == config.reference_condition_id:
        return {
            "reference_condition_id": condition.condition_id,
            "reference_trainable_parameters": trainable_parameters,
            "observed_trainable_parameters": trainable_parameters,
            "relative_error": 0.0,
            "within_tolerance": True,
        }
    if reference_id is None:
        return {
            "reference_condition_id": None,
            "reference_trainable_parameters": None,
            "observed_trainable_parameters": trainable_parameters,
            "relative_error": None,
            "within_tolerance": None,
        }
    reference_key = unit_key(reference_id, lesson_id, seed)
    reference = manifest.payload["units"].get(reference_key)
    if not isinstance(reference, dict) or reference.get("state") != "verified":
        raise RuntimeError(
            f"budget reference must be verified before {condition.condition_id}: "
            f"{reference_key}"
        )
    reference_summary = reference.get("training_summary")
    if not isinstance(reference_summary, dict):
        raise RuntimeError(f"budget reference lacks training summary: {reference_key}")
    reference_parameters = int(reference_summary["trainable_parameters"])
    relative_error = (
        abs(trainable_parameters - reference_parameters) / reference_parameters
    )
    if relative_error > config.budget_tolerance:
        raise RuntimeError(
            f"actual parameter budget mismatch for {condition.condition_id}: "
            f"{trainable_parameters} vs {reference_parameters} "
            f"(relative_error={relative_error:.6f}, "
            f"tolerance={config.budget_tolerance:.6f})"
        )
    return {
        "reference_condition_id": reference_id,
        "reference_trainable_parameters": reference_parameters,
        "observed_trainable_parameters": trainable_parameters,
        "relative_error": relative_error,
        "within_tolerance": True,
    }


def _clear_accelerator_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
