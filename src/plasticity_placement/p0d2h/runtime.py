from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.compiler import ACTIONS, load_compiled_bank
from plasticity_placement.p0c.domain import Arm, CompiledLesson, P0CProbe
from plasticity_placement.p0c.modeling import (
    ModelBundle,
    activate_adapter,
    deactivate_adapter,
    evaluate_probes,
    load_base_model,
    parse_unique_action,
    read_probe_results,
    release_model,
    resolve_model_revision,
    write_probe_rows,
)
from plasticity_placement.p0c.runtime import (
    current_code_hash,
    prepare_experiment,
)
from plasticity_placement.p0d.runtime import (
    _adapter_hash,
    _json_hash,
    _write_environment,
)
from plasticity_placement.p0d2.analysis import (
    _validate_manifest as _validate_p0d2,
)
from plasticity_placement.p0d2.analysis import (
    _validate_manifest_metadata as _validate_p0d2_metadata,
)
from plasticity_placement.p0d2.manifest import unit_key as source_unit_key
from plasticity_placement.p0d2h.config import (
    STRESS_CONDITION_IDS,
    P0D2HRequest,
    ResolvedP0D2HConfig,
    StressCondition,
)
from plasticity_placement.p0d2h.manifest import P0D2HManifest, unit_key
from plasticity_placement.p0d2h.probes import (
    HARD_PROBE_COMPILER_VERSION,
    PROBES_PER_LESSON,
    write_hard_probe_bank,
)

ACTIVE_ADAPTER_NAME = "p0d2h_active"


def _progress(message: str) -> None:
    print(f"[p0d2h] {message}", flush=True)


def _source_validation_progress(index: int, total: int, key: str) -> None:
    if index == 1 or index == total or index % 25 == 0:
        _progress(
            f"source integrity CPU/Drive {index}/{total}: {key}"
        )


def _require_cuda_runtime() -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "P0-D2H requires PyTorch with CUDA support"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            "P0-D2H formal evaluation requires CUDA; "
            "select a GPU runtime and reinstall dependencies"
        )
    _progress(
        f"CUDA ready: {torch.cuda.get_device_name(0)} "
        f"(torch CUDA {torch.version.cuda})"
    )


def _require_cuda_bundle(bundle: ModelBundle) -> None:
    device = next(bundle.model.parameters()).device
    if getattr(device, "type", None) != "cuda":
        raise RuntimeError(
            f"P0-D2H model loaded on {device}, expected a CUDA device"
        )
    _progress(
        f"reusable model loaded on {device}; evaluation_precision={bundle.precision}"
    )


def resolve_request(
    request: P0D2HRequest,
    *,
    deep_source_validation: bool = True,
) -> tuple[
    ResolvedP0D2HConfig,
    dict[str, tuple[P0CProbe, ...]],
    tuple[CompiledLesson, ...],
]:
    source_path = request.source_manifest
    if not source_path.exists():
        raise FileNotFoundError(f"missing P0-D2 source manifest: {source_path}")
    source_dir = source_path.parent
    if request.output_dir.resolve() == source_dir.resolve():
        raise ValueError("P0-D2H output cannot be the source P0-D2 directory")
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    if deep_source_validation:
        _progress(
            "source integrity validation started: 504 adapters on CPU/Drive; "
            "GPU memory remains idle during this block"
        )
        _validate_p0d2(
            source_dir,
            source,
            progress=_source_validation_progress,
        )
        _progress("source integrity validation complete")
    else:
        _validate_p0d2_metadata(source)

    source_compiler_hashes = prepare_experiment(source_dir)
    if source_compiler_hashes != source["compiler_hashes"]:
        raise ValueError("P0-D2 source compiler artifacts changed")
    compiled = load_compiled_bank(source_dir)
    selected_ids = tuple(str(value) for value in source["selected_lessons"])
    compiled_by_id = {item.lesson.lesson_id: item for item in compiled}
    if set(selected_ids) - set(compiled_by_id):
        raise ValueError("P0-D2 selected lessons are missing from its compiled bank")
    selected = tuple(compiled_by_id[lesson_id] for lesson_id in selected_ids)

    summary_path = source_dir / "results" / "aggregate" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"missing P0-D2 source aggregate summary: {summary_path}"
        )
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    if (
        summary.get("schema_version") != "p0d2-summary-v1"
        or summary.get("run_id") != source.get("run_id")
        or summary.get("gates", {}).get("run_valid") is not True
        or int(summary.get("adapter_unit_count", -1)) != 504
        or int(summary.get("probe_row_count", -1)) != 27_456
        or "late-matched"
        not in summary.get("gates", {}).get(
            "eligible_locus_conditions",
            [],
        )
    ):
        raise ValueError(
            "P0-D2 source aggregate is not complete, run-valid, and "
            "late-matched eligible"
        )

    hard_probe_hashes, hard_probe_bank = write_hard_probe_bank(
        request.output_dir,
        selected,
    )
    source_config = source["config"]
    model_name = str(source_config["model_name"])
    model_revision = str(source_config["model_revision"])
    if resolve_model_revision(model_name, model_revision) != model_revision:
        raise ValueError("P0-D2H source model revision did not resolve exactly")

    conditions = tuple(
        _source_condition(source, condition_id, selected_ids)
        for condition_id in STRESS_CONDITION_IDS
    )
    resolved = ResolvedP0D2HConfig(
        output_dir=request.output_dir,
        source_output_dir=source_dir,
        code_sha256=current_code_hash(),
        source_manifest_sha256=sha256(source_bytes).hexdigest(),
        source_run_id=str(source["run_id"]),
        source_summary_sha256=sha256(summary_bytes).hexdigest(),
        source_compiler_hashes={
            str(key): str(value)
            for key, value in source_compiler_hashes.items()
        },
        selected_lesson_ids=selected_ids,
        model_name=model_name,
        model_revision=model_revision,
        use_4bit=bool(source_config["use_4bit"]),
        max_length=int(source_config["max_length"]),
        max_new_tokens=int(source_config["max_new_tokens"]),
        training_seeds=tuple(
            int(seed) for seed in source_config["training_seeds"]
        ),
        conditions=conditions,
        hard_probe_hashes=hard_probe_hashes,
        resilience_margin=request.resilience_margin,
    )
    return resolved, hard_probe_bank, selected


def run_experiment(request: P0D2HRequest) -> Path:
    _require_cuda_runtime()
    config, hard_probe_bank, selected = resolve_request(
        request,
        deep_source_validation=True,
    )
    run_id = _run_id(config)
    manifest = P0D2HManifest.load_or_create(
        config.output_dir / "manifest.json",
        run_id=run_id,
        config=config.identity_dict(),
        source_manifest_path=str(request.source_manifest),
    )
    _write_environment(config.output_dir)
    _progress(
        f"starting run_id={run_id} lessons={len(selected)} "
        f"units={config.expected_unit_count} probes_per_lesson={PROBES_PER_LESSON}"
    )
    bundle: ModelBundle | None = None
    if _requires_model(config, manifest, selected):
        _progress(
            "loading one reusable base model for base arms and all adapters"
        )
        bundle = load_base_model(config)
        _require_cuda_bundle(bundle)
    try:
        bundle = _run_base_arms(
            config,
            run_id,
            manifest,
            selected,
            hard_probe_bank,
            bundle,
        )
        bundle = _run_adapter_units(
            config,
            run_id,
            manifest,
            selected,
            hard_probe_bank,
            bundle,
        )
    finally:
        if bundle is not None:
            release_model(bundle)
    _progress(f"completed manifest={manifest.path}")
    return manifest.path


def _run_id(config: ResolvedP0D2HConfig) -> str:
    return f"p0d2h-hard_probe-{_json_hash(config.identity_dict())[:10]}"


def _requires_model(
    config: ResolvedP0D2HConfig,
    manifest: P0D2HManifest,
    selected: tuple[CompiledLesson, ...],
) -> bool:
    for item in selected:
        lesson_id = item.lesson.lesson_id
        state = manifest.base_state(lesson_id)
        if (
            state not in {"verified", "failed"}
            and not _base_result_path(config.output_dir, lesson_id).exists()
        ):
            return True
    for condition in config.conditions:
        for lesson_id in config.selected_lesson_ids:
            for seed in config.training_seeds:
                state = manifest.unit_state(
                    condition.condition_id,
                    lesson_id,
                    seed,
                )
                if (
                    state not in {"verified", "failed"}
                    and not _adapter_result_path(
                        config.output_dir,
                        condition.condition_id,
                        lesson_id,
                        seed,
                    ).exists()
                ):
                    return True
    return False


def _source_condition(
    source: dict[str, Any],
    condition_id: str,
    lesson_ids: tuple[str, ...],
) -> StressCondition:
    source_condition = source["conditions"].get(condition_id)
    if not isinstance(source_condition, dict):
        raise ValueError(f"P0-D2 source condition is missing: {condition_id}")
    seeds = tuple(int(seed) for seed in source["config"]["training_seeds"])
    parameter_values = {
        int(
            source["units"][
                source_unit_key(condition_id, lesson_id, seed)
            ]["training_summary"]["trainable_parameters"]
        )
        for lesson_id in lesson_ids
        for seed in seeds
    }
    if len(parameter_values) != 1:
        raise ValueError(
            f"P0-D2 source condition has mixed parameter counts: {condition_id}"
        )
    return StressCondition(
        condition_id=condition_id,
        layer_band=str(source_condition["layer_band"]),
        selected_layers=tuple(
            int(layer) for layer in source_condition["selected_layers"]
        ),
        rank=int(source_condition["rank"]),
        alpha=int(source_condition["alpha"]),
        target_modules=tuple(
            str(module) for module in source_condition["target_modules"]
        ),
        trainable_parameters=parameter_values.pop(),
    )


def _run_base_arms(
    config: ResolvedP0D2HConfig,
    run_id: str,
    manifest: P0D2HManifest,
    selected: tuple[CompiledLesson, ...],
    hard_probe_bank: dict[str, tuple[P0CProbe, ...]],
    bundle: ModelBundle | None,
) -> ModelBundle | None:
    failed = [
        item.lesson.lesson_id
        for item in selected
        if manifest.base_state(item.lesson.lesson_id) == "failed"
    ]
    if failed:
        raise RuntimeError(
            f"P0-D2H base failures are immutable; use a new attempt: {failed}"
        )
    pending: list[CompiledLesson] = []
    for item in selected:
        lesson_id = item.lesson.lesson_id
        state = manifest.base_state(lesson_id)
        path = _base_result_path(config.output_dir, lesson_id)
        if state == "verified":
            base_metadata = manifest.payload["base_arms"][lesson_id]
            _verify_rows(
                config,
                run_id,
                path,
                hard_probe_bank[lesson_id],
                lesson_id=lesson_id,
                condition_id=None,
                seed=None,
                expected_arms={Arm.NO_WRITE, Arm.EXTERNAL},
                adapter_sha256=None,
                expected_precision=str(base_metadata["evaluation_precision"]),
                source_training_precision=None,
            )
            continue
        if path.exists():
            evaluation_precision = _verify_rows(
                config,
                run_id,
                path,
                hard_probe_bank[lesson_id],
                lesson_id=lesson_id,
                condition_id=None,
                seed=None,
                expected_arms={Arm.NO_WRITE, Arm.EXTERNAL},
                adapter_sha256=None,
                expected_precision=None,
                source_training_precision=None,
            )
            manifest.mark_base(
                lesson_id,
                "verified",
                result_path=str(path),
                evaluation_precision=evaluation_precision,
            )
            continue
        pending.append(item)
    if not pending:
        _progress("all hard-probe base arms already verified")
        return bundle

    if bundle is None:
        raise RuntimeError("base evaluation requires the reusable CUDA model")
    for item in pending:
        lesson_id = item.lesson.lesson_id
        path = _base_result_path(config.output_dir, lesson_id)
        try:
            _progress(f"base evaluating lesson={lesson_id} on GPU")
            manifest.mark_base(lesson_id, "evaluating")
            no_write = evaluate_probes(
                bundle=bundle,
                probes=hard_probe_bank[lesson_id],
                arm=Arm.NO_WRITE,
                run_id=run_id,
                config=config,
                training_seed=None,
            )
            external = evaluate_probes(
                bundle=bundle,
                probes=hard_probe_bank[lesson_id],
                arm=Arm.EXTERNAL,
                run_id=run_id,
                config=config,
                training_seed=None,
                external_note=item.external_note,
            )
            rows = [
                _stress_row(
                    result.to_dict(),
                    config,
                    condition_id=None,
                    source_training_precision=None,
                )
                for result in [*no_write, *external]
            ]
            write_probe_rows(path, rows)
            evaluation_precision = _verify_rows(
                config,
                run_id,
                path,
                hard_probe_bank[lesson_id],
                lesson_id=lesson_id,
                condition_id=None,
                seed=None,
                expected_arms={Arm.NO_WRITE, Arm.EXTERNAL},
                adapter_sha256=None,
                expected_precision=bundle.precision,
                source_training_precision=None,
            )
            manifest.mark_base(
                lesson_id,
                "verified",
                result_path=str(path),
                evaluation_precision=evaluation_precision,
            )
            _progress(f"base verified lesson={lesson_id}")
        except (RuntimeError, ValueError, OSError) as error:
            manifest.mark_base(lesson_id, "failed")
            manifest.record_error(f"base::{lesson_id}", error)
            raise
    return bundle


def _run_adapter_units(
    config: ResolvedP0D2HConfig,
    run_id: str,
    manifest: P0D2HManifest,
    selected: tuple[CompiledLesson, ...],
    hard_probe_bank: dict[str, tuple[P0CProbe, ...]],
    bundle: ModelBundle | None,
) -> ModelBundle | None:
    selected_by_id = {
        item.lesson.lesson_id: item
        for item in selected
    }
    source_manifest = _load_frozen_source_manifest(config)
    total = config.expected_unit_count
    index = 0
    for condition in config.conditions:
        for lesson_id in config.selected_lesson_ids:
            for seed in config.training_seeds:
                index += 1
                bundle = _run_adapter_unit(
                    config,
                    condition,
                    run_id,
                    manifest,
                    selected_by_id[lesson_id],
                    hard_probe_bank[lesson_id],
                    seed,
                    source_manifest,
                    bundle,
                    index=index,
                    total=total,
                )
    _load_frozen_source_manifest(config)
    return bundle


def _run_adapter_unit(
    config: ResolvedP0D2HConfig,
    condition: StressCondition,
    run_id: str,
    manifest: P0D2HManifest,
    item: CompiledLesson,
    probes: tuple[P0CProbe, ...],
    seed: int,
    source_manifest: dict[str, Any],
    bundle: ModelBundle | None,
    *,
    index: int,
    total: int,
) -> ModelBundle | None:
    lesson_id = item.lesson.lesson_id
    key = unit_key(condition.condition_id, lesson_id, seed)
    state = manifest.unit_state(condition.condition_id, lesson_id, seed)
    if state == "failed":
        raise RuntimeError(f"P0-D2H unit is immutable after failure: {key}")

    source_key = source_unit_key(condition.condition_id, lesson_id, seed)
    source_unit = source_manifest["units"].get(source_key)
    if not isinstance(source_unit, dict) or source_unit.get("state") != "verified":
        raise RuntimeError(f"P0-D2 source unit is no longer verified: {source_key}")
    source_adapter_hash = str(source_unit["adapter_sha256"])
    adapter_dir = (
        config.source_output_dir
        / "adapters"
        / condition.condition_id
        / lesson_id
        / f"seed-{seed}"
    )
    _progress(
        f"unit {index}/{total} validating source adapter on CPU/Drive: {key}"
    )
    if _adapter_hash(adapter_dir) != source_adapter_hash:
        raise RuntimeError(f"P0-D2 source adapter changed: {source_key}")
    result_path = _adapter_result_path(
        config.output_dir,
        condition.condition_id,
        lesson_id,
        seed,
    )
    source_training_precision = str(
        source_unit["training_summary"]["precision"]
    )
    if state == "verified" or result_path.exists():
        unit_metadata = manifest.payload["units"].get(key, {})
        expected_evaluation_precision = (
            str(unit_metadata["evaluation_precision"])
            if state == "verified"
            else None
        )
        if state == "verified" and (
            unit_metadata.get("source_training_precision")
            != source_training_precision
        ):
            raise ValueError(f"P0-D2H source training precision changed: {key}")
        evaluation_precision = _verify_rows(
            config,
            run_id,
            result_path,
            probes,
            lesson_id=lesson_id,
            condition_id=condition.condition_id,
            seed=seed,
            expected_arms={Arm.PARAMETRIC},
            adapter_sha256=source_adapter_hash,
            expected_precision=expected_evaluation_precision,
            source_training_precision=source_training_precision,
        )
        if state != "verified":
            manifest.mark_unit(
                condition.condition_id,
                lesson_id,
                seed,
                "verified",
                source_unit_key=source_key,
                source_adapter_path=str(adapter_dir),
                source_adapter_sha256=source_adapter_hash,
                result_path=str(result_path),
                source_training_precision=source_training_precision,
                evaluation_precision=evaluation_precision,
            )
        _progress(f"unit {index}/{total} already verified: {key}")
        return bundle

    try:
        _progress(f"unit {index}/{total} preparing GPU evaluation: {key}")
        manifest.mark_unit(
            condition.condition_id,
            lesson_id,
            seed,
            "evaluating",
            source_unit_key=source_key,
            source_adapter_path=str(adapter_dir),
            source_adapter_sha256=source_adapter_hash,
            source_training_precision=source_training_precision,
        )
        if bundle is None:
            raise RuntimeError(
                "adapter evaluation requires the reusable CUDA model"
            )
        adapter_active = False
        try:
            activate_adapter(
                bundle,
                adapter_dir,
                adapter_name=ACTIVE_ADAPTER_NAME,
            )
            adapter_active = True
            _progress(f"unit {index}/{total} evaluating on GPU: {key}")
            parametric = evaluate_probes(
                bundle=bundle,
                probes=probes,
                arm=Arm.PARAMETRIC,
                run_id=run_id,
                config=config,
                training_seed=seed,
                adapter_sha256=source_adapter_hash,
            )
            rows = [
                _stress_row(
                    result.to_dict(),
                    config,
                    condition_id=condition.condition_id,
                    source_training_precision=source_training_precision,
                )
                for result in parametric
            ]
            write_probe_rows(result_path, rows)
        finally:
            if adapter_active:
                deactivate_adapter(
                    bundle,
                    adapter_name=ACTIVE_ADAPTER_NAME,
                )
        if _adapter_hash(adapter_dir) != source_adapter_hash:
            raise RuntimeError(
                f"P0-D2 source adapter changed during read-only evaluation: {source_key}"
            )
        evaluation_precision = _verify_rows(
            config,
            run_id,
            result_path,
            probes,
            lesson_id=lesson_id,
            condition_id=condition.condition_id,
            seed=seed,
            expected_arms={Arm.PARAMETRIC},
            adapter_sha256=source_adapter_hash,
            expected_precision=bundle.precision,
            source_training_precision=source_training_precision,
        )
        manifest.mark_unit(
            condition.condition_id,
            lesson_id,
            seed,
            "verified",
            result_path=str(result_path),
            source_training_precision=source_training_precision,
            evaluation_precision=evaluation_precision,
        )
        _progress(f"unit {index}/{total} verified: {key}")
    except (RuntimeError, ValueError, OSError) as error:
        manifest.mark_unit(
            condition.condition_id,
            lesson_id,
            seed,
            "failed",
        )
        manifest.record_error(key, error)
        _progress(f"failed {key}: {type(error).__name__}: {error}")
        raise
    return bundle


def _stress_row(
    row: dict[str, Any],
    config: ResolvedP0D2HConfig,
    *,
    condition_id: str | None,
    source_training_precision: str | None,
) -> dict[str, Any]:
    return {
        **row,
        "condition_id": condition_id,
        "source_training_precision": source_training_precision,
        "source_p0d2_run_id": config.source_run_id,
        "source_manifest_sha256": config.source_manifest_sha256,
        "hard_probe_compiler_version": HARD_PROBE_COMPILER_VERSION,
        "hard_probes_sha256": config.hard_probe_hashes[
            "hard_probes_sha256"
        ],
    }


def _verify_rows(
    config: ResolvedP0D2HConfig,
    run_id: str,
    path: Path,
    probes: tuple[P0CProbe, ...],
    *,
    lesson_id: str,
    condition_id: str | None,
    seed: int | None,
    expected_arms: set[Arm],
    adapter_sha256: str | None,
    expected_precision: str | None,
    source_training_precision: str | None,
) -> str:
    if not path.exists():
        raise FileNotFoundError(f"missing P0-D2H result: {path}")
    rows = read_probe_results(path)
    probe_by_id = {probe.probe_id: probe for probe in probes}
    expected = {
        (arm.value, probe.probe_id)
        for arm in expected_arms
        for probe in probes
    }
    observed = [
        (str(row.get("arm")), str(row.get("probe_id")))
        for row in rows
    ]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(f"P0-D2H result matrix mismatch: {path}")
    precisions = {str(row.get("precision")) for row in rows}
    if len(precisions) != 1 or (
        expected_precision is not None and precisions != {expected_precision}
    ):
        raise ValueError(f"P0-D2H precision mismatch: {path}")
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
    }
    for row in rows:
        missing = required_fields - row.keys()
        if missing:
            raise ValueError(
                f"P0-D2H result row is missing {sorted(missing)}: {path}"
            )
        probe = probe_by_id[str(row["probe_id"])]
        predicted = row.get("predicted_action")
        if (
            row.get("run_id") != run_id
            or row.get("model") != config.model_name
            or row.get("model_revision") != config.model_revision
            or row.get("lesson_id") != lesson_id
            or row.get("pair_id") != probe.pair_id
            or row.get("lesson_type") != probe.lesson_type
            or row.get("category") != probe.category
            or row.get("expected_action") != probe.expected_action
            or row.get("training_seed") != seed
            or row.get("condition_id") != condition_id
            or row.get("adapter_sha256") != adapter_sha256
            or row.get("source_training_precision")
            != source_training_precision
            or row.get("source_p0d2_run_id") != config.source_run_id
            or row.get("source_manifest_sha256")
            != config.source_manifest_sha256
            or row.get("hard_probe_compiler_version")
            != HARD_PROBE_COMPILER_VERSION
            or row.get("hard_probes_sha256")
            != config.hard_probe_hashes["hard_probes_sha256"]
            or not isinstance(row.get("prompt_sha256"), str)
            or len(str(row["prompt_sha256"])) != 64
            or parse_unique_action(str(row["generated_text"]), ACTIONS)
            != predicted
            or bool(row["correct"])
            != (predicted == probe.expected_action)
            or bool(row["invalid"]) != (predicted is None)
            or int(row["input_tokens"]) <= 0
            or int(row["generated_tokens"]) < 0
            or float(row["latency_seconds"]) < 0.0
        ):
            raise ValueError(
                f"P0-D2H result provenance mismatch: "
                f"{path}/{row.get('probe_id')}"
            )
    return next(iter(precisions))


def _load_frozen_source_manifest(
    config: ResolvedP0D2HConfig,
) -> dict[str, Any]:
    path = config.source_output_dir / "manifest.json"
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != config.source_manifest_sha256:
        raise RuntimeError("P0-D2 source manifest changed during stress evaluation")
    manifest = json.loads(payload)
    if manifest.get("run_id") != config.source_run_id:
        raise RuntimeError("P0-D2 source run ID changed during stress evaluation")
    return manifest


def _base_result_path(output_dir: Path, lesson_id: str) -> Path:
    return output_dir / "results" / "base" / f"{lesson_id}.jsonl"


def _adapter_result_path(
    output_dir: Path,
    condition_id: str,
    lesson_id: str,
    seed: int,
) -> Path:
    return (
        output_dir
        / "results"
        / "adapters"
        / condition_id
        / lesson_id
        / f"seed-{seed}.jsonl"
    )
