from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.compiler import (
    ACTIONS,
    load_compiled_bank,
)
from plasticity_placement.p0c.domain import Arm, CompiledLesson, P0CProbe
from plasticity_placement.p0c.modeling import (
    ModelBundle,
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
from plasticity_placement.p0d.runtime import _json_hash, _write_environment
from plasticity_placement.p0d2h.probes import (
    HARD_PROBE_COMPILER_VERSION,
    load_hard_probe_bank,
    verify_hard_probe_hashes,
)
from plasticity_placement.p0d2h.prompting import audit_prompt
from plasticity_placement.p0d2hc.config import (
    CALIBRATION_ARMS,
    CANARY_MODEL_ID,
    SOURCE_MODEL_ID,
    CalibrationModel,
    ModelEvaluationConfig,
    P0D2HCRequest,
    ResolvedP0D2HCConfig,
)
from plasticity_placement.p0d2hc.manifest import (
    P0D2HCManifest,
    unit_key,
)
from plasticity_placement.p0d2hc.prompting import (
    prompt_variant_for_arm,
    render_calibration_probe,
)
from plasticity_placement.training.model_utils import load_tokenizer

PromptAuditIndex = dict[
    tuple[str, str, str, str],
    dict[str, Any],
]


def _progress(message: str) -> None:
    print(f"[p0d2hc] {message}", flush=True)


def _require_cuda_runtime() -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "P0-D2H-CAL requires PyTorch with CUDA support"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            "P0-D2H-CAL formal evaluation requires CUDA; select a GPU runtime"
        )
    _progress(
        f"CUDA ready: {torch.cuda.get_device_name(0)} "
        f"(torch CUDA {torch.version.cuda})"
    )


def _require_cuda_bundle(bundle: ModelBundle) -> None:
    device = next(bundle.model.parameters()).device
    if getattr(device, "type", None) != "cuda":
        raise RuntimeError(
            f"calibration model loaded on {device}, expected CUDA"
        )


def resolve_request(
    request: P0D2HCRequest,
) -> tuple[
    ResolvedP0D2HCConfig,
    dict[str, tuple[P0CProbe, ...]],
    tuple[CompiledLesson, ...],
]:
    source_path = request.source_manifest
    if not source_path.exists():
        raise FileNotFoundError(
            f"missing P0-D2H-R source manifest: {source_path}"
        )
    source_dir = source_path.parent
    if request.output_dir.resolve() == source_dir.resolve():
        raise ValueError(
            "P0-D2H-CAL output cannot be the P0-D2H-R source directory"
        )
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    source_config = source.get("config", {})
    if (
        source.get("schema_version") != "p0d2h-manifest-v1"
        or source_config.get("schema_version") != "p0d2h-config-v2"
        or source_config.get("stage") != "hard_probe"
        or source.get("errors")
        or len(source.get("selected_lessons", [])) != 24
        or len(source.get("units", {})) != 288
        or any(
            unit.get("state") != "verified"
            for unit in source.get("units", {}).values()
        )
        or len(source.get("base_arms", {})) != 24
        or any(
            unit.get("state") != "verified"
            for unit in source.get("base_arms", {}).values()
        )
    ):
        raise ValueError(
            "P0-D2H-R source manifest is not complete and verified"
        )

    summary_path = source_dir / "results" / "aggregate" / "summary.json"
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    if (
        summary.get("schema_version") != "p0d2h-summary-v2"
        or summary.get("run_id") != source.get("run_id")
        or summary.get("adapter_unit_count") != 288
        or summary.get("probe_row_count") != 5_376
        or summary.get("gates", {}).get("run_valid") is not True
    ):
        raise ValueError(
            "P0-D2H-R source aggregate is not complete and run-valid"
        )

    source_audit_path = (
        source_dir / "preflight" / "prompt_token_audit.json"
    )
    source_audit_bytes = source_audit_path.read_bytes()
    source_audit = json.loads(source_audit_bytes)
    if (
        sha256(source_audit_bytes).hexdigest()
        != source.get("prompt_token_audit_sha256")
        or source_audit.get("all_prompts_fit") is not True
        or int(source_audit.get("input_truncated_count", -1)) != 0
        or int(source_audit.get("output_instruction_missing_count", -1))
        != 0
    ):
        raise ValueError(
            "P0-D2H-R source prompt-token audit is not valid"
        )

    source_p0d2_path = Path(str(source["source_manifest_path"]))
    source_p0d2_bytes = source_p0d2_path.read_bytes()
    if (
        sha256(source_p0d2_bytes).hexdigest()
        != source_config["source_manifest_sha256"]
    ):
        raise ValueError("P0-D2 source manifest changed")
    source_p0d2 = json.loads(source_p0d2_bytes)
    if source_p0d2.get("run_id") != source_config["source_run_id"]:
        raise ValueError("P0-D2 source run ID changed")
    source_p0d2_dir = source_p0d2_path.parent
    if not (source_p0d2_dir / "compiled" / "hashes.json").exists():
        raise FileNotFoundError(
            "P0-D2 source compiled hashes are missing; refusing to write "
            "inside the read-only source"
        )
    compiler_hashes = prepare_experiment(source_p0d2_dir)
    if compiler_hashes != source_config["source_compiler_hashes"]:
        raise ValueError("P0-D2 compiled lesson bank changed")

    selected_ids = tuple(
        str(value) for value in source["selected_lessons"]
    )
    compiled_by_id = {
        item.lesson.lesson_id: item
        for item in load_compiled_bank(source_p0d2_dir)
    }
    if set(selected_ids) - set(compiled_by_id):
        raise ValueError("calibration lessons are missing from the source bank")
    selected = tuple(compiled_by_id[lesson_id] for lesson_id in selected_ids)

    hard_probe_hashes = {
        str(key): str(value)
        for key, value in source_config["hard_probe_hashes"].items()
    }
    verify_hard_probe_hashes(source_dir, hard_probe_hashes)
    hard_bank = load_hard_probe_bank(source_dir)
    if tuple(hard_bank) != selected_ids:
        raise ValueError("P0-D2H-R hard-probe lesson order changed")

    source_model_name = str(source_config["model_name"])
    source_model_revision = str(source_config["model_revision"])
    if (
        resolve_model_revision(source_model_name, source_model_revision)
        != source_model_revision
    ):
        raise ValueError("source model revision no longer resolves exactly")
    models = [
        CalibrationModel(
            model_id=SOURCE_MODEL_ID,
            role="source",
            model_name=source_model_name,
            model_revision=source_model_revision,
            use_4bit=request.use_4bit,
        )
    ]
    if request.canary_model_name:
        canary_revision = resolve_model_revision(
            request.canary_model_name,
            request.canary_model_revision,
        )
        if not canary_revision:
            raise ValueError(
                "scale canary did not resolve to an immutable revision"
            )
        if (
            request.canary_model_name == source_model_name
            and canary_revision == source_model_revision
        ):
            raise ValueError(
                "scale canary must differ from the source model revision"
            )
        models.append(
            CalibrationModel(
                model_id=CANARY_MODEL_ID,
                role="scale_canary",
                model_name=request.canary_model_name,
                model_revision=canary_revision,
                use_4bit=request.use_4bit,
            )
        )

    resolved = ResolvedP0D2HCConfig(
        output_dir=request.output_dir,
        source_output_dir=source_dir,
        source_p0d2_output_dir=source_p0d2_dir,
        code_sha256=current_code_hash(),
        source_manifest_sha256=sha256(source_bytes).hexdigest(),
        source_run_id=str(source["run_id"]),
        source_summary_sha256=sha256(summary_bytes).hexdigest(),
        source_p0d2_run_id=str(source_p0d2["run_id"]),
        source_compiler_hashes={
            str(key): str(value) for key, value in compiler_hashes.items()
        },
        hard_probe_hashes=hard_probe_hashes,
        selected_lesson_ids=selected_ids,
        models=tuple(models),
        evaluation_max_length=request.evaluation_max_length,
        max_new_tokens=int(source_config["max_new_tokens"]),
        oracle_min_accuracy=request.oracle_min_accuracy,
        oracle_min_category_accuracy=(
            request.oracle_min_category_accuracy
        ),
        oracle_max_invalid_rate=request.oracle_max_invalid_rate,
        external_min_accuracy=request.external_min_accuracy,
        external_max_invalid_rate=request.external_max_invalid_rate,
    )
    return resolved, hard_bank, selected


def audit_request(request: P0D2HCRequest) -> Path:
    config, hard_bank, selected = resolve_request(request)
    path, _, _, _ = _prepare_prompt_token_audit(
        config,
        hard_bank,
        selected,
    )
    return path


def _prepare_prompt_token_audit(
    config: ResolvedP0D2HCConfig,
    hard_bank: dict[str, tuple[P0CProbe, ...]],
    selected: tuple[CompiledLesson, ...],
) -> tuple[Path, str, PromptAuditIndex, dict[str, Any]]:
    audits: PromptAuditIndex = {}
    records: list[dict[str, Any]] = []
    for model in config.models:
        _progress(
            f"token audit loading {model.model_id} tokenizer on CPU"
        )
        tokenizer = load_tokenizer(
            model.model_name,
            model.model_revision,
        )
        for item in selected:
            lesson_id = item.lesson.lesson_id
            for arm_value in CALIBRATION_ARMS:
                arm = Arm(arm_value)
                for probe in hard_bank[lesson_id]:
                    rendered, variant, renderer = (
                        render_calibration_probe(
                            probe,
                            arm=arm,
                            external_note=item.external_note,
                        )
                    )
                    audit = audit_prompt(
                        tokenizer,
                        rendered,
                        prompt_variant=variant,
                        prompt_rendering_version=renderer,
                        evaluation_max_length=(
                            config.evaluation_max_length
                        ),
                    )
                    record = {
                        "model_id": model.model_id,
                        "model_name": model.model_name,
                        "model_revision": model.model_revision,
                        **audit.to_dict(),
                    }
                    key = _prompt_audit_key(
                        model.model_id,
                        lesson_id,
                        probe.probe_id,
                        variant,
                    )
                    if key in audits:
                        raise ValueError(
                            f"duplicate calibration audit key: {key}"
                        )
                    audits[key] = record
                    records.append(record)
    truncated = [
        record for record in records if record["input_truncated"]
    ]
    missing = [
        record
        for record in records
        if not record["output_instruction_preserved"]
    ]
    report = {
        "schema_version": "p0d2hc-prompt-token-audit-v1",
        "evaluation_max_length": config.evaluation_max_length,
        "model_count": len(config.models),
        "prompt_count": len(records),
        "input_truncated_count": len(truncated),
        "output_instruction_missing_count": len(missing),
        "all_prompts_fit": not truncated and not missing,
        "max_untruncated_input_tokens_by_model": {
            model.model_id: max(
                int(record["untruncated_input_tokens"])
                for record in records
                if record["model_id"] == model.model_id
            )
            for model in config.models
        },
        "records": records,
    }
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    path = config.output_dir / "preflight" / "prompt_token_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(
                "calibration prompt-token audit changed; use a new attempt"
            )
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    audit_sha256 = sha256(payload).hexdigest()
    _progress(
        f"token audit complete: prompts={len(records)} "
        f"truncated={len(truncated)} missing_instruction={len(missing)}"
    )
    return path, audit_sha256, audits, report


def _require_prompt_audit_ready(report: dict[str, Any]) -> None:
    if report.get("all_prompts_fit") is not True:
        raise RuntimeError(
            "calibration prompt-token audit failed before inference: "
            f"truncated={report.get('input_truncated_count')} "
            f"missing_instruction="
            f"{report.get('output_instruction_missing_count')}"
        )


def run_experiment(request: P0D2HCRequest) -> Path:
    _require_cuda_runtime()
    config, hard_bank, selected = resolve_request(request)
    _, audit_hash, prompt_audits, audit_report = (
        _prepare_prompt_token_audit(config, hard_bank, selected)
    )
    _require_prompt_audit_ready(audit_report)
    run_id = f"p0d2hc-calibration-{_json_hash(config.identity_dict())[:10]}"
    manifest = P0D2HCManifest.load_or_create(
        config.output_dir / "manifest.json",
        run_id=run_id,
        config=config.identity_dict(),
        source_manifest_path=str(request.source_manifest),
        prompt_token_audit_sha256=audit_hash,
    )
    _write_environment(config.output_dir)
    selected_by_id = {
        item.lesson.lesson_id: item for item in selected
    }
    for model in config.models:
        _run_model(
            config,
            model,
            run_id,
            manifest,
            selected_by_id,
            hard_bank,
            prompt_audits,
        )
    _progress(f"completed manifest={manifest.path}")
    return manifest.path


def _run_model(
    config: ResolvedP0D2HCConfig,
    model: CalibrationModel,
    run_id: str,
    manifest: P0D2HCManifest,
    selected_by_id: dict[str, CompiledLesson],
    hard_bank: dict[str, tuple[P0CProbe, ...]],
    prompt_audits: PromptAuditIndex,
) -> None:
    failed = [
        lesson_id
        for lesson_id in config.selected_lesson_ids
        if manifest.unit_state(model.model_id, lesson_id) == "failed"
    ]
    if failed:
        raise RuntimeError(
            f"calibration failures are immutable for {model.model_id}: {failed}"
        )
    for lesson_id in config.selected_lesson_ids:
        path = _result_path(
            config.output_dir,
            model.model_id,
            lesson_id,
        )
        state = manifest.unit_state(model.model_id, lesson_id)
        if state != "verified" and path.exists():
            precision = _verify_rows(
                config,
                model,
                run_id,
                path,
                hard_bank[lesson_id],
                lesson_id,
                prompt_audits,
                expected_precision=None,
            )
            manifest.mark_unit(
                model.model_id,
                lesson_id,
                "verified",
                result_path=str(path),
                evaluation_precision=precision,
            )
    pending = [
        lesson_id
        for lesson_id in config.selected_lesson_ids
        if manifest.unit_state(model.model_id, lesson_id) != "verified"
    ]
    if not pending:
        for lesson_id in config.selected_lesson_ids:
            _verify_rows(
                config,
                model,
                run_id,
                _result_path(
                    config.output_dir,
                    model.model_id,
                    lesson_id,
                ),
                hard_bank[lesson_id],
                lesson_id,
                prompt_audits,
                expected_precision=str(
                    manifest.payload["units"][
                        unit_key(model.model_id, lesson_id)
                    ]["evaluation_precision"]
                ),
            )
        _progress(f"all units already verified: {model.model_id}")
        return

    _progress(
        f"loading base-only model on GPU: {model.model_id}/{model.model_name}"
    )
    evaluation_config = config.evaluation_config(model.model_id)
    bundle = load_base_model(evaluation_config)
    _require_cuda_bundle(bundle)
    try:
        for lesson_id in config.selected_lesson_ids:
            _run_unit(
                config,
                model,
                evaluation_config,
                bundle,
                run_id,
                manifest,
                selected_by_id[lesson_id],
                hard_bank[lesson_id],
                prompt_audits,
            )
    finally:
        release_model(bundle)


def _run_unit(
    config: ResolvedP0D2HCConfig,
    model: CalibrationModel,
    evaluation_config: ModelEvaluationConfig,
    bundle: ModelBundle,
    run_id: str,
    manifest: P0D2HCManifest,
    item: CompiledLesson,
    probes: tuple[P0CProbe, ...],
    prompt_audits: PromptAuditIndex,
) -> None:
    lesson_id = item.lesson.lesson_id
    path = _result_path(config.output_dir, model.model_id, lesson_id)
    state = manifest.unit_state(model.model_id, lesson_id)
    if state == "failed":
        raise RuntimeError(
            f"calibration unit is immutable after failure: "
            f"{model.model_id}/{lesson_id}"
        )
    if state == "verified" or path.exists():
        expected_precision = None
        if state == "verified":
            expected_precision = str(
                manifest.payload["units"][
                    unit_key(model.model_id, lesson_id)
                ]["evaluation_precision"]
            )
        precision = _verify_rows(
            config,
            model,
            run_id,
            path,
            probes,
            lesson_id,
            prompt_audits,
            expected_precision=expected_precision,
        )
        if state != "verified":
            manifest.mark_unit(
                model.model_id,
                lesson_id,
                "verified",
                result_path=str(path),
                evaluation_precision=precision,
            )
        return
    try:
        _progress(f"evaluating {model.model_id}/{lesson_id}")
        manifest.mark_unit(model.model_id, lesson_id, "evaluating")
        rows: list[dict[str, Any]] = []
        for arm_value in CALIBRATION_ARMS:
            arm = Arm(arm_value)
            rendered = tuple(
                render_calibration_probe(
                    probe,
                    arm=arm,
                    external_note=item.external_note,
                )[0]
                for probe in probes
            )
            results = evaluate_probes(
                bundle=bundle,
                probes=rendered,
                arm=arm,
                run_id=run_id,
                config=evaluation_config,
                training_seed=None,
            )
            rows.extend(
                _calibration_row(
                    result.to_dict(),
                    config,
                    model,
                    prompt_audits[
                        _prompt_audit_key(
                            model.model_id,
                            lesson_id,
                            result.probe_id,
                            prompt_variant_for_arm(arm),
                        )
                    ],
                )
                for result in results
            )
        write_probe_rows(path, rows)
        precision = _verify_rows(
            config,
            model,
            run_id,
            path,
            probes,
            lesson_id,
            prompt_audits,
            expected_precision=bundle.precision,
        )
        manifest.mark_unit(
            model.model_id,
            lesson_id,
            "verified",
            result_path=str(path),
            evaluation_precision=precision,
        )
    except (RuntimeError, ValueError, OSError) as error:
        manifest.mark_unit(model.model_id, lesson_id, "failed")
        manifest.record_error(
            unit_key(model.model_id, lesson_id),
            error,
        )
        raise


def _calibration_row(
    row: dict[str, Any],
    config: ResolvedP0D2HCConfig,
    model: CalibrationModel,
    audit: dict[str, Any],
) -> dict[str, Any]:
    if (
        row.get("prompt_sha256") != audit["prompt_sha256"]
        or int(row.get("input_tokens", -1))
        != int(audit["retained_input_tokens"])
    ):
        raise ValueError(
            f"calibration input differs from audit: {row.get('probe_id')}"
        )
    return {
        **row,
        "calibration_model_id": model.model_id,
        "calibration_model_role": model.role,
        "evaluation_max_length": config.evaluation_max_length,
        "prompt_variant": audit["prompt_variant"],
        "prompt_rendering_version": audit["prompt_rendering_version"],
        "untruncated_input_tokens": audit["untruncated_input_tokens"],
        "truncated_token_count": audit["truncated_token_count"],
        "input_truncated": audit["input_truncated"],
        "output_instruction_preserved": (
            audit["output_instruction_preserved"]
        ),
        "source_p0d2h_run_id": config.source_run_id,
        "source_manifest_sha256": config.source_manifest_sha256,
        "source_hard_probes_sha256": config.hard_probe_hashes[
            "hard_probes_sha256"
        ],
        "source_hard_probe_compiler_version": (
            HARD_PROBE_COMPILER_VERSION
        ),
    }


def _verify_rows(
    config: ResolvedP0D2HCConfig,
    model: CalibrationModel,
    run_id: str,
    path: Path,
    probes: tuple[P0CProbe, ...],
    lesson_id: str,
    prompt_audits: PromptAuditIndex,
    *,
    expected_precision: str | None,
) -> str:
    if not path.exists():
        raise FileNotFoundError(f"missing calibration result: {path}")
    rows = read_probe_results(path)
    probe_by_id = {probe.probe_id: probe for probe in probes}
    expected = {
        (arm, probe.probe_id)
        for arm in CALIBRATION_ARMS
        for probe in probes
    }
    observed = [
        (str(row.get("arm")), str(row.get("probe_id"))) for row in rows
    ]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(f"calibration result matrix mismatch: {path}")
    precisions = {str(row.get("precision")) for row in rows}
    if len(precisions) != 1 or (
        expected_precision is not None
        and precisions != {expected_precision}
    ):
        raise ValueError(f"calibration precision mismatch: {path}")
    required = {
        "predicted_action",
        "correct",
        "invalid",
        "generated_text",
        "input_tokens",
        "generated_tokens",
        "latency_seconds",
        "prompt_sha256",
        "calibration_model_id",
        "calibration_model_role",
        "evaluation_max_length",
        "prompt_variant",
        "prompt_rendering_version",
        "untruncated_input_tokens",
        "truncated_token_count",
        "input_truncated",
        "output_instruction_preserved",
    }
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(
                f"calibration row is missing {sorted(missing)}: {path}"
            )
        probe = probe_by_id[str(row["probe_id"])]
        arm = Arm(str(row["arm"]))
        variant = prompt_variant_for_arm(arm)
        audit = prompt_audits.get(
            _prompt_audit_key(
                model.model_id,
                lesson_id,
                probe.probe_id,
                variant,
            )
        )
        predicted = row.get("predicted_action")
        if (
            audit is None
            or row.get("run_id") != run_id
            or row.get("model") != model.model_name
            or row.get("model_revision") != model.model_revision
            or row.get("lesson_id") != lesson_id
            or row.get("pair_id") != probe.pair_id
            or row.get("lesson_type") != probe.lesson_type
            or row.get("category") != probe.category
            or row.get("expected_action") != probe.expected_action
            or row.get("training_seed") is not None
            or row.get("adapter_sha256") is not None
            or row.get("calibration_model_id") != model.model_id
            or row.get("calibration_model_role") != model.role
            or int(row["evaluation_max_length"])
            != config.evaluation_max_length
            or row.get("prompt_variant") != variant
            or row.get("prompt_sha256") != audit["prompt_sha256"]
            or int(row["input_tokens"])
            != int(audit["retained_input_tokens"])
            or int(row["untruncated_input_tokens"])
            != int(audit["untruncated_input_tokens"])
            or int(row["truncated_token_count"]) != 0
            or bool(row["input_truncated"])
            or not bool(row["output_instruction_preserved"])
            or row.get("source_p0d2h_run_id") != config.source_run_id
            or row.get("source_manifest_sha256")
            != config.source_manifest_sha256
            or row.get("source_hard_probes_sha256")
            != config.hard_probe_hashes["hard_probes_sha256"]
            or row.get("source_hard_probe_compiler_version")
            != HARD_PROBE_COMPILER_VERSION
            or parse_unique_action(str(row["generated_text"]), ACTIONS)
            != predicted
            or bool(row["correct"])
            != (predicted == probe.expected_action)
            or bool(row["invalid"]) != (predicted is None)
            or int(row["generated_tokens"]) < 0
            or float(row["latency_seconds"]) < 0.0
        ):
            raise ValueError(
                f"calibration provenance mismatch: "
                f"{path}/{row.get('probe_id')}/{row.get('arm')}"
            )
    return next(iter(precisions))


def _prompt_audit_key(
    model_id: str,
    lesson_id: str,
    probe_id: str,
    variant: str,
) -> tuple[str, str, str, str]:
    return model_id, lesson_id, probe_id, variant


def _result_path(
    output_dir: Path,
    model_id: str,
    lesson_id: str,
) -> Path:
    return output_dir / "results" / "raw" / model_id / f"{lesson_id}.jsonl"
