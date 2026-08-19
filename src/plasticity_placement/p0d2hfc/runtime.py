from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from plasticity_placement.p0c.domain import Arm, CompiledLesson, P0CProbe
from plasticity_placement.p0c.modeling import (
    ModelBundle,
    load_base_model,
    read_probe_results,
    release_model,
    write_probe_rows,
)
from plasticity_placement.p0c.runtime import (
    current_code_hash,
    current_environment_fingerprint,
    current_environment_snapshot,
)
from plasticity_placement.p0d.runtime import _write_environment
from plasticity_placement.p0d2h.probes import load_hard_probe_bank
from plasticity_placement.p0d2h.prompting import (
    EXTERNAL_PROMPT_RENDERER_VERSION,
)
from plasticity_placement.p0d2hc.analysis import _validate_and_load
from plasticity_placement.p0d2hc.config import CALIBRATION_ARMS
from plasticity_placement.p0d2hc.prompting import (
    ORACLE_PROMPT_RENDERER_VERSION,
    render_calibration_probe,
)
from plasticity_placement.p0d2hfc.config import (
    CANDIDATE_SCORING_VERSION,
    EVALUATION_MAX_LENGTH,
    ForcedChoiceModel,
    P0D2HFCRequest,
    ResolvedP0D2HFCConfig,
)
from plasticity_placement.p0d2hfc.manifest import (
    P0D2HFCManifest,
    unit_key,
)
from plasticity_placement.p0d2hfc.scoring import (
    CandidateTokenMismatch,
    audit_candidate_tokenization,
    rank_candidate_scores,
    score_candidate_batch,
)
from plasticity_placement.training.model_utils import (
    chat_prompt,
    load_tokenizer,
)

CandidateAuditIndex = dict[tuple[str, str, str, str], dict[str, Any]]
SourceRowIndex = dict[tuple[str, str, str, str], dict[str, Any]]


def _progress(message: str) -> None:
    print(f"[p0d2hfc] {message}", flush=True)


def _require_cuda_runtime() -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "P0-D2H-CAL-FC requires PyTorch with CUDA support"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            "formal forced-choice scoring requires CUDA; select a GPU runtime"
        )
    _progress(
        f"CUDA ready: {torch.cuda.get_device_name(0)} "
        f"(torch CUDA {torch.version.cuda})"
    )


def _require_cuda_bundle(bundle: ModelBundle) -> None:
    device = next(bundle.model.parameters()).device
    if getattr(device, "type", None) != "cuda":
        raise RuntimeError(f"forced-choice model loaded on {device}, expected CUDA")


def resolve_request(
    request: P0D2HFCRequest,
) -> tuple[
    ResolvedP0D2HFCConfig,
    dict[str, tuple[P0CProbe, ...]],
    tuple[CompiledLesson, ...],
    SourceRowIndex,
]:
    source_path = request.source_manifest
    if not source_path.exists():
        raise FileNotFoundError(
            f"missing P0-D2H-CAL source manifest: {source_path}"
        )
    source_dir = source_path.parent
    _require_independent_output(source_dir, request.output_dir)
    source_manifest_bytes = source_path.read_bytes()
    source_manifest = json.loads(source_manifest_bytes)
    source_config, validated_rows = _validate_and_load(
        source_dir,
        source_manifest,
    )
    if (
        source_manifest.get("schema_version") != "p0d2hc-manifest-v1"
        or len(source_config.models) != 2
        or source_config.expected_probe_row_count != 2_304
        or len(validated_rows) != 2_304
        or source_config.evaluation_max_length != EVALUATION_MAX_LENGTH
    ):
        raise ValueError(
            "P0-D2H-CAL-FC requires the complete frozen two-model source"
        )

    source_summary_path = source_dir / "results" / "aggregate" / "summary.json"
    source_summary_bytes = source_summary_path.read_bytes()
    source_prompt_audit_path = (
        source_dir / "preflight" / "prompt_token_audit.json"
    )
    source_prompt_audit_bytes = source_prompt_audit_path.read_bytes()
    source_raw_hash = _raw_tree_hash(source_dir)

    p0d2h_manifest_path = Path(str(source_manifest["source_manifest_path"]))
    p0d2h_manifest_bytes = p0d2h_manifest_path.read_bytes()
    if (
        sha256(p0d2h_manifest_bytes).hexdigest()
        != source_config.source_manifest_sha256
    ):
        raise ValueError("transitive P0-D2H-R source manifest changed")

    hard_bank = load_hard_probe_bank(source_config.source_output_dir)
    from plasticity_placement.p0c.compiler import load_compiled_bank

    compiled_by_id = {
        item.lesson.lesson_id: item
        for item in load_compiled_bank(source_config.source_p0d2_output_dir)
    }
    selected = tuple(
        compiled_by_id[lesson_id]
        for lesson_id in source_config.selected_lesson_ids
    )
    if (
        tuple(hard_bank) != source_config.selected_lesson_ids
        or len(selected) != 24
    ):
        raise ValueError("forced-choice source lesson/probe order changed")

    source_rows = _source_row_index(
        source_dir,
        source_config.models,
        source_config.selected_lesson_ids,
    )
    validated_index = {
        _source_row_key(row): row for row in validated_rows
    }
    if (
        len(validated_index) != 2_304
        or set(source_rows) != set(validated_index)
        or any(
            source_rows[key]["row"] != validated_index[key]
            for key in validated_index
        )
    ):
        raise ValueError("source raw-row identity differs after validation")

    models = tuple(
        ForcedChoiceModel(
            model_id=model.model_id,
            role=model.role,
            model_name=model.model_name,
            model_revision=model.model_revision,
            use_4bit=model.use_4bit,
        )
        for model in source_config.models
    )
    config = ResolvedP0D2HFCConfig(
        output_dir=request.output_dir,
        source_output_dir=source_dir,
        source_p0d2h_output_dir=source_config.source_output_dir,
        code_sha256=current_code_hash(),
        source_manifest_sha256=sha256(source_manifest_bytes).hexdigest(),
        source_run_id=str(source_manifest["run_id"]),
        source_summary_sha256=sha256(source_summary_bytes).hexdigest(),
        source_prompt_token_audit_sha256=sha256(
            source_prompt_audit_bytes
        ).hexdigest(),
        source_raw_results_sha256=source_raw_hash,
        source_p0d2h_run_id=source_config.source_run_id,
        source_p0d2h_manifest_sha256=source_config.source_manifest_sha256,
        hard_probe_hashes=dict(source_config.hard_probe_hashes),
        selected_lesson_ids=source_config.selected_lesson_ids,
        models=models,
        prompt_renderers={
            "no_write": "frozen_hard_probe",
            "external": EXTERNAL_PROMPT_RENDERER_VERSION,
            "answer_copy_oracle": ORACLE_PROMPT_RENDERER_VERSION,
        },
    )
    return config, hard_bank, selected, source_rows


def audit_request(request: P0D2HFCRequest) -> Path:
    config, hard_bank, selected, source_rows = resolve_request(request)
    path, _, _, report = _prepare_candidate_token_audit(
        config,
        hard_bank,
        selected,
        source_rows,
    )
    if report.get("all_checks_passed") is not True:
        raise RuntimeError(
            "candidate-token audit failed; formal scoring is forbidden"
        )
    return path


def _prepare_candidate_token_audit(
    config: ResolvedP0D2HFCConfig,
    hard_bank: dict[str, tuple[P0CProbe, ...]],
    selected: tuple[CompiledLesson, ...],
    source_rows: SourceRowIndex,
) -> tuple[Path, str, CandidateAuditIndex, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    audits: CandidateAuditIndex = {}
    chat_template_hashes: dict[str, str] = {}
    for model in config.models:
        _progress(f"CPU audit loading tokenizer: {model.model_id}")
        tokenizer = load_tokenizer(model.model_name, model.model_revision)
        chat_template = getattr(tokenizer, "chat_template", None)
        if not isinstance(chat_template, str) or not chat_template:
            raise ValueError(
                f"tokenizer has no frozen chat template: {model.model_id}"
            )
        chat_template_hashes[model.model_id] = sha256(
            chat_template.encode()
        ).hexdigest()
        for item in selected:
            lesson_id = item.lesson.lesson_id
            for arm_value in CALIBRATION_ARMS:
                arm = Arm(arm_value)
                for probe in hard_bank[lesson_id]:
                    key = _audit_key(
                        model.model_id,
                        lesson_id,
                        arm_value,
                        probe.probe_id,
                    )
                    source = source_rows[key]
                    source_row = source["row"]
                    rendered, variant, renderer = render_calibration_probe(
                        probe,
                        arm=arm,
                        external_note=item.external_note,
                    )
                    formatted = chat_prompt(tokenizer, rendered.prompt)
                    token_audit = audit_candidate_tokenization(
                        tokenizer,
                        formatted,
                        probe.action_choices,
                        evaluation_max_length=config.evaluation_max_length,
                    )
                    source_matches = (
                        source_row["prompt_sha256"]
                        == token_audit["prompt_sha256"]
                        and int(source_row["input_tokens"])
                        == int(
                            token_audit["untruncated_prompt_token_count"]
                        )
                        and source_row["expected_action"]
                        == probe.expected_action
                        and source_row["category"] == probe.category
                        and source_row["pair_id"] == probe.pair_id
                        and source_row["lesson_type"] == probe.lesson_type
                        and source_row["prompt_variant"] == variant
                        and source_row["prompt_rendering_version"] == renderer
                    )
                    record = {
                        "model_id": model.model_id,
                        "model_name": model.model_name,
                        "model_revision": model.model_revision,
                        "lesson_id": lesson_id,
                        "pair_id": probe.pair_id,
                        "lesson_type": probe.lesson_type,
                        "probe_id": probe.probe_id,
                        "category": probe.category,
                        "arm": arm_value,
                        "expected_action": probe.expected_action,
                        "prompt_variant": variant,
                        "prompt_rendering_version": renderer,
                        "source_row_key": _source_row_key_text(key),
                        "source_row_sha256": source["source_row_sha256"],
                        "source_raw_file_sha256": (
                            source["source_raw_file_sha256"]
                        ),
                        "source_result_path": source["source_result_path"],
                        "source_prompt_matches": source_matches,
                        **token_audit,
                    }
                    record["record_sha256"] = _json_hash(record)
                    if key in audits:
                        raise ValueError(f"duplicate candidate audit key: {key}")
                    audits[key] = record
                    records.append(record)
    failed = [
        record
        for record in records
        if not record["source_prompt_matches"]
        or not record["all_candidates_valid"]
    ]
    candidate_records = [
        candidate
        for record in records
        for candidate in record["candidates"]
    ]
    report = {
        "schema_version": "p0d2hfc-candidate-token-audit-v1",
        "candidate_scoring_version": CANDIDATE_SCORING_VERSION,
        "config_identity_sha256": _json_hash(config.identity_dict()),
        "source_manifest_sha256": config.source_manifest_sha256,
        "source_summary_sha256": config.source_summary_sha256,
        "source_prompt_token_audit_sha256": (
            config.source_prompt_token_audit_sha256
        ),
        "source_raw_results_sha256": config.source_raw_results_sha256,
        "evaluation_max_length": config.evaluation_max_length,
        "model_count": len(config.models),
        "decision_count": len(records),
        "candidate_count": len(records) * 4,
        "failed_decision_count": len(failed),
        "input_truncated_count": 0,
        "source_prompt_mismatch_count": sum(
            not bool(record["source_prompt_matches"])
            for record in records
        ),
        "prompt_prefix_mismatch_count": sum(
            not bool(candidate["prompt_prefix_verified"])
            for candidate in candidate_records
        ),
        "standalone_candidate_mismatch_count": sum(
            not bool(candidate["standalone_token_ids_match"])
            for candidate in candidate_records
        ),
        "decoded_candidate_mismatch_count": sum(
            not bool(candidate["decoded_continuation_matches"])
            for candidate in candidate_records
        ),
        "full_sequence_overflow_count": sum(
            not bool(candidate["full_sequence_fits"])
            for candidate in candidate_records
        ),
        "all_checks_passed": not failed and len(records) == 2_304,
        "chat_template_sha256_by_model": chat_template_hashes,
        "max_untruncated_prompt_tokens_by_model": {
            model.model_id: max(
                int(record["untruncated_prompt_token_count"])
                for record in records
                if record["model_id"] == model.model_id
            )
            for model in config.models
        },
        "max_full_input_tokens_by_model": {
            model.model_id: max(
                int(candidate["full_input_token_count"])
                for record in records
                if record["model_id"] == model.model_id
                for candidate in record["candidates"]
            )
            for model in config.models
        },
        "records": records,
    }
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    path = config.output_dir / "preflight" / "candidate_token_audit.json"
    _write_immutable(path, payload, "candidate-token audit")
    audit_hash = sha256(payload).hexdigest()
    _progress(
        f"CPU audit complete: decisions={len(records)} "
        f"candidates={len(records) * 4} failed={len(failed)}"
    )
    return path, audit_hash, audits, report


def _load_candidate_token_audit(
    config: ResolvedP0D2HFCConfig,
) -> tuple[str, CandidateAuditIndex, dict[str, Any]]:
    path = config.output_dir / "preflight" / "candidate_token_audit.json"
    payload = path.read_bytes()
    report = json.loads(payload)
    if (
        report.get("schema_version")
        != "p0d2hfc-candidate-token-audit-v1"
        or report.get("candidate_scoring_version")
        != CANDIDATE_SCORING_VERSION
        or report.get("config_identity_sha256")
        != _json_hash(config.identity_dict())
        or report.get("source_manifest_sha256")
        != config.source_manifest_sha256
        or report.get("source_raw_results_sha256")
        != config.source_raw_results_sha256
        or report.get("decision_count") != 2_304
        or report.get("candidate_count") != 9_216
        or report.get("failed_decision_count") != 0
        or report.get("input_truncated_count") != 0
        or report.get("all_checks_passed") is not True
    ):
        raise ValueError("candidate-token audit is incomplete or mismatched")
    audits = {
        _audit_key(
            str(record["model_id"]),
            str(record["lesson_id"]),
            str(record["arm"]),
            str(record["probe_id"]),
        ): record
        for record in report["records"]
    }
    records_valid = all(
        record.get("record_sha256")
        == _json_hash(
            {
                key: value
                for key, value in record.items()
                if key != "record_sha256"
            }
        )
        and record.get("source_prompt_matches") is True
        and record.get("all_candidates_valid") is True
        and len(record.get("candidates", [])) == 4
        for record in report["records"]
    )
    if len(audits) != 2_304 or not records_valid:
        raise ValueError("candidate-token audit contains duplicate row keys")
    return sha256(payload).hexdigest(), audits, report


def run_experiment(request: P0D2HFCRequest) -> Path:
    config, hard_bank, selected, source_rows = resolve_request(request)
    audit_hash, audits, _ = _load_candidate_token_audit(config)
    _require_cuda_runtime()
    run_id = "p0d2hfc-forced-choice-" + _json_hash(
        {
            "config": config.identity_dict(),
            "candidate_token_audit_sha256": audit_hash,
        }
    )[:10]
    manifest = P0D2HFCManifest.load_or_create(
        config.output_dir / "manifest.json",
        run_id=run_id,
        config=config.identity_dict(),
        source_manifest_path=str(request.source_manifest),
        candidate_token_audit_sha256=audit_hash,
    )
    _write_environment(config.output_dir)
    session_id, environment_fingerprint = _record_source_session(
        config,
        run_id,
    )
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
            source_rows,
            audits,
            session_id,
            environment_fingerprint,
        )
    _progress(f"completed manifest={manifest.path}")
    return manifest.path


def _run_model(
    config: ResolvedP0D2HFCConfig,
    model: ForcedChoiceModel,
    run_id: str,
    manifest: P0D2HFCManifest,
    selected_by_id: dict[str, CompiledLesson],
    hard_bank: dict[str, tuple[P0CProbe, ...]],
    source_rows: SourceRowIndex,
    audits: CandidateAuditIndex,
    session_id: str,
    environment_fingerprint: str,
) -> None:
    failed = [
        lesson_id
        for lesson_id in config.selected_lesson_ids
        if manifest.unit_state(model.model_id, lesson_id) == "failed"
    ]
    if failed:
        raise RuntimeError(
            f"forced-choice failures are immutable for {model.model_id}: {failed}"
        )
    for lesson_id in config.selected_lesson_ids:
        path = _result_path(config.output_dir, model.model_id, lesson_id)
        state = manifest.unit_state(model.model_id, lesson_id)
        if state != "verified" and path.exists():
            try:
                precision = _verify_rows(
                    config,
                    model,
                    run_id,
                    path,
                    hard_bank[lesson_id],
                    lesson_id,
                    source_rows,
                    audits,
                    expected_precision=None,
                )
                manifest.mark_unit(
                    model.model_id,
                    lesson_id,
                    "verified",
                    result_path=str(path),
                    evaluation_precision=precision,
                    result_sha256=_file_hash(path),
                )
            except (RuntimeError, ValueError, OSError) as error:
                manifest.mark_unit(model.model_id, lesson_id, "failed")
                manifest.record_error(
                    unit_key(model.model_id, lesson_id),
                    error,
                )
                raise
    pending = [
        lesson_id
        for lesson_id in config.selected_lesson_ids
        if manifest.unit_state(model.model_id, lesson_id) != "verified"
    ]
    if not pending:
        for lesson_id in config.selected_lesson_ids:
            path = _result_path(
                config.output_dir,
                model.model_id,
                lesson_id,
            )
            unit = manifest.payload["units"][
                unit_key(model.model_id, lesson_id)
            ]
            _verify_rows(
                config,
                model,
                run_id,
                path,
                hard_bank[lesson_id],
                lesson_id,
                source_rows,
                audits,
                expected_precision=str(unit["evaluation_precision"]),
            )
            if unit.get("result_sha256") != _file_hash(path):
                raise ValueError(
                    f"verified forced-choice raw file changed: {path}"
                )
        _progress(f"all units already verified and hash-checked: {model.model_id}")
        return

    _progress(f"loading one base model on CUDA: {model.model_id}/{model.model_name}")
    evaluation_config = config.evaluation_config(model.model_id)
    bundle = load_base_model(evaluation_config)
    _require_cuda_bundle(bundle)
    if bundle.model_revision != model.model_revision:
        release_model(bundle)
        raise RuntimeError(
            f"loaded revision differs from frozen source: "
            f"{bundle.model_revision} != {model.model_revision}"
        )
    try:
        for lesson_id in config.selected_lesson_ids:
            _run_unit(
                config,
                model,
                bundle,
                run_id,
                manifest,
                selected_by_id[lesson_id],
                hard_bank[lesson_id],
                source_rows,
                audits,
                session_id,
                environment_fingerprint,
            )
    finally:
        release_model(bundle)


def _run_unit(
    config: ResolvedP0D2HFCConfig,
    model: ForcedChoiceModel,
    bundle: ModelBundle,
    run_id: str,
    manifest: P0D2HFCManifest,
    item: CompiledLesson,
    probes: tuple[P0CProbe, ...],
    source_rows: SourceRowIndex,
    audits: CandidateAuditIndex,
    session_id: str,
    environment_fingerprint: str,
) -> None:
    lesson_id = item.lesson.lesson_id
    path = _result_path(config.output_dir, model.model_id, lesson_id)
    state = manifest.unit_state(model.model_id, lesson_id)
    if state == "failed":
        raise RuntimeError(
            f"forced-choice unit is immutable after failure: "
            f"{model.model_id}/{lesson_id}"
        )
    if state == "verified" or path.exists():
        if state == "verified":
            unit = manifest.payload["units"][
                unit_key(model.model_id, lesson_id)
            ]
            expected_precision = str(unit["evaluation_precision"])
            if unit.get("result_sha256") != _file_hash(path):
                raise ValueError(
                    f"verified forced-choice raw file changed: {path}"
                )
        else:
            expected_precision = None
        precision = _verify_rows(
            config,
            model,
            run_id,
            path,
            probes,
            lesson_id,
            source_rows,
            audits,
            expected_precision=expected_precision,
        )
        if state != "verified":
            manifest.mark_unit(
                model.model_id,
                lesson_id,
                "verified",
                result_path=str(path),
                evaluation_precision=precision,
                result_sha256=_file_hash(path),
            )
        return
    try:
        _progress(f"scoring {model.model_id}/{lesson_id}")
        manifest.mark_unit(model.model_id, lesson_id, "evaluating")
        rows: list[dict[str, Any]] = []
        for arm_value in CALIBRATION_ARMS:
            _progress(f"heartbeat {model.model_id}/{lesson_id}/{arm_value}")
            arm = Arm(arm_value)
            for probe in probes:
                key = _audit_key(
                    model.model_id,
                    lesson_id,
                    arm_value,
                    probe.probe_id,
                )
                audit = audits[key]
                rendered, _, _ = render_calibration_probe(
                    probe,
                    arm=arm,
                    external_note=item.external_note,
                )
                formatted = chat_prompt(bundle.tokenizer, rendered.prompt)
                try:
                    outcome = score_candidate_batch(
                        bundle,
                        formatted,
                        audit,
                    )
                except CandidateTokenMismatch as error:
                    outcome = _candidate_token_error(audit, error)
                rows.append(
                    _decision_row(
                        config,
                        model,
                        run_id,
                        bundle.precision,
                        probe,
                        arm_value,
                        audit,
                        source_rows[key],
                        outcome,
                        session_id,
                        environment_fingerprint,
                    )
                )
        write_probe_rows(path, rows)
        precision = _verify_rows(
            config,
            model,
            run_id,
            path,
            probes,
            lesson_id,
            source_rows,
            audits,
            expected_precision=bundle.precision,
        )
        manifest.mark_unit(
            model.model_id,
            lesson_id,
            "verified",
            result_path=str(path),
            evaluation_precision=precision,
            result_sha256=_file_hash(path),
        )
    except (RuntimeError, ValueError, OSError) as error:
        manifest.mark_unit(model.model_id, lesson_id, "failed")
        manifest.record_error(unit_key(model.model_id, lesson_id), error)
        raise


def _decision_row(
    config: ResolvedP0D2HFCConfig,
    model: ForcedChoiceModel,
    run_id: str,
    precision: str,
    probe: P0CProbe,
    arm: str,
    audit: dict[str, Any],
    source: dict[str, Any],
    outcome: dict[str, Any],
    session_id: str,
    environment_fingerprint: str,
) -> dict[str, Any]:
    source_row = source["row"]
    predicted = outcome["predicted_action"]
    return {
        "schema_version": "p0d2hfc-decision-row-v1",
        "run_id": run_id,
        "model_id": model.model_id,
        "model_role": model.role,
        "model_name": model.model_name,
        "model_revision": model.model_revision,
        "precision": precision,
        "lesson_id": probe.lesson_id,
        "pair_id": probe.pair_id,
        "lesson_type": probe.lesson_type,
        "probe_id": probe.probe_id,
        "category": probe.category,
        "arm": arm,
        "expected_action": probe.expected_action,
        "ordered_allowed_actions": list(probe.action_choices),
        "prompt_variant": audit["prompt_variant"],
        "prompt_rendering_version": audit["prompt_rendering_version"],
        "prompt_sha256": audit["prompt_sha256"],
        "untruncated_prompt_token_count": audit[
            "untruncated_prompt_token_count"
        ],
        "evaluation_max_length": config.evaluation_max_length,
        "candidate_scoring_version": CANDIDATE_SCORING_VERSION,
        "candidate_audit_record_sha256": audit["record_sha256"],
        "candidates": outcome["candidates"],
        "predicted_action": predicted,
        "mean_predicted_action": outcome["mean_predicted_action"],
        "top1_top2_margin": outcome["top1_top2_margin"],
        "sum_mean_disagreement": outcome["sum_mean_disagreement"],
        "correct": predicted == probe.expected_action,
        "tie": outcome["tie"],
        "mean_score_tie": outcome["mean_score_tie"],
        "non_finite": outcome["non_finite"],
        "error_status": outcome["error_status"],
        "error_message": outcome["error_message"],
        "latency_seconds": outcome["latency_seconds"],
        "source_run_id": config.source_run_id,
        "source_manifest_sha256": config.source_manifest_sha256,
        "source_summary_sha256": config.source_summary_sha256,
        "source_prompt_token_audit_sha256": (
            config.source_prompt_token_audit_sha256
        ),
        "source_raw_results_sha256": config.source_raw_results_sha256,
        "source_result_path": source["source_result_path"],
        "source_raw_file_sha256": source["source_raw_file_sha256"],
        "source_row_key": _source_row_key_text(
            _source_row_key(source_row)
        ),
        "source_row_sha256": source["source_row_sha256"],
        "source_strict_predicted_action": source_row["predicted_action"],
        "source_strict_correct": bool(source_row["correct"]),
        "source_strict_invalid": bool(source_row["invalid"]),
        "source_generated_text_sha256": sha256(
            str(source_row["generated_text"]).encode()
        ).hexdigest(),
        "source_p0d2h_run_id": config.source_p0d2h_run_id,
        "source_p0d2h_manifest_sha256": (
            config.source_p0d2h_manifest_sha256
        ),
        "source_hard_probes_sha256": config.hard_probe_hashes[
            "hard_probes_sha256"
        ],
        "training_seed": None,
        "adapter_sha256": None,
        "execution_session_id": session_id,
        "environment_fingerprint": environment_fingerprint,
    }


def _candidate_token_error(
    audit: dict[str, Any],
    error: CandidateTokenMismatch,
) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "action": candidate["action"],
                "continuation": candidate["continuation"],
                "token_ids": candidate["token_ids"],
                "token_count": candidate["token_count"],
                "sum_logprob": None,
                "mean_logprob": None,
                "token_logprobs": [],
                "score_finite": None,
                "sum_rank": None,
                "mean_rank": None,
            }
            for candidate in audit["candidates"]
        ],
        "predicted_action": None,
        "mean_predicted_action": None,
        "top1_top2_margin": None,
        "sum_mean_disagreement": None,
        "tie": False,
        "mean_score_tie": False,
        "non_finite": False,
        "error_status": "candidate_token_mismatch",
        "error_message": str(error),
        "latency_seconds": 0.0,
    }


def _verify_rows(
    config: ResolvedP0D2HFCConfig,
    model: ForcedChoiceModel,
    run_id: str,
    path: Path,
    probes: tuple[P0CProbe, ...],
    lesson_id: str,
    source_rows: SourceRowIndex,
    audits: CandidateAuditIndex,
    *,
    expected_precision: str | None,
) -> str:
    if not path.exists():
        raise FileNotFoundError(f"missing forced-choice result: {path}")
    rows = read_probe_results(path)
    expected = {
        (arm, probe.probe_id)
        for arm in CALIBRATION_ARMS
        for probe in probes
    }
    observed = [
        (str(row.get("arm")), str(row.get("probe_id"))) for row in rows
    ]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(f"forced-choice result matrix mismatch: {path}")
    precisions = {str(row.get("precision")) for row in rows}
    if len(precisions) != 1 or (
        expected_precision is not None
        and precisions != {expected_precision}
    ):
        raise ValueError(f"forced-choice precision mismatch: {path}")
    probe_by_id = {probe.probe_id: probe for probe in probes}
    sessions = _session_index(config.output_dir)
    for row in rows:
        probe = probe_by_id[str(row["probe_id"])]
        key = _audit_key(
            model.model_id,
            lesson_id,
            str(row["arm"]),
            probe.probe_id,
        )
        audit = audits.get(key)
        source = source_rows.get(key)
        if audit is None or source is None:
            raise ValueError(f"forced-choice audit/source row is missing: {key}")
        source_row = source["row"]
        session = sessions.get(str(row.get("execution_session_id")))
        basic_valid = (
            row.get("schema_version") == "p0d2hfc-decision-row-v1"
            and row.get("run_id") == run_id
            and row.get("model_id") == model.model_id
            and row.get("model_role") == model.role
            and row.get("model_name") == model.model_name
            and row.get("model_revision") == model.model_revision
            and row.get("lesson_id") == lesson_id
            and row.get("pair_id") == probe.pair_id
            and row.get("lesson_type") == probe.lesson_type
            and row.get("category") == probe.category
            and row.get("expected_action") == probe.expected_action
            and row.get("ordered_allowed_actions") == list(probe.action_choices)
            and row.get("prompt_sha256") == audit["prompt_sha256"]
            and row.get("prompt_variant") == audit["prompt_variant"]
            and row.get("prompt_rendering_version")
            == audit["prompt_rendering_version"]
            and row.get("candidate_audit_record_sha256")
            == audit["record_sha256"]
            and row.get("candidate_scoring_version")
            == CANDIDATE_SCORING_VERSION
            and int(row.get("untruncated_prompt_token_count", -1))
            == int(audit["untruncated_prompt_token_count"])
            and int(row.get("evaluation_max_length", -1))
            == config.evaluation_max_length
            and row.get("source_run_id") == config.source_run_id
            and row.get("source_manifest_sha256")
            == config.source_manifest_sha256
            and row.get("source_summary_sha256")
            == config.source_summary_sha256
            and row.get("source_prompt_token_audit_sha256")
            == config.source_prompt_token_audit_sha256
            and row.get("source_raw_results_sha256")
            == config.source_raw_results_sha256
            and row.get("source_result_path") == source["source_result_path"]
            and row.get("source_raw_file_sha256")
            == source["source_raw_file_sha256"]
            and row.get("source_row_sha256") == source["source_row_sha256"]
            and row.get("source_row_key") == _source_row_key_text(key)
            and bool(row.get("source_strict_correct"))
            == bool(source_row["correct"])
            and bool(row.get("source_strict_invalid"))
            == bool(source_row["invalid"])
            and row.get("source_strict_predicted_action")
            == source_row["predicted_action"]
            and row.get("source_generated_text_sha256")
            == sha256(str(source_row["generated_text"]).encode()).hexdigest()
            and row.get("source_p0d2h_run_id")
            == config.source_p0d2h_run_id
            and row.get("source_p0d2h_manifest_sha256")
            == config.source_p0d2h_manifest_sha256
            and row.get("source_hard_probes_sha256")
            == config.hard_probe_hashes["hard_probes_sha256"]
            and row.get("training_seed") is None
            and row.get("adapter_sha256") is None
            and session is not None
            and row.get("environment_fingerprint")
            == session.get("environment_fingerprint")
            and float(row.get("latency_seconds", -1.0)) >= 0.0
        )
        if not basic_valid:
            raise ValueError(
                f"forced-choice provenance mismatch: {path}/{key}"
            )
        _verify_candidate_outcome(row, audit, probe)
    return next(iter(precisions))


def _verify_candidate_outcome(
    row: dict[str, Any],
    audit: dict[str, Any],
    probe: P0CProbe,
) -> None:
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 4:
        raise ValueError("forced-choice row requires four candidates")
    for observed, expected in zip(
        candidates,
        audit["candidates"],
        strict=True,
    ):
        if (
            observed.get("action") != expected["action"]
            or observed.get("continuation") != expected["continuation"]
            or observed.get("token_ids") != expected["token_ids"]
            or int(observed.get("token_count", -1))
            != int(expected["token_count"])
        ):
            raise ValueError("forced-choice candidate token provenance mismatch")
    status = str(row.get("error_status"))
    if status == "candidate_token_mismatch":
        if (
            row.get("predicted_action") is not None
            or bool(row.get("correct"))
            or any(
                candidate.get("sum_logprob") is not None
                for candidate in candidates
            )
        ):
            raise ValueError("malformed explicit candidate-token error row")
        return
    if status not in {"ok", "tie", "non_finite"}:
        raise ValueError(f"unsupported forced-choice error status: {status}")
    ranked = rank_candidate_scores(candidates)
    fields = (
        "predicted_action",
        "mean_predicted_action",
        "top1_top2_margin",
        "sum_mean_disagreement",
        "tie",
        "mean_score_tie",
        "non_finite",
        "error_status",
    )
    if any(row.get(field) != ranked[field] for field in fields):
        raise ValueError("stored forced-choice ranking differs from scores")
    if bool(row.get("correct")) != (
        row.get("predicted_action") == probe.expected_action
    ):
        raise ValueError("stored forced-choice correctness differs from ranking")
    if status == "non_finite":
        if not any(
            candidate.get("score_finite") is False
            and candidate.get("sum_logprob") is None
            and candidate.get("mean_logprob") is None
            for candidate in candidates
        ):
            raise ValueError("malformed explicit non-finite score row")
        return
    for observed, expected in zip(
        candidates,
        ranked["candidates"],
        strict=True,
    ):
        if (
            observed.get("sum_rank") != expected["sum_rank"]
            or observed.get("mean_rank") != expected["mean_rank"]
        ):
            raise ValueError("stored forced-choice ranks differ from scores")
        token_logprobs = observed.get("token_logprobs")
        if (
            not isinstance(token_logprobs, list)
            or len(token_logprobs) != int(observed["token_count"])
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in token_logprobs
            )
            or not math.isclose(
                float(observed["sum_logprob"]),
                sum(float(value) for value in token_logprobs),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            or not math.isclose(
                float(observed["mean_logprob"]),
                float(observed["sum_logprob"])
                / int(observed["token_count"]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        ):
            raise ValueError("candidate token log-probabilities are inconsistent")


def _source_row_index(
    source_dir: Path,
    models: tuple[Any, ...],
    lesson_ids: tuple[str, ...],
) -> SourceRowIndex:
    result: SourceRowIndex = {}
    for model in models:
        for lesson_id in lesson_ids:
            path = (
                source_dir
                / "results"
                / "raw"
                / model.model_id
                / f"{lesson_id}.jsonl"
            )
            payload = path.read_bytes()
            file_hash = sha256(payload).hexdigest()
            rows = [
                json.loads(line)
                for line in payload.decode().splitlines()
                if line.strip()
            ]
            for row in rows:
                key = _source_row_key(row)
                if key in result:
                    raise ValueError(f"duplicate source row key: {key}")
                result[key] = {
                    "row": row,
                    "source_result_path": str(path),
                    "source_raw_file_sha256": file_hash,
                    "source_row_sha256": _json_hash(row),
                }
    if len(result) != 2_304:
        raise ValueError(f"source row count mismatch: {len(result)} != 2304")
    return result


def _source_row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["calibration_model_id"]),
        str(row["lesson_id"]),
        str(row["arm"]),
        str(row["probe_id"]),
    )


def _audit_key(
    model_id: str,
    lesson_id: str,
    arm: str,
    probe_id: str,
) -> tuple[str, str, str, str]:
    return model_id, lesson_id, arm, probe_id


def _source_row_key_text(key: tuple[str, str, str, str]) -> str:
    return "::".join(key)


def _result_path(
    output_dir: Path,
    model_id: str,
    lesson_id: str,
) -> Path:
    return output_dir / "results" / "raw" / model_id / f"{lesson_id}.jsonl"


def _record_source_session(
    config: ResolvedP0D2HFCConfig,
    run_id: str,
) -> tuple[str, str]:
    session_id = f"p0d2hfc-session-{uuid4().hex[:12]}"
    fingerprint = current_environment_fingerprint()
    record = {
        "session_id": session_id,
        "recorded_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "environment_fingerprint": fingerprint,
        "environment": current_environment_snapshot(),
        "source_run_id": config.source_run_id,
        "source_manifest_sha256": config.source_manifest_sha256,
        "source_raw_results_sha256": config.source_raw_results_sha256,
    }
    path = config.output_dir / "source_sessions.json"
    sessions = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if not isinstance(sessions, list):
        raise ValueError(f"invalid source session history: {path}")
    sessions.append(record)
    _atomic_json_write(path, sessions)
    return session_id, fingerprint


def _session_index(output_dir: Path) -> dict[str, dict[str, Any]]:
    path = output_dir / "source_sessions.json"
    if not path.exists():
        raise FileNotFoundError(f"missing forced-choice source sessions: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"invalid source sessions: {path}")
    indexed = {
        str(record["session_id"]): record
        for record in records
        if isinstance(record, dict) and record.get("session_id")
    }
    if not indexed:
        raise ValueError("source sessions contain no execution session")
    return indexed


def _raw_tree_hash(source_output_dir: Path) -> str:
    root = source_output_dir / "results" / "raw"
    files = sorted(root.rglob("*.jsonl"))
    if len(files) != 48:
        raise ValueError(f"expected 48 P0-D2H-CAL raw files, found {len(files)}")
    digest = sha256()
    for path in files:
        digest.update(str(path.relative_to(source_output_dir)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _require_independent_output(source_dir: Path, output_dir: Path) -> None:
    source = source_dir.resolve()
    output = output_dir.resolve()
    if (
        source == output
        or output.is_relative_to(source)
        or source.is_relative_to(output)
    ):
        raise ValueError(
            "forced-choice output must be independent from P0-D2H-CAL"
        )


def _json_hash(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _write_immutable(path: Path, payload: bytes, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing {label} changed; use a new attempt")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
