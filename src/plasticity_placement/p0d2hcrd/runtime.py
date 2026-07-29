from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from plasticity_placement.p0c.compiler import (
    load_compiled_bank,
)
from plasticity_placement.p0c.domain import CompiledLesson
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
    prepare_experiment,
)
from plasticity_placement.p0d.analysis import _atomic_json_write
from plasticity_placement.p0d.runtime import _write_environment
from plasticity_placement.p0d2h.probes import (
    load_hard_probe_bank,
    verify_hard_probe_hashes,
)
from plasticity_placement.p0d2hcrd.config import (
    CANDIDATE_SCORING_VERSION,
    ENDPOINTS,
    EXPECTED_MODEL_NAME,
    EXPECTED_MODEL_REVISION,
    EXPECTED_SOURCE_CANDIDATE_AUDIT_SHA256,
    EXPECTED_SOURCE_MANIFEST_SHA256,
    EXPECTED_SOURCE_RAW_TREE_SHA256,
    EXPECTED_SOURCE_RUN_ID,
    EXPECTED_SOURCE_SUMMARY_SHA256,
    CRDModel,
    P0D2HCRDRequest,
    ResolvedP0D2HCRDConfig,
)
from plasticity_placement.p0d2hcrd.manifest import (
    P0D2HCRDManifest,
    unit_key,
)
from plasticity_placement.p0d2hcrd.probes import (
    PROMPT_RENDERERS,
    DecompositionProbe,
    compile_decomposition_bank,
)
from plasticity_placement.p0d2hcrd.scoring import (
    CandidateTokenMismatch,
    audit_candidate_tokenization,
    rank_candidate_scores,
    score_candidate_batch,
)
from plasticity_placement.p0d2hfc.scoring import (
    rank_candidate_scores as rank_source_candidate_scores,
)
from plasticity_placement.training.model_utils import (
    chat_prompt,
    load_tokenizer,
)

CandidateAuditIndex = dict[str, dict[str, Any]]
SourceRowIndex = dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class FrozenSource:
    forced_choice_dir: Path
    calibration_dir: Path
    hard_probe_dir: Path
    compiled_dir: Path
    forced_choice_manifest: dict[str, Any]
    hard_probe_hashes: dict[str, str]
    compiler_hashes: dict[str, str]
    selected: tuple[CompiledLesson, ...]
    source_rows: SourceRowIndex


def _progress(message: str) -> None:
    print(f"[p0d2hcrd] {message}", flush=True)


def _require_cuda_runtime() -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "P0-D2H-CRD requires PyTorch with CUDA support"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            "formal route decomposition requires CUDA; select a GPU runtime"
        )
    _progress(
        f"CUDA ready: {torch.cuda.get_device_name(0)} "
        f"(torch CUDA {torch.version.cuda})"
    )


def _require_cuda_bundle(bundle: ModelBundle) -> None:
    device = next(bundle.model.parameters()).device
    if getattr(device, "type", None) != "cuda":
        raise RuntimeError(
            f"route-decomposition model loaded on {device}, expected CUDA"
        )


def resolve_request(
    request: P0D2HCRDRequest,
) -> tuple[
    ResolvedP0D2HCRDConfig,
    dict[str, tuple[DecompositionProbe, ...]],
    tuple[CompiledLesson, ...],
    SourceRowIndex,
    dict[str, Any],
]:
    source = validate_frozen_source(request.source_manifest)
    _require_independent_output(source.forced_choice_dir, request.output_dir)
    bank, bank_audit = compile_decomposition_bank(
        source.selected,
        load_hard_probe_bank(source.hard_probe_dir),
    )
    if bank_audit.get("all_checks_passed") is not True:
        raise ValueError("route-decomposition bank audit failed")
    model = CRDModel(
        model_id="scale_canary",
        role="scale_canary",
        model_name=EXPECTED_MODEL_NAME,
        model_revision=EXPECTED_MODEL_REVISION,
        use_4bit=True,
    )
    forced_config = source.forced_choice_manifest["config"]
    config = ResolvedP0D2HCRDConfig(
        output_dir=request.output_dir,
        source_output_dir=source.forced_choice_dir,
        source_calibration_output_dir=source.calibration_dir,
        source_hard_probe_output_dir=source.hard_probe_dir,
        source_compiled_output_dir=source.compiled_dir,
        code_sha256=current_code_hash(),
        source_manifest_sha256=EXPECTED_SOURCE_MANIFEST_SHA256,
        source_run_id=EXPECTED_SOURCE_RUN_ID,
        source_summary_sha256=EXPECTED_SOURCE_SUMMARY_SHA256,
        source_candidate_token_audit_sha256=(
            EXPECTED_SOURCE_CANDIDATE_AUDIT_SHA256
        ),
        source_raw_results_sha256=EXPECTED_SOURCE_RAW_TREE_SHA256,
        source_calibration_manifest_sha256=str(
            forced_config["source_manifest_sha256"]
        ),
        source_calibration_raw_results_sha256=str(
            forced_config["source_raw_results_sha256"]
        ),
        source_hard_probe_manifest_sha256=str(
            forced_config["source_p0d2h_manifest_sha256"]
        ),
        hard_probe_hashes=source.hard_probe_hashes,
        compiler_hashes=source.compiler_hashes,
        selected_lesson_ids=tuple(forced_config["selected_lesson_ids"]),
        decomposition_bank_sha256=str(bank_audit["bank_sha256"]),
        model=model,
        prompt_renderers=dict(PROMPT_RENDERERS),
    )
    return config, bank, source.selected, source.source_rows, bank_audit


def validate_frozen_source(source_manifest_path: Path) -> FrozenSource:
    if not source_manifest_path.exists():
        raise FileNotFoundError(
            f"missing frozen P0-D2H-CAL-FC manifest: {source_manifest_path}"
        )
    forced_choice_dir = source_manifest_path.parent
    manifest_bytes = source_manifest_path.read_bytes()
    _require_hash(
        manifest_bytes,
        EXPECTED_SOURCE_MANIFEST_SHA256,
        "P0-D2H-CAL-FC manifest",
    )
    manifest = json.loads(manifest_bytes)
    forced_config = manifest.get("config", {})
    if (
        manifest.get("schema_version") != "p0d2hfc-manifest-v1"
        or manifest.get("run_id") != EXPECTED_SOURCE_RUN_ID
        or manifest.get("errors")
        or forced_config.get("schema_version") != "p0d2hfc-config-v1"
        or forced_config.get("stage") != "hard_probe_forced_choice"
        or manifest.get("candidate_token_audit_sha256")
        != EXPECTED_SOURCE_CANDIDATE_AUDIT_SHA256
        or forced_config.get("automatic_training_started") is not False
        or forced_config.get("automatic_narrow_scan_started") is not False
    ):
        raise ValueError("frozen P0-D2H-CAL-FC manifest identity changed")
    selected_ids = tuple(str(value) for value in manifest["selected_lessons"])
    if (
        len(selected_ids) != 24
        or tuple(forced_config.get("selected_lesson_ids", ())) != selected_ids
    ):
        raise ValueError("frozen P0-D2H-CAL-FC lesson cohort changed")
    model = manifest.get("models", {}).get("scale_canary", {})
    if (
        model.get("model_name") != EXPECTED_MODEL_NAME
        or model.get("model_revision") != EXPECTED_MODEL_REVISION
        or model.get("role") != "scale_canary"
        or model.get("use_4bit") is not True
    ):
        raise ValueError("frozen scale-canary identity changed")

    summary_path = forced_choice_dir / "results" / "aggregate" / "summary.json"
    summary_bytes = summary_path.read_bytes()
    _require_hash(
        summary_bytes,
        EXPECTED_SOURCE_SUMMARY_SHA256,
        "P0-D2H-CAL-FC summary",
    )
    summary = json.loads(summary_bytes)
    if (
        summary.get("run_id") != EXPECTED_SOURCE_RUN_ID
        or summary.get("decision_row_count") != 2_304
        or summary.get("gates", {}).get("run_valid") is not True
        or summary.get("gates", {}).get("eligible_model_ids") != []
        or summary.get("gates", {}).get("automatic_training_started") is not False
        or summary.get("gates", {}).get("automatic_narrow_scan_started")
        is not False
    ):
        raise ValueError("frozen P0-D2H-CAL-FC summary changed")
    observed_raw_hash = _raw_tree_hash(forced_choice_dir, 48)
    if observed_raw_hash != EXPECTED_SOURCE_RAW_TREE_SHA256:
        raise ValueError("frozen P0-D2H-CAL-FC raw tree changed")

    audit_path = forced_choice_dir / "preflight" / "candidate_token_audit.json"
    audit_bytes = audit_path.read_bytes()
    _require_hash(
        audit_bytes,
        EXPECTED_SOURCE_CANDIDATE_AUDIT_SHA256,
        "P0-D2H-CAL-FC candidate-token audit",
    )
    source_rows = _validate_forced_choice_rows(
        forced_choice_dir,
        manifest,
        json.loads(audit_bytes),
        selected_ids,
    )

    calibration_path = Path(str(manifest["source_manifest_path"]))
    calibration_bytes = calibration_path.read_bytes()
    _require_hash(
        calibration_bytes,
        str(forced_config["source_manifest_sha256"]),
        "P0-D2H-CAL manifest",
    )
    calibration = json.loads(calibration_bytes)
    calibration_dir = calibration_path.parent
    _validate_calibration_source(calibration_dir, calibration, forced_config)

    hard_probe_path = Path(str(calibration["source_manifest_path"]))
    hard_probe_bytes = hard_probe_path.read_bytes()
    _require_hash(
        hard_probe_bytes,
        str(forced_config["source_p0d2h_manifest_sha256"]),
        "P0-D2H-R manifest",
    )
    hard_probe = json.loads(hard_probe_bytes)
    hard_probe_dir = hard_probe_path.parent
    hard_config = hard_probe.get("config", {})
    if (
        hard_probe.get("schema_version") != "p0d2h-manifest-v1"
        or hard_probe.get("errors")
        or hard_config.get("selected_lesson_ids") != list(selected_ids)
    ):
        raise ValueError("P0-D2H-R source manifest is incomplete")
    hard_probe_hashes = dict(forced_config["hard_probe_hashes"])
    if hard_config.get("hard_probe_hashes") != hard_probe_hashes:
        raise ValueError("P0-D2H-R hard-probe identities changed")
    verify_hard_probe_hashes(hard_probe_dir, hard_probe_hashes)

    compiled_path = Path(str(hard_probe["source_manifest_path"]))
    compiled_bytes = compiled_path.read_bytes()
    _require_hash(
        compiled_bytes,
        str(hard_config["source_manifest_sha256"]),
        "P0-D2 manifest",
    )
    compiled_dir = compiled_path.parent
    if not (compiled_dir / "compiled" / "hashes.json").exists():
        raise FileNotFoundError(
            "P0-D2 compiled hashes are missing; refusing to write in source"
        )
    compiler_hashes = dict(hard_config["source_compiler_hashes"])
    if prepare_experiment(compiled_dir) != compiler_hashes:
        raise ValueError("P0-D2 compiled bank changed")
    hard_bank = load_hard_probe_bank(hard_probe_dir)
    compiled_by_id = {
        item.lesson.lesson_id: item
        for item in load_compiled_bank(compiled_dir)
    }
    try:
        selected = tuple(compiled_by_id[lesson_id] for lesson_id in selected_ids)
    except KeyError as error:
        raise ValueError("frozen lesson is missing from compiled bank") from error
    if tuple(hard_bank) != selected_ids:
        raise ValueError("frozen hard-probe bank order changed")
    return FrozenSource(
        forced_choice_dir=forced_choice_dir,
        calibration_dir=calibration_dir,
        hard_probe_dir=hard_probe_dir,
        compiled_dir=compiled_dir,
        forced_choice_manifest=manifest,
        hard_probe_hashes=hard_probe_hashes,
        compiler_hashes=compiler_hashes,
        selected=selected,
        source_rows=source_rows,
    )


def _validate_calibration_source(
    source_dir: Path,
    manifest: dict[str, Any],
    forced_config: dict[str, Any],
) -> None:
    if (
        manifest.get("schema_version") != "p0d2hc-manifest-v1"
        or manifest.get("errors")
        or manifest.get("run_id") != forced_config["source_run_id"]
    ):
        raise ValueError("P0-D2H-CAL source manifest is incomplete")
    _require_file_hash(
        source_dir / "results" / "aggregate" / "summary.json",
        str(forced_config["source_summary_sha256"]),
        "P0-D2H-CAL summary",
    )
    _require_file_hash(
        source_dir / "preflight" / "prompt_token_audit.json",
        str(forced_config["source_prompt_token_audit_sha256"]),
        "P0-D2H-CAL prompt-token audit",
    )
    if _raw_tree_hash(source_dir, 48) != forced_config["source_raw_results_sha256"]:
        raise ValueError("P0-D2H-CAL raw tree changed")
    units = manifest.get("units", {})
    expected = {
        f"{model_id}::{lesson_id}"
        for model_id in ("source_model", "scale_canary")
        for lesson_id in forced_config["selected_lesson_ids"]
    }
    if set(units) != expected:
        raise ValueError("P0-D2H-CAL unit matrix changed")
    for key, unit in units.items():
        if unit.get("state") != "verified":
            raise ValueError(f"P0-D2H-CAL source unit is not verified: {key}")
        model_id, lesson_id = key.split("::", maxsplit=1)
        path = source_dir / "results" / "raw" / model_id / f"{lesson_id}.jsonl"
        _require_file_hash(
            path,
            str(unit["result_sha256"]),
            f"P0-D2H-CAL raw unit {key}",
        )


def _validate_forced_choice_rows(
    source_dir: Path,
    manifest: dict[str, Any],
    audit: dict[str, Any],
    lesson_ids: tuple[str, ...],
) -> SourceRowIndex:
    if (
        audit.get("schema_version") != "p0d2hfc-candidate-token-audit-v1"
        or audit.get("decision_count") != 2_304
        or audit.get("candidate_count") != 9_216
        or audit.get("all_checks_passed") is not True
    ):
        raise ValueError("frozen source candidate-token audit is incomplete")
    audit_index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in audit.get("records", []):
        key = (
            str(record["model_id"]),
            str(record["lesson_id"]),
            str(record["arm"]),
            str(record["probe_id"]),
        )
        record_without_hash = {
            name: value
            for name, value in record.items()
            if name != "record_sha256"
        }
        if (
            key in audit_index
            or record.get("record_sha256") != _json_hash(record_without_hash)
            or record.get("source_prompt_matches") is not True
            or record.get("all_candidates_valid") is not True
        ):
            raise ValueError("frozen source candidate-token record changed")
        audit_index[key] = record
    if len(audit_index) != 2_304:
        raise ValueError("frozen source candidate-token audit keys changed")

    units = manifest.get("units", {})
    expected_units = {
        f"{model_id}::{lesson_id}"
        for model_id in ("source_model", "scale_canary")
        for lesson_id in lesson_ids
    }
    if set(units) != expected_units:
        raise ValueError("frozen P0-D2H-CAL-FC unit matrix changed")
    all_keys: set[tuple[str, str, str, str]] = set()
    selected_rows: SourceRowIndex = {}
    for key in sorted(expected_units):
        model_id, lesson_id = key.split("::", maxsplit=1)
        unit = units[key]
        if unit.get("state") != "verified":
            raise ValueError(f"frozen source unit is not verified: {key}")
        path = source_dir / "results" / "raw" / model_id / f"{lesson_id}.jsonl"
        payload = path.read_bytes()
        _require_hash(payload, str(unit["result_sha256"]), f"source unit {key}")
        rows = [
            json.loads(line)
            for line in payload.decode().splitlines()
            if line.strip()
        ]
        if len(rows) != 48:
            raise ValueError(f"frozen source row count changed: {key}")
        for row in rows:
            row_key = (
                str(row["model_id"]),
                str(row["lesson_id"]),
                str(row["arm"]),
                str(row["probe_id"]),
            )
            record = audit_index.get(row_key)
            if (
                row_key in all_keys
                or record is None
                or row.get("schema_version") != "p0d2hfc-decision-row-v1"
                or row.get("run_id") != EXPECTED_SOURCE_RUN_ID
                or row.get("candidate_audit_record_sha256")
                != record["record_sha256"]
                or row.get("training_seed") is not None
                or row.get("adapter_sha256") is not None
                or row.get("model_name") != record["model_name"]
                or row.get("model_revision") != record["model_revision"]
            ):
                raise ValueError(f"frozen source row provenance changed: {row_key}")
            _verify_source_candidate_outcome(row)
            all_keys.add(row_key)
            if (
                row_key[0] == "scale_canary"
                and row_key[2] == "external"
                and row.get("category") == "conditional_route"
            ):
                text_key = "::".join(row_key)
                selected_rows[text_key] = {
                    "row": row,
                    "source_result_path": str(path),
                    "source_raw_file_sha256": sha256(payload).hexdigest(),
                    "source_row_sha256": _json_hash(row),
                }
    if len(all_keys) != 2_304 or len(selected_rows) != 96:
        raise ValueError("frozen forced-choice source matrix changed")
    return selected_rows


def _verify_source_candidate_outcome(row: dict[str, Any]) -> None:
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 4:
        raise ValueError("frozen source row requires four candidate scores")
    ranked = rank_source_candidate_scores(candidates)
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
        raise ValueError("frozen source ranking changed")
    if bool(row.get("correct")) != (
        row.get("predicted_action") == row.get("expected_action")
    ):
        raise ValueError("frozen source correctness changed")


def audit_request(request: P0D2HCRDRequest) -> Path:
    config, bank, _, source_rows, bank_audit = resolve_request(request)
    bank_path, bank_audit_hash = _write_bank_audit(config, bank_audit)
    path, _, _, report = _prepare_candidate_token_audit(
        config,
        bank,
        source_rows,
        bank_audit_hash,
    )
    if report.get("all_checks_passed") is not True:
        raise RuntimeError(
            "route-decomposition candidate-token audit failed; scoring forbidden"
        )
    _progress(f"bank audit ready: {bank_path}")
    return path


def _write_bank_audit(
    config: ResolvedP0D2HCRDConfig,
    bank_audit: dict[str, Any],
) -> tuple[Path, str]:
    if (
        bank_audit.get("bank_sha256") != config.decomposition_bank_sha256
        or bank_audit.get("all_checks_passed") is not True
        or bank_audit.get("decision_count") != config.expected_decision_row_count
        or bank_audit.get("candidate_count")
        != config.expected_candidate_sequence_count
    ):
        raise ValueError("route-decomposition bank audit identity changed")
    payload = (
        json.dumps(bank_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    path = config.output_dir / "preflight" / "decomposition_bank.json"
    _write_immutable(path, payload, "decomposition-bank audit")
    return path, sha256(payload).hexdigest()


def _prepare_candidate_token_audit(
    config: ResolvedP0D2HCRDConfig,
    bank: dict[str, tuple[DecompositionProbe, ...]],
    source_rows: SourceRowIndex,
    bank_audit_sha256: str,
) -> tuple[Path, str, CandidateAuditIndex, dict[str, Any]]:
    _progress("CPU audit loading scale-canary tokenizer")
    tokenizer = load_tokenizer(
        config.model.model_name,
        config.model.model_revision,
    )
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("scale-canary tokenizer has no frozen chat template")
    records: list[dict[str, Any]] = []
    audits: CandidateAuditIndex = {}
    for lesson_id in config.selected_lesson_ids:
        for probe in bank[lesson_id]:
            source = source_rows.get(probe.source_row_key)
            if source is None:
                raise ValueError(
                    f"missing linked forced-choice source row: {probe.source_row_key}"
                )
            source_row = source["row"]
            formatted = chat_prompt(tokenizer, probe.prompt)
            token_audit = audit_candidate_tokenization(
                tokenizer,
                formatted,
                probe.ordered_candidates,
                evaluation_max_length=config.evaluation_max_length,
            )
            source_matches = (
                source_row["lesson_id"] == probe.lesson_id
                and source_row["probe_id"] == probe.source_probe_id
                and source_row["category"] == "conditional_route"
                and source_row["arm"] == "external"
                and source_row["expected_action"]
                == (
                    probe.expected_candidate
                    if probe.endpoint != "route_only"
                    else source_row["expected_action"]
                )
            )
            record = {
                "model_id": config.model.model_id,
                "model_name": config.model.model_name,
                "model_revision": config.model.model_revision,
                "lesson_id": probe.lesson_id,
                "pair_id": probe.pair_id,
                "lesson_type": probe.lesson_type,
                "lesson_side": probe.lesson_side,
                "probe_id": probe.probe_id,
                "endpoint": probe.endpoint,
                "expected_candidate": probe.expected_candidate,
                "ordered_candidates": list(probe.ordered_candidates),
                "prompt_rendering_version": config.prompt_renderers[probe.endpoint],
                "bank_record_sha256": _json_hash(probe.to_dict()),
                "source_probe_id": probe.source_probe_id,
                "source_row_key": probe.source_row_key,
                "source_row_sha256": source["source_row_sha256"],
                "source_raw_file_sha256": source["source_raw_file_sha256"],
                "source_result_path": source["source_result_path"],
                "source_matches": source_matches,
                **token_audit,
            }
            record["record_sha256"] = _json_hash(record)
            if probe.probe_id in audits:
                raise ValueError(f"duplicate candidate audit key: {probe.probe_id}")
            audits[probe.probe_id] = record
            records.append(record)
    failed = [
        record
        for record in records
        if not record["source_matches"]
        or not record["all_candidates_valid"]
    ]
    candidates = [
        candidate
        for record in records
        for candidate in record["candidates"]
    ]
    report = {
        "schema_version": "p0d2hcrd-candidate-token-audit-v1",
        "candidate_scoring_version": CANDIDATE_SCORING_VERSION,
        "config_identity_sha256": _json_hash(config.identity_dict()),
        "bank_audit_sha256": bank_audit_sha256,
        "source_manifest_sha256": config.source_manifest_sha256,
        "source_raw_results_sha256": config.source_raw_results_sha256,
        "evaluation_max_length": config.evaluation_max_length,
        "model_count": 1,
        "decision_count": len(records),
        "candidate_count": len(candidates),
        "failed_decision_count": len(failed),
        "input_truncated_count": 0,
        "source_mismatch_count": sum(
            not bool(record["source_matches"]) for record in records
        ),
        "prompt_prefix_mismatch_count": sum(
            not bool(candidate["prompt_prefix_verified"])
            for candidate in candidates
        ),
        "standalone_candidate_mismatch_count": sum(
            not bool(candidate["standalone_token_ids_match"])
            for candidate in candidates
        ),
        "decoded_candidate_mismatch_count": sum(
            not bool(candidate["decoded_continuation_matches"])
            for candidate in candidates
        ),
        "full_sequence_overflow_count": sum(
            not bool(candidate["full_sequence_fits"])
            for candidate in candidates
        ),
        "all_checks_passed": (
            not failed
            and len(records) == config.expected_decision_row_count
            and len(candidates) == config.expected_candidate_sequence_count
        ),
        "chat_template_sha256": sha256(chat_template.encode()).hexdigest(),
        "max_untruncated_prompt_tokens": max(
            int(record["untruncated_prompt_token_count"]) for record in records
        ),
        "max_full_input_tokens": max(
            int(candidate["full_input_token_count"])
            for candidate in candidates
        ),
        "records": records,
    }
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    path = config.output_dir / "preflight" / "candidate_token_audit.json"
    _write_immutable(path, payload, "candidate-token audit")
    _progress(
        f"CPU audit complete: decisions={len(records)} "
        f"candidates={len(candidates)} failed={len(failed)}"
    )
    return path, sha256(payload).hexdigest(), audits, report


def _load_bank_audit(
    config: ResolvedP0D2HCRDConfig,
) -> tuple[str, dict[str, Any]]:
    path = config.output_dir / "preflight" / "decomposition_bank.json"
    payload = path.read_bytes()
    report = json.loads(payload)
    if (
        report.get("schema_version") != "p0d2hcrd-bank-audit-v1"
        or report.get("bank_sha256") != config.decomposition_bank_sha256
        or report.get("decision_count") != config.expected_decision_row_count
        or report.get("candidate_count")
        != config.expected_candidate_sequence_count
        or report.get("all_checks_passed") is not True
    ):
        raise ValueError("decomposition-bank audit is incomplete or mismatched")
    return sha256(payload).hexdigest(), report


def _load_candidate_token_audit(
    config: ResolvedP0D2HCRDConfig,
    bank_audit_sha256: str,
) -> tuple[str, CandidateAuditIndex, dict[str, Any]]:
    path = config.output_dir / "preflight" / "candidate_token_audit.json"
    payload = path.read_bytes()
    report = json.loads(payload)
    if (
        report.get("schema_version")
        != "p0d2hcrd-candidate-token-audit-v1"
        or report.get("candidate_scoring_version")
        != CANDIDATE_SCORING_VERSION
        or report.get("config_identity_sha256")
        != _json_hash(config.identity_dict())
        or report.get("bank_audit_sha256") != bank_audit_sha256
        or report.get("source_manifest_sha256")
        != config.source_manifest_sha256
        or report.get("source_raw_results_sha256")
        != config.source_raw_results_sha256
        or report.get("decision_count") != config.expected_decision_row_count
        or report.get("candidate_count")
        != config.expected_candidate_sequence_count
        or report.get("failed_decision_count") != 0
        or report.get("input_truncated_count") != 0
        or report.get("all_checks_passed") is not True
    ):
        raise ValueError("candidate-token audit is incomplete or mismatched")
    audits: CandidateAuditIndex = {}
    records_valid = True
    for record in report.get("records", []):
        probe_id = str(record["probe_id"])
        without_hash = {
            key: value
            for key, value in record.items()
            if key != "record_sha256"
        }
        if (
            probe_id in audits
            or record.get("record_sha256") != _json_hash(without_hash)
            or record.get("source_matches") is not True
            or record.get("all_candidates_valid") is not True
            or len(record.get("candidates", []))
            not in {2, 4}
        ):
            records_valid = False
        audits[probe_id] = record
    if (
        len(audits) != config.expected_decision_row_count
        or not records_valid
    ):
        raise ValueError("candidate-token audit contains invalid row records")
    return sha256(payload).hexdigest(), audits, report


def run_experiment(request: P0D2HCRDRequest) -> Path:
    config, bank, _, source_rows, _ = resolve_request(request)
    bank_audit_hash, _ = _load_bank_audit(config)
    candidate_audit_hash, audits, _ = _load_candidate_token_audit(
        config,
        bank_audit_hash,
    )
    _require_cuda_runtime()
    run_id = "p0d2hcrd-route-decomposition-" + _json_hash(
        {
            "config": config.identity_dict(),
            "bank_audit_sha256": bank_audit_hash,
            "candidate_token_audit_sha256": candidate_audit_hash,
        }
    )[:10]
    manifest = P0D2HCRDManifest.load_or_create(
        config.output_dir / "manifest.json",
        run_id=run_id,
        config=config.identity_dict(),
        source_manifest_path=str(request.source_manifest),
        bank_audit_sha256=bank_audit_hash,
        candidate_token_audit_sha256=candidate_audit_hash,
    )
    _write_environment(config.output_dir)
    session_id, environment_fingerprint = _record_source_session(config, run_id)
    _run_model(
        config,
        run_id,
        manifest,
        bank,
        source_rows,
        audits,
        session_id,
        environment_fingerprint,
    )
    _progress(f"completed manifest={manifest.path}")
    return manifest.path


def _run_model(
    config: ResolvedP0D2HCRDConfig,
    run_id: str,
    manifest: P0D2HCRDManifest,
    bank: dict[str, tuple[DecompositionProbe, ...]],
    source_rows: SourceRowIndex,
    audits: CandidateAuditIndex,
    session_id: str,
    environment_fingerprint: str,
) -> None:
    failed = [
        lesson_id
        for lesson_id in config.selected_lesson_ids
        if manifest.unit_state(lesson_id) == "failed"
    ]
    if failed:
        raise RuntimeError(
            f"route-decomposition failures are immutable: {failed}"
        )
    for lesson_id in config.selected_lesson_ids:
        path = _result_path(config.output_dir, lesson_id)
        state = manifest.unit_state(lesson_id)
        if state != "verified" and path.exists():
            try:
                precision = _verify_rows(
                    config,
                    run_id,
                    path,
                    bank[lesson_id],
                    source_rows,
                    audits,
                    expected_precision=None,
                )
                manifest.mark_unit(
                    lesson_id,
                    "verified",
                    result_path=str(path),
                    evaluation_precision=precision,
                    result_sha256=_file_hash(path),
                )
            except (RuntimeError, ValueError, OSError) as error:
                manifest.mark_unit(lesson_id, "failed")
                manifest.record_error(lesson_id, error)
                raise
    pending = [
        lesson_id
        for lesson_id in config.selected_lesson_ids
        if manifest.unit_state(lesson_id) != "verified"
    ]
    if not pending:
        for lesson_id in config.selected_lesson_ids:
            path = _result_path(config.output_dir, lesson_id)
            unit = manifest.payload["units"][unit_key(lesson_id)]
            _verify_rows(
                config,
                run_id,
                path,
                bank[lesson_id],
                source_rows,
                audits,
                expected_precision=str(unit["evaluation_precision"]),
            )
            if unit.get("result_sha256") != _file_hash(path):
                raise ValueError(
                    f"verified route-decomposition raw file changed: {path}"
                )
        _progress("all 24 units already verified and hash-checked")
        return

    _progress(
        f"loading one base model on CUDA: "
        f"{config.model.model_id}/{config.model.model_name}"
    )
    bundle = load_base_model(config.evaluation_config())
    _require_cuda_bundle(bundle)
    if bundle.model_revision != config.model.model_revision:
        release_model(bundle)
        raise RuntimeError(
            f"loaded revision differs from frozen source: "
            f"{bundle.model_revision} != {config.model.model_revision}"
        )
    try:
        for lesson_id in config.selected_lesson_ids:
            _run_unit(
                config,
                bundle,
                run_id,
                manifest,
                bank[lesson_id],
                source_rows,
                audits,
                session_id,
                environment_fingerprint,
            )
    finally:
        release_model(bundle)


def _run_unit(
    config: ResolvedP0D2HCRDConfig,
    bundle: ModelBundle,
    run_id: str,
    manifest: P0D2HCRDManifest,
    probes: tuple[DecompositionProbe, ...],
    source_rows: SourceRowIndex,
    audits: CandidateAuditIndex,
    session_id: str,
    environment_fingerprint: str,
) -> None:
    lesson_id = probes[0].lesson_id
    path = _result_path(config.output_dir, lesson_id)
    state = manifest.unit_state(lesson_id)
    if state == "failed":
        raise RuntimeError(
            f"route-decomposition unit is immutable after failure: {lesson_id}"
        )
    if state == "verified" or path.exists():
        expected_precision = None
        if state == "verified":
            unit = manifest.payload["units"][unit_key(lesson_id)]
            expected_precision = str(unit["evaluation_precision"])
            if unit.get("result_sha256") != _file_hash(path):
                raise ValueError(
                    f"verified route-decomposition raw file changed: {path}"
                )
        precision = _verify_rows(
            config,
            run_id,
            path,
            probes,
            source_rows,
            audits,
            expected_precision=expected_precision,
        )
        if state != "verified":
            manifest.mark_unit(
                lesson_id,
                "verified",
                result_path=str(path),
                evaluation_precision=precision,
                result_sha256=_file_hash(path),
            )
        return
    try:
        _progress(f"scoring scale_canary/{lesson_id}")
        manifest.mark_unit(lesson_id, "evaluating")
        rows: list[dict[str, Any]] = []
        for endpoint in ENDPOINTS:
            _progress(f"heartbeat scale_canary/{lesson_id}/{endpoint}")
            for probe in probes:
                if probe.endpoint != endpoint:
                    continue
                audit = audits[probe.probe_id]
                formatted = chat_prompt(bundle.tokenizer, probe.prompt)
                try:
                    outcome = score_candidate_batch(bundle, formatted, audit)
                except CandidateTokenMismatch as error:
                    outcome = _candidate_token_error(audit, error)
                rows.append(
                    _decision_row(
                        config,
                        run_id,
                        bundle.precision,
                        probe,
                        audit,
                        source_rows[probe.source_row_key],
                        outcome,
                        session_id,
                        environment_fingerprint,
                    )
                )
        write_probe_rows(path, rows)
        precision = _verify_rows(
            config,
            run_id,
            path,
            probes,
            source_rows,
            audits,
            expected_precision=bundle.precision,
        )
        manifest.mark_unit(
            lesson_id,
            "verified",
            result_path=str(path),
            evaluation_precision=precision,
            result_sha256=_file_hash(path),
        )
    except (RuntimeError, ValueError, OSError) as error:
        manifest.mark_unit(lesson_id, "failed")
        manifest.record_error(lesson_id, error)
        raise


def _decision_row(
    config: ResolvedP0D2HCRDConfig,
    run_id: str,
    precision: str,
    probe: DecompositionProbe,
    audit: dict[str, Any],
    source: dict[str, Any],
    outcome: dict[str, Any],
    session_id: str,
    environment_fingerprint: str,
) -> dict[str, Any]:
    source_row = source["row"]
    predicted = outcome["predicted_candidate"]
    return {
        "schema_version": "p0d2hcrd-decision-row-v1",
        "run_id": run_id,
        "model_id": config.model.model_id,
        "model_role": config.model.role,
        "model_name": config.model.model_name,
        "model_revision": config.model.model_revision,
        "precision": precision,
        "lesson_id": probe.lesson_id,
        "pair_id": probe.pair_id,
        "lesson_type": probe.lesson_type,
        "lesson_side": probe.lesson_side,
        "probe_id": probe.probe_id,
        "endpoint": probe.endpoint,
        "expected_candidate": probe.expected_candidate,
        "ordered_candidates": list(probe.ordered_candidates),
        "route_variant": probe.route_variant,
        "target_slot": probe.target_slot,
        "current_marker": probe.current_marker,
        "slot_candidate_order": probe.slot_candidate_order,
        "slot_content_order": probe.slot_content_order,
        "action_panel_position": probe.action_panel_position,
        "prompt_rendering_version": audit["prompt_rendering_version"],
        "prompt_sha256": audit["prompt_sha256"],
        "untruncated_prompt_token_count": audit[
            "untruncated_prompt_token_count"
        ],
        "evaluation_max_length": config.evaluation_max_length,
        "candidate_scoring_version": CANDIDATE_SCORING_VERSION,
        "candidate_audit_record_sha256": audit["record_sha256"],
        "bank_record_sha256": audit["bank_record_sha256"],
        "candidates": outcome["candidates"],
        "predicted_candidate": predicted,
        "mean_predicted_candidate": outcome["mean_predicted_candidate"],
        "top1_top2_margin": outcome["top1_top2_margin"],
        "sum_mean_disagreement": outcome["sum_mean_disagreement"],
        "correct": predicted == probe.expected_candidate,
        "tie": outcome["tie"],
        "mean_score_tie": outcome["mean_score_tie"],
        "non_finite": outcome["non_finite"],
        "error_status": outcome["error_status"],
        "error_message": outcome["error_message"],
        "latency_seconds": outcome["latency_seconds"],
        "source_run_id": config.source_run_id,
        "source_manifest_sha256": config.source_manifest_sha256,
        "source_summary_sha256": config.source_summary_sha256,
        "source_candidate_token_audit_sha256": (
            config.source_candidate_token_audit_sha256
        ),
        "source_raw_results_sha256": config.source_raw_results_sha256,
        "source_probe_id": probe.source_probe_id,
        "source_row_key": probe.source_row_key,
        "source_row_sha256": source["source_row_sha256"],
        "source_result_path": source["source_result_path"],
        "source_raw_file_sha256": source["source_raw_file_sha256"],
        "source_fc_predicted_action": source_row["predicted_action"],
        "source_fc_correct": bool(source_row["correct"]),
        "source_fc_top1_top2_margin": source_row["top1_top2_margin"],
        "source_calibration_manifest_sha256": (
            config.source_calibration_manifest_sha256
        ),
        "source_hard_probe_manifest_sha256": (
            config.source_hard_probe_manifest_sha256
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
                "candidate": candidate["candidate"],
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
        "predicted_candidate": None,
        "mean_predicted_candidate": None,
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
    config: ResolvedP0D2HCRDConfig,
    run_id: str,
    path: Path,
    probes: tuple[DecompositionProbe, ...],
    source_rows: SourceRowIndex,
    audits: CandidateAuditIndex,
    *,
    expected_precision: str | None,
) -> str:
    if not path.exists():
        raise FileNotFoundError(f"missing route-decomposition result: {path}")
    rows = read_probe_results(path)
    expected = {probe.probe_id for probe in probes}
    observed = [str(row.get("probe_id")) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(f"route-decomposition result matrix mismatch: {path}")
    precisions = {str(row.get("precision")) for row in rows}
    if len(precisions) != 1 or (
        expected_precision is not None
        and precisions != {expected_precision}
    ):
        raise ValueError(f"route-decomposition precision mismatch: {path}")
    probes_by_id = {probe.probe_id: probe for probe in probes}
    sessions = _session_index(config.output_dir)
    for row in rows:
        probe = probes_by_id[str(row["probe_id"])]
        audit = audits.get(probe.probe_id)
        source = source_rows.get(probe.source_row_key)
        if audit is None or source is None:
            raise ValueError(
                f"route-decomposition audit/source missing: {probe.probe_id}"
            )
        source_row = source["row"]
        session = sessions.get(str(row.get("execution_session_id")))
        basic_valid = (
            row.get("schema_version") == "p0d2hcrd-decision-row-v1"
            and row.get("run_id") == run_id
            and row.get("model_id") == config.model.model_id
            and row.get("model_role") == config.model.role
            and row.get("model_name") == config.model.model_name
            and row.get("model_revision") == config.model.model_revision
            and row.get("lesson_id") == probe.lesson_id
            and row.get("pair_id") == probe.pair_id
            and row.get("lesson_type") == probe.lesson_type
            and row.get("lesson_side") == probe.lesson_side
            and row.get("endpoint") == probe.endpoint
            and row.get("expected_candidate") == probe.expected_candidate
            and row.get("ordered_candidates") == list(probe.ordered_candidates)
            and row.get("route_variant") == probe.route_variant
            and row.get("target_slot") == probe.target_slot
            and row.get("current_marker") == probe.current_marker
            and row.get("slot_candidate_order") == probe.slot_candidate_order
            and row.get("slot_content_order") == probe.slot_content_order
            and row.get("action_panel_position") == probe.action_panel_position
            and row.get("prompt_sha256") == audit["prompt_sha256"]
            and row.get("prompt_rendering_version")
            == audit["prompt_rendering_version"]
            and row.get("candidate_audit_record_sha256")
            == audit["record_sha256"]
            and row.get("bank_record_sha256") == audit["bank_record_sha256"]
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
            and row.get("source_candidate_token_audit_sha256")
            == config.source_candidate_token_audit_sha256
            and row.get("source_raw_results_sha256")
            == config.source_raw_results_sha256
            and row.get("source_probe_id") == probe.source_probe_id
            and row.get("source_row_key") == probe.source_row_key
            and row.get("source_row_sha256") == source["source_row_sha256"]
            and row.get("source_result_path") == source["source_result_path"]
            and row.get("source_raw_file_sha256")
            == source["source_raw_file_sha256"]
            and row.get("source_fc_predicted_action")
            == source_row["predicted_action"]
            and bool(row.get("source_fc_correct"))
            == bool(source_row["correct"])
            and row.get("source_fc_top1_top2_margin")
            == source_row["top1_top2_margin"]
            and row.get("source_calibration_manifest_sha256")
            == config.source_calibration_manifest_sha256
            and row.get("source_hard_probe_manifest_sha256")
            == config.source_hard_probe_manifest_sha256
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
                f"route-decomposition provenance mismatch: "
                f"{path}/{probe.probe_id}"
            )
        _verify_candidate_outcome(row, audit, probe)
    return next(iter(precisions))


def _verify_candidate_outcome(
    row: dict[str, Any],
    audit: dict[str, Any],
    probe: DecompositionProbe,
) -> None:
    candidates = row.get("candidates")
    expected_count = len(probe.ordered_candidates)
    if not isinstance(candidates, list) or len(candidates) != expected_count:
        raise ValueError("route-decomposition candidate count changed")
    for observed, expected in zip(
        candidates,
        audit["candidates"],
        strict=True,
    ):
        if (
            observed.get("candidate") != expected["candidate"]
            or observed.get("continuation") != expected["continuation"]
            or observed.get("token_ids") != expected["token_ids"]
            or int(observed.get("token_count", -1))
            != int(expected["token_count"])
        ):
            raise ValueError("candidate token provenance mismatch")
    status = str(row.get("error_status"))
    if status == "candidate_token_mismatch":
        if (
            row.get("predicted_candidate") is not None
            or bool(row.get("correct"))
            or any(
                candidate.get("sum_logprob") is not None
                for candidate in candidates
            )
        ):
            raise ValueError("malformed explicit candidate-token error row")
        return
    if status not in {"ok", "tie", "non_finite"}:
        raise ValueError(f"unsupported route-decomposition status: {status}")
    ranked = rank_candidate_scores(candidates)
    fields = (
        "predicted_candidate",
        "mean_predicted_candidate",
        "top1_top2_margin",
        "sum_mean_disagreement",
        "tie",
        "mean_score_tie",
        "non_finite",
        "error_status",
    )
    if any(row.get(field) != ranked[field] for field in fields):
        raise ValueError("stored route-decomposition ranking differs from scores")
    if bool(row.get("correct")) != (
        row.get("predicted_candidate") == probe.expected_candidate
    ):
        raise ValueError("stored route-decomposition correctness changed")
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
        token_logprobs = observed.get("token_logprobs")
        if (
            observed.get("sum_rank") != expected["sum_rank"]
            or observed.get("mean_rank") != expected["mean_rank"]
            or not isinstance(token_logprobs, list)
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
            raise ValueError("candidate scores or ranks are inconsistent")


def _record_source_session(
    config: ResolvedP0D2HCRDConfig,
    run_id: str,
) -> tuple[str, str]:
    session_id = f"p0d2hcrd-session-{uuid4().hex[:12]}"
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
        raise FileNotFoundError(
            f"missing route-decomposition source sessions: {path}"
        )
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


def _raw_tree_hash(output_dir: Path, expected_files: int) -> str:
    root = output_dir / "results" / "raw"
    files = sorted(root.rglob("*.jsonl"))
    if len(files) != expected_files:
        raise ValueError(
            f"expected {expected_files} raw files, found {len(files)}"
        )
    digest = sha256()
    for path in files:
        digest.update(str(path.relative_to(output_dir)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _result_path(output_dir: Path, lesson_id: str) -> Path:
    return (
        output_dir
        / "results"
        / "raw"
        / "scale_canary"
        / f"{lesson_id}.jsonl"
    )


def _require_hash(payload: bytes, expected: str, label: str) -> None:
    observed = sha256(payload).hexdigest()
    if observed != expected:
        raise ValueError(f"{label} hash changed: {observed} != {expected}")


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    _require_hash(path.read_bytes(), expected, label)


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
            "route-decomposition output must be independent from its source"
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
