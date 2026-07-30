from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.domain import Arm, P0CProbe
from plasticity_placement.p0c.modeling import (
    ModelBundle,
    load_adapter_model,
    release_model,
    write_probe_rows,
)
from plasticity_placement.p0c.runtime import current_code_hash
from plasticity_placement.p0d2h.probes import load_hard_probe_bank
from plasticity_placement.p0d2hc.config import CALIBRATION_ARMS
from plasticity_placement.p0d2hc.prompting import render_calibration_probe
from plasticity_placement.p0d2hcrd.probes import (
    DecompositionProbe,
    compile_decomposition_bank,
)
from plasticity_placement.p0d2hcrd.runtime import validate_frozen_source
from plasticity_placement.p0d2hcrd.scoring import (
    score_candidate_batch as score_crd_candidate_batch,
)
from plasticity_placement.p0d2hfc.scoring import (
    score_candidate_batch as score_fc_candidate_batch,
)
from plasticity_placement.p0d2hrr.authorization import adopt_authorization
from plasticity_placement.p0d2hrr.config import ResolvedPilotConfig
from plasticity_placement.p0d2hrr.io import file_hash, json_hash, read_json_object
from plasticity_placement.p0d2hrr.manifest import PilotManifest
from plasticity_placement.p0d2hrr.preflight import (
    load_resolved_config,
    load_scale_canary_source_rows,
)
from plasticity_placement.training.lora import (
    _adapter_bundle_files,
    _files_hash,
    train_lora,
)
from plasticity_placement.training.model_utils import chat_prompt


@dataclass(frozen=True, slots=True)
class _AdapterLoadConfig:
    model_name: str
    model_revision: str
    use_4bit: bool
    training_seeds: tuple[int, ...]


def authorize_experiment(
    output_dir: Path,
    authorization_path: Path,
) -> Path:
    output_dir = output_dir.resolve()
    authorization_path = authorization_path.resolve()
    config = load_resolved_config(output_dir)
    manifest = PilotManifest.load(output_dir / "manifest.json")
    if manifest.state != "planned":
        raise ValueError(f"authorization requires state=planned, found {manifest.state}")
    payload, digest = adopt_authorization(
        output_dir,
        authorization_path,
        preregistration_sha256=config.preregistration_sha256,
    )
    manifest.transition(
        "authorized",
        authorization={
            "sha256": digest,
            "approved_by": payload["approved_by"],
            "approved_at": payload["approved_at"],
            "scope": payload["scope"],
        },
    )
    return manifest.path


def train_experiment(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    config = load_resolved_config(output_dir)
    manifest = PilotManifest.load(output_dir / "manifest.json")
    if manifest.state != "authorized":
        raise PermissionError(f"training requires state=authorized, found {manifest.state}")
    _validate_execution_identity(config, manifest)
    adapter_dir = output_dir / "adapter"
    if adapter_dir.exists():
        raise FileExistsError("route-remediation adapter output already exists; use a new attempt")
    training_config = config.spec.training.to_lora_config(
        model_name=config.spec.model_name,
        model_revision=config.spec.model_revision,
        data_path=output_dir / "preflight" / "route_train.jsonl",
        output_dir=adapter_dir,
    )
    manifest.transition("training")
    try:
        summary = train_lora(training_config)
        if (
            summary.optimizer_steps != config.spec.training.max_steps
            or summary.training_data_sha256 != config.train_data_sha256
            or summary.model_revision != config.spec.model_revision
        ):
            raise ValueError("route-remediation training summary differs from preregistration")
        metadata_path = adapter_dir / "training_metadata.json"
        manifest.transition(
            "trained",
            training={
                "summary": asdict(summary),
                "training_metadata_sha256": file_hash(metadata_path),
                "completed_training_runs": 1,
                "checkpoint_selection": (config.spec.training.checkpoint_selection),
            },
        )
    except BaseException as error:
        manifest.record_error("training", error)
        raise
    return manifest.path


def evaluate_experiment(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    config = load_resolved_config(output_dir)
    manifest = PilotManifest.load(output_dir / "manifest.json")
    if manifest.state != "trained":
        raise PermissionError(f"locked evaluation requires state=trained, found {manifest.state}")
    _validate_execution_identity(config, manifest)
    training = manifest.payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("route-remediation training record is missing")
    adapter_sha256 = _adapter_hash(output_dir / "adapter")
    if adapter_sha256 != training["summary"]["adapter_sha256"]:
        raise ValueError("route-remediation adapter hash changed")

    source = validate_frozen_source(config.source_manifest_path)
    source_rows = load_scale_canary_source_rows(source)
    hard_bank = load_hard_probe_bank(source.hard_probe_dir)
    crd_bank, crd_bank_audit = compile_decomposition_bank(
        source.selected,
        hard_bank,
    )
    stored_crd_bank_audit = read_json_object(
        output_dir / "preflight" / "crd_bank_audit.json",
        "CRD bank audit",
    )
    if crd_bank_audit != stored_crd_bank_audit:
        raise ValueError("route-remediation CRD bank changed after preregistration")

    fc_audits = _load_fc_audits(output_dir, config)
    crd_audits = _load_crd_audits(output_dir, config)
    dev_records = _load_dev_records(output_dir)
    dev_audits = _load_dev_audits(output_dir, config)
    load_config = _AdapterLoadConfig(
        model_name=config.spec.model_name,
        model_revision=config.spec.model_revision,
        use_4bit=config.spec.training.use_4bit,
        training_seeds=(config.spec.training.seed,),
    )
    manifest.transition("evaluating")
    bundle: ModelBundle | None = None
    try:
        _require_cuda()
        bundle = load_adapter_model(load_config, output_dir / "adapter")
        _require_source_compatible_bundle(bundle, config)
        dev_rows = _evaluate_dev(
            bundle=bundle,
            config=config,
            run_id=str(manifest.payload["run_id"]),
            adapter_sha256=adapter_sha256,
            records=dev_records,
            audits=dev_audits,
        )
        selected_by_id = {item.lesson.lesson_id: item for item in source.selected}
        fc_rows = _evaluate_forced_choice(
            bundle=bundle,
            config=config,
            run_id=str(manifest.payload["run_id"]),
            adapter_sha256=adapter_sha256,
            selected_by_id=selected_by_id,
            hard_bank=hard_bank,
            source_rows=source_rows,
            audits=fc_audits,
        )
        crd_rows = _evaluate_crd(
            bundle=bundle,
            config=config,
            run_id=str(manifest.payload["run_id"]),
            adapter_sha256=adapter_sha256,
            bank=crd_bank,
            source_rows=source.source_rows,
            audits=crd_audits,
        )
        dev_path = output_dir / "results" / "dev" / "rows.jsonl"
        fc_path = output_dir / "results" / "forced_choice" / "rows.jsonl"
        crd_path = output_dir / "results" / "crd" / "rows.jsonl"
        write_probe_rows(dev_path, dev_rows)
        write_probe_rows(fc_path, fc_rows)
        write_probe_rows(crd_path, crd_rows)
        manifest.transition(
            "evaluated",
            evaluation={
                "precision": bundle.precision,
                "adapter_sha256": adapter_sha256,
                "dev_result_path": str(dev_path),
                "dev_result_sha256": file_hash(dev_path),
                "dev_decision_count": len(dev_rows),
                "dev_accuracy": (sum(bool(row["correct"]) for row in dev_rows) / len(dev_rows)),
                "dev_used_for_checkpoint_selection": False,
                "forced_choice_result_path": str(fc_path),
                "forced_choice_result_sha256": file_hash(fc_path),
                "forced_choice_decision_count": len(fc_rows),
                "crd_result_path": str(crd_path),
                "crd_result_sha256": file_hash(crd_path),
                "crd_decision_count": len(crd_rows),
                "completed_locked_external_evaluations": 1,
                "strict_fp32_full_set_audit_completed": False,
            },
        )
    except BaseException as error:
        manifest.record_error("evaluation", error)
        raise
    finally:
        if bundle is not None:
            release_model(bundle)
    return manifest.path


def _evaluate_dev(
    *,
    bundle: ModelBundle,
    config: ResolvedPilotConfig,
    run_id: str,
    adapter_sha256: str,
    records: list[dict[str, Any]],
    audits: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        example_id = str(record["example_id"])
        audit = audits[example_id]
        formatted = chat_prompt(bundle.tokenizer, str(record["prompt"]))
        outcome = score_crd_candidate_batch(bundle, formatted, audit)
        predicted = outcome["predicted_candidate"]
        rows.append(
            {
                "schema_version": "p0d2hrr-dev-decision-row-v1",
                "run_id": run_id,
                "model_id": "scale_canary_route_remediated",
                "model_name": config.spec.model_name,
                "model_revision": config.spec.model_revision,
                "precision": bundle.precision,
                "adapter_sha256": adapter_sha256,
                "example_id": example_id,
                "group_id": record["group_id"],
                "task": record["task"],
                "template_id": record["template_id"],
                "target_slot": record["target_slot"],
                "slot_content_order": record["slot_content_order"],
                "expected_candidate": record["expected_completion"],
                "prompt_sha256": audit["prompt_sha256"],
                "candidate_audit_record_sha256": audit["record_sha256"],
                "candidates": outcome["candidates"],
                "predicted_candidate": predicted,
                "mean_predicted_candidate": outcome["mean_predicted_candidate"],
                "top1_top2_margin": outcome["top1_top2_margin"],
                "sum_mean_disagreement": outcome["sum_mean_disagreement"],
                "correct": predicted == record["expected_completion"],
                "tie": outcome["tie"],
                "mean_score_tie": outcome["mean_score_tie"],
                "non_finite": outcome["non_finite"],
                "error_status": outcome["error_status"],
                "error_message": outcome["error_message"],
                "latency_seconds": outcome["latency_seconds"],
                "source_run_id": config.source_run_id,
                "source_manifest_sha256": config.source_manifest_sha256,
            }
        )
    if len(rows) != 96:
        raise ValueError("route-remediation dev matrix is incomplete")
    return rows


def _evaluate_forced_choice(
    *,
    bundle: ModelBundle,
    config: ResolvedPilotConfig,
    run_id: str,
    adapter_sha256: str,
    selected_by_id: dict[str, Any],
    hard_bank: dict[str, tuple[P0CProbe, ...]],
    source_rows: dict[tuple[str, str, str], dict[str, Any]],
    audits: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lesson_id in _source_lesson_ids(selected_by_id):
        item = selected_by_id[lesson_id]
        for arm_value in CALIBRATION_ARMS:
            for probe in hard_bank[lesson_id]:
                key = (lesson_id, arm_value, probe.probe_id)
                audit = audits[key]
                source = source_rows[key]
                rendered, variant, renderer = render_calibration_probe(
                    probe,
                    arm=Arm(arm_value),
                    external_note=item.external_note,
                )
                formatted = chat_prompt(bundle.tokenizer, rendered.prompt)
                outcome = score_fc_candidate_batch(bundle, formatted, audit)
                predicted = outcome["predicted_action"]
                source_row = source["row"]
                rows.append(
                    {
                        "schema_version": "p0d2hrr-fc-decision-row-v1",
                        "run_id": run_id,
                        "model_id": "scale_canary_route_remediated",
                        "model_name": config.spec.model_name,
                        "model_revision": config.spec.model_revision,
                        "precision": bundle.precision,
                        "adapter_sha256": adapter_sha256,
                        "lesson_id": lesson_id,
                        "pair_id": probe.pair_id,
                        "lesson_type": probe.lesson_type,
                        "probe_id": probe.probe_id,
                        "category": probe.category,
                        "arm": arm_value,
                        "expected_action": probe.expected_action,
                        "ordered_allowed_actions": list(probe.action_choices),
                        "prompt_variant": variant,
                        "prompt_rendering_version": renderer,
                        "prompt_sha256": audit["prompt_sha256"],
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
                        "source_row_sha256": source["source_row_sha256"],
                        "source_predicted_action": source_row["predicted_action"],
                        "source_correct": bool(source_row["correct"]),
                        "source_strict_correct": bool(source_row["source_strict_correct"]),
                    }
                )
    if len(rows) != 1_152:
        raise ValueError("route-remediation forced-choice matrix is incomplete")
    return rows


def _evaluate_crd(
    *,
    bundle: ModelBundle,
    config: ResolvedPilotConfig,
    run_id: str,
    adapter_sha256: str,
    bank: dict[str, tuple[DecompositionProbe, ...]],
    source_rows: dict[str, dict[str, Any]],
    audits: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for probes in bank.values():
        for probe in probes:
            audit = audits[probe.probe_id]
            formatted = chat_prompt(bundle.tokenizer, probe.prompt)
            outcome = score_crd_candidate_batch(bundle, formatted, audit)
            predicted = outcome["predicted_candidate"]
            source = source_rows[probe.source_row_key]
            rows.append(
                {
                    "schema_version": "p0d2hrr-crd-decision-row-v1",
                    "run_id": run_id,
                    "model_id": "scale_canary_route_remediated",
                    "model_name": config.spec.model_name,
                    "model_revision": config.spec.model_revision,
                    "precision": bundle.precision,
                    "adapter_sha256": adapter_sha256,
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
                    "prompt_sha256": audit["prompt_sha256"],
                    "candidate_audit_record_sha256": audit["record_sha256"],
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
                    "source_probe_id": probe.source_probe_id,
                    "source_row_key": probe.source_row_key,
                    "source_row_sha256": source["source_row_sha256"],
                    "source_fc_correct": bool(source["row"]["correct"]),
                }
            )
    if len(rows) != 1_536:
        raise ValueError("route-remediation CRD matrix is incomplete")
    return rows


def _load_fc_audits(
    output_dir: Path,
    config: ResolvedPilotConfig,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    report = read_json_object(
        output_dir / "preflight" / "forced_choice_token_audit.json",
        "forced-choice token audit",
    )
    if (
        file_hash(output_dir / "preflight" / "forced_choice_token_audit.json")
        != config.forced_choice_token_audit_sha256
        or report.get("schema_version") != "p0d2hrr-fc-token-audit-v1"
        or report.get("decision_count") != 1_152
        or report.get("all_checks_passed") is not True
    ):
        raise ValueError("route-remediation forced-choice token audit changed")
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in report["records"]:
        key = (
            str(record["lesson_id"]),
            str(record["arm"]),
            str(record["probe_id"]),
        )
        _verify_audit_record(record)
        if key in result:
            raise ValueError(f"duplicate forced-choice audit key: {key}")
        result[key] = record
    return result


def _load_dev_records(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "preflight" / "route_dev.jsonl"
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(rows) != 96 or len({row.get("example_id") for row in rows}) != 96:
        raise ValueError("route-remediation dev data changed")
    return rows


def _load_dev_audits(
    output_dir: Path,
    config: ResolvedPilotConfig,
) -> dict[str, dict[str, Any]]:
    report = read_json_object(
        output_dir / "preflight" / "dev_token_audit.json",
        "dev token audit",
    )
    if (
        file_hash(output_dir / "preflight" / "dev_token_audit.json")
        != config.dev_token_audit_sha256
        or report.get("schema_version") != "p0d2hrr-dev-token-audit-v1"
        or report.get("decision_count") != 96
        or report.get("all_checks_passed") is not True
    ):
        raise ValueError("route-remediation dev token audit changed")
    result: dict[str, dict[str, Any]] = {}
    for record in report["records"]:
        key = str(record["example_id"])
        _verify_audit_record(record)
        if key in result:
            raise ValueError(f"duplicate dev audit key: {key}")
        result[key] = record
    return result


def _load_crd_audits(
    output_dir: Path,
    config: ResolvedPilotConfig,
) -> dict[str, dict[str, Any]]:
    report = read_json_object(
        output_dir / "preflight" / "crd_token_audit.json",
        "CRD token audit",
    )
    if (
        file_hash(output_dir / "preflight" / "crd_token_audit.json")
        != config.crd_token_audit_sha256
        or report.get("schema_version") != "p0d2hrr-crd-token-audit-v1"
        or report.get("decision_count") != 1_536
        or report.get("all_checks_passed") is not True
    ):
        raise ValueError("route-remediation CRD token audit changed")
    result: dict[str, dict[str, Any]] = {}
    for record in report["records"]:
        key = str(record["probe_id"])
        _verify_audit_record(record)
        if key in result:
            raise ValueError(f"duplicate CRD audit key: {key}")
        result[key] = record
    return result


def _verify_audit_record(record: dict[str, Any]) -> None:
    observed = record.get("record_sha256")
    expected = json_hash({key: value for key, value in record.items() if key != "record_sha256"})
    if observed != expected or record.get("all_candidates_valid") is not True:
        raise ValueError(f"candidate audit record changed: {record.get('probe_id')}")


def _validate_execution_identity(
    config: ResolvedPilotConfig,
    manifest: PilotManifest,
) -> None:
    if current_code_hash() != config.code_sha256:
        raise ValueError("code changed after route-remediation preregistration")
    if file_hash(config.source_manifest_path) != config.source_manifest_sha256:
        raise ValueError("frozen source manifest changed")
    authorization = read_json_object(
        config.output_dir / "authorization.json",
        "authorization",
    )
    if (
        manifest.payload.get("authorization", {}).get("sha256")
        != file_hash(config.output_dir / "authorization.json")
        or authorization.get("preregistration_sha256") != config.preregistration_sha256
        or authorization.get("decision") != "approved"
    ):
        raise PermissionError("route-remediation authorization changed")


def _adapter_hash(adapter_dir: Path) -> str:
    files = _adapter_bundle_files(adapter_dir)
    return _files_hash(files, adapter_dir)


def _require_cuda() -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "route-remediation evaluation requires the training dependencies"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("route-remediation locked evaluation requires CUDA")


def _require_source_compatible_bundle(
    bundle: ModelBundle,
    config: ResolvedPilotConfig,
) -> None:
    device = next(bundle.model.parameters()).device
    if getattr(device, "type", None) != "cuda":
        raise RuntimeError("route-remediation model is not on CUDA")
    if bundle.model_revision != config.spec.model_revision:
        raise ValueError("route-remediation model revision changed")
    if bundle.precision != config.source_evaluation_precision:
        raise RuntimeError(
            "official evaluation precision differs from the frozen source: "
            f"{bundle.precision} != {config.source_evaluation_precision}"
        )


def _source_lesson_ids(selected_by_id: dict[str, Any]) -> tuple[str, ...]:
    lesson_ids = tuple(selected_by_id)
    if len(lesson_ids) != 24:
        raise ValueError("route-remediation lesson cohort changed")
    return lesson_ids
