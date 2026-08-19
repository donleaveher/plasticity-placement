from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.compiler import ACTIONS
from plasticity_placement.p0c.domain import Arm, CompiledLesson, P0CProbe
from plasticity_placement.p0c.runtime import current_code_hash
from plasticity_placement.p0d2h.probes import load_hard_probe_bank
from plasticity_placement.p0d2hc.config import CALIBRATION_ARMS
from plasticity_placement.p0d2hc.prompting import render_calibration_probe
from plasticity_placement.p0d2hcrd.probes import (
    DecompositionProbe,
    compile_decomposition_bank,
)
from plasticity_placement.p0d2hcrd.runtime import (
    FrozenSource,
    validate_frozen_source,
)
from plasticity_placement.p0d2hcrd.scoring import (
    audit_candidate_tokenization as audit_crd_tokenization,
)
from plasticity_placement.p0d2hfc.scoring import (
    audit_candidate_tokenization as audit_fc_tokenization,
)
from plasticity_placement.p0d2hrr.authorization import authorization_template
from plasticity_placement.p0d2hrr.config import (
    PilotSpec,
    ResolvedPilotConfig,
)
from plasticity_placement.p0d2hrr.data import (
    RouteTrainingRecord,
    compile_route_data,
    write_route_data,
)
from plasticity_placement.p0d2hrr.io import (
    file_hash,
    immutable_json_write,
    json_hash,
    read_json_object,
)
from plasticity_placement.p0d2hrr.manifest import PilotManifest
from plasticity_placement.training.model_utils import chat_prompt, load_tokenizer

EVALUATION_MAX_LENGTH = 512


def plan_experiment(
    *,
    output_dir: Path,
    source_manifest: Path,
    spec_path: Path,
) -> Path:
    output_dir = output_dir.resolve()
    source_manifest = source_manifest.resolve()
    spec_path = spec_path.resolve()
    _require_independent_output(source_manifest.parent, output_dir)
    spec = PilotSpec.from_path(spec_path)
    source = validate_frozen_source(source_manifest)
    source_rows = load_scale_canary_source_rows(source)
    source_precision = source_evaluation_precision(source)
    code_sha256 = current_code_hash()
    spec_sha256 = json_hash(spec.to_dict())
    context = {
        "code_sha256": code_sha256,
        "source_run_id": source.forced_choice_manifest["run_id"],
        "source_manifest_sha256": file_hash(source_manifest),
        "spec_sha256": spec_sha256,
    }
    context_sha256 = json_hash(context)

    forbidden = _frozen_lesson_strings(source.selected)
    train, dev, data_audit = compile_route_data(
        spec.data,
        forbidden_strings=forbidden,
    )
    train_hash, dev_hash, data_audit_hash = write_route_data(
        output_dir,
        train,
        dev,
        data_audit,
    )

    tokenizer = load_tokenizer(spec.model_name, spec.model_revision)
    dev_token_audit = build_dev_token_audit(
        tokenizer=tokenizer,
        records=dev,
        context_sha256=context_sha256,
    )
    dev_token_audit_hash = immutable_json_write(
        output_dir / "preflight" / "dev_token_audit.json",
        dev_token_audit,
        "route-remediation dev token audit",
    )
    hard_bank = load_hard_probe_bank(source.hard_probe_dir)
    fc_audit = build_forced_choice_token_audit(
        tokenizer=tokenizer,
        source=source,
        source_rows=source_rows,
        hard_bank=hard_bank,
        context_sha256=context_sha256,
    )
    fc_audit_hash = immutable_json_write(
        output_dir / "preflight" / "forced_choice_token_audit.json",
        fc_audit,
        "route-remediation forced-choice token audit",
    )

    crd_bank, crd_bank_audit = compile_decomposition_bank(
        source.selected,
        hard_bank,
    )
    if crd_bank_audit.get("all_checks_passed") is not True:
        raise ValueError("route-remediation CRD bank audit failed")
    crd_bank_audit_hash = immutable_json_write(
        output_dir / "preflight" / "crd_bank_audit.json",
        crd_bank_audit,
        "route-remediation CRD bank audit",
    )
    crd_audit = build_crd_token_audit(
        tokenizer=tokenizer,
        source=source,
        bank=crd_bank,
        context_sha256=context_sha256,
        bank_audit_sha256=crd_bank_audit_hash,
    )
    crd_audit_hash = immutable_json_write(
        output_dir / "preflight" / "crd_token_audit.json",
        crd_audit,
        "route-remediation CRD token audit",
    )

    summary_path = source.forced_choice_dir / "results" / "aggregate" / "summary.json"
    candidate_audit_path = source.forced_choice_dir / "preflight" / "candidate_token_audit.json"
    config = ResolvedPilotConfig(
        output_dir=output_dir,
        source_manifest_path=source_manifest,
        code_sha256=code_sha256,
        source_run_id=str(source.forced_choice_manifest["run_id"]),
        source_manifest_sha256=file_hash(source_manifest),
        source_summary_sha256=file_hash(summary_path),
        source_candidate_token_audit_sha256=file_hash(candidate_audit_path),
        source_raw_results_sha256=_forced_choice_raw_hash(source.forced_choice_dir),
        spec=spec,
        spec_sha256=spec_sha256,
        train_data_sha256=train_hash,
        dev_data_sha256=dev_hash,
        data_audit_sha256=data_audit_hash,
        dev_token_audit_sha256=dev_token_audit_hash,
        forced_choice_token_audit_sha256=fc_audit_hash,
        crd_bank_audit_sha256=crd_bank_audit_hash,
        crd_token_audit_sha256=crd_audit_hash,
        source_evaluation_precision=source_precision,
    )
    config_payload = config.identity_dict()
    immutable_json_write(
        output_dir / "preregistration.json",
        config_payload,
        "route-remediation preregistration",
    )
    immutable_json_write(
        output_dir / "authorization.template.json",
        authorization_template(config.preregistration_sha256),
        "route-remediation authorization template",
    )
    run_id = f"p0d2hrr-{config.preregistration_sha256[:10]}"
    manifest = PilotManifest.load_or_create(
        output_dir / "manifest.json",
        run_id=run_id,
        config=config_payload,
        source_manifest_path=str(source_manifest),
    )
    return manifest.path


def load_resolved_config(output_dir: Path) -> ResolvedPilotConfig:
    output_dir = output_dir.resolve()
    manifest = PilotManifest.load(output_dir / "manifest.json")
    identity = manifest.payload.get("config")
    if not isinstance(identity, dict):
        raise ValueError("route-remediation manifest config is missing")
    spec_payload = identity.get("spec")
    if not isinstance(spec_payload, dict):
        raise ValueError("route-remediation manifest spec is missing")
    config = ResolvedPilotConfig(
        output_dir=output_dir,
        source_manifest_path=Path(str(manifest.payload["source_manifest_path"])),
        code_sha256=str(identity["code_sha256"]),
        source_run_id=str(identity["source_run_id"]),
        source_manifest_sha256=str(identity["source_manifest_sha256"]),
        source_summary_sha256=str(identity["source_summary_sha256"]),
        source_candidate_token_audit_sha256=str(identity["source_candidate_token_audit_sha256"]),
        source_raw_results_sha256=str(identity["source_raw_results_sha256"]),
        spec=PilotSpec.from_dict(spec_payload),
        spec_sha256=str(identity["spec_sha256"]),
        train_data_sha256=str(identity["train_data_sha256"]),
        dev_data_sha256=str(identity["dev_data_sha256"]),
        data_audit_sha256=str(identity["data_audit_sha256"]),
        dev_token_audit_sha256=str(identity["dev_token_audit_sha256"]),
        forced_choice_token_audit_sha256=str(identity["forced_choice_token_audit_sha256"]),
        crd_bank_audit_sha256=str(identity["crd_bank_audit_sha256"]),
        crd_token_audit_sha256=str(identity["crd_token_audit_sha256"]),
        source_evaluation_precision=str(identity["source_evaluation_precision"]),
    )
    if identity != config.identity_dict():
        raise ValueError("route-remediation preregistration identity changed")
    _validate_preflight_files(config)
    return config


def load_scale_canary_source_rows(
    source: FrozenSource,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    root = source.forced_choice_dir / "results" / "raw" / "scale_canary"
    files = sorted(root.glob("*.jsonl"))
    if len(files) != 24:
        raise ValueError(f"expected 24 scale-canary source files, found {len(files)}")
    for path in files:
        payload = path.read_bytes()
        file_sha256 = sha256(payload).hexdigest()
        rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        if len(rows) != 48:
            raise ValueError(f"scale-canary source row count changed: {path}")
        for row in rows:
            key = (
                str(row["lesson_id"]),
                str(row["arm"]),
                str(row["probe_id"]),
            )
            if key in result or row.get("model_id") != "scale_canary":
                raise ValueError(f"duplicate or invalid scale-canary source row: {key}")
            result[key] = {
                "row": row,
                "source_result_path": str(path),
                "source_raw_file_sha256": file_sha256,
                "source_row_sha256": json_hash(row),
            }
    if len(result) != 1_152:
        raise ValueError("scale-canary source matrix is incomplete")
    return result


def source_evaluation_precision(source: FrozenSource) -> str:
    units = source.forced_choice_manifest.get("units", {})
    values = {
        str(unit["evaluation_precision"])
        for key, unit in units.items()
        if key.startswith("scale_canary::")
    }
    if len(values) != 1:
        raise ValueError("scale-canary source precision is inconsistent")
    return next(iter(values))


def build_forced_choice_token_audit(
    *,
    tokenizer: Any,
    source: FrozenSource,
    source_rows: dict[tuple[str, str, str], dict[str, Any]],
    hard_bank: dict[str, tuple[P0CProbe, ...]],
    context_sha256: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    by_id = {item.lesson.lesson_id: item for item in source.selected}
    for lesson_id in source.forced_choice_manifest["selected_lessons"]:
        item = by_id[str(lesson_id)]
        for arm_value in CALIBRATION_ARMS:
            for probe in hard_bank[str(lesson_id)]:
                source_record = source_rows[(str(lesson_id), arm_value, probe.probe_id)]
                source_row = source_record["row"]
                rendered, variant, renderer = render_calibration_probe(
                    probe,
                    arm=Arm(arm_value),
                    external_note=item.external_note,
                )
                formatted = chat_prompt(tokenizer, rendered.prompt)
                token_audit = audit_fc_tokenization(
                    tokenizer,
                    formatted,
                    probe.action_choices,
                    evaluation_max_length=EVALUATION_MAX_LENGTH,
                )
                source_matches = (
                    source_row["prompt_sha256"] == token_audit["prompt_sha256"]
                    and source_row["expected_action"] == probe.expected_action
                    and source_row["ordered_allowed_actions"] == list(probe.action_choices)
                    and source_row["category"] == probe.category
                    and source_row["prompt_variant"] == variant
                    and source_row["prompt_rendering_version"] == renderer
                )
                record = {
                    "lesson_id": str(lesson_id),
                    "arm": arm_value,
                    "probe_id": probe.probe_id,
                    "category": probe.category,
                    "expected_action": probe.expected_action,
                    "prompt": rendered.prompt,
                    "formatted_prompt_sha256": token_audit["prompt_sha256"],
                    "prompt_variant": variant,
                    "prompt_rendering_version": renderer,
                    "source_row_sha256": source_record["source_row_sha256"],
                    "source_result_path": source_record["source_result_path"],
                    "source_raw_file_sha256": (source_record["source_raw_file_sha256"]),
                    "source_matches": source_matches,
                    **token_audit,
                }
                record["record_sha256"] = json_hash(record)
                records.append(record)
    failed = [
        record["probe_id"]
        for record in records
        if not record["source_matches"] or not record["all_candidates_valid"]
    ]
    return {
        "schema_version": "p0d2hrr-fc-token-audit-v1",
        "context_sha256": context_sha256,
        "decision_count": len(records),
        "candidate_count": sum(len(record["candidates"]) for record in records),
        "failed_decision_count": len(failed),
        "input_truncated_count": sum(
            not bool(record["all_candidates_valid"]) for record in records
        ),
        "failed_probe_ids": failed,
        "all_checks_passed": not failed and len(records) == 1_152,
        "records": records,
    }


def build_dev_token_audit(
    *,
    tokenizer: Any,
    records: list[RouteTrainingRecord],
    context_sha256: str,
) -> dict[str, Any]:
    audits: list[dict[str, Any]] = []
    for record in records:
        ordered_candidates = ("slot_a", "slot_b") if record.task == "route_only" else tuple(ACTIONS)
        formatted = chat_prompt(tokenizer, record.prompt)
        token_audit = audit_crd_tokenization(
            tokenizer,
            formatted,
            ordered_candidates,
            evaluation_max_length=EVALUATION_MAX_LENGTH,
        )
        audit = {
            "example_id": record.example_id,
            "group_id": record.group_id,
            "task": record.task,
            "expected_candidate": record.expected_completion,
            **token_audit,
        }
        audit["record_sha256"] = json_hash(audit)
        audits.append(audit)
    failed = [record["example_id"] for record in audits if not record["all_candidates_valid"]]
    return {
        "schema_version": "p0d2hrr-dev-token-audit-v1",
        "context_sha256": context_sha256,
        "decision_count": len(audits),
        "candidate_count": sum(len(record["candidates"]) for record in audits),
        "failed_decision_count": len(failed),
        "input_truncated_count": len(failed),
        "failed_example_ids": failed,
        "all_checks_passed": not failed and len(audits) == 96,
        "records": audits,
    }


def build_crd_token_audit(
    *,
    tokenizer: Any,
    source: FrozenSource,
    bank: dict[str, tuple[DecompositionProbe, ...]],
    context_sha256: str,
    bank_audit_sha256: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for lesson_id in source.forced_choice_manifest["selected_lessons"]:
        for probe in bank[str(lesson_id)]:
            source_record = source.source_rows[probe.source_row_key]
            formatted = chat_prompt(tokenizer, probe.prompt)
            token_audit = audit_crd_tokenization(
                tokenizer,
                formatted,
                probe.ordered_candidates,
                evaluation_max_length=EVALUATION_MAX_LENGTH,
            )
            record = {
                "lesson_id": str(lesson_id),
                "probe_id": probe.probe_id,
                "endpoint": probe.endpoint,
                "expected_candidate": probe.expected_candidate,
                "prompt": probe.prompt,
                "formatted_prompt_sha256": token_audit["prompt_sha256"],
                "source_row_key": probe.source_row_key,
                "source_row_sha256": source_record["source_row_sha256"],
                **token_audit,
            }
            record["record_sha256"] = json_hash(record)
            records.append(record)
    failed = [record["probe_id"] for record in records if not record["all_candidates_valid"]]
    return {
        "schema_version": "p0d2hrr-crd-token-audit-v1",
        "context_sha256": context_sha256,
        "bank_audit_sha256": bank_audit_sha256,
        "decision_count": len(records),
        "candidate_count": sum(len(record["candidates"]) for record in records),
        "failed_decision_count": len(failed),
        "input_truncated_count": sum(
            not bool(record["all_candidates_valid"]) for record in records
        ),
        "failed_probe_ids": failed,
        "all_checks_passed": not failed and len(records) == 1_536,
        "records": records,
    }


def _validate_preflight_files(config: ResolvedPilotConfig) -> None:
    paths = {
        config.train_data_sha256: config.output_dir / "preflight" / "route_train.jsonl",
        config.dev_data_sha256: config.output_dir / "preflight" / "route_dev.jsonl",
        config.data_audit_sha256: (config.output_dir / "preflight" / "route_data_audit.json"),
        config.dev_token_audit_sha256: (config.output_dir / "preflight" / "dev_token_audit.json"),
        config.forced_choice_token_audit_sha256: (
            config.output_dir / "preflight" / "forced_choice_token_audit.json"
        ),
        config.crd_bank_audit_sha256: (config.output_dir / "preflight" / "crd_bank_audit.json"),
        config.crd_token_audit_sha256: (config.output_dir / "preflight" / "crd_token_audit.json"),
    }
    for expected, path in paths.items():
        if file_hash(path) != expected:
            raise ValueError(f"route-remediation preflight file changed: {path}")
    preregistration = read_json_object(
        config.output_dir / "preregistration.json",
        "preregistration",
    )
    if preregistration != config.identity_dict():
        raise ValueError("route-remediation preregistration changed")


def _frozen_lesson_strings(
    selected: tuple[CompiledLesson, ...],
) -> tuple[str, ...]:
    values: set[str] = set()
    for item in selected:
        values.update(
            {
                item.lesson.lesson_id,
                item.lesson.context_id,
                item.lesson.condition,
                item.external_note,
            }
        )
    return tuple(sorted(value for value in values if len(value) >= 4))


def _forced_choice_raw_hash(output_dir: Path) -> str:
    files = sorted((output_dir / "results" / "raw").rglob("*.jsonl"))
    if len(files) != 48:
        raise ValueError("frozen forced-choice raw tree is incomplete")
    digest = sha256()
    for path in files:
        digest.update(str(path.relative_to(output_dir)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _require_independent_output(source_dir: Path, output_dir: Path) -> None:
    source = source_dir.resolve()
    output = output_dir.resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("route-remediation output must be independent from the frozen source")
