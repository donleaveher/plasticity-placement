from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.domain import Arm
from plasticity_placement.p0c.modeling import (
    ModelBundle,
    activate_adapter,
    load_base_model,
    release_model,
)
from plasticity_placement.p0c.runtime import (
    current_environment_fingerprint,
    current_environment_snapshot,
)
from plasticity_placement.p0d2h.probes import load_hard_probe_bank
from plasticity_placement.p0d2hc.prompting import render_calibration_probe
from plasticity_placement.p0d2hcbr.analysis import analyze_factorial, classify_unit
from plasticity_placement.p0d2hcbr.config import unit_id
from plasticity_placement.p0d2hcbr.evaluation import (
    _score_heldout,
    _source_snapshot,
    _stage,
)
from plasticity_placement.p0d2hcbr.preflight import (
    _validate_rabx_source,
    load_config,
    load_heldout_audits,
    load_heldout_probes,
)
from plasticity_placement.p0d2hcbr.resume import validate_resumable_authorization
from plasticity_placement.p0d2hcbr.runtime import _adapter_dir
from plasticity_placement.p0d2hcrd.probes import compile_decomposition_bank
from plasticity_placement.p0d2hcrd.runtime import validate_frozen_source
from plasticity_placement.p0d2hcrd.scoring import (
    score_candidate_batch as score_crd_candidate_batch,
)
from plasticity_placement.p0d2hfc.scoring import (
    score_candidate_batch as score_fc_candidate_batch,
)
from plasticity_placement.p0d2hrab.analysis import analyze_binding, build_binding_records
from plasticity_placement.p0d2hrab.config import AuditSpec as RabAuditSpec
from plasticity_placement.p0d2hrab.runtime import (
    _load_binding_probes,
    _load_token_audits,
)
from plasticity_placement.p0d2hrab.runtime import _score_bank as score_rab_bank
from plasticity_placement.p0d2hrabx.config import AuditSpec as RabxAuditSpec
from plasticity_placement.p0d2hrabx.runtime import _validate_rabc_source
from plasticity_placement.p0d2hrr.io import (
    file_hash,
    immutable_json_write,
    immutable_write,
    json_hash,
    read_json_object,
)
from plasticity_placement.p0d2hrr.paired_audit import (
    _json_bytes,
    build_crd_pairs,
    build_forced_choice_pairs,
    build_paired_analysis,
)
from plasticity_placement.p0d2hrr.preflight import (
    load_resolved_config,
    load_scale_canary_source_rows,
)
from plasticity_placement.p0d2hrr.runtime import (
    _adapter_hash,
    _AdapterLoadConfig,
    _load_crd_audits,
    _load_fc_audits,
    _require_source_compatible_bundle,
    _source_lesson_ids,
    _verify_crd_bank_audit,
)
from plasticity_placement.p0d2hrr.same_runtime_audit import (
    ROW_SCHEMA_VERSION as GUARDRAIL_ROW_SCHEMA_VERSION,
)
from plasticity_placement.p0d2hrr.same_runtime_audit import (
    _decision_row as _guardrail_decision_row,
)
from plasticity_placement.p0d2hrr.same_runtime_audit import (
    compare_sentinel_scores,
    score_adapter_off_on,
)
from plasticity_placement.training.model_utils import chat_prompt

SCHEMA_VERSION = "p0d2hcbr-resumable-unit-evaluation-v1"
PLAN_SCHEMA_VERSION = "p0d2hcbr-resumable-shard-plan-v1"
SHARD_SCHEMA_VERSION = "p0d2hcbr-resumable-score-shard-v1"
CLAIM_SCHEMA_VERSION = "p0d2hcbr-resumable-evaluation-claim-v1"
STAGES = ("heldout", "historical_rab", "forced_choice", "crd")


@dataclass(frozen=True, slots=True)
class ResumableEvaluationRequest:
    output_dir: Path
    curriculum: str
    placement: str
    seed: int


@dataclass(slots=True)
class EvaluationInputs:
    heldout_probes: list[Any]
    heldout_audits: dict[str, dict[str, Any]]
    rab_probes: list[Any]
    rab_audits: dict[str, dict[str, Any]]
    fc_items: list[tuple[str, Any, Any]]
    fc_audits: dict[tuple[str, str, str], dict[str, Any]]
    source_rows: dict[tuple[str, str, str], dict[str, Any]]
    crd_probes: list[Any]
    crd_audits: dict[str, dict[str, Any]]
    crd_source_rows: dict[str, dict[str, Any]]
    rr_config: Any
    frozen: Any


def evaluate_unit_resumable(request: ResumableEvaluationRequest) -> Path:
    output_dir = request.output_dir.resolve()
    authorization = validate_resumable_authorization(output_dir)
    key = unit_id(request.curriculum, request.placement, request.seed)
    if key not in authorization["eligible_unit_ids"]:
        raise PermissionError(f"unit is outside resumable authorization: {key}")
    analysis_output = Path(authorization["analysis_root"]) / key
    manifest, spec, identity = load_config(output_dir)
    complete_claim = manifest.payload.get("evaluation_claims", {}).get(key)
    if isinstance(complete_claim, dict) and complete_claim.get("state") == "complete":
        summary = analysis_output / "summary.json"
        if complete_claim.get("analysis_output") != str(analysis_output) or not summary.is_file():
            raise ValueError(f"completed resumable claim differs: {key}")
        if file_hash(summary) != complete_claim.get("summary_sha256"):
            raise ValueError(f"completed resumable summary changed: {key}")
        _progress(key, "already complete; no inference required")
        return summary

    budget_audit = authorization["budget_audit"]
    entry = manifest.payload["training_units"][key]
    adapter_dir = _adapter_dir(output_dir, entry, key)
    adapter_sha256 = _adapter_hash(adapter_dir)
    if adapter_sha256 != entry["training"]["summary"]["adapter_sha256"]:
        raise ValueError(f"CBR adapter changed after training: {key}")
    source_now = _validate_rabx_source(
        Path(identity["rabx_output"]), Path(identity["source_code_revision_lock"])
    )
    if source_now["snapshot"] != identity["source_snapshot"]:
        raise ValueError("RABX/RABC/RAB source changed after CBR planning")
    source_snapshot = _source_snapshot(output_dir, identity, key)
    inputs = _load_inputs(output_dir, identity)
    plan = _build_plan(
        analysis_output=analysis_output,
        cbr_run_id=manifest.payload["run_id"],
        key=key,
        adapter_sha256=adapter_sha256,
        code_sha256=authorization["resumable_code_sha256"],
        budget_audit=budget_audit,
        source_snapshot=source_snapshot,
        shard_size=int(authorization["shard_size"]),
        inputs=inputs,
    )
    plan_path = analysis_output / "checkpoints" / "plan.json"
    plan_sha256 = immutable_json_write(
        plan_path, plan, "CBR resumable shard plan"
    )
    _record_plan(output_dir, key, plan, plan_sha256)
    completed, pending = _load_checkpoint_shards(analysis_output, plan)
    _reconcile_completed_shards(output_dir, key, analysis_output, plan, completed)
    _progress(
        key,
        f"validated shards: {len(completed)}/{len(plan['shards'])}; "
        f"pending: {len(pending)}",
    )
    if pending:
        _score_pending_shards(
            output_dir=output_dir,
            analysis_output=analysis_output,
            key=key,
            request=request,
            adapter_dir=adapter_dir,
            adapter_sha256=adapter_sha256,
            spec=spec,
            inputs=inputs,
            plan=plan,
            pending=pending,
        )
        completed, pending = _load_checkpoint_shards(analysis_output, plan)
        _reconcile_completed_shards(output_dir, key, analysis_output, plan, completed)
    if pending:
        raise RuntimeError(f"resumable shards remain incomplete: {len(pending)}")
    return _finalize(
        output_dir=output_dir,
        analysis_output=analysis_output,
        key=key,
        request=request,
        adapter_sha256=adapter_sha256,
        spec=spec,
        identity=identity,
        budget_audit=budget_audit,
        plan=plan,
        plan_sha256=plan_sha256,
        shards=completed,
    )


def _load_inputs(output_dir: Path, identity: dict[str, Any]) -> EvaluationInputs:
    rabc = _validate_rabc_source(Path(identity["rabc_output"]), RabxAuditSpec())
    context = rabc["rab"]["rsh"]["upstream"]["context"]
    rr_output = Path(context["cpr_identity"]["source_output"])
    rr_config = load_resolved_config(rr_output)
    frozen = validate_frozen_source(rr_config.source_manifest_path)
    source_rows = load_scale_canary_source_rows(frozen)
    hard_bank = load_hard_probe_bank(frozen.hard_probe_dir)
    crd_bank, crd_bank_audit = compile_decomposition_bank(frozen.selected, hard_bank)
    _verify_crd_bank_audit(
        crd_bank_audit,
        json.loads((rr_output / "preflight" / "crd_bank_audit.json").read_text()),
    )
    selected_by_id = {item.lesson.lesson_id: item for item in frozen.selected}
    fc_items = [
        (lesson_id, selected_by_id[lesson_id], probe)
        for lesson_id in _source_lesson_ids(selected_by_id)
        for probe in hard_bank[lesson_id]
        if probe.category == "conditional_route"
    ]
    if len(fc_items) != 96:
        raise ValueError("resumable forced-choice matrix must contain 96 prompts")
    crd_probes = [probe for probes in crd_bank.values() for probe in probes]
    if len(crd_probes) != 1_536:
        raise ValueError("resumable CRD matrix must contain 1,536 prompts")
    rab_output = Path(identity["rab_output"])
    return EvaluationInputs(
        heldout_probes=load_heldout_probes(output_dir),
        heldout_audits=load_heldout_audits(output_dir),
        rab_probes=_load_binding_probes(rab_output),
        rab_audits=_load_token_audits(rab_output),
        fc_items=fc_items,
        fc_audits=_load_fc_audits(rr_output, rr_config),
        source_rows=source_rows,
        crd_probes=crd_probes,
        crd_audits=_load_crd_audits(rr_output, rr_config),
        crd_source_rows=frozen.source_rows,
        rr_config=rr_config,
        frozen=frozen,
    )


def _build_plan(
    *,
    analysis_output: Path,
    cbr_run_id: str,
    key: str,
    adapter_sha256: str,
    code_sha256: str,
    budget_audit: dict[str, Any],
    source_snapshot: dict[str, Any],
    shard_size: int,
    inputs: EvaluationInputs,
) -> dict[str, Any]:
    item_ids = {
        "heldout": [probe.probe_id for probe in inputs.heldout_probes],
        "historical_rab": [probe.probe_id for probe in inputs.rab_probes],
        "forced_choice": [
            f"{lesson_id}::{probe.probe_id}" for lesson_id, _, probe in inputs.fc_items
        ],
        "crd": [probe.probe_id for probe in inputs.crd_probes],
    }
    expected = {
        "heldout": 1_536,
        "historical_rab": 192,
        "forced_choice": 96,
        "crd": 1_536,
    }
    observed = {stage: len(values) for stage, values in item_ids.items()}
    if observed != expected:
        raise ValueError(f"resumable evaluation matrix differs: {observed}")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "cbr_run_id": cbr_run_id,
        "unit_id": key,
        "adapter_sha256": adapter_sha256,
        "analysis_output": str(analysis_output),
        "resumable_code_sha256": code_sha256,
        "budget_audit_sha256": json_hash(budget_audit),
        "source_snapshot_sha256": json_hash(source_snapshot),
        "shard_size": shard_size,
        "pair_order": "adapter_off_then_adapter_on_per_prompt",
    }
    run_id = "p0d2hcbr-resume-" + json_hash(identity)[:10]
    shards: list[dict[str, Any]] = []
    for stage in STAGES:
        values = item_ids[stage]
        for start in range(0, len(values), shard_size):
            index = start // shard_size
            shard_id = f"{stage}-{index:04d}"
            shards.append(
                {
                    "shard_id": shard_id,
                    "stage": stage,
                    "start": start,
                    "stop": min(start + shard_size, len(values)),
                    "item_ids": values[start : start + shard_size],
                    "relative_path": f"checkpoints/shards/{shard_id}.json",
                }
            )
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "run_id": run_id,
        "identity": identity,
        "stage_order": list(STAGES),
        "stage_prompt_counts": expected,
        "prompt_count": sum(expected.values()),
        "decision_count": 2 * sum(expected.values()),
        "shard_count": len(shards),
        "shards": shards,
        "source_snapshot": source_snapshot,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }


def _record_plan(
    output_dir: Path, key: str, plan: dict[str, Any], plan_sha256: str
) -> None:
    manifest = load_config(output_dir)[0]
    resume = manifest.payload.get("resumable_evaluation")
    if not isinstance(resume, dict):
        raise PermissionError("resumable evaluation authorization disappeared")
    units = resume.setdefault("units", {})
    existing = units.get(key)
    expected = {
        "state": "checkpointing",
        "run_id": plan["run_id"],
        "analysis_output": plan["identity"]["analysis_output"],
        "plan_sha256": plan_sha256,
        "shard_count": plan["shard_count"],
        "completed_shards": {},
    }
    if isinstance(existing, dict):
        fixed = {name: expected[name] for name in expected if name != "completed_shards"}
        differences = {
            name: {"expected": value, "observed": existing.get(name)}
            for name, value in fixed.items()
            if existing.get(name) != value and existing.get("state") != "complete"
        }
        if differences:
            raise ValueError(f"resumable unit identity differs: {key}: {differences}")
        return
    units[key] = expected
    manifest.save()


def _load_checkpoint_shards(
    analysis_output: Path, plan: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for shard in plan["shards"]:
        path = analysis_output / shard["relative_path"]
        if not path.exists():
            pending.append(shard)
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"resumable shard path is unsafe: {path}")
        payload = read_json_object(path, f"CBR resumable shard {shard['shard_id']}")
        _validate_checkpoint_shard(payload, shard, plan)
        completed.append(payload)
    return completed, pending


def _validate_checkpoint_shard(
    payload: dict[str, Any], shard: dict[str, Any], plan: dict[str, Any]
) -> None:
    runtime = payload.get("runtime_segment", {})
    expected_items = shard["item_ids"]
    if (
        payload.get("schema_version") != SHARD_SCHEMA_VERSION
        or payload.get("run_id") != plan["run_id"]
        or payload.get("shard_id") != shard["shard_id"]
        or payload.get("stage") != shard["stage"]
        or payload.get("item_ids") != expected_items
        or payload.get("adapter_sha256") != plan["identity"]["adapter_sha256"]
        or payload.get("source_snapshot_sha256")
        != plan["identity"]["source_snapshot_sha256"]
        or payload.get("pair_order") != "adapter_off_then_adapter_on_per_prompt"
        or payload.get("prompt_count") != len(expected_items)
        or payload.get("decision_count") != 2 * len(expected_items)
        or runtime.get("single_process_for_shard") is not True
        or runtime.get("single_loaded_model_for_shard") is not True
        or runtime.get("object_identity_preserved") is not True
        or runtime.get("sentinel", {}).get("passed") is not True
        or payload.get("training_authorized") is not False
        or payload.get("mappings_per_adapter_authorized") is not False
    ):
        raise ValueError(f"resumable shard identity or integrity differs: {shard['shard_id']}")
    stage = shard["stage"]
    if stage in {"heldout", "historical_rab"}:
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != 2 * len(expected_items):
            raise ValueError(f"resumable shard row count differs: {shard['shard_id']}")
        observed = [rows[index]["probe_id"] for index in range(0, len(rows), 2)]
    else:
        off_rows = payload.get("off_rows")
        on_rows = payload.get("on_rows")
        if (
            not isinstance(off_rows, list)
            or not isinstance(on_rows, list)
            or len(off_rows) != len(expected_items)
            or len(on_rows) != len(expected_items)
        ):
            raise ValueError(f"resumable shard row count differs: {shard['shard_id']}")
        if stage == "forced_choice":
            observed = [
                f"{row['lesson_id']}::{row['probe_id']}" for row in off_rows
            ]
        else:
            observed = [row["probe_id"] for row in off_rows]
    if observed != expected_items:
        raise ValueError(f"resumable shard item order differs: {shard['shard_id']}")


def _score_pending_shards(
    *,
    output_dir: Path,
    analysis_output: Path,
    key: str,
    request: ResumableEvaluationRequest,
    adapter_dir: Path,
    adapter_sha256: str,
    spec: Any,
    inputs: EvaluationInputs,
    plan: dict[str, Any],
    pending: list[dict[str, Any]],
) -> None:
    load_value = _AdapterLoadConfig(
        model_name=spec.model_name,
        model_revision=spec.model_revision,
        use_4bit=spec.training.use_4bit,
        training_seeds=(request.seed,),
    )
    started_at = datetime.now(UTC).isoformat()
    segment_id = "p0d2hcbr-segment-" + json_hash(
        {
            "run_id": plan["run_id"],
            "started_at": started_at,
            "environment_fingerprint": current_environment_fingerprint(),
        }
    )[:10]
    environment = current_environment_snapshot()
    bundle: ModelBundle | None = None
    try:
        _progress(key, f"loading model for runtime segment {segment_id}")
        bundle = load_base_model(load_value)
        _require_source_compatible_bundle(bundle, inputs.rr_config)
        base_id = id(bundle.model)
        activate_adapter(bundle, adapter_dir, adapter_name=f"cbr_{key}")
        if not hasattr(bundle.model, "disable_adapter"):
            raise RuntimeError("installed PEFT version lacks disable_adapter()")
        bundle.model.requires_grad_(False)
        bundle.model.eval()
        peft_id = id(bundle.model)
        wrapped_id = id(bundle.model.get_base_model())
        if wrapped_id != base_id:
            raise RuntimeError("CBR adapter activation replaced the base-model object")
        sentinel_probe = inputs.heldout_probes[0]
        sentinel_prompt = chat_prompt(bundle.tokenizer, sentinel_probe.prompt)
        sentinel_audit = inputs.heldout_audits[sentinel_probe.probe_id]
        for number, shard in enumerate(pending, start=1):
            with bundle.model.disable_adapter():
                sentinel_before = score_crd_candidate_batch(
                    bundle, sentinel_prompt, sentinel_audit
                )
            _progress(
                key,
                f"scoring shard {shard['shard_id']} ({number}/{len(pending)})",
            )
            rows = _score_shard(bundle, inputs, plan["run_id"], adapter_sha256, shard)
            with bundle.model.disable_adapter():
                sentinel_after = score_crd_candidate_batch(
                    bundle, sentinel_prompt, sentinel_audit
                )
            sentinel = compare_sentinel_scores(
                sentinel_before,
                sentinel_after,
                tolerance=spec.gates.sentinel_score_tolerance,
            )
            runtime = {
                "segment_id": segment_id,
                "segment_started_at": started_at,
                "single_process_for_shard": True,
                "single_loaded_model_for_shard": True,
                "pair_order": "adapter_off_then_adapter_on_per_prompt",
                "object_identity_preserved": id(bundle.model) == peft_id
                and id(bundle.model.get_base_model()) == wrapped_id,
                "base_model_object_id": base_id,
                "peft_model_object_id": peft_id,
                "wrapped_base_object_id": wrapped_id,
                "model_revision": bundle.model_revision,
                "precision": bundle.precision,
                "adapter_sha256": adapter_sha256,
                "environment": environment,
                "sentinel": sentinel,
            }
            if not runtime["object_identity_preserved"] or not sentinel["passed"]:
                raise RuntimeError(f"CBR shard runtime integrity failed: {shard['shard_id']}")
            payload = {
                "schema_version": SHARD_SCHEMA_VERSION,
                "run_id": plan["run_id"],
                "shard_id": shard["shard_id"],
                "stage": shard["stage"],
                "item_ids": shard["item_ids"],
                "adapter_sha256": adapter_sha256,
                "source_snapshot_sha256": plan["identity"]["source_snapshot_sha256"],
                "pair_order": "adapter_off_then_adapter_on_per_prompt",
                "prompt_count": len(shard["item_ids"]),
                "decision_count": 2 * len(shard["item_ids"]),
                **rows,
                "runtime_segment": runtime,
                "training_authorized": False,
                "mappings_per_adapter_authorized": False,
            }
            path = analysis_output / shard["relative_path"]
            digest = immutable_json_write(path, payload, f"CBR shard {shard['shard_id']}")
            _record_completed_shard(output_dir, key, shard["shard_id"], path, digest)
            _progress(key, f"checkpoint saved: {shard['shard_id']}")
    finally:
        if bundle is not None:
            release_model(bundle)


def _score_shard(
    bundle: ModelBundle,
    inputs: EvaluationInputs,
    run_id: str,
    adapter_sha256: str,
    shard: dict[str, Any],
) -> dict[str, Any]:
    start, stop = int(shard["start"]), int(shard["stop"])
    if shard["stage"] == "heldout":
        return {
            "rows": _score_heldout(
                bundle,
                inputs.heldout_probes[start:stop],
                inputs.heldout_audits,
                run_id,
                adapter_sha256,
            )
        }
    if shard["stage"] == "historical_rab":
        return {
            "rows": score_rab_bank(
                bundle,
                inputs.rab_probes[start:stop],
                inputs.rab_audits,
                run_id,
                adapter_sha256,
            )
        }
    if shard["stage"] == "forced_choice":
        off_rows, on_rows = _score_forced_choice_items(
            bundle,
            inputs.fc_items[start:stop],
            inputs,
            run_id,
            adapter_sha256,
        )
        return {"off_rows": off_rows, "on_rows": on_rows}
    if shard["stage"] == "crd":
        off_rows, on_rows = _score_crd_items(
            bundle,
            inputs.crd_probes[start:stop],
            inputs,
            run_id,
            adapter_sha256,
        )
        return {"off_rows": off_rows, "on_rows": on_rows}
    raise AssertionError(shard["stage"])


def _score_forced_choice_items(
    bundle: ModelBundle,
    items: list[tuple[str, Any, Any]],
    inputs: EvaluationInputs,
    run_id: str,
    adapter_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    off_rows: list[dict[str, Any]] = []
    on_rows: list[dict[str, Any]] = []
    config = inputs.rr_config
    for lesson_id, item, probe in items:
        key = (lesson_id, Arm.EXTERNAL.value, probe.probe_id)
        audit = inputs.fc_audits[key]
        source_record = inputs.source_rows[key]
        rendered, variant, renderer = render_calibration_probe(
            probe,
            arm=Arm.EXTERNAL,
            external_note=item.external_note,
        )
        formatted = chat_prompt(bundle.tokenizer, rendered.prompt)
        off, on = score_adapter_off_on(bundle, formatted, audit, score_fc_candidate_batch)
        common = {
            "schema_version": GUARDRAIL_ROW_SCHEMA_VERSION,
            "run_id": run_id,
            "model_name": config.spec.model_name,
            "model_revision": config.spec.model_revision,
            "precision": bundle.precision,
            "lesson_id": lesson_id,
            "pair_id": probe.pair_id,
            "lesson_type": probe.lesson_type,
            "probe_id": probe.probe_id,
            "category": probe.category,
            "arm": Arm.EXTERNAL.value,
            "expected_action": probe.expected_action,
            "ordered_allowed_actions": list(probe.action_choices),
            "prompt_variant": variant,
            "prompt_rendering_version": renderer,
            "prompt_sha256": audit["prompt_sha256"],
            "candidate_audit_record_sha256": audit["record_sha256"],
            "frozen_source_row_sha256": source_record["source_row_sha256"],
        }
        off_row = _guardrail_decision_row(
            common, off, "predicted_action", "adapter_off"
        )
        on_row = _guardrail_decision_row(
            common, on, "predicted_action", "adapter_on"
        )
        off_row["correct"] = off_row["predicted_action"] == probe.expected_action
        on_row["correct"] = on_row["predicted_action"] == probe.expected_action
        off_row["adapter_sha256"] = None
        on_row["adapter_sha256"] = adapter_sha256
        on_row["source_row_sha256"] = json_hash(off_row)
        off_rows.append(off_row)
        on_rows.append(on_row)
    return off_rows, on_rows


def _score_crd_items(
    bundle: ModelBundle,
    probes: list[Any],
    inputs: EvaluationInputs,
    run_id: str,
    adapter_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    off_rows: list[dict[str, Any]] = []
    on_rows: list[dict[str, Any]] = []
    config = inputs.rr_config
    for probe in probes:
        audit = inputs.crd_audits[probe.probe_id]
        source = inputs.crd_source_rows[probe.source_row_key]
        formatted = chat_prompt(bundle.tokenizer, probe.prompt)
        off, on = score_adapter_off_on(bundle, formatted, audit, score_crd_candidate_batch)
        common = {
            "schema_version": GUARDRAIL_ROW_SCHEMA_VERSION,
            "run_id": run_id,
            "model_name": config.spec.model_name,
            "model_revision": config.spec.model_revision,
            "precision": bundle.precision,
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
            "source_probe_id": probe.source_probe_id,
            "source_row_key": probe.source_row_key,
            "source_row_sha256": source["source_row_sha256"],
            "source_fc_correct": bool(source["row"]["correct"]),
        }
        off_row = _guardrail_decision_row(
            common, off, "predicted_candidate", "adapter_off"
        )
        on_row = _guardrail_decision_row(
            common, on, "predicted_candidate", "adapter_on"
        )
        off_row["correct"] = off_row["predicted_candidate"] == probe.expected_candidate
        on_row["correct"] = on_row["predicted_candidate"] == probe.expected_candidate
        off_row["adapter_sha256"] = None
        on_row["adapter_sha256"] = adapter_sha256
        off_rows.append(off_row)
        on_rows.append(on_row)
    return off_rows, on_rows


def _record_completed_shard(
    output_dir: Path, key: str, shard_id: str, path: Path, digest: str
) -> None:
    manifest = load_config(output_dir)[0]
    unit = manifest.payload["resumable_evaluation"]["units"][key]
    completed = unit.setdefault("completed_shards", {})
    existing = completed.get(shard_id)
    value = {"path": str(path), "sha256": digest}
    if existing is not None and existing != value:
        raise ValueError(f"resumable shard checkpoint changed: {shard_id}")
    completed[shard_id] = value
    manifest.save()


def _reconcile_completed_shards(
    output_dir: Path,
    key: str,
    analysis_output: Path,
    plan: dict[str, Any],
    shards: list[dict[str, Any]],
) -> None:
    manifest = load_config(output_dir)[0]
    unit = manifest.payload["resumable_evaluation"]["units"][key]
    recorded = unit.setdefault("completed_shards", {})
    observed = {
        shard["shard_id"]: {
            "path": str(
                analysis_output
                / _plan_shard(plan, shard["shard_id"])["relative_path"]
            ),
            "sha256": file_hash(
                analysis_output
                / _plan_shard(plan, shard["shard_id"])["relative_path"]
            ),
        }
        for shard in shards
    }
    differences = {
        shard_id: {"expected": value, "observed": observed.get(shard_id)}
        for shard_id, value in recorded.items()
        if observed.get(shard_id) != value
    }
    if differences:
        raise ValueError(f"recorded resumable checkpoints changed: {differences}")
    additions = {key: value for key, value in observed.items() if key not in recorded}
    if additions:
        recorded.update(additions)
        manifest.save()


def _finalize(
    *,
    output_dir: Path,
    analysis_output: Path,
    key: str,
    request: ResumableEvaluationRequest,
    adapter_sha256: str,
    spec: Any,
    identity: dict[str, Any],
    budget_audit: dict[str, Any],
    plan: dict[str, Any],
    plan_sha256: str,
    shards: list[dict[str, Any]],
) -> Path:
    by_stage: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGES}
    for shard in shards:
        by_stage[shard["stage"]].append(shard)
    heldout_rows = [row for shard in by_stage["heldout"] for row in shard["rows"]]
    rab_rows = [row for shard in by_stage["historical_rab"] for row in shard["rows"]]
    off_fc = [row for shard in by_stage["forced_choice"] for row in shard["off_rows"]]
    on_fc = [row for shard in by_stage["forced_choice"] for row in shard["on_rows"]]
    off_crd = [row for shard in by_stage["crd"] for row in shard["off_rows"]]
    on_crd = [row for shard in by_stage["crd"] for row in shard["on_rows"]]
    expected_counts = (3_072, 384, 96, 96, 1_536, 1_536)
    observed_counts = tuple(
        len(rows) for rows in (heldout_rows, rab_rows, off_fc, on_fc, off_crd, on_crd)
    )
    if observed_counts != expected_counts:
        raise ValueError(f"assembled resumable row counts differ: {observed_counts}")
    heldout_analysis, heldout_pairs = analyze_factorial(heldout_rows)
    rab_analysis = analyze_binding(rab_rows, RabAuditSpec())
    rab_pairs, rab_integrity = build_binding_records(rab_rows, RabAuditSpec())
    if not rab_integrity:
        raise ValueError("historical RAB matrix failed reconstruction")
    fc_pairs = build_forced_choice_pairs(off_fc, on_fc)
    crd_pairs = build_crd_pairs(off_crd, on_crd)
    paired_analysis, failures = build_paired_analysis(
        fc_pairs,
        crd_pairs,
        off_crd,
        on_crd,
        bootstrap_samples=spec.gates.bootstrap_samples,
        bootstrap_seed=spec.gates.bootstrap_seed,
    )
    decision = classify_unit(
        heldout=heldout_analysis,
        rab=rab_analysis,
        paired=paired_analysis,
        guardrail_pairs=fc_pairs + crd_pairs,
        gates=spec.gates,
    )
    after = _source_snapshot(output_dir, identity, key)
    if after != plan["source_snapshot"]:
        raise RuntimeError("CBR source artifacts changed during resumable evaluation")
    segment_records = _runtime_segment_records(shards)
    runtime_identity = {
        "protocol": "immutable_prompt_shards_v1",
        "whole_unit_single_process": len(segment_records) == 1,
        "whole_unit_single_loaded_base_model": len(segment_records) == 1,
        "runtime_segment_count": len(segment_records),
        "single_loaded_model_per_segment": True,
        "pair_order": "adapter_off_then_adapter_on_per_prompt",
        "all_shards_object_identity_preserved": all(
            shard["runtime_segment"]["object_identity_preserved"] for shard in shards
        ),
        "all_shard_sentinels_passed": all(
            shard["runtime_segment"]["sentinel"]["passed"] for shard in shards
        ),
        "adapter_sha256": adapter_sha256,
        "segments": segment_records,
    }
    summary = {
        "schema_version": "p0d2hcbr-unit-evaluation-v1",
        "run_id": plan["run_id"],
        "cbr_run_id": plan["identity"]["cbr_run_id"],
        "unit": {
            "unit_id": key,
            "curriculum": request.curriculum,
            "placement": request.placement,
            "seed": request.seed,
            "adapter_sha256": adapter_sha256,
        },
        "runtime_identity": runtime_identity,
        "environment": {
            "protocol": "multi_segment_resumable",
            "segment_environments": [record["environment"] for record in segment_records],
        },
        "heldout_factorial": heldout_analysis,
        "historical_rab": rab_analysis,
        "preservation": paired_analysis,
        "decision": decision,
        "budget_audit": budget_audit,
        "placement_comparison_scope": budget_audit.get(
            "comparison_scope", "preregistered_parameter_matched"
        ),
        "checkpoint_provenance": {
            "plan_sha256": plan_sha256,
            "shard_count": len(shards),
            "shard_size": plan["identity"]["shard_size"],
            "runtime_segment_count": len(segment_records),
            "automatic_resume": True,
        },
        "source_snapshot_before": plan["source_snapshot"],
        "source_snapshot_after": after,
        "source_artifacts_modified": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    claim_hash = _claim_final(output_dir, key, plan, adapter_sha256)
    manifest = load_config(output_dir)[0]
    existing_claim = manifest.payload.get("evaluation_claims", {}).get(key)
    claimed = {
        "state": "claimed",
        "run_id": plan["run_id"],
        "analysis_output": str(analysis_output),
        "claim_sha256": claim_hash,
        "summary_sha256": None,
        "resume_protocol": "immutable_prompt_shards_v1",
    }
    if existing_claim is not None and existing_claim != claimed:
        raise ValueError(f"final resumable claim differs: {key}")
    manifest.payload["evaluation_claims"][key] = claimed
    manifest.save()
    paths, hashes = _stage(
        analysis_output,
        summary,
        heldout_rows,
        rab_rows,
        off_fc,
        on_fc,
        off_crd,
        on_crd,
        heldout_pairs,
        rab_pairs,
        fc_pairs,
        crd_pairs,
        failures,
    )
    checkpoint_artifacts = {
        "checkpoint_plan": {
            "path": str(analysis_output / "checkpoints" / "plan.json"),
            "sha256": plan_sha256,
        },
        **{
            f"checkpoint_shard_{shard['shard_id']}": {
                "path": str(
                    analysis_output
                    / _plan_shard(plan, shard["shard_id"])["relative_path"]
                ),
                "sha256": file_hash(
                    analysis_output / _plan_shard(plan, shard["shard_id"])["relative_path"]
                ),
            }
            for shard in shards
        },
    }
    audit_manifest = {
        "schema_version": "p0d2hcbr-unit-evaluation-manifest-v1",
        "run_id": plan["run_id"],
        "identity": plan["identity"],
        "source_snapshot_sha256": plan["identity"]["source_snapshot_sha256"],
        "artifacts": {
            **{
                name: {"path": str(paths[name]), "sha256": digest}
                for name, digest in hashes.items()
            },
            **checkpoint_artifacts,
        },
        "resume_protocol": "immutable_prompt_shards_v1",
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    manifest_hash = immutable_write(
        paths["manifest"],
        _json_bytes(audit_manifest),
        "CBR resumable unit evaluation manifest",
    )
    manifest = load_config(output_dir)[0]
    manifest.payload["evaluation_claims"][key].update(
        {
            "state": "complete",
            "summary_sha256": hashes["summary"],
            "evaluation_manifest_sha256": manifest_hash,
        }
    )
    resumable_unit = manifest.payload["resumable_evaluation"]["units"][key]
    resumable_unit["state"] = "complete"
    resumable_unit["summary_sha256"] = hashes["summary"]
    if len(manifest.payload["evaluation_claims"]) == 12 and all(
        value["state"] == "complete"
        for value in manifest.payload["evaluation_claims"].values()
    ):
        manifest.payload["state"] = "evaluated"
    manifest.save()
    _progress(key, f"complete: {paths['summary']}")
    return paths["summary"]


def _claim_final(
    output_dir: Path,
    key: str,
    plan: dict[str, Any],
    adapter_sha256: str,
) -> str:
    payload = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "run_id": plan["run_id"],
        "unit_id": key,
        "analysis_output": plan["identity"]["analysis_output"],
        "adapter_sha256": adapter_sha256,
        "plan_sha256": file_hash(
            Path(plan["identity"]["analysis_output"]) / "checkpoints" / "plan.json"
        ),
        "shard_count": plan["shard_count"],
        "allowed_locked_evaluations": 1,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    return immutable_json_write(
        output_dir / "claims" / f"evaluation-{key}.json",
        payload,
        f"CBR resumable evaluation claim {key}",
    )


def _runtime_segment_records(shards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for shard in shards:
        runtime = shard["runtime_segment"]
        segment_id = runtime["segment_id"]
        record = records.setdefault(
            segment_id,
            {
                "segment_id": segment_id,
                "segment_started_at": runtime["segment_started_at"],
                "model_revision": runtime["model_revision"],
                "precision": runtime["precision"],
                "environment": runtime["environment"],
                "shard_ids": [],
            },
        )
        record["shard_ids"].append(shard["shard_id"])
    return [records[key] for key in sorted(records)]


def _plan_shard(plan: dict[str, Any], shard_id: str) -> dict[str, Any]:
    return next(shard for shard in plan["shards"] if shard["shard_id"] == shard_id)


def _progress(key: str, message: str) -> None:
    print(f"[p0d2hcbr-resume:{key}] {message}", flush=True)
