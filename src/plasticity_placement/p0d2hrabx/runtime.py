from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.modeling import (
    ModelBundle,
    activate_adapter,
    load_base_model,
    release_model,
)
from plasticity_placement.p0c.runtime import current_code_hash, current_environment_snapshot
from plasticity_placement.p0d2hcrd.scoring import (
    audit_candidate_tokenization,
    score_candidate_batch,
)
from plasticity_placement.p0d2hrab.probes import BindingProbe
from plasticity_placement.p0d2hrabc.runtime import (
    verify_complete_result as verify_rabc_result,
)
from plasticity_placement.p0d2hrabd.runtime import _validate_rab_result
from plasticity_placement.p0d2hrabx.analysis import analyze_label_disentanglement
from plasticity_placement.p0d2hrabx.config import AUTHORIZATION_SCOPE, AuditSpec
from plasticity_placement.p0d2hrabx.probes import (
    LabelDisentanglementProbe,
    compile_label_disentanglement_bank,
)
from plasticity_placement.p0d2hrr.io import (
    atomic_json_write,
    file_hash,
    immutable_json_write,
    immutable_write,
    json_hash,
    read_json_object,
)
from plasticity_placement.p0d2hrr.preflight import load_resolved_config
from plasticity_placement.p0d2hrr.runtime import (
    _AdapterLoadConfig,
    _require_source_compatible_bundle,
)
from plasticity_placement.p0d2hrr.same_runtime_audit import (
    _tree_hash,
    compare_sentinel_scores,
    score_adapter_off_on,
)
from plasticity_placement.p0d2hrsh.runtime import spec_context_model, spec_context_revision
from plasticity_placement.training.model_utils import chat_prompt, load_tokenizer

SCHEMA_VERSION = "p0d2hrabx-audit-v1"
MANIFEST_SCHEMA_VERSION = "p0d2hrabx-manifest-v1"
AUTHORIZATION_SCHEMA_VERSION = "p0d2hrabx-authorization-v1"
ROW_SCHEMA_VERSION = "p0d2hrabx-decision-row-v1"


def plan_audit(*, output_dir: Path, rabc_output: Path) -> Path:
    output_dir = output_dir.resolve()
    rabc_output = rabc_output.resolve()
    _require_independent(rabc_output, output_dir)
    if output_dir.exists():
        raise FileExistsError(f"RAB label-disentanglement output already exists: {output_dir}")
    spec = AuditSpec()
    source = _validate_rabc_source(rabc_output, spec)
    rab_output = Path(source["rabc_identity"]["rab_output"])
    source_probes = _load_source_binding_probes(rab_output, spec)
    probes, bank_audit = compile_label_disentanglement_bank(source_probes, spec)
    context = source["rab"]["rsh"]["upstream"]["context"]
    tokenizer = load_tokenizer(spec_context_model(context), spec_context_revision(context))
    token_records: list[dict[str, Any]] = []
    for probe in probes:
        formatted = chat_prompt(tokenizer, probe.prompt)
        audit = audit_candidate_tokenization(
            tokenizer,
            formatted,
            probe.ordered_candidates,
            evaluation_max_length=512,
        )
        record = {"probe_id": probe.probe_id, **audit}
        record["record_sha256"] = json_hash(record)
        token_records.append(record)
    token_audit = {
        "schema_version": "p0d2hrabx-token-audit-v1",
        "decision_count": len(token_records),
        "failed_probe_ids": [
            row["probe_id"] for row in token_records if row["all_candidates_valid"] is not True
        ],
        "records": token_records,
    }
    token_audit["all_checks_passed"] = not token_audit["failed_probe_ids"]
    if token_audit["all_checks_passed"] is not True:
        raise ValueError("RAB label-disentanglement candidate-token audit failed")

    output_dir.mkdir(parents=True)
    probe_hash = immutable_write(
        output_dir / "preflight" / "label_disentanglement_probes.jsonl",
        _jsonl([probe.to_dict() for probe in probes]),
        "RAB label-disentanglement probes",
    )
    bank_hash = immutable_json_write(
        output_dir / "preflight" / "bank_audit.json",
        bank_audit,
        "RAB label-disentanglement bank audit",
    )
    token_hash = immutable_json_write(
        output_dir / "preflight" / "token_audit.json",
        token_audit,
        "RAB label-disentanglement token audit",
    )
    source_audit = {
        "schema_version": "p0d2hrabx-source-audit-v1",
        "rabc_run_id": source["rabc_summary"]["run_id"],
        "rabc_analysis_status": source["rabc_summary"]["analysis"]["analysis_status"],
        "rabc_attribution": source["rabc_summary"]["analysis"]["causal_attribution"],
        "rabc_summary_sha256": file_hash(rabc_output / "summary.json"),
        "rab_run_id": source["rab"]["summary"]["run_id"],
        "rab_summary_sha256": file_hash(rab_output / "summary.json"),
        "adapter_sha256": context["adapter_sha256"],
        "source_snapshot": source["snapshot"],
        "all_checks_passed": True,
    }
    source_hash = immutable_json_write(
        output_dir / "preflight" / "source_audit.json",
        source_audit,
        "RAB label-disentanglement source audit",
    )
    analysis_plan = {
        "schema_version": "p0d2hrabx-analysis-plan-v1",
        "crossing": (
            "orientation x receipt A/B x canonical/crossed codebook x display AB/BA x "
            "candidate rotations 0/1/2/3"
        ),
        "primary_factors": [
            "receipt_token_A_vs_B",
            "selected_slot_label_A_vs_B",
            "selected_slot_display_position_first_vs_second",
            "expected_candidate_position_0_vs_1_2_3",
        ],
        "primary_endpoint": "forced-choice correctness with tie-identified bounds",
        "secondary_endpoint": "selected-minus-counterfactual sum-logprob margin",
        "uncertainty": "pair_id cluster bootstrap",
        "interaction_orders": [2, 3, 4],
        "identifiability": (
            "receipt token and selected slot label are independently crossed by the explicit "
            "canonical/crossed receipt-to-slot codebook"
        ),
        "allows_training": False,
        "allows_historical_reclassification": False,
        "allows_mappings_per_adapter_scan": False,
    }
    analysis_plan["analysis_plan_sha256"] = json_hash(analysis_plan)
    analysis_plan_hash = immutable_json_write(
        output_dir / "preflight" / "analysis_plan.json",
        analysis_plan,
        "RAB label-disentanglement analysis plan",
    )
    identity = {
        "schema_version": "p0d2hrabx-preregistration-v1",
        "code_sha256": current_code_hash(),
        "spec": spec.to_dict(),
        "rabc_output": str(rabc_output),
        "rabc_run_id": source["rabc_summary"]["run_id"],
        "rabc_summary_sha256": file_hash(rabc_output / "summary.json"),
        "rab_output": str(rab_output),
        "rab_run_id": source["rab"]["summary"]["run_id"],
        "adapter_sha256": context["adapter_sha256"],
        "source_snapshot": source["snapshot"],
        "probe_sha256": probe_hash,
        "bank_audit_sha256": bank_hash,
        "token_audit_sha256": token_hash,
        "source_audit_sha256": source_hash,
        "analysis_plan_sha256": analysis_plan_hash,
        "historical_rab_decision_changed": False,
        "historical_rabc_attribution_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    identity["preregistration_sha256"] = json_hash(identity)
    immutable_json_write(
        output_dir / "preregistration.json",
        identity,
        "RAB label-disentanglement preregistration",
    )
    immutable_json_write(
        output_dir / "authorization.template.json",
        authorization_template(identity["preregistration_sha256"]),
        "RAB label-disentanglement authorization template",
    )
    now = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": "p0d2hrabx-" + identity["preregistration_sha256"][:10],
        "state": "planned",
        "created_at": now,
        "updated_at": now,
        "config": identity,
        "authorization": None,
        "errors": [],
        "historical_rab_decision_changed": False,
        "historical_rabc_attribution_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    atomic_json_write(output_dir / "manifest.json", manifest)
    return output_dir / "manifest.json"


def authorization_template(preregistration_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "decision": "pending",
        "scope": AUTHORIZATION_SCOPE,
        "preregistration_sha256": preregistration_sha256,
        "approved_by": "TBD",
        "approved_at": "TBD",
        "allowed_audit_runs": 1,
        "allows_training": False,
        "allows_prompt_revision_after_results": False,
        "allows_historical_reclassification": False,
        "allows_mappings_per_adapter_scan": False,
    }


def authorize_audit(output_dir: Path, authorization_path: Path) -> Path:
    output_dir = output_dir.resolve()
    authorization_path = authorization_path.resolve()
    manifest = _load_manifest(output_dir)
    if manifest["state"] == "authorized":
        adopted_path = output_dir / "authorization.json"
        external = read_json_object(authorization_path, "external label authorization")
        adopted = read_json_object(adopted_path, "adopted label authorization")
        if external != adopted or manifest.get("authorization", {}).get("sha256") != file_hash(
            adopted_path
        ):
            raise PermissionError("existing RAB label-disentanglement authorization differs")
        return output_dir / "manifest.json"
    if manifest["state"] != "planned":
        raise PermissionError("RAB label-disentanglement authorization requires planned state")
    if authorization_path == output_dir or authorization_path.is_relative_to(output_dir):
        raise ValueError("RAB label-disentanglement authorization must be external")
    payload = read_json_object(authorization_path, "RAB label-disentanglement authorization")
    expected = authorization_template(manifest["config"]["preregistration_sha256"])
    required = {
        **expected,
        "decision": "approved",
        "approved_by": payload.get("approved_by"),
        "approved_at": payload.get("approved_at"),
    }
    differences = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in required.items()
        if payload.get(key) != value
    }
    if differences:
        raise PermissionError(f"RAB label-disentanglement authorization differs: {differences}")
    if payload.get("approved_by") in {None, "", "TBD"} or payload.get("approved_at") == "TBD":
        raise ValueError("RAB label-disentanglement authorization requires approver and timestamp")
    approved_at = datetime.fromisoformat(str(payload["approved_at"]).replace("Z", "+00:00"))
    if approved_at.tzinfo is None:
        raise ValueError("RAB label-disentanglement approval timestamp must be timezone-aware")
    digest = immutable_json_write(
        output_dir / "authorization.json",
        payload,
        "RAB label-disentanglement authorization",
    )
    manifest["state"] = "authorized"
    manifest["authorization"] = {"sha256": digest, "approved_by": payload["approved_by"]}
    _save_manifest(output_dir, manifest)
    return output_dir / "manifest.json"


def run_audit(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    manifest = _load_manifest(output_dir)
    if manifest["state"] != "authorized":
        raise PermissionError("RAB label-disentanglement run requires authorized state")
    identity = manifest["config"]
    if current_code_hash() != identity["code_sha256"]:
        raise ValueError("RAB label-disentanglement code changed after preregistration")
    _verify_preflight(output_dir, identity)
    _verify_authorization(output_dir, manifest)
    spec = AuditSpec()
    rabc_output = Path(identity["rabc_output"])
    source = _validate_rabc_source(rabc_output, spec)
    if source["snapshot"] != identity["source_snapshot"]:
        raise ValueError("RABC/RAB source artifacts changed after label planning")
    probes = _load_label_probes(output_dir, spec)
    audits = _load_token_audits(output_dir, spec)
    upstream = source["rab"]["rsh"]["upstream"]
    context = upstream["context"]
    cpr_output = Path(upstream["rtb_identity"]["cpr_output"])
    rr_config = load_resolved_config(Path(context["cpr_identity"]["source_output"]))
    load_config = _AdapterLoadConfig(
        model_name=spec_context_model(context),
        model_revision=spec_context_revision(context),
        use_4bit=context["cpr_spec"].training.use_4bit,
        training_seeds=(context["cpr_spec"].training.seed,),
    )
    manifest["state"] = "running"
    _save_manifest(output_dir, manifest)
    bundle: ModelBundle | None = None
    try:
        _progress("loading one frozen base-model runtime")
        bundle = load_base_model(load_config)
        _require_source_compatible_bundle(bundle, rr_config)
        base_id = id(bundle.model)
        activate_adapter(bundle, cpr_output / "adapter", adapter_name="rab_label_disentanglement")
        if not hasattr(bundle.model, "disable_adapter"):
            raise RuntimeError("installed PEFT version lacks disable_adapter()")
        bundle.model.requires_grad_(False)
        bundle.model.eval()
        wrapped_id = id(bundle.model.get_base_model())
        if wrapped_id != base_id:
            raise RuntimeError("label adapter activation replaced the base-model object")
        peft_id = id(bundle.model)
        first = probes[0]
        first_prompt = chat_prompt(bundle.tokenizer, first.prompt)
        sentinel_before = _score_off(bundle, first_prompt, audits[first.probe_id])
        raw_rows = _score_bank(
            bundle, probes, audits, manifest["run_id"], context["adapter_sha256"]
        )
        sentinel_after = _score_off(bundle, first_prompt, audits[first.probe_id])
        sentinel = compare_sentinel_scores(
            sentinel_before, sentinel_after, tolerance=spec.sentinel_score_tolerance
        )
        runtime_identity = {
            "single_process": True,
            "single_loaded_base_model": True,
            "pair_order": "adapter_off_then_adapter_on_per_frozen_prompt",
            "object_identity_preserved": id(bundle.model) == peft_id
            and id(bundle.model.get_base_model()) == wrapped_id,
            "base_model_object_id": base_id,
            "peft_model_object_id": peft_id,
            "wrapped_base_object_id": wrapped_id,
            "model_revision": bundle.model_revision,
            "precision": bundle.precision,
            "adapter_sha256": context["adapter_sha256"],
            "sentinel": sentinel,
        }
        if not runtime_identity["object_identity_preserved"] or not sentinel["passed"]:
            raise RuntimeError("RAB label-disentanglement same-runtime integrity check failed")
    except BaseException as error:
        _mark_failed(output_dir, error)
        raise
    finally:
        if bundle is not None:
            release_model(bundle)

    try:
        raw_hashes = _publish_raw(output_dir, manifest["run_id"], raw_rows)
        analysis, paired_records = analyze_label_disentanglement(raw_rows, spec)
        after = _validate_rabc_source(rabc_output, spec)["snapshot"]
        if after != identity["source_snapshot"]:
            raise RuntimeError("RABC/RAB source artifacts changed during label audit")
        summary = {
            "schema_version": SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "analysis_status": analysis["analysis_status"],
            "source": {
                "rabc_run_id": source["rabc_summary"]["run_id"],
                "rabc_summary_sha256": file_hash(rabc_output / "summary.json"),
                "rab_run_id": source["rab"]["summary"]["run_id"],
                "rab_summary_sha256": file_hash(Path(identity["rab_output"]) / "summary.json"),
                "adapter_sha256": context["adapter_sha256"],
            },
            "runtime_identity": runtime_identity,
            "raw_score_checkpoint": raw_hashes,
            "environment": current_environment_snapshot(),
            "analysis": analysis,
            "source_snapshot_before": identity["source_snapshot"],
            "source_snapshot_after": after,
            "source_artifacts_modified": False,
            "historical_rab_decision_changed": False,
            "historical_rabc_attribution_changed": False,
            "training_authorized": False,
            "mappings_per_adapter_authorized": False,
            "interpretation_boundary": (
                "This frozen-adapter inference audit separates receipt-token and selected-slot-"
                "label effects under an explicit codebook while retaining display/candidate "
                "controls. It cannot reclassify prior results or authorize training/1/4/8."
            ),
        }
        artifact_hashes = {**raw_hashes, **_publish_analysis(output_dir, summary, paired_records)}
        audit_manifest = {
            "schema_version": "p0d2hrabx-audit-manifest-v1",
            "run_id": manifest["run_id"],
            "preregistration_sha256": identity["preregistration_sha256"],
            "source_snapshot_sha256": json_hash(after),
            "artifacts": artifact_hashes,
            "historical_rab_decision_changed": False,
            "historical_rabc_attribution_changed": False,
            "training_authorized": False,
            "mappings_per_adapter_authorized": False,
        }
        audit_manifest_sha256 = immutable_json_write(
            output_dir / "audit_manifest.json",
            audit_manifest,
            "RAB label-disentanglement audit manifest",
        )
        manifest = _load_manifest(output_dir)
        manifest["state"] = "complete"
        manifest["result"] = {
            "summary_sha256": artifact_hashes["summary"],
            "audit_manifest_sha256": audit_manifest_sha256,
            "analysis_status": analysis["analysis_status"],
            "causal_attribution": analysis["causal_attribution"]["status"],
        }
        _save_manifest(output_dir, manifest)
    except BaseException as error:
        _mark_failed(output_dir, error)
        raise
    _progress(f"complete: {output_dir / 'summary.json'}")
    return output_dir / "summary.json"


def verify_complete_result(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    manifest = _load_manifest(output_dir)
    if manifest.get("state") != "complete":
        raise ValueError("RAB label-disentanglement manifest is not complete")
    _verify_preflight(output_dir, manifest["config"])
    audit_path = output_dir / "audit_manifest.json"
    if not audit_path.is_file() or file_hash(audit_path) != manifest.get("result", {}).get(
        "audit_manifest_sha256"
    ):
        raise ValueError("RAB label-disentanglement audit manifest is missing or changed")
    audit = read_json_object(audit_path, "RAB label-disentanglement audit manifest")
    required = _published_artifact_paths(output_dir)
    differences = {
        name: {
            "expected": audit.get("artifacts", {}).get(name),
            "observed": file_hash(path) if path.is_file() else None,
        }
        for name, path in required.items()
        if (file_hash(path) if path.is_file() else None) != audit.get("artifacts", {}).get(name)
    }
    if differences:
        raise ValueError(f"RAB label-disentanglement artifacts changed: {differences}")
    if (
        audit.get("schema_version") != "p0d2hrabx-audit-manifest-v1"
        or audit.get("run_id") != manifest.get("run_id")
        or audit.get("preregistration_sha256")
        != manifest.get("config", {}).get("preregistration_sha256")
        or audit.get("training_authorized") is not False
        or audit.get("mappings_per_adapter_authorized") is not False
        or manifest.get("result", {}).get("summary_sha256") != audit["artifacts"]["summary"]
    ):
        raise ValueError("RAB label-disentanglement complete manifests disagree")
    checkpoint = read_json_object(
        required["raw_score_checkpoint"], "RAB label-disentanglement raw checkpoint"
    )
    off_count = _jsonl_row_count(required["adapter_off"])
    on_count = _jsonl_row_count(required["adapter_on"])
    if (
        checkpoint.get("schema_version") != "p0d2hrabx-raw-score-checkpoint-v1"
        or checkpoint.get("run_id") != manifest.get("run_id")
        or checkpoint.get("adapter_off_count") != off_count
        or checkpoint.get("adapter_on_count") != on_count
        or off_count != AuditSpec().prompt_count
        or on_count != AuditSpec().prompt_count
        or checkpoint.get("artifacts")
        != {
            "adapter_off": audit["artifacts"]["adapter_off"],
            "adapter_on": audit["artifacts"]["adapter_on"],
        }
    ):
        raise ValueError("RAB label-disentanglement raw checkpoint is inconsistent")
    summary = read_json_object(required["summary"], "RAB label-disentanglement summary")
    if (
        summary.get("run_id") != manifest.get("run_id")
        or summary.get("analysis", {}).get("analysis_status")
        != manifest.get("result", {}).get("analysis_status")
        or summary.get("analysis", {}).get("causal_attribution", {}).get("status")
        != manifest.get("result", {}).get("causal_attribution")
        or summary.get("historical_rab_decision_changed") is not False
        or summary.get("historical_rabc_attribution_changed") is not False
        or summary.get("training_authorized") is not False
        or summary.get("mappings_per_adapter_authorized") is not False
        or _jsonl_row_count(required["paired_records"]) != AuditSpec().prompt_count
    ):
        raise ValueError("RAB label-disentanglement summary or record counts are inconsistent")
    return required["summary"]


def _validate_rabc_source(rabc_output: Path, spec: AuditSpec) -> dict[str, Any]:
    verify_rabc_result(rabc_output)
    manifest = read_json_object(rabc_output / "manifest.json", "RABC manifest")
    identity = read_json_object(rabc_output / "preregistration.json", "RABC preregistration")
    summary = read_json_object(rabc_output / "summary.json", "RABC summary")
    flags = summary.get("analysis", {}).get("causal_attribution", {}).get("flags", {})
    expected_flags = {
        "candidate_position_0": True,
        "composition_interaction": True,
        "receipt_or_slot_label": True,
        "serial_position": True,
    }
    if (
        manifest.get("config") != identity
        or summary.get("run_id") != spec.expected_rabc_run_id
        or summary.get("analysis", {}).get("analysis_status") != "counterbalancing_complete"
        or summary.get("analysis", {}).get("causal_attribution", {}).get("status")
        != "multiple_mechanisms"
        or flags != expected_flags
        or summary.get("historical_rab_decision_changed") is not False
        or summary.get("training_authorized") is not False
        or summary.get("mappings_per_adapter_authorized") is not False
    ):
        raise ValueError("source is not the frozen completed RABC result")
    rab_output = Path(identity["rab_output"])
    rab = _validate_rab_result(rab_output)
    if (
        summary.get("source", {}).get("rab_run_id") != rab["summary"]["run_id"]
        or summary.get("source", {}).get("rab_summary_sha256")
        != file_hash(rab_output / "summary.json")
        or summary.get("source_snapshot_after") != rab["snapshot"]
    ):
        raise ValueError("RABC transitive RAB source changed")
    return {
        "rabc_manifest": manifest,
        "rabc_identity": identity,
        "rabc_summary": summary,
        "rab": rab,
        "snapshot": {
            "rabc_output_tree": _tree_hash(rabc_output),
            "rab_source_snapshot": rab["snapshot"],
        },
    }


def _score_bank(
    bundle: ModelBundle,
    probes: list[LabelDisentanglementProbe],
    audits: dict[str, dict[str, Any]],
    run_id: str,
    adapter_sha256: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for probe in probes:
        formatted = chat_prompt(bundle.tokenizer, probe.prompt)
        off, on = score_adapter_off_on(
            bundle, formatted, audits[probe.probe_id], score_candidate_batch
        )
        common = {
            "schema_version": ROW_SCHEMA_VERSION,
            "run_id": run_id,
            **{
                key: value
                for key, value in probe.to_dict().items()
                if key not in {"prompt", "source_prompt_sha256"}
            },
            "ordered_candidates": list(probe.ordered_candidates),
            "formatted_prompt_sha256": audits[probe.probe_id]["prompt_sha256"],
        }
        rows.append(_decision_row(common, off, "adapter_off", None))
        rows.append(_decision_row(common, on, "adapter_on", adapter_sha256))
        if len(rows) % 128 == 0:
            _progress(f"label decisions complete: {len(rows)}/{len(probes) * 2}")
    return rows


def _decision_row(
    common: dict[str, Any],
    outcome: dict[str, Any],
    state: str,
    adapter_sha256: str | None,
) -> dict[str, Any]:
    return {
        **common,
        "adapter_state": state,
        "adapter_sha256": adapter_sha256,
        "candidates": outcome["candidates"],
        "predicted_candidate": outcome["predicted_candidate"],
        "mean_predicted_candidate": outcome["mean_predicted_candidate"],
        "top1_top2_margin": outcome["top1_top2_margin"],
        "sum_mean_disagreement": outcome["sum_mean_disagreement"],
        "tie": outcome["tie"],
        "mean_score_tie": outcome["mean_score_tie"],
        "non_finite": outcome["non_finite"],
        "error_status": outcome["error_status"],
        "error_message": outcome["error_message"],
        "latency_seconds": outcome["latency_seconds"],
    }


def _score_off(bundle: ModelBundle, prompt: str, audit: dict[str, Any]) -> dict[str, Any]:
    with bundle.model.disable_adapter():
        return score_candidate_batch(bundle, prompt, audit)


def _publish_raw(output: Path, run_id: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    off = [row for row in rows if row["adapter_state"] == "adapter_off"]
    on = [row for row in rows if row["adapter_state"] == "adapter_on"]
    hashes = {
        "adapter_off": immutable_write(
            output / "results" / "adapter_off.jsonl", _jsonl(off), "label-disentanglement OFF rows"
        ),
        "adapter_on": immutable_write(
            output / "results" / "adapter_on.jsonl", _jsonl(on), "label-disentanglement ON rows"
        ),
    }
    checkpoint = {
        "schema_version": "p0d2hrabx-raw-score-checkpoint-v1",
        "run_id": run_id,
        "adapter_off_count": len(off),
        "adapter_on_count": len(on),
        "artifacts": dict(hashes),
        "checkpoint_stage": "inference_complete_before_analysis",
        "automatic_analysis_recovery_allowed": False,
    }
    hashes["raw_score_checkpoint"] = immutable_json_write(
        output / "raw_score_checkpoint.json",
        checkpoint,
        "label-disentanglement raw checkpoint",
    )
    return hashes


def _publish_analysis(
    output: Path, summary: dict[str, Any], paired_records: list[dict[str, Any]]
) -> dict[str, str]:
    return {
        "summary": immutable_json_write(
            output / "summary.json", summary, "RAB label-disentanglement summary"
        ),
        "report": immutable_write(
            output / "report.md",
            _render_report(summary).encode(),
            "RAB label-disentanglement report",
        ),
        "paired_records": immutable_write(
            output / "paired_records.jsonl",
            _jsonl(paired_records),
            "RAB label-disentanglement paired records",
        ),
    }


def _render_report(summary: dict[str, Any]) -> str:
    analysis = summary["analysis"]
    lines = [
        "# Frozen RAB receipt/slot-label disentanglement audit",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Status: `{analysis['analysis_status']}`",
        f"- Attribution: `{analysis['causal_attribution']['status']}`",
        "- Historical RAB decision changed: `false`",
        "- Historical RABC attribution changed: `false`",
        "- Training authorized: `false`",
        "- 1/4/8 authorized: `false`",
        "",
    ]
    if analysis["analysis_status"] != "label_disentanglement_complete":
        lines.extend(["Scoring integrity failed; factor effects are not interpreted.", ""])
        return "\n".join(lines)
    lines.extend(
        [
            "## Adapter factor contrasts",
            "",
            "Negative values mean the named level is disadvantaged.",
            "",
            "| Contrast | Accuracy midpoint (95% pair-cluster CI) | Identified interval | Margin |",
            "|---|---:|---:|---:|",
        ]
    )
    labels = {
        "receipt_B_minus_A": "Receipt token B − A",
        "selected_slot_B_minus_A": "Selected Slot B − A",
        "selected_display_second_minus_first": "Selected slot second − first",
        "candidate_position_0_minus_1_2_3": "Candidate position 0 − positions 1/2/3",
    }
    for key, label in labels.items():
        values = analysis["main_effects"][key]["adapter"]
        accuracy = values["accuracy_midpoint"]
        interval = values["accuracy_identified_interval"]
        margin = values["selected_minus_counterfactual_margin"]
        lines.append(
            f"| {label} | {accuracy['estimate']:.4f} "
            f"[{accuracy['ci95'][0]:.4f}, {accuracy['ci95'][1]:.4f}] | "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] | {margin['estimate']:.4f} "
            f"[{margin['ci95'][0]:.4f}, {margin['ci95'][1]:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Attribution flags",
            "",
            *[
                f"- {name}: `{str(value).lower()}`"
                for name, value in analysis["causal_attribution"]["flags"].items()
            ],
            "",
            analysis["causal_attribution"]["identifiability_note"],
            "",
            "This audit does not authorize remediation training or the 1/4/8 experiment.",
            "",
        ]
    )
    return "\n".join(lines)


def _verify_preflight(output: Path, identity: dict[str, Any]) -> None:
    files = {
        "probe_sha256": output / "preflight" / "label_disentanglement_probes.jsonl",
        "bank_audit_sha256": output / "preflight" / "bank_audit.json",
        "token_audit_sha256": output / "preflight" / "token_audit.json",
        "source_audit_sha256": output / "preflight" / "source_audit.json",
        "analysis_plan_sha256": output / "preflight" / "analysis_plan.json",
    }
    for field, path in files.items():
        if file_hash(path) != identity[field]:
            raise ValueError(f"RAB label-disentanglement preflight changed: {field}")
    prereg = read_json_object(
        output / "preregistration.json", "RAB label-disentanglement preregistration"
    )
    unhashed = {key: value for key, value in identity.items() if key != "preregistration_sha256"}
    if prereg != identity or json_hash(unhashed) != identity["preregistration_sha256"]:
        raise ValueError("RAB label-disentanglement preregistration changed")
    for name in ("bank_audit", "token_audit", "source_audit"):
        payload = read_json_object(output / "preflight" / f"{name}.json", name)
        if payload.get("all_checks_passed") is not True:
            raise ValueError(f"RAB label-disentanglement preflight failed: {name}")
    plan = read_json_object(output / "preflight" / "analysis_plan.json", "analysis plan")
    plan_unhashed = {key: value for key, value in plan.items() if key != "analysis_plan_sha256"}
    if json_hash(plan_unhashed) != plan.get("analysis_plan_sha256"):
        raise ValueError("RAB label-disentanglement analysis plan self-hash changed")


def _verify_authorization(output: Path, manifest: dict[str, Any]) -> None:
    path = output / "authorization.json"
    authorization = read_json_object(path, "adopted RAB label authorization")
    if (
        manifest.get("authorization", {}).get("sha256") != file_hash(path)
        or authorization.get("decision") != "approved"
        or authorization.get("preregistration_sha256")
        != manifest["config"]["preregistration_sha256"]
        or authorization.get("allows_training") is not False
        or authorization.get("allows_historical_reclassification") is not False
        or authorization.get("allows_mappings_per_adapter_scan") is not False
    ):
        raise ValueError("RAB label-disentanglement authorization changed after adoption")


def _load_source_binding_probes(rab_output: Path, spec: AuditSpec) -> list[BindingProbe]:
    rows = _read_jsonl(rab_output / "preflight" / "binding_probes.jsonl")
    probes = [
        BindingProbe(**{**row, "ordered_candidates": tuple(row["ordered_candidates"])})
        for row in rows
    ]
    if len(probes) != spec.source_prompt_count:
        raise ValueError("completed RAB source probe bank is incomplete")
    return probes


def _load_label_probes(output: Path, spec: AuditSpec) -> list[LabelDisentanglementProbe]:
    rows = _read_jsonl(output / "preflight" / "label_disentanglement_probes.jsonl")
    probes = [
        LabelDisentanglementProbe(
            **{**row, "ordered_candidates": tuple(row["ordered_candidates"])}
        )
        for row in rows
    ]
    if len(probes) != spec.prompt_count:
        raise ValueError("RAB label-disentanglement probe bank is incomplete")
    return probes


def _load_token_audits(output: Path, spec: AuditSpec) -> dict[str, dict[str, Any]]:
    payload = read_json_object(output / "preflight" / "token_audit.json", "token audit")
    result = {str(row["probe_id"]): row for row in payload["records"]}
    if len(result) != spec.prompt_count or payload.get("all_checks_passed") is not True:
        raise ValueError("RAB label-disentanglement token audit is incomplete")
    return result


def _published_artifact_paths(output: Path) -> dict[str, Path]:
    return {
        "summary": output / "summary.json",
        "report": output / "report.md",
        "paired_records": output / "paired_records.jsonl",
        "adapter_off": output / "results" / "adapter_off.jsonl",
        "adapter_on": output / "results" / "adapter_on.jsonl",
        "raw_score_checkpoint": output / "raw_score_checkpoint.json",
    }


def _load_manifest(output: Path) -> dict[str, Any]:
    manifest = read_json_object(output / "manifest.json", "RAB label-disentanglement manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("not an RAB label-disentanglement manifest")
    return manifest


def _save_manifest(output: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    atomic_json_write(output / "manifest.json", manifest)


def _mark_failed(output: Path, error: BaseException) -> None:
    manifest = _load_manifest(output)
    manifest["state"] = "failed"
    manifest["errors"].append(
        {"type": type(error).__name__, "message": str(error), "time": datetime.now(UTC).isoformat()}
    )
    _save_manifest(output, manifest)


def _require_independent(source: Path, output: Path) -> None:
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("RAB label-disentanglement output must be independent of source")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    return (payload + "\n").encode()


def _jsonl_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def _progress(message: str) -> None:
    print(f"[p0d2hrabx] {message}", flush=True)
