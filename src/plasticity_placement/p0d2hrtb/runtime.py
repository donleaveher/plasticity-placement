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
from plasticity_placement.p0d2h.probes import load_hard_probe_bank
from plasticity_placement.p0d2hcpr.preflight import load_config as load_cpr_config
from plasticity_placement.p0d2hcrd.runtime import validate_frozen_source
from plasticity_placement.p0d2hcrd.scoring import (
    audit_candidate_tokenization,
    score_candidate_batch,
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
    _adapter_hash,
    _AdapterLoadConfig,
    _require_source_compatible_bundle,
)
from plasticity_placement.p0d2hrr.same_runtime_audit import (
    _tree_hash,
    compare_sentinel_scores,
    score_adapter_off_on,
)
from plasticity_placement.p0d2hrtb.analysis import analyze_bridge, build_pairs
from plasticity_placement.p0d2hrtb.config import AUTHORIZATION_SCOPE, AuditSpec
from plasticity_placement.p0d2hrtb.probes import (
    BridgeProbe,
    compile_bridge_bank,
    route_anchor_static_sha256,
)
from plasticity_placement.training.model_utils import chat_prompt, load_tokenizer

SCHEMA_VERSION = "p0d2hrtb-audit-v1"
MANIFEST_SCHEMA_VERSION = "p0d2hrtb-manifest-v1"
AUTHORIZATION_SCHEMA_VERSION = "p0d2hrtb-authorization-v1"
ROW_SCHEMA_VERSION = "p0d2hrtb-decision-row-v1"


def plan_audit(*, output_dir: Path, cpr_output: Path, qualification_summary: Path) -> Path:
    output_dir = output_dir.resolve()
    cpr_output = cpr_output.resolve()
    qualification_summary = qualification_summary.resolve()
    _require_independent(cpr_output, output_dir)
    _require_independent(qualification_summary.parent, output_dir)
    if output_dir.exists():
        raise FileExistsError(f"route-transfer audit output already exists: {output_dir}")
    spec = AuditSpec()
    context = _validate_sources(cpr_output, qualification_summary)
    probes, bank_audit = compile_bridge_bank(context["source"].selected, context["hard_bank"], spec)
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
        record = {
            "probe_id": probe.probe_id,
            "formatted_prompt_sha256": audit["prompt_sha256"],
            **audit,
        }
        record["record_sha256"] = json_hash(record)
        token_records.append(record)
    token_audit = {
        "schema_version": "p0d2hrtb-token-audit-v1",
        "decision_count": len(token_records),
        "failed_probe_ids": [
            row["probe_id"] for row in token_records if row["all_candidates_valid"] is not True
        ],
        "records": token_records,
    }
    token_audit["all_checks_passed"] = not token_audit["failed_probe_ids"]
    q2_prompt_hashes = context["q2_external_prompt_hashes"]
    action_anchor_hashes = {
        probe.source_probe_id: record["prompt_sha256"]
        for probe, record in zip(probes, token_records, strict=True)
        if probe.cell == "external_action"
    }
    token_audit["external_action_anchor_matches_q2"] = action_anchor_hashes == q2_prompt_hashes
    q2_route_prompt_hashes = context["q2_route_prompt_hashes"]
    route_anchor_matches = {
        str(probe.route_anchor_probe_id): record["prompt_sha256"]
        for probe, record in zip(probes, token_records, strict=True)
        if probe.cell == "slot_map_snake_opaque"
    }
    token_audit["exact_crd_route_anchor_matches_q2"] = len(route_anchor_matches) == 96 and all(
        q2_route_prompt_hashes.get(probe_id) == digest
        for probe_id, digest in route_anchor_matches.items()
    )
    q2_route_static_hashes = context["q2_route_static_hashes"]
    token_audit["exact_crd_scoring_identity_matches_q2"] = all(
        probe.route_anchor_static_sha256
        == q2_route_static_hashes.get(str(probe.route_anchor_probe_id))
        for probe in probes
        if probe.response_type == "slot"
    )
    token_audit["all_checks_passed"] = (
        token_audit["all_checks_passed"]
        and token_audit["external_action_anchor_matches_q2"]
        and token_audit["exact_crd_route_anchor_matches_q2"]
        and token_audit["exact_crd_scoring_identity_matches_q2"]
    )
    if token_audit["all_checks_passed"] is not True:
        raise ValueError("route-transfer candidate-token audit failed")
    output_dir.mkdir(parents=True)
    probe_hash = immutable_write(
        output_dir / "preflight" / "bridge_probes.jsonl",
        _jsonl([probe.to_dict() for probe in probes]),
        "route-transfer bridge probes",
    )
    bank_hash = immutable_json_write(
        output_dir / "preflight" / "bank_audit.json",
        bank_audit,
        "route-transfer bank audit",
    )
    token_hash = immutable_json_write(
        output_dir / "preflight" / "token_audit.json",
        token_audit,
        "route-transfer token audit",
    )
    source_snapshot = _source_snapshot(cpr_output, qualification_summary.parent)
    identity = {
        "schema_version": "p0d2hrtb-preregistration-v1",
        "code_sha256": current_code_hash(),
        "spec": spec.to_dict(),
        "cpr_output": str(cpr_output),
        "qualification_summary": str(qualification_summary),
        "cpr_run_id": context["cpr_manifest"]["run_id"],
        "qualification_run_id": context["qualification"]["run_id"],
        "adapter_sha256": context["adapter_sha256"],
        "source_snapshot": source_snapshot,
        "probe_sha256": probe_hash,
        "bank_audit_sha256": bank_hash,
        "token_audit_sha256": token_hash,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    preregistration_sha256 = json_hash(identity)
    identity["preregistration_sha256"] = preregistration_sha256
    immutable_json_write(
        output_dir / "preregistration.json", identity, "route-transfer preregistration"
    )
    immutable_json_write(
        output_dir / "authorization.template.json",
        authorization_template(preregistration_sha256),
        "route-transfer authorization template",
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": "p0d2hrtb-" + preregistration_sha256[:10],
        "state": "planned",
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "config": identity,
        "authorization": None,
        "errors": [],
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
        "allows_mappings_per_adapter_scan": False,
    }


def authorize_audit(output_dir: Path, authorization_path: Path) -> Path:
    output_dir = output_dir.resolve()
    authorization_path = authorization_path.resolve()
    manifest = _load_manifest(output_dir)
    if manifest["state"] != "planned":
        raise PermissionError("route-transfer authorization requires planned state")
    if authorization_path == output_dir or authorization_path.is_relative_to(output_dir):
        raise ValueError("route-transfer authorization must be external")
    payload = read_json_object(authorization_path, "route-transfer authorization")
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
        raise PermissionError(f"route-transfer authorization differs: {differences}")
    if payload.get("approved_by") in {None, "", "TBD"} or payload.get("approved_at") == "TBD":
        raise ValueError("route-transfer authorization requires approver and timestamp")
    datetime.fromisoformat(str(payload["approved_at"]).replace("Z", "+00:00"))
    digest = immutable_json_write(
        output_dir / "authorization.json", payload, "route-transfer authorization"
    )
    manifest["state"] = "authorized"
    manifest["authorization"] = {"sha256": digest, "approved_by": payload["approved_by"]}
    _save_manifest(output_dir, manifest)
    return output_dir / "manifest.json"


def run_audit(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    manifest = _load_manifest(output_dir)
    if manifest["state"] != "authorized":
        raise PermissionError("route-transfer run requires authorized state")
    identity = manifest["config"]
    if current_code_hash() != identity["code_sha256"]:
        raise ValueError("route-transfer code changed after preregistration")
    _verify_preflight(output_dir, identity)
    cpr_output = Path(identity["cpr_output"])
    qualification_summary = Path(identity["qualification_summary"])
    context = _validate_sources(cpr_output, qualification_summary)
    before = _source_snapshot(cpr_output, qualification_summary.parent)
    if before != identity["source_snapshot"]:
        raise ValueError("route-transfer source artifacts changed after preregistration")
    probes = _load_probes(output_dir)
    audits = _load_token_audits(output_dir)
    spec = AuditSpec()
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
        _progress("loading one base-model runtime")
        bundle = load_base_model(load_config)
        _require_source_compatible_bundle(bundle, rr_config)
        base_id = id(bundle.model)
        activate_adapter(bundle, cpr_output / "adapter", adapter_name="cpr_route_transfer_bridge")
        if not hasattr(bundle.model, "disable_adapter"):
            raise RuntimeError("installed PEFT version lacks disable_adapter()")
        bundle.model.requires_grad_(False)
        bundle.model.eval()
        wrapped_id = id(bundle.model.get_base_model())
        if wrapped_id != base_id:
            raise RuntimeError("bridge adapter activation replaced the base-model object")
        peft_id = id(bundle.model)
        first = probes[0]
        first_formatted = chat_prompt(bundle.tokenizer, first.prompt)
        sentinel_before = _score_off(bundle, first_formatted, audits[first.probe_id])
        off_rows, on_rows = _score_bank(
            bundle,
            probes,
            audits,
            run_id=manifest["run_id"],
            adapter_sha256=context["adapter_sha256"],
        )
        sentinel_after = _score_off(bundle, first_formatted, audits[first.probe_id])
        sentinel = compare_sentinel_scores(
            sentinel_before,
            sentinel_after,
            tolerance=spec.sentinel_score_tolerance,
        )
        runtime_identity = {
            "single_process": True,
            "single_loaded_base_model": True,
            "pair_order": "adapter_off_then_adapter_on_per_prompt",
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
            raise RuntimeError("route-transfer same-runtime integrity check failed")
    except BaseException as error:
        _mark_failed(output_dir, error)
        raise
    finally:
        if bundle is not None:
            release_model(bundle)
    try:
        raw_artifact_hashes = _publish_raw_scores(
            output_dir,
            run_id=manifest["run_id"],
            off_rows=off_rows,
            on_rows=on_rows,
        )
    except BaseException as error:
        _mark_failed(output_dir, error)
        raise
    try:
        analysis, artifact_hashes = _analyze_and_publish(
            output_dir=output_dir,
            manifest=manifest,
            identity=identity,
            context=context,
            runtime_identity=runtime_identity,
            spec=spec,
            cpr_output=cpr_output,
            qualification_summary=qualification_summary,
            before=before,
            off_rows=off_rows,
            on_rows=on_rows,
            raw_artifact_hashes=raw_artifact_hashes,
        )
    except BaseException as error:
        _mark_failed(output_dir, error)
        raise
    try:
        manifest = _load_manifest(output_dir)
        manifest["state"] = "complete"
        manifest["result"] = {
            "summary_sha256": artifact_hashes["summary"],
            "decision": analysis["decision"]["status"],
        }
        _save_manifest(output_dir, manifest)
    except BaseException as error:
        _mark_failed(output_dir, error)
        raise
    _progress(f"complete: {output_dir / 'summary.json'}")
    return output_dir / "summary.json"


def _analyze_and_publish(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    identity: dict[str, Any],
    context: dict[str, Any],
    runtime_identity: dict[str, Any],
    spec: AuditSpec,
    cpr_output: Path,
    qualification_summary: Path,
    before: dict[str, Any],
    off_rows: list[dict[str, Any]],
    on_rows: list[dict[str, Any]],
    raw_artifact_hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    pairs = build_pairs(off_rows, on_rows)
    analysis = analyze_bridge(
        pairs,
        bootstrap_samples=spec.bootstrap_samples,
        bootstrap_seed=spec.bootstrap_seed,
        adjusted_confidence=spec.primary_two_sided_confidence,
        material_threshold=spec.material_rescue_threshold,
    )
    after = _source_snapshot(cpr_output, qualification_summary.parent)
    if after != before:
        raise RuntimeError("route-transfer source artifacts changed during audit")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "analysis_status": "post_failure_inference_only_bridge_complete",
        "source": {
            "cpr_run_id": context["cpr_manifest"]["run_id"],
            "qualification_run_id": context["qualification"]["run_id"],
            "adapter_sha256": context["adapter_sha256"],
        },
        "runtime_identity": runtime_identity,
        "raw_score_checkpoint": raw_artifact_hashes,
        "environment": current_environment_snapshot(),
        "analysis": analysis,
        "source_snapshot_before": before,
        "source_snapshot_after": after,
        "source_artifacts_modified": False,
        "historical_gate_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
        "interpretation_boundary": (
            "This post-failure bridge audit localizes prompt-transfer barriers for one frozen "
            "adapter. It is not a qualification retry and cannot authorize training or 1/4/8."
        ),
    }
    analysis_hashes = _publish_analysis(output_dir, summary, pairs)
    artifact_hashes = {**raw_artifact_hashes, **analysis_hashes}
    if _source_snapshot(cpr_output, qualification_summary.parent) != before:
        raise RuntimeError("route-transfer source artifacts changed before publication")
    audit_manifest = {
        "schema_version": "p0d2hrtb-audit-manifest-v1",
        "run_id": manifest["run_id"],
        "preregistration_sha256": identity["preregistration_sha256"],
        "source_snapshot_sha256": json_hash(before),
        "artifacts": artifact_hashes,
        "historical_gate_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    immutable_json_write(output_dir / "audit_manifest.json", audit_manifest, "bridge manifest")
    return analysis, artifact_hashes


def _validate_sources(cpr_output: Path, qualification_summary: Path) -> dict[str, Any]:
    cpr_manifest_obj, cpr_spec, cpr_identity = load_cpr_config(cpr_output)
    cpr_manifest = cpr_manifest_obj.payload
    if cpr_manifest_obj.state != "trained":
        raise ValueError("route-transfer source CPR adapter is not trained")
    adapter_sha256 = _adapter_hash(cpr_output / "adapter")
    if adapter_sha256 != cpr_manifest["training"]["summary"]["adapter_sha256"]:
        raise ValueError("route-transfer source adapter changed")
    qualification = read_json_object(qualification_summary, "CPR q2 qualification summary")
    qualification_manifest = read_json_object(
        qualification_summary.parent / "qualification_manifest.json",
        "CPR q2 qualification manifest",
    )
    if (
        qualification.get("schema_version") != "p0d2hcpr-same-runtime-qualification-v1"
        or qualification.get("source", {}).get("cpr_run_id") != cpr_manifest["run_id"]
        or qualification.get("runtime_identity", {}).get("adapter_sha256") != adapter_sha256
        or qualification.get("source_artifacts_modified") is not False
        or qualification.get("mappings_per_adapter_authorized") is not False
        or qualification.get("decision", {}).get("status") != "scoring_integrity_failed"
        or qualification.get("decision", {})
        .get("checks", {})
        .get("conditional_route_tie_robust_at_least_0_75")
        is not False
    ):
        raise ValueError("route-transfer source is not the closed failed CPR q2 result")
    if (
        qualification_manifest.get("run_id") != qualification["run_id"]
        or qualification_manifest.get("artifacts", {}).get("summary", {}).get("sha256")
        != file_hash(qualification_summary)
        or qualification_manifest.get("artifacts", {}).get("paired_records", {}).get("sha256")
        != file_hash(qualification_summary.parent / "paired_records.jsonl")
        or qualification_manifest.get("artifacts", {}).get("off_forced_choice", {}).get("sha256")
        != file_hash(qualification_summary.parent / "results" / "adapter_off_fc.jsonl")
        or qualification_manifest.get("artifacts", {}).get("off_crd", {}).get("sha256")
        != file_hash(qualification_summary.parent / "results" / "adapter_off_crd.jsonl")
        or qualification_manifest.get("mappings_per_adapter_authorized") is not False
    ):
        raise ValueError("CPR q2 qualification manifest does not bind its summary")
    violations = []
    for row in _read_jsonl(qualification_summary.parent / "paired_records.jsonl"):
        for state in ("base", "adapter"):
            if row.get(f"{state}_error_status") not in {"ok", "tie"} or (
                row.get(f"{state}_tie") and not row.get(f"{state}_tie_expected_compatible")
            ):
                violations.append(
                    (
                        row.get("matrix"),
                        row.get("endpoint"),
                        row.get("probe_id"),
                        state,
                        row.get(f"{state}_error_status"),
                    )
                )
    expected_violation = [
        (
            "crd",
            "combined",
            "F_pair04_a_crd_combined_r04_ap3_so1",
            "base",
            "tie",
        )
    ]
    if violations != expected_violation:
        raise ValueError("CPR q2 scoring-integrity evidence differs from the reviewed failure")
    q2_external_rows = _read_jsonl(
        qualification_summary.parent / "results" / "adapter_off_fc.jsonl"
    )
    q2_external_prompt_hashes = {
        str(row["probe_id"]): str(row["prompt_sha256"]) for row in q2_external_rows
    }
    if len(q2_external_prompt_hashes) != 96:
        raise ValueError("CPR q2 external conditional-route anchor is incomplete")
    q2_crd_rows = _read_jsonl(qualification_summary.parent / "results" / "adapter_off_crd.jsonl")
    q2_route_prompt_hashes = {
        str(row["probe_id"]): str(row["prompt_sha256"])
        for row in q2_crd_rows
        if row.get("endpoint") == "route_only"
    }
    if len(q2_route_prompt_hashes) != 384:
        raise ValueError("CPR q2 CRD route-only anchor is incomplete")
    q2_route_static_hashes = {
        str(row["probe_id"]): route_anchor_static_sha256(row)
        for row in q2_crd_rows
        if row.get("endpoint") == "route_only"
    }
    rr_config = load_resolved_config(Path(cpr_identity["source_output"]))
    source = validate_frozen_source(rr_config.source_manifest_path)
    hard_bank = load_hard_probe_bank(source.hard_probe_dir)
    return {
        "cpr_manifest": cpr_manifest,
        "cpr_spec": cpr_spec,
        "cpr_identity": cpr_identity,
        "qualification": qualification,
        "adapter_sha256": adapter_sha256,
        "source": source,
        "hard_bank": hard_bank,
        "q2_external_prompt_hashes": q2_external_prompt_hashes,
        "q2_route_prompt_hashes": q2_route_prompt_hashes,
        "q2_route_static_hashes": q2_route_static_hashes,
    }


def _score_bank(
    bundle: ModelBundle,
    probes: list[BridgeProbe],
    audits: dict[str, dict[str, Any]],
    *,
    run_id: str,
    adapter_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    off_rows: list[dict[str, Any]] = []
    on_rows: list[dict[str, Any]] = []
    for probe in probes:
        formatted = chat_prompt(bundle.tokenizer, probe.prompt)
        off, on = score_adapter_off_on(
            bundle, formatted, audits[probe.probe_id], score_candidate_batch
        )
        common = {
            "schema_version": ROW_SCHEMA_VERSION,
            "run_id": run_id,
            **probe.to_dict(),
            "prompt_sha256": audits[probe.probe_id]["prompt_sha256"],
            "candidate_audit_record_sha256": audits[probe.probe_id]["record_sha256"],
            "precision": bundle.precision,
        }
        off_row = _decision_row(common, off, "adapter_off", None)
        on_row = _decision_row(common, on, "adapter_on", adapter_sha256)
        off_rows.append(off_row)
        on_rows.append(on_row)
        if len(off_rows) % 96 == 0:
            _progress(f"bridge pairs complete: {len(off_rows)}/{len(probes)}")
    return off_rows, on_rows


def _decision_row(
    common: dict[str, Any], outcome: dict[str, Any], state: str, adapter_sha256: str | None
) -> dict[str, Any]:
    predicted = outcome["predicted_candidate"]
    return {
        **common,
        "adapter_state": state,
        "adapter_sha256": adapter_sha256,
        "candidates": outcome["candidates"],
        "predicted_candidate": predicted,
        "mean_predicted_candidate": outcome["mean_predicted_candidate"],
        "correct": predicted == common["expected_candidate"],
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


def _publish_raw_scores(
    output: Path,
    *,
    run_id: str,
    off_rows: list[dict[str, Any]],
    on_rows: list[dict[str, Any]],
) -> dict[str, str]:
    hashes = {
        "adapter_off": immutable_write(
            output / "results" / "adapter_off.jsonl",
            _jsonl(off_rows),
            "route-transfer raw adapter_off",
        ),
        "adapter_on": immutable_write(
            output / "results" / "adapter_on.jsonl",
            _jsonl(on_rows),
            "route-transfer raw adapter_on",
        ),
    }
    checkpoint = {
        "schema_version": "p0d2hrtb-raw-score-checkpoint-v1",
        "run_id": run_id,
        "adapter_off_count": len(off_rows),
        "adapter_on_count": len(on_rows),
        "artifacts": dict(hashes),
        "checkpoint_stage": "inference_complete_before_analysis",
        "automatic_analysis_recovery_allowed": False,
    }
    hashes["raw_score_checkpoint"] = immutable_json_write(
        output / "raw_score_checkpoint.json",
        checkpoint,
        "route-transfer raw score checkpoint",
    )
    return hashes


def _publish_analysis(
    output: Path,
    summary: dict[str, Any],
    pairs: list[dict[str, Any]],
) -> dict[str, str]:
    payloads = {
        "summary": (output / "summary.json", _json(summary)),
        "report": (output / "report.md", _render_report(summary).encode()),
        "paired_records": (output / "paired_records.jsonl", _jsonl(pairs)),
    }
    return {
        name: immutable_write(path, payload, f"route-transfer {name}")
        for name, (path, payload) in payloads.items()
    }


def _render_report(summary: dict[str, Any]) -> str:
    analysis = summary["analysis"]
    lines = [
        "# CPR route-transfer bridge audit",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Decision: `{analysis['decision']['status']}`",
        "- Primary factorial zero-tie integrity: "
        f"`{str(analysis['decision']['primary_factorial_zero_tie_integrity']).lower()}`",
        "- Action expected-compatible-tie integrity: "
        f"`{str(analysis['decision']['action_expected_compatible_tie_integrity']).lower()}`",
        "- Historical gate changed: `false`",
        "- Training authorized: `false`",
        "- 1/4/8 authorized: `false`",
        "",
        "| Cell | N | OFF | ON | ON−OFF (95% lesson-cluster CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell, values in analysis["by_cell"].items():
        difference = values["paired_accuracy_difference"]
        interval = difference["ci95"]
        lines.append(
            f"| {cell} | {values['decision_count']} | {values['base_accuracy']:.4f} | "
            f"{values['adapter_accuracy']:.4f} | {difference['estimate']:.4f} "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] |"
        )
    lines.extend(["", "## Primary rescue contrasts", ""])
    for factor, values in analysis["primary_rescue_contrasts"].items():
        lines.append(
            f"- `{factor}`: {values['estimate']:.4f} "
            f"[{values['ci'][0]:.4f}, {values['ci'][1]:.4f}], "
            f"supported=`{str(values['supported_barrier']).lower()}`."
        )
    lines.extend(
        [
            "",
            "This post-failure audit is diagnostic only. It cannot reopen CPR-v1 or authorize "
            "training or the 1/4/8 experiment.",
            "",
        ]
    )
    return "\n".join(lines)


def _verify_preflight(output: Path, identity: dict[str, Any]) -> None:
    files = {
        "probe_sha256": output / "preflight" / "bridge_probes.jsonl",
        "bank_audit_sha256": output / "preflight" / "bank_audit.json",
        "token_audit_sha256": output / "preflight" / "token_audit.json",
    }
    for field, path in files.items():
        if file_hash(path) != identity[field]:
            raise ValueError(f"route-transfer preflight artifact changed: {field}")
    prereg = read_json_object(output / "preregistration.json", "route-transfer preregistration")
    if (
        prereg != identity
        or json_hash({k: v for k, v in identity.items() if k != "preregistration_sha256"})
        != identity["preregistration_sha256"]
    ):
        raise ValueError("route-transfer preregistration changed")


def _load_probes(output: Path) -> list[BridgeProbe]:
    return [BridgeProbe(**row) for row in _read_jsonl(output / "preflight" / "bridge_probes.jsonl")]


def _load_token_audits(output: Path) -> dict[str, dict[str, Any]]:
    payload = read_json_object(output / "preflight" / "token_audit.json", "bridge token audit")
    result = {str(row["probe_id"]): row for row in payload["records"]}
    if len(result) != 960 or payload.get("all_checks_passed") is not True:
        raise ValueError("route-transfer token audit is incomplete")
    return result


def _source_snapshot(cpr_output: Path, qualification_output: Path) -> dict[str, Any]:
    return {
        "cpr_output_tree": _tree_hash(cpr_output),
        "qualification_output_tree": _tree_hash(qualification_output),
    }


def _load_manifest(output: Path) -> dict[str, Any]:
    manifest = read_json_object(output / "manifest.json", "route-transfer manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("not a route-transfer manifest")
    return manifest


def _save_manifest(output: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    atomic_json_write(output / "manifest.json", manifest)


def _mark_failed(output: Path, error: BaseException) -> None:
    manifest = _load_manifest(output)
    manifest["state"] = "failed"
    manifest["errors"].append(
        {
            "type": type(error).__name__,
            "message": str(error),
            "time": datetime.now(UTC).isoformat(),
        }
    )
    _save_manifest(output, manifest)


def spec_context_model(context: dict[str, Any]) -> str:
    return str(context["cpr_spec"].model_name)


def spec_context_revision(context: dict[str, Any]) -> str:
    return str(context["cpr_spec"].model_revision)


def _require_independent(source: Path, output: Path) -> None:
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("route-transfer output must be independent from source artifacts")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()


def _json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _progress(message: str) -> None:
    print(f"[p0d2hrtb] {message}", flush=True)
