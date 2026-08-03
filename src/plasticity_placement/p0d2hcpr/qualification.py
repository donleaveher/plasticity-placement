from __future__ import annotations

from dataclasses import dataclass
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
from plasticity_placement.p0d2hcpr.preflight import load_config
from plasticity_placement.p0d2hcrd.probes import compile_decomposition_bank
from plasticity_placement.p0d2hcrd.runtime import validate_frozen_source
from plasticity_placement.p0d2hfc.scoring import score_candidate_batch as score_fc_candidate_batch
from plasticity_placement.p0d2hrr.io import file_hash, immutable_write, json_hash, read_json_object
from plasticity_placement.p0d2hrr.paired_audit import (
    ENDPOINTS,
    PAIRED_ROW_SCHEMA_VERSION,
    _json_bytes,
    _jsonl_bytes,
    _validate_route_remediation,
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
    _verify_crd_bank_audit,
)
from plasticity_placement.p0d2hrr.same_runtime_audit import (
    _evaluate_crd_pairs,
    _evaluate_forced_choice_pairs,
    _first_forced_choice_item,
    _score_adapter_off,
    _tree_hash,
    compare_sentinel_scores,
)

SCHEMA_VERSION = "p0d2hcpr-same-runtime-qualification-v1"
RECOVERY_AUTHORIZATION_SCHEMA_VERSION = "p0d2hcpr-qualification-recovery-authorization-v1"
RECOVERY_AUTHORIZATION_SCOPE = "single_post_inference_classification_failure_recovery"
RECOVERABLE_FAILURE_SIGNATURE = "KeyError:accuracy_interval_after_complete_scoring"


@dataclass(frozen=True, slots=True)
class QualificationRequest:
    output_dir: Path
    analysis_output: Path
    recovery_authorization: Path | None = None


def run_qualification(request: QualificationRequest) -> Path:
    output_dir = request.output_dir.resolve()
    analysis_output = request.analysis_output.resolve()
    _require_new_independent(output_dir, analysis_output)
    manifest, spec, identity = load_config(output_dir)
    if manifest.state != "trained":
        raise PermissionError(f"CPR qualification requires state=trained, found {manifest.state}")
    analysis_code_sha256 = current_code_hash()
    recovery_mode = request.recovery_authorization is not None
    if analysis_code_sha256 != identity["code_sha256"] and not recovery_mode:
        raise ValueError("code changed after CPR preregistration")
    adapter_sha256 = _adapter_hash(output_dir / "adapter")
    if adapter_sha256 != manifest.payload["training"]["summary"]["adapter_sha256"]:
        raise ValueError("CPR adapter changed after training")
    _validate_trained_artifacts(output_dir, manifest.payload, spec, identity, adapter_sha256)

    rr_output = Path(str(identity["source_output"]))
    revision_lock = Path(str(identity["source_code_revision_lock"]))
    _require_new_independent(rr_output, analysis_output)
    _require_new_independent(Path(str(identity["same_runtime_summary"])).parent, analysis_output)
    if file_hash(revision_lock) != identity["source_code_revision_lock_sha256"]:
        raise ValueError("RR code revision lock changed")
    if (
        file_hash(Path(str(identity["same_runtime_summary"])))
        != identity["same_runtime_summary_sha256"]
    ):
        raise ValueError("prior same-runtime summary changed")
    if (
        file_hash(Path(str(identity["same_runtime_manifest"])))
        != identity["same_runtime_manifest_sha256"]
    ):
        raise ValueError("prior same-runtime audit manifest changed")
    source_provenance, _, _, _, _ = _validate_route_remediation(rr_output, revision_lock)
    if source_provenance != identity["source_route_remediation"]:
        raise ValueError("RR1 provenance changed after CPR preregistration")
    rr_config = load_resolved_config(rr_output)
    source = validate_frozen_source(rr_config.source_manifest_path)
    source_rows = load_scale_canary_source_rows(source)
    hard_bank = load_hard_probe_bank(source.hard_probe_dir)
    crd_bank, crd_bank_audit = compile_decomposition_bank(source.selected, hard_bank)
    stored_bank_audit = read_json_object(
        rr_output / "preflight" / "crd_bank_audit.json", "CRD bank audit"
    )
    _verify_crd_bank_audit(crd_bank_audit, stored_bank_audit)
    fc_audits = _load_fc_audits(rr_output, rr_config)
    crd_audits = _load_crd_audits(rr_output, rr_config)
    load_config_value = _AdapterLoadConfig(
        model_name=spec.model_name,
        model_revision=spec.model_revision,
        use_4bit=spec.training.use_4bit,
        training_seeds=(spec.training.seed,),
    )
    run_identity = {
        "schema_version": SCHEMA_VERSION,
        "cpr_run_id": manifest.payload["run_id"],
        "cpr_manifest_sha256": file_hash(manifest.path),
        "adapter_sha256": adapter_sha256,
        "training_code_sha256": identity["code_sha256"],
        "analysis_code_sha256": analysis_code_sha256,
        "qualification_mode": "approved_recovery" if recovery_mode else "primary",
        "bootstrap_samples": spec.gates.bootstrap_samples,
        "bootstrap_seed": spec.gates.bootstrap_seed,
        "combined_noninferiority_margin": spec.gates.combined_noninferiority_margin,
    }
    run_id = "p0d2hcpr-qualification-" + json_hash(run_identity)[:10]
    recovery_context = _claim_qualification(
        output_dir,
        analysis_output,
        run_id,
        adapter_sha256,
        analysis_code_sha256=analysis_code_sha256,
        recovery_authorization=request.recovery_authorization,
    )
    before = _source_snapshot(output_dir, identity)

    bundle: ModelBundle | None = None
    try:
        _progress("loading one base-model runtime")
        bundle = load_base_model(load_config_value)
        _require_source_compatible_bundle(bundle, rr_config)
        base_id = id(bundle.model)
        activate_adapter(bundle, output_dir / "adapter", adapter_name="composition_preserving")
        if not hasattr(bundle.model, "disable_adapter"):
            raise RuntimeError("installed PEFT version lacks disable_adapter()")
        bundle.model.requires_grad_(False)
        bundle.model.eval()
        peft_id = id(bundle.model)
        wrapped_id = id(bundle.model.get_base_model())
        if wrapped_id != base_id:
            raise RuntimeError("adapter activation replaced the base-model object")
        sentinel_item = _first_forced_choice_item(bundle, source, hard_bank, fc_audits)
        sentinel_before = _score_adapter_off(
            bundle,
            sentinel_item["formatted_prompt"],
            sentinel_item["audit"],
            score_fc_candidate_batch,
        )
        _progress("scoring 96 external conditional-route OFF/ON pairs")
        off_fc, on_fc = _evaluate_forced_choice_pairs(
            bundle=bundle,
            run_id=run_id,
            adapter_sha256=adapter_sha256,
            config=rr_config,
            source=source,
            hard_bank=hard_bank,
            source_rows=source_rows,
            audits=fc_audits,
        )
        _progress("scoring 1,536 CRD OFF/ON pairs")
        off_crd, on_crd = _evaluate_crd_pairs(
            bundle=bundle,
            run_id=run_id,
            adapter_sha256=adapter_sha256,
            config=rr_config,
            bank=crd_bank,
            source_rows=source.source_rows,
            audits=crd_audits,
        )
        sentinel_after = _score_adapter_off(
            bundle,
            sentinel_item["formatted_prompt"],
            sentinel_item["audit"],
            score_fc_candidate_batch,
        )
        sentinel = compare_sentinel_scores(
            sentinel_before,
            sentinel_after,
            tolerance=spec.gates.sentinel_score_tolerance,
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
            "adapter_name": "composition_preserving",
            "adapter_sha256": adapter_sha256,
            "sentinel": sentinel,
        }
        if not runtime_identity["object_identity_preserved"] or not sentinel["passed"]:
            raise RuntimeError("CPR same-runtime integrity check failed")
    finally:
        if bundle is not None:
            release_model(bundle)

    fc_pairs = build_forced_choice_pairs(off_fc, on_fc)
    crd_pairs = build_crd_pairs(off_crd, on_crd)
    analysis, failures = build_paired_analysis(
        fc_pairs,
        crd_pairs,
        off_crd,
        on_crd,
        bootstrap_samples=spec.gates.bootstrap_samples,
        bootstrap_seed=spec.gates.bootstrap_seed,
    )
    decision = classify_result(analysis, fc_pairs + crd_pairs, spec.gates)
    after = _source_snapshot(output_dir, identity)
    if before != after:
        raise RuntimeError("CPR source artifacts changed during qualification")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "analysis_status": "same_runtime_adapter_off_on_complete",
        "source": {
            "cpr_run_id": manifest.payload["run_id"],
            "cpr_manifest_sha256": file_hash(manifest.path),
            "prior_failed_same_runtime_run_id": identity["same_runtime_run_id"],
            "qualification_recovery": recovery_context,
        },
        "runtime_identity": runtime_identity,
        "environment": current_environment_snapshot(),
        "analysis": analysis,
        "decision": decision,
        "source_snapshot_before": before,
        "source_snapshot_after": after,
        "source_artifacts_modified": False,
        "historical_gate_changed": False,
        "mappings_per_adapter_authorized": False,
        "interpretation_boundary": (
            "This fixed-pilot qualification can identify a reading-qualification candidate. "
            "It never changes the historical gate or automatically authorizes 1/4/8."
        ),
    }
    paths, hashes = _stage(
        analysis_output, summary, off_fc, on_fc, off_crd, on_crd, fc_pairs, crd_pairs, failures
    )
    if _source_snapshot(output_dir, identity) != before:
        raise RuntimeError("CPR source artifacts changed before publication")
    audit_manifest = {
        "schema_version": "p0d2hcpr-qualification-manifest-v1",
        "run_id": run_id,
        "identity": run_identity,
        "source_snapshot_sha256": json_hash(before),
        "artifacts": {
            name: {"path": str(paths[name]), "sha256": digest} for name, digest in hashes.items()
        },
        "historical_gate_changed": False,
        "mappings_per_adapter_authorized": False,
    }
    immutable_write(paths["manifest"], _json_bytes(audit_manifest), "CPR qualification manifest")
    _progress(f"complete: {paths['summary']}")
    return paths["summary"]


def classify_result(
    analysis: dict[str, Any], pairs: list[dict[str, Any]], gates: Any
) -> dict[str, Any]:
    fc = analysis["forced_choice"]["external_conditional_route"]
    endpoints = analysis["crd"]["by_endpoint"]
    route, retrieval, combined = (endpoints[name] for name in ENDPOINTS)
    compatible_ties = all(
        (not row[f"{state}_tie"] or row[f"{state}_tie_expected_compatible"])
        and row[f"{state}_error_status"] in {"ok", "tie"}
        for row in pairs
        for state in ("base", "adapter")
    )
    fc_lower, _ = _tie_accuracy_bounds(fc, "adapter")
    route_lower, _ = _tie_accuracy_bounds(route, "adapter")
    retrieval_lower, _ = _tie_accuracy_bounds(retrieval, "adapter")
    combined_adapter_lower, _ = _tie_accuracy_bounds(combined, "adapter")
    _, combined_base_upper = _tie_accuracy_bounds(combined, "base")
    combined_worst_delta = combined_adapter_lower - combined_base_upper
    route_ci = route["paired_accuracy_difference"]["ci95"]
    combined_ci = combined["paired_accuracy_difference"]["ci95"]
    material_combined_regression = combined_ci[1] < -gates.combined_noninferiority_margin
    checks = {
        "scoring_integrity_and_expected_compatible_ties": compatible_ties,
        "conditional_route_tie_robust_at_least_0_75": fc_lower
        >= gates.conditional_route_min_accuracy,
        "route_only_tie_robust_strictly_above_0_625": route_lower
        > gates.route_only_min_accuracy_exclusive,
        "route_delta_ci_lower_strictly_positive": route_ci[0]
        > gates.route_delta_ci_lower_exclusive,
        "retrieval_only_tie_robust_at_least_1_0": retrieval_lower
        >= gates.retrieval_only_min_accuracy,
        "combined_cluster_ci_noninferior": combined_ci[0] >= -gates.combined_noninferiority_margin,
        "combined_tie_robust_noninferior": combined_worst_delta
        >= -gates.combined_noninferiority_margin,
    }
    if not compatible_ties:
        status = "scoring_integrity_failed"
    elif all(checks.values()):
        status = "reading_qualification_candidate_review_required"
    elif checks["route_delta_ci_lower_strictly_positive"] and material_combined_regression:
        status = "composition_interference_confirmed"
    else:
        status = "qualification_failed"
    return {
        "status": status,
        "checks": checks,
        "diagnostics": {
            "combined_material_regression_supported": material_combined_regression,
        },
        "tie_robust_values": {
            "conditional_route_adapter_accuracy_lower": fc_lower,
            "route_only_adapter_accuracy_lower": route_lower,
            "retrieval_only_adapter_accuracy_lower": retrieval_lower,
            "combined_adapter_minus_base_lower": combined_worst_delta,
        },
        "thresholds": {
            "conditional_route_min_accuracy": gates.conditional_route_min_accuracy,
            "route_only_min_accuracy_exclusive": gates.route_only_min_accuracy_exclusive,
            "combined_noninferiority_margin": gates.combined_noninferiority_margin,
        },
        "historical_gate_changed": False,
        "mappings_per_adapter_authorized": False,
        "next_action": (
            "independent_review_then_formal_gate_recheck"
            if status == "reading_qualification_candidate_review_required"
            else "stop_cpr_v1_and_review_before_any_new_intervention"
        ),
    }


def _tie_accuracy_bounds(metric: dict[str, Any], state: str) -> tuple[float, float]:
    tie = metric.get("tie_sensitivity", {}).get(state)
    if not isinstance(tie, dict) or tie.get("bounds_valid") is not True:
        raise ValueError(f"{state} tie-sensitivity bounds are missing or invalid")
    lower = tie.get("all_ties_incorrect_accuracy")
    upper = tie.get("all_ties_compatible_accuracy")
    if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
        raise ValueError(f"{state} tie-sensitivity accuracy fields are missing")
    if not 0.0 <= float(lower) <= float(upper) <= 1.0:
        raise ValueError(f"{state} tie-sensitivity accuracy bounds are invalid")
    return float(lower), float(upper)


def _stage(
    output: Path,
    summary: dict[str, Any],
    off_fc: list[dict[str, Any]],
    on_fc: list[dict[str, Any]],
    off_crd: list[dict[str, Any]],
    on_crd: list[dict[str, Any]],
    fc_pairs: list[dict[str, Any]],
    crd_pairs: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> tuple[dict[str, Path], dict[str, str]]:
    paths = {
        "summary": output / "summary.json",
        "report": output / "report.md",
        "off_forced_choice": output / "results" / "adapter_off_fc.jsonl",
        "on_forced_choice": output / "results" / "adapter_on_fc.jsonl",
        "off_crd": output / "results" / "adapter_off_crd.jsonl",
        "on_crd": output / "results" / "adapter_on_crd.jsonl",
        "paired_records": output / "paired_records.jsonl",
        "combined_failures": output / "combined_failure_records.jsonl",
        "manifest": output / "qualification_manifest.json",
    }
    pair_rows = [
        {"schema_version": PAIRED_ROW_SCHEMA_VERSION, "matrix": "forced_choice", **row}
        for row in fc_pairs
    ] + [{"schema_version": PAIRED_ROW_SCHEMA_VERSION, "matrix": "crd", **row} for row in crd_pairs]
    payloads = {
        "summary": _json_bytes(summary),
        "report": _render_report(summary).encode(),
        "off_forced_choice": _jsonl_bytes(off_fc),
        "on_forced_choice": _jsonl_bytes(on_fc),
        "off_crd": _jsonl_bytes(off_crd),
        "on_crd": _jsonl_bytes(on_crd),
        "paired_records": _jsonl_bytes(pair_rows),
        "combined_failures": _jsonl_bytes(failures),
    }
    hashes = {
        name: immutable_write(paths[name], value, f"CPR qualification {name}")
        for name, value in payloads.items()
    }
    return paths, hashes


def _render_report(summary: dict[str, Any]) -> str:
    analysis = summary["analysis"]
    rows = [("External conditional-route", analysis["forced_choice"]["external_conditional_route"])]
    rows.extend((f"CRD {name}", analysis["crd"]["by_endpoint"][name]) for name in ENDPOINTS)
    lines = [
        "# CPR-v1 same-runtime qualification",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Decision: `{summary['decision']['status']}`",
        "- Historical gate changed: `false`",
        "- 1/4/8 authorized: `false`",
        "",
        "| Endpoint | N | OFF | ON | ON-OFF (95% lesson-cluster CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in rows:
        difference = values["paired_accuracy_difference"]
        interval = difference["ci95"]
        lines.append(
            f"| {label} | {values['decision_count']} | {values['base_accuracy']:.4f} | "
            f"{values['adapter_accuracy']:.4f} | {difference['estimate']:.4f} "
            f"[{interval[0]:.4f}, {interval[1]:.4f}] |"
        )
    lines.extend(
        ["", "The decision uses lesson-cluster uncertainty and tie-robust accuracy bounds.", ""]
    )
    return "\n".join(lines)


def _source_snapshot(output_dir: Path, identity: dict[str, Any]) -> dict[str, Any]:
    rr_output = Path(str(identity["source_output"]))
    frozen_source_root = Path(str(identity["source_context"]["source_manifest_path"])).parent
    paths = {
        "cpr_manifest": output_dir / "manifest.json",
        "cpr_preregistration": output_dir / "preregistration.json",
        "cpr_authorization": output_dir / "authorization.json",
        "cpr_training_metadata": output_dir / "adapter" / "training_metadata.json",
        "qualification_claim": output_dir / "qualification_claim.json",
        "rr_manifest": rr_output / "manifest.json",
        "rr_revision_lock": Path(str(identity["source_code_revision_lock"])),
        "prior_same_runtime_summary": Path(str(identity["same_runtime_summary"])),
        "prior_same_runtime_manifest": Path(str(identity["same_runtime_manifest"])),
    }
    optional_paths = {
        "qualification_recovery_authorization": (
            output_dir / "qualification_recovery_authorization.json"
        ),
        "qualification_recovery_claim": output_dir / "qualification_recovery_claim.json",
    }
    paths.update({name: path for name, path in optional_paths.items() if path.exists()})
    snapshot = {
        name: {"path": str(path), "sha256": file_hash(path)} for name, path in paths.items()
    }
    snapshot["adapter_bundle"] = {
        "path": str(output_dir / "adapter"),
        "sha256": _adapter_hash(output_dir / "adapter"),
    }
    snapshot["rr_output_tree"] = _tree_hash(rr_output)
    snapshot["frozen_forced_choice_tree"] = _tree_hash(frozen_source_root)
    return snapshot


def _require_new_independent(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"CPR qualification output already exists: {output}")
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("CPR qualification output must be independent from training output")


def _claim_qualification(
    output_dir: Path,
    analysis_output: Path,
    run_id: str,
    adapter_sha256: str,
    *,
    analysis_code_sha256: str | None = None,
    recovery_authorization: Path | None = None,
) -> dict[str, Any]:
    path = output_dir / "qualification_claim.json"
    if recovery_authorization is not None:
        return _claim_recovery_qualification(
            output_dir,
            analysis_output,
            run_id,
            adapter_sha256,
            recovery_authorization,
            analysis_code_sha256 or current_code_hash(),
        )
    if path.exists():
        raise PermissionError("CPR-v1 locked qualification has already been claimed")
    payload = {
        "schema_version": "p0d2hcpr-qualification-claim-v1",
        "run_id": run_id,
        "analysis_output": str(analysis_output),
        "adapter_sha256": adapter_sha256,
        "allowed_locked_qualifications": 1,
        "mappings_per_adapter_authorized": False,
    }
    from plasticity_placement.p0d2hrr.io import immutable_json_write

    immutable_json_write(path, payload, "CPR qualification claim")
    return {
        "mode": "primary",
        "original_claim_sha256": None,
        "recovery_authorization_sha256": None,
    }


def _claim_recovery_qualification(
    output_dir: Path,
    analysis_output: Path,
    run_id: str,
    adapter_sha256: str,
    authorization_path: Path,
    analysis_code_sha256: str,
) -> dict[str, Any]:
    from datetime import datetime

    from plasticity_placement.p0d2hrr.io import immutable_json_write

    original_claim_path = output_dir / "qualification_claim.json"
    recovery_claim_path = output_dir / "qualification_recovery_claim.json"
    adopted_authorization_path = output_dir / "qualification_recovery_authorization.json"
    if not original_claim_path.is_file():
        raise PermissionError("recovery requires the immutable original qualification claim")
    if recovery_claim_path.exists():
        raise PermissionError("the single CPR qualification recovery has already been claimed")
    original_claim = read_json_object(original_claim_path, "original qualification claim")
    original_analysis_output = Path(str(original_claim.get("analysis_output", ""))).resolve()
    if original_analysis_output.exists():
        raise PermissionError("recovery is forbidden because original result artifacts exist")
    if original_claim.get("adapter_sha256") != adapter_sha256:
        raise ValueError("original qualification claim does not bind this adapter")
    authorization_path = authorization_path.resolve()
    if authorization_path == output_dir or authorization_path.is_relative_to(output_dir):
        raise ValueError("recovery authorization must be authored outside the CPR output")
    authorization = read_json_object(authorization_path, "qualification recovery authorization")
    expected_claim_sha256 = file_hash(original_claim_path)
    required = {
        "schema_version": RECOVERY_AUTHORIZATION_SCHEMA_VERSION,
        "decision": "approved",
        "scope": RECOVERY_AUTHORIZATION_SCOPE,
        "failure_signature": RECOVERABLE_FAILURE_SIGNATURE,
        "original_claim_sha256": expected_claim_sha256,
        "original_analysis_output": str(original_analysis_output),
        "recovery_analysis_output": str(analysis_output),
        "analysis_code_sha256": analysis_code_sha256,
        "allowed_recovery_runs": 1,
        "allows_training": False,
        "allows_mappings_per_adapter_scan": False,
    }
    differences = {
        key: {"expected": value, "observed": authorization.get(key)}
        for key, value in required.items()
        if authorization.get(key) != value
    }
    if differences:
        raise PermissionError(f"recovery authorization differs: {differences}")
    approver = authorization.get("approved_by")
    approved_at = authorization.get("approved_at")
    if not isinstance(approver, str) or not approver.strip() or approver == "TBD":
        raise ValueError("recovery authorization requires a named approver")
    if not isinstance(approved_at, str) or approved_at == "TBD":
        raise ValueError("recovery authorization requires an approval timestamp")
    datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    if adopted_authorization_path.exists():
        adopted_authorization = read_json_object(
            adopted_authorization_path, "adopted qualification recovery authorization"
        )
        if adopted_authorization != authorization:
            raise PermissionError("adopted recovery authorization differs from external approval")
        authorization_sha256 = file_hash(adopted_authorization_path)
    else:
        authorization_sha256 = immutable_json_write(
            adopted_authorization_path,
            authorization,
            "qualification recovery authorization",
        )
    claim = {
        "schema_version": "p0d2hcpr-qualification-recovery-claim-v1",
        "run_id": run_id,
        "analysis_output": str(analysis_output),
        "adapter_sha256": adapter_sha256,
        "analysis_code_sha256": analysis_code_sha256,
        "original_claim_sha256": expected_claim_sha256,
        "recovery_authorization_sha256": authorization_sha256,
        "allowed_recovery_runs": 1,
        "mappings_per_adapter_authorized": False,
    }
    immutable_json_write(recovery_claim_path, claim, "qualification recovery claim")
    return {
        "mode": "approved_post_inference_classification_failure_recovery",
        "failure_signature": RECOVERABLE_FAILURE_SIGNATURE,
        "analysis_code_sha256": analysis_code_sha256,
        "original_claim_sha256": expected_claim_sha256,
        "recovery_authorization_sha256": authorization_sha256,
    }


def _validate_trained_artifacts(
    output_dir: Path,
    manifest: dict[str, Any],
    spec: Any,
    identity: dict[str, Any],
    adapter_sha256: str,
) -> None:
    authorization_path = output_dir / "authorization.json"
    training_metadata_path = output_dir / "adapter" / "training_metadata.json"
    authorization = manifest.get("authorization")
    training = manifest.get("training")
    if not isinstance(authorization, dict) or not isinstance(training, dict):
        raise ValueError("CPR authorization or training provenance is missing")
    if authorization.get("sha256") != file_hash(authorization_path):
        raise ValueError("CPR authorization changed after training")
    approved = read_json_object(authorization_path, "CPR authorization")
    if approved.get("preregistration_sha256") != identity["preregistration_sha256"]:
        raise ValueError("CPR authorization does not bind the preregistration")
    metadata = read_json_object(training_metadata_path, "CPR training metadata")
    expected_config = spec.training.to_lora_config(
        data_path=output_dir / "preflight" / "composition_train.jsonl",
        output_dir=output_dir / "adapter",
    ).to_dict()
    if (
        training.get("completed_training_runs") != 1
        or training.get("resumed_from_rr1") is not False
        or training.get("objective") != spec.training.objective
        or training.get("training_metadata_sha256") != file_hash(training_metadata_path)
        or metadata.get("config") != expected_config
        or metadata.get("summary") != training.get("summary")
        or training.get("summary", {}).get("training_data_sha256")
        != identity["data_hashes"]["train"]
        or training.get("summary", {}).get("adapter_sha256") != adapter_sha256
    ):
        raise ValueError("CPR trained artifact provenance differs")


def _progress(message: str) -> None:
    print(f"[p0d2hcpr] {message}", flush=True)
