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
from plasticity_placement.p0d2hrabc.analysis import analyze_counterbalancing
from plasticity_placement.p0d2hrabc.config import AUTHORIZATION_SCOPE, AuditSpec
from plasticity_placement.p0d2hrabc.probes import (
    CounterbalancedProbe,
    compile_counterbalanced_bank,
)
from plasticity_placement.p0d2hrabd.runtime import (
    _source_snapshot as _rab_source_snapshot,
)
from plasticity_placement.p0d2hrabd.runtime import _validate_rab_result
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
    compare_sentinel_scores,
    score_adapter_off_on,
)
from plasticity_placement.p0d2hrsh.runtime import spec_context_model, spec_context_revision
from plasticity_placement.training.model_utils import chat_prompt, load_tokenizer

SCHEMA_VERSION = "p0d2hrabc-audit-v1"
MANIFEST_SCHEMA_VERSION = "p0d2hrabc-manifest-v1"
AUTHORIZATION_SCHEMA_VERSION = "p0d2hrabc-authorization-v1"
ROW_SCHEMA_VERSION = "p0d2hrabc-decision-row-v1"


def plan_audit(*, output_dir: Path, rab_output: Path) -> Path:
    output_dir = output_dir.resolve()
    rab_output = rab_output.resolve()
    _require_independent(rab_output, output_dir)
    if output_dir.exists():
        raise FileExistsError(f"RAB counterbalancing output already exists: {output_dir}")
    spec = AuditSpec()
    source = _validate_rab_result(rab_output)
    source_probes = _load_source_binding_probes(rab_output)
    probes, bank_audit = compile_counterbalanced_bank(source_probes, spec)
    context = source["rsh"]["upstream"]["context"]
    tokenizer = load_tokenizer(
        spec_context_model(context),
        spec_context_revision(context),
    )
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
        "schema_version": "p0d2hrabc-token-audit-v1",
        "decision_count": len(token_records),
        "failed_probe_ids": [
            row["probe_id"] for row in token_records if row["all_candidates_valid"] is not True
        ],
        "records": token_records,
    }
    token_audit["all_checks_passed"] = not token_audit["failed_probe_ids"]
    if token_audit["all_checks_passed"] is not True:
        raise ValueError("RAB counterbalancing candidate-token audit failed")

    output_dir.mkdir(parents=True)
    probe_hash = immutable_write(
        output_dir / "preflight" / "counterbalanced_probes.jsonl",
        _jsonl([probe.to_dict() for probe in probes]),
        "RAB counterbalanced probes",
    )
    bank_hash = immutable_json_write(
        output_dir / "preflight" / "bank_audit.json",
        bank_audit,
        "RAB counterbalanced bank audit",
    )
    token_hash = immutable_json_write(
        output_dir / "preflight" / "token_audit.json",
        token_audit,
        "RAB counterbalancing token audit",
    )
    analysis_plan = {
        "schema_version": "p0d2hrabc-analysis-plan-v1",
        "primary_factors": [
            "receipt_A_vs_B",
            "selected_slot_display_position_first_vs_second",
            "expected_candidate_position_0_vs_1_2_3",
        ],
        "crossing": "canonical/swapped x receipt A/B x display AB/BA x rotations 0/1/2/3",
        "primary_endpoint": "forced-choice correctness with tie-identified bounds",
        "secondary_endpoint": "selected-minus-counterfactual sum-logprob margin",
        "uncertainty": "pair_id cluster bootstrap",
        "interaction_terms": [
            "receipt_x_display_position",
            "receipt_x_candidate_position_0",
            "display_position_x_candidate_position_0",
            "receipt_x_display_position_x_candidate_position_0",
        ],
        "receipt_slot_identifiability_limit": (
            "receipt token and selected slot label remain linked; their joint label effect is "
            "separable from display and candidate position, but not from each other"
        ),
        "allows_training": False,
        "allows_historical_rab_reclassification": False,
        "allows_mappings_per_adapter_scan": False,
    }
    analysis_plan["analysis_plan_sha256"] = json_hash(analysis_plan)
    analysis_plan_hash = immutable_json_write(
        output_dir / "preflight" / "analysis_plan.json",
        analysis_plan,
        "RAB counterbalancing analysis plan",
    )
    identity = {
        "schema_version": "p0d2hrabc-preregistration-v1",
        "code_sha256": current_code_hash(),
        "spec": spec.to_dict(),
        "rab_output": str(rab_output),
        "rab_run_id": source["summary"]["run_id"],
        "rab_summary_sha256": file_hash(rab_output / "summary.json"),
        "adapter_sha256": context["adapter_sha256"],
        "source_snapshot": source["snapshot"],
        "probe_sha256": probe_hash,
        "bank_audit_sha256": bank_hash,
        "token_audit_sha256": token_hash,
        "analysis_plan_sha256": analysis_plan_hash,
        "historical_rab_decision_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    identity["preregistration_sha256"] = json_hash(identity)
    immutable_json_write(
        output_dir / "preregistration.json",
        identity,
        "RAB counterbalancing preregistration",
    )
    immutable_json_write(
        output_dir / "authorization.template.json",
        authorization_template(identity["preregistration_sha256"]),
        "RAB counterbalancing authorization template",
    )
    now = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": "p0d2hrabc-" + identity["preregistration_sha256"][:10],
        "state": "planned",
        "created_at": now,
        "updated_at": now,
        "config": identity,
        "authorization": None,
        "errors": [],
        "historical_rab_decision_changed": False,
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
        "allows_historical_rab_reclassification": False,
        "allows_mappings_per_adapter_scan": False,
    }


def authorize_audit(output_dir: Path, authorization_path: Path) -> Path:
    output_dir = output_dir.resolve()
    authorization_path = authorization_path.resolve()
    manifest = _load_manifest(output_dir)
    if manifest["state"] == "authorized":
        adopted_path = output_dir / "authorization.json"
        external = read_json_object(authorization_path, "external counterbalancing authorization")
        adopted = read_json_object(adopted_path, "adopted counterbalancing authorization")
        if external != adopted or manifest.get("authorization", {}).get("sha256") != file_hash(
            adopted_path
        ):
            raise PermissionError("existing RAB counterbalancing authorization differs")
        return output_dir / "manifest.json"
    if manifest["state"] != "planned":
        raise PermissionError("RAB counterbalancing authorization requires planned state")
    if authorization_path == output_dir or authorization_path.is_relative_to(output_dir):
        raise ValueError("RAB counterbalancing authorization must be external")
    payload = read_json_object(authorization_path, "RAB counterbalancing authorization")
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
        raise PermissionError(f"RAB counterbalancing authorization differs: {differences}")
    if payload.get("approved_by") in {None, "", "TBD"} or payload.get("approved_at") == "TBD":
        raise ValueError("RAB counterbalancing authorization requires approver and timestamp")
    approved_at = datetime.fromisoformat(str(payload["approved_at"]).replace("Z", "+00:00"))
    if approved_at.tzinfo is None:
        raise ValueError("RAB counterbalancing approval timestamp must be timezone-aware")
    digest = immutable_json_write(
        output_dir / "authorization.json", payload, "RAB counterbalancing authorization"
    )
    manifest["state"] = "authorized"
    manifest["authorization"] = {"sha256": digest, "approved_by": payload["approved_by"]}
    _save_manifest(output_dir, manifest)
    return output_dir / "manifest.json"


def run_audit(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    manifest = _load_manifest(output_dir)
    if manifest["state"] != "authorized":
        raise PermissionError("RAB counterbalancing run requires authorized state")
    identity = manifest["config"]
    if current_code_hash() != identity["code_sha256"]:
        raise ValueError("RAB counterbalancing code changed after preregistration")
    _verify_preflight(output_dir, identity)
    _verify_authorization(output_dir, manifest)
    rab_output = Path(identity["rab_output"])
    source = _validate_rab_result(rab_output)
    if source["snapshot"] != identity["source_snapshot"]:
        raise ValueError("RAB source artifacts changed after counterbalancing planning")
    probes = _load_counterbalanced_probes(output_dir)
    audits = _load_token_audits(output_dir)
    spec = AuditSpec()
    upstream = source["rsh"]["upstream"]
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
        activate_adapter(bundle, cpr_output / "adapter", adapter_name="rab_counterbalancing")
        if not hasattr(bundle.model, "disable_adapter"):
            raise RuntimeError("installed PEFT version lacks disable_adapter()")
        bundle.model.requires_grad_(False)
        bundle.model.eval()
        wrapped_id = id(bundle.model.get_base_model())
        if wrapped_id != base_id:
            raise RuntimeError("counterbalancing adapter activation replaced the base-model object")
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
            raise RuntimeError("RAB counterbalancing same-runtime integrity check failed")
    except BaseException as error:
        _mark_failed(output_dir, error)
        raise
    finally:
        if bundle is not None:
            release_model(bundle)

    try:
        raw_hashes = _publish_raw(output_dir, manifest["run_id"], raw_rows)
        analysis, paired_records = analyze_counterbalancing(raw_rows, spec)
        after = _rab_source_snapshot(rab_output)
        if after != identity["source_snapshot"]:
            raise RuntimeError("RAB source artifacts changed during counterbalancing audit")
        summary = {
            "schema_version": SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "analysis_status": analysis["analysis_status"],
            "source": {
                "rab_run_id": source["summary"]["run_id"],
                "rab_summary_sha256": file_hash(rab_output / "summary.json"),
                "historical_rab_decision": source["summary"]["analysis"]["decision"]["status"],
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
            "training_authorized": False,
            "mappings_per_adapter_authorized": False,
            "interpretation_boundary": (
                "This frozen-adapter inference audit separates joint receipt/slot-label, "
                "serial display-position, candidate-position, and interaction signatures. "
                "It cannot reclassify historical RAB, authorize remediation training, or "
                "authorize the 1/4/8 mapping experiment."
            ),
        }
        artifact_hashes = {
            **raw_hashes,
            **_publish_analysis(output_dir, summary, paired_records),
        }
        audit_manifest = {
            "schema_version": "p0d2hrabc-audit-manifest-v1",
            "run_id": manifest["run_id"],
            "preregistration_sha256": identity["preregistration_sha256"],
            "source_snapshot_sha256": json_hash(after),
            "artifacts": artifact_hashes,
            "historical_rab_decision_changed": False,
            "training_authorized": False,
            "mappings_per_adapter_authorized": False,
        }
        audit_manifest_sha256 = immutable_json_write(
            output_dir / "audit_manifest.json",
            audit_manifest,
            "RAB counterbalancing audit manifest",
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
        raise ValueError("RAB counterbalancing manifest is not complete")
    _verify_preflight(output_dir, manifest["config"])
    audit_path = output_dir / "audit_manifest.json"
    if not audit_path.is_file() or file_hash(audit_path) != manifest.get("result", {}).get(
        "audit_manifest_sha256"
    ):
        raise ValueError("RAB counterbalancing audit manifest is missing or changed")
    audit = read_json_object(audit_path, "RAB counterbalancing audit manifest")
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
        raise ValueError(f"RAB counterbalancing published artifacts changed: {differences}")
    if (
        audit.get("schema_version") != "p0d2hrabc-audit-manifest-v1"
        or audit.get("run_id") != manifest.get("run_id")
        or audit.get("preregistration_sha256")
        != manifest.get("config", {}).get("preregistration_sha256")
        or audit.get("training_authorized") is not False
        or audit.get("mappings_per_adapter_authorized") is not False
        or manifest.get("result", {}).get("summary_sha256") != audit["artifacts"]["summary"]
    ):
        raise ValueError("RAB counterbalancing complete manifests disagree")
    checkpoint = read_json_object(
        required["raw_score_checkpoint"], "RAB counterbalancing raw checkpoint"
    )
    off_count = _jsonl_row_count(required["adapter_off"])
    on_count = _jsonl_row_count(required["adapter_on"])
    if (
        checkpoint.get("schema_version") != "p0d2hrabc-raw-score-checkpoint-v1"
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
        raise ValueError("RAB counterbalancing raw checkpoint is inconsistent")
    summary = read_json_object(required["summary"], "RAB counterbalancing summary")
    if (
        summary.get("run_id") != manifest.get("run_id")
        or summary.get("analysis", {}).get("analysis_status")
        != manifest.get("result", {}).get("analysis_status")
        or summary.get("analysis", {}).get("causal_attribution", {}).get("status")
        != manifest.get("result", {}).get("causal_attribution")
        or summary.get("historical_rab_decision_changed") is not False
        or summary.get("training_authorized") is not False
        or summary.get("mappings_per_adapter_authorized") is not False
        or _jsonl_row_count(required["paired_records"]) != AuditSpec().prompt_count
    ):
        raise ValueError("RAB counterbalancing summary or record counts are inconsistent")
    return required["summary"]


def _score_bank(
    bundle: ModelBundle,
    probes: list[CounterbalancedProbe],
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
        if len(rows) % 96 == 0:
            _progress(f"counterbalanced decisions complete: {len(rows)}/{len(probes) * 2}")
    return rows


def _decision_row(
    common: dict[str, Any], outcome: dict[str, Any], state: str, adapter_sha256: str | None
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
            output / "results" / "adapter_off.jsonl", _jsonl(off), "counterbalancing OFF rows"
        ),
        "adapter_on": immutable_write(
            output / "results" / "adapter_on.jsonl", _jsonl(on), "counterbalancing ON rows"
        ),
    }
    checkpoint = {
        "schema_version": "p0d2hrabc-raw-score-checkpoint-v1",
        "run_id": run_id,
        "adapter_off_count": len(off),
        "adapter_on_count": len(on),
        "artifacts": dict(hashes),
        "checkpoint_stage": "inference_complete_before_analysis",
        "automatic_analysis_recovery_allowed": False,
    }
    hashes["raw_score_checkpoint"] = immutable_json_write(
        output / "raw_score_checkpoint.json", checkpoint, "counterbalancing raw checkpoint"
    )
    return hashes


def _publish_analysis(
    output: Path, summary: dict[str, Any], paired_records: list[dict[str, Any]]
) -> dict[str, str]:
    return {
        "summary": immutable_json_write(
            output / "summary.json", summary, "RAB counterbalancing summary"
        ),
        "report": immutable_write(
            output / "report.md", _render_report(summary).encode(), "counterbalancing report"
        ),
        "paired_records": immutable_write(
            output / "paired_records.jsonl",
            _jsonl(paired_records),
            "counterbalancing paired records",
        ),
    }


def _render_report(summary: dict[str, Any]) -> str:
    analysis = summary["analysis"]
    lines = [
        "# Frozen RAB counterbalancing audit",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Status: `{analysis['analysis_status']}`",
        f"- Attribution: `{analysis['causal_attribution']['status']}`",
        "- Historical RAB decision changed: `false`",
        "- Training authorized: `false`",
        "- 1/4/8 authorized: `false`",
        "",
    ]
    if analysis["analysis_status"] != "counterbalancing_complete":
        lines.extend(["Scoring integrity failed; factor effects are not interpreted.", ""])
        return "\n".join(lines)
    lines.extend(
        [
            "## Adapter factor contrasts",
            "",
            "Negative accuracy values mean the named level is disadvantaged.",
            "",
            "| Contrast | Accuracy midpoint (95% pair-cluster CI) | Identified interval | Margin |",
            "|---|---:|---:|---:|",
        ]
    )
    labels = {
        "receipt_B_minus_A": "Receipt B − A",
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
            analysis["causal_attribution"]["receipt_or_slot_label_note"],
            "",
            "This audit does not authorize remediation training or the 1/4/8 mapping experiment.",
            "",
        ]
    )
    return "\n".join(lines)


def _verify_preflight(output: Path, identity: dict[str, Any]) -> None:
    files = {
        "probe_sha256": output / "preflight" / "counterbalanced_probes.jsonl",
        "bank_audit_sha256": output / "preflight" / "bank_audit.json",
        "token_audit_sha256": output / "preflight" / "token_audit.json",
        "analysis_plan_sha256": output / "preflight" / "analysis_plan.json",
    }
    for field, path in files.items():
        if file_hash(path) != identity[field]:
            raise ValueError(f"RAB counterbalancing preflight changed: {field}")
    prereg = read_json_object(
        output / "preregistration.json", "RAB counterbalancing preregistration"
    )
    unhashed = {key: value for key, value in identity.items() if key != "preregistration_sha256"}
    if prereg != identity or json_hash(unhashed) != identity["preregistration_sha256"]:
        raise ValueError("RAB counterbalancing preregistration changed")
    for name in ("bank_audit", "token_audit"):
        payload = read_json_object(output / "preflight" / f"{name}.json", name)
        if payload.get("all_checks_passed") is not True:
            raise ValueError(f"RAB counterbalancing preflight is not qualified: {name}")
    plan = read_json_object(output / "preflight" / "analysis_plan.json", "analysis plan")
    plan_unhashed = {key: value for key, value in plan.items() if key != "analysis_plan_sha256"}
    if json_hash(plan_unhashed) != plan.get("analysis_plan_sha256"):
        raise ValueError("RAB counterbalancing analysis plan self-hash changed")


def _verify_authorization(output: Path, manifest: dict[str, Any]) -> None:
    path = output / "authorization.json"
    authorization = read_json_object(path, "adopted RAB counterbalancing authorization")
    if (
        manifest.get("authorization", {}).get("sha256") != file_hash(path)
        or authorization.get("decision") != "approved"
        or authorization.get("preregistration_sha256")
        != manifest["config"]["preregistration_sha256"]
        or authorization.get("allows_training") is not False
        or authorization.get("allows_historical_rab_reclassification") is not False
        or authorization.get("allows_mappings_per_adapter_scan") is not False
    ):
        raise ValueError("RAB counterbalancing authorization changed after adoption")


def _load_source_binding_probes(rab_output: Path) -> list[BindingProbe]:
    rows = _read_jsonl(rab_output / "preflight" / "binding_probes.jsonl")
    probes = [
        BindingProbe(**{**row, "ordered_candidates": tuple(row["ordered_candidates"])})
        for row in rows
    ]
    if len(probes) != AuditSpec().source_prompt_count:
        raise ValueError("completed RAB source probe bank is incomplete")
    return probes


def _load_counterbalanced_probes(output: Path) -> list[CounterbalancedProbe]:
    rows = _read_jsonl(output / "preflight" / "counterbalanced_probes.jsonl")
    probes = [
        CounterbalancedProbe(**{**row, "ordered_candidates": tuple(row["ordered_candidates"])})
        for row in rows
    ]
    if len(probes) != AuditSpec().prompt_count:
        raise ValueError("RAB counterbalanced probe bank is incomplete")
    return probes


def _load_token_audits(output: Path) -> dict[str, dict[str, Any]]:
    payload = read_json_object(output / "preflight" / "token_audit.json", "token audit")
    result = {str(row["probe_id"]): row for row in payload["records"]}
    if len(result) != AuditSpec().prompt_count or payload.get("all_checks_passed") is not True:
        raise ValueError("RAB counterbalancing token audit is incomplete")
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
    manifest = read_json_object(output / "manifest.json", "RAB counterbalancing manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("not an RAB counterbalancing manifest")
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
        raise ValueError("RAB counterbalancing output must be independent of RAB source")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    return (payload + "\n").encode()


def _jsonl_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def _progress(message: str) -> None:
    print(f"[p0d2hrabc] {message}", flush=True)
