from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
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
from plasticity_placement.p0d2hrab.analysis import analyze_binding, build_binding_records
from plasticity_placement.p0d2hrab.config import AUTHORIZATION_SCOPE, AuditSpec
from plasticity_placement.p0d2hrab.probes import BindingProbe, compile_binding_bank
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
from plasticity_placement.p0d2hrsh.probes import HandoffProbe
from plasticity_placement.p0d2hrsh.runtime import (
    _source_snapshot as _rsh_upstream_snapshot,
)
from plasticity_placement.p0d2hrsh.runtime import (
    _validate_rtb_result,
    spec_context_model,
    spec_context_revision,
)
from plasticity_placement.p0d2hrsh.runtime import (
    _verify_preflight as _verify_rsh_preflight,
)
from plasticity_placement.p0d2hrsh.runtime import (
    verify_complete_result as verify_rsh_result,
)
from plasticity_placement.training.model_utils import chat_prompt, load_tokenizer

SCHEMA_VERSION = "p0d2hrab-audit-v1"
MANIFEST_SCHEMA_VERSION = "p0d2hrab-manifest-v1"
AUTHORIZATION_SCHEMA_VERSION = "p0d2hrab-authorization-v1"
ROW_SCHEMA_VERSION = "p0d2hrab-decision-row-v1"


def plan_audit(*, output_dir: Path, rsh_output: Path) -> Path:
    output_dir = output_dir.resolve()
    rsh_output = rsh_output.resolve()
    _require_independent(rsh_output, output_dir)
    if output_dir.exists():
        raise FileExistsError(f"receipt/action binding output already exists: {output_dir}")
    spec = AuditSpec()
    source = _validate_rsh_result(rsh_output)
    handoff_probes = _load_handoff_probes(rsh_output)
    probes, bank_audit = compile_binding_bank(handoff_probes, spec)
    diagnostic = build_rsh_quadrant_diagnostic(rsh_output)
    if diagnostic["all_checks_passed"] is not True:
        raise ValueError("RSH quadrant diagnostic failed source-integrity checks")

    tokenizer = load_tokenizer(
        spec_context_model(source["upstream"]["context"]),
        spec_context_revision(source["upstream"]["context"]),
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
        record = {
            "probe_id": probe.probe_id,
            "formatted_prompt_sha256": audit["prompt_sha256"],
            **audit,
        }
        record["record_sha256"] = json_hash(record)
        token_records.append(record)
    token_audit = {
        "schema_version": "p0d2hrab-token-audit-v1",
        "decision_count": len(token_records),
        "failed_probe_ids": [
            row["probe_id"] for row in token_records if row["all_candidates_valid"] is not True
        ],
        "records": token_records,
    }
    token_audit["all_checks_passed"] = not token_audit["failed_probe_ids"]
    if token_audit["all_checks_passed"] is not True:
        raise ValueError("receipt/action candidate-token audit failed")

    output_dir.mkdir(parents=True)
    probe_hash = immutable_write(
        output_dir / "preflight" / "binding_probes.jsonl",
        _jsonl([probe.to_dict() for probe in probes]),
        "receipt/action probes",
    )
    bank_hash = immutable_json_write(
        output_dir / "preflight" / "bank_audit.json",
        bank_audit,
        "receipt/action bank audit",
    )
    diagnostic_hash = immutable_json_write(
        output_dir / "preflight" / "rsh_quadrant_diagnostic.json",
        diagnostic,
        "RSH quadrant diagnostic",
    )
    immutable_write(
        output_dir / "preflight" / "rsh_quadrant_diagnostic.md",
        _render_quadrant_report(diagnostic).encode(),
        "RSH quadrant report",
    )
    token_hash = immutable_json_write(
        output_dir / "preflight" / "token_audit.json",
        token_audit,
        "receipt/action token audit",
    )
    identity = {
        "schema_version": "p0d2hrab-preregistration-v1",
        "code_sha256": current_code_hash(),
        "spec": spec.to_dict(),
        "rsh_output": str(rsh_output),
        "rsh_run_id": source["summary"]["run_id"],
        "rsh_summary_sha256": file_hash(rsh_output / "summary.json"),
        "adapter_sha256": source["upstream"]["context"]["adapter_sha256"],
        "source_snapshot": source["snapshot"],
        "probe_sha256": probe_hash,
        "bank_audit_sha256": bank_hash,
        "rsh_quadrant_diagnostic_sha256": diagnostic_hash,
        "token_audit_sha256": token_hash,
        "historical_rsh_decision_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    identity["preregistration_sha256"] = json_hash(identity)
    immutable_json_write(
        output_dir / "preregistration.json", identity, "receipt/action preregistration"
    )
    immutable_json_write(
        output_dir / "authorization.template.json",
        authorization_template(identity["preregistration_sha256"]),
        "receipt/action authorization template",
    )
    now = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": "p0d2hrab-" + identity["preregistration_sha256"][:10],
        "state": "planned",
        "created_at": now,
        "updated_at": now,
        "config": identity,
        "authorization": None,
        "errors": [],
        "historical_rsh_decision_changed": False,
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
        "allows_historical_rsh_reclassification": False,
        "allows_mappings_per_adapter_scan": False,
    }


def authorize_audit(output_dir: Path, authorization_path: Path) -> Path:
    output_dir = output_dir.resolve()
    authorization_path = authorization_path.resolve()
    manifest = _load_manifest(output_dir)
    if manifest["state"] == "authorized":
        adopted_path = output_dir / "authorization.json"
        external = read_json_object(authorization_path, "external receipt/action authorization")
        adopted = read_json_object(adopted_path, "adopted receipt/action authorization")
        if external != adopted or manifest.get("authorization", {}).get("sha256") != file_hash(
            adopted_path
        ):
            raise PermissionError("existing receipt/action authorization differs")
        return output_dir / "manifest.json"
    if manifest["state"] != "planned":
        raise PermissionError("receipt/action authorization requires planned state")
    if authorization_path == output_dir or authorization_path.is_relative_to(output_dir):
        raise ValueError("receipt/action authorization must be external")
    payload = read_json_object(authorization_path, "receipt/action authorization")
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
        raise PermissionError(f"receipt/action authorization differs: {differences}")
    if payload.get("approved_by") in {None, "", "TBD"} or payload.get("approved_at") == "TBD":
        raise ValueError("receipt/action authorization requires approver and timestamp")
    approved_at = datetime.fromisoformat(str(payload["approved_at"]).replace("Z", "+00:00"))
    if approved_at.tzinfo is None:
        raise ValueError("receipt/action approval timestamp must be timezone-aware")
    digest = immutable_json_write(
        output_dir / "authorization.json", payload, "receipt/action authorization"
    )
    manifest["state"] = "authorized"
    manifest["authorization"] = {"sha256": digest, "approved_by": payload["approved_by"]}
    _save_manifest(output_dir, manifest)
    return output_dir / "manifest.json"


def run_audit(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    manifest = _load_manifest(output_dir)
    if manifest["state"] != "authorized":
        raise PermissionError("receipt/action run requires authorized state")
    identity = manifest["config"]
    if current_code_hash() != identity["code_sha256"]:
        raise ValueError("receipt/action code changed after preregistration")
    _verify_preflight(output_dir, identity)
    _verify_authorization(output_dir, manifest)
    rsh_output = Path(identity["rsh_output"])
    source = _validate_rsh_result(rsh_output)
    if source["snapshot"] != identity["source_snapshot"]:
        raise ValueError("receipt/action source artifacts changed after planning")
    probes = _load_binding_probes(output_dir)
    audits = _load_token_audits(output_dir)
    spec = AuditSpec()
    upstream = source["upstream"]
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
        _progress("loading one base-model runtime")
        bundle = load_base_model(load_config)
        _require_source_compatible_bundle(bundle, rr_config)
        base_id = id(bundle.model)
        activate_adapter(bundle, cpr_output / "adapter", adapter_name="receipt_action_binding")
        if not hasattr(bundle.model, "disable_adapter"):
            raise RuntimeError("installed PEFT version lacks disable_adapter()")
        bundle.model.requires_grad_(False)
        bundle.model.eval()
        wrapped_id = id(bundle.model.get_base_model())
        if wrapped_id != base_id:
            raise RuntimeError("receipt/action adapter activation replaced the base-model object")
        peft_id = id(bundle.model)
        first = probes[0]
        first_prompt = chat_prompt(bundle.tokenizer, first.prompt)
        sentinel_before = _score_off(bundle, first_prompt, audits[first.probe_id])
        raw_rows = _score_bank(
            bundle,
            probes,
            audits,
            manifest["run_id"],
            context["adapter_sha256"],
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
            raise RuntimeError("receipt/action same-runtime integrity check failed")
    except BaseException as error:
        _mark_failed(output_dir, error)
        raise
    finally:
        if bundle is not None:
            release_model(bundle)

    try:
        raw_hashes = _publish_raw(output_dir, manifest["run_id"], raw_rows)
        analysis = analyze_binding(raw_rows, spec)
        paired_records, _ = build_binding_records(raw_rows, spec)
        after = _source_snapshot(rsh_output, source["upstream"])
        if after != identity["source_snapshot"]:
            raise RuntimeError("receipt/action source artifacts changed during audit")
        summary = {
            "schema_version": SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "analysis_status": "post_rsh_receipt_action_binding_complete",
            "source": {
                "rsh_run_id": source["summary"]["run_id"],
                "rsh_summary_sha256": file_hash(rsh_output / "summary.json"),
                "cpr_run_id": context["cpr_manifest"]["run_id"],
                "adapter_sha256": context["adapter_sha256"],
            },
            "runtime_identity": runtime_identity,
            "raw_score_checkpoint": raw_hashes,
            "environment": current_environment_snapshot(),
            "rsh_quadrant_diagnostic": read_json_object(
                output_dir / "preflight" / "rsh_quadrant_diagnostic.json",
                "RSH quadrant diagnostic",
            ),
            "analysis": analysis,
            "source_snapshot_before": identity["source_snapshot"],
            "source_snapshot_after": after,
            "source_artifacts_modified": False,
            "historical_rsh_decision_changed": False,
            "training_authorized": False,
            "mappings_per_adapter_authorized": False,
            "interpretation_boundary": (
                "This inference-only audit tests causal receipt/action binding for one frozen "
                "adapter. It cannot reclassify RSH, authorize training, or authorize 1/4/8."
            ),
        }
        artifact_hashes = {**raw_hashes, **_publish_analysis(output_dir, summary, paired_records)}
        audit_manifest = {
            "schema_version": "p0d2hrab-audit-manifest-v1",
            "run_id": manifest["run_id"],
            "preregistration_sha256": identity["preregistration_sha256"],
            "source_snapshot_sha256": json_hash(after),
            "artifacts": artifact_hashes,
            "historical_rsh_decision_changed": False,
            "training_authorized": False,
            "mappings_per_adapter_authorized": False,
        }
        audit_manifest_sha256 = immutable_json_write(
            output_dir / "audit_manifest.json", audit_manifest, "receipt/action audit manifest"
        )
        manifest = _load_manifest(output_dir)
        manifest["state"] = "complete"
        manifest["result"] = {
            "summary_sha256": artifact_hashes["summary"],
            "audit_manifest_sha256": audit_manifest_sha256,
            "decision": analysis["decision"]["status"],
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
        raise ValueError("receipt/action manifest is not complete")
    audit_path = output_dir / "audit_manifest.json"
    if not audit_path.is_file() or file_hash(audit_path) != manifest.get("result", {}).get(
        "audit_manifest_sha256"
    ):
        raise ValueError("receipt/action audit manifest is missing or changed")
    audit = read_json_object(audit_path, "receipt/action audit manifest")
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
        raise ValueError(f"receipt/action published artifacts changed: {differences}")
    if (
        audit.get("schema_version") != "p0d2hrab-audit-manifest-v1"
        or audit.get("run_id") != manifest.get("run_id")
        or audit.get("preregistration_sha256")
        != manifest.get("config", {}).get("preregistration_sha256")
        or audit.get("training_authorized") is not False
        or audit.get("mappings_per_adapter_authorized") is not False
        or manifest.get("result", {}).get("summary_sha256") != audit["artifacts"]["summary"]
    ):
        raise ValueError("receipt/action complete manifests disagree")
    checkpoint = read_json_object(
        output_dir / "raw_score_checkpoint.json", "receipt/action raw checkpoint"
    )
    off_count = _jsonl_row_count(required["adapter_off"])
    on_count = _jsonl_row_count(required["adapter_on"])
    if (
        checkpoint.get("schema_version") != "p0d2hrab-raw-score-checkpoint-v1"
        or checkpoint.get("run_id") != manifest.get("run_id")
        or checkpoint.get("adapter_off_count") != off_count
        or checkpoint.get("adapter_on_count") != on_count
        or off_count != 192
        or on_count != 192
        or checkpoint.get("artifacts")
        != {
            "adapter_off": audit["artifacts"]["adapter_off"],
            "adapter_on": audit["artifacts"]["adapter_on"],
        }
    ):
        raise ValueError("receipt/action raw checkpoint is inconsistent")
    summary = read_json_object(required["summary"], "receipt/action summary")
    if (
        summary.get("run_id") != manifest.get("run_id")
        or summary.get("historical_rsh_decision_changed") is not False
        or summary.get("training_authorized") is not False
        or summary.get("mappings_per_adapter_authorized") is not False
        or summary.get("analysis", {}).get("decision", {}).get("status")
        != manifest.get("result", {}).get("decision")
        or _jsonl_row_count(required["paired_records"]) != 192
    ):
        raise ValueError("receipt/action summary or paired records are inconsistent")
    return required["summary"]


def build_rsh_quadrant_diagnostic(rsh_output: Path) -> dict[str, Any]:
    _validate_rsh_result(rsh_output)
    records = _read_jsonl(rsh_output / "paired_records.jsonl")
    checks = {
        "record_count": len(records) == 96,
        "unique_probe_ids": len({row.get("probe_id") for row in records}) == 96,
        "states_complete": all("base" in row and "adapter" in row for row in records),
    }
    states: dict[str, Any] = {}
    for state in ("base", "adapter"):
        counts = Counter()
        tied_or_bounded = 0
        for row in records:
            values = row[state]
            oracle = bool(values["oracle_correct"])
            wrong = bool(values["wrong_target_selected"])
            if oracle and wrong:
                counts["both_target"] += 1
            elif oracle:
                counts["oracle_only_target"] += 1
            elif wrong:
                counts["wrong_only_target"] += 1
            else:
                counts["neither_target"] += 1
            tied_or_bounded += int(
                values["oracle_lower"] != values["oracle_upper"]
                or values["wrong_target_lower"] != values["wrong_target_upper"]
            )
        states[state] = {
            "counts": {
                name: counts[name]
                for name in (
                    "both_target",
                    "oracle_only_target",
                    "wrong_only_target",
                    "neither_target",
                )
            },
            "rates": {
                name: counts[name] / len(records)
                for name in (
                    "both_target",
                    "oracle_only_target",
                    "wrong_only_target",
                    "neither_target",
                )
            },
            "tie_affected_record_count": tied_or_bounded,
            "oracle_accuracy_bounds": [
                mean(float(row[state]["oracle_lower"]) for row in records),
                mean(float(row[state]["oracle_upper"]) for row in records),
            ],
            "wrong_target_rate_bounds": [
                mean(float(row[state]["wrong_target_lower"]) for row in records),
                mean(float(row[state]["wrong_target_upper"]) for row in records),
            ],
            "specificity_bounds": [
                mean(
                    float(row[state]["oracle_lower"]) - float(row[state]["wrong_target_upper"])
                    for row in records
                ),
                mean(
                    float(row[state]["oracle_upper"]) - float(row[state]["wrong_target_lower"])
                    for row in records
                ),
            ],
        }
    checks["quadrants_partition_each_state"] = all(
        sum(payload["counts"].values()) == 96 for payload in states.values()
    )
    return {
        "schema_version": "p0d2hrab-rsh-quadrant-diagnostic-v1",
        "rsh_summary_sha256": file_hash(rsh_output / "summary.json"),
        "rsh_paired_records_sha256": file_hash(rsh_output / "paired_records.jsonl"),
        "state_results": states,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "descriptive_only": True,
        "historical_rsh_decision_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }


def _validate_rsh_result(rsh_output: Path) -> dict[str, Any]:
    verify_rsh_result(rsh_output)
    manifest = read_json_object(rsh_output / "manifest.json", "RSH manifest")
    identity = read_json_object(rsh_output / "preregistration.json", "RSH preregistration")
    summary = read_json_object(rsh_output / "summary.json", "RSH summary")
    _verify_rsh_preflight(rsh_output, identity)
    decision = summary.get("analysis", {}).get("decision", {})
    primary = summary.get("analysis", {}).get("primary_handoff_rescue", {})
    safeguards = summary.get("analysis", {}).get("safeguards", {})
    if (
        manifest.get("config") != identity
        or decision.get("status") != "handoff_not_supported"
        or decision.get("scoring_integrity") is not True
        or primary.get("supported") is not True
        or safeguards.get("adapter_on_chain_minus_oracle_conservative", {}).get("passed")
        is not True
        or safeguards.get("adapter_on_oracle_minus_wrong_target_conservative", {}).get("passed")
        is not False
        or summary.get("historical_rtb_decision_changed") is not False
        or summary.get("training_authorized") is not False
        or summary.get("mappings_per_adapter_authorized") is not False
    ):
        raise ValueError("receipt/action source is not the reviewed completed RSH result")
    rtb_output = Path(identity["rtb_output"])
    upstream = _validate_rtb_result(rtb_output)
    current_upstream_snapshot = _rsh_upstream_snapshot(rtb_output, upstream["rtb_identity"])
    if summary.get("source_snapshot_after") != current_upstream_snapshot:
        raise ValueError("RSH upstream RTB/CPR artifacts changed")
    return {
        "manifest": manifest,
        "identity": identity,
        "summary": summary,
        "upstream": upstream,
        "snapshot": _source_snapshot(rsh_output, upstream),
    }


def _source_snapshot(rsh_output: Path, upstream: dict[str, Any]) -> dict[str, Any]:
    configured_rtb = Path(
        read_json_object(rsh_output / "preregistration.json", "RSH preregistration")["rtb_output"]
    )
    return {
        "rsh_output_tree": _tree_hash(rsh_output),
        "rsh_bound_upstream": _rsh_upstream_snapshot(configured_rtb, upstream["rtb_identity"]),
    }


def _score_bank(
    bundle: ModelBundle,
    probes: list[BindingProbe],
    audits: dict[str, dict[str, Any]],
    run_id: str,
    adapter_sha256: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for probe in probes:
        formatted = chat_prompt(bundle.tokenizer, probe.prompt)
        off, on = score_adapter_off_on(
            bundle,
            formatted,
            audits[probe.probe_id],
            score_candidate_batch,
        )
        common = {
            "schema_version": ROW_SCHEMA_VERSION,
            "run_id": run_id,
            "probe_id": probe.probe_id,
            "unit_id": probe.unit_id,
            "pair_id": probe.pair_id,
            "route_variant": probe.route_variant,
            "orientation": probe.orientation,
            "receipt": probe.receipt,
            "cell": probe.cell,
            "lesson_a_id": probe.lesson_a_id,
            "lesson_b_id": probe.lesson_b_id,
            "slot_a_lesson_id": probe.slot_a_lesson_id,
            "slot_b_lesson_id": probe.slot_b_lesson_id,
            "slot_a_action": probe.slot_a_action,
            "slot_b_action": probe.slot_b_action,
            "expected_action": probe.expected_action,
            "counterfactual_action": probe.counterfactual_action,
            "ordered_candidates": list(probe.ordered_candidates),
            "formatted_prompt_sha256": audits[probe.probe_id]["prompt_sha256"],
        }
        rows.append(_decision_row(common, off, "adapter_off", None))
        rows.append(_decision_row(common, on, "adapter_on", adapter_sha256))
        if len(rows) % 96 == 0:
            _progress(f"binding decisions complete: {len(rows)}/{len(probes) * 2}")
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
            output / "results" / "adapter_off.jsonl", _jsonl(off), "receipt/action OFF rows"
        ),
        "adapter_on": immutable_write(
            output / "results" / "adapter_on.jsonl", _jsonl(on), "receipt/action ON rows"
        ),
    }
    checkpoint = {
        "schema_version": "p0d2hrab-raw-score-checkpoint-v1",
        "run_id": run_id,
        "adapter_off_count": len(off),
        "adapter_on_count": len(on),
        "artifacts": dict(hashes),
        "checkpoint_stage": "inference_complete_before_analysis",
        "automatic_analysis_recovery_allowed": False,
    }
    hashes["raw_score_checkpoint"] = immutable_json_write(
        output / "raw_score_checkpoint.json", checkpoint, "receipt/action raw checkpoint"
    )
    return hashes


def _publish_analysis(
    output: Path, summary: dict[str, Any], paired_records: list[dict[str, Any]]
) -> dict[str, str]:
    return {
        "summary": immutable_json_write(output / "summary.json", summary, "receipt/action summary"),
        "report": immutable_write(
            output / "report.md", _render_report(summary).encode(), "receipt/action report"
        ),
        "paired_records": immutable_write(
            output / "paired_records.jsonl",
            _jsonl(paired_records),
            "receipt/action paired records",
        ),
    }


def _render_report(summary: dict[str, Any]) -> str:
    analysis = summary["analysis"]
    endpoints = analysis["endpoint_summaries"]
    lines = [
        "# CPR receipt/action binding audit",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Decision: `{analysis['decision']['status']}`",
        "- Historical RSH decision changed: `false`",
        "- Training authorized: `false`",
        "- 1/4/8 authorized: `false`",
        "",
        "| Endpoint | OFF | ON | ON-OFF (95% pair-cluster CI) |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "selected_correct": "Selected-binding accuracy",
        "counterfactual_selected": "Counterfactual-action rate",
        "specificity": "Selected-counterfactual specificity",
    }
    for key, label in labels.items():
        values = endpoints[key]
        difference = values["adapter_minus_base"]
        lines.append(
            f"| {label} | {values['base']:.4f} | {values['adapter']:.4f} | "
            f"{difference['estimate']:.4f} "
            f"[{difference['ci95'][0]:.4f}, {difference['ci95'][1]:.4f}] |"
        )
    binding = analysis["adapter_on_conservative_binding_accuracy"]
    specificity = analysis["adapter_on_conservative_specificity"]
    orientations = analysis["orientation_conservative_accuracy"]
    noninferiority = analysis["adapter_on_minus_off_noninferiority"]
    lines.extend(
        [
            "",
            "## Conservative gates",
            "",
            f"- Binding accuracy: {binding['estimate']:.4f} "
            f"[{binding['ci95'][0]:.4f}, {binding['ci95'][1]:.4f}], "
            f"passed=`{str(binding['passed']).lower()}`.",
            f"- Causal specificity: {specificity['estimate']:.4f} "
            f"[{specificity['ci95'][0]:.4f}, {specificity['ci95'][1]:.4f}], "
            f"passed=`{str(specificity['passed']).lower()}`.",
            f"- Canonical orientation accuracy: {orientations['canonical']['estimate']:.4f} "
            f"[{orientations['canonical']['ci95'][0]:.4f}, "
            f"{orientations['canonical']['ci95'][1]:.4f}], "
            f"passed=`{str(orientations['canonical']['passed']).lower()}`.",
            f"- Swapped orientation accuracy: {orientations['swapped']['estimate']:.4f} "
            f"[{orientations['swapped']['ci95'][0]:.4f}, "
            f"{orientations['swapped']['ci95'][1]:.4f}], "
            f"passed=`{str(orientations['swapped']['passed']).lower()}`.",
            f"- Adapter ON-OFF noninferiority: {noninferiority['estimate']:.4f} "
            f"[{noninferiority['ci95'][0]:.4f}, {noninferiority['ci95'][1]:.4f}], "
            f"passed=`{str(noninferiority['passed']).lower()}`.",
            "",
            "This inference-only audit cannot reclassify RSH, authorize training, or authorize "
            "the 1/4/8 experiment.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_quadrant_report(diagnostic: dict[str, Any]) -> str:
    lines = [
        "# RSH oracle/wrong-receipt paired diagnostic",
        "",
        "This table is descriptive only and does not change the RSH decision.",
        "",
        "| State | Both target | Oracle only | Wrong only | Neither | Tie-affected |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for state in ("base", "adapter"):
        values = diagnostic["state_results"][state]
        counts = values["counts"]
        lines.append(
            f"| {state} | {counts['both_target']} | {counts['oracle_only_target']} | "
            f"{counts['wrong_only_target']} | {counts['neither_target']} | "
            f"{values['tie_affected_record_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _verify_preflight(output: Path, identity: dict[str, Any]) -> None:
    files = {
        "probe_sha256": output / "preflight" / "binding_probes.jsonl",
        "bank_audit_sha256": output / "preflight" / "bank_audit.json",
        "rsh_quadrant_diagnostic_sha256": output / "preflight" / "rsh_quadrant_diagnostic.json",
        "token_audit_sha256": output / "preflight" / "token_audit.json",
    }
    for field, path in files.items():
        if file_hash(path) != identity[field]:
            raise ValueError(f"receipt/action preflight changed: {field}")
    prereg = read_json_object(output / "preregistration.json", "receipt/action preregistration")
    unhashed = {key: value for key, value in identity.items() if key != "preregistration_sha256"}
    if prereg != identity or json_hash(unhashed) != identity["preregistration_sha256"]:
        raise ValueError("receipt/action preregistration changed")
    for name in ("bank_audit", "rsh_quadrant_diagnostic", "token_audit"):
        payload = read_json_object(output / "preflight" / f"{name}.json", name)
        if payload.get("all_checks_passed") is not True:
            raise ValueError(f"receipt/action preflight is not qualified: {name}")


def _verify_authorization(output: Path, manifest: dict[str, Any]) -> None:
    path = output / "authorization.json"
    authorization = read_json_object(path, "adopted receipt/action authorization")
    if (
        manifest.get("authorization", {}).get("sha256") != file_hash(path)
        or authorization.get("decision") != "approved"
        or authorization.get("preregistration_sha256")
        != manifest["config"]["preregistration_sha256"]
        or authorization.get("allows_training") is not False
        or authorization.get("allows_historical_rsh_reclassification") is not False
        or authorization.get("allows_mappings_per_adapter_scan") is not False
    ):
        raise ValueError("receipt/action authorization changed after adoption")


def _load_handoff_probes(rsh_output: Path) -> list[HandoffProbe]:
    rows = _read_jsonl(rsh_output / "preflight" / "handoff_probes.jsonl")
    return [
        HandoffProbe(
            **{
                **row,
                "slot_candidates": tuple(row["slot_candidates"]),
                "action_candidates": tuple(row["action_candidates"]),
            }
        )
        for row in rows
    ]


def _load_binding_probes(output: Path) -> list[BindingProbe]:
    rows = _read_jsonl(output / "preflight" / "binding_probes.jsonl")
    return [
        BindingProbe(**{**row, "ordered_candidates": tuple(row["ordered_candidates"])})
        for row in rows
    ]


def _load_token_audits(output: Path) -> dict[str, dict[str, Any]]:
    payload = read_json_object(output / "preflight" / "token_audit.json", "token audit")
    result = {str(row["probe_id"]): row for row in payload["records"]}
    if len(result) != 192 or payload.get("all_checks_passed") is not True:
        raise ValueError("receipt/action token audit is incomplete")
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
    manifest = read_json_object(output / "manifest.json", "receipt/action manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("not a receipt/action binding manifest")
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
        raise ValueError("receipt/action output must be independent of RSH source")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    return (payload + "\n").encode()


def _jsonl_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def _progress(message: str) -> None:
    print(f"[p0d2hrab] {message}", flush=True)
