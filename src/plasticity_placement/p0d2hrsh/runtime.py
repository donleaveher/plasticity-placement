from __future__ import annotations

import json
from collections import Counter
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
from plasticity_placement.p0d2hrsh.analysis import analyze_handoff, build_handoff_records
from plasticity_placement.p0d2hrsh.config import (
    AUTHORIZATION_SCOPE,
    PROMPT_KINDS,
    AuditSpec,
)
from plasticity_placement.p0d2hrsh.probes import HandoffProbe, compile_handoff_bank
from plasticity_placement.p0d2hrtb.probes import BridgeProbe
from plasticity_placement.p0d2hrtb.runtime import (
    _source_snapshot as _rtb_source_snapshot,
)
from plasticity_placement.p0d2hrtb.runtime import (
    _validate_sources as _validate_rtb_sources,
)
from plasticity_placement.p0d2hrtb.runtime import _verify_preflight as _verify_rtb_preflight
from plasticity_placement.p0d2hrtb.runtime import (
    spec_context_model,
    spec_context_revision,
)
from plasticity_placement.training.model_utils import chat_prompt, load_tokenizer

SCHEMA_VERSION = "p0d2hrsh-audit-v1"
MANIFEST_SCHEMA_VERSION = "p0d2hrsh-manifest-v1"
AUTHORIZATION_SCHEMA_VERSION = "p0d2hrsh-authorization-v1"
ROW_SCHEMA_VERSION = "p0d2hrsh-decision-row-v1"


def plan_audit(*, output_dir: Path, rtb_output: Path) -> Path:
    output_dir = output_dir.resolve()
    rtb_output = rtb_output.resolve()
    _require_independent(rtb_output, output_dir)
    if output_dir.exists():
        raise FileExistsError(f"route-state handoff output already exists: {output_dir}")
    spec = AuditSpec()
    source = _validate_rtb_result(rtb_output)
    bridge_probes = _load_bridge_probes(rtb_output)
    probes, bank_audit = compile_handoff_bank(bridge_probes, spec)
    replay_audit = _build_replay_audit(rtb_output)
    if replay_audit["all_checks_passed"] is not True:
        raise ValueError("route-state handoff replay qualification failed")

    tokenizer = load_tokenizer(
        spec_context_model(source["context"]),
        spec_context_revision(source["context"]),
    )
    token_records: list[dict[str, Any]] = []
    for probe in probes:
        for kind in PROMPT_KINDS:
            formatted = chat_prompt(tokenizer, probe.prompt(kind))
            audit = audit_candidate_tokenization(
                tokenizer,
                formatted,
                probe.candidates(kind),
                evaluation_max_length=512,
            )
            record = {
                "probe_id": probe.probe_id,
                "prompt_kind": kind,
                "formatted_prompt_sha256": audit["prompt_sha256"],
                **audit,
            }
            record["record_sha256"] = json_hash(record)
            token_records.append(record)
    token_audit = {
        "schema_version": "p0d2hrsh-token-audit-v1",
        "decision_count": len(token_records),
        "failed_keys": [
            [row["probe_id"], row["prompt_kind"]]
            for row in token_records
            if row["all_candidates_valid"] is not True
        ],
        "records": token_records,
    }
    token_audit["all_checks_passed"] = not token_audit["failed_keys"]
    if token_audit["all_checks_passed"] is not True:
        raise ValueError("route-state handoff candidate-token audit failed")
    anchor_audit = _audit_handoff_anchors(
        probes,
        token_records,
        _read_jsonl(rtb_output / "paired_records.jsonl"),
    )
    if anchor_audit["all_checks_passed"] is not True:
        raise ValueError("route-state handoff prompts differ from bound RTB paired rows")
    bank_audit["rtb_paired_anchor_audit"] = anchor_audit
    bank_audit["all_checks_passed"] = (
        bank_audit["all_checks_passed"] and anchor_audit["all_checks_passed"]
    )

    output_dir.mkdir(parents=True)
    probe_hash = immutable_write(
        output_dir / "preflight" / "handoff_probes.jsonl",
        _jsonl([probe.to_dict() for probe in probes]),
        "handoff probes",
    )
    bank_hash = immutable_json_write(
        output_dir / "preflight" / "bank_audit.json", bank_audit, "handoff bank audit"
    )
    replay_hash = immutable_json_write(
        output_dir / "preflight" / "replay_audit.json", replay_audit, "handoff replay audit"
    )
    token_hash = immutable_json_write(
        output_dir / "preflight" / "token_audit.json", token_audit, "handoff token audit"
    )
    identity = {
        "schema_version": "p0d2hrsh-preregistration-v1",
        "code_sha256": current_code_hash(),
        "spec": spec.to_dict(),
        "rtb_output": str(rtb_output),
        "rtb_run_id": source["summary"]["run_id"],
        "rtb_summary_sha256": file_hash(rtb_output / "summary.json"),
        "adapter_sha256": source["context"]["adapter_sha256"],
        "source_snapshot": source["snapshot"],
        "probe_sha256": probe_hash,
        "bank_audit_sha256": bank_hash,
        "replay_audit_sha256": replay_hash,
        "token_audit_sha256": token_hash,
        "historical_rtb_decision_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    identity["preregistration_sha256"] = json_hash(identity)
    immutable_json_write(output_dir / "preregistration.json", identity, "handoff preregistration")
    immutable_json_write(
        output_dir / "authorization.template.json",
        authorization_template(identity["preregistration_sha256"]),
        "handoff authorization template",
    )
    now = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": "p0d2hrsh-" + identity["preregistration_sha256"][:10],
        "state": "planned",
        "created_at": now,
        "updated_at": now,
        "config": identity,
        "authorization": None,
        "errors": [],
        "historical_rtb_decision_changed": False,
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
    if manifest["state"] == "authorized":
        adopted_path = output_dir / "authorization.json"
        external = read_json_object(authorization_path, "external handoff authorization")
        adopted = read_json_object(adopted_path, "adopted handoff authorization")
        if (
            external != adopted
            or manifest.get("authorization", {}).get("sha256") != file_hash(adopted_path)
        ):
            raise PermissionError("existing route-state handoff authorization differs")
        return output_dir / "manifest.json"
    if manifest["state"] != "planned":
        raise PermissionError("route-state handoff authorization requires planned state")
    if authorization_path == output_dir or authorization_path.is_relative_to(output_dir):
        raise ValueError("route-state handoff authorization must be external")
    payload = read_json_object(authorization_path, "route-state handoff authorization")
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
        raise PermissionError(f"route-state handoff authorization differs: {differences}")
    if payload.get("approved_by") in {None, "", "TBD"} or payload.get("approved_at") == "TBD":
        raise ValueError("route-state handoff authorization requires approver and timestamp")
    approved_at = datetime.fromisoformat(str(payload["approved_at"]).replace("Z", "+00:00"))
    if approved_at.tzinfo is None:
        raise ValueError("route-state handoff approval timestamp must be timezone-aware")
    digest = immutable_json_write(
        output_dir / "authorization.json", payload, "route-state handoff authorization"
    )
    manifest["state"] = "authorized"
    manifest["authorization"] = {"sha256": digest, "approved_by": payload["approved_by"]}
    _save_manifest(output_dir, manifest)
    return output_dir / "manifest.json"


def run_audit(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    manifest = _load_manifest(output_dir)
    if manifest["state"] != "authorized":
        raise PermissionError("route-state handoff run requires authorized state")
    identity = manifest["config"]
    if current_code_hash() != identity["code_sha256"]:
        raise ValueError("route-state handoff code changed after preregistration")
    _verify_preflight(output_dir, identity)
    _verify_authorization(output_dir, manifest)
    rtb_output = Path(identity["rtb_output"])
    source = _validate_rtb_result(rtb_output)
    if source["snapshot"] != identity["source_snapshot"]:
        raise ValueError("route-state handoff source artifacts changed after planning")
    probes = _load_handoff_probes(output_dir)
    audits = _load_token_audits(output_dir)
    spec = AuditSpec()
    context = source["context"]
    cpr_output = Path(source["rtb_identity"]["cpr_output"])
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
        activate_adapter(bundle, cpr_output / "adapter", adapter_name="route_state_handoff")
        if not hasattr(bundle.model, "disable_adapter"):
            raise RuntimeError("installed PEFT version lacks disable_adapter()")
        bundle.model.requires_grad_(False)
        bundle.model.eval()
        wrapped_id = id(bundle.model.get_base_model())
        if wrapped_id != base_id:
            raise RuntimeError("handoff adapter activation replaced the base-model object")
        peft_id = id(bundle.model)
        first = probes[0]
        first_prompt = chat_prompt(bundle.tokenizer, first.slot_prompt)
        sentinel_before = _score_off(bundle, first_prompt, audits[(first.probe_id, "slot_readout")])
        raw_rows = _score_bank(
            bundle,
            probes,
            audits,
            manifest["run_id"],
            context["adapter_sha256"],
        )
        sentinel_after = _score_off(bundle, first_prompt, audits[(first.probe_id, "slot_readout")])
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
            raise RuntimeError("route-state handoff same-runtime integrity check failed")
    except BaseException as error:
        _mark_failed(output_dir, error)
        raise
    finally:
        if bundle is not None:
            release_model(bundle)

    try:
        raw_hashes = _publish_raw(output_dir, manifest["run_id"], raw_rows)
        analysis = analyze_handoff(raw_rows, spec)
        paired_records, _ = build_handoff_records(raw_rows, spec)
        after = _source_snapshot(rtb_output, source["rtb_identity"])
        if after != identity["source_snapshot"]:
            raise RuntimeError("route-state handoff source artifacts changed during audit")
        summary = {
            "schema_version": SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "analysis_status": "post_failure_route_state_handoff_complete",
            "source": {
                "rtb_run_id": source["summary"]["run_id"],
                "rtb_summary_sha256": file_hash(rtb_output / "summary.json"),
                "cpr_run_id": context["cpr_manifest"]["run_id"],
                "adapter_sha256": context["adapter_sha256"],
            },
            "runtime_identity": runtime_identity,
            "raw_score_checkpoint": raw_hashes,
            "environment": current_environment_snapshot(),
            "analysis": analysis,
            "source_snapshot_before": identity["source_snapshot"],
            "source_snapshot_after": after,
            "source_artifacts_modified": False,
            "historical_rtb_decision_changed": False,
            "training_authorized": False,
            "mappings_per_adapter_authorized": False,
            "interpretation_boundary": (
                "This inference-only audit tests route-state handoff for one frozen adapter. "
                "It cannot reclassify RTB, authorize training, or authorize 1/4/8."
            ),
        }
        artifact_hashes = _publish_analysis(output_dir, summary, paired_records)
        artifact_hashes = {**raw_hashes, **artifact_hashes}
        audit_manifest = {
            "schema_version": "p0d2hrsh-audit-manifest-v1",
            "run_id": manifest["run_id"],
            "preregistration_sha256": identity["preregistration_sha256"],
            "source_snapshot_sha256": json_hash(after),
            "artifacts": artifact_hashes,
            "historical_rtb_decision_changed": False,
            "training_authorized": False,
            "mappings_per_adapter_authorized": False,
        }
        audit_manifest_sha256 = immutable_json_write(
            output_dir / "audit_manifest.json", audit_manifest, "handoff manifest"
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
    """Verify every published artifact before a completed audit is displayed or reused."""
    output_dir = output_dir.resolve()
    manifest = _load_manifest(output_dir)
    if manifest.get("state") != "complete":
        raise ValueError("route-state handoff manifest is not complete")
    audit_path = output_dir / "audit_manifest.json"
    expected_audit_hash = manifest.get("result", {}).get("audit_manifest_sha256")
    if not audit_path.is_file() or file_hash(audit_path) != expected_audit_hash:
        raise ValueError("route-state handoff audit manifest is missing or changed")
    audit = read_json_object(audit_path, "handoff audit manifest")
    required = {
        "summary": output_dir / "summary.json",
        "report": output_dir / "report.md",
        "paired_records": output_dir / "paired_records.jsonl",
        "adapter_off": output_dir / "results" / "adapter_off.jsonl",
        "adapter_on": output_dir / "results" / "adapter_on.jsonl",
        "raw_score_checkpoint": output_dir / "raw_score_checkpoint.json",
    }
    differences: dict[str, dict[str, str | None]] = {}
    for name, path in required.items():
        expected = audit.get("artifacts", {}).get(name)
        observed = file_hash(path) if path.is_file() else None
        if observed != expected:
            differences[name] = {"expected": expected, "observed": observed}
    if differences:
        raise ValueError(f"route-state handoff published artifacts changed: {differences}")
    if (
        audit.get("schema_version") != "p0d2hrsh-audit-manifest-v1"
        or audit.get("run_id") != manifest.get("run_id")
        or audit.get("preregistration_sha256")
        != manifest.get("config", {}).get("preregistration_sha256")
        or audit.get("training_authorized") is not False
        or audit.get("mappings_per_adapter_authorized") is not False
        or manifest.get("result", {}).get("summary_sha256")
        != audit.get("artifacts", {}).get("summary")
    ):
        raise ValueError("route-state handoff complete manifests disagree")
    checkpoint = read_json_object(
        output_dir / "raw_score_checkpoint.json", "handoff raw checkpoint"
    )
    off_count = _jsonl_row_count(required["adapter_off"])
    on_count = _jsonl_row_count(required["adapter_on"])
    if (
        checkpoint.get("schema_version") != "p0d2hrsh-raw-score-checkpoint-v1"
        or checkpoint.get("run_id") != manifest.get("run_id")
        or checkpoint.get("adapter_off_count") != off_count
        or checkpoint.get("adapter_on_count") != on_count
        or off_count != 384
        or on_count != 384
        or checkpoint.get("artifacts")
        != {
            "adapter_off": audit["artifacts"]["adapter_off"],
            "adapter_on": audit["artifacts"]["adapter_on"],
        }
    ):
        raise ValueError("route-state handoff raw checkpoint is inconsistent")
    summary = read_json_object(required["summary"], "handoff summary")
    if (
        summary.get("run_id") != manifest.get("run_id")
        or summary.get("training_authorized") is not False
        or summary.get("mappings_per_adapter_authorized") is not False
        or summary.get("historical_rtb_decision_changed") is not False
        or summary.get("analysis", {}).get("decision", {}).get("status")
        != manifest.get("result", {}).get("decision")
        or _jsonl_row_count(required["paired_records"]) != 96
    ):
        raise ValueError("route-state handoff summary or paired records are inconsistent")
    return required["summary"]


def _validate_rtb_result(rtb_output: Path) -> dict[str, Any]:
    manifest = read_json_object(rtb_output / "manifest.json", "RTB manifest")
    identity = read_json_object(rtb_output / "preregistration.json", "RTB preregistration")
    summary = read_json_object(rtb_output / "summary.json", "RTB summary")
    audit = read_json_object(rtb_output / "audit_manifest.json", "RTB audit manifest")
    _verify_rtb_preflight(rtb_output, identity)
    if (
        manifest.get("state") != "complete"
        or manifest.get("config") != identity
        or manifest.get("run_id") != summary.get("run_id")
        or audit.get("run_id") != summary.get("run_id")
        or audit.get("preregistration_sha256") != identity.get("preregistration_sha256")
        or audit.get("training_authorized") is not False
        or audit.get("mappings_per_adapter_authorized") is not False
        or manifest.get("result", {}).get("summary_sha256")
        != file_hash(rtb_output / "summary.json")
        or summary.get("analysis", {}).get("decision", {}).get("status")
        != "scoring_integrity_failed"
        or summary.get("analysis", {})
        .get("decision", {})
        .get("primary_factorial_zero_tie_integrity")
        is not False
        or summary.get("analysis", {})
        .get("decision", {})
        .get("action_expected_compatible_tie_integrity")
        is not True
        or summary.get("training_authorized") is not False
        or summary.get("mappings_per_adapter_authorized") is not False
    ):
        raise ValueError("route-state handoff source is not the reviewed failed RTB result")
    artifact_paths = {
        "summary": rtb_output / "summary.json",
        "report": rtb_output / "report.md",
        "paired_records": rtb_output / "paired_records.jsonl",
        "adapter_off": rtb_output / "results" / "adapter_off.jsonl",
        "adapter_on": rtb_output / "results" / "adapter_on.jsonl",
        "raw_score_checkpoint": rtb_output / "raw_score_checkpoint.json",
    }
    differences: dict[str, dict[str, Any]] = {}
    for name, path in artifact_paths.items():
        observed = file_hash(path) if path.is_file() else None
        expected = audit.get("artifacts", {}).get(name)
        if observed != expected:
            differences[name] = {"expected": expected, "observed": observed}
    checkpoint = read_json_object(
        rtb_output / "raw_score_checkpoint.json", "RTB raw score checkpoint"
    )
    if (
        checkpoint.get("adapter_off_count") != 960
        or checkpoint.get("adapter_on_count") != 960
        or checkpoint.get("checkpoint_stage") != "inference_complete_before_analysis"
        or checkpoint.get("artifacts", {}).get("adapter_off")
        != audit.get("artifacts", {}).get("adapter_off")
        or checkpoint.get("artifacts", {}).get("adapter_on")
        != audit.get("artifacts", {}).get("adapter_on")
    ):
        differences["raw_score_checkpoint_contents"] = {
            "expected": "complete 960/960 rows bound to audit manifest",
            "observed": checkpoint,
        }
    if differences:
        raise ValueError(f"RTB audit manifest does not bind complete artifacts: {differences}")
    cpr_output = Path(identity["cpr_output"])
    qualification_summary = Path(identity["qualification_summary"])
    context = _validate_rtb_sources(cpr_output, qualification_summary)
    if context["adapter_sha256"] != identity["adapter_sha256"]:
        raise ValueError("RTB adapter identity changed")
    return {
        "manifest": manifest,
        "rtb_identity": identity,
        "summary": summary,
        "audit_manifest": audit,
        "context": context,
        "snapshot": _source_snapshot(rtb_output, identity),
    }


def _build_replay_audit(rtb_output: Path) -> dict[str, Any]:
    source = _validate_rtb_result(rtb_output)
    rtb_pairs = _read_jsonl(rtb_output / "paired_records.jsonl")
    primary = [row for row in rtb_pairs if str(row.get("cell", "")).startswith("slot_")]
    tie_counts = Counter()
    violations: list[dict[str, Any]] = []
    for row in primary:
        for state in ("base", "adapter"):
            error = row.get(f"{state}_error_status")
            tie = bool(row.get(f"{state}_tie"))
            if error != "ok" or tie:
                record = {
                    "probe_id": row.get("probe_id"),
                    "cell": row.get("cell"),
                    "state": state,
                    "error_status": error,
                    "tie": tie,
                    "expected_compatible": row.get(f"{state}_tie_expected_compatible"),
                    "signed_expected_margin": row.get(f"{state}_signed_expected_margin"),
                }
                violations.append(record)
                tie_counts[state] += int(tie)
    by_cell: dict[str, dict[str, Any]] = {}
    for cell in sorted({str(row["cell"]) for row in primary}):
        rows = [row for row in primary if row["cell"] == cell]
        delta = sum(int(row["adapter_correct"]) - int(row["base_correct"]) for row in rows) / 96
        tie_n = sum(
            int(row[f"{state}_tie"])
            for row in rows
            for state in ("base", "adapter")
        )
        by_cell[cell] = {
            "observed_delta": delta,
            "conservative_delta_interval": [delta - tie_n / 96, delta + tie_n / 96],
            "tie_count": tie_n,
        }

    q2_path = Path(source["rtb_identity"]["qualification_summary"]).parent / "paired_records.jsonl"
    q2_pairs = {
        str(row["probe_id"]): row
        for row in _read_jsonl(q2_path)
        if row.get("matrix") == "forced_choice" and row.get("category") == "conditional_route"
    }
    external = [row for row in rtb_pairs if row.get("cell") == "external_action"]
    identity_mismatches: list[str] = []
    drift = Counter()
    replay_records: list[dict[str, Any]] = []
    for row in external:
        q2 = q2_pairs.get(str(row["source_probe_id"]))
        if q2 is None or any(
            row.get(field) != q2.get(field)
            for field in ("prompt_sha256", "expected_candidate", "ordered_candidates")
        ):
            identity_mismatches.append(str(row.get("probe_id")))
            continue
        replay_record: dict[str, Any] = {
            "rtb_probe_id": row["probe_id"],
            "q2_probe_id": row["source_probe_id"],
            "prompt_sha256": row["prompt_sha256"],
        }
        for state in ("base", "adapter"):
            decision_changed = bool(row[f"{state}_correct"]) != bool(q2[f"{state}_correct"])
            drift[state] += int(decision_changed)
            q2_margin = q2.get(f"{state}_signed_expected_margin")
            rtb_margin = row.get(f"{state}_signed_expected_margin")
            replay_record[state] = {
                "q2_correct": bool(q2[f"{state}_correct"]),
                "rtb_correct": bool(row[f"{state}_correct"]),
                "decision_changed": decision_changed,
                "q2_signed_expected_margin": q2_margin,
                "rtb_signed_expected_margin": rtb_margin,
                "signed_expected_margin_delta": (
                    None
                    if not isinstance(q2_margin, (int, float))
                    or not isinstance(rtb_margin, (int, float))
                    else float(rtb_margin) - float(q2_margin)
                ),
            }
        replay_records.append(replay_record)
    replay_margin_summary = {
        state: _margin_replay_summary(replay_records, state) for state in ("base", "adapter")
    }
    checks = {
        "primary_row_count": len(primary) == 768,
        "reviewed_violation_count": len(violations) == 26,
        "reviewed_state_tie_counts": tie_counts == {"base": 18, "adapter": 8},
        "all_violations_expected_compatible_zero_margin": all(
            row["tie"] is True
            and row["expected_compatible"] is True
            and row["signed_expected_margin"] == 0.0
            for row in violations
        ),
        "all_slot_cell_conservative_deltas_positive": len(by_cell) == 8
        and all(values["conservative_delta_interval"][0] > 0.0 for values in by_cell.values()),
        "q2_external_row_count": len(q2_pairs) == 96,
        "rtb_external_row_count": len(external) == 96,
        "exact_external_static_identity": not identity_mismatches,
        "complete_external_margin_replay": len(replay_records) == 96
        and all(
            row[state]["signed_expected_margin_delta"] is not None
            for row in replay_records
            for state in ("base", "adapter")
        ),
        "historical_rtb_decision_unchanged": True,
    }
    return {
        "schema_version": "p0d2hrsh-replay-audit-v1",
        "rtb_run_id": source["summary"]["run_id"],
        "rtb_summary_sha256": file_hash(rtb_output / "summary.json"),
        "q2_paired_records_sha256": file_hash(q2_path),
        "violation_count": len(violations),
        "tie_counts": dict(tie_counts),
        "violations_sha256": json_hash(violations),
        "slot_cell_sensitivity": by_cell,
        "q2_to_rtb_decision_drift": dict(drift),
        "q2_to_rtb_margin_drift": replay_margin_summary,
        "external_replay_records": replay_records,
        "external_replay_records_sha256": json_hash(replay_records),
        "external_identity_mismatches": identity_mismatches,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "historical_rtb_decision_changed": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }


def _audit_handoff_anchors(
    probes: list[HandoffProbe],
    token_records: list[dict[str, Any]],
    rtb_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    token_index = {
        (str(row["probe_id"]), str(row["prompt_kind"])): row for row in token_records
    }
    pair_index = {str(row["probe_id"]): row for row in rtb_pairs}
    mismatches: list[dict[str, str]] = []
    for probe in probes:
        anchors = (
            (
                "slot_readout",
                probe.rtb_slot_probe_id,
                probe.target_slot,
                list(probe.slot_candidates),
            ),
            (
                "direct_action",
                probe.rtb_direct_probe_id,
                probe.target_action,
                list(probe.action_candidates),
            ),
            (
                f"receipt_{probe.target_slot}",
                probe.rtb_oracle_probe_id,
                probe.target_action,
                list(probe.action_candidates),
            ),
        )
        for kind, rtb_probe_id, expected, candidates in anchors:
            pair = pair_index.get(rtb_probe_id)
            token = token_index.get((probe.probe_id, kind))
            if (
                pair is None
                or token is None
                or pair.get("source_probe_id") != probe.source_probe_id
                or pair.get("expected_candidate") != expected
                or pair.get("ordered_candidates") != candidates
                or pair.get("prompt_sha256") != token.get("prompt_sha256")
            ):
                mismatches.append(
                    {
                        "probe_id": probe.probe_id,
                        "prompt_kind": kind,
                        "rtb_probe_id": rtb_probe_id,
                    }
                )
    return {
        "schema_version": "p0d2hrsh-rtb-anchor-audit-v1",
        "anchor_count": len(probes) * 3,
        "mismatches": mismatches,
        "all_checks_passed": not mismatches,
    }


def _score_bank(
    bundle: ModelBundle,
    probes: list[HandoffProbe],
    audits: dict[tuple[str, str], dict[str, Any]],
    run_id: str,
    adapter_sha256: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for probe in probes:
        for kind in PROMPT_KINDS:
            formatted = chat_prompt(bundle.tokenizer, probe.prompt(kind))
            off, on = score_adapter_off_on(
                bundle,
                formatted,
                audits[(probe.probe_id, kind)],
                score_candidate_batch,
            )
            common = {
                "schema_version": ROW_SCHEMA_VERSION,
                "run_id": run_id,
                "probe_id": probe.probe_id,
                "lesson_id": probe.lesson_id,
                "pair_id": probe.pair_id,
                "lesson_type": probe.lesson_type,
                "lesson_side": probe.lesson_side,
                "route_variant": probe.route_variant,
                "source_probe_id": probe.source_probe_id,
                "target_slot": probe.target_slot,
                "target_action": probe.target_action,
                "prompt_kind": kind,
                "ordered_candidates": list(probe.candidates(kind)),
                "prompt_sha256": audits[(probe.probe_id, kind)]["prompt_sha256"],
                "candidate_audit_record_sha256": audits[(probe.probe_id, kind)]["record_sha256"],
                "precision": bundle.precision,
            }
            rows.append(_decision_row(common, off, "adapter_off", None))
            rows.append(_decision_row(common, on, "adapter_on", adapter_sha256))
        if len(rows) % 192 == 0:
            _progress(f"handoff decisions complete: {len(rows)}/{len(probes) * 8}")
    return rows


def _margin_replay_summary(records: list[dict[str, Any]], state: str) -> dict[str, Any]:
    deltas = [
        float(row[state]["signed_expected_margin_delta"])
        for row in records
        if row[state]["signed_expected_margin_delta"] is not None
    ]
    return {
        "row_count": len(deltas),
        "mean_delta": sum(deltas) / len(deltas) if deltas else None,
        "mean_absolute_delta": sum(abs(value) for value in deltas) / len(deltas)
        if deltas
        else None,
        "max_absolute_delta": max(map(abs, deltas)) if deltas else None,
    }


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


def _publish_raw(output: Path, run_id: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    off = [row for row in rows if row["adapter_state"] == "adapter_off"]
    on = [row for row in rows if row["adapter_state"] == "adapter_on"]
    hashes = {
        "adapter_off": immutable_write(
            output / "results" / "adapter_off.jsonl",
            _jsonl(off),
            "handoff OFF rows",
        ),
        "adapter_on": immutable_write(
            output / "results" / "adapter_on.jsonl",
            _jsonl(on),
            "handoff ON rows",
        ),
    }
    checkpoint = {
        "schema_version": "p0d2hrsh-raw-score-checkpoint-v1",
        "run_id": run_id,
        "adapter_off_count": len(off),
        "adapter_on_count": len(on),
        "artifacts": dict(hashes),
        "checkpoint_stage": "inference_complete_before_analysis",
        "automatic_analysis_recovery_allowed": False,
    }
    hashes["raw_score_checkpoint"] = immutable_json_write(
        output / "raw_score_checkpoint.json", checkpoint, "handoff raw checkpoint"
    )
    return hashes


def _publish_analysis(
    output: Path, summary: dict[str, Any], paired_records: list[dict[str, Any]]
) -> dict[str, str]:
    report = _render_report(summary)
    return {
        "summary": immutable_json_write(output / "summary.json", summary, "handoff summary"),
        "report": immutable_write(output / "report.md", report.encode(), "handoff report"),
        "paired_records": immutable_write(
            output / "paired_records.jsonl",
            _jsonl(paired_records),
            "handoff paired records",
        ),
    }


def _render_report(summary: dict[str, Any]) -> str:
    analysis = summary["analysis"]
    lines = [
        "# CPR route-state handoff audit",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Decision: `{analysis['decision']['status']}`",
        "- Historical RTB decision changed: `false`",
        "- Training authorized: `false`",
        "- 1/4/8 authorized: `false`",
        "",
        "| Endpoint | OFF | ON | ON-OFF (95% lesson-cluster CI) |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "slot_correct": "Slot readout",
        "direct_correct": "Direct action",
        "predicted_chain_correct": "Predicted-slot chain",
        "oracle_correct": "Oracle-slot action",
        "wrong_target_selected": "Wrong-slot target-action rate",
    }
    for key, label in labels.items():
        values = analysis["endpoint_summaries"][key]
        difference = values["adapter_minus_base"]
        lines.append(
            f"| {label} | {values['base']:.4f} | {values['adapter']:.4f} | "
            f"{difference['estimate']:.4f} "
            f"[{difference['ci95'][0]:.4f}, {difference['ci95'][1]:.4f}] |"
        )
    primary = analysis["primary_handoff_rescue"]
    conservative = primary["conservative_compatible_tie_bound"]
    lines.extend(
        [
            "",
            "## Primary handoff rescue",
            "",
            f"- Conservative estimate: {conservative['estimate']:.4f} "
            f"[{conservative['ci95'][0]:.4f}, {conservative['ci95'][1]:.4f}].",
            f"- Supported: `{str(primary['supported']).lower()}`.",
            "",
            "This inference-only audit cannot reclassify RTB, authorize training, or authorize "
            "the 1/4/8 experiment.",
            "",
        ]
    )
    return "\n".join(lines)


def _verify_preflight(output: Path, identity: dict[str, Any]) -> None:
    files = {
        "probe_sha256": output / "preflight" / "handoff_probes.jsonl",
        "bank_audit_sha256": output / "preflight" / "bank_audit.json",
        "replay_audit_sha256": output / "preflight" / "replay_audit.json",
        "token_audit_sha256": output / "preflight" / "token_audit.json",
    }
    for field, path in files.items():
        if file_hash(path) != identity[field]:
            raise ValueError(f"route-state handoff preflight changed: {field}")
    prereg = read_json_object(output / "preregistration.json", "handoff preregistration")
    unhashed_identity = {
        key: value for key, value in identity.items() if key != "preregistration_sha256"
    }
    if (
        prereg != identity
        or json_hash(unhashed_identity) != identity["preregistration_sha256"]
    ):
        raise ValueError("route-state handoff preregistration changed")
    replay = read_json_object(output / "preflight" / "replay_audit.json", "handoff replay audit")
    if replay.get("all_checks_passed") is not True:
        raise ValueError("route-state handoff replay audit is not qualified")


def _verify_authorization(output: Path, manifest: dict[str, Any]) -> None:
    path = output / "authorization.json"
    authorization = read_json_object(path, "adopted handoff authorization")
    if (
        manifest.get("authorization", {}).get("sha256") != file_hash(path)
        or authorization.get("decision") != "approved"
        or authorization.get("preregistration_sha256")
        != manifest["config"]["preregistration_sha256"]
        or authorization.get("allows_training") is not False
        or authorization.get("allows_mappings_per_adapter_scan") is not False
    ):
        raise ValueError("route-state handoff authorization changed after adoption")


def _load_bridge_probes(rtb_output: Path) -> list[BridgeProbe]:
    path = rtb_output / "preflight" / "bridge_probes.jsonl"
    return [BridgeProbe(**row) for row in _read_jsonl(path)]


def _load_handoff_probes(output: Path) -> list[HandoffProbe]:
    path = output / "preflight" / "handoff_probes.jsonl"
    return [HandoffProbe(**row) for row in _read_jsonl(path)]


def _load_token_audits(output: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = read_json_object(output / "preflight" / "token_audit.json", "handoff token audit")
    result = {(str(row["probe_id"]), str(row["prompt_kind"])): row for row in payload["records"]}
    if len(result) != 384 or payload.get("all_checks_passed") is not True:
        raise ValueError("route-state handoff token audit is incomplete")
    return result


def _source_snapshot(rtb_output: Path, rtb_identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "rtb_output_tree": _tree_hash(rtb_output),
        "rtb_bound_sources": _rtb_source_snapshot(
            Path(rtb_identity["cpr_output"]),
            Path(rtb_identity["qualification_summary"]).parent,
        ),
    }


def _score_off(bundle: ModelBundle, prompt: str, audit: dict[str, Any]) -> dict[str, Any]:
    with bundle.model.disable_adapter():
        return score_candidate_batch(bundle, prompt, audit)


def _load_manifest(output: Path) -> dict[str, Any]:
    manifest = read_json_object(output / "manifest.json", "route-state handoff manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("not a route-state handoff manifest")
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
        raise ValueError("route-state handoff output must be independent of RTB source")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    return (payload + "\n").encode()


def _jsonl_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def _progress(message: str) -> None:
    print(f"[p0d2hrsh] {message}", flush=True)
