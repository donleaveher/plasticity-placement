from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.modeling import (
    ModelBundle,
    activate_adapter,
    load_base_model,
    release_model,
)
from plasticity_placement.p0c.runtime import current_environment_snapshot
from plasticity_placement.p0d2h.probes import load_hard_probe_bank
from plasticity_placement.p0d2hcbr.analysis import analyze_factorial, classify_unit
from plasticity_placement.p0d2hcbr.config import unit_id
from plasticity_placement.p0d2hcbr.deviation import evaluation_code_sha256
from plasticity_placement.p0d2hcbr.preflight import (
    _validate_rabx_source,
    load_config,
    load_heldout_audits,
    load_heldout_probes,
)
from plasticity_placement.p0d2hcbr.runtime import _adapter_dir, validate_all_training_units
from plasticity_placement.p0d2hcrd.probes import compile_decomposition_bank
from plasticity_placement.p0d2hcrd.runtime import validate_frozen_source
from plasticity_placement.p0d2hcrd.scoring import score_candidate_batch
from plasticity_placement.p0d2hrab.analysis import analyze_binding, build_binding_records
from plasticity_placement.p0d2hrab.config import AuditSpec as RabAuditSpec
from plasticity_placement.p0d2hrab.runtime import (
    _load_binding_probes,
    _load_token_audits,
)
from plasticity_placement.p0d2hrab.runtime import (
    _score_bank as score_rab_bank,
)
from plasticity_placement.p0d2hrabx.config import AuditSpec as RabxAuditSpec
from plasticity_placement.p0d2hrabx.runtime import _validate_rabc_source
from plasticity_placement.p0d2hrr.io import (
    file_hash,
    immutable_json_write,
    immutable_write,
    json_hash,
)
from plasticity_placement.p0d2hrr.paired_audit import (
    PAIRED_ROW_SCHEMA_VERSION,
    _json_bytes,
    _jsonl_bytes,
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
    compare_sentinel_scores,
    score_adapter_off_on,
)
from plasticity_placement.training.model_utils import chat_prompt

SCHEMA_VERSION = "p0d2hcbr-unit-evaluation-v1"
ROW_SCHEMA_VERSION = "p0d2hcbr-heldout-decision-row-v1"


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    output_dir: Path
    curriculum: str
    placement: str
    seed: int
    analysis_output: Path


def evaluate_unit(request: EvaluationRequest) -> Path:
    output_dir = request.output_dir.resolve()
    analysis_output = request.analysis_output.resolve()
    _require_new_independent(output_dir, analysis_output)
    manifest, spec, identity = load_config(output_dir)
    budget_audit = validate_all_training_units(output_dir, allow_authorized_deviation=True)
    evaluator_sha256 = evaluation_code_sha256(output_dir, identity)
    key = unit_id(request.curriculum, request.placement, request.seed)
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
    run_identity = {
        "schema_version": SCHEMA_VERSION,
        "cbr_run_id": manifest.payload["run_id"],
        "unit_id": key,
        "adapter_sha256": adapter_sha256,
        "analysis_output": str(analysis_output),
        "experiment_code_sha256": identity["code_sha256"],
        "evaluation_code_sha256": evaluator_sha256,
        "budget_audit_sha256": json_hash(budget_audit),
    }
    run_id = "p0d2hcbr-eval-" + json_hash(run_identity)[:10]
    claim_hash = _claim_evaluation(output_dir, key, run_id, analysis_output, adapter_sha256)
    manifest = load_config(output_dir)[0]
    manifest.payload["evaluation_claims"][key] = {
        "state": "claimed",
        "run_id": run_id,
        "analysis_output": str(analysis_output),
        "claim_sha256": claim_hash,
        "summary_sha256": None,
    }
    manifest.save()
    before = _source_snapshot(output_dir, identity, key)

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
    fc_audits = _load_fc_audits(rr_output, rr_config)
    crd_audits = _load_crd_audits(rr_output, rr_config)
    heldout_probes = load_heldout_probes(output_dir)
    heldout_audits = load_heldout_audits(output_dir)
    rab_output = Path(identity["rab_output"])
    rab_probes = _load_binding_probes(rab_output)
    rab_audits = _load_token_audits(rab_output)
    load_value = _AdapterLoadConfig(
        model_name=spec.model_name,
        model_revision=spec.model_revision,
        use_4bit=spec.training.use_4bit,
        training_seeds=(request.seed,),
    )

    bundle: ModelBundle | None = None
    try:
        _progress(key, "loading one base-model runtime")
        bundle = load_base_model(load_value)
        _require_source_compatible_bundle(bundle, rr_config)
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
        first = heldout_probes[0]
        first_prompt = chat_prompt(bundle.tokenizer, first.prompt)
        with bundle.model.disable_adapter():
            sentinel_before = score_candidate_batch(
                bundle, first_prompt, heldout_audits[first.probe_id]
            )
        _progress(key, "scoring 1,536 held-out factorial OFF/ON pairs")
        heldout_rows = _score_heldout(
            bundle, heldout_probes, heldout_audits, run_id, adapter_sha256
        )
        _progress(key, "scoring 192 historical RAB OFF/ON pairs")
        rab_rows = score_rab_bank(bundle, rab_probes, rab_audits, run_id, adapter_sha256)
        _progress(key, "scoring 96 external conditional-route OFF/ON pairs")
        off_fc, on_fc = _evaluate_forced_choice_pairs(
            bundle=bundle,
            run_id=run_id,
            adapter_sha256=adapter_sha256,
            config=rr_config,
            source=frozen,
            hard_bank=hard_bank,
            source_rows=source_rows,
            audits=fc_audits,
        )
        _progress(key, "scoring 1,536 CRD OFF/ON pairs")
        off_crd, on_crd = _evaluate_crd_pairs(
            bundle=bundle,
            run_id=run_id,
            adapter_sha256=adapter_sha256,
            config=rr_config,
            bank=crd_bank,
            source_rows=frozen.source_rows,
            audits=crd_audits,
        )
        with bundle.model.disable_adapter():
            sentinel_after = score_candidate_batch(
                bundle, first_prompt, heldout_audits[first.probe_id]
            )
        sentinel = compare_sentinel_scores(
            sentinel_before, sentinel_after, tolerance=spec.gates.sentinel_score_tolerance
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
            "adapter_sha256": adapter_sha256,
            "sentinel": sentinel,
        }
        if not runtime_identity["object_identity_preserved"] or not sentinel["passed"]:
            raise RuntimeError("CBR same-runtime integrity check failed")
    finally:
        if bundle is not None:
            release_model(bundle)

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
    if before != after:
        raise RuntimeError("CBR source artifacts changed during unit evaluation")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "cbr_run_id": manifest.payload["run_id"],
        "unit": {
            "unit_id": key,
            "curriculum": request.curriculum,
            "placement": request.placement,
            "seed": request.seed,
            "adapter_sha256": adapter_sha256,
        },
        "runtime_identity": runtime_identity,
        "environment": current_environment_snapshot(),
        "heldout_factorial": heldout_analysis,
        "historical_rab": rab_analysis,
        "preservation": paired_analysis,
        "decision": decision,
        "budget_audit": budget_audit,
        "placement_comparison_scope": budget_audit.get(
            "comparison_scope", "preregistered_parameter_matched"
        ),
        "source_snapshot_before": before,
        "source_snapshot_after": after,
        "source_artifacts_modified": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
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
    audit_manifest = {
        "schema_version": "p0d2hcbr-unit-evaluation-manifest-v1",
        "run_id": run_id,
        "identity": run_identity,
        "source_snapshot_sha256": json_hash(before),
        "artifacts": {
            name: {"path": str(paths[name]), "sha256": digest} for name, digest in hashes.items()
        },
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    manifest_hash = immutable_write(
        paths["manifest"], _json_bytes(audit_manifest), "CBR unit evaluation manifest"
    )
    manifest = load_config(output_dir)[0]
    manifest.payload["evaluation_claims"][key].update(
        {
            "state": "complete",
            "summary_sha256": hashes["summary"],
            "evaluation_manifest_sha256": manifest_hash,
        }
    )
    if len(manifest.payload["evaluation_claims"]) == 12 and all(
        value["state"] == "complete" for value in manifest.payload["evaluation_claims"].values()
    ):
        manifest.payload["state"] = "evaluated"
    manifest.save()
    _progress(key, f"complete: {paths['summary']}")
    return paths["summary"]


def _score_heldout(
    bundle: ModelBundle,
    probes: list[Any],
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
            **{key: value for key, value in probe.to_dict().items() if key != "prompt"},
            "ordered_candidates": list(probe.ordered_candidates),
            "formatted_prompt_sha256": audits[probe.probe_id]["prompt_sha256"],
        }
        rows.append(_decision_row(common, off, "adapter_off", None))
        rows.append(_decision_row(common, on, "adapter_on", adapter_sha256))
        if len(rows) % 256 == 0:
            print(f"[p0d2hcbr] held-out decisions: {len(rows)}/{len(probes) * 2}", flush=True)
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


def _stage(
    output: Path,
    summary: dict[str, Any],
    heldout_rows: list[dict[str, Any]],
    rab_rows: list[dict[str, Any]],
    off_fc: list[dict[str, Any]],
    on_fc: list[dict[str, Any]],
    off_crd: list[dict[str, Any]],
    on_crd: list[dict[str, Any]],
    heldout_pairs: list[dict[str, Any]],
    rab_pairs: list[dict[str, Any]],
    fc_pairs: list[dict[str, Any]],
    crd_pairs: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> tuple[dict[str, Path], dict[str, str]]:
    paths = {
        "summary": output / "summary.json",
        "report": output / "report.md",
        "heldout_raw": output / "results" / "heldout_off_on.jsonl",
        "rab_raw": output / "results" / "historical_rab_off_on.jsonl",
        "off_forced_choice": output / "results" / "adapter_off_fc.jsonl",
        "on_forced_choice": output / "results" / "adapter_on_fc.jsonl",
        "off_crd": output / "results" / "adapter_off_crd.jsonl",
        "on_crd": output / "results" / "adapter_on_crd.jsonl",
        "heldout_pairs": output / "heldout_paired_records.jsonl",
        "rab_pairs": output / "rab_paired_records.jsonl",
        "guardrail_pairs": output / "guardrail_paired_records.jsonl",
        "combined_failures": output / "combined_failure_records.jsonl",
        "manifest": output / "evaluation_manifest.json",
    }
    guardrail = [
        {"schema_version": PAIRED_ROW_SCHEMA_VERSION, "matrix": "forced_choice", **row}
        for row in fc_pairs
    ] + [{"schema_version": PAIRED_ROW_SCHEMA_VERSION, "matrix": "crd", **row} for row in crd_pairs]
    payloads = {
        "summary": _json_bytes(summary),
        "report": _render_report(summary).encode(),
        "heldout_raw": _jsonl_bytes(heldout_rows),
        "rab_raw": _jsonl_bytes(rab_rows),
        "off_forced_choice": _jsonl_bytes(off_fc),
        "on_forced_choice": _jsonl_bytes(on_fc),
        "off_crd": _jsonl_bytes(off_crd),
        "on_crd": _jsonl_bytes(on_crd),
        "heldout_pairs": _jsonl_bytes(heldout_pairs),
        "rab_pairs": _jsonl_bytes(rab_pairs),
        "guardrail_pairs": _jsonl_bytes(guardrail),
        "combined_failures": _jsonl_bytes(failures),
    }
    hashes = {
        name: immutable_write(paths[name], value, f"CBR unit evaluation {name}")
        for name, value in payloads.items()
    }
    return paths, hashes


def _render_report(summary: dict[str, Any]) -> str:
    unit = summary["unit"]
    decision = summary["decision"]
    heldout = summary["heldout_factorial"]["state_summaries"]["adapter"]
    return "\n".join(
        [
            "# CBR-v1 unit evaluation",
            "",
            f"- Unit: `{unit['unit_id']}`",
            f"- Decision: `{decision['status']}`",
            f"- Held-out tie-worst accuracy: `{heldout['accuracy_identified_interval'][0]:.4f}`",
            "- Training authorized: `false`",
            "- 1/4/8 authorized: `false`",
            "",
        ]
    )


def _claim_evaluation(
    output_dir: Path,
    key: str,
    run_id: str,
    analysis_output: Path,
    adapter_sha256: str,
) -> str:
    path = output_dir / "claims" / f"evaluation-{key}.json"
    payload = {
        "schema_version": "p0d2hcbr-evaluation-claim-v1",
        "run_id": run_id,
        "unit_id": key,
        "analysis_output": str(analysis_output),
        "adapter_sha256": adapter_sha256,
        "allowed_locked_evaluations": 1,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    return immutable_json_write(path, payload, f"CBR evaluation claim {key}")


def _source_snapshot(output_dir: Path, identity: dict[str, Any], key: str) -> dict[str, Any]:
    manifest = load_config(output_dir)[0]
    adapter_dir = _adapter_dir(output_dir, manifest.payload["training_units"][key], key)
    snapshot = {
        "preregistration_sha256": file_hash(output_dir / "preregistration.json"),
        "authorization_sha256": file_hash(output_dir / "authorization.json"),
        "training_metadata_sha256": file_hash(adapter_dir / "training_metadata.json"),
        "adapter_sha256": _adapter_hash(adapter_dir),
        "rabx_source": _validate_rabx_source(
            Path(identity["rabx_output"]), Path(identity["source_code_revision_lock"])
        )["snapshot"],
    }
    deviation = output_dir / "budget_deviation_authorization.json"
    if deviation.is_file():
        snapshot["budget_deviation_authorization_sha256"] = file_hash(deviation)
    corrected_source = identity.get("corrected_source_snapshot")
    if corrected_source is not None:
        snapshot["corrected_source_snapshot"] = corrected_source
    return snapshot


def _require_new_independent(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"CBR unit evaluation output already exists: {output}")
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("CBR unit evaluation output must be independent from training output")


def _progress(key: str, message: str) -> None:
    print(f"[p0d2hcbr:{key}] {message}", flush=True)
