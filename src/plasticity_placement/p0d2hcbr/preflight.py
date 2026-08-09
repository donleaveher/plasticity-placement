from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.runtime import current_code_hash
from plasticity_placement.p0d2hcbr.authorization import template
from plasticity_placement.p0d2hcbr.config import (
    CONFIG_SCHEMA_VERSION,
    CORRECTED_CONFIG_SCHEMA_VERSION,
    CORRECTED_SPEC_SCHEMA_VERSION,
    CURRICULA,
    CorrectedExperimentSpec,
    ExperimentSpec,
)
from plasticity_placement.p0d2hcbr.data import (
    HeldoutProbe,
    TrainingRecord,
    compile_heldout_bank,
    compile_training_banks,
    write_banks,
)
from plasticity_placement.p0d2hcbr.manifest import Manifest
from plasticity_placement.p0d2hcrd.scoring import audit_candidate_tokenization
from plasticity_placement.p0d2hrabx.config import AuditSpec as RabxAuditSpec
from plasticity_placement.p0d2hrabx.runtime import (
    _tree_hash,
    _validate_rabc_source,
)
from plasticity_placement.p0d2hrabx.runtime import (
    verify_complete_result as verify_rabx_result,
)
from plasticity_placement.p0d2hrr.io import (
    file_hash,
    immutable_json_write,
    json_hash,
    read_json_object,
)
from plasticity_placement.training.model_utils import chat_prompt, load_tokenizer


def plan_experiment(
    *,
    output_dir: Path,
    rabx_output: Path,
    source_code_revision_lock: Path,
    spec_path: Path,
) -> Path:
    output_dir = output_dir.resolve()
    rabx_output = rabx_output.resolve()
    source_code_revision_lock = source_code_revision_lock.resolve()
    _require_independent(rabx_output, output_dir)
    if output_dir.exists():
        raise FileExistsError(f"CBR output already exists: {output_dir}")
    spec = ExperimentSpec.from_path(spec_path.resolve())
    source = _validate_rabx_source(rabx_output, source_code_revision_lock)
    train, dev, training_audit = compile_training_banks(spec.data)
    heldout, heldout_audit = compile_heldout_bank(spec.data)
    tokenizer = load_tokenizer(spec.model_name, spec.model_revision)
    training_token_audit = {
        curriculum: _training_token_audit(
            tokenizer,
            train[curriculum] + dev[curriculum],
            max_length=spec.training.max_length,
        )
        for curriculum in CURRICULA
    }
    if not all(value["all_checks_passed"] for value in training_token_audit.values()):
        raise ValueError("CBR training token audit failed")
    heldout_token_audit = _heldout_token_audit(tokenizer, heldout)
    source_overlap_audit = _source_overlap_audit(
        rabx_output,
        Path(source["rab_output"]),
        [record for curriculum in CURRICULA for record in train[curriculum] + dev[curriculum]],
        heldout,
    )

    output_dir.mkdir(parents=True)
    data_hashes = write_banks(output_dir, train, dev, training_audit, heldout, heldout_audit)
    data_hashes["training_token_audit"] = immutable_json_write(
        output_dir / "preflight" / "training_token_audit.json",
        {
            "schema_version": "p0d2hcbr-training-token-audit-v1",
            "curricula": training_token_audit,
        },
        "CBR training token audit",
    )
    data_hashes["source_overlap_audit"] = immutable_json_write(
        output_dir / "preflight" / "source_overlap_audit.json",
        source_overlap_audit,
        "CBR source-overlap audit",
    )
    data_hashes["heldout_token_audit"] = immutable_json_write(
        output_dir / "preflight" / "rab_gen_token_audit.json",
        heldout_token_audit,
        "CBR held-out token audit",
    )
    analysis_plan = {
        "schema_version": "p0d2hcbr-analysis-plan-v1",
        "training_matrix": "curriculum coupled/disentangled x placement full/late x 3 seeds",
        "primary_panel": "24 held-out groups x 64 factorial cells",
        "primary_gate": "tie-worst binding accuracy and four factor-equivalence intervals",
        "preservation_panels": ["historical_rab", "external_conditional_route", "full_crd"],
        "uncertainty": "semantic-group/lesson clustered bootstrap with seed replication",
        "training_authorization_is_external": True,
        "mappings_per_adapter_authorized": False,
    }
    analysis_plan["analysis_plan_sha256"] = json_hash(analysis_plan)
    data_hashes["analysis_plan"] = immutable_json_write(
        output_dir / "preflight" / "analysis_plan.json", analysis_plan, "CBR analysis plan"
    )
    identity = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "stage": "counterbalanced_binding_remediation",
        "code_sha256": current_code_hash(),
        "spec": spec.to_dict(),
        "spec_sha256": json_hash(spec.to_dict()),
        "rabx_output": str(rabx_output),
        "rabx_run_id": source["rabx_summary"]["run_id"],
        "rabc_output": source["rabx_identity"]["rabc_output"],
        "rab_output": source["rab_output"],
        "source_code_revision_lock": str(source_code_revision_lock),
        "source_code_revision_lock_sha256": file_hash(source_code_revision_lock),
        "source_snapshot": source["snapshot"],
        "data_hashes": data_hashes,
        "training_run_limit": 12,
        "locked_evaluation_limit": 12,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    identity["preregistration_sha256"] = json_hash(identity)
    immutable_json_write(output_dir / "preregistration.json", identity, "CBR preregistration")
    immutable_json_write(
        output_dir / "authorization.template.json",
        template(identity["preregistration_sha256"]),
        "CBR authorization template",
    )
    manifest = Manifest.create(
        output_dir / "manifest.json",
        {
            "run_id": "p0d2hcbr-" + identity["preregistration_sha256"][:10],
            "config": identity,
            "rabx_output": str(rabx_output),
        },
        spec.unit_ids(),
    )
    return manifest.path


def load_config(
    output_dir: Path,
) -> tuple[Manifest, ExperimentSpec | CorrectedExperimentSpec, dict[str, Any]]:
    output_dir = output_dir.resolve()
    manifest = Manifest.load(output_dir / "manifest.json")
    identity = manifest.payload.get("config")
    if not isinstance(identity, dict) or identity.get("schema_version") not in {
        CONFIG_SCHEMA_VERSION,
        CORRECTED_CONFIG_SCHEMA_VERSION,
    }:
        raise ValueError("CBR preregistration identity is missing")
    spec_payload = identity.get("spec")
    if not isinstance(spec_payload, dict):
        raise ValueError("CBR spec is missing")
    spec = _spec_from_dict(spec_payload)
    prereg = read_json_object(output_dir / "preregistration.json", "CBR preregistration")
    if (
        prereg != identity
        or json_hash(
            {key: value for key, value in identity.items() if key != "preregistration_sha256"}
        )
        != identity["preregistration_sha256"]
    ):
        raise ValueError("CBR preregistration changed")
    paths = _preflight_paths(output_dir)
    for name, digest in identity["data_hashes"].items():
        if name not in paths or file_hash(paths[name]) != digest:
            raise ValueError(f"CBR preflight artifact changed: {name}")
    if (
        file_hash(Path(identity["source_code_revision_lock"]))
        != identity["source_code_revision_lock_sha256"]
    ):
        raise ValueError("CBR source code revision lock changed")
    if identity["schema_version"] == CORRECTED_CONFIG_SCHEMA_VERSION:
        from plasticity_placement.p0d2hcbr.corrected import validate_corrected_source

        validate_corrected_source(identity)
    return manifest, spec, identity


def _validate_rabx_source(rabx_output: Path, source_code_revision_lock: Path) -> dict[str, Any]:
    verify_rabx_result(rabx_output)
    manifest = read_json_object(rabx_output / "manifest.json", "RABX manifest")
    identity = read_json_object(rabx_output / "preregistration.json", "RABX preregistration")
    summary = read_json_object(rabx_output / "summary.json", "RABX summary")
    flags = summary.get("analysis", {}).get("causal_attribution", {}).get("flags")
    expected_flags = {
        "candidate_position_0": True,
        "composition_interaction": True,
        "receipt_token": True,
        "serial_position": True,
        "slot_label": True,
    }
    if (
        manifest.get("config") != identity
        or summary.get("analysis_status") != "label_disentanglement_complete"
        or summary.get("analysis", {}).get("causal_attribution", {}).get("status")
        != "multiple_mechanisms"
        or flags != expected_flags
        or summary.get("training_authorized") is not False
        or summary.get("mappings_per_adapter_authorized") is not False
    ):
        raise ValueError("CBR source is not the completed multiple-mechanism RABX audit")
    rabc_output = Path(identity["rabc_output"])
    source = _validate_rabc_source(rabc_output, RabxAuditSpec())
    rab_output = Path(identity["rab_output"])
    if str(rab_output) != str(source["rabc_identity"]["rab_output"]):
        raise ValueError("RABX transitive RAB source differs")
    return {
        "rabx_manifest": manifest,
        "rabx_identity": identity,
        "rabx_summary": summary,
        "rab_output": str(rab_output),
        "snapshot": {
            "rabx_output_tree": _tree_hash(rabx_output),
            "transitive_rabc_rab": source["snapshot"],
            "source_code_revision_lock_sha256": file_hash(source_code_revision_lock),
        },
    }


def _training_token_audit(
    tokenizer: Any, records: list[TrainingRecord], *, max_length: int
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in records:
        formatted = chat_prompt(tokenizer, record.prompt)
        prompt_ids = tokenizer(formatted, add_special_tokens=True)["input_ids"]
        completion_ids = tokenizer(record.completion, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            completion_ids = [*completion_ids, tokenizer.eos_token_id]
        survives = len(prompt_ids) + len(completion_ids) <= max_length
        rows.append(
            {
                "example_id": record.example_id,
                "prompt_token_count": len(prompt_ids),
                "completion_token_count": len(completion_ids),
                "completion_fully_survives": survives,
            }
        )
    failed = [row["example_id"] for row in rows if not row["completion_fully_survives"]]
    return {
        "decision_count": len(rows),
        "max_length": max_length,
        "failed_example_ids": failed,
        "records": rows,
        "all_checks_passed": not failed,
    }


def _heldout_token_audit(tokenizer: Any, probes: list[HeldoutProbe]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for probe in probes:
        audit = audit_candidate_tokenization(
            tokenizer,
            chat_prompt(tokenizer, probe.prompt),
            probe.ordered_candidates,
            evaluation_max_length=512,
        )
        record = {"probe_id": probe.probe_id, **audit}
        record["record_sha256"] = json_hash(record)
        rows.append(record)
    failed = [row["probe_id"] for row in rows if row["all_candidates_valid"] is not True]
    if failed:
        raise ValueError(f"CBR held-out token audit failed: {failed[:10]}")
    return {
        "schema_version": "p0d2hcbr-heldout-token-audit-v1",
        "decision_count": len(rows),
        "failed_probe_ids": failed,
        "records": rows,
        "all_checks_passed": True,
    }


def _preflight_paths(output_dir: Path) -> dict[str, Path]:
    root = output_dir / "preflight"
    paths = {f"train_{curriculum}": root / f"train_{curriculum}.jsonl" for curriculum in CURRICULA}
    paths.update(
        {f"dev_{curriculum}": root / f"dev_{curriculum}.jsonl" for curriculum in CURRICULA}
    )
    paths.update(
        {
            "training_audit": root / "training_data_audit.json",
            "heldout_probes": root / "rab_gen_probes.jsonl",
            "heldout_audit": root / "rab_gen_bank_audit.json",
            "training_token_audit": root / "training_token_audit.json",
            "heldout_token_audit": root / "rab_gen_token_audit.json",
            "source_overlap_audit": root / "source_overlap_audit.json",
            "analysis_plan": root / "analysis_plan.json",
        }
    )
    return paths


def _spec_from_dict(
    payload: dict[str, Any],
) -> ExperimentSpec | CorrectedExperimentSpec:
    from plasticity_placement.p0d2hcbr.config import (
        CorrectedTrainingSpec,
        DataSpec,
        GateSpec,
        TrainingSpec,
    )

    copy = dict(payload)
    data = DataSpec(**dict(copy.pop("data")))
    training_payload = dict(copy.pop("training"))
    training_payload["target_modules"] = tuple(training_payload["target_modules"])
    training_payload["seeds"] = tuple(training_payload["seeds"])
    corrected = copy.get("schema_version") == CORRECTED_SPEC_SCHEMA_VERSION
    if corrected:
        training_payload["late_explicit_layers"] = tuple(
            training_payload["late_explicit_layers"]
        )
        training = CorrectedTrainingSpec(**training_payload)
    else:
        training = TrainingSpec(**training_payload)
    gates = GateSpec(**dict(copy.pop("gates")))
    copy["curricula"] = tuple(copy["curricula"])
    copy["placements"] = tuple(copy["placements"])
    spec_type = CorrectedExperimentSpec if corrected else ExperimentSpec
    return spec_type(**copy, data=data, training=training, gates=gates)


def load_heldout_probes(output_dir: Path) -> list[HeldoutProbe]:
    rows = [
        json.loads(line)
        for line in (output_dir / "preflight" / "rab_gen_probes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for row in rows:
        row["ordered_candidates"] = tuple(row["ordered_candidates"])
    return [HeldoutProbe(**row) for row in rows]


def load_heldout_audits(output_dir: Path) -> dict[str, dict[str, Any]]:
    payload = read_json_object(
        output_dir / "preflight" / "rab_gen_token_audit.json", "CBR held-out token audit"
    )
    return {str(row["probe_id"]): row for row in payload["records"]}


def _source_overlap_audit(
    rabx_output: Path,
    rab_output: Path,
    training_records: list[TrainingRecord],
    heldout_probes: list[HeldoutProbe],
) -> dict[str, Any]:
    source_paths = (
        rabx_output / "preflight" / "label_disentanglement_probes.jsonl",
        rab_output / "preflight" / "binding_probes.jsonl",
    )
    source_hashes: set[str] = set()
    for path in source_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                prompt = json.loads(line).get("prompt")
                if isinstance(prompt, str):
                    source_hashes.add(sha256(prompt.encode()).hexdigest())
    training_hashes = {sha256(row.prompt.encode()).hexdigest() for row in training_records}
    heldout_hashes = {sha256(row.prompt.encode()).hexdigest() for row in heldout_probes}
    checks = {
        "source_prompts_loaded": bool(source_hashes),
        "training_exact_prompt_disjoint": source_hashes.isdisjoint(training_hashes),
        "heldout_exact_prompt_disjoint": source_hashes.isdisjoint(heldout_hashes),
        "training_heldout_exact_prompt_disjoint": training_hashes.isdisjoint(heldout_hashes),
    }
    payload = {
        "schema_version": "p0d2hcbr-source-overlap-audit-v1",
        "source_prompt_count": len(source_hashes),
        "training_prompt_count": len(training_hashes),
        "heldout_prompt_count": len(heldout_hashes),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    if payload["all_checks_passed"] is not True:
        raise ValueError(f"CBR source-overlap audit failed: {checks}")
    return payload


def _require_independent(source: Path, output: Path) -> None:
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("CBR output must be independent from the RABX source")
