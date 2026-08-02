from __future__ import annotations

from pathlib import Path
from typing import Any

from plasticity_placement.p0c.runtime import current_code_hash
from plasticity_placement.p0d2hcpr.authorization import template
from plasticity_placement.p0d2hcpr.config import CONFIG_SCHEMA_VERSION, ExperimentSpec
from plasticity_placement.p0d2hcpr.data import compile_data, write_data
from plasticity_placement.p0d2hcpr.manifest import Manifest
from plasticity_placement.p0d2hcrd.runtime import validate_frozen_source
from plasticity_placement.p0d2hrr.io import (
    file_hash,
    immutable_json_write,
    json_hash,
    read_json_object,
)
from plasticity_placement.p0d2hrr.paired_audit import _validate_route_remediation
from plasticity_placement.p0d2hrr.preflight import _frozen_lesson_strings, load_resolved_config
from plasticity_placement.training.model_utils import chat_prompt, load_tokenizer


def plan_experiment(
    *,
    output_dir: Path,
    source_output: Path,
    same_runtime_summary: Path,
    source_code_revision_lock: Path,
    spec_path: Path,
) -> Path:
    output_dir = output_dir.resolve()
    source_output = source_output.resolve()
    same_runtime_summary = same_runtime_summary.resolve()
    source_code_revision_lock = source_code_revision_lock.resolve()
    _require_independent((source_output, same_runtime_summary.parent), output_dir)
    spec = ExperimentSpec.from_path(spec_path.resolve())
    provenance, _, _, _, source_context = _validate_route_remediation(
        source_output, source_code_revision_lock
    )
    source_config = load_resolved_config(source_output)
    frozen = validate_frozen_source(source_config.source_manifest_path)
    audit = _validate_failed_same_runtime(same_runtime_summary, provenance)
    same_runtime_manifest = same_runtime_summary.parent / "audit_manifest.json"

    train, dev, data_audit = compile_data(
        spec.data,
        forbidden_strings=_frozen_lesson_strings(frozen.selected),
    )
    data_hashes = write_data(output_dir, train, dev, data_audit)
    tokenizer = load_tokenizer(spec.model_name, spec.model_revision)
    token_audit = _build_training_token_audit(
        tokenizer,
        train + dev,
        max_length=spec.training.max_length,
    )
    data_hashes["token_audit"] = immutable_json_write(
        output_dir / "preflight" / "training_token_audit.json",
        token_audit,
        "CPR training token audit",
    )
    identity = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "stage": "composition_preserving_route_remediation",
        "code_sha256": current_code_hash(),
        "spec": spec.to_dict(),
        "spec_sha256": json_hash(spec.to_dict()),
        "source_route_remediation": provenance,
        "source_context": source_context,
        "source_output": str(source_output),
        "source_code_revision_lock": str(source_code_revision_lock),
        "source_code_revision_lock_sha256": file_hash(source_code_revision_lock),
        "same_runtime_summary": str(same_runtime_summary),
        "same_runtime_summary_sha256": file_hash(same_runtime_summary),
        "same_runtime_manifest": str(same_runtime_manifest),
        "same_runtime_manifest_sha256": file_hash(same_runtime_manifest),
        "same_runtime_run_id": audit["run_id"],
        "data_hashes": data_hashes,
        "training_run_limit": 1,
        "locked_qualification_limit": 1,
        "automatic_mappings_per_adapter_authorized": False,
    }
    preregistration_sha256 = json_hash(identity)
    identity["preregistration_sha256"] = preregistration_sha256
    immutable_json_write(output_dir / "preregistration.json", identity, "CPR preregistration")
    immutable_json_write(
        output_dir / "authorization.template.json",
        template(preregistration_sha256),
        "CPR authorization template",
    )
    run_id = "p0d2hcpr-" + preregistration_sha256[:10]
    manifest = Manifest.load_or_create(
        output_dir / "manifest.json",
        {
            "run_id": run_id,
            "config": identity,
            "source_output": str(source_output),
            "same_runtime_summary": str(same_runtime_summary),
        },
    )
    return manifest.path


def load_config(output_dir: Path) -> tuple[Manifest, ExperimentSpec, dict[str, Any]]:
    output_dir = output_dir.resolve()
    manifest = Manifest.load(output_dir / "manifest.json")
    identity = manifest.payload.get("config")
    if not isinstance(identity, dict) or identity.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("CPR preregistration identity is missing")
    spec_payload = identity.get("spec")
    if not isinstance(spec_payload, dict):
        raise ValueError("CPR spec is missing")
    spec = _spec_from_dict(spec_payload)
    prereg = read_json_object(output_dir / "preregistration.json", "CPR preregistration")
    if (
        prereg != identity
        or json_hash({k: v for k, v in identity.items() if k != "preregistration_sha256"})
        != identity["preregistration_sha256"]
    ):
        raise ValueError("CPR preregistration changed")
    files = {
        "train": output_dir / "preflight" / "composition_train.jsonl",
        "dev": output_dir / "preflight" / "composition_dev.jsonl",
        "audit": output_dir / "preflight" / "data_audit.json",
        "token_audit": output_dir / "preflight" / "training_token_audit.json",
    }
    for name, path in files.items():
        if file_hash(path) != identity["data_hashes"][name]:
            raise ValueError(f"CPR {name} artifact changed")
    return manifest, spec, identity


def _spec_from_dict(payload: dict[str, Any]) -> ExperimentSpec:
    from plasticity_placement.p0d2hcpr.config import DataSpec, GateSpec, TrainingSpec

    copy = dict(payload)
    data = DataSpec(**dict(copy.pop("data")))
    training_payload = dict(copy.pop("training"))
    training_payload["target_modules"] = tuple(training_payload["target_modules"])
    training = TrainingSpec(**training_payload)
    gates = GateSpec(**dict(copy.pop("gates")))
    return ExperimentSpec(**copy, data=data, training=training, gates=gates)


def _validate_failed_same_runtime(path: Path, rr_provenance: dict[str, Any]) -> dict[str, Any]:
    summary = read_json_object(path, "same-runtime summary")
    audit_manifest = read_json_object(
        path.parent / "audit_manifest.json",
        "same-runtime audit manifest",
    )
    if summary.get("schema_version") != "p0d2hrr-same-runtime-audit-v1":
        raise ValueError("same-runtime source schema changed")
    if (
        audit_manifest.get("schema_version") != "p0d2hrr-same-runtime-audit-manifest-v1"
        or audit_manifest.get("run_id") != summary.get("run_id")
        or audit_manifest.get("artifacts", {}).get("summary", {}).get("sha256") != file_hash(path)
        or audit_manifest.get("source_artifacts_modified") is not False
        or audit_manifest.get("mappings_per_adapter_authorized") is not False
    ):
        raise ValueError("same-runtime audit manifest does not bind the summary")
    source = summary.get("source", {}).get("route_remediation", {})
    if (
        source.get("run_id") != rr_provenance["run_id"]
        or source.get("adapter_sha256") != rr_provenance["adapter_sha256"]
    ):
        raise ValueError("same-runtime result is not linked to the source RR adapter")
    if summary.get("runtime_identity", {}).get("sentinel", {}).get("passed") is not True:
        raise ValueError("same-runtime OFF replay sentinel did not pass")
    if (
        summary.get("source_artifacts_modified") is not False
        or summary.get("source_gate_status_changed") is not False
        or summary.get("source", {})
        .get("route_remediation", {})
        .get("current_artifact_checks_passed")
        is not True
    ):
        raise ValueError("same-runtime source provenance is incomplete")
    decision = summary.get("decision", {})
    if decision.get("status") == "same_runtime_reading_qualification_candidate":
        raise ValueError("composition remediation is not justified by a passing source adapter")
    if decision.get("mappings_per_adapter_authorized") is not False:
        raise ValueError("source same-runtime result unexpectedly authorizes 1/4/8")
    return summary


def _build_training_token_audit(
    tokenizer: Any,
    records: list[Any],
    *,
    max_length: int,
) -> dict[str, Any]:
    audits: list[dict[str, Any]] = []
    for record in records:
        formatted = chat_prompt(tokenizer, record.prompt)
        prompt_ids = tokenizer(formatted, add_special_tokens=True)["input_ids"]
        completion_ids = tokenizer(record.completion, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            completion_ids = [*completion_ids, tokenizer.eos_token_id]
        surviving = max(0, min(len(completion_ids), max_length - len(prompt_ids)))
        fully_survives = surviving == len(completion_ids)
        audits.append(
            {
                "example_id": record.example_id,
                "split": record.split,
                "task": record.task,
                "prompt_token_count": len(prompt_ids),
                "completion_token_count": len(completion_ids),
                "completion_tokens_surviving": surviving,
                "input_truncated": len(prompt_ids) + len(completion_ids) > max_length,
                "completion_fully_survives": fully_survives,
            }
        )
    failed = [row["example_id"] for row in audits if not row["completion_fully_survives"]]
    return {
        "schema_version": "p0d2hcpr-training-token-audit-v1",
        "decision_count": len(audits),
        "max_length": max_length,
        "input_truncated_count": sum(row["input_truncated"] for row in audits),
        "failed_example_ids": failed,
        "all_checks_passed": not failed,
        "records": audits,
    }


def _require_independent(sources: tuple[Path, ...], output: Path) -> None:
    for source in sources:
        source = source.resolve()
        if source == output or source.is_relative_to(output) or output.is_relative_to(source):
            raise ValueError("CPR output must be independent from all source artifacts")
