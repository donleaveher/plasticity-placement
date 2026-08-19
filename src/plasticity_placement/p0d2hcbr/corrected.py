from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.runtime import current_code_hash
from plasticity_placement.p0d2hcbr.aggregate import verify_complete_result
from plasticity_placement.p0d2hcbr.authorization import template
from plasticity_placement.p0d2hcbr.config import (
    CONFIG_SCHEMA_VERSION,
    CORRECTED_CONFIG_SCHEMA_VERSION,
    CURRICULA,
    CorrectedExperimentSpec,
    unit_id,
)
from plasticity_placement.p0d2hcbr.manifest import Manifest
from plasticity_placement.p0d2hcbr.preflight import _preflight_paths, load_config
from plasticity_placement.p0d2hcbr.runtime import audit_all_training_units
from plasticity_placement.p0d2hrr.io import (
    file_hash,
    immutable_json_write,
    immutable_write,
    json_hash,
    read_json_object,
)
from plasticity_placement.p0d2hrr.runtime import _adapter_hash


def plan_corrected_experiment(
    *, output_dir: Path, source_cbr_output: Path, spec_path: Path
) -> Path:
    output_dir = output_dir.resolve()
    source_cbr_output = source_cbr_output.resolve()
    _require_independent(source_cbr_output, output_dir)
    if output_dir.exists():
        raise FileExistsError(f"corrected CBR output already exists: {output_dir}")
    source_summary_path = verify_complete_result(source_cbr_output)
    source_manifest, source_spec, source_identity = load_config(source_cbr_output)
    if (
        source_identity.get("schema_version") != CONFIG_SCHEMA_VERSION
        or source_manifest.state != "complete"
        or source_spec.training_run_limit != 12
    ):
        raise ValueError("corrected CBR requires the completed original CBR-v1 run")
    source_summary = read_json_object(source_summary_path, "CBR-v1 aggregate summary")
    if (
        source_summary.get("decision", {}).get("status")
        != "budget_deviation_full_matrix_complete"
        or source_summary.get("placement_comparison_scope")
        != "exploratory_parameter_count_confounded"
    ):
        raise ValueError("CBR-v1 source is not the completed authorized deviation analysis")
    source_budget = audit_all_training_units(source_cbr_output)
    if source_budget["all_checks_passed"] is not False:
        raise ValueError("corrected CBR source does not contain the registered budget deviation")
    spec = CorrectedExperimentSpec.from_path(spec_path.resolve())
    source_snapshot = corrected_source_snapshot(source_cbr_output)

    output_dir.mkdir(parents=True)
    source_paths = _preflight_paths(source_cbr_output)
    target_paths = _preflight_paths(output_dir)
    data_hashes: dict[str, str] = {}
    for name, source_path in source_paths.items():
        if name == "analysis_plan":
            continue
        expected = source_identity["data_hashes"].get(name)
        if expected is None or file_hash(source_path) != expected:
            raise ValueError(f"CBR-v1 preflight source changed: {name}")
        data_hashes[name] = immutable_write(
            target_paths[name], source_path.read_bytes(), f"corrected CBR preflight {name}"
        )
    analysis_plan = {
        "schema_version": "p0d2hcbr-corrected-analysis-plan-v2",
        "source_cbr_run_id": source_manifest.payload["run_id"],
        "imported_controls": "six immutable full-depth adapters",
        "new_training": "six late adapters on explicit layers 20-27 at rank 28 alpha 56",
        "exact_nominal_budget_identity": "28 layers * rank 8 == 8 layers * rank 28",
        "primary_panel": "same frozen 24 held-out groups x 64 factorial cells",
        "preservation_panels": ["historical_rab", "external_conditional_route", "full_crd"],
        "additional_training_authorization_is_external": True,
        "mappings_per_adapter_authorized": False,
    }
    analysis_plan["analysis_plan_sha256"] = json_hash(analysis_plan)
    data_hashes["analysis_plan"] = immutable_json_write(
        target_paths["analysis_plan"], analysis_plan, "corrected CBR analysis plan"
    )
    identity = {
        "schema_version": CORRECTED_CONFIG_SCHEMA_VERSION,
        "stage": "counterbalanced_binding_remediation_corrected",
        "code_sha256": current_code_hash(),
        "spec": spec.to_dict(),
        "spec_sha256": json_hash(spec.to_dict()),
        "source_cbr_output": str(source_cbr_output),
        "source_cbr_run_id": source_manifest.payload["run_id"],
        "source_cbr_snapshot": source_snapshot,
        "rabx_output": source_identity["rabx_output"],
        "rabx_run_id": source_identity["rabx_run_id"],
        "rabc_output": source_identity["rabc_output"],
        "rab_output": source_identity["rab_output"],
        "source_code_revision_lock": source_identity["source_code_revision_lock"],
        "source_code_revision_lock_sha256": source_identity[
            "source_code_revision_lock_sha256"
        ],
        "source_snapshot": source_identity["source_snapshot"],
        "corrected_source_snapshot": source_snapshot,
        "data_hashes": data_hashes,
        "training_run_limit": 6,
        "locked_evaluation_limit": 12,
        "imported_full_depth_unit_count": 6,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    identity["preregistration_sha256"] = json_hash(identity)
    immutable_json_write(
        output_dir / "preregistration.json", identity, "corrected CBR preregistration"
    )
    immutable_json_write(
        output_dir / "authorization.template.json",
        template(identity["preregistration_sha256"], allowed_training_runs=6),
        "corrected CBR authorization template",
    )
    manifest = Manifest.create(
        output_dir / "manifest.json",
        {
            "run_id": "p0d2hcbr2-" + identity["preregistration_sha256"][:10],
            "config": identity,
            "rabx_output": identity["rabx_output"],
            "source_cbr_output": str(source_cbr_output),
        },
        spec.unit_ids(),
    )
    for curriculum in CURRICULA:
        for seed in spec.training.seeds:
            key = unit_id(curriculum, "full_depth", seed)
            source_entry = source_manifest.payload["training_units"][key]
            source_adapter = source_cbr_output / "adapters" / key
            metadata_path = source_adapter / "training_metadata.json"
            metadata = read_json_object(metadata_path, f"imported CBR metadata {key}")
            training = deepcopy(source_entry["training"])
            training.update(
                {
                    "artifact_origin": "imported",
                    "adapter_dir": str(source_adapter),
                    "adapter_sha256": _adapter_hash(source_adapter),
                    "source_training_config": metadata["config"],
                    "training_data_sha256": metadata["summary"]["training_data_sha256"],
                    "source_cbr_run_id": source_manifest.payload["run_id"],
                }
            )
            manifest.payload["training_units"][key] = {
                "state": "trained",
                "training": training,
                "error": None,
            }
    manifest.save()
    return manifest.path


def corrected_source_snapshot(source_cbr_output: Path) -> dict[str, Any]:
    source_cbr_output = source_cbr_output.resolve()
    manifest = Manifest.load(source_cbr_output / "manifest.json")
    result = manifest.payload.get("result")
    if manifest.state != "complete" or not isinstance(result, dict):
        raise ValueError("corrected CBR source is not complete")
    imported: dict[str, Any] = {}
    for curriculum in CURRICULA:
        for seed in (20260810, 20260811, 20260812):
            key = unit_id(curriculum, "full_depth", seed)
            adapter = source_cbr_output / "adapters" / key
            imported[key] = {
                "adapter_sha256": _adapter_hash(adapter),
                "training_metadata_sha256": file_hash(adapter / "training_metadata.json"),
            }
    paths = {
        "preregistration": source_cbr_output / "preregistration.json",
        "authorization": source_cbr_output / "authorization.json",
        "budget_deviation_authorization": source_cbr_output
        / "budget_deviation_authorization.json",
        "aggregate_summary": source_cbr_output / "aggregate" / "summary.json",
        "aggregate_audit": source_cbr_output / "aggregate" / "audit_manifest.json",
        "manifest": source_cbr_output / "manifest.json",
    }
    return {
        "source_cbr_run_id": manifest.payload["run_id"],
        "artifacts": {name: file_hash(path) for name, path in paths.items()},
        "imported_full_depth": imported,
    }


def validate_corrected_source(identity: dict[str, Any]) -> None:
    source = Path(identity["source_cbr_output"])
    observed = corrected_source_snapshot(source)
    if observed != identity.get("source_cbr_snapshot"):
        raise ValueError("corrected CBR-v2 source artifacts changed")


def _require_independent(source: Path, output: Path) -> None:
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("corrected CBR output must be independent from CBR-v1 source")
