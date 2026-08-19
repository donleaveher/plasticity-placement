from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.io import (
    canonical_json_bytes,
    file_hash,
    immutable_json_write,
    json_hash,
)
from plasticity_placement.pathmem_ropcd_p0.bundle import (
    load_plan_bundle,
    verify_recorded_plan_bundle,
)
from plasticity_placement.pathmem_ropcd_p0.config import (
    AUTHORIZATION_SCHEMA_VERSION,
    EXECUTION_MANIFEST_SCHEMA_VERSION,
    P0_ROPCD_RECIPE,
    execution_permissions,
)
from plasticity_placement.pathmem_ropcd_p0.identity import (
    implementation_identity,
    verify_external_approval_object,
)
from plasticity_placement.pathmem_ropcd_p0.manifest import P0ExecutionManifest
from plasticity_placement.pathmem_ropcd_p0.runtime import (
    RunContext,
    _artifact_integrity,
    _immutable_text_write,
    _load_rows,
    _render_report,
    _summary,
)

REPAIR_AUTHORIZATION_SCHEMA_VERSION = "pathmem-ropcd-p0-analysis-repair-authorization-v1"
REPAIR_MANIFEST_SCHEMA_VERSION = "pathmem-ropcd-p0-analysis-repair-manifest-v1"
REPAIR_RULE_ID = "qualification-null-obsolete-from-parametric-core-v1"


def repair_permissions() -> dict[str, bool]:
    return {
        "read_immutable_p0_results_authorized": True,
        "recompute_frozen_p0_aggregate_authorized": True,
        "source_artifact_mutation_authorized": False,
        "training_authorized": False,
        "gpu_inference_authorized": False,
        "candidate_rescoring_authorized": False,
        "p1_authorized": False,
        "kill_or_reserve_access_authorized": False,
        "learned_router_authorized": False,
        "rl_controller_authorized": False,
        "automatic_hyperparameter_search_authorized": False,
    }


def inspect_analysis_repair(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    plan_root: Path,
    output_root: Path,
    repair_output_root: Path,
) -> dict[str, Any]:
    _require_separate_output(output_root, repair_output_root)
    context, rows, integrity = _source_snapshot(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
        output_root=output_root,
    )
    missing = _qualification_null_count(rows)
    return {
        "passed": all(integrity.values()) and missing == 64,
        "repair_rule_id": REPAIR_RULE_ID,
        "source_run_id": context.manifest.payload["run_id"],
        "source_plan_id": context.bundle["plan"]["plan_id"],
        "source_matrix_sha256": json_hash(rows),
        "source_rows": len(rows),
        "source_boundary": _source_boundary(output_root),
        "qualification_rows_with_null_obsolete_action": missing,
        "repair_output_root": str(repair_output_root.resolve()),
        "permissions": repair_permissions(),
        "source_mutated": False,
        "training_started": False,
        "gpu_inference_started": False,
        "p1_authorized": False,
    }


def analysis_repair_authorization_template(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    plan_root: Path,
    output_root: Path,
    repair_output_root: Path,
) -> dict[str, Any]:
    _require_separate_output(output_root, repair_output_root)
    context, rows, integrity = _source_snapshot(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
        output_root=output_root,
    )
    if not all(integrity.values()):
        raise PermissionError("R-OPCD P0 analysis repair requires intact source artifacts")
    return _authorization_template_from_snapshot(
        context=context,
        rows=rows,
        output_root=output_root,
        repair_output_root=repair_output_root,
    )


def _authorization_template_from_snapshot(
    *,
    context: RunContext,
    rows: list[dict[str, Any]],
    output_root: Path,
    repair_output_root: Path,
) -> dict[str, Any]:
    null_rows = _qualification_null_count(rows)
    if null_rows != 64:
        raise ValueError(
            "R-OPCD P0 analysis repair requires exactly 64 null-obsolete "
            f"qualification rows, observed {null_rows}"
        )
    implementation, implementation_sha256 = implementation_identity()
    source_manifest = context.manifest
    return {
        "schema_version": REPAIR_AUTHORIZATION_SCHEMA_VERSION,
        "decision": "pending_human_review",
        "scope": "ropcd_p0_qualification_label_analysis_repair_only",
        "source": {
            "run_id": context.manifest.payload["run_id"],
            "plan_id": context.bundle["plan"]["plan_id"],
            "plan_manifest_id": source_manifest.payload["identity"]["plan_manifest_id"],
            "authorization_id": source_manifest.payload["identity"]["authorization_id"],
            "execution_implementation_sha256": source_manifest.payload["identity"][
                "implementation_sha256"
            ],
            "matrix_sha256": json_hash(rows),
            "rows": len(rows),
            **_source_boundary(output_root),
        },
        "repair_rule": {
            "repair_rule_id": REPAIR_RULE_ID,
            "missing_field": "path_qualification.obsolete_action",
            "recovery_source": "matching_parametric_path_core_rows",
            "required_reference_rows_per_item_path": 14,
            "required_qualification_rows_per_item_path": 4,
            "probability_outcomes_used_for_label_recovery": False,
            "failed_paths_filtered": False,
        },
        "implementation": implementation,
        "implementation_sha256": implementation_sha256,
        "source_output_root": str(output_root.resolve()),
        "repair_output_root": str(repair_output_root.resolve()),
        "permissions": repair_permissions(),
    }


def adopt_analysis_repair_authorization(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    plan_root: Path,
    output_root: Path,
    repair_output_root: Path,
    approval_path: Path,
) -> dict[str, Any]:
    _require_separate_output(output_root, repair_output_root)
    template = analysis_repair_authorization_template(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
        output_root=output_root,
        repair_output_root=repair_output_root,
    )
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    verified = verify_external_approval_object(approval, expected_template=template)
    repair_output_root.mkdir(parents=True, exist_ok=True)
    immutable_json_write(
        repair_output_root / "authorization.json",
        verified,
        "R-OPCD P0 analysis-repair authorization",
    )
    return {
        "adopted": True,
        "authorization_id": verified["authorization_id"],
        "source_run_id": template["source"]["run_id"],
        "repair_rule_id": REPAIR_RULE_ID,
        "permissions": verified["permissions"],
    }


def aggregate_analysis_repair(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    plan_root: Path,
    output_root: Path,
    repair_output_root: Path,
) -> dict[str, Any]:
    _require_separate_output(output_root, repair_output_root)
    boundary_before = _source_boundary(output_root)
    context, rows, integrity = _source_snapshot(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
        output_root=output_root,
    )
    template = _authorization_template_from_snapshot(
        context=context,
        rows=rows,
        output_root=output_root,
        repair_output_root=repair_output_root,
    )
    authorization = _verify_repair_authorization(
        repair_output_root=repair_output_root,
        expected_template=template,
    )
    summary = _repaired_summary(context, rows, integrity, authorization)
    if _source_boundary(output_root) != boundary_before:
        raise RuntimeError("R-OPCD P0 source boundary changed during analysis repair")
    aggregate_root = repair_output_root / "aggregate"
    summary_path = aggregate_root / "summary.json"
    report_path = aggregate_root / "report.md"
    immutable_json_write(summary_path, summary, "R-OPCD P0 repaired summary")
    _immutable_text_write(
        report_path,
        _repair_report(summary),
        "R-OPCD P0 repaired report",
    )
    manifest = _repair_manifest(
        context=context,
        rows=rows,
        authorization=authorization,
        summary_path=summary_path,
        report_path=report_path,
    )
    immutable_json_write(
        repair_output_root / "repair_manifest.json",
        manifest,
        "R-OPCD P0 analysis-repair manifest",
    )
    return summary


def verify_analysis_repair(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    plan_root: Path,
    output_root: Path,
    repair_output_root: Path,
) -> dict[str, Any]:
    _require_separate_output(output_root, repair_output_root)
    boundary_before = _source_boundary(output_root)
    context, rows, integrity = _source_snapshot(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
        output_root=output_root,
    )
    template = _authorization_template_from_snapshot(
        context=context,
        rows=rows,
        output_root=output_root,
        repair_output_root=repair_output_root,
    )
    authorization = _verify_repair_authorization(
        repair_output_root=repair_output_root,
        expected_template=template,
    )
    expected_summary = _repaired_summary(context, rows, integrity, authorization)
    if _source_boundary(output_root) != boundary_before:
        raise RuntimeError("R-OPCD P0 source boundary changed during repair verification")
    summary_path = repair_output_root / "aggregate/summary.json"
    report_path = repair_output_root / "aggregate/report.md"
    observed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if canonical_json_bytes(observed_summary) != canonical_json_bytes(expected_summary):
        raise ValueError("R-OPCD P0 repaired summary regeneration mismatch")
    if report_path.read_text(encoding="utf-8") != _repair_report(expected_summary):
        raise ValueError("R-OPCD P0 repaired report regeneration mismatch")
    expected_manifest = _repair_manifest(
        context=context,
        rows=rows,
        authorization=authorization,
        summary_path=summary_path,
        report_path=report_path,
    )
    observed_manifest = json.loads(
        (repair_output_root / "repair_manifest.json").read_text(encoding="utf-8")
    )
    if canonical_json_bytes(observed_manifest) != canonical_json_bytes(expected_manifest):
        raise ValueError("R-OPCD P0 analysis-repair manifest changed")
    return {
        "passed": True,
        "repair_id": expected_manifest["repair_id"],
        "repair_rule_id": REPAIR_RULE_ID,
        "source_run_id": context.manifest.payload["run_id"],
        "source_matrix_sha256": json_hash(rows),
        "g2_gate": expected_summary["gate"],
        "source_mutated": False,
        "training_started": False,
        "gpu_inference_started": False,
        "p1_authorized": False,
    }


def _source_snapshot(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    plan_root: Path,
    output_root: Path,
) -> tuple[RunContext, list[dict[str, Any]], dict[str, bool]]:
    plan_report = verify_recorded_plan_bundle(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
    )
    bundle = load_plan_bundle(plan_root)
    plan_manifest = bundle["manifest"]
    template = _recorded_execution_authorization_template(
        plan_root=plan_root,
        output_root=output_root,
        plan_report=plan_report,
        plan_manifest=plan_manifest,
    )
    authorization_payload = json.loads(
        (output_root / "authorization.json").read_text(encoding="utf-8")
    )
    authorization_id = authorization_payload.pop("authorization_id", None)
    authorization = verify_external_approval_object(
        authorization_payload,
        expected_template=template,
    )
    if authorization_id != authorization["authorization_id"]:
        raise ValueError("recorded R-OPCD P0 authorization identity changed")
    manifest = P0ExecutionManifest.load(output_root / "manifest.json")
    expected_identity = _execution_identity(
        plan_report=plan_report,
        bundle=bundle,
        authorization=authorization,
        implementation_sha256=str(plan_manifest["implementation_sha256"]),
    )
    if manifest.payload.get("identity") != expected_identity or manifest.payload.get(
        "run_id"
    ) != json_hash(expected_identity):
        raise ValueError("recorded R-OPCD P0 execution identity changed")
    manifest.require_all_verified()
    context = RunContext(
        plan_report=plan_report,
        bundle=bundle,
        authorization=authorization,
        manifest=manifest,
    )
    rows = _load_rows(output_root, context)
    integrity = _artifact_integrity(output_root, context, plan_report)
    if not all(integrity.values()):
        failed = sorted(name for name, passed in integrity.items() if not passed)
        raise ValueError(f"R-OPCD P0 source integrity failed: {failed}")
    return context, rows, integrity


def _recorded_execution_authorization_template(
    *,
    plan_root: Path,
    output_root: Path,
    plan_report: dict[str, Any],
    plan_manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "decision": "pending_human_review",
        "scope": "ropcd_specific_p0_smoke_only",
        "plan": {
            "manifest_id": plan_report["manifest_id"],
            "manifest_sha256": file_hash(plan_root / "manifest.json"),
            "plan_id": plan_report["plan_id"],
            "plan_sha256": file_hash(plan_root / "p0_plan.json"),
            "g1c_handoff_id": plan_report["g1c_handoff_id"],
            "g1c_run_id": plan_report["g1c_run_id"],
        },
        "recipe": P0_ROPCD_RECIPE.to_dict(),
        "recipe_sha256": json_hash(P0_ROPCD_RECIPE.to_dict()),
        "implementation": plan_manifest["implementation"],
        "implementation_sha256": plan_manifest["implementation_sha256"],
        "output_root": str(output_root.resolve()),
        "permissions": execution_permissions(),
    }


def _execution_identity(
    *,
    plan_report: dict[str, Any],
    bundle: dict[str, Any],
    authorization: dict[str, Any],
    implementation_sha256: str,
) -> dict[str, Any]:
    planned = {
        str(unit["unit_id"]): {
            "unit_id": str(unit["unit_id"]),
            "sequence_index": int(unit["sequence_index"]),
            "item_id": str(unit["item_id"]),
            "logical_name": str(unit["logical_name"]),
            "parent_unit_id": unit["parent_unit_id"],
            "technical_duplicate": bool(unit["technical_duplicate"]),
            "unit_sha256": json_hash(unit),
        }
        for unit in bundle["plan"]["units"]
    }
    return {
        "schema_version": EXECUTION_MANIFEST_SCHEMA_VERSION,
        "plan_manifest_id": plan_report["manifest_id"],
        "plan_id": bundle["plan"]["plan_id"],
        "g1c_handoff_id": bundle["plan"]["g1c_handoff"]["handoff_id"],
        "g1c_run_id": bundle["plan"]["g1c_handoff"]["run_id"],
        "authorization_id": authorization["authorization_id"],
        "recipe": bundle["plan"]["recipe"],
        "recipe_sha256": bundle["plan"]["recipe_sha256"],
        "implementation_sha256": implementation_sha256,
        "planned_units": planned,
    }


def _verify_repair_authorization(
    *,
    repair_output_root: Path,
    expected_template: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads((repair_output_root / "authorization.json").read_text(encoding="utf-8"))
    authorization_id = payload.pop("authorization_id", None)
    verified = verify_external_approval_object(payload, expected_template=expected_template)
    if authorization_id != verified["authorization_id"]:
        raise ValueError("R-OPCD P0 analysis-repair authorization identity changed")
    return verified


def _repaired_summary(
    context: RunContext,
    rows: list[dict[str, Any]],
    integrity: dict[str, bool],
    authorization: dict[str, Any],
) -> dict[str, Any]:
    summary = _summary(rows, integrity, context)
    return {
        **summary,
        "analysis_repair": {
            "repair_rule_id": REPAIR_RULE_ID,
            "authorization_id": authorization["authorization_id"],
            "source_matrix_sha256": json_hash(rows),
            "source_execution_manifest_sha256": authorization["source"][
                "source_execution_manifest_sha256"
            ],
            "source_execution_authorization_sha256": authorization["source"][
                "source_execution_authorization_sha256"
            ],
            "source_rows_mutated": False,
            "training_repeated": False,
            "gpu_inference_repeated": False,
        },
    }


def _repair_report(summary: dict[str, Any]) -> str:
    return (
        _render_report(summary)
        + "\n\n## Analysis repair\n\n"
        + f"- Rule: `{REPAIR_RULE_ID}`\n"
        + f"- Source matrix: `{summary['analysis_repair']['source_matrix_sha256']}`\n"
        + "- Source rows mutated: `False`\n"
        + "- Training or GPU inference repeated: `False`\n"
    )


def _repair_manifest(
    *,
    context: RunContext,
    rows: list[dict[str, Any]],
    authorization: dict[str, Any],
    summary_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    implementation, implementation_sha256 = implementation_identity()
    identity = {
        "schema_version": REPAIR_MANIFEST_SCHEMA_VERSION,
        "repair_rule_id": REPAIR_RULE_ID,
        "source_run_id": context.manifest.payload["run_id"],
        "source_plan_id": context.bundle["plan"]["plan_id"],
        "source_matrix_sha256": json_hash(rows),
        "source_execution_implementation_sha256": context.manifest.payload["identity"][
            "implementation_sha256"
        ],
        "source_execution_manifest_sha256": authorization["source"][
            "source_execution_manifest_sha256"
        ],
        "source_execution_authorization_sha256": authorization["source"][
            "source_execution_authorization_sha256"
        ],
        "authorization_id": authorization["authorization_id"],
        "implementation": implementation,
        "implementation_sha256": implementation_sha256,
        "files": {
            "aggregate/summary.json": file_hash(summary_path),
            "aggregate/report.md": file_hash(report_path),
        },
        "permissions": repair_permissions(),
        "source_mutated": False,
        "training_started": False,
        "gpu_inference_started": False,
        "p1_authorized": False,
    }
    return {**identity, "repair_id": json_hash(identity)}


def _require_separate_output(output_root: Path, repair_output_root: Path) -> None:
    source = output_root.resolve()
    repair = repair_output_root.resolve()
    if source == repair or repair.is_relative_to(source) or source.is_relative_to(repair):
        raise ValueError("analysis repair requires a separate output directory")


def _source_boundary(output_root: Path) -> dict[str, str]:
    return {
        "source_execution_manifest_sha256": file_hash(output_root / "manifest.json"),
        "source_execution_authorization_sha256": file_hash(output_root / "authorization.json"),
    }


def _qualification_null_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        row.get("arm") == "path_qualification" and row.get("obsolete_action") is None
        for row in rows
    )
