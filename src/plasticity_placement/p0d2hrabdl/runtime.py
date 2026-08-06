from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.runtime import (
    current_code_hash,
    current_environment_snapshot,
)
from plasticity_placement.p0d2hrabd.config import CATEGORIES, FACTORS
from plasticity_placement.p0d2hrabd.runtime import (
    verify_complete_result as verify_rabd_result,
)
from plasticity_placement.p0d2hrabdl.analysis import analyze_localization
from plasticity_placement.p0d2hrabdl.config import (
    EXPECTED_ADAPTER_TAXONOMY,
    EXPECTED_BASE_TAXONOMY,
    EXPECTED_TRANSITION_COUNTS,
    SOURCE_RUN_ID,
    TRANSITIONS,
    LocalizationSpec,
)
from plasticity_placement.p0d2hrr.io import (
    atomic_json_write,
    file_hash,
    immutable_json_write,
    immutable_write,
    json_hash,
    read_json_object,
)
from plasticity_placement.p0d2hrr.same_runtime_audit import _tree_hash

SCHEMA_VERSION = "p0d2hrabdl-localization-v1"
MANIFEST_SCHEMA_VERSION = "p0d2hrabdl-manifest-v1"


def plan_localization(*, output_dir: Path, rabd_output: Path) -> Path:
    output_dir = output_dir.resolve()
    rabd_output = rabd_output.resolve()
    _require_independent(rabd_output, output_dir)
    if output_dir.exists():
        raise FileExistsError(f"RAB localization output already exists: {output_dir}")
    spec = LocalizationSpec()
    source = _validate_rabd_result(rabd_output)
    analysis_plan = {
        "schema_version": "p0d2hrabdl-analysis-plan-v1",
        "spec": spec.to_dict(),
        "source_run_id": SOURCE_RUN_ID,
        "transition_order": list(TRANSITIONS),
        "taxonomy_order": list(CATEGORIES),
        "factor_order": list(FACTORS),
        "adverse_definition": "C→W or W→W",
        "margin_estimand": "adapter selected-minus-counterfactual margin minus base margin",
        "margin_uncertainty": "10000-sample pair_id cluster bootstrap within transition",
        "subgroup_inference": "descriptive_only_no_multiplicity_adjusted_claims",
        "qualification_gate": None,
        "allows_inference": False,
        "allows_training": False,
        "allows_historical_reclassification": False,
        "allows_mappings_per_adapter_scan": False,
    }
    analysis_plan["analysis_plan_sha256"] = json_hash(analysis_plan)
    output_dir.mkdir(parents=True)
    plan_hash = immutable_json_write(
        output_dir / "preflight" / "analysis_plan.json",
        analysis_plan,
        "RAB localization analysis plan",
    )
    identity = {
        "schema_version": "p0d2hrabdl-preregistration-v1",
        "code_sha256": current_code_hash(),
        "spec": spec.to_dict(),
        "rabd_output": str(rabd_output),
        "rabd_run_id": source["summary"]["run_id"],
        "rabd_summary_sha256": file_hash(rabd_output / "summary.json"),
        "source_snapshot": source["snapshot"],
        "analysis_plan_sha256": plan_hash,
        "historical_rab_decision_changed": False,
        "historical_rabd_status_changed": False,
        "inference_authorized": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    identity["preregistration_sha256"] = json_hash(identity)
    immutable_json_write(
        output_dir / "preregistration.json",
        identity,
        "RAB localization preregistration",
    )
    now = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": "p0d2hrabdl-" + identity["preregistration_sha256"][:10],
        "state": "planned",
        "created_at": now,
        "updated_at": now,
        "config": identity,
        "errors": [],
        "historical_rab_decision_changed": False,
        "historical_rabd_status_changed": False,
        "inference_authorized": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    atomic_json_write(output_dir / "manifest.json", manifest)
    return output_dir / "manifest.json"


def run_localization(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    manifest = _load_manifest(output_dir)
    if manifest.get("state") != "planned":
        raise PermissionError("RAB localization run requires planned state")
    identity = manifest["config"]
    if current_code_hash() != identity["code_sha256"]:
        raise ValueError("RAB localization code changed after planning")
    _verify_preflight(output_dir, identity)
    rabd_output = Path(identity["rabd_output"])
    source = _validate_rabd_result(rabd_output)
    if source["snapshot"] != identity["source_snapshot"]:
        raise ValueError("RABD source artifacts changed after localization planning")
    manifest["state"] = "running"
    _save_manifest(output_dir, manifest)
    try:
        cell_records = _read_jsonl(rabd_output / "cell_records.jsonl")
        unit_records = _read_jsonl(rabd_output / "unit_records.jsonl")
        source_factor_slices = read_json_object(
            rabd_output / "factor_slices.json", "RABD factor slices"
        )
        analysis, cell_localization, taxonomy, margins, hotspots = analyze_localization(
            source["summary"],
            source_factor_slices,
            cell_records,
            unit_records,
            LocalizationSpec(),
        )
        after = _source_snapshot(rabd_output)
        if after != identity["source_snapshot"]:
            raise RuntimeError("RABD source artifacts changed during localization")
        summary = {
            "schema_version": SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "source": {
                "rabd_run_id": source["summary"]["run_id"],
                "rabd_summary_sha256": file_hash(rabd_output / "summary.json"),
                "rabd_status": source["summary"]["analysis"]["analysis_status"],
            },
            "environment": current_environment_snapshot(),
            "analysis": analysis,
            "source_snapshot_before": identity["source_snapshot"],
            "source_snapshot_after": after,
            "source_artifacts_modified": False,
            "historical_rab_decision_changed": False,
            "historical_rabd_status_changed": False,
            "inference_authorized": False,
            "training_authorized": False,
            "mappings_per_adapter_authorized": False,
            "interpretation_boundary": (
                "This deterministic CPU-only reader is descriptive. It cannot reclassify RAB "
                "or RABD, load a model, authorize training, or authorize 1/4/8."
            ),
        }
        artifacts = _publish_analysis(
            output_dir,
            summary,
            cell_localization,
            taxonomy,
            margins,
            hotspots,
        )
        audit_manifest = {
            "schema_version": "p0d2hrabdl-audit-manifest-v1",
            "run_id": manifest["run_id"],
            "preregistration_sha256": identity["preregistration_sha256"],
            "source_snapshot_sha256": json_hash(after),
            "artifacts": artifacts,
            "historical_rab_decision_changed": False,
            "historical_rabd_status_changed": False,
            "inference_authorized": False,
            "training_authorized": False,
            "mappings_per_adapter_authorized": False,
        }
        audit_hash = immutable_json_write(
            output_dir / "audit_manifest.json",
            audit_manifest,
            "RAB localization audit manifest",
        )
        manifest = _load_manifest(output_dir)
        manifest["state"] = "complete"
        manifest["result"] = {
            "summary_sha256": artifacts["summary"],
            "audit_manifest_sha256": audit_hash,
            "analysis_status": analysis["analysis_status"],
        }
        _save_manifest(output_dir, manifest)
    except BaseException as error:
        _mark_failed(output_dir, error)
        raise
    return output_dir / "summary.json"


def verify_complete_result(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    manifest = _load_manifest(output_dir)
    if manifest.get("state") != "complete":
        raise ValueError("RAB localization manifest is not complete")
    _verify_preflight(output_dir, manifest["config"])
    audit_path = output_dir / "audit_manifest.json"
    if not audit_path.is_file() or file_hash(audit_path) != manifest.get("result", {}).get(
        "audit_manifest_sha256"
    ):
        raise ValueError("RAB localization audit manifest is missing or changed")
    audit = read_json_object(audit_path, "RAB localization audit manifest")
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
        raise ValueError(f"RAB localization artifacts changed: {differences}")
    if (
        audit.get("schema_version") != "p0d2hrabdl-audit-manifest-v1"
        or audit.get("run_id") != manifest.get("run_id")
        or audit.get("preregistration_sha256")
        != manifest.get("config", {}).get("preregistration_sha256")
        or audit.get("source_snapshot_sha256")
        != json_hash(manifest.get("config", {}).get("source_snapshot"))
        or manifest.get("historical_rab_decision_changed") is not False
        or manifest.get("historical_rabd_status_changed") is not False
        or manifest.get("inference_authorized") is not False
        or manifest.get("training_authorized") is not False
        or manifest.get("mappings_per_adapter_authorized") is not False
        or audit.get("historical_rab_decision_changed") is not False
        or audit.get("historical_rabd_status_changed") is not False
        or audit.get("inference_authorized") is not False
        or audit.get("training_authorized") is not False
        or audit.get("mappings_per_adapter_authorized") is not False
        or manifest.get("result", {}).get("summary_sha256")
        != audit.get("artifacts", {}).get("summary")
    ):
        raise ValueError("RAB localization complete manifests disagree")
    summary = read_json_object(required["summary"], "RAB localization summary")
    identity = manifest["config"]
    source = summary.get("source", {})
    if (
        summary.get("run_id") != manifest.get("run_id")
        or summary.get("analysis", {}).get("analysis_status")
        != manifest.get("result", {}).get("analysis_status")
        or summary.get("historical_rab_decision_changed") is not False
        or summary.get("historical_rabd_status_changed") is not False
        or summary.get("inference_authorized") is not False
        or summary.get("training_authorized") is not False
        or summary.get("mappings_per_adapter_authorized") is not False
        or summary.get("source_artifacts_modified") is not False
        or summary.get("source_snapshot_before") != identity.get("source_snapshot")
        or summary.get("source_snapshot_after") != identity.get("source_snapshot")
        or source.get("rabd_run_id") != identity.get("rabd_run_id")
        or source.get("rabd_summary_sha256") != identity.get("rabd_summary_sha256")
        or source.get("rabd_status") != "descriptive_error_topology_complete"
        or _jsonl_row_count(required["unit_hotspots"]) != 48
    ):
        raise ValueError("RAB localization summary or hotspot count is inconsistent")
    return required["summary"]


def _validate_rabd_result(rabd_output: Path) -> dict[str, Any]:
    verify_rabd_result(rabd_output)
    manifest = read_json_object(rabd_output / "manifest.json", "RABD manifest")
    identity = read_json_object(rabd_output / "preregistration.json", "RABD preregistration")
    summary = read_json_object(rabd_output / "summary.json", "RABD summary")
    analysis = summary.get("analysis", {})
    if (
        manifest.get("config") != identity
        or manifest.get("state") != "complete"
        or summary.get("run_id") != SOURCE_RUN_ID
        or analysis.get("analysis_status") != "descriptive_error_topology_complete"
        or analysis.get("correctness_transitions") != EXPECTED_TRANSITION_COUNTS
        or analysis.get("unit_taxonomy", {}).get("base", {}).get("counts") != EXPECTED_BASE_TAXONOMY
        or analysis.get("unit_taxonomy", {}).get("adapter", {}).get("counts")
        != EXPECTED_ADAPTER_TAXONOMY
        or analysis.get("integrity", {}).get("all_checks_passed") is not True
        or summary.get("source_artifacts_modified") is not False
        or summary.get("source_snapshot_before") != summary.get("source_snapshot_after")
        or summary.get("historical_rab_decision_changed") is not False
        or summary.get("inference_authorized") is not False
        or summary.get("training_authorized") is not False
        or summary.get("mappings_per_adapter_authorized") is not False
    ):
        raise ValueError("localization source is not the reviewed completed RABD result")
    return {
        "manifest": manifest,
        "identity": identity,
        "summary": summary,
        "snapshot": _source_snapshot(rabd_output),
    }


def _source_snapshot(rabd_output: Path) -> dict[str, Any]:
    return {"rabd_output_tree": _tree_hash(rabd_output)}


def _publish_analysis(
    output: Path,
    summary: dict[str, Any],
    cell_localization: dict[str, Any],
    taxonomy: dict[str, Any],
    margins: dict[str, Any],
    hotspots: list[dict[str, Any]],
) -> dict[str, str]:
    return {
        "summary": immutable_json_write(
            output / "summary.json", summary, "RAB localization summary"
        ),
        "report": immutable_write(
            output / "report.md", _render_report(summary).encode(), "RAB localization report"
        ),
        "cell_transition_localization": immutable_json_write(
            output / "cell_transition_localization.json",
            cell_localization,
            "RAB cell-transition localization",
        ),
        "taxonomy_migrations": immutable_json_write(
            output / "taxonomy_migrations.json",
            taxonomy,
            "RAB taxonomy migrations",
        ),
        "margin_transition_profiles": immutable_json_write(
            output / "margin_transition_profiles.json",
            margins,
            "RAB margin-transition profiles",
        ),
        "unit_hotspots": immutable_write(
            output / "unit_hotspots.jsonl",
            _jsonl(hotspots),
            "RAB unit hotspots",
        ),
    }


def _render_report(summary: dict[str, Any]) -> str:
    analysis = summary["analysis"]
    lines = [
        "# RAB error-localization reader",
        "",
        f"- Run: `{summary['run_id']}`",
        "- Status: `descriptive_error_localization_complete`",
        "- Historical RAB/RABD status changed: `false`",
        "- Model inference performed: `false`",
        "- Training authorized: `false`",
        "- 1/4/8 authorized: `false`",
        "",
        "## Correctness-transition totals",
        "",
        "| C→C | C→W | W→C | W→W |",
        "|---:|---:|---:|---:|",
        f"| {analysis['transition_totals']['C→C']} | "
        f"{analysis['transition_totals']['C→W']} | "
        f"{analysis['transition_totals']['W→C']} | "
        f"{analysis['transition_totals']['W→W']} |",
        "",
        "## Highest adverse level per factor",
        "",
        "| Factor | Level | Adverse | Share | C→W | W→W |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for factor in FACTORS:
        top = analysis["top_adverse_levels"][factor][0]
        lines.append(
            f"| {factor} | {top['level']} | {top['adverse_count']} | "
            f"{top['adverse_share']:.4f} | {top['C→W']} | {top['W→W']} |"
        )
    lines.extend(
        [
            "",
            "## Margin change by transition",
            "",
            "| Transition | N | Mean change | 95% pair-cluster CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for transition in TRANSITIONS:
        profile = analysis["margin_transition_profiles"][transition]
        interval = profile["mean_change_pair_cluster_ci"]["ci95"]
        lines.append(
            f"| {transition} | {profile['observation_count']} | "
            f"{profile['mean_change']:.6f} | [{interval[0]:.6f}, {interval[1]:.6f}] |"
        )
    lines.extend(
        [
            "",
            "All rankings and factor slices are descriptive. This reader cannot reclassify "
            "RAB/RABD or authorize training/1/4/8.",
            "",
        ]
    )
    return "\n".join(lines)


def _verify_preflight(output: Path, identity: dict[str, Any]) -> None:
    plan_path = output / "preflight" / "analysis_plan.json"
    if file_hash(plan_path) != identity["analysis_plan_sha256"]:
        raise ValueError("RAB localization analysis plan changed")
    prereg = read_json_object(output / "preregistration.json", "localization preregistration")
    unhashed = {key: value for key, value in identity.items() if key != "preregistration_sha256"}
    if prereg != identity or json_hash(unhashed) != identity["preregistration_sha256"]:
        raise ValueError("RAB localization preregistration changed")
    plan = read_json_object(plan_path, "RAB localization analysis plan")
    unhashed_plan = {key: value for key, value in plan.items() if key != "analysis_plan_sha256"}
    if json_hash(unhashed_plan) != plan.get("analysis_plan_sha256"):
        raise ValueError("RAB localization analysis plan self-hash changed")


def _published_artifact_paths(output: Path) -> dict[str, Path]:
    return {
        "summary": output / "summary.json",
        "report": output / "report.md",
        "cell_transition_localization": output / "cell_transition_localization.json",
        "taxonomy_migrations": output / "taxonomy_migrations.json",
        "margin_transition_profiles": output / "margin_transition_profiles.json",
        "unit_hotspots": output / "unit_hotspots.jsonl",
    }


def _load_manifest(output: Path) -> dict[str, Any]:
    manifest = read_json_object(output / "manifest.json", "RAB localization manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("not an RAB localization manifest")
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
        raise ValueError("RAB localization output must be independent of RABD source")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    return (payload + "\n").encode()


def _jsonl_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)
