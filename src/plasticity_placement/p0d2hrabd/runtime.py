from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.runtime import (
    current_code_hash,
    current_environment_snapshot,
)
from plasticity_placement.p0d2hrab.runtime import (
    _validate_rsh_result,
)
from plasticity_placement.p0d2hrab.runtime import (
    _verify_preflight as _verify_rab_preflight,
)
from plasticity_placement.p0d2hrab.runtime import (
    verify_complete_result as verify_rab_result,
)
from plasticity_placement.p0d2hrabd.analysis import analyze_error_topology
from plasticity_placement.p0d2hrabd.config import CATEGORIES, CELLS, FACTORS, DiagnosticSpec
from plasticity_placement.p0d2hrr.io import (
    atomic_json_write,
    file_hash,
    immutable_json_write,
    immutable_write,
    json_hash,
    read_json_object,
)
from plasticity_placement.p0d2hrr.same_runtime_audit import _tree_hash

SCHEMA_VERSION = "p0d2hrabd-diagnostic-v1"
MANIFEST_SCHEMA_VERSION = "p0d2hrabd-manifest-v1"


def plan_diagnostic(*, output_dir: Path, rab_output: Path) -> Path:
    output_dir = output_dir.resolve()
    rab_output = rab_output.resolve()
    _require_independent(rab_output, output_dir)
    if output_dir.exists():
        raise FileExistsError(f"RAB error-topology output already exists: {output_dir}")
    spec = DiagnosticSpec()
    source = _validate_rab_result(rab_output)
    analysis_plan = {
        "schema_version": "p0d2hrabd-analysis-plan-v1",
        "spec": spec.to_dict(),
        "fixed_cell_order": list(CELLS),
        "taxonomy_priority": list(CATEGORIES),
        "factor_slices": list(FACTORS),
        "margin_score": "candidate sum_logprob(selected) - sum_logprob(counterfactual)",
        "overall_uncertainty": "10000-sample pair_id cluster bootstrap",
        "factor_slice_uncertainty": "descriptive_only_no_multiplicity_adjusted_claims",
        "qualification_gate": None,
        "allows_inference": False,
        "allows_training": False,
        "allows_historical_rab_reclassification": False,
        "allows_mappings_per_adapter_scan": False,
    }
    analysis_plan["analysis_plan_sha256"] = json_hash(analysis_plan)
    output_dir.mkdir(parents=True)
    plan_hash = immutable_json_write(
        output_dir / "preflight" / "analysis_plan.json",
        analysis_plan,
        "RAB error-topology analysis plan",
    )
    identity = {
        "schema_version": "p0d2hrabd-preregistration-v1",
        "code_sha256": current_code_hash(),
        "spec": spec.to_dict(),
        "rab_output": str(rab_output),
        "rab_run_id": source["summary"]["run_id"],
        "rab_summary_sha256": file_hash(rab_output / "summary.json"),
        "source_snapshot": source["snapshot"],
        "analysis_plan_sha256": plan_hash,
        "historical_rab_decision_changed": False,
        "inference_authorized": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    identity["preregistration_sha256"] = json_hash(identity)
    immutable_json_write(
        output_dir / "preregistration.json", identity, "RAB error-topology preregistration"
    )
    now = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": "p0d2hrabd-" + identity["preregistration_sha256"][:10],
        "state": "planned",
        "created_at": now,
        "updated_at": now,
        "config": identity,
        "errors": [],
        "historical_rab_decision_changed": False,
        "inference_authorized": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    atomic_json_write(output_dir / "manifest.json", manifest)
    return output_dir / "manifest.json"


def run_diagnostic(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    manifest = _load_manifest(output_dir)
    if manifest["state"] != "planned":
        raise PermissionError("RAB error-topology run requires planned state")
    identity = manifest["config"]
    if current_code_hash() != identity["code_sha256"]:
        raise ValueError("RAB error-topology code changed after planning")
    _verify_preflight(output_dir, identity)
    rab_output = Path(identity["rab_output"])
    source = _validate_rab_result(rab_output)
    if source["snapshot"] != identity["source_snapshot"]:
        raise ValueError("RAB source artifacts changed after diagnostic planning")
    manifest["state"] = "running"
    _save_manifest(output_dir, manifest)
    try:
        paired_records = _read_jsonl(rab_output / "paired_records.jsonl")
        adapter_off = _read_jsonl(rab_output / "results" / "adapter_off.jsonl")
        adapter_on = _read_jsonl(rab_output / "results" / "adapter_on.jsonl")
        binding_probes = _read_jsonl(rab_output / "preflight" / "binding_probes.jsonl")
        lesson_types = _lesson_type_map(Path(source["identity"]["rsh_output"]))
        analysis, cell_records, unit_records = analyze_error_topology(
            paired_records,
            adapter_off,
            adapter_on,
            binding_probes,
            lesson_types,
            DiagnosticSpec(),
        )
        after = _source_snapshot(rab_output)
        if after != identity["source_snapshot"]:
            raise RuntimeError("RAB source artifacts changed during error-topology analysis")
        summary = {
            "schema_version": SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "source": {
                "rab_run_id": source["summary"]["run_id"],
                "rab_summary_sha256": file_hash(rab_output / "summary.json"),
                "rab_decision": source["summary"]["analysis"]["decision"]["status"],
            },
            "environment": current_environment_snapshot(),
            "analysis": analysis,
            "source_snapshot_before": identity["source_snapshot"],
            "source_snapshot_after": after,
            "source_artifacts_modified": False,
            "historical_rab_decision_changed": False,
            "inference_authorized": False,
            "training_authorized": False,
            "mappings_per_adapter_authorized": False,
            "interpretation_boundary": (
                "This CPU-only post-hoc audit is descriptive. It cannot reclassify RAB, load a "
                "model, authorize training, or authorize the 1/4/8 experiment."
            ),
        }
        artifact_hashes = _publish_analysis(
            output_dir,
            summary,
            cell_records,
            unit_records,
            analysis["factor_slices"],
        )
        audit_manifest = {
            "schema_version": "p0d2hrabd-audit-manifest-v1",
            "run_id": manifest["run_id"],
            "preregistration_sha256": identity["preregistration_sha256"],
            "source_snapshot_sha256": json_hash(after),
            "artifacts": artifact_hashes,
            "historical_rab_decision_changed": False,
            "inference_authorized": False,
            "training_authorized": False,
            "mappings_per_adapter_authorized": False,
        }
        audit_manifest_sha256 = immutable_json_write(
            output_dir / "audit_manifest.json",
            audit_manifest,
            "RAB error-topology audit manifest",
        )
        manifest = _load_manifest(output_dir)
        manifest["state"] = "complete"
        manifest["result"] = {
            "summary_sha256": artifact_hashes["summary"],
            "audit_manifest_sha256": audit_manifest_sha256,
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
        raise ValueError("RAB error-topology manifest is not complete")
    _verify_preflight(output_dir, manifest["config"])
    audit_path = output_dir / "audit_manifest.json"
    if not audit_path.is_file() or file_hash(audit_path) != manifest.get("result", {}).get(
        "audit_manifest_sha256"
    ):
        raise ValueError("RAB error-topology audit manifest is missing or changed")
    audit = read_json_object(audit_path, "RAB error-topology audit manifest")
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
        raise ValueError(f"RAB error-topology artifacts changed: {differences}")
    if (
        audit.get("schema_version") != "p0d2hrabd-audit-manifest-v1"
        or audit.get("run_id") != manifest.get("run_id")
        or audit.get("preregistration_sha256")
        != manifest.get("config", {}).get("preregistration_sha256")
        or audit.get("inference_authorized") is not False
        or audit.get("training_authorized") is not False
        or audit.get("mappings_per_adapter_authorized") is not False
        or manifest.get("result", {}).get("summary_sha256") != audit["artifacts"]["summary"]
    ):
        raise ValueError("RAB error-topology complete manifests disagree")
    summary = read_json_object(required["summary"], "RAB error-topology summary")
    if (
        summary.get("run_id") != manifest.get("run_id")
        or summary.get("analysis", {}).get("analysis_status")
        != manifest.get("result", {}).get("analysis_status")
        or summary.get("historical_rab_decision_changed") is not False
        or summary.get("inference_authorized") is not False
        or summary.get("training_authorized") is not False
        or summary.get("mappings_per_adapter_authorized") is not False
        or _jsonl_row_count(required["cell_records"]) != 192
        or _jsonl_row_count(required["unit_records"]) != 48
    ):
        raise ValueError("RAB error-topology summary or record counts are inconsistent")
    return required["summary"]


def _validate_rab_result(rab_output: Path) -> dict[str, Any]:
    verify_rab_result(rab_output)
    manifest = read_json_object(rab_output / "manifest.json", "RAB manifest")
    identity = read_json_object(rab_output / "preregistration.json", "RAB preregistration")
    summary = read_json_object(rab_output / "summary.json", "RAB summary")
    _verify_rab_preflight(rab_output, identity)
    decision = summary.get("analysis", {}).get("decision", {})
    if (
        manifest.get("config") != identity
        or decision.get("status") != "binding_not_supported"
        or decision.get("scoring_integrity") is not True
        or decision.get("binding_accuracy_passed") is not False
        or decision.get("causal_specificity_passed") is not True
        or decision.get("orientation_robustness_passed") is not False
        or decision.get("adapter_noninferiority_passed") is not True
        or summary.get("historical_rsh_decision_changed") is not False
        or summary.get("training_authorized") is not False
        or summary.get("mappings_per_adapter_authorized") is not False
    ):
        raise ValueError("error-topology source is not the reviewed completed RAB result")
    rsh = _validate_rsh_result(Path(identity["rsh_output"]))
    if summary.get("source_snapshot_after") != rsh["snapshot"]:
        raise ValueError("RAB upstream RSH artifacts changed")
    return {
        "manifest": manifest,
        "identity": identity,
        "summary": summary,
        "rsh": rsh,
        "snapshot": _source_snapshot(rab_output),
    }


def _lesson_type_map(rsh_output: Path) -> dict[str, str]:
    rows = _read_jsonl(rsh_output / "preflight" / "handoff_probes.jsonl")
    grouped: dict[str, set[str]] = {}
    for row in rows:
        grouped.setdefault(str(row["pair_id"]), set()).add(str(row["lesson_type"]))
    if len(grouped) != 12 or any(len(values) != 1 for values in grouped.values()):
        raise ValueError("RSH lesson types are not unique within 12 frozen pairs")
    return {pair_id: next(iter(values)) for pair_id, values in grouped.items()}


def _source_snapshot(rab_output: Path) -> dict[str, Any]:
    return {"rab_output_tree": _tree_hash(rab_output)}


def _publish_analysis(
    output: Path,
    summary: dict[str, Any],
    cell_records: list[dict[str, Any]],
    unit_records: list[dict[str, Any]],
    factor_slices: dict[str, Any],
) -> dict[str, str]:
    return {
        "summary": immutable_json_write(
            output / "summary.json", summary, "RAB error-topology summary"
        ),
        "report": immutable_write(
            output / "report.md",
            _render_report(summary).encode(),
            "RAB error-topology report",
        ),
        "cell_records": immutable_write(
            output / "cell_records.jsonl",
            _jsonl(cell_records),
            "RAB topology cell records",
        ),
        "unit_records": immutable_write(
            output / "unit_records.jsonl",
            _jsonl(unit_records),
            "RAB topology unit records",
        ),
        "factor_slices": immutable_json_write(
            output / "factor_slices.json", factor_slices, "RAB topology factor slices"
        ),
    }


def _render_report(summary: dict[str, Any]) -> str:
    analysis = summary["analysis"]
    transitions = analysis["correctness_transitions"]
    margins = analysis["margin_summary"]
    lines = [
        "# RAB paired error-topology audit",
        "",
        f"- Run: `{summary['run_id']}`",
        "- Status: `descriptive_error_topology_complete`",
        "- Historical RAB decision changed: `false`",
        "- Model inference performed: `false`",
        "- Training authorized: `false`",
        "- 1/4/8 authorized: `false`",
        "",
        "## Cell correctness transitions",
        "",
        "| C→C | C→W | W→C | W→W | Net gain |",
        "|---:|---:|---:|---:|---:|",
        f"| {transitions['C→C']} | {transitions['C→W']} | {transitions['W→C']} | "
        f"{transitions['W→W']} | {transitions['W→C'] - transitions['C→W']} |",
        "",
        "## Unit topology",
        "",
        "| Category | OFF | ON |",
        "|---|---:|---:|",
    ]
    for category in CATEGORIES:
        lines.append(
            f"| {category} | {analysis['unit_taxonomy']['base']['counts'][category]} | "
            f"{analysis['unit_taxonomy']['adapter']['counts'][category]} |"
        )
    difference = margins["adapter_minus_base"]
    lines.extend(
        [
            "",
            "## Selected-minus-counterfactual margin",
            "",
            f"- OFF mean: {margins['base']['estimate']:.6f} "
            f"[{margins['base']['ci95'][0]:.6f}, {margins['base']['ci95'][1]:.6f}].",
            f"- ON mean: {margins['adapter']['estimate']:.6f} "
            f"[{margins['adapter']['ci95'][0]:.6f}, "
            f"{margins['adapter']['ci95'][1]:.6f}].",
            f"- ON−OFF: {difference['estimate']:.6f} "
            f"[{difference['ci95'][0]:.6f}, {difference['ci95'][1]:.6f}].",
            "",
            f"Dominant ON noncompliant category: "
            f"`{analysis['unit_taxonomy']['adapter']['dominant_noncompliant_category']}` "
            f"({analysis['unit_taxonomy']['adapter']['dominant_noncompliant_count']}/48 units).",
            "",
            "All factor slices are descriptive. This audit cannot reclassify RAB or authorize "
            "training/1/4/8.",
            "",
        ]
    )
    return "\n".join(lines)


def _verify_preflight(output: Path, identity: dict[str, Any]) -> None:
    if file_hash(output / "preflight" / "analysis_plan.json") != identity["analysis_plan_sha256"]:
        raise ValueError("RAB error-topology analysis plan changed")
    prereg = read_json_object(output / "preregistration.json", "topology preregistration")
    unhashed = {key: value for key, value in identity.items() if key != "preregistration_sha256"}
    if prereg != identity or json_hash(unhashed) != identity["preregistration_sha256"]:
        raise ValueError("RAB error-topology preregistration changed")
    plan = read_json_object(output / "preflight" / "analysis_plan.json", "analysis plan")
    unhashed_plan = {key: value for key, value in plan.items() if key != "analysis_plan_sha256"}
    if json_hash(unhashed_plan) != plan.get("analysis_plan_sha256"):
        raise ValueError("RAB error-topology analysis plan self-hash changed")


def _published_artifact_paths(output: Path) -> dict[str, Path]:
    return {
        "summary": output / "summary.json",
        "report": output / "report.md",
        "cell_records": output / "cell_records.jsonl",
        "unit_records": output / "unit_records.jsonl",
        "factor_slices": output / "factor_slices.json",
    }


def _load_manifest(output: Path) -> dict[str, Any]:
    manifest = read_json_object(output / "manifest.json", "RAB error-topology manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("not an RAB error-topology manifest")
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
        raise ValueError("RAB error-topology output must be independent of RAB source")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    return (payload + "\n").encode()


def _jsonl_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)
