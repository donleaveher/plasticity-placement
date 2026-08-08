from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from plasticity_placement.p0d2hcbr.config import CURRICULA, PLACEMENTS, unit_id
from plasticity_placement.p0d2hcbr.preflight import load_config
from plasticity_placement.p0d2hcbr.runtime import validate_all_training_units
from plasticity_placement.p0d2hrr.io import (
    file_hash,
    immutable_json_write,
    immutable_write,
    read_json_object,
)

QUALIFIED = "binding_remediation_candidate_review_required"


def aggregate_experiment(output_dir: Path, evaluations_root: Path) -> Path:
    output_dir = output_dir.resolve()
    evaluations_root = evaluations_root.resolve()
    manifest, spec, identity = load_config(output_dir)
    if manifest.state != "evaluated":
        raise ValueError(f"CBR aggregation requires state=evaluated, found {manifest.state}")
    budget_audit = validate_all_training_units(output_dir)
    summaries = {
        key: _verify_unit_evaluation(
            evaluations_root / key,
            key,
            manifest.payload["evaluation_claims"][key],
            manifest.payload["run_id"],
        )
        for key in spec.unit_ids()
    }
    arms: dict[str, Any] = {}
    for curriculum in CURRICULA:
        for placement in PLACEMENTS:
            arm = f"{curriculum}__{placement}"
            values = [
                summaries[unit_id(curriculum, placement, seed)] for seed in spec.training.seeds
            ]
            heldout = [
                float(value["decision"]["values"]["heldout_tie_worst_accuracy"]) for value in values
            ]
            drift = [
                float(value["decision"]["values"]["guardrail_c_to_w"]["estimate"])
                for value in values
            ]
            arms[arm] = {
                "unit_ids": [value["unit"]["unit_id"] for value in values],
                "unit_statuses": [value["decision"]["status"] for value in values],
                "all_seeds_qualified": all(
                    value["decision"]["status"] == QUALIFIED for value in values
                ),
                "heldout_tie_worst_accuracy_by_seed": heldout,
                "heldout_tie_worst_accuracy_mean": mean(heldout),
                "guardrail_c_to_w_by_seed": drift,
                "guardrail_c_to_w_mean": mean(drift),
            }
    contrasts = _condition_contrasts(summaries, spec.training.seeds)
    qualified = sorted(name for name, value in arms.items() if value["all_seeds_qualified"])
    status = _classify_matrix(
        arms, contrasts, qualified, spec.gates.placement_noninferiority_margin
    )
    summary = {
        "schema_version": "p0d2hcbr-aggregate-v1",
        "run_id": manifest.payload["run_id"],
        "analysis_status": "cbr_matrix_complete",
        "source": {
            "preregistration_sha256": identity["preregistration_sha256"],
            "rabx_run_id": identity["rabx_run_id"],
        },
        "budget_audit": budget_audit,
        "arms": arms,
        "paired_seed_contrasts": contrasts,
        "decision": {
            "status": status,
            "qualified_arms": qualified,
            "requires_independent_review": bool(qualified),
            "additional_training_authorized": False,
            "mappings_per_adapter_authorized": False,
        },
        "unit_summary_hashes": {
            key: file_hash(evaluations_root / key / "summary.json") for key in summaries
        },
        "historical_rab_decision_changed": False,
        "historical_rabx_attribution_changed": False,
        "additional_training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    aggregate_root = output_dir / "aggregate"
    if aggregate_root.exists():
        raise FileExistsError(f"CBR aggregate output already exists: {aggregate_root}")
    summary_hash = immutable_json_write(
        aggregate_root / "summary.json", summary, "CBR aggregate summary"
    )
    report_hash = immutable_write(
        aggregate_root / "report.md", _render_report(summary).encode(), "CBR aggregate report"
    )
    audit = {
        "schema_version": "p0d2hcbr-aggregate-manifest-v1",
        "run_id": manifest.payload["run_id"],
        "preregistration_sha256": identity["preregistration_sha256"],
        "artifacts": {"summary": summary_hash, "report": report_hash},
        "unit_summary_hashes": summary["unit_summary_hashes"],
        "additional_training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    audit_hash = immutable_json_write(
        aggregate_root / "audit_manifest.json", audit, "CBR aggregate manifest"
    )
    manifest.payload["result"] = {
        "summary_sha256": summary_hash,
        "report_sha256": report_hash,
        "audit_manifest_sha256": audit_hash,
        "decision": status,
        "evaluations_root": str(evaluations_root),
    }
    manifest.set_state("complete")
    return aggregate_root / "summary.json"


def verify_complete_result(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    manifest, spec, identity = load_config(output_dir)
    if manifest.state != "complete":
        raise ValueError("CBR manifest is not complete")
    result = manifest.payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("CBR aggregate result is missing")
    root = output_dir / "aggregate"
    paths = {
        "summary": root / "summary.json",
        "report": root / "report.md",
        "audit_manifest": root / "audit_manifest.json",
    }
    expected = {
        "summary": result["summary_sha256"],
        "report": result["report_sha256"],
        "audit_manifest": result["audit_manifest_sha256"],
    }
    differences = {
        name: {"expected": expected[name], "observed": file_hash(path) if path.is_file() else None}
        for name, path in paths.items()
        if (file_hash(path) if path.is_file() else None) != expected[name]
    }
    if differences:
        raise ValueError(f"CBR aggregate artifacts changed: {differences}")
    audit = read_json_object(paths["audit_manifest"], "CBR aggregate manifest")
    summary = read_json_object(paths["summary"], "CBR aggregate summary")
    if (
        audit.get("schema_version") != "p0d2hcbr-aggregate-manifest-v1"
        or audit.get("run_id") != manifest.payload["run_id"]
        or audit.get("preregistration_sha256") != identity["preregistration_sha256"]
        or audit.get("artifacts") != {"summary": expected["summary"], "report": expected["report"]}
        or summary.get("analysis_status") != "cbr_matrix_complete"
        or summary.get("additional_training_authorized") is not False
        or summary.get("mappings_per_adapter_authorized") is not False
        or len(summary.get("unit_summary_hashes", {})) != len(spec.unit_ids())
    ):
        raise ValueError("CBR aggregate manifests disagree")
    evaluations_root = Path(result["evaluations_root"])
    for key in spec.unit_ids():
        _verify_unit_evaluation(
            evaluations_root / key,
            key,
            manifest.payload["evaluation_claims"][key],
            manifest.payload["run_id"],
        )
    return paths["summary"]


def _verify_unit_evaluation(
    root: Path, key: str, claim: dict[str, Any], cbr_run_id: str
) -> dict[str, Any]:
    summary_path = root / "summary.json"
    manifest_path = root / "evaluation_manifest.json"
    if claim.get("state") != "complete" or claim.get("analysis_output") != str(root):
        raise ValueError(f"CBR evaluation claim is incomplete: {key}")
    if file_hash(summary_path) != claim.get("summary_sha256"):
        raise ValueError(f"CBR unit summary changed: {key}")
    if file_hash(manifest_path) != claim.get("evaluation_manifest_sha256"):
        raise ValueError(f"CBR unit evaluation manifest changed: {key}")
    audit = read_json_object(manifest_path, f"CBR evaluation manifest {key}")
    summary = read_json_object(summary_path, f"CBR unit summary {key}")
    differences = {
        name: {
            "expected": value.get("sha256"),
            "observed": file_hash(Path(value["path"])) if Path(value["path"]).is_file() else None,
        }
        for name, value in audit.get("artifacts", {}).items()
        if (file_hash(Path(value["path"])) if Path(value["path"]).is_file() else None)
        != value.get("sha256")
    }
    if differences:
        raise ValueError(f"CBR unit artifacts changed for {key}: {differences}")
    if (
        audit.get("schema_version") != "p0d2hcbr-unit-evaluation-manifest-v1"
        or audit.get("run_id") != summary.get("run_id")
        or summary.get("cbr_run_id") != cbr_run_id
        or summary.get("unit", {}).get("unit_id") != key
        or summary.get("source_artifacts_modified") is not False
        or summary.get("training_authorized") is not False
        or summary.get("mappings_per_adapter_authorized") is not False
    ):
        raise ValueError(f"CBR unit evaluation manifests disagree: {key}")
    return summary


def _condition_contrasts(
    summaries: dict[str, dict[str, Any]], seeds: tuple[int, ...]
) -> dict[str, Any]:
    values: dict[str, Any] = defaultdict(dict)
    for placement in PLACEMENTS:
        label = f"disentangled_minus_coupled__{placement}"
        values[label] = {
            str(seed): _heldout(summaries[unit_id("disentangled", placement, seed)])
            - _heldout(summaries[unit_id("coupled", placement, seed)])
            for seed in seeds
        }
        values[label]["mean"] = mean(values[label][str(seed)] for seed in seeds)
    for curriculum in CURRICULA:
        label = f"late_minus_full__{curriculum}"
        values[label] = {
            str(seed): _heldout(summaries[unit_id(curriculum, "late_matched", seed)])
            - _heldout(summaries[unit_id(curriculum, "full_depth", seed)])
            for seed in seeds
        }
        values[label]["mean"] = mean(values[label][str(seed)] for seed in seeds)
        drift_label = f"late_minus_full_guardrail_c_to_w__{curriculum}"
        values[drift_label] = {
            str(seed): _drift(summaries[unit_id(curriculum, "late_matched", seed)])
            - _drift(summaries[unit_id(curriculum, "full_depth", seed)])
            for seed in seeds
        }
        values[drift_label]["mean"] = mean(values[drift_label][str(seed)] for seed in seeds)
    return dict(values)


def _classify_matrix(
    arms: dict[str, Any],
    contrasts: dict[str, Any],
    qualified: list[str],
    noninferiority_margin: float,
) -> str:
    late = "disentangled__late_matched"
    full = "disentangled__full_depth"
    if not qualified:
        return "no_condition_qualified"
    if late in qualified and full in qualified:
        binding = contrasts["late_minus_full__disentangled"]
        drift = contrasts["late_minus_full_guardrail_c_to_w__disentangled"]
        seed_keys = [key for key in binding if key != "mean"]
        if all(binding[key] >= -noninferiority_margin for key in seed_keys) and all(
            drift[key] <= 0.0 for key in seed_keys
        ):
            return "late_placement_reduces_drift"
        return "binding_remediation_supported_scope_tradeoff"
    if full in qualified:
        return "full_depth_required_for_binding"
    if late in qualified:
        return "late_binding_remediation_supported"
    if any(name.startswith("coupled__") for name in qualified):
        return "balanced_data_advantage_not_supported"
    return "binding_remediation_supported"


def _heldout(summary: dict[str, Any]) -> float:
    return float(summary["decision"]["values"]["heldout_tie_worst_accuracy"])


def _drift(summary: dict[str, Any]) -> float:
    return float(summary["decision"]["values"]["guardrail_c_to_w"]["estimate"])


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# P0-D2H-CBR-v1 aggregate",
        "",
        f"- Run: `{summary['run_id']}`",
        f"- Decision: `{summary['decision']['status']}`",
        f"- Qualified arms: `{', '.join(summary['decision']['qualified_arms']) or 'none'}`",
        "- Additional training authorized: `false`",
        "- 1/4/8 authorized: `false`",
        "",
        "| Arm | All seeds qualified | Held-out mean | Guardrail C→W mean |",
        "|---|---:|---:|---:|",
    ]
    for name, value in sorted(summary["arms"].items()):
        lines.append(
            f"| {name} | {str(value['all_seeds_qualified']).lower()} | "
            f"{value['heldout_tie_worst_accuracy_mean']:.4f} | "
            f"{value['guardrail_c_to_w_mean']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)
