from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from plasticity_placement.pathmem.io import file_hash
from plasticity_placement.pathmem_exec.artifacts import read_jsonl

G1_DIAGNOSTIC_VERSION = "pathmem-g1-posthoc-diagnostic-v1"


def diagnose_g1_run(run_root: Path) -> dict[str, Any]:
    """Build a read-only diagnosis from a completed G1 run, including Recipe A."""
    summary_path = run_root / "summary.json"
    manifest_path = run_root / "manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("G1 diagnosis requires summary.json and manifest.json")
    summary = _read_object(summary_path)
    manifest = _read_object(manifest_path)
    rows = _load_verified_rows(run_root, manifest)
    lineages = _load_lineages(run_root)
    diagnostic = build_g1_diagnostics(rows, lineages=lineages)
    expected_rows = summary.get("counts", {}).get("rows")
    expected_run_id = summary.get("run_id")
    observed_run_ids = sorted({str(row.get("run_id")) for row in rows})
    required_source_checks = {
        "summary_row_count_matches": len(rows) == expected_rows,
        "single_run_id_matches_summary": observed_run_ids == [expected_run_id],
        "manifest_run_id_matches_summary": manifest.get("run_id") == expected_run_id,
    }
    if not all(required_source_checks.values()):
        failed = sorted(name for name, passed in required_source_checks.items() if not passed)
        raise ValueError(f"G1 diagnostic source-integrity checks failed: {failed}")
    source_checks = {
        **required_source_checks,
        "verified_unit_result_hashes_verified": True,
        "control_hash_anchored_in_manifest": manifest.get("control_results_sha256")
        is not None,
    }
    return {
        "schema_version": G1_DIAGNOSTIC_VERSION,
        "source": {
            "run_root": str(run_root),
            "run_id": expected_run_id,
            "summary_sha256": file_hash(summary_path),
            "recipe_id": summary.get("recipe_id", "A-legacy"),
            "gate_passed": summary.get("gate", {}).get("passed"),
            "source_checks": source_checks,
        },
        **diagnostic,
        "scope": (
            "post-hoc descriptive diagnosis only; it does not alter G1 gates or authorize P0"
        ),
    }


def build_g1_diagnostics(
    rows: list[dict[str, Any]], *, lineages: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    parametric = [row for row in rows if row.get("arm") == "parametric_current"]
    bare_base = [row for row in rows if row.get("arm") == "bare_base"]
    base_unrelated = [row for row in rows if row.get("arm") == "base_unrelated"]
    parametric_unrelated = [row for row in rows if row.get("arm") == "parametric_unrelated"]
    if not parametric:
        raise ValueError("G1 diagnosis found no parametric_current rows")

    base_by_key = _unique_by_key(bare_base)
    action_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    paired_deltas: dict[str, list[float]] = defaultdict(list)
    for row in parametric:
        expected = str(row["expected_action"])
        predicted = str(row["predicted_action"])
        action_rows[expected].append(row)
        confusion[expected][predicted] += 1
        base = base_by_key[_row_key(row)]
        paired_deltas[expected].append(
            _probability(row, expected) - _probability(base, expected)
        )

    per_action = {}
    for action, observed in sorted(action_rows.items()):
        margins = [_expected_margin(row) for row in observed]
        per_action[action] = {
            "count": len(observed),
            "top1": mean(bool(row.get("correct")) for row in observed),
            "mean_expected_probability": mean(_probability(row, action) for row in observed),
            "mean_expected_vs_best_other_margin": mean(margins),
            "mean_base_to_adapter_expected_probability_delta": mean(paired_deltas[action]),
        }

    unrelated_transitions = _paired_correctness_transitions(
        base_unrelated, parametric_unrelated
    )
    training = _training_diagnostics(lineages or [], action_rows)
    action_top1 = [float(values["top1"]) for values in per_action.values()]
    unrelated_regression_pp = 100.0 * (
        mean(bool(row.get("correct")) for row in base_unrelated)
        - mean(bool(row.get("correct")) for row in parametric_unrelated)
    )
    return {
        "counts": {
            "rows": len(rows),
            "parametric_current": len(parametric),
            "paired_current_base": len(base_by_key),
            "paired_unrelated": len(base_unrelated),
            "lineages": len(lineages or []),
        },
        "confusion_matrix": {
            expected: dict(sorted(predictions.items()))
            for expected, predictions in sorted(confusion.items())
        },
        "per_action": per_action,
        "aggregate": {
            "parametric_current_top1": mean(
                bool(row.get("correct")) for row in parametric
            ),
            "per_action_top1_range": max(action_top1) - min(action_top1),
            "unrelated_regression_pp": unrelated_regression_pp,
        },
        "unrelated_correctness_transitions": unrelated_transitions,
        "training": training,
        "observed_pattern": {
            "strong_action_asymmetry": max(action_top1) - min(action_top1) >= 0.25,
            "material_unrelated_regression": unrelated_regression_pp > 2.0,
            "interpretation": (
                "The completed run is consistent with action-asymmetric optimization/capacity "
                "and non-local interference. This diagnostic does not identify a unique cause."
            ),
        },
    }


def _load_verified_rows(run_root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    results_root = run_root / "results"
    controls_path = results_root / "base_controls.jsonl"
    if not controls_path.is_file():
        raise FileNotFoundError(f"missing G1 controls: {controls_path}")
    controls_hash = manifest.get("control_results_sha256")
    if controls_hash is not None and file_hash(controls_path) != controls_hash:
        raise ValueError("G1 control result hash changed")
    rows = read_jsonl(controls_path)
    units = manifest.get("units")
    if not isinstance(units, dict) or not units:
        raise ValueError("G1 manifest contains no units")
    for attempt_id, unit in sorted(units.items()):
        if unit.get("state") != "verified":
            raise ValueError(f"G1 diagnostic requires a verified unit: {attempt_id}")
        result_path = results_root / f"{attempt_id.replace('::', '__')}.jsonl"
        expected_hash = unit.get("metadata", {}).get("result_sha256")
        if not result_path.is_file() or file_hash(result_path) != expected_hash:
            raise ValueError(f"G1 verified result hash changed: {attempt_id}")
        rows.extend(read_jsonl(result_path))
    return rows


def _load_lineages(run_root: Path) -> list[dict[str, Any]]:
    adapter_root = run_root / "adapters"
    if not adapter_root.is_dir():
        return []
    return [
        _read_object(path)
        for path in sorted(adapter_root.glob("*/pathmem_lineage.json"))
    ]


def _training_diagnostics(
    lineages: list[dict[str, Any]], action_rows: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    action_by_attempt = {
        str(row["attempt_id"]): action
        for action, rows in action_rows.items()
        for row in rows
    }
    losses: dict[str, list[float]] = defaultdict(list)
    recipes: Counter[str] = Counter()
    for lineage in lineages:
        attempt_id = str(lineage.get("attempt_id"))
        action = action_by_attempt.get(attempt_id)
        summary = lineage.get("training_summary")
        if action is not None and isinstance(summary, dict):
            final_loss = summary.get("final_loss")
            if isinstance(final_loss, int | float):
                losses[action].append(float(final_loss))
        recipes[str(lineage.get("recipe_id", "A-legacy"))] += 1
    return {
        "recipe_lineage_counts": dict(sorted(recipes.items())),
        "mean_final_loss_by_action": {
            action: mean(values) for action, values in sorted(losses.items())
        },
        "note": "final minibatch loss is descriptive and is not a convergence certificate",
    }


def _paired_correctness_transitions(
    base_rows: list[dict[str, Any]], adapter_rows: list[dict[str, Any]]
) -> dict[str, int]:
    base_by_key = _unique_by_key(base_rows)
    adapter_by_key = _unique_by_key(adapter_rows)
    if base_by_key.keys() != adapter_by_key.keys():
        raise ValueError("G1 unrelated rows do not form exact base/adapter pairs")
    transitions: Counter[str] = Counter()
    for key in sorted(base_by_key):
        before = "correct" if bool(base_by_key[key].get("correct")) else "wrong"
        after = "correct" if bool(adapter_by_key[key].get("correct")) else "wrong"
        transitions[f"{before}_to_{after}"] += 1
    return dict(sorted(transitions.items()))


def _unique_by_key(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        if key in result:
            raise ValueError(f"duplicate G1 diagnostic pair key: {key}")
        result[key] = row
    return result


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["attempt_id"]), str(row["probe_id"])


def _probability(row: dict[str, Any], action: str) -> float:
    actions = row.get("ordered_actions")
    probabilities = row.get("candidate_probabilities")
    if not isinstance(actions, list) or not isinstance(probabilities, list):
        raise ValueError("G1 diagnostic row has no candidate probabilities")
    return float(probabilities[actions.index(action)])


def _expected_margin(row: dict[str, Any]) -> float:
    expected = str(row["expected_action"])
    expected_probability = _probability(row, expected)
    other_probabilities = [
        _probability(row, str(action))
        for action in row["ordered_actions"]
        if action != expected
    ]
    return expected_probability - max(other_probabilities)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value
