from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable
from statistics import mean
from typing import Any

from plasticity_placement.p0d2hrr.paired_audit import paired_summary, signed_expected_margin

FULL_EXTERNAL_CELL = "slot_default_natural_external"
PRIMARY_RESCUES = {
    "grammar": "slot_map_natural_external",
    "lexicon": "slot_default_snake_external",
    "payload": "slot_default_natural_opaque",
}


def build_pairs(
    off_rows: list[dict[str, Any]], on_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    off = _index(off_rows, "adapter OFF")
    on = _index(on_rows, "adapter ON")
    if set(off) != set(on):
        raise ValueError("route-transfer OFF/ON probe sets differ")
    pairs: list[dict[str, Any]] = []
    for probe_id in sorted(off):
        base = off[probe_id]
        adapter = on[probe_id]
        static = (
            "lesson_id",
            "pair_id",
            "lesson_type",
            "lesson_side",
            "cell",
            "response_type",
            "route_variant",
            "grammar",
            "lexicon",
            "payload",
            "target_slot",
            "expected_candidate",
            "ordered_candidates",
            "prompt_sha256",
            "source_probe_id",
            "route_anchor_probe_id",
            "route_anchor_prompt_sha256",
            "route_anchor_static_sha256",
        )
        if any(base.get(field) != adapter.get(field) for field in static):
            raise ValueError(f"route-transfer static pair fields differ: {probe_id}")
        base_margin = signed_expected_margin(base, "expected_candidate")
        adapter_margin = signed_expected_margin(adapter, "expected_candidate")
        pairs.append(
            {
                **{field: base.get(field) for field in static},
                "probe_id": probe_id,
                "base_predicted_candidate": base.get("predicted_candidate"),
                "adapter_predicted_candidate": adapter.get("predicted_candidate"),
                "base_correct": bool(base.get("correct")),
                "adapter_correct": bool(adapter.get("correct")),
                "base_error_status": str(base.get("error_status")),
                "adapter_error_status": str(adapter.get("error_status")),
                "base_tie": bool(base.get("tie")),
                "adapter_tie": bool(adapter.get("tie")),
                "base_tie_expected_compatible": _expected_in_top_tie(base),
                "adapter_tie_expected_compatible": _expected_in_top_tie(adapter),
                "base_signed_expected_margin": base_margin,
                "adapter_signed_expected_margin": adapter_margin,
                "signed_expected_margin_delta": (
                    None
                    if base_margin is None or adapter_margin is None
                    else adapter_margin - base_margin
                ),
                "prediction_changed": base.get("predicted_candidate")
                != adapter.get("predicted_candidate"),
                "transition": _transition(base, adapter),
            }
        )
    return pairs


def analyze_bridge(
    pairs: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    adjusted_confidence: float,
    material_threshold: float,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[str(row["cell"])].append(row)
    if set(grouped) != {
        "slot_map_snake_opaque",
        "slot_map_snake_external",
        "slot_map_natural_opaque",
        "slot_map_natural_external",
        "slot_default_snake_opaque",
        "slot_default_snake_external",
        "slot_default_natural_opaque",
        FULL_EXTERNAL_CELL,
        "external_action",
        "forced_slot_action",
    } or set(map(len, grouped.values())) != {96}:
        raise ValueError("route-transfer analysis matrix is incomplete")
    by_cell = {
        cell: paired_summary(
            rows,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=_derived_seed(bootstrap_seed, cell),
        )
        for cell, rows in sorted(grouped.items())
    }
    primary_rows = [row for row in pairs if str(row["cell"]).startswith("slot_")]
    action_rows = [
        row for row in pairs if str(row["cell"]) in {"external_action", "forced_slot_action"}
    ]
    primary_scoring_integrity = all(
        row[f"{state}_error_status"] == "ok" and not row[f"{state}_tie"]
        for row in primary_rows
        for state in ("base", "adapter")
    )
    action_scoring_integrity = all(
        row[f"{state}_error_status"] in {"ok", "tie"}
        and (not row[f"{state}_tie"] or row[f"{state}_tie_expected_compatible"])
        for row in action_rows
        for state in ("base", "adapter")
    )
    primary = {
        factor: _rescue_contrast(
            grouped[reverted],
            grouped[FULL_EXTERNAL_CELL],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=_derived_seed(bootstrap_seed, factor),
            confidence=adjusted_confidence,
            material_threshold=material_threshold,
            scoring_eligible=primary_scoring_integrity,
        )
        for factor, reverted in PRIMARY_RESCUES.items()
    }
    supported = [factor for factor, values in primary.items() if values["supported_barrier"]]
    scoring_integrity = primary_scoring_integrity and action_scoring_integrity
    if not scoring_integrity:
        status = "scoring_integrity_failed"
    elif supported:
        status = "one_or_more_transfer_barriers_supported"
    else:
        status = "no_single_factor_barrier_supported"
    return {
        "by_cell": by_cell,
        "primary_rescue_contrasts": primary,
        "supported_barriers": supported,
        "action_triangle": {
            "fully_external_slot_readout": by_cell[FULL_EXTERNAL_CELL],
            "actual_external_action": by_cell["external_action"],
            "forced_slot_external_action": by_cell["forced_slot_action"],
            "scope": (
                "Cross-response accuracies are diagnostic and are not treated as a qualification "
                "gate or as interchangeable outcomes."
            ),
        },
        "decision": {
            "status": status,
            "scoring_integrity_and_expected_compatible_ties": scoring_integrity,
            "primary_factorial_zero_tie_integrity": primary_scoring_integrity,
            "action_expected_compatible_tie_integrity": action_scoring_integrity,
            "material_rescue_threshold": material_threshold,
            "primary_two_sided_confidence": adjusted_confidence,
            "multiplicity_control": "Bonferroni over three preregistered rescue contrasts",
            "historical_gate_changed": False,
            "training_authorized": False,
            "mappings_per_adapter_authorized": False,
            "next_action": "review_bridge_evidence_before_any_new_intervention",
        },
    }


def _rescue_contrast(
    reverted_rows: list[dict[str, Any]],
    external_rows: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence: float,
    material_threshold: float,
    scoring_eligible: bool,
) -> dict[str, Any]:
    reverted = {_unit_key(row): row for row in reverted_rows}
    external = {_unit_key(row): row for row in external_rows}
    if set(reverted) != set(external) or len(reverted) != 96:
        raise ValueError("primary rescue contrast units differ")
    contrast_rows = [
        {
            "lesson_id": key[0],
            "route_variant": key[1],
            "rescue": _adapter_effect(reverted[key]) - _adapter_effect(external[key]),
        }
        for key in sorted(reverted)
    ]
    interval = _cluster_bootstrap(
        contrast_rows,
        lambda row: float(row["rescue"]),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    estimate = float(interval["estimate"])
    lower = float(interval["ci"][0])
    return {
        "estimand": (
            "[ON-OFF correctness with one external factor reverted] - "
            "[ON-OFF correctness in the fully external slot-readout cell]"
        ),
        **interval,
        "material_threshold": material_threshold,
        "scoring_eligible": scoring_eligible,
        "supported_barrier": (scoring_eligible and estimate >= material_threshold and lower > 0.0),
    }


def _cluster_bootstrap(
    rows: list[dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        number = float(value(row))
        if not math.isfinite(number):
            raise ValueError("non-finite route-transfer contrast")
        grouped[str(row["lesson_id"])].append(number)
    labels = sorted(grouped)
    generator = random.Random(bootstrap_seed)
    estimates: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = [labels[generator.randrange(len(labels))] for _ in labels]
        values = [number for label in sampled for number in grouped[label]]
        estimates.append(mean(values))
    alpha = 1.0 - confidence
    return {
        "estimate": mean(value(row) for row in rows),
        "ci": [_percentile(estimates, alpha / 2.0), _percentile(estimates, 1.0 - alpha / 2.0)],
        "confidence": confidence,
        "cluster_unit": "lesson_id",
        "cluster_count": len(labels),
        "observation_count": len(rows),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }


def _expected_in_top_tie(row: dict[str, Any]) -> bool:
    if not row.get("tie"):
        return False
    expected = row.get("expected_candidate")
    return any(
        candidate.get("candidate") == expected and candidate.get("sum_rank") == 1
        for candidate in row.get("candidates", [])
    )


def _transition(base: dict[str, Any], adapter: dict[str, Any]) -> str:
    return {
        (True, True): "correct_to_correct",
        (True, False): "correct_to_wrong",
        (False, True): "wrong_to_correct",
        (False, False): "wrong_to_wrong",
    }[(bool(base.get("correct")), bool(adapter.get("correct")))]


def _adapter_effect(row: dict[str, Any]) -> float:
    return float(bool(row["adapter_correct"])) - float(bool(row["base_correct"]))


def _unit_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["lesson_id"]), int(row["route_variant"])


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {str(row["probe_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {label} probe IDs")
    return result


def _derived_seed(seed: int, label: str) -> int:
    return seed + sum((index + 1) * ord(char) for index, char in enumerate(label))


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid percentile request")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
