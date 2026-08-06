from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from plasticity_placement.p0d2hrabdl.analysis import (
    analyze_localization,
    build_margin_transition_profiles,
    build_taxonomy_migrations,
    build_unit_hotspots,
    validate_source_rows,
)
from plasticity_placement.p0d2hrabdl.config import (
    CELLS,
    EXPECTED_ADAPTER_TAXONOMY,
    EXPECTED_BASE_TAXONOMY,
    EXPECTED_TRANSITION_COUNTS,
    FACTORS,
    LocalizationSpec,
)
from plasticity_placement.p0d2hrabdl.runtime import (
    MANIFEST_SCHEMA_VERSION,
    plan_localization,
    verify_complete_result,
)
from plasticity_placement.p0d2hrr.io import file_hash, json_hash


def test_localization_spec_is_frozen_and_disallows_intervention() -> None:
    spec = LocalizationSpec()

    assert spec.allows_inference is False
    assert spec.allows_training is False
    assert spec.allows_historical_reclassification is False
    assert spec.allows_mappings_per_adapter_scan is False
    with pytest.raises(ValueError, match="frozen"):
        LocalizationSpec(bootstrap_seed=1)


def test_localization_reconstructs_all_frozen_views() -> None:
    summary, factor_slices, cells, units = _synthetic_source()

    analysis, localization, taxonomy, margins, hotspots = analyze_localization(
        summary,
        factor_slices,
        cells,
        units,
        LocalizationSpec(),
    )

    assert analysis["integrity"]["all_checks_passed"] is True
    assert analysis["transition_totals"] == EXPECTED_TRANSITION_COUNTS
    assert localization["adverse_total"] == 69
    assert len(localization["factors"]["pair_id"]["levels"]) == 12
    assert sum(row["count"] for row in taxonomy["nonzero_transitions"]) == 48
    assert margins["profiles"]["C→W"]["mean_change"] == -3.0
    assert margins["profiles"]["W→C"]["mean_change"] == 3.0
    assert len(hotspots) == 48
    assert [row["adverse_rank"] for row in hotspots] == list(range(1, 49))
    assert analysis["descriptive_only"] is True
    assert analysis["training_authorized"] is False
    assert analysis["mappings_per_adapter_authorized"] is False


@pytest.mark.parametrize(
    "corruption",
    ("summary_transition", "factor_count", "duplicate_probe", "unit_join", "nonfinite"),
)
def test_source_integrity_rejects_corruption(corruption: str) -> None:
    summary, factor_slices, cells, units = _synthetic_source()
    if corruption == "summary_transition":
        summary["analysis"]["correctness_transitions"]["C→W"] = 20
    elif corruption == "factor_count":
        factor_slices["receipt"][0]["observation_count"] += 1
    elif corruption == "duplicate_probe":
        cells[1]["probe_id"] = cells[0]["probe_id"]
    elif corruption == "unit_join":
        units[0]["pair_id"] = "wrong-pair"
    elif corruption == "nonfinite":
        cells[0]["adapter"]["selected_minus_counterfactual_margin"] = float("inf")
    else:
        raise AssertionError(corruption)

    integrity = validate_source_rows(
        summary,
        factor_slices,
        cells,
        units,
        LocalizationSpec(),
    )

    assert integrity["all_checks_passed"] is False


def test_taxonomy_migration_matrix_preserves_all_units() -> None:
    _, _, _, units = _synthetic_source()

    result = build_taxonomy_migrations(units)

    assert sum(sum(row.values()) for row in result["matrix"].values()) == 48
    assert sum(result["base_single_action_locked_destinations"].values()) == 30
    assert sum(result["adapter_receipt_invariant_origins"].values()) == 13
    assert sum(result["adapter_partial_mixed_origins"].values()) == 23


def test_margin_profiles_cluster_within_each_transition() -> None:
    _, _, cells, _ = _synthetic_source()

    profiles = build_margin_transition_profiles(cells, LocalizationSpec())["profiles"]

    for transition, count in EXPECTED_TRANSITION_COUNTS.items():
        assert profiles[transition]["observation_count"] == count
        assert profiles[transition]["mean_change_pair_cluster_ci"]["cluster_unit"] == "pair_id"
        assert profiles[transition]["mean_change_pair_cluster_ci"]["bootstrap_samples"] == 10_000


def test_hotspot_ranking_is_deterministic_and_exhaustive() -> None:
    _, _, cells, units = _synthetic_source()

    first = build_unit_hotspots(cells, units)
    second = build_unit_hotspots(list(reversed(cells)), list(reversed(units)))

    assert first == second
    assert len(first) == 48
    assert all(len(row["cells"]) == 4 for row in first)


def test_plan_is_safe_and_preregistered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import plasticity_placement.p0d2hrabdl.runtime as runtime

    source = tmp_path / "rabd"
    source.mkdir()
    (source / "summary.json").write_text("{}")
    monkeypatch.setattr(
        runtime,
        "_validate_rabd_result",
        lambda _: {
            "summary": {"run_id": "p0d2hrabd-a82d35757b"},
            "snapshot": {"rabd_output_tree": "a" * 64},
        },
    )
    monkeypatch.setattr(runtime, "current_code_hash", lambda: "b" * 64)

    path = plan_localization(output_dir=tmp_path / "localization", rabd_output=source)
    manifest = json.loads(path.read_text())
    prereg = json.loads((path.parent / "preregistration.json").read_text())

    assert manifest["state"] == "planned"
    assert prereg["inference_authorized"] is False
    assert prereg["training_authorized"] is False
    assert prereg["mappings_per_adapter_authorized"] is False
    assert not (path.parent / "summary.json").exists()


def test_complete_verifier_rejects_artifact_mutation(tmp_path: Path) -> None:
    output = _complete_result_fixture(tmp_path)

    assert verify_complete_result(output) == output / "summary.json"
    (output / "taxonomy_migrations.json").write_text('{"mutated": true}')
    with pytest.raises(ValueError, match="artifacts changed"):
        verify_complete_result(output)


def _synthetic_source() -> tuple[
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    transition_sequence = [
        transition for transition, count in EXPECTED_TRANSITION_COUNTS.items() for _ in range(count)
    ]
    base_categories = [
        category for category, count in EXPECTED_BASE_TAXONOMY.items() for _ in range(count)
    ]
    adapter_categories = [
        category for category, count in EXPECTED_ADAPTER_TAXONOMY.items() for _ in range(count)
    ]
    cells: list[dict[str, object]] = []
    units: list[dict[str, object]] = []
    for unit_index in range(48):
        pair_index = unit_index // 4
        variant = unit_index % 4 + 1
        pair_id = f"pair-{pair_index:02d}"
        unit_id = f"{pair_id}-r{variant}"
        base_category = base_categories[unit_index]
        adapter_category = adapter_categories[unit_index]
        units.append(
            {
                "unit_id": unit_id,
                "pair_id": pair_id,
                "lesson_type": f"lesson-{pair_index % 3}",
                "route_variant": variant,
                "base": {"category": base_category},
                "adapter": {"category": adapter_category},
                "taxonomy_transition": f"{base_category}→{adapter_category}",
            }
        )
        for cell_index, cell in enumerate(CELLS):
            transition = transition_sequence[unit_index * 4 + cell_index]
            base_correct = transition.startswith("C")
            adapter_correct = transition.endswith("C")
            expected_action = "act_a" if cell_index in (0, 3) else "act_b"
            cells.append(
                {
                    "probe_id": f"{unit_id}-{cell}",
                    "unit_id": unit_id,
                    "pair_id": pair_id,
                    "lesson_type": f"lesson-{pair_index % 3}",
                    "route_variant": variant,
                    "orientation": "canonical" if cell_index < 2 else "swapped",
                    "receipt": "A" if cell_index % 2 == 0 else "B",
                    "cell": cell,
                    "expected_action": expected_action,
                    "candidate_position": 0 if expected_action == "act_a" else 1,
                    "expected_token_count": 1,
                    "correctness_transition": transition,
                    "base": {
                        "selected_correct": float(base_correct),
                        "selected_minus_counterfactual_margin": 1.0 if base_correct else -1.0,
                    },
                    "adapter": {
                        "selected_correct": float(adapter_correct),
                        "selected_minus_counterfactual_margin": 2.0 if adapter_correct else -2.0,
                    },
                }
            )
    factor_slices = _factor_slices(cells)
    summary = {
        "run_id": "p0d2hrabd-a82d35757b",
        "analysis": {
            "analysis_status": "descriptive_error_topology_complete",
            "correctness_transitions": dict(EXPECTED_TRANSITION_COUNTS),
            "unit_taxonomy": {
                "base": {"counts": dict(EXPECTED_BASE_TAXONOMY)},
                "adapter": {"counts": dict(EXPECTED_ADAPTER_TAXONOMY)},
            },
        },
        "historical_rab_decision_changed": False,
        "inference_authorized": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    return summary, factor_slices, cells, units


def _factor_slices(cells: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for factor in FACTORS:
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in cells:
            grouped[str(row[factor])].append(row)
        result[factor] = [
            {
                "factor": factor,
                "level": level,
                "observation_count": len(rows),
                "correctness_transitions": {
                    label: Counter(str(row["correctness_transition"]) for row in rows)[label]
                    for label in EXPECTED_TRANSITION_COUNTS
                },
            }
            for level, rows in sorted(grouped.items())
        ]
    return result


def _complete_result_fixture(tmp_path: Path) -> Path:
    output = tmp_path / "complete"
    (output / "preflight").mkdir(parents=True)
    run_id = "p0d2hrabdl-fixture"
    plan = {"schema_version": "fixture-plan"}
    plan["analysis_plan_sha256"] = json_hash(plan)
    (output / "preflight" / "analysis_plan.json").write_text(json.dumps(plan))
    identity = {
        "analysis_plan_sha256": file_hash(output / "preflight" / "analysis_plan.json"),
        "inference_authorized": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    identity["preregistration_sha256"] = json_hash(identity)
    (output / "preregistration.json").write_text(json.dumps(identity))
    summary = {
        "run_id": run_id,
        "analysis": {"analysis_status": "descriptive_error_localization_complete"},
        "historical_rab_decision_changed": False,
        "historical_rabd_status_changed": False,
        "inference_authorized": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    (output / "summary.json").write_text(json.dumps(summary))
    (output / "report.md").write_text("# report\n")
    (output / "cell_transition_localization.json").write_text("{}")
    (output / "taxonomy_migrations.json").write_text("{}")
    (output / "margin_transition_profiles.json").write_text("{}")
    (output / "unit_hotspots.jsonl").write_text("{}\n" * 48)
    filenames = {
        "summary": "summary.json",
        "report": "report.md",
        "cell_transition_localization": "cell_transition_localization.json",
        "taxonomy_migrations": "taxonomy_migrations.json",
        "margin_transition_profiles": "margin_transition_profiles.json",
        "unit_hotspots": "unit_hotspots.jsonl",
    }
    artifacts = {name: file_hash(output / filename) for name, filename in filenames.items()}
    audit = {
        "schema_version": "p0d2hrabdl-audit-manifest-v1",
        "run_id": run_id,
        "preregistration_sha256": identity["preregistration_sha256"],
        "artifacts": artifacts,
        "historical_rab_decision_changed": False,
        "historical_rabd_status_changed": False,
        "inference_authorized": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    (output / "audit_manifest.json").write_text(json.dumps(audit))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "state": "complete",
        "config": identity,
        "result": {
            "summary_sha256": artifacts["summary"],
            "audit_manifest_sha256": file_hash(output / "audit_manifest.json"),
            "analysis_status": "descriptive_error_localization_complete",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    return output
