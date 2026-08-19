from __future__ import annotations

import json
from pathlib import Path

import pytest

from plasticity_placement.p0d2hrabd.analysis import (
    _classify_unit,
    analyze_error_topology,
    build_cell_records,
)
from plasticity_placement.p0d2hrabd.config import CATEGORIES, CELLS, DiagnosticSpec
from plasticity_placement.p0d2hrabd.runtime import (
    MANIFEST_SCHEMA_VERSION,
    plan_diagnostic,
    verify_complete_result,
)
from plasticity_placement.p0d2hrr.io import file_hash, json_hash


def test_diagnostic_spec_is_frozen_and_disallows_intervention() -> None:
    spec = DiagnosticSpec()

    assert spec.allows_inference is False
    assert spec.allows_training is False
    assert spec.allows_historical_rab_reclassification is False
    assert spec.allows_mappings_per_adapter_scan is False
    with pytest.raises(ValueError, match="frozen"):
        DiagnosticSpec(bootstrap_seed=1)


@pytest.mark.parametrize(
    ("signature", "category", "tie"),
    (
        ("abba", "fully_compliant", False),
        ("baab", "fully_inverted", False),
        ("aaaa", "single_action_locked", False),
        ("aabb", "receipt_invariant", False),
        ("abab", "binding_invariant", False),
        ("aaab", "partial_mixed", False),
        ("obba", "other_action_intrusion", False),
        ("abba", "invalid_or_tied", True),
    ),
)
def test_unit_taxonomy_priority(signature: str, category: str, tie: bool) -> None:
    by_cell = _taxonomy_cells(signature, tie=tie)

    result = _classify_unit(by_cell, "adapter")

    assert result["category"] == category
    assert result["prediction_signature_CA_CB_SA_SB"] == signature
    assert category in CATEGORIES


def test_full_analysis_reconstructs_transitions_margins_and_taxonomy() -> None:
    paired, off, on, probes, lesson_types = _synthetic_source()

    result, cell_records, unit_records = analyze_error_topology(
        paired, off, on, probes, lesson_types, DiagnosticSpec()
    )

    assert len(cell_records) == 192
    assert len(unit_records) == 48
    assert result["integrity"]["all_checks_passed"] is True
    assert result["correctness_transitions"] == {
        "C→C": 96,
        "C→W": 96,
        "W→C": 0,
        "W→W": 0,
    }
    assert result["margin_summary"]["base"]["estimate"] == 2.0
    assert result["margin_summary"]["adapter"]["estimate"] == 0.0
    assert result["margin_summary"]["adapter_minus_base"]["estimate"] == -2.0
    assert result["unit_taxonomy"]["base"]["counts"]["fully_compliant"] == 48
    assert result["unit_taxonomy"]["adapter"]["counts"]["single_action_locked"] == 48
    assert result["descriptive_only"] is True
    assert result["inference_authorized"] is False
    assert result["training_authorized"] is False
    assert result["mappings_per_adapter_authorized"] is False


@pytest.mark.parametrize(
    "corruption", ("static", "score_rank", "nonfinite", "fractional_token_count")
)
def test_row_reconstruction_rejects_source_corruption(corruption: str) -> None:
    paired, off, on, probes, lesson_types = _synthetic_source()
    if corruption == "static":
        on[0]["receipt"] = "B"
    elif corruption == "score_rank":
        on[0]["candidates"][1]["sum_rank"] = 1
    elif corruption == "nonfinite":
        on[0]["candidates"][0]["mean_logprob"] = float("inf")
    elif corruption == "fractional_token_count":
        on[0]["candidates"][0]["token_count"] = 1.9
    else:
        raise AssertionError(corruption)

    _, integrity = build_cell_records(paired, off, on, probes, lesson_types, DiagnosticSpec())

    assert integrity["all_checks_passed"] is False
    assert len(integrity["mismatches"]) == 1


def test_row_reconstruction_rejects_consistent_result_mutation_against_probe_bank() -> None:
    paired, off, on, probes, lesson_types = _synthetic_source()
    for row in (paired[0], off[0], on[0]):
        row["expected_action"] = "act_c"

    _, integrity = build_cell_records(paired, off, on, probes, lesson_types, DiagnosticSpec())

    assert integrity["all_checks_passed"] is False
    assert "frozen binding probe" in integrity["mismatches"][0]["error"]


def test_plan_is_safe_and_preregistered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import plasticity_placement.p0d2hrabd.runtime as runtime

    rab = tmp_path / "rab"
    rab.mkdir()
    (rab / "summary.json").write_text("{}")
    monkeypatch.setattr(
        runtime,
        "_validate_rab_result",
        lambda _: {
            "summary": {"run_id": "p0d2hrab-reviewed"},
            "snapshot": {"rab_output_tree": "a" * 64},
        },
    )
    monkeypatch.setattr(runtime, "current_code_hash", lambda: "b" * 64)

    manifest_path = plan_diagnostic(output_dir=tmp_path / "diagnostic", rab_output=rab)
    manifest = json.loads(manifest_path.read_text())
    prereg = json.loads((manifest_path.parent / "preregistration.json").read_text())

    assert manifest["state"] == "planned"
    assert prereg["inference_authorized"] is False
    assert prereg["training_authorized"] is False
    assert prereg["mappings_per_adapter_authorized"] is False
    assert not (manifest_path.parent / "summary.json").exists()


def test_complete_verifier_rejects_published_artifact_mutation(tmp_path: Path) -> None:
    output = _complete_result_fixture(tmp_path)

    assert verify_complete_result(output) == output / "summary.json"
    (output / "factor_slices.json").write_text('{"mutated": true}')
    with pytest.raises(ValueError, match="artifacts changed"):
        verify_complete_result(output)


@pytest.mark.parametrize("relative", ("preregistration.json", "preflight/analysis_plan.json"))
def test_complete_verifier_rejects_preflight_mutation(tmp_path: Path, relative: str) -> None:
    output = _complete_result_fixture(tmp_path)
    (output / relative).write_text("{}")

    with pytest.raises(ValueError, match="changed"):
        verify_complete_result(output)


def _taxonomy_cells(signature: str, *, tie: bool) -> dict[str, dict[str, object]]:
    expected = {"canonical_A": "a", "canonical_B": "b", "swapped_A": "b", "swapped_B": "a"}
    rows: dict[str, dict[str, object]] = {}
    for cell, predicted in zip(CELLS, signature, strict=True):
        target = expected[cell]
        rows[cell] = {
            "expected_action": target,
            "adapter": {
                "predicted_action": predicted,
                "selected_correct": float(predicted == target),
                "counterfactual_selected": float(predicted in {"a", "b"} and predicted != target),
                "tie": tie and cell == "canonical_A",
            },
        }
    return rows


def _synthetic_source() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, str],
]:
    paired: list[dict[str, object]] = []
    off: list[dict[str, object]] = []
    on: list[dict[str, object]] = []
    probes: list[dict[str, object]] = []
    lesson_types = {f"pair-{index:02d}": f"lesson-{index % 3}" for index in range(12)}
    for pair_index in range(12):
        pair_id = f"pair-{pair_index:02d}"
        for variant in range(1, 5):
            unit_id = f"{pair_id}-r{variant}"
            for orientation, receipt, cell in (
                ("canonical", "A", "canonical_A"),
                ("canonical", "B", "canonical_B"),
                ("swapped", "A", "swapped_A"),
                ("swapped", "B", "swapped_B"),
            ):
                expected = (
                    "act_a"
                    if (orientation, receipt) in {("canonical", "A"), ("swapped", "B")}
                    else "act_b"
                )
                counterfactual = "act_b" if expected == "act_a" else "act_a"
                static = {
                    "probe_id": f"{unit_id}-{cell}",
                    "unit_id": unit_id,
                    "pair_id": pair_id,
                    "route_variant": variant,
                    "orientation": orientation,
                    "receipt": receipt,
                    "cell": cell,
                    "expected_action": expected,
                    "counterfactual_action": counterfactual,
                    "ordered_candidates": ["act_a", "act_b", "act_c", "act_d"],
                }
                base = _raw_row(static, "adapter_off", expected)
                adapter = _raw_row(static, "adapter_on", "act_a")
                paired.append(
                    {
                        **static,
                        "base": _paired_state(expected, counterfactual, expected),
                        "adapter": _paired_state(expected, counterfactual, "act_a"),
                    }
                )
                off.append(base)
                on.append(adapter)
                probes.append(dict(static))
    return paired, off, on, probes, lesson_types


def _raw_row(static: dict[str, object], state: str, predicted: str) -> dict[str, object]:
    candidates = list(static["ordered_candidates"])
    return {
        **static,
        "adapter_state": state,
        "predicted_candidate": predicted,
        "candidates": [
            {
                "candidate": candidate,
                "sum_rank": 1 if candidate == predicted else 2 + index,
                "sum_logprob": 0.0 if candidate == predicted else -2.0,
                "mean_logprob": 0.0 if candidate == predicted else -1.0,
                "token_count": index + 1,
            }
            for index, candidate in enumerate(candidates)
        ],
        "tie": False,
        "non_finite": False,
        "error_status": "ok",
        "top1_top2_margin": 2.0,
    }


def _paired_state(expected: str, counterfactual: str, predicted: str) -> dict[str, object]:
    return {
        "predicted_action": predicted,
        "selected_correct": float(predicted == expected),
        "counterfactual_selected": float(predicted == counterfactual),
        "scoring_valid": True,
    }


def _complete_result_fixture(tmp_path: Path) -> Path:
    output = tmp_path / "complete"
    (output / "preflight").mkdir(parents=True)
    run_id = "p0d2hrabd-fixture"
    analysis_plan = {"schema_version": "fixture-plan"}
    analysis_plan["analysis_plan_sha256"] = json_hash(analysis_plan)
    (output / "preflight" / "analysis_plan.json").write_text(json.dumps(analysis_plan))
    identity = {
        "analysis_plan_sha256": file_hash(output / "preflight" / "analysis_plan.json"),
        "inference_authorized": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    identity["preregistration_sha256"] = json_hash(identity)
    preregistration_sha256 = identity["preregistration_sha256"]
    (output / "preregistration.json").write_text(json.dumps(identity))
    summary = {
        "run_id": run_id,
        "analysis": {"analysis_status": "descriptive_error_topology_complete"},
        "historical_rab_decision_changed": False,
        "inference_authorized": False,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    (output / "summary.json").write_text(json.dumps(summary))
    (output / "report.md").write_text("# report\n")
    (output / "cell_records.jsonl").write_text("{}\n" * 192)
    (output / "unit_records.jsonl").write_text("{}\n" * 48)
    (output / "factor_slices.json").write_text(json.dumps({}))
    artifacts = {
        name: file_hash(output / filename)
        for name, filename in {
            "summary": "summary.json",
            "report": "report.md",
            "cell_records": "cell_records.jsonl",
            "unit_records": "unit_records.jsonl",
            "factor_slices": "factor_slices.json",
        }.items()
    }
    audit = {
        "schema_version": "p0d2hrabd-audit-manifest-v1",
        "run_id": run_id,
        "preregistration_sha256": preregistration_sha256,
        "artifacts": artifacts,
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
            "analysis_status": "descriptive_error_topology_complete",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    return output
