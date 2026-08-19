from __future__ import annotations

import json
from pathlib import Path

import pytest

from plasticity_placement.pathmem.cli import build_parser
from plasticity_placement.pathmem.gates import (
    evaluate_g0_gate,
    evaluate_g1_gate,
    require_authorization,
)
from plasticity_placement.pathmem.manifest import (
    PROTOCOL_FILES,
    verify_g0_bundle,
    write_g0_bundle,
)
from plasticity_placement.pathmem.scoring import (
    CandidateDistribution,
    jensen_shannon_nats,
    mean_endpoint_defect,
    normalize_candidate_scores,
    symmetric_item_defect,
    write_qualified,
)


def _protocol_root(tmp_path: Path) -> Path:
    root = tmp_path / "protocol"
    for relative_path in PROTOCOL_FILES:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen protocol source: {relative_path}\n", encoding="utf-8")
    return root


def test_js_matches_frozen_analytic_calibration_scale() -> None:
    actions = ("current", "obsolete", "other1", "other2")
    first = CandidateDistribution(actions, (0.70, 0.10, 0.10, 0.10))
    second = CandidateDistribution(actions, (0.55, 0.25, 0.10, 0.10))
    assert jensen_shannon_nats(first, second) == pytest.approx(0.021114, abs=5e-7)
    assert mean_endpoint_defect((first, first), (second, second)) == pytest.approx(
        0.021114,
        abs=5e-7,
    )
    assert symmetric_item_defect(0.01, 0.03) == pytest.approx(0.02)


def test_score_normalization_and_write_qualification() -> None:
    actions = ("current", "obsolete", "other1", "other2")
    distribution = normalize_candidate_scores(actions, (0.0, -2.0, -3.0, -4.0))
    assert sum(distribution.probabilities) == pytest.approx(1.0)
    assert write_qualified(
        (distribution, distribution, distribution, distribution),
        current_action="current",
        obsolete_action="obsolete",
    )


def test_gates_are_explicit_and_authorization_is_sequential() -> None:
    g0 = evaluate_g0_gate(
        {
            "protocol_sources_hashed": True,
            "schemas_frozen": True,
            "split_bank_complete": True,
            "endpoint_equivalence_verified": True,
            "exposure_controls_verified": True,
            "observability_matrix_complete": True,
            "claim_falsifiers_complete": True,
            "manifest_identity_verified": True,
            "training_not_started": True,
        }
    )
    assert g0.passed is True
    g1 = evaluate_g1_gate(
        {
            "answer_copy_top1": 0.99,
            "parser_validity": 1.0,
            "external_current_top1": 0.96,
            "external_obsolete_intrusion": 0.04,
            "parametric_current_top1": 0.81,
            "parametric_per_action_min_top1": 0.71,
            "parametric_unrelated_regression_pp": 4.0,
            "adapter_disable_max_candidate_delta": 1e-6,
        }
    )
    assert g1.passed is True
    require_authorization("P0", {"G0", "G1"})
    with pytest.raises(PermissionError, match="G1"):
        require_authorization("P0", {"G0"})


def test_g0_bundle_is_immutable_regenerable_and_tamper_evident(tmp_path: Path) -> None:
    protocol_root = _protocol_root(tmp_path)
    output = tmp_path / "g0"
    manifest_path = write_g0_bundle(output, protocol_root)
    first = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = verify_g0_bundle(output, protocol_root)
    assert report["passed"] is True
    assert report["counts"] == {
        "items": 68,
        "event_blocks": 272,
        "probes": 4_080,
        "history_families": 136,
        "diagnostic_specs": 5,
    }
    write_g0_bundle(output, protocol_root)
    second = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert first == second
    items_path = output / "compiled" / "items.jsonl"
    items_path.write_text(items_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    tampered = verify_g0_bundle(output, protocol_root)
    assert tampered["passed"] is False
    assert "hash mismatch" in tampered["error"]


def test_g0_bundle_rejects_extra_files_and_protocol_drift(tmp_path: Path) -> None:
    protocol_root = _protocol_root(tmp_path)
    output = tmp_path / "g0"
    write_g0_bundle(output, protocol_root)
    (output / "unplanned.txt").write_text("not authorized", encoding="utf-8")
    assert verify_g0_bundle(output, protocol_root)["error"] == "unexpected G0 file set"
    (output / "unplanned.txt").unlink()
    (protocol_root / "README.md").write_text("changed\n", encoding="utf-8")
    assert verify_g0_bundle(output, protocol_root)["error"] == "protocol source hash mismatch"


def test_cli_requires_output_and_protocol_root_for_prepare_and_verify() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["prepare", "--output", "/tmp/g0", "--protocol-root", "/tmp/protocol"]
    )
    assert args.command == "prepare"
    assert args.output == Path("/tmp/g0")
    verify = parser.parse_args(
        ["verify", "--output", "/tmp/g0", "--protocol-root", "/tmp/protocol"]
    )
    assert verify.command == "verify"
