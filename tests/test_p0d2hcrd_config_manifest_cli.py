from __future__ import annotations

from pathlib import Path

import pytest

from plasticity_placement.p0d2hcrd.cli import build_parser
from plasticity_placement.p0d2hcrd.config import (
    CRDModel,
    P0D2HCRDRequest,
    ResolvedP0D2HCRDConfig,
)
from plasticity_placement.p0d2hcrd.manifest import P0D2HCRDManifest


def config(tmp_path: Path) -> ResolvedP0D2HCRDConfig:
    return ResolvedP0D2HCRDConfig(
        output_dir=tmp_path / "out",
        source_output_dir=tmp_path / "source",
        source_calibration_output_dir=tmp_path / "calibration",
        source_hard_probe_output_dir=tmp_path / "hard",
        source_compiled_output_dir=tmp_path / "compiled",
        code_sha256="code",
        source_manifest_sha256="manifest",
        source_run_id="source-run",
        source_summary_sha256="summary",
        source_candidate_token_audit_sha256="source-audit",
        source_raw_results_sha256="source-raw",
        source_calibration_manifest_sha256="calibration-manifest",
        source_calibration_raw_results_sha256="calibration-raw",
        source_hard_probe_manifest_sha256="hard-manifest",
        hard_probe_hashes={"hard_probes_sha256": "hard-probes"},
        compiler_hashes={"lessons_sha256": "lessons"},
        selected_lesson_ids=tuple(f"L{index:02d}" for index in range(24)),
        decomposition_bank_sha256="bank",
        model=CRDModel(
            model_id="scale_canary",
            role="scale_canary",
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            model_revision=(
                "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
            ),
            use_4bit=True,
        ),
        prompt_renderers={
            "route_only": "route",
            "retrieval_only": "retrieval",
            "combined": "combined",
        },
    )


def test_config_freezes_matrix_and_never_authorizes_training(
    tmp_path: Path,
) -> None:
    resolved = config(tmp_path)
    identity = resolved.identity_dict()
    assert resolved.expected_unit_count == 24
    assert resolved.expected_decision_row_count == 1_536
    assert resolved.expected_candidate_sequence_count == 5_376
    assert identity["endpoints"] == [
        "route_only",
        "retrieval_only",
        "combined",
    ]
    assert identity["training_complexity_review_eligible"] is False
    assert identity["automatic_training_started"] is False
    assert identity["automatic_narrow_scan_started"] is False


def test_request_rejects_nested_source_and_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    with pytest.raises(ValueError, match="independent"):
        P0D2HCRDRequest(
            output_dir=source / "derived",
            source_manifest=source / "manifest.json",
        )


def test_manifest_has_terminal_verified_and_failed_states(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    identity = config(tmp_path).identity_dict()
    manifest = P0D2HCRDManifest.load_or_create(
        path,
        run_id="run",
        config=identity,
        source_manifest_path="/source/manifest.json",
        bank_audit_sha256="bank-audit",
        candidate_token_audit_sha256="candidate-audit",
    )
    manifest.mark_unit("L00", "evaluating")
    manifest.mark_unit("L00", "verified")
    with pytest.raises(ValueError, match="transition"):
        manifest.mark_unit("L00", "evaluating")
    manifest.mark_unit("L01", "failed")
    with pytest.raises(ValueError, match="transition"):
        manifest.mark_unit("L01", "evaluating")
    resumed = P0D2HCRDManifest.load_or_create(
        path,
        run_id="run",
        config=identity,
        source_manifest_path="/source/manifest.json",
        bank_audit_sha256="bank-audit",
        candidate_token_audit_sha256="candidate-audit",
    )
    assert resumed.unit_state("L00") == "verified"
    assert resumed.unit_state("L01") == "failed"


def test_cli_exposes_only_base_diagnostic_workflow(tmp_path: Path) -> None:
    parser = build_parser()
    help_text = parser.format_help().casefold()
    for command in ("environment", "plan", "audit", "run", "aggregate"):
        assert command in help_text
    for forbidden in ("train", "adapter", "scan", "grpo", "rlvr"):
        assert forbidden not in help_text
    args = parser.parse_args(
        [
            "plan",
            "--output",
            str(tmp_path / "out"),
            "--source-manifest",
            str(tmp_path / "source" / "manifest.json"),
        ]
    )
    assert args.command == "plan"


def test_package_has_no_training_or_adapter_execution_path() -> None:
    package = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "plasticity_placement"
        / "p0d2hcrd"
    )
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(package.glob("*.py"))
    )
    for forbidden in (
        "load_adapter_model",
        "activate_adapter(",
        "PeftModel",
        "train_lora",
        "run_narrow_scan",
    ):
        assert forbidden not in source

