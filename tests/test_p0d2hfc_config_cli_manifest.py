from __future__ import annotations

from pathlib import Path

import pytest

from plasticity_placement.p0d2hfc.cli import build_parser
from plasticity_placement.p0d2hfc.config import (
    EXTERNAL_MIN_ACCURACY,
    FORCED_CHOICE_SCHEMA_VERSION,
    ForcedChoiceModel,
    P0D2HFCRequest,
    ResolvedP0D2HFCConfig,
)
from plasticity_placement.p0d2hfc.manifest import P0D2HFCManifest


def _model(model_id: str) -> ForcedChoiceModel:
    return ForcedChoiceModel(
        model_id=model_id,
        role="source" if model_id == "source_model" else "scale_canary",
        model_name=f"model-{model_id}",
        model_revision=f"revision-{model_id}",
        use_4bit=True,
    )


def _config(tmp_path: Path) -> ResolvedP0D2HFCConfig:
    return ResolvedP0D2HFCConfig(
        output_dir=tmp_path / "out",
        source_output_dir=tmp_path / "source",
        source_p0d2h_output_dir=tmp_path / "hard",
        code_sha256="code",
        source_manifest_sha256="manifest",
        source_run_id="source-run",
        source_summary_sha256="summary",
        source_prompt_token_audit_sha256="prompt-audit",
        source_raw_results_sha256="raw",
        source_p0d2h_run_id="hard-run",
        source_p0d2h_manifest_sha256="hard-manifest",
        hard_probe_hashes={"hard_probes_sha256": "probes"},
        selected_lesson_ids=tuple(f"L{index:02d}" for index in range(24)),
        models=(_model("source_model"), _model("scale_canary")),
        prompt_renderers={
            "no_write": "plain",
            "external": "external",
            "answer_copy_oracle": "oracle",
        },
    )


def test_config_freezes_full_matrix_score_and_gate_identity(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    identity = config.identity_dict()
    assert config.expected_unit_count == 48
    assert config.expected_decision_row_count == 2_304
    assert identity["schema_version"] == FORCED_CHOICE_SCHEMA_VERSION
    assert identity["primary_score_definition"] == "candidate_token_sum_logprob"
    assert identity["canonical_leading_whitespace"] == ""
    assert (
        identity["gate_thresholds"]["external_min_accuracy"]
        == EXTERNAL_MIN_ACCURACY
    )
    assert identity["automatic_training_started"] is False
    assert identity["automatic_narrow_scan_started"] is False


def test_request_rejects_source_destination_collision(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    with pytest.raises(ValueError, match="independent"):
        P0D2HFCRequest(
            output_dir=source / "derived",
            source_manifest=source / "manifest.json",
        )


def test_cli_exposes_only_base_scoring_workflow(tmp_path: Path) -> None:
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


def test_manifest_resume_conflict_and_failed_attempt_are_immutable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    config = _config(tmp_path).identity_dict()
    manifest = P0D2HFCManifest.load_or_create(
        path,
        run_id="run",
        config=config,
        source_manifest_path="/source/manifest.json",
        candidate_token_audit_sha256="audit",
    )
    manifest.mark_unit("source_model", "L00", "evaluating")
    manifest.mark_unit("source_model", "L00", "verified")
    resumed = P0D2HFCManifest.load_or_create(
        path,
        run_id="run",
        config=config,
        source_manifest_path="/source/manifest.json",
        candidate_token_audit_sha256="audit",
    )
    assert resumed.unit_state("source_model", "L00") == "verified"
    with pytest.raises(ValueError, match="transition"):
        resumed.mark_unit("source_model", "L00", "evaluating")
    resumed.mark_unit("scale_canary", "L00", "failed")
    with pytest.raises(ValueError, match="transition"):
        resumed.mark_unit("scale_canary", "L00", "evaluating")
    with pytest.raises(ValueError, match="does not match"):
        P0D2HFCManifest.load_or_create(
            path,
            run_id="changed-run",
            config=config,
            source_manifest_path="/source/manifest.json",
            candidate_token_audit_sha256="audit",
        )
