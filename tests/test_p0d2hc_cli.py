from __future__ import annotations

import json
import sys
from pathlib import Path

import plasticity_placement.p0d2hc.cli as cli
from plasticity_placement.p0d2hc.cli import build_parser


def test_p0d2hc_cli_freezes_calibration_defaults(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "plan",
            "--output",
            str(tmp_path / "out"),
            "--source-manifest",
            str(tmp_path / "source" / "manifest.json"),
        ]
    )
    assert args.evaluation_max_length == 512
    assert args.oracle_min_accuracy == 0.90
    assert args.oracle_min_category_accuracy == 0.80
    assert args.oracle_max_invalid_rate == 0.01
    assert args.external_min_accuracy == 0.75
    assert args.external_max_invalid_rate == 0.05
    assert args.canary_model_name is None
    assert args.no_4bit is False


def test_p0d2hc_cli_exposes_no_training_or_scan_action() -> None:
    help_text = build_parser().format_help().casefold()
    for command in (
        "environment",
        "plan",
        "audit",
        "run",
        "aggregate",
        "audit-invalid",
    ):
        assert command in help_text
    assert "train" not in help_text
    assert "scan" not in help_text


def test_p0d2hc_invalid_audit_requires_independent_output(
    tmp_path: Path,
) -> None:
    args = build_parser().parse_args(
        [
            "audit-invalid",
            "--output",
            str(tmp_path / "source"),
            "--audit-output",
            str(tmp_path / "audit"),
        ]
    )
    assert args.output == tmp_path / "source"
    assert args.audit_output == tmp_path / "audit"
    assert args.bootstrap_samples == 10_000


def test_p0d2hc_invalid_audit_cli_dispatches(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    source = tmp_path / "source"
    audit = tmp_path / "audit"
    result = audit / "invalid_output_audit.json"
    observed = []

    def run(source_dir, audit_dir, bootstrap_samples):
        observed.append((source_dir, audit_dir, bootstrap_samples))
        return result

    monkeypatch.setattr(cli, "audit_invalid_outputs", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plasticity-p0d2hc",
            "audit-invalid",
            "--output",
            str(source),
            "--audit-output",
            str(audit),
            "--bootstrap-samples",
            "20",
        ],
    )
    cli.main()
    assert observed == [(source, audit, 20)]
    assert json.loads(capsys.readouterr().out) == {
        "invalid_output_audit_path": str(result)
    }
