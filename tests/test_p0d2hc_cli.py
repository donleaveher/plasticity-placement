from __future__ import annotations

from pathlib import Path

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
    for command in ("environment", "plan", "audit", "run", "aggregate"):
        assert command in help_text
    assert "train" not in help_text
    assert "scan" not in help_text
