from pathlib import Path

from plasticity_placement.p0d2h.cli import build_parser


def test_p0d2h_cli_plan_uses_frozen_defaults(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "plan",
            "--output",
            str(tmp_path / "out"),
            "--source-manifest",
            str(tmp_path / "source" / "manifest.json"),
        ]
    )
    assert args.command == "plan"
    assert args.resilience_margin == 0.05


def test_p0d2h_cli_exposes_no_training_or_narrow_action() -> None:
    help_text = build_parser().format_help().casefold()
    for command in ("environment", "plan", "run", "aggregate"):
        assert command in help_text
    assert "train" not in help_text
    assert "narrow" not in help_text
