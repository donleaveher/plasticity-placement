from pathlib import Path

from plasticity_placement.p0d2.cli import build_parser


def test_p0d2_cli_plan_uses_frozen_defaults(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "plan",
            "--output",
            str(tmp_path / "out"),
            "--source-manifest",
            str(tmp_path / "source.json"),
        ]
    )
    assert args.command == "plan"
    assert args.retention_margin == 0.05
    assert args.budget_tolerance == 0.01


def test_p0d2_cli_exposes_only_independent_actions() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "environment" in help_text
    assert "plan" in help_text
    assert "run" in help_text
    assert "aggregate" in help_text
    assert "narrow" not in help_text.casefold()
