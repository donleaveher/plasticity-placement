from __future__ import annotations

from plasticity_placement.p0d2hcbr.cli import _selected_units, build_parser


def test_cli_exposes_frozen_lifecycle_and_unit_selection() -> None:
    parser = build_parser()
    for command in (
        "plan",
        "authorize",
        "train",
        "evaluate",
        "resumable-evaluation-template",
        "authorize-resumable-evaluation",
        "evaluate-resumable",
        "aggregate",
        "verify",
        "status",
    ):
        assert command in parser._subparsers._group_actions[0].choices
    args = parser.parse_args(
        [
            "train",
            "--output",
            "/tmp/cbr",
            "--curriculum",
            "disentangled",
            "--placement",
            "late_matched",
            "--seed",
            "20260810",
        ]
    )
    assert _selected_units(args) == [("disentangled", "late_matched", 20260810)]


def test_cli_default_matrix_contains_twelve_units() -> None:
    args = build_parser().parse_args(["train", "--output", "/tmp/cbr"])

    assert len(_selected_units(args)) == 12
