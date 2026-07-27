from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import plasticity_placement.p0d2h.cli as cli
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


def test_p0d2h_plan_uses_metadata_only_source_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observed: list[bool] = []
    config = SimpleNamespace(
        identity_dict=lambda: {"conditions": []},
        expected_unit_count=288,
        expected_probe_row_count=5_376,
    )

    def resolve(request, *, deep_source_validation):
        observed.append(deep_source_validation)
        return config, {}, ()

    monkeypatch.setattr(cli, "resolve_request", resolve)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plasticity-p0d2h",
            "plan",
            "--output",
            str(tmp_path / "out"),
            "--source-manifest",
            str(tmp_path / "source" / "manifest.json"),
        ],
    )
    cli.main()
    assert observed == [False]
