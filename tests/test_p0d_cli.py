import argparse
import json
from hashlib import sha256
from pathlib import Path

import pytest

from plasticity_placement.p0d.cli import _request_from_args
from plasticity_placement.p0d.config import P0DStage


def test_narrow_cli_freezes_condition_file_and_parent_run(tmp_path: Path) -> None:
    path = tmp_path / "conditions.json"
    payload = {
        "stage": "narrow_scan",
        "source_band_run_id": "p0d-band-run",
        "reference_condition_id": "full",
        "conditions": [
            {"condition_id": "full", "layer_band": "full"},
            {
                "condition_id": "layer-11",
                "layer_band": "explicit",
                "explicit_layers": [11],
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    request = _request_from_args(
        argparse.Namespace(
            output=tmp_path / "out",
            source_manifest=tmp_path / "source.json",
            stage=P0DStage.NARROW_SCAN,
            conditions_config=path,
            retention_margin=0.05,
        )
    )
    assert request.parent_band_run_id == "p0d-band-run"
    assert request.conditions_config_sha256 == sha256(path.read_bytes()).hexdigest()
    assert request.conditions[1].explicit_layers == (11,)


def test_narrow_cli_rejects_missing_parent_band_run(tmp_path: Path) -> None:
    path = tmp_path / "conditions.json"
    path.write_text(
        json.dumps(
            {
                "stage": "narrow_scan",
                "conditions": [
                    {"condition_id": "full", "layer_band": "full"},
                    {
                        "condition_id": "layer-11",
                        "layer_band": "explicit",
                        "explicit_layers": [11],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="parent band"):
        _request_from_args(
            argparse.Namespace(
                output=tmp_path / "out",
                source_manifest=tmp_path / "source.json",
                stage=P0DStage.NARROW_SCAN,
                conditions_config=path,
                retention_margin=0.05,
            )
        )
