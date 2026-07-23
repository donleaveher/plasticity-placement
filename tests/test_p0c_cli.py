import argparse

import pytest

import plasticity_placement.p0c.cli as cli_module
from plasticity_placement.p0c.cli import _validate_calibration_report


def test_calibration_report_must_match_run_model_and_precision(monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "current_code_hash", lambda: "code-1")
    args = argparse.Namespace(
        model="model-a",
        model_revision="rev-1",
        use_4bit=True,
        max_length=256,
        max_new_tokens=8,
    )
    matching = {
        "calibration_config": {
            "model_name": "model-a",
            "model_revision": "rev-1",
            "use_4bit": True,
            "max_length": 256,
            "max_new_tokens": 8,
        },
        "selected_config": {"model_revision": "resolved-1"},
        "provenance": {
            "resolved_model_revision": "resolved-1",
            "code_sha256": "code-1",
            "compiler_hashes": {},
        },
    }
    _validate_calibration_report(matching, args)
    mismatched = {
        "calibration_config": {
            **matching["calibration_config"],
            "model_name": "model-b",
        }
    }
    with pytest.raises(SystemExit, match="configuration mismatch"):
        _validate_calibration_report(mismatched, args)
