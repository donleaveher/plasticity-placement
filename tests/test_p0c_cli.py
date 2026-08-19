import argparse
import json
import sys

import pytest

import plasticity_placement.p0c.cli as cli_module
from plasticity_placement.p0c.cli import _validate_calibration_report, main


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


def test_environment_command_emits_machine_readable_context(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli_module, "current_environment_fingerprint", lambda: "env-1")
    monkeypatch.setattr(
        cli_module,
        "current_environment_snapshot",
        lambda: {"gpu": "T4", "code_sha256": "code-1"},
    )
    monkeypatch.setattr(sys, "argv", ["plasticity-p0c", "environment"])

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "fingerprint": "env-1",
        "environment": {"gpu": "T4", "code_sha256": "code-1"},
    }


def test_resolve_model_command_emits_immutable_revision(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "resolve_model_revision",
        lambda model_name, revision: "resolved-revision",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plasticity-p0c",
            "resolve-model",
            "--model",
            "model-a",
            "--model-revision",
            "requested-revision",
        ],
    )

    main()

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "model_name": "model-a",
        "requested_revision": "requested-revision",
        "resolved_revision": "resolved-revision",
    }
