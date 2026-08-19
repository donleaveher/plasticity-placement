from __future__ import annotations

import json
from pathlib import Path

import pytest

from plasticity_placement.p0d2hcbr import deviation
from plasticity_placement.p0d2hcbr.config import ExperimentSpec
from plasticity_placement.p0d2hcbr.manifest import Manifest


def _audit() -> dict[str, object]:
    return {
        "comparisons": {
            "coupled__seed-20260810": {
                "full_depth": 1_089_536,
                "late_matched": 1_167_360,
                "relative_difference": 1 / 14,
            }
        },
        "relative_tolerance": 0.01,
        "all_checks_passed": False,
    }


def test_budget_deviation_requires_external_human_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    identity = {
        "preregistration_sha256": "p" * 64,
        "code_sha256": "e" * 64,
    }
    manifest = Manifest.create(
        output / "manifest.json",
        {"run_id": "cbr-source", "config": identity},
        ExperimentSpec().unit_ids(),
    )
    manifest.payload["state"] = "trained"
    for entry in manifest.payload["training_units"].values():
        entry["state"] = "trained"
    manifest.save()
    monkeypatch.setattr(deviation, "load_config", lambda _: (manifest, ExperimentSpec(), identity))
    monkeypatch.setattr(
        "plasticity_placement.p0d2hcbr.runtime.audit_all_training_units", lambda _: _audit()
    )
    monkeypatch.setattr(deviation, "current_code_hash", lambda: "r" * 64)

    template = deviation.deviation_template(output)
    assert template["placement_comparison_scope"] == deviation.PLACEMENT_SCOPE
    assert template["allows_additional_training"] is False

    internal = output / "approval.json"
    internal.write_text(json.dumps(template))
    with pytest.raises(ValueError, match="outside"):
        deviation.adopt_budget_deviation(output, internal)

    approval = {
        **template,
        "decision": "approved",
        "approved_by": "reviewer",
        "approved_at": "2026-08-09T00:00:00+00:00",
    }
    external = tmp_path / "approval.json"
    external.write_text(json.dumps(approval))
    adopted = deviation.adopt_budget_deviation(output, external)

    assert adopted.is_file()
    assert manifest.payload["budget_deviation"]["additional_training_authorized"] is False
    assert deviation.validate_adopted_deviation(output)["recovery_code_sha256"] == "r" * 64


def test_budget_deviation_rejects_scope_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    identity = {"preregistration_sha256": "p" * 64, "code_sha256": "e" * 64}
    manifest = Manifest.create(
        output / "manifest.json",
        {"run_id": "cbr-source", "config": identity},
        ExperimentSpec().unit_ids(),
    )
    manifest.payload["state"] = "trained"
    manifest.save()
    monkeypatch.setattr(deviation, "load_config", lambda _: (manifest, ExperimentSpec(), identity))
    monkeypatch.setattr(
        "plasticity_placement.p0d2hcbr.runtime.audit_all_training_units", lambda _: _audit()
    )
    monkeypatch.setattr(deviation, "current_code_hash", lambda: "r" * 64)
    approval = {
        **deviation.deviation_template(output),
        "decision": "approved",
        "approved_by": "reviewer",
        "approved_at": "2026-08-09T00:00:00+00:00",
        "allows_additional_training": True,
    }
    path = tmp_path / "expanded.json"
    path.write_text(json.dumps(approval))

    with pytest.raises(PermissionError, match="differs"):
        deviation.adopt_budget_deviation(output, path)
