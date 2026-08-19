from __future__ import annotations

import json
from pathlib import Path

import pytest

from plasticity_placement.p0d2hcbr.authorization import adopt, template
from plasticity_placement.p0d2hcbr.config import ExperimentSpec
from plasticity_placement.p0d2hcbr.manifest import Manifest


def test_cbr_authorization_is_external_and_cannot_expand_scope(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    digest = "a" * 64
    approval = {
        **template(digest),
        "decision": "approved",
        "approved_by": "reviewer",
        "approved_at": "2026-08-09T00:00:00+00:00",
    }
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(approval))

    adopted, _ = adopt(output, approval_path, digest)

    assert adopted["allowed_training_runs"] == 12
    assert adopted["allows_full_parameter_training"] is False
    approval["allows_mappings_per_adapter_scan"] = True
    expanded = tmp_path / "expanded.json"
    expanded.write_text(json.dumps(approval))
    with pytest.raises(PermissionError, match="differs"):
        adopt(tmp_path / "other-output", expanded, digest)


def test_manifest_tracks_exact_twelve_units(tmp_path: Path) -> None:
    spec = ExperimentSpec()
    manifest = Manifest.create(
        tmp_path / "manifest.json",
        {"run_id": "cbr-test", "config": {}},
        spec.unit_ids(),
    )

    assert manifest.state == "planned"
    assert set(manifest.payload["training_units"]) == set(spec.unit_ids())
    assert set(value["state"] for value in manifest.payload["training_units"].values()) == {
        "pending"
    }
    manifest.set_state("authorized")
    manifest.set_state("training")
    assert Manifest.load(manifest.path).state == "training"
