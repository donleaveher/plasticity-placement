from __future__ import annotations

from pathlib import Path

import pytest

from plasticity_placement.p0d2hrr.authorization import (
    authorization_template,
    validate_authorization,
)
from plasticity_placement.p0d2hrr.manifest import PilotManifest

PREREGISTRATION_HASH = "a" * 64


def _approval() -> dict[str, object]:
    payload = authorization_template(PREREGISTRATION_HASH)
    payload.update(
        {
            "decision": "approved",
            "approved_by": "independent-reviewer",
            "approved_at": "2026-07-29T12:00:00+08:00",
        }
    )
    return payload


def test_authorization_must_bind_exact_preregistration_and_narrow_scope() -> None:
    approved = validate_authorization(
        _approval(),
        preregistration_sha256=PREREGISTRATION_HASH,
    )
    assert approved["allowed_training_runs"] == 1
    assert approved["allows_narrow_scan"] is False

    wrong_hash = _approval()
    wrong_hash["preregistration_sha256"] = "b" * 64
    with pytest.raises(PermissionError, match="does not bind"):
        validate_authorization(
            wrong_hash,
            preregistration_sha256=PREREGISTRATION_HASH,
        )

    expanded = _approval()
    expanded["allows_rlvr"] = True
    with pytest.raises(PermissionError, match="expand"):
        validate_authorization(
            expanded,
            preregistration_sha256=PREREGISTRATION_HASH,
        )


def test_pending_template_is_not_authorization() -> None:
    with pytest.raises(PermissionError, match="not approved"):
        validate_authorization(
            authorization_template(PREREGISTRATION_HASH),
            preregistration_sha256=PREREGISTRATION_HASH,
        )


def test_manifest_enforces_one_way_experiment_lifecycle(tmp_path: Path) -> None:
    manifest = PilotManifest.load_or_create(
        tmp_path / "manifest.json",
        run_id="p0d2hrr-test",
        config={"preregistration_sha256": PREREGISTRATION_HASH},
        source_manifest_path="/frozen/source/manifest.json",
    )
    assert manifest.state == "planned"
    manifest.transition("authorized")
    manifest.transition("training")
    manifest.transition("trained")
    manifest.transition("evaluating")
    manifest.transition("evaluated")
    manifest.transition("verified")
    with pytest.raises(ValueError, match="invalid"):
        manifest.transition("training")
