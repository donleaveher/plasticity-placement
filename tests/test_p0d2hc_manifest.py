from pathlib import Path

import pytest

from plasticity_placement.p0d2hc.manifest import (
    P0D2HCManifest,
    unit_key,
)


def _config() -> dict[str, object]:
    return {
        "selected_lesson_ids": ["L1"],
        "models": [
            {
                "model_id": "source_model",
                "role": "source",
                "model_name": "model",
                "model_revision": "revision",
                "use_4bit": True,
            }
        ],
    }


def test_p0d2hc_manifest_resumes_and_preserves_terminal_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    manifest = P0D2HCManifest.load_or_create(
        path,
        run_id="calibration-run",
        config=_config(),
        source_manifest_path="/source/manifest.json",
        prompt_token_audit_sha256="audit",
    )
    manifest.mark_unit("source_model", "L1", "evaluating")
    manifest.mark_unit(
        "source_model",
        "L1",
        "verified",
        evaluation_precision="float16",
    )
    resumed = P0D2HCManifest.load_or_create(
        path,
        run_id="calibration-run",
        config=_config(),
        source_manifest_path="/source/manifest.json",
        prompt_token_audit_sha256="audit",
    )
    assert resumed.unit_state("source_model", "L1") == "verified"
    assert unit_key("source_model", "L1") in resumed.payload["units"]
    with pytest.raises(ValueError, match="transition"):
        resumed.mark_unit("source_model", "L1", "evaluating")
