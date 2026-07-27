from pathlib import Path

import pytest

from plasticity_placement.p0d2h.manifest import P0D2HManifest, unit_key


def _config() -> dict[str, object]:
    return {
        "selected_lesson_ids": ["L1"],
        "conditions": [{"condition_id": "late-matched"}],
    }


def test_p0d2h_manifest_resumes_and_preserves_terminal_states(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    manifest = P0D2HManifest.load_or_create(
        path,
        run_id="hard-run",
        config=_config(),
        source_manifest_path="/source/manifest.json",
        prompt_token_audit_sha256="audit-1",
    )
    manifest.mark_base("L1", "evaluating")
    manifest.mark_base("L1", "verified", result_path="base.jsonl")
    manifest.mark_unit("late-matched", "L1", 41, "evaluating")
    manifest.mark_unit("late-matched", "L1", 41, "verified")

    resumed = P0D2HManifest.load_or_create(
        path,
        run_id="hard-run",
        config=_config(),
        source_manifest_path="/source/manifest.json",
        prompt_token_audit_sha256="audit-1",
    )
    assert resumed.base_state("L1") == "verified"
    assert resumed.unit_state("late-matched", "L1", 41) == "verified"
    assert unit_key("late-matched", "L1", 41) in resumed.payload["units"]
    with pytest.raises(ValueError, match="transition"):
        resumed.mark_unit("late-matched", "L1", 41, "evaluating")


def test_p0d2h_manifest_rejects_source_path_change(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    P0D2HManifest.load_or_create(
        path,
        run_id="hard-run",
        config=_config(),
        source_manifest_path="/source/a/manifest.json",
        prompt_token_audit_sha256="audit-1",
    )
    with pytest.raises(ValueError, match="source manifest path"):
        P0D2HManifest.load_or_create(
            path,
            run_id="hard-run",
            config=_config(),
            source_manifest_path="/source/b/manifest.json",
            prompt_token_audit_sha256="audit-1",
        )
