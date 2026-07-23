from pathlib import Path

import pytest

from plasticity_placement.p0c.manifest import RunManifest


def test_manifest_round_trip_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = RunManifest.load_or_create(
        path,
        run_id="run-1",
        config={"tier": "smoke"},
        compiler_hashes={"compiler_version": "v1"},
    )
    manifest.set_selected_lessons(["D_01", "D_02"])
    manifest.mark_unit("D_01", 42, "training")
    manifest.mark_unit("D_01", 42, "trained")
    manifest.mark_unit("D_01", 42, "evaluated")
    manifest.mark_unit("D_01", 42, "verified", rollback_exact_match_rate=1.0)

    resumed = RunManifest.load_or_create(
        path,
        run_id="run-1",
        config={"tier": "smoke"},
        compiler_hashes={"compiler_version": "v1"},
    )
    assert resumed.unit_state("D_01", 42) == "verified"
    assert resumed.payload["selected_lessons"] == ["D_01", "D_02"]


def test_failed_units_are_immutable(tmp_path: Path) -> None:
    manifest = RunManifest.load_or_create(
        tmp_path / "manifest.json",
        run_id="run-1",
        config={"tier": "smoke"},
        compiler_hashes={},
    )
    manifest.mark_unit("D_01", 42, "training")
    manifest.mark_unit("D_01", 42, "failed")
    with pytest.raises(ValueError, match="failed -> training"):
        manifest.mark_unit("D_01", 42, "training")


def test_manifest_rejects_changed_config(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    RunManifest.load_or_create(
        path,
        run_id="run-1",
        config={"tier": "smoke"},
        compiler_hashes={},
    )
    with pytest.raises(ValueError):
        RunManifest.load_or_create(
            path,
            run_id="run-1",
            config={"tier": "pilot"},
            compiler_hashes={},
        )
