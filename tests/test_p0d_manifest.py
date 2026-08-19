from pathlib import Path

import pytest

from plasticity_placement.p0d.manifest import P0DManifest, unit_key


def _config() -> dict[str, object]:
    return {
        "selected_lesson_ids": ["L1"],
        "conditions": [
            {
                "condition_id": "middle",
                "layer_band": "middle",
                "explicit_layers": [],
                "selected_layers": [1],
                "parameter_budget_regime": "fixed_local",
            }
        ],
    }


def test_condition_aware_manifest_resume_and_immutability(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = P0DManifest.load_or_create(
        path,
        run_id="p0d-run",
        config=_config(),
        compiler_hashes={"compiler_version": "v4"},
    )
    manifest.mark_unit("middle", "L1", 42, "training")
    manifest.mark_unit("middle", "L1", 42, "trained")
    manifest.mark_unit("middle", "L1", 42, "evaluated")
    manifest.mark_unit(
        "middle",
        "L1",
        42,
        "verified",
        rollback_exact_match_rate=1.0,
    )
    resumed = P0DManifest.load_or_create(
        path,
        run_id="p0d-run",
        config=_config(),
        compiler_hashes={"compiler_version": "v4"},
    )
    assert resumed.unit_state("middle", "L1", 42) == "verified"
    assert unit_key("middle", "L1", 42) in resumed.payload["units"]
    with pytest.raises(ValueError, match="transition"):
        resumed.mark_unit("middle", "L1", 42, "training")


def test_failed_unit_is_immutable(tmp_path: Path) -> None:
    manifest = P0DManifest.load_or_create(
        tmp_path / "manifest.json",
        run_id="p0d-run",
        config=_config(),
        compiler_hashes={"compiler_version": "v4"},
    )
    manifest.mark_unit("middle", "L1", 42, "failed")
    with pytest.raises(ValueError, match="transition"):
        manifest.mark_unit("middle", "L1", 42, "training")


def test_manifest_rejects_changed_condition_matrix(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    P0DManifest.load_or_create(
        path,
        run_id="p0d-run",
        config=_config(),
        compiler_hashes={"compiler_version": "v4"},
    )
    changed = _config()
    changed["conditions"] = [
        {
            "condition_id": "late",
            "layer_band": "late",
            "explicit_layers": [],
            "selected_layers": [2],
            "parameter_budget_regime": "fixed_local",
        }
    ]
    with pytest.raises(ValueError, match="config"):
        P0DManifest.load_or_create(
            path,
            run_id="p0d-run",
            config=changed,
            compiler_hashes={"compiler_version": "v4"},
        )
