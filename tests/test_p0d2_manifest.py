from pathlib import Path

import pytest

from plasticity_placement.p0d2.manifest import P0D2Manifest, unit_key


def _config() -> dict[str, object]:
    return {
        "selected_lesson_ids": ["L1"],
        "conditions": [
            {
                "condition_id": "middle-matched",
                "rank": 24,
                "alpha": 48,
                "target_modules": ["q_proj", "v_proj"],
            }
        ],
    }


def test_p0d2_manifest_resume_and_immutability(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = P0D2Manifest.load_or_create(
        path,
        run_id="p0d2-run",
        config=_config(),
        compiler_hashes={"compiler_version": "v4"},
    )
    manifest.mark_unit("middle-matched", "L1", 42, "training")
    manifest.mark_unit("middle-matched", "L1", 42, "trained")
    manifest.mark_unit("middle-matched", "L1", 42, "evaluated")
    manifest.mark_unit(
        "middle-matched",
        "L1",
        42,
        "verified",
        rollback_exact_match_rate=1.0,
    )
    resumed = P0D2Manifest.load_or_create(
        path,
        run_id="p0d2-run",
        config=_config(),
        compiler_hashes={"compiler_version": "v4"},
    )
    assert resumed.unit_state("middle-matched", "L1", 42) == "verified"
    assert unit_key("middle-matched", "L1", 42) in resumed.payload["units"]
    with pytest.raises(ValueError, match="transition"):
        resumed.mark_unit("middle-matched", "L1", 42, "training")


def test_p0d2_manifest_rejects_condition_hyperparameter_change(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    P0D2Manifest.load_or_create(
        path,
        run_id="p0d2-run",
        config=_config(),
        compiler_hashes={"compiler_version": "v4"},
    )
    changed = _config()
    changed["conditions"] = [
        {
            "condition_id": "middle-matched",
            "rank": 32,
            "alpha": 64,
            "target_modules": ["q_proj", "v_proj"],
        }
    ]
    with pytest.raises(ValueError, match="config"):
        P0D2Manifest.load_or_create(
            path,
            run_id="p0d2-run",
            config=changed,
            compiler_hashes={"compiler_version": "v4"},
        )
