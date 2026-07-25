import json
from pathlib import Path

import pytest

import plasticity_placement.p0c.calibration as calibration_module
from plasticity_placement.p0c.calibration import (
    CalibrationConfig,
    _calibration_score,
    _combine_summaries,
    _qualified,
    _stage1_sort_key,
    calibration_candidates,
    run_calibration,
)


def test_calibration_grid_and_successive_halving_counts() -> None:
    candidates = calibration_candidates()
    assert len(candidates) == 18
    assert len({candidate.candidate_id for candidate in candidates}) == 18
    assert len(candidates) * 2 == 36
    assert 6 * 4 == 24


def test_calibration_qualification_uses_development_metrics() -> None:
    good = {
        "p_target": 0.9,
        "n_target": 0.2,
        "target_gain": 0.7,
        "p_exact": 1.0,
        "p_interference": 0.0,
        "invalid_increase": 0.0,
        "p_write_time": 1.0,
        "p_peak_memory": 1.0,
        "p_adapter_bytes": 1.0,
        "resolved_model_revision": "revision-1",
    }
    bad = {**good, "target_gain": 0.1}
    assert _qualified(good)
    assert not _qualified(bad)
    assert _calibration_score(good) == 0.7
    combined = _combine_summaries(good, bad, 2, 4)
    assert combined["target_gain"] == 0.3


def test_calibration_orchestrates_sixty_adapter_units(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run_candidate(**kwargs):
        lesson_ids = kwargs["lesson_ids"]
        calls.append(lesson_ids)
        return {
            "p_target": 0.9,
            "n_target": 0.2,
            "target_gain": 0.7,
            "p_exact": 1.0,
            "p_interference": 0.0,
            "invalid_increase": 0.0,
            "p_write_time": 1.0,
            "p_peak_memory": 1.0,
            "p_adapter_bytes": 1.0,
            "resolved_model_revision": "revision-1",
        }

    monkeypatch.setattr(calibration_module, "_run_candidate", fake_run_candidate)
    monkeypatch.setattr(
        calibration_module,
        "resolve_model_revision",
        lambda model_name, revision: "revision-1",
    )
    path = run_calibration(CalibrationConfig(output_dir=tmp_path))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["stage1_adapter_count"] == 36
    assert report["stage2_adapter_count"] == 24
    assert sum(len(lesson_ids) for lesson_ids in calls) == 60
    assert calls[:18] == [("D_01", "D_04")] * 18
    assert calls[18:] == [("D_02", "D_03", "D_05", "D_06")] * 6
    assert report["selected_config"] is not None
    assert report["provenance"]["resolved_model_revision"] == "revision-1"


def test_stage1_tie_break_uses_write_time_before_candidate_id() -> None:
    common = {
        "stage1_score": 0.5,
        "max_steps": 4,
        "rank": 4,
    }
    slow = {
        **common,
        "candidate_id": "a-first-lexically",
        "stage1": {"p_write_time": 2.0},
    }
    fast = {
        **common,
        "candidate_id": "z-last-lexically",
        "stage1": {"p_write_time": 1.0},
    }
    assert sorted([slow, fast], key=_stage1_sort_key)[0] is fast


def test_calibration_records_candidate_failure_and_continues(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = 0

    def fake_run_candidate(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic OOM")
        return {
            "p_target": 0.9,
            "n_target": 0.2,
            "target_gain": 0.7,
            "p_exact": 1.0,
            "p_interference": 0.0,
            "invalid_increase": 0.0,
            "p_write_time": 1.0,
            "p_peak_memory": 1.0,
            "p_adapter_bytes": 1.0,
            "resolved_model_revision": "revision-1",
        }

    monkeypatch.setattr(calibration_module, "_run_candidate", fake_run_candidate)
    monkeypatch.setattr(
        calibration_module,
        "resolve_model_revision",
        lambda model_name, revision: "revision-1",
    )
    report_path = run_calibration(CalibrationConfig(output_dir=tmp_path))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["stage1_results"][0]["status"] == "failed"
    assert report["stage1_results"][0]["error"]["message"] == "synthetic OOM"
    assert report["selected_config"] is not None


def test_completed_calibration_report_is_terminal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = 0

    def fake_run_candidate(**kwargs):
        nonlocal calls
        calls += 1
        return {
            "p_target": 0.9,
            "n_target": 0.2,
            "target_gain": 0.7,
            "p_exact": 1.0,
            "p_interference": 0.0,
            "invalid_increase": 0.0,
            "p_write_time": 1.0,
            "p_peak_memory": 1.0,
            "p_adapter_bytes": 1.0,
            "resolved_model_revision": "revision-1",
        }

    monkeypatch.setattr(calibration_module, "_run_candidate", fake_run_candidate)
    monkeypatch.setattr(
        calibration_module,
        "resolve_model_revision",
        lambda model_name, revision: "revision-1",
    )
    config = CalibrationConfig(output_dir=tmp_path)
    first_path = run_calibration(config)
    assert calls == 24

    second_path = run_calibration(config)

    assert second_path == first_path
    assert calls == 24
    mismatched = CalibrationConfig(output_dir=tmp_path, max_length=192)
    with pytest.raises(ValueError, match="config mismatch"):
        run_calibration(mismatched)
