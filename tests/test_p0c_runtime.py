import json
from hashlib import sha256
from pathlib import Path

import pytest

import plasticity_placement.p0c.runtime as runtime_module
from plasticity_placement.p0c.compiler import compile_bank
from plasticity_placement.p0c.config import P0CConfig
from plasticity_placement.p0c.domain import Tier
from plasticity_placement.p0c.manifest import RunManifest
from plasticity_placement.p0c.runtime import (
    _adapter_hash,
    _adapter_training_config,
    _load_training_metadata,
    _run_adapter_unit,
    _validate_selected_lessons,
    _write_environment,
    current_environment_fingerprint,
    select_screened_lessons,
)


def test_pilot_selection_preserves_complete_counterbalanced_pairs() -> None:
    compiled = compile_bank()
    screening_results = [
        {
            "lesson_id": item.lesson.lesson_id,
            "correct": False,
            "invalid": False,
        }
        for item in compiled
        if item.lesson.split in {"confirmatory", "reserve"}
        for _ in range(4)
    ]
    selected = select_screened_lessons(
        compiled=compiled,
        screening_results=screening_results,
        tier=Tier.PILOT,
        accuracy_threshold=0.5,
        invalid_threshold=0.5,
    )
    assert len(selected) == 8
    assert sum(item.lesson.lesson_type == "fact_mapping" for item in selected) == 4
    assert sum(item.lesson.lesson_type == "procedure_recovery" for item in selected) == 4
    pair_counts: dict[str, int] = {}
    for item in selected:
        pair_counts[item.lesson.pair_id] = pair_counts.get(item.lesson.pair_id, 0) + 1
    assert set(pair_counts.values()) == {2}
    for lesson_type in ("fact_mapping", "procedure_recovery"):
        actions = [
            item.lesson.desired_action
            for item in selected
            if item.lesson.lesson_type == lesson_type
        ]
        assert len(set(actions)) == 4


def test_pilot_selection_uses_reserves_before_accepting_action_imbalance(
    tmp_path: Path,
) -> None:
    compiled = compile_bank()
    eligible_pair_ids = {
        "F_pair01",
        "F_pair04",
        "RF_pair01",
        "RF_pair02",
        "P_pair01",
        "P_pair04",
        "RP_pair01",
        "RP_pair02",
    }
    screening_results = [
        {
            "lesson_id": item.lesson.lesson_id,
            "correct": item.lesson.pair_id not in eligible_pair_ids,
            "invalid": False,
        }
        for item in compiled
        if item.lesson.split in {"confirmatory", "reserve"}
        for _ in range(4)
    ]

    selected = select_screened_lessons(
        compiled=compiled,
        screening_results=screening_results,
        tier=Tier.PILOT,
        accuracy_threshold=0.5,
        invalid_threshold=0.5,
    )

    config = P0CConfig(
        output_dir=tmp_path,
        tier=Tier.PILOT,
        calibration_report_sha256="calibration-report",
    )
    _validate_selected_lessons(config, selected)
    assert {item.lesson.split for item in selected} == {"reserve"}
    for lesson_type in ("fact_mapping", "procedure_recovery"):
        actions = {
            item.lesson.desired_action
            for item in selected
            if item.lesson.lesson_type == lesson_type
        }
        assert len(actions) == 4


def test_confirmatory_v4_selection_recovers_v3_fact_screening_shortfall(
    tmp_path: Path,
) -> None:
    compiled = compile_bank()
    eligible_pair_ids = {
        "F_pair01",
        "F_pair04",
        "F_pair06",
        "RF_pair01",
        "RF_pair02",
        "RF_pair03",
        "RF_pair04",
        "P_pair01",
        "P_pair02",
        "P_pair03",
        "P_pair04",
        "P_pair05",
        "P_pair06",
        "RP_pair01",
        "RP_pair02",
    }
    screening_results = [
        {
            "lesson_id": item.lesson.lesson_id,
            "correct": item.lesson.pair_id not in eligible_pair_ids,
            "invalid": False,
        }
        for item in compiled
        if item.lesson.split in {"confirmatory", "reserve"}
        for _ in range(4)
    ]

    selected = select_screened_lessons(
        compiled=compiled,
        screening_results=screening_results,
        tier=Tier.CONFIRMATORY,
        accuracy_threshold=0.5,
        invalid_threshold=0.5,
    )

    config = P0CConfig(
        output_dir=tmp_path,
        tier=Tier.CONFIRMATORY,
        training_seeds=(41, 42, 43),
        calibration_report_sha256="calibration-report",
    )
    _validate_selected_lessons(config, selected)
    assert len(selected) == 24
    selected_pair_ids = {item.lesson.pair_id for item in selected}
    assert {"RF_pair03", "RF_pair04"}.issubset(selected_pair_ids)
    for lesson_type in ("fact_mapping", "procedure_recovery"):
        lessons = [item for item in selected if item.lesson.lesson_type == lesson_type]
        action_counts = {
            action: sum(item.lesson.desired_action == action for item in lessons)
            for action in ("act_n7", "act_p3", "act_v9", "act_k2")
        }
        assert action_counts == {
            "act_n7": 3,
            "act_p3": 3,
            "act_v9": 3,
            "act_k2": 3,
        }


def test_training_metadata_recovery_validates_hashes(tmp_path: Path) -> None:
    item = compile_bank()[0]
    output_dir = tmp_path / "run"
    data_path = output_dir / "compiled" / "training" / "D_01.jsonl"
    data_path.parent.mkdir(parents=True)
    data_path.write_text('{"prompt":"p","completion":"a"}\n', encoding="utf-8")
    adapter_dir = output_dir / "adapters" / "D_01" / "seed-42"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    config = P0CConfig(
        output_dir=output_dir,
        tier=Tier.DEVELOPMENT,
        model_revision="revision-1",
    )
    training_config = _adapter_training_config(config, item, 42, adapter_dir)
    config_payload = json.dumps(
        training_config.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    summary = {
        "elapsed_seconds": 1.0,
        "peak_memory_bytes": 2,
        "adapter_bytes": 3,
        "training_data_sha256": sha256(data_path.read_bytes()).hexdigest(),
        "config_sha256": sha256(config_payload.encode()).hexdigest(),
        "adapter_sha256": _adapter_hash(adapter_dir),
        "model_revision": "revision-1",
        "precision": "float32",
    }
    metadata_path = adapter_dir / "training_metadata.json"
    metadata_path.write_text(
        json.dumps({"config": training_config.to_dict(), "summary": summary}),
        encoding="utf-8",
    )
    assert _load_training_metadata(metadata_path, training_config, adapter_dir) == summary
    metadata_path.write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        _load_training_metadata(metadata_path, training_config, adapter_dir)


def test_environment_change_is_rejected(tmp_path: Path, monkeypatch) -> None:
    snapshot = {
        "python": "3.12",
        "platform": "test",
        "packages": {"torch": "1"},
        "cuda_available": True,
        "cuda_version": "12",
        "gpu": "T4",
        "code_sha256": "code-1",
    }
    current = [snapshot]
    monkeypatch.setattr(
        runtime_module,
        "_environment_snapshot",
        lambda: current[0],
    )
    _write_environment(tmp_path)
    _write_environment(tmp_path)
    sessions = json.loads((tmp_path / "environment_sessions.json").read_text(encoding="utf-8"))
    assert len(sessions) == 2
    current[0] = {**snapshot, "gpu": "L4"}
    with pytest.raises(RuntimeError, match="environment changed"):
        _write_environment(tmp_path)


def test_environment_fingerprint_uses_only_critical_fields(monkeypatch) -> None:
    snapshot = {
        "python": "3.12",
        "platform": "platform-a",
        "packages": {"torch": "1"},
        "cuda_available": True,
        "cuda_version": "12",
        "gpu": "T4",
        "git_commit": "commit-a",
        "code_sha256": "code-1",
    }
    current = [snapshot]
    monkeypatch.setattr(runtime_module, "_environment_snapshot", lambda: current[0])
    first = current_environment_fingerprint()
    current[0] = {**snapshot, "platform": "platform-b", "git_commit": "commit-b"}
    assert current_environment_fingerprint() == first
    current[0] = {**snapshot, "gpu": "L4"}
    assert current_environment_fingerprint() != first


def test_evaluated_unit_resume_records_rollback_rate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = P0CConfig(
        output_dir=tmp_path,
        tier=Tier.DEVELOPMENT,
        model_revision="revision-1",
    )
    item = compile_bank()[0]
    lesson_id = item.lesson.lesson_id
    unit_id = f"{lesson_id}::seed-42"
    result_path = tmp_path / "results" / "raw" / lesson_id / "seed-42" / "adapter_arms.jsonl"
    result_path.parent.mkdir(parents=True)
    result_path.write_text("", encoding="utf-8")
    manifest = RunManifest(
        path=tmp_path / "manifest.json",
        payload={
            "units": {
                unit_id: {
                    "lesson_id": lesson_id,
                    "seed": 42,
                    "state": "evaluated",
                    "adapter_sha256": "adapter-hash",
                }
            },
            "base_arms": {},
            "errors": [],
        },
    )
    monkeypatch.setattr(runtime_module, "_adapter_hash", lambda path: "adapter-hash")
    monkeypatch.setattr(runtime_module, "_verify_adapter_result_rows", lambda *args: None)
    monkeypatch.setattr(runtime_module, "_verify_rollback", lambda *args: 1.0)

    _run_adapter_unit(config, "run-1", manifest, item, 42)

    unit = manifest.payload["units"][unit_id]
    assert unit["state"] == "verified"
    assert unit["rollback_exact_match_rate"] == 1.0
