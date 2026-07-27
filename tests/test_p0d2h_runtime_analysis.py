from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

import plasticity_placement.p0d2.runtime as source_runtime
import plasticity_placement.p0d2h.runtime as stress_runtime
from plasticity_placement.p0c.compiler import (
    ACTIONS,
    compile_bank,
    write_compiled_bank,
)
from plasticity_placement.p0c.domain import Arm
from plasticity_placement.p0c.modeling import P0CProbeResult
from plasticity_placement.p0d2.analysis import (
    aggregate_experiment as aggregate_source,
)
from plasticity_placement.p0d2.config import P0D2Request
from plasticity_placement.p0d2.runtime import run_experiment as run_source
from plasticity_placement.p0d2h.analysis import aggregate_experiment
from plasticity_placement.p0d2h.config import P0D2HRequest
from plasticity_placement.p0d2h.runtime import run_experiment
from plasticity_placement.training.config import select_layers
from plasticity_placement.training.lora import TrainingSummary


class _FakeModel:
    def __init__(self, condition_id: str) -> None:
        self.condition_id = condition_id
        self.disabled = False

    @contextmanager
    def disable_adapter(self):
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = False


class _FakeBundle:
    def __init__(self, condition_id: str) -> None:
        self.model = _FakeModel(condition_id)
        self.precision = "float32"


def test_fake_hard_probe_run_resumes_and_aggregates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p0c_manifest = _write_p0c_source(tmp_path / "p0c")
    p0d2_dir = tmp_path / "p0d2"
    _patch_source_backend(monkeypatch)
    p0d2_manifest = run_source(
        P0D2Request(
            output_dir=p0d2_dir,
            source_manifest=p0c_manifest,
        ),
        num_hidden_layers=24,
    )
    aggregate_source(p0d2_dir, bootstrap_samples=20)
    source_adapter_mtimes = {
        path: path.stat().st_mtime_ns
        for path in p0d2_dir.glob(
            "adapters/*/*/seed-*/adapter_model.safetensors"
        )
    }
    assert len(source_adapter_mtimes) == 504

    _patch_stress_backend(monkeypatch)
    output_dir = tmp_path / "hard"
    request = P0D2HRequest(
        output_dir=output_dir,
        source_manifest=p0d2_manifest,
    )
    manifest_path = run_experiment(request)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["selected_lessons"]) == 24
    assert len(manifest["conditions"]) == 4
    assert len(manifest["units"]) == 288
    assert {
        value["state"] for value in manifest["base_arms"].values()
    } == {"verified"}
    assert {
        value["state"] for value in manifest["units"].values()
    } == {"verified"}
    assert {
        path: path.stat().st_mtime_ns for path in source_adapter_mtimes
    } == source_adapter_mtimes

    result_mtimes = {
        path: path.stat().st_mtime_ns
        for path in output_dir.glob("results/adapters/*/*/seed-*.jsonl")
    }
    assert len(result_mtimes) == 288
    run_experiment(request)
    assert {
        path: path.stat().st_mtime_ns for path in result_mtimes
    } == result_mtimes

    summary_path = aggregate_experiment(
        output_dir,
        bootstrap_samples=20,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["adapter_unit_count"] == 288
    assert summary["probe_row_count"] == 5_376
    assert (
        summary["condition_summary"]["late-matched"]["hard_accuracy"]
        == 1.0
    )
    assert summary["condition_summary"]["full-base"]["hard_accuracy"] == 0.0
    assert summary["gates"]["decision_status"] == "hard_locus_robust"
    assert summary["gates"]["automatic_training_started"] is False

    first_result = next(
        output_dir.glob("results/adapters/*/*/seed-*.jsonl")
    )
    original_result = first_result.read_bytes()
    first_result.write_bytes(original_result.splitlines(keepends=True)[0])
    with pytest.raises(ValueError, match="result matrix"):
        aggregate_experiment(output_dir, bootstrap_samples=10)
    first_result.write_bytes(original_result)

    first_adapter = next(iter(source_adapter_mtimes))
    original_adapter = first_adapter.read_bytes()
    first_adapter.write_bytes(original_adapter + b"tampered")
    with pytest.raises(ValueError, match="adapter bundle changed"):
        aggregate_experiment(output_dir, bootstrap_samples=10)
    first_adapter.write_bytes(original_adapter)


def _write_p0c_source(source_dir: Path) -> Path:
    hashes = write_compiled_bank(source_dir)
    selected = [
        item for item in compile_bank() if item.lesson.split == "confirmatory"
    ]
    lesson_ids = [item.lesson.lesson_id for item in selected]
    seeds = [41, 42, 43]
    manifest = {
        "run_id": "p0c-confirmatory-source",
        "config": {
            "tier": "confirmatory",
            "output_dir": str(source_dir),
            "model_name": "model-a",
            "model_revision": "revision-1",
            "use_4bit": False,
            "rank": 4,
            "alpha": 8,
            "learning_rate": 2e-4,
            "max_steps": 16,
            "max_length": 256,
            "max_new_tokens": 8,
            "training_seeds": seeds,
            "calibration_report_sha256": "calibration-1",
        },
        "compiler_hashes": hashes,
        "selected_lessons": lesson_ids,
        "base_arms": {
            lesson_id: "verified" for lesson_id in lesson_ids
        },
        "units": {
            f"{lesson_id}::seed-{seed}": {
                "state": "verified",
                "rollback_exact_match_rate": 1.0,
            }
            for lesson_id in lesson_ids
            for seed in seeds
        },
    }
    path = source_dir / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _patch_source_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        source_runtime,
        "resolve_model_revision",
        lambda model_name, revision: revision,
    )
    monkeypatch.setattr(
        source_runtime,
        "current_code_hash",
        lambda: "source-code-hash",
    )
    monkeypatch.setattr(
        source_runtime,
        "_write_environment",
        lambda output: None,
    )
    monkeypatch.setattr(
        source_runtime,
        "load_base_model",
        lambda config: _FakeBundle("base"),
    )
    monkeypatch.setattr(
        source_runtime,
        "load_adapter_model",
        lambda config, path: _FakeBundle(path.parts[-3]),
    )
    monkeypatch.setattr(
        source_runtime,
        "release_model",
        lambda bundle: None,
    )
    monkeypatch.setattr(
        source_runtime,
        "_clear_accelerator_cache",
        lambda: None,
    )
    monkeypatch.setattr(
        source_runtime,
        "train_lora",
        _fake_train_lora,
    )
    monkeypatch.setattr(
        source_runtime,
        "evaluate_probes",
        _fake_source_evaluate,
    )


def _patch_stress_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        stress_runtime,
        "resolve_model_revision",
        lambda model_name, revision: revision,
    )
    monkeypatch.setattr(
        stress_runtime,
        "current_code_hash",
        lambda: "stress-code-hash",
    )
    monkeypatch.setattr(
        stress_runtime,
        "_write_environment",
        lambda output: None,
    )
    monkeypatch.setattr(
        stress_runtime,
        "load_base_model",
        lambda config: _FakeBundle("base"),
    )
    monkeypatch.setattr(
        stress_runtime,
        "load_adapter_model",
        lambda config, path: _FakeBundle(path.parts[-3]),
    )
    monkeypatch.setattr(
        stress_runtime,
        "release_model",
        lambda bundle: None,
    )
    monkeypatch.setattr(
        stress_runtime,
        "evaluate_probes",
        _fake_stress_evaluate,
    )


def _fake_train_lora(config) -> TrainingSummary:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    payload = f"{config.layer_band}:{config.rank}:{config.seed}".encode()
    (config.output_dir / "adapter_model.safetensors").write_bytes(payload)
    (config.output_dir / "adapter_config.json").write_text(
        "{}",
        encoding="utf-8",
    )
    adapter_hash = source_runtime._adapter_hash(config.output_dir)
    selected = select_layers(config.layer_band, 24, config.explicit_layers)
    selected_count = 24 if selected is None else len(selected)
    trainable_parameters = selected_count * config.rank * 100
    summary = TrainingSummary(
        output_dir=str(config.output_dir),
        selected_layers=selected,
        trainable_parameters=trainable_parameters,
        total_parameters=100_000,
        optimizer_steps=config.max_steps,
        final_loss=0.1,
        elapsed_seconds=1.0,
        peak_memory_bytes=1_000,
        adapter_bytes=trainable_parameters,
        model_revision=config.model_revision,
        training_data_sha256=sha256(
            config.data_path.read_bytes()
        ).hexdigest(),
        config_sha256=source_runtime._json_hash(config.to_dict()),
        adapter_sha256=adapter_hash,
        precision="float32",
    )
    (config.output_dir / "training_metadata.json").write_text(
        json.dumps(
            {"config": config.to_dict(), "summary": asdict(summary)}
        ),
        encoding="utf-8",
    )
    return summary


def _fake_source_evaluate(**kwargs: Any) -> list[P0CProbeResult]:
    bundle = kwargs["bundle"]
    arm = Arm(kwargs["arm"])
    condition_id = bundle.model.condition_id
    results: list[P0CProbeResult] = []
    for probe in kwargs["probes"]:
        target = probe.category in {"paraphrase", "compositional"}
        if arm is Arm.EXTERNAL:
            correct = True
        elif arm is Arm.PARAMETRIC:
            correct = (
                not target
                or condition_id in {"full-base", "late-matched"}
            )
        elif arm in {Arm.NO_WRITE, Arm.ROLLBACK}:
            correct = not target
        else:
            raise AssertionError(arm)
        results.append(
            _probe_result(kwargs, probe, arm, correct)
        )
    return results


def _fake_stress_evaluate(**kwargs: Any) -> list[P0CProbeResult]:
    bundle = kwargs["bundle"]
    arm = Arm(kwargs["arm"])
    if arm is Arm.EXTERNAL:
        correct = True
    elif arm is Arm.PARAMETRIC:
        correct = bundle.model.condition_id == "late-matched"
    elif arm is Arm.NO_WRITE:
        correct = False
    else:
        raise AssertionError(arm)
    return [
        _probe_result(kwargs, probe, arm, correct)
        for probe in kwargs["probes"]
    ]


def _probe_result(
    kwargs: dict[str, Any],
    probe,
    arm: Arm,
    correct: bool,
) -> P0CProbeResult:
    predicted = (
        probe.expected_action
        if correct
        else next(
            action
            for action in ACTIONS
            if action != probe.expected_action
        )
    )
    return P0CProbeResult(
        run_id=kwargs["run_id"],
        model=kwargs["config"].model_name,
        model_revision=kwargs["config"].model_revision,
        precision="float32",
        lesson_id=probe.lesson_id,
        pair_id=probe.pair_id,
        lesson_type=probe.lesson_type,
        training_seed=kwargs["training_seed"],
        arm=arm,
        probe_id=probe.probe_id,
        category=probe.category,
        expected_action=probe.expected_action,
        predicted_action=predicted,
        correct=correct,
        invalid=False,
        generated_text=predicted,
        input_tokens=160,
        generated_tokens=1,
        latency_seconds=0.1,
        prompt_sha256=sha256(probe.prompt.encode()).hexdigest(),
        adapter_sha256=kwargs.get("adapter_sha256"),
    )
