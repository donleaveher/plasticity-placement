from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

import plasticity_placement.p0d.runtime as runtime_module
from plasticity_placement.p0c.compiler import ACTIONS, compile_bank, write_compiled_bank
from plasticity_placement.p0c.domain import Arm
from plasticity_placement.p0c.modeling import P0CProbeResult
from plasticity_placement.p0d.analysis import aggregate_experiment
from plasticity_placement.p0d.config import P0DRequest
from plasticity_placement.p0d.runtime import run_experiment
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


def test_fake_band_scan_runs_resumes_and_aggregates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_manifest = _write_source_manifest(tmp_path / "source")
    output_dir = tmp_path / "p0d-band"
    _patch_fake_backend(monkeypatch)
    request = P0DRequest(output_dir=output_dir, source_manifest=source_manifest)

    manifest_path = run_experiment(request, num_hidden_layers=24)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["selected_lessons"]) == 24
    assert len(manifest["conditions"]) == 4
    assert len(manifest["units"]) == 288
    assert set(manifest["base_arms"].values()) == {"verified"}
    assert {
        str(unit["state"]) for unit in manifest["units"].values()
    } == {"verified"}

    adapter_mtimes = {
        path: path.stat().st_mtime_ns
        for path in output_dir.glob("adapters/*/*/seed-*/adapter_model.safetensors")
    }
    assert len(adapter_mtimes) == 288
    run_experiment(request, num_hidden_layers=24)
    assert {
        path: path.stat().st_mtime_ns for path in adapter_mtimes
    } == adapter_mtimes

    summary_path = aggregate_experiment(output_dir, bootstrap_samples=100)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["adapter_unit_count"] == 288
    assert summary["probe_row_count"] == 16_224
    assert summary["condition_summary"]["full"]["target_generalization"] == 1.0
    assert summary["condition_summary"]["middle"]["target_generalization"] == 1.0
    assert summary["condition_summary"]["early"]["target_generalization"] == 0.0
    assert summary["contrasts"]["full-vs-no_write"]["target_mean"] == 1.0
    assert summary["gates"]["conditions"]["middle"]["retains_full_gain"] is True
    assert summary["gates"]["eligible_locus_conditions"] == ["middle"]
    candidates = json.loads(
        (
            output_dir / "results" / "aggregate" / "next_stage_candidates.json"
        ).read_text(encoding="utf-8")
    )
    assert candidates["status"] == "manual_freeze_required"
    assert candidates["selection_automatic"] is False
    first_adapter = next(
        output_dir.glob("adapters/*/*/seed-*/adapter_model.safetensors")
    )
    first_adapter.write_bytes(first_adapter.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="adapter bundle changed"):
        aggregate_experiment(output_dir, bootstrap_samples=10)


def test_aggregate_rejects_partial_condition_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_manifest = _write_source_manifest(tmp_path / "source")
    output_dir = tmp_path / "p0d-band"
    _patch_fake_backend(monkeypatch)
    run_experiment(
        P0DRequest(output_dir=output_dir, source_manifest=source_manifest),
        num_hidden_layers=24,
    )
    path = output_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["units"].pop(next(iter(manifest["units"])))
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unit matrix"):
        aggregate_experiment(output_dir, bootstrap_samples=10)


def test_source_compiler_mismatch_is_rejected(tmp_path: Path) -> None:
    source_manifest = _write_source_manifest(tmp_path / "source")
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    source["compiler_hashes"]["lessons_sha256"] = "changed"
    source_manifest.write_text(json.dumps(source), encoding="utf-8")
    request = P0DRequest(
        output_dir=tmp_path / "p0d-band",
        source_manifest=source_manifest,
    )
    with pytest.raises(ValueError, match="compiler hashes"):
        runtime_module.resolve_request(request, num_hidden_layers=24)


def test_run_id_covers_code_conditions_seeds_and_compiler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_manifest = _write_source_manifest(tmp_path / "source")
    monkeypatch.setattr(
        runtime_module,
        "resolve_model_revision",
        lambda model_name, revision: revision,
    )
    monkeypatch.setattr(runtime_module, "current_code_hash", lambda: "code-hash")
    config, hashes, _ = runtime_module.resolve_request(
        P0DRequest(
            output_dir=tmp_path / "p0d-band",
            source_manifest=source_manifest,
        ),
        num_hidden_layers=24,
    )
    original = runtime_module._run_id(config, hashes)
    assert runtime_module._run_id(
        replace(config, code_sha256="changed"),
        hashes,
    ) != original
    changed_condition = replace(
        config.conditions[1],
        selected_layers=(0,),
    )
    assert runtime_module._run_id(
        replace(
            config,
            conditions=(config.conditions[0], changed_condition, *config.conditions[2:]),
        ),
        hashes,
    ) != original
    assert runtime_module._run_id(
        replace(config, training_seeds=(41, 42, 44)),
        hashes,
    ) != original
    assert runtime_module._run_id(
        config,
        {**hashes, "lessons_sha256": "changed"},
    ) != original


def _write_source_manifest(source_dir: Path) -> Path:
    hashes = write_compiled_bank(source_dir)
    selected = [
        item for item in compile_bank() if item.lesson.split == "confirmatory"
    ]
    assert len(selected) == 24
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
        "base_arms": {lesson_id: "verified" for lesson_id in lesson_ids},
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


def _patch_fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime_module,
        "resolve_model_revision",
        lambda model_name, revision: revision,
    )
    monkeypatch.setattr(runtime_module, "current_code_hash", lambda: "code-hash")
    monkeypatch.setattr(runtime_module, "_write_environment", lambda output: None)
    monkeypatch.setattr(
        runtime_module,
        "load_base_model",
        lambda config: _FakeBundle("base"),
    )
    monkeypatch.setattr(
        runtime_module,
        "load_adapter_model",
        lambda config, path: _FakeBundle(path.parts[-3]),
    )
    monkeypatch.setattr(runtime_module, "release_model", lambda bundle: None)
    monkeypatch.setattr(runtime_module, "_clear_accelerator_cache", lambda: None)
    monkeypatch.setattr(runtime_module, "train_lora", _fake_train_lora)
    monkeypatch.setattr(runtime_module, "evaluate_probes", _fake_evaluate_probes)


def _fake_train_lora(config) -> TrainingSummary:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "adapter_model.safetensors").write_bytes(
        f"{config.layer_band}:{config.explicit_layers}:{config.seed}".encode()
    )
    (config.output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    adapter_hash = runtime_module._adapter_hash(config.output_dir)
    selected = select_layers(config.layer_band, 24, config.explicit_layers)
    summary = TrainingSummary(
        output_dir=str(config.output_dir),
        selected_layers=selected,
        trainable_parameters=100 * (24 if selected is None else len(selected)),
        total_parameters=10_000,
        optimizer_steps=config.max_steps,
        final_loss=0.1,
        elapsed_seconds=1.0,
        peak_memory_bytes=1_000,
        adapter_bytes=100,
        model_revision=config.model_revision,
        training_data_sha256=sha256(config.data_path.read_bytes()).hexdigest(),
        config_sha256=runtime_module._json_hash(config.to_dict()),
        adapter_sha256=adapter_hash,
        precision="float32",
    )
    (config.output_dir / "training_metadata.json").write_text(
        json.dumps({"config": config.to_dict(), "summary": asdict(summary)}),
        encoding="utf-8",
    )
    return summary


def _fake_evaluate_probes(**kwargs: Any) -> list[P0CProbeResult]:
    bundle = kwargs["bundle"]
    arm = Arm(kwargs["arm"])
    probes = list(kwargs["probes"])
    condition_id = bundle.model.condition_id
    results: list[P0CProbeResult] = []
    for probe in probes:
        target = probe.category in {"paraphrase", "compositional"}
        if arm is Arm.EXTERNAL:
            correct = True
        elif arm is Arm.PARAMETRIC:
            correct = not target or condition_id in {"full", "middle"}
        elif arm in {Arm.NO_WRITE, Arm.ROLLBACK}:
            correct = not target
        else:
            raise AssertionError(arm)
        predicted = (
            probe.expected_action
            if correct
            else next(action for action in ACTIONS if action != probe.expected_action)
        )
        results.append(
            P0CProbeResult(
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
                input_tokens=10,
                generated_tokens=1,
                latency_seconds=0.1,
                prompt_sha256=f"prompt-{probe.probe_id}",
                adapter_sha256=kwargs.get("adapter_sha256"),
            )
        )
    return results
