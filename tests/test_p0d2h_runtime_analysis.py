from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
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
from plasticity_placement.p0d2h.analysis import (
    _suite_quality,
    aggregate_experiment,
)
from plasticity_placement.p0d2h.config import P0D2HRequest
from plasticity_placement.p0d2h.prompting import (
    EXTERNAL_PROMPT_VARIANT,
    PLAIN_PROMPT_VARIANT,
    PromptTokenAudit,
    render_evaluation_probe,
)
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
    def __init__(
        self,
        condition_id: str,
        *,
        precision: str = "float32",
    ) -> None:
        self.model = _FakeModel(condition_id)
        self.precision = precision


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
        for path in p0d2_dir.glob("adapters/*/*/seed-*/adapter_model.safetensors")
    }
    assert len(source_adapter_mtimes) == 504

    stress_calls = _patch_stress_backend(monkeypatch)
    output_dir = tmp_path / "hard"
    request = P0D2HRequest(
        output_dir=output_dir,
        source_manifest=p0d2_manifest,
    )
    manifest_path = run_experiment(request)
    assert stress_calls == {
        "base_loads": 1,
        "adapter_activations": 288,
        "adapter_deactivations": 288,
        "releases": 1,
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["selected_lessons"]) == 24
    assert len(manifest["conditions"]) == 4
    assert len(manifest["units"]) == 288
    assert {value["state"] for value in manifest["base_arms"].values()} == {"verified"}
    assert {value["state"] for value in manifest["units"].values()} == {"verified"}
    assert {value["source_training_precision"] for value in manifest["units"].values()} == {
        "float32"
    }
    assert {value["evaluation_precision"] for value in manifest["units"].values()} == {"float16"}
    assert {
        path: path.stat().st_mtime_ns for path in source_adapter_mtimes
    } == source_adapter_mtimes

    result_mtimes = {
        path: path.stat().st_mtime_ns
        for path in output_dir.glob("results/adapters/*/*/seed-*.jsonl")
    }
    assert len(result_mtimes) == 288
    run_experiment(request)
    assert stress_calls == {
        "base_loads": 1,
        "adapter_activations": 288,
        "adapter_deactivations": 288,
        "releases": 1,
    }
    assert {path: path.stat().st_mtime_ns for path in result_mtimes} == result_mtimes

    summary_path = aggregate_experiment(
        output_dir,
        bootstrap_samples=20,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["adapter_unit_count"] == 288
    assert summary["probe_row_count"] == 5_376
    assert summary["condition_summary"]["late-matched"]["hard_accuracy"] == 1.0
    assert summary["condition_summary"]["full-base"]["hard_accuracy"] == 0.0
    assert summary["gates"]["decision_status"] == "hard_locus_robust"
    assert summary["suite_quality"]["status"] == "ready"
    assert summary["gates"]["next_stage_status"] == "eligible_for_reviewed_followup"
    assert summary["prompt_token_audit"]["max_untruncated_input_tokens"] == 160
    assert summary["gates"]["automatic_training_started"] is False

    first_result = next(output_dir.glob("results/adapters/*/*/seed-*.jsonl"))
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


def test_reusable_model_is_released_when_evaluation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        output_dir=tmp_path / "hard",
        expected_unit_count=288,
        identity_dict=lambda: {"schema_version": "test"},
    )
    manifest = SimpleNamespace(path=config.output_dir / "manifest.json")
    bundle = _FakeBundle("base")
    released: list[_FakeBundle] = []

    monkeypatch.setattr(
        stress_runtime,
        "_require_cuda_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        stress_runtime,
        "resolve_request",
        lambda request, *, deep_source_validation: (config, {}, ()),
    )
    monkeypatch.setattr(
        stress_runtime.P0D2HManifest,
        "load_or_create",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(stress_runtime, "_write_environment", lambda output: None)
    monkeypatch.setattr(
        stress_runtime,
        "_prepare_prompt_token_audit",
        lambda *args: (
            config.output_dir / "preflight" / "prompt_token_audit.json",
            "audit",
            {},
            {"all_prompts_fit": True},
        ),
    )
    monkeypatch.setattr(stress_runtime, "_requires_model", lambda *args: True)
    monkeypatch.setattr(stress_runtime, "load_base_model", lambda value: bundle)
    monkeypatch.setattr(stress_runtime, "_require_cuda_bundle", lambda value: None)
    monkeypatch.setattr(
        stress_runtime,
        "_run_base_arms",
        lambda *args: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    monkeypatch.setattr(
        stress_runtime,
        "release_model",
        lambda value: released.append(value),
    )

    with pytest.raises(RuntimeError, match="synthetic failure"):
        run_experiment(
            P0D2HRequest(
                output_dir=config.output_dir,
                source_manifest=tmp_path / "source" / "manifest.json",
            )
        )
    assert released == [bundle]


def test_suite_quality_blocks_external_failure_and_common_floor() -> None:
    categories = (
        "binding_decoys",
        "conflict_stack",
        "conditional_route",
        "long_context",
    )
    condition_summary = {
        condition_id: {
            "hard_accuracy": 0.5 if condition_id == "late-matched" else 0.4,
            "invalid_rate": 0.0,
            "categories": {
                category: {"accuracy": (0.25 if category == "binding_decoys" else 0.6)}
                for category in categories
            },
        }
        for condition_id in (
            "full-base",
            "early-matched",
            "middle-matched",
            "late-matched",
            "no_write",
        )
    }
    condition_summary["external"] = {
        "hard_accuracy": 0.31,
        "invalid_rate": 1.0 / 3.0,
        "categories": {
            category: {"accuracy": 0.0 if category == "long_context" else 0.25}
            for category in categories
        },
    }
    quality = _suite_quality(
        condition_summary,
        [
            {
                "input_truncated": False,
                "output_instruction_preserved": True,
            }
        ],
        {
            "common_floor_tolerance": 0.05,
            "external_anchor_min_accuracy": 0.75,
            "external_anchor_max_invalid_rate": 0.05,
        },
    )
    assert quality["status"] == "suite_repair_required"
    assert quality["common_floor_categories"] == ["binding_decoys"]
    assert quality["checks"]["external_accuracy_at_least_threshold"] is False
    assert quality["checks"]["external_invalid_rate_at_most_threshold"] is False


def _write_p0c_source(source_dir: Path) -> Path:
    hashes = write_compiled_bank(source_dir)
    selected = [item for item in compile_bank() if item.lesson.split == "confirmatory"]
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


def _patch_stress_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    calls = {
        "base_loads": 0,
        "adapter_activations": 0,
        "adapter_deactivations": 0,
        "releases": 0,
    }

    def load_base(config) -> _FakeBundle:
        calls["base_loads"] += 1
        return _FakeBundle("base", precision="float16")

    def activate(bundle, path, *, adapter_name) -> None:
        assert adapter_name == stress_runtime.ACTIVE_ADAPTER_NAME
        assert bundle.model.condition_id == "base"
        calls["adapter_activations"] += 1
        bundle.model.condition_id = path.parts[-3]

    def deactivate(bundle, *, adapter_name) -> None:
        assert adapter_name == stress_runtime.ACTIVE_ADAPTER_NAME
        assert bundle.model.condition_id != "base"
        calls["adapter_deactivations"] += 1
        bundle.model.condition_id = "base"

    def release(bundle) -> None:
        assert bundle.model.condition_id == "base"
        calls["releases"] += 1

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
        "_require_cuda_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        stress_runtime,
        "_require_cuda_bundle",
        lambda bundle: None,
    )
    monkeypatch.setattr(
        stress_runtime,
        "load_base_model",
        load_base,
    )
    monkeypatch.setattr(
        stress_runtime,
        "activate_adapter",
        activate,
    )
    monkeypatch.setattr(
        stress_runtime,
        "deactivate_adapter",
        deactivate,
    )
    monkeypatch.setattr(
        stress_runtime,
        "release_model",
        release,
    )
    monkeypatch.setattr(
        stress_runtime,
        "evaluate_probes",
        _fake_stress_evaluate,
    )
    monkeypatch.setattr(
        stress_runtime,
        "_prepare_prompt_token_audit",
        _fake_prompt_token_audit,
    )
    return calls


def _fake_prompt_token_audit(
    config,
    hard_probe_bank,
    selected,
):
    audits = {}
    records = []
    for item in selected:
        lesson_id = item.lesson.lesson_id
        for external_note in (None, item.external_note):
            for probe in hard_probe_bank[lesson_id]:
                rendered, variant, rendering_version = render_evaluation_probe(
                    probe,
                    external_note=external_note,
                )
                audit = PromptTokenAudit(
                    lesson_id=lesson_id,
                    probe_id=probe.probe_id,
                    category=probe.category,
                    prompt_variant=variant,
                    prompt_rendering_version=rendering_version,
                    prompt_sha256=sha256(rendered.prompt.encode()).hexdigest(),
                    untruncated_input_tokens=160,
                    retained_input_tokens=160,
                    truncated_token_count=0,
                    input_truncated=False,
                    output_instruction_preserved=True,
                    evaluation_max_length=config.evaluation_max_length,
                )
                audits[(lesson_id, probe.probe_id, variant)] = audit
                records.append(audit.to_dict())
    report = {
        "schema_version": "p0d2h-prompt-token-audit-v1",
        "source_training_max_length": config.source_training_max_length,
        "evaluation_max_length": config.evaluation_max_length,
        "external_prompt_renderer_version": (
            config.identity_dict()["external_prompt_renderer_version"]
        ),
        "prompt_count": len(records),
        "input_truncated_count": 0,
        "output_instruction_missing_count": 0,
        "all_prompts_fit": True,
        "max_untruncated_input_tokens": 160,
        "max_untruncated_input_tokens_by_variant": {
            PLAIN_PROMPT_VARIANT: 160,
            EXTERNAL_PROMPT_VARIANT: 160,
        },
        "records": records,
    }
    payload = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    path = config.output_dir / "preflight" / "prompt_token_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path, sha256(payload).hexdigest(), audits, report


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
        training_data_sha256=sha256(config.data_path.read_bytes()).hexdigest(),
        config_sha256=source_runtime._json_hash(config.to_dict()),
        adapter_sha256=adapter_hash,
        precision="float32",
    )
    (config.output_dir / "training_metadata.json").write_text(
        json.dumps({"config": config.to_dict(), "summary": asdict(summary)}),
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
            correct = not target or condition_id in {"full-base", "late-matched"}
        elif arm in {Arm.NO_WRITE, Arm.ROLLBACK}:
            correct = not target
        else:
            raise AssertionError(arm)
        results.append(_probe_result(kwargs, probe, arm, correct))
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
    return [_probe_result(kwargs, probe, arm, correct) for probe in kwargs["probes"]]


def _probe_result(
    kwargs: dict[str, Any],
    probe,
    arm: Arm,
    correct: bool,
) -> P0CProbeResult:
    predicted = (
        probe.expected_action
        if correct
        else next(action for action in ACTIONS if action != probe.expected_action)
    )
    return P0CProbeResult(
        run_id=kwargs["run_id"],
        model=kwargs["config"].model_name,
        model_revision=kwargs["config"].model_revision,
        precision=kwargs["bundle"].precision,
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
