from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import plasticity_placement.p0d2hc.runtime as calibration_runtime
from plasticity_placement.p0c.compiler import (
    ACTIONS,
    compile_bank,
    write_compiled_bank,
)
from plasticity_placement.p0c.domain import Arm
from plasticity_placement.p0c.modeling import P0CProbeResult
from plasticity_placement.p0d2h.probes import write_hard_probe_bank
from plasticity_placement.p0d2hc.analysis import aggregate_experiment
from plasticity_placement.p0d2hc.config import P0D2HCRequest
from plasticity_placement.p0d2hc.invalid_audit import (
    audit_invalid_outputs,
)
from plasticity_placement.p0d2hc.prompting import render_calibration_probe
from plasticity_placement.p0d2hc.runtime import run_experiment


def test_fake_calibration_run_resumes_and_classifies_scale_bottleneck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_manifest = _write_verified_stress_source(tmp_path)
    calls = _patch_backend(monkeypatch)
    output_dir = tmp_path / "calibration"
    request = P0D2HCRequest(
        output_dir=output_dir,
        source_manifest=source_manifest,
        canary_model_name="model-canary",
        canary_model_revision="revision-canary",
        use_4bit=False,
    )
    manifest_path = run_experiment(request)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["models"]) == 2
    assert len(manifest["units"]) == 48
    assert {unit["state"] for unit in manifest["units"].values()} == {
        "verified"
    }
    assert calls == {"loads": 2, "releases": 2, "evaluations": 144}
    assert len(list(output_dir.glob("results/raw/*/*.jsonl"))) == 48

    result_mtimes = {
        path: path.stat().st_mtime_ns
        for path in output_dir.glob("results/raw/*/*.jsonl")
    }
    run_experiment(request)
    assert calls == {"loads": 2, "releases": 2, "evaluations": 144}
    assert {
        path: path.stat().st_mtime_ns for path in result_mtimes
    } == result_mtimes

    summary_path = aggregate_experiment(output_dir, bootstrap_samples=20)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["unit_count"] == 48
    assert summary["probe_row_count"] == 2_304
    assert summary["gates"]["run_valid"] is True
    assert (
        summary["model_gates"]["source_model"]["status"]
        == "oracle_pass_external_failed"
    )
    assert (
        summary["model_gates"]["scale_canary"]["status"]
        == "calibrated"
    )
    assert (
        summary["cross_scale_gate"]["status"]
        == "scale_bottleneck_supported"
    )
    assert summary["cross_scale_gate"]["eligible_model_ids"] == [
        "scale_canary"
    ]
    assert (
        summary["cross_scale_gate"]["next_stage_status"]
        == "training_complexity_design_eligible"
    )
    assert summary["gates"]["automatic_training_started"] is False

    source_mtimes = {
        path: path.stat().st_mtime_ns
        for path in (
            manifest_path,
            *output_dir.glob("results/raw/*/*.jsonl"),
        )
    }
    invalid_audit_path = audit_invalid_outputs(
        output_dir,
        tmp_path / "invalid-audit",
        bootstrap_samples=20,
    )
    invalid_audit = json.loads(
        invalid_audit_path.read_text(encoding="utf-8")
    )
    assert invalid_audit["row_count"] == 2_304
    assert invalid_audit["invalid_record_count"] == 0
    assert invalid_audit["strict_gate_status_changed"] is False
    assert (
        tmp_path / "invalid-audit" / "invalid_output_audit.md"
    ).exists()
    assert (
        tmp_path / "invalid-audit" / "invalid_output_records.jsonl"
    ).read_text(encoding="utf-8") == ""
    assert {
        path: path.stat().st_mtime_ns for path in source_mtimes
    } == source_mtimes
    assert (
        audit_invalid_outputs(
            output_dir,
            tmp_path / "invalid-audit",
            bootstrap_samples=20,
        )
        == invalid_audit_path
    )

    first_result = next(output_dir.glob("results/raw/*/*.jsonl"))
    original = first_result.read_bytes()
    first_result.write_bytes(original.splitlines(keepends=True)[0])
    with pytest.raises(ValueError, match="result matrix"):
        aggregate_experiment(output_dir, bootstrap_samples=10)


def _write_verified_stress_source(tmp_path: Path) -> Path:
    p0d2_dir = tmp_path / "p0d2"
    compiler_hashes = write_compiled_bank(p0d2_dir)
    p0d2_manifest = p0d2_dir / "manifest.json"
    p0d2_manifest.write_text(
        json.dumps({"run_id": "p0d2-source"}),
        encoding="utf-8",
    )
    selected = tuple(
        item
        for item in compile_bank()
        if item.lesson.split == "confirmatory"
    )
    assert len(selected) == 24

    source_dir = tmp_path / "hard"
    hard_hashes, _ = write_hard_probe_bank(source_dir, selected)
    source_audit = {
        "all_prompts_fit": True,
        "input_truncated_count": 0,
        "output_instruction_missing_count": 0,
    }
    source_audit_bytes = (
        json.dumps(
            source_audit,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    source_audit_path = source_dir / "preflight" / "prompt_token_audit.json"
    source_audit_path.parent.mkdir(parents=True, exist_ok=True)
    source_audit_path.write_bytes(source_audit_bytes)

    lesson_ids = [item.lesson.lesson_id for item in selected]
    source = {
        "schema_version": "p0d2h-manifest-v1",
        "run_id": "hard-source",
        "config": {
            "schema_version": "p0d2h-config-v2",
            "stage": "hard_probe",
            "source_manifest_sha256": sha256(
                p0d2_manifest.read_bytes()
            ).hexdigest(),
            "source_run_id": "p0d2-source",
            "source_compiler_hashes": compiler_hashes,
            "hard_probe_hashes": hard_hashes,
            "model_name": "model-source",
            "model_revision": "revision-source",
            "max_new_tokens": 8,
        },
        "source_manifest_path": str(p0d2_manifest),
        "prompt_token_audit_sha256": sha256(
            source_audit_bytes
        ).hexdigest(),
        "selected_lessons": lesson_ids,
        "units": {
            f"unit-{index:03d}": {"state": "verified"}
            for index in range(288)
        },
        "base_arms": {
            lesson_id: {"state": "verified"} for lesson_id in lesson_ids
        },
        "errors": [],
    }
    source_manifest = source_dir / "manifest.json"
    source_manifest.write_text(
        json.dumps(source, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    aggregate_dir = source_dir / "results" / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    (aggregate_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "p0d2h-summary-v2",
                "run_id": "hard-source",
                "adapter_unit_count": 288,
                "probe_row_count": 5_376,
                "gates": {"run_valid": True},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return source_manifest


def _patch_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    calls = {"loads": 0, "releases": 0, "evaluations": 0}

    def load_base(config):
        calls["loads"] += 1
        return SimpleNamespace(
            model=SimpleNamespace(),
            precision="float16",
            model_name=config.model_name,
        )

    def release(bundle) -> None:
        calls["releases"] += 1

    def evaluate(**kwargs: Any) -> list[P0CProbeResult]:
        calls["evaluations"] += 1
        arm = Arm(kwargs["arm"])
        model_name = kwargs["config"].model_name
        correct = (
            arm is Arm.ANSWER_COPY_ORACLE
            or (
                arm is Arm.EXTERNAL
                and model_name == "model-canary"
            )
        )
        return [
            _probe_result(kwargs, probe, arm, correct)
            for probe in kwargs["probes"]
        ]

    monkeypatch.setattr(
        calibration_runtime,
        "resolve_model_revision",
        lambda model_name, revision: revision,
    )
    monkeypatch.setattr(
        calibration_runtime,
        "current_code_hash",
        lambda: "calibration-code",
    )
    monkeypatch.setattr(
        calibration_runtime,
        "_write_environment",
        lambda output: None,
    )
    monkeypatch.setattr(
        calibration_runtime,
        "_require_cuda_runtime",
        lambda: None,
    )
    monkeypatch.setattr(
        calibration_runtime,
        "_require_cuda_bundle",
        lambda bundle: None,
    )
    monkeypatch.setattr(calibration_runtime, "load_base_model", load_base)
    monkeypatch.setattr(calibration_runtime, "release_model", release)
    monkeypatch.setattr(calibration_runtime, "evaluate_probes", evaluate)
    monkeypatch.setattr(
        calibration_runtime,
        "_prepare_prompt_token_audit",
        _fake_prompt_token_audit,
    )
    return calls


def _fake_prompt_token_audit(config, hard_bank, selected):
    audits = {}
    records = []
    for model in config.models:
        for item in selected:
            lesson_id = item.lesson.lesson_id
            for arm in (
                Arm.NO_WRITE,
                Arm.EXTERNAL,
                Arm.ANSWER_COPY_ORACLE,
            ):
                for probe in hard_bank[lesson_id]:
                    rendered, variant, renderer = render_calibration_probe(
                        probe,
                        arm=arm,
                        external_note=item.external_note,
                    )
                    record = {
                        "model_id": model.model_id,
                        "model_name": model.model_name,
                        "model_revision": model.model_revision,
                        "lesson_id": lesson_id,
                        "probe_id": probe.probe_id,
                        "category": probe.category,
                        "prompt_variant": variant,
                        "prompt_rendering_version": renderer,
                        "prompt_sha256": sha256(
                            rendered.prompt.encode()
                        ).hexdigest(),
                        "untruncated_input_tokens": 160,
                        "retained_input_tokens": 160,
                        "truncated_token_count": 0,
                        "input_truncated": False,
                        "output_instruction_preserved": True,
                        "evaluation_max_length": (
                            config.evaluation_max_length
                        ),
                    }
                    audits[
                        (
                            model.model_id,
                            lesson_id,
                            probe.probe_id,
                            variant,
                        )
                    ] = record
                    records.append(record)
    report = {
        "schema_version": "p0d2hc-prompt-token-audit-v1",
        "evaluation_max_length": config.evaluation_max_length,
        "model_count": len(config.models),
        "prompt_count": len(records),
        "input_truncated_count": 0,
        "output_instruction_missing_count": 0,
        "all_prompts_fit": True,
        "max_untruncated_input_tokens_by_model": {
            model.model_id: 160 for model in config.models
        },
        "records": records,
    }
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    path = config.output_dir / "preflight" / "prompt_token_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path, sha256(payload).hexdigest(), audits, report


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
            action for action in ACTIONS if action != probe.expected_action
        )
    )
    return P0CProbeResult(
        run_id=kwargs["run_id"],
        model=kwargs["config"].model_name,
        model_revision=kwargs["config"].model_revision,
        precision=kwargs["bundle"].precision,
        lesson_id=probe.lesson_id,
        pair_id=probe.pair_id,
        lesson_type=probe.lesson_type,
        training_seed=None,
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
        adapter_sha256=None,
    )
