import json
from pathlib import Path

import pytest

from plasticity_placement.p0c.analysis import aggregate_experiment
from plasticity_placement.p0c.compiler import load_compiled_bank, write_compiled_bank
from plasticity_placement.p0c.domain import CompiledLesson


def test_aggregate_builds_strict_paired_summary(tmp_path: Path) -> None:
    item = _prepare_one_lesson_run(tmp_path)

    summary_path = aggregate_experiment(tmp_path, bootstrap_samples=100)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["arm_summary"]["external"]["target_generalization"] == 1.0
    assert summary["arm_summary"]["no_write"]["target_generalization"] == 0.0
    assert summary["contrasts"]["P-N"]["target_mean"] == 1.0
    assert summary["arm_summary"]["parametric"]["target_generalization_ci95"] == [
        1.0,
        1.0,
    ]
    assert summary["seed_category_metrics"]
    assert summary["seed_stability"]
    assert summary["pair_action_diagnostics"][0]["pair_id"] == item.lesson.pair_id
    assert summary["action_token_diagnostics"]
    assert (tmp_path / "results" / "aggregate" / "main_table.md").exists()


def test_aggregate_rejects_missing_configured_seed(tmp_path: Path) -> None:
    _prepare_one_lesson_run(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["training_seeds"] = [42, 43]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest unit set"):
        aggregate_experiment(tmp_path, bootstrap_samples=10)


def test_aggregate_rejects_duplicate_probe_rows(tmp_path: Path) -> None:
    item = _prepare_one_lesson_run(tmp_path)
    path = tmp_path / "results" / "raw" / item.lesson.lesson_id / "base_arms.jsonl"
    first = path.read_text(encoding="utf-8").splitlines()[0]
    path.write_text(path.read_text(encoding="utf-8") + first + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid result panel"):
        aggregate_experiment(tmp_path, bootstrap_samples=10)


def _prepare_one_lesson_run(tmp_path: Path) -> CompiledLesson:
    hashes = write_compiled_bank(tmp_path)
    item = load_compiled_bank(tmp_path)[0]
    lesson_id = item.lesson.lesson_id
    adapter_hash = "adapter-1"
    training_summary = {
        "elapsed_seconds": 1.0,
        "peak_memory_bytes": 2.0,
        "adapter_bytes": 3.0,
        "training_data_sha256": "data-1",
        "config_sha256": "config-1",
        "adapter_sha256": adapter_hash,
        "precision": "float32",
        "model_revision": "revision-1",
    }
    manifest = {
        "run_id": "test-run",
        "selected_lessons": [lesson_id],
        "compiler_hashes": hashes,
        "config": {
            "tier": "development",
            "training_seeds": [42],
            "model_name": "model-a",
            "model_revision": "revision-1",
            "calibration_only": False,
        },
        "base_arms": {lesson_id: "verified"},
        "units": {
            f"{lesson_id}::seed-42": {
                "lesson_id": lesson_id,
                "seed": 42,
                "state": "verified",
                "adapter_sha256": adapter_hash,
                "rollback_exact_match_rate": 1.0,
                "training_summary": training_summary,
            }
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    lesson_dir = tmp_path / "results" / "raw" / lesson_id
    seed_dir = lesson_dir / "seed-42"
    seed_dir.mkdir(parents=True)

    no_write = _arm_rows(item, "no_write", None, None, target_correct=False)
    external = _arm_rows(item, "external", None, None, target_correct=True)
    parametric = _arm_rows(
        item,
        "parametric",
        42,
        adapter_hash,
        target_correct=True,
    )
    both = _arm_rows(item, "both", 42, adapter_hash, target_correct=True)
    rollback = _arm_rows(
        item,
        "rollback",
        42,
        adapter_hash,
        target_correct=False,
    )
    _write_jsonl(lesson_dir / "base_arms.jsonl", [*no_write, *external])
    _write_jsonl(
        seed_dir / "adapter_arms.jsonl",
        [*parametric, *both, *rollback],
    )
    return item


def _arm_rows(
    item: CompiledLesson,
    arm: str,
    seed: int | None,
    adapter_hash: str | None,
    *,
    target_correct: bool,
) -> list[dict[str, object]]:
    rows = []
    for probe in item.evaluation_probes:
        correct = target_correct if probe.category in {"paraphrase", "compositional"} else True
        rows.append(
            {
                "run_id": "test-run",
                "model": "model-a",
                "model_revision": "revision-1",
                "precision": "float32",
                "lesson_id": item.lesson.lesson_id,
                "pair_id": item.lesson.pair_id,
                "lesson_type": item.lesson.lesson_type,
                "arm": arm,
                "probe_id": probe.probe_id,
                "category": probe.category,
                "training_seed": seed,
                "expected_action": probe.expected_action,
                "predicted_action": probe.expected_action if correct else None,
                "correct": correct,
                "invalid": not correct,
                "generated_text": probe.expected_action if correct else "",
                "input_tokens": 10,
                "generated_tokens": 1,
                "latency_seconds": 0.1,
                "prompt_sha256": f"prompt-{probe.probe_id}",
                "adapter_sha256": adapter_hash,
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
