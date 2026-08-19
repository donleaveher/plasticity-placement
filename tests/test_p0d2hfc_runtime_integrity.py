from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from plasticity_placement.p0d2hc.config import CALIBRATION_ARMS
from plasticity_placement.p0d2hfc.config import (
    ForcedChoiceModel,
    ResolvedP0D2HFCConfig,
)
from plasticity_placement.p0d2hfc.runtime import (
    _load_candidate_token_audit,
    _raw_tree_hash,
    _source_row_index,
)


def _write_raw_matrix(root: Path) -> tuple[tuple[SimpleNamespace, ...], tuple[str, ...]]:
    models = (
        SimpleNamespace(model_id="source_model"),
        SimpleNamespace(model_id="scale_canary"),
    )
    lessons = tuple(f"L{index:02d}" for index in range(24))
    for model in models:
        for lesson in lessons:
            path = root / "results" / "raw" / model.model_id / f"{lesson}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "calibration_model_id": model.model_id,
                    "lesson_id": lesson,
                    "arm": arm,
                    "probe_id": f"{lesson}-probe-{probe:02d}",
                }
                for arm in CALIBRATION_ARMS
                for probe in range(16)
            ]
            path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in rows
                ),
                encoding="utf-8",
            )
    return models, lessons


def _model(model_id: str) -> ForcedChoiceModel:
    return ForcedChoiceModel(
        model_id=model_id,
        role="source" if model_id == "source_model" else "scale_canary",
        model_name=model_id,
        model_revision=f"{model_id}-revision",
        use_4bit=True,
    )


def _config(tmp_path: Path) -> ResolvedP0D2HFCConfig:
    return ResolvedP0D2HFCConfig(
        output_dir=tmp_path / "out",
        source_output_dir=tmp_path / "source",
        source_p0d2h_output_dir=tmp_path / "hard",
        code_sha256="code",
        source_manifest_sha256="manifest",
        source_run_id="run",
        source_summary_sha256="summary",
        source_prompt_token_audit_sha256="prompt",
        source_raw_results_sha256="raw",
        source_p0d2h_run_id="hard-run",
        source_p0d2h_manifest_sha256="hard-manifest",
        hard_probe_hashes={"hard_probes_sha256": "probes"},
        selected_lesson_ids=tuple(f"L{index:02d}" for index in range(24)),
        models=(_model("source_model"), _model("scale_canary")),
        prompt_renderers={
            "no_write": "plain",
            "external": "external",
            "answer_copy_oracle": "oracle",
        },
    )


def test_source_raw_matrix_rejects_missing_and_duplicate_rows(
    tmp_path: Path,
) -> None:
    models, lessons = _write_raw_matrix(tmp_path)
    indexed = _source_row_index(tmp_path, models, lessons)
    assert len(indexed) == 2_304
    path = (
        tmp_path
        / "results"
        / "raw"
        / "source_model"
        / f"{lessons[0]}.jsonl"
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row count mismatch"):
        _source_row_index(tmp_path, models, lessons)

    path.write_text("\n".join([*lines[:-1], lines[0]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate source row key"):
        _source_row_index(tmp_path, models, lessons)


def test_source_raw_tree_hash_detects_post_freeze_mutation(
    tmp_path: Path,
) -> None:
    _, lessons = _write_raw_matrix(tmp_path)
    frozen = _raw_tree_hash(tmp_path)
    path = (
        tmp_path
        / "results"
        / "raw"
        / "scale_canary"
        / f"{lessons[-1]}.jsonl"
    )
    path.write_text(
        path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    assert _raw_tree_hash(tmp_path) != frozen


def test_candidate_audit_rejects_source_hash_mismatch(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    path = config.output_dir / "preflight" / "candidate_token_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "p0d2hfc-candidate-token-audit-v1",
                "candidate_scoring_version": (
                    "p0d2hfc-full-string-sum-logprob-v1"
                ),
                "config_identity_sha256": "wrong",
                "source_manifest_sha256": "changed",
                "source_raw_results_sha256": "raw",
                "decision_count": 2_304,
                "candidate_count": 9_216,
                "failed_decision_count": 0,
                "input_truncated_count": 0,
                "all_checks_passed": True,
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete or mismatched"):
        _load_candidate_token_audit(config)


def test_package_has_no_adapter_discovery_or_training_path() -> None:
    package = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "plasticity_placement"
        / "p0d2hfc"
    )
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(package.glob("*.py"))
    )
    for forbidden in (
        "load_adapter_model",
        "activate_adapter(",
        "PeftModel",
        "train_lora",
        "run_narrow_scan",
    ):
        assert forbidden not in source
