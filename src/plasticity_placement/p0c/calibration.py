from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.analysis import aggregate_experiment
from plasticity_placement.p0c.config import P0CConfig
from plasticity_placement.p0c.domain import Arm, Tier
from plasticity_placement.p0c.modeling import resolve_model_revision
from plasticity_placement.p0c.runtime import (
    _write_environment,
    current_code_hash,
    prepare_experiment,
    run_experiment,
)


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    output_dir: Path
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_revision: str | None = None
    use_4bit: bool = False
    max_length: int = 256
    max_new_tokens: int = 8
    seed: int = 42
    bootstrap_samples: int = 1_000


@dataclass(frozen=True, slots=True)
class Candidate:
    rank: int
    learning_rate: float
    max_steps: int

    @property
    def alpha(self) -> int:
        return self.rank * 2

    @property
    def candidate_id(self) -> str:
        learning_rate = f"{self.learning_rate:.0e}".replace("+", "")
        return f"r{self.rank}-lr{learning_rate}-s{self.max_steps}"

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
        }


def calibration_candidates() -> tuple[Candidate, ...]:
    return tuple(
        Candidate(rank=rank, learning_rate=learning_rate, max_steps=max_steps)
        for rank in (4, 8, 16)
        for learning_rate in (5e-5, 2e-4)
        for max_steps in (4, 16, 64)
    )


def _progress(message: str) -> None:
    print(f"[calibration] {message}", flush=True)


def run_calibration(config: CalibrationConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.output_dir / "calibration_report.json"
    requested_config = _calibration_config_payload(config)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _validate_terminal_report(report, requested_config, report_path)
        _progress(f"reusing terminal report {report_path}")
        return report_path

    compiler_hashes = prepare_experiment(config.output_dir)
    requested_revision = config.model_revision
    resolved_revision = resolve_model_revision(config.model_name, requested_revision)
    runtime_config = replace(config, model_revision=resolved_revision)
    _write_environment(config.output_dir)
    _freeze_calibration_request(
        config.output_dir,
        {
            "calibration_config": requested_config,
            "resolved_model_revision": resolved_revision,
            "code_sha256": current_code_hash(),
            "compiler_hashes": compiler_hashes,
        },
    )

    stage1_results: list[dict[str, Any]] = []
    candidates = calibration_candidates()
    _progress(f"stage1 starting candidates={len(candidates)} adapters={len(candidates) * 2}")
    for index, candidate in enumerate(candidates, start=1):
        stage_dir = config.output_dir / "candidates" / candidate.candidate_id / "stage1"
        _progress(f"stage1 {index}/{len(candidates)} start {candidate.candidate_id}")
        try:
            summary = _run_candidate(
                config=runtime_config,
                candidate=candidate,
                output_dir=stage_dir,
                lesson_ids=("D_01", "D_04"),
            )
        except (RuntimeError, ValueError, OSError) as error:
            stage1_results.append(
                {
                    **candidate.to_dict(),
                    "status": "failed",
                    "error": _error_record(error),
                }
            )
            _progress(
                f"stage1 {index}/{len(candidates)} failed {candidate.candidate_id}: "
                f"{type(error).__name__}: {error}"
            )
            continue
        stage1_results.append(
            {
                **candidate.to_dict(),
                "status": "completed",
                "stage1": summary,
                "stage1_score": _calibration_score(summary),
            }
        )
        _progress(
            f"stage1 {index}/{len(candidates)} completed {candidate.candidate_id} "
            f"score={_calibration_score(summary):.4f}"
        )

    survivors = sorted(
        [row for row in stage1_results if row["status"] == "completed"],
        key=_stage1_sort_key,
    )[:6]
    _progress(f"stage1 survivors: {', '.join(str(row['candidate_id']) for row in survivors)}")
    final_results: list[dict[str, Any]] = []
    _progress(f"stage2 starting candidates={len(survivors)} adapters={len(survivors) * 4}")
    for index, survivor in enumerate(survivors, start=1):
        candidate = Candidate(
            rank=int(survivor["rank"]),
            learning_rate=float(survivor["learning_rate"]),
            max_steps=int(survivor["max_steps"]),
        )
        stage_dir = config.output_dir / "candidates" / candidate.candidate_id / "stage2"
        _progress(f"stage2 {index}/{len(survivors)} start {candidate.candidate_id}")
        try:
            stage2 = _run_candidate(
                config=runtime_config,
                candidate=candidate,
                output_dir=stage_dir,
                lesson_ids=("D_02", "D_03", "D_05", "D_06"),
            )
        except (RuntimeError, ValueError, OSError) as error:
            final_results.append(
                {
                    **candidate.to_dict(),
                    "status": "failed",
                    "stage1": survivor["stage1"],
                    "stage2_error": _error_record(error),
                    "qualified": False,
                }
            )
            _progress(
                f"stage2 {index}/{len(survivors)} failed {candidate.candidate_id}: "
                f"{type(error).__name__}: {error}"
            )
            continue
        combined = _combine_summaries(survivor["stage1"], stage2, 2, 4)
        final_results.append(
            {
                **candidate.to_dict(),
                "status": "completed",
                "stage1": survivor["stage1"],
                "stage2": stage2,
                "combined": combined,
                "qualified": _qualified(combined),
            }
        )
        _progress(
            f"stage2 {index}/{len(survivors)} completed {candidate.candidate_id} "
            f"qualified={_qualified(combined)}"
        )

    qualified = [row for row in final_results if row["status"] == "completed" and row["qualified"]]
    qualified.sort(
        key=lambda row: (
            int(row["max_steps"]),
            int(row["rank"]),
            float(row["combined"]["p_write_time"]),
            -float(row["combined"]["p_target"]),
        )
    )
    selected = (
        {
            key: qualified[0][key]
            for key in (
                "candidate_id",
                "rank",
                "alpha",
                "learning_rate",
                "max_steps",
            )
        }
        if qualified
        else None
    )
    if selected is not None:
        selected["model_revision"] = qualified[0]["combined"]["resolved_model_revision"]
    report = {
        "calibration_config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "model_revision": requested_revision,
            "resolved_model_revision": resolved_revision,
        },
        "candidate_count": len(candidates),
        "stage1_adapter_count": len(candidates) * 2,
        "stage2_adapter_count": len(survivors) * 4,
        "survivor_ids": [str(row["candidate_id"]) for row in survivors],
        "stage1_results": stage1_results,
        "final_results": final_results,
        "selected_config": selected,
        "provenance": {
            "compiler_hashes": compiler_hashes,
            "code_sha256": current_code_hash(),
            "resolved_model_revision": resolved_revision,
        },
    }
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(report_path)
    _progress(f"completed report={report_path} selected={selected}")
    return report_path


def _calibration_config_payload(config: CalibrationConfig) -> dict[str, Any]:
    return {
        **asdict(config),
        "output_dir": str(config.output_dir),
    }


def _validate_terminal_report(
    report: dict[str, Any],
    requested_config: dict[str, Any],
    report_path: Path,
) -> None:
    observed = report.get("calibration_config")
    if not isinstance(observed, dict):
        raise ValueError(f"terminal calibration report is missing config: {report_path}")
    mismatches = {
        key: (observed.get(key), value)
        for key, value in requested_config.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise ValueError(f"terminal calibration report config mismatch: {mismatches}")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"terminal calibration report is missing provenance: {report_path}")
    if provenance.get("code_sha256") != current_code_hash():
        raise ValueError("terminal calibration report was produced by different experiment code")


def _freeze_calibration_request(output_dir: Path, request: dict[str, Any]) -> None:
    path = output_dir / "calibration_request.json"
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != request:
            raise ValueError(
                "existing calibration request differs from the current request; "
                f"use a new output directory: {path}"
            )
        return
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_candidate(
    *,
    config: CalibrationConfig,
    candidate: Candidate,
    output_dir: Path,
    lesson_ids: tuple[str, ...],
) -> dict[str, Any]:
    p0c_config = P0CConfig(
        output_dir=output_dir,
        tier=Tier.DEVELOPMENT,
        model_name=config.model_name,
        model_revision=config.model_revision,
        use_4bit=config.use_4bit,
        rank=candidate.rank,
        alpha=candidate.alpha,
        learning_rate=candidate.learning_rate,
        max_steps=candidate.max_steps,
        max_length=config.max_length,
        max_new_tokens=config.max_new_tokens,
        training_seeds=(config.seed,),
        lesson_ids=lesson_ids,
        calibration_only=True,
        base_results_cache=config.output_dir / "base_cache",
    )
    run_experiment(p0c_config)
    summary_path = aggregate_experiment(output_dir, config.bootstrap_samples)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return _extract_calibration_metrics(summary)


def _extract_calibration_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    arms = summary["arm_summary"]
    parametric = arms[Arm.PARAMETRIC]
    no_write = arms[Arm.NO_WRITE]
    return {
        "p_target": float(parametric["target_generalization"]),
        "n_target": float(no_write["target_generalization"]),
        "target_gain": (
            float(parametric["target_generalization"]) - float(no_write["target_generalization"])
        ),
        "p_exact": float(parametric["exact_heldout"]),
        "p_interference": float(parametric["interference_regression"]),
        "invalid_increase": (float(parametric["invalid_rate"]) - float(no_write["invalid_rate"])),
        "p_write_time": float(parametric["write_time_seconds"]),
        "p_peak_memory": float(parametric["peak_memory_bytes"]),
        "p_adapter_bytes": float(parametric["adapter_bytes"]),
        "resolved_model_revision": _single_model_revision(summary),
    }


def _calibration_score(metrics: dict[str, Any]) -> float:
    return (
        metrics["target_gain"]
        - max(metrics["p_interference"], 0.0)
        - max(metrics["invalid_increase"], 0.0)
    )


def _stage1_sort_key(
    row: dict[str, Any],
) -> tuple[float, int, int, float, str]:
    return (
        -float(row["stage1_score"]),
        int(row["max_steps"]),
        int(row["rank"]),
        float(row["stage1"]["p_write_time"]),
        str(row["candidate_id"]),
    )


def _combine_summaries(
    first: dict[str, Any],
    second: dict[str, Any],
    first_weight: int,
    second_weight: int,
) -> dict[str, Any]:
    if first["resolved_model_revision"] != second["resolved_model_revision"]:
        raise RuntimeError("model revision changed between calibration stages")
    denominator = first_weight + second_weight
    combined = {
        key: (float(first[key]) * first_weight + float(second[key]) * second_weight) / denominator
        for key in first
        if key != "resolved_model_revision"
    }
    combined["resolved_model_revision"] = first["resolved_model_revision"]
    return combined


def _qualified(metrics: dict[str, Any]) -> bool:
    return (
        metrics["target_gain"] >= 0.40
        and metrics["p_exact"] >= 0.75
        and metrics["p_interference"] <= 0.05
        and metrics["invalid_increase"] <= 0.05
    )


def _single_model_revision(summary: dict[str, Any]) -> str | None:
    revisions = summary.get("model_revisions", [])
    if len(revisions) > 1:
        raise RuntimeError(f"multiple model revisions in one calibration run: {revisions}")
    return str(revisions[0]) if revisions else None


def _error_record(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": str(error),
    }
