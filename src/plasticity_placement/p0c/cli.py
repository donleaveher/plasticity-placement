from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from plasticity_placement.p0c.analysis import aggregate_experiment
from plasticity_placement.p0c.calibration import CalibrationConfig, run_calibration
from plasticity_placement.p0c.config import P0CConfig
from plasticity_placement.p0c.domain import Tier
from plasticity_placement.p0c.runtime import (
    current_code_hash,
    prepare_experiment,
    run_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 P0-C 真实四载体 Colab 实验")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="编译 frozen lesson/probe artifacts")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--overwrite", action="store_true")

    run = subparsers.add_parser("run", help="运行 smoke、pilot 或 confirmatory tier")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--tier", choices=list(Tier), default=Tier.SMOKE)
    run.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    run.add_argument("--model-revision")
    run.add_argument("--use-4bit", action="store_true")
    run.add_argument("--rank", type=int, default=8)
    run.add_argument("--alpha", type=int, default=16)
    run.add_argument("--learning-rate", type=float, default=2e-4)
    run.add_argument("--max-steps", type=int, default=16)
    run.add_argument("--max-length", type=int, default=256)
    run.add_argument("--max-new-tokens", type=int, default=8)
    run.add_argument("--seeds", type=int, nargs="+")
    run.add_argument("--lesson-ids", nargs="+")
    run.add_argument("--calibration-config", type=Path)
    run.add_argument("--overwrite-compiled", action="store_true")

    aggregate = subparsers.add_parser("aggregate", help="聚合已完成的配对结果")
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--bootstrap-samples", type=int, default=10_000)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="运行 18-config development successive halving",
    )
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    calibrate.add_argument("--model-revision")
    calibrate.add_argument("--use-4bit", action="store_true")
    calibrate.add_argument("--max-length", type=int, default=256)
    calibrate.add_argument("--max-new-tokens", type=int, default=8)
    calibrate.add_argument("--seed", type=int, default=42)
    calibrate.add_argument("--bootstrap-samples", type=int, default=1_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        hashes = prepare_experiment(args.output, overwrite=args.overwrite)
        print(f"compiled artifacts: {args.output / 'compiled'}")
        print(f"compiler version: {hashes['compiler_version']}")
        return
    if args.command == "aggregate":
        path = aggregate_experiment(args.output, args.bootstrap_samples)
        print(f"aggregate summary: {path}")
        return
    if args.command == "calibrate":
        path = run_calibration(
            CalibrationConfig(
                output_dir=args.output,
                model_name=args.model,
                model_revision=args.model_revision,
                use_4bit=args.use_4bit,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                bootstrap_samples=args.bootstrap_samples,
            )
        )
        report = json.loads(path.read_text(encoding="utf-8"))
        print(f"calibration report: {path}")
        print(f"selected config: {report['selected_config']}")
        return

    tier = Tier(args.tier)
    calibrated: dict[str, object] = {}
    if args.calibration_config:
        report = json.loads(args.calibration_config.read_text(encoding="utf-8"))
        _validate_calibration_report(report, args)
        calibrated = report.get("selected_config") or {}
        if not calibrated:
            raise SystemExit("calibration report does not contain a selected_config")
    if tier in {Tier.PILOT, Tier.CONFIRMATORY} and not calibrated:
        raise SystemExit(f"{tier.value} tier requires --calibration-config")
    default_seeds = (41, 42, 43) if tier is Tier.CONFIRMATORY else (42,)
    selected_revision = calibrated.get("model_revision", args.model_revision)
    config = P0CConfig(
        output_dir=args.output,
        tier=tier,
        model_name=args.model,
        model_revision=(str(selected_revision) if selected_revision is not None else None),
        use_4bit=args.use_4bit,
        rank=int(calibrated.get("rank", args.rank)),
        alpha=int(calibrated.get("alpha", args.alpha)),
        learning_rate=float(calibrated.get("learning_rate", args.learning_rate)),
        max_steps=int(calibrated.get("max_steps", args.max_steps)),
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        training_seeds=tuple(args.seeds or default_seeds),
        lesson_ids=tuple(args.lesson_ids or ()),
        overwrite_compiled=args.overwrite_compiled,
        calibration_report_sha256=(
            sha256(args.calibration_config.read_bytes()).hexdigest()
            if args.calibration_config
            else None
        ),
    )
    manifest_path = run_experiment(config)
    print(f"run manifest: {manifest_path}")


def _validate_calibration_report(
    report: dict[str, object],
    args: argparse.Namespace,
) -> None:
    calibration = report.get("calibration_config")
    if not isinstance(calibration, dict):
        raise SystemExit("calibration report is missing calibration_config")
    expected = {
        "model_name": args.model,
        "model_revision": args.model_revision,
        "use_4bit": args.use_4bit,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
    }
    mismatches = {
        key: (calibration.get(key), value)
        for key, value in expected.items()
        if calibration.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"calibration/run configuration mismatch: {mismatches}")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise SystemExit("calibration report is missing provenance")
    selected = report.get("selected_config")
    if isinstance(selected, dict) and selected.get("model_revision") != provenance.get(
        "resolved_model_revision"
    ):
        raise SystemExit("calibration selected model revision does not match provenance")
    if provenance.get("code_sha256") != current_code_hash():
        raise SystemExit("calibration report was produced by different experiment code")
    output = getattr(args, "output", None)
    if output is not None:
        observed_hashes = prepare_experiment(output)
        if provenance.get("compiler_hashes") != observed_hashes:
            raise SystemExit("calibration report compiler artifacts do not match this run")


if __name__ == "__main__":
    main()
