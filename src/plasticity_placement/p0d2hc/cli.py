from __future__ import annotations

import argparse
import json
from pathlib import Path

from plasticity_placement.p0c.runtime import (
    current_environment_fingerprint,
    current_environment_snapshot,
)
from plasticity_placement.p0d2h.probes import (
    HARD_CATEGORIES,
    PROBES_PER_LESSON,
)
from plasticity_placement.p0d2hc.analysis import aggregate_experiment
from plasticity_placement.p0d2hc.config import (
    CALIBRATION_ARMS,
    P0D2HCRequest,
)
from plasticity_placement.p0d2hc.invalid_audit import (
    audit_invalid_outputs,
)
from plasticity_placement.p0d2hc.runtime import (
    audit_request,
    resolve_request,
    run_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 P0-D2H-CAL base-only oracle calibration",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("environment", help="输出当前环境与 fingerprint")
    for command in ("plan", "audit", "run"):
        command_parser = subparsers.add_parser(
            command,
            help=(
                "审计三臂 prompt token 容量"
                if command == "audit"
                else (
                    "解析冻结 calibration 矩阵"
                    if command == "plan"
                    else "运行 base-only calibration 矩阵"
                )
            ),
        )
        _add_request_arguments(command_parser)
    aggregate = subparsers.add_parser(
        "aggregate",
        help="聚合 verified P0-D2H-CAL 结果",
    )
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--bootstrap-samples", type=int, default=10_000)
    invalid_audit = subparsers.add_parser(
        "audit-invalid",
        help="只读分类 verified P0-D2H-CAL invalid outputs",
    )
    invalid_audit.add_argument(
        "--output",
        type=Path,
        required=True,
        help="完整 verified P0-D2H-CAL source run",
    )
    invalid_audit.add_argument(
        "--audit-output",
        type=Path,
        required=True,
        help="独立的 supplementary audit 输出目录",
    )
    invalid_audit.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
    )
    return parser


def _add_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
        help="完整 verified P0-D2H-R manifest",
    )
    parser.add_argument(
        "--canary-model-name",
        help="可选的同系列更大规模 base-only canary",
    )
    parser.add_argument(
        "--canary-model-revision",
        help="可选 canary revision；运行前解析并冻结为 commit",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="关闭 base model 4-bit 加载",
    )
    parser.add_argument(
        "--evaluation-max-length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--oracle-min-accuracy",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--oracle-min-category-accuracy",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--oracle-max-invalid-rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--external-min-accuracy",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--external-max-invalid-rate",
        type=float,
        default=0.05,
    )


def _request_from_args(args: argparse.Namespace) -> P0D2HCRequest:
    return P0D2HCRequest(
        output_dir=args.output,
        source_manifest=args.source_manifest,
        canary_model_name=args.canary_model_name,
        canary_model_revision=args.canary_model_revision,
        use_4bit=not args.no_4bit,
        evaluation_max_length=args.evaluation_max_length,
        oracle_min_accuracy=args.oracle_min_accuracy,
        oracle_min_category_accuracy=args.oracle_min_category_accuracy,
        oracle_max_invalid_rate=args.oracle_max_invalid_rate,
        external_min_accuracy=args.external_min_accuracy,
        external_max_invalid_rate=args.external_max_invalid_rate,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "environment":
        print(
            json.dumps(
                {
                    "fingerprint": current_environment_fingerprint(),
                    "environment": current_environment_snapshot(),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    if args.command == "aggregate":
        path = aggregate_experiment(args.output, args.bootstrap_samples)
        print(json.dumps({"summary_path": str(path)}, ensure_ascii=False))
        return
    if args.command == "audit-invalid":
        path = audit_invalid_outputs(
            args.output,
            args.audit_output,
            args.bootstrap_samples,
        )
        print(
            json.dumps(
                {"invalid_output_audit_path": str(path)},
                ensure_ascii=False,
            )
        )
        return

    request = _request_from_args(args)
    if args.command == "plan":
        config, _, _ = resolve_request(request)
        print(
            json.dumps(
                {
                    "config": config.identity_dict(),
                    "calibration_arms": list(CALIBRATION_ARMS),
                    "hard_categories": list(HARD_CATEGORIES),
                    "probes_per_lesson": PROBES_PER_LESSON,
                    "expected_unit_count": config.expected_unit_count,
                    "expected_probe_row_count": (
                        config.expected_probe_row_count
                    ),
                    "training_requested": False,
                    "narrow_scan_requested": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    if args.command == "audit":
        path = audit_request(request)
        report = json.loads(path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "audit_path": str(path),
                    "audit": {
                        key: value
                        for key, value in report.items()
                        if key != "records"
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    path = run_experiment(request)
    print(json.dumps({"manifest_path": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
