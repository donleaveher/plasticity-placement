from __future__ import annotations

import argparse
import json
from pathlib import Path

from plasticity_placement.p0c.runtime import (
    current_environment_fingerprint,
    current_environment_snapshot,
)
from plasticity_placement.p0d2hcrd.analysis import aggregate_experiment
from plasticity_placement.p0d2hcrd.config import P0D2HCRDRequest
from plasticity_placement.p0d2hcrd.conservative_analysis import (
    aggregate_conservative_ties,
)
from plasticity_placement.p0d2hcrd.integrity_audit import (
    audit_scoring_integrity,
)
from plasticity_placement.p0d2hcrd.precision_audit import (
    audit_fp32_precision,
)
from plasticity_placement.p0d2hcrd.runtime import (
    audit_request,
    resolve_request,
    run_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 P0-D2H-CRD base-only route/retrieval decomposition",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("environment", help="输出当前环境与 fingerprint")
    for command in ("plan", "audit", "run"):
        command_parser = subparsers.add_parser(
            command,
            help={
                "plan": "验证冻结 source 并解析 1,536-row CRD matrix",
                "audit": "CPU 审计拆分 prompt、泄漏与候选 token 边界",
                "run": "GPU 运行 route/retrieval/combined candidate scoring",
            }[command],
        )
        _add_request_arguments(command_parser)
    aggregate = subparsers.add_parser(
        "aggregate",
        help="聚合 verified CRD rows 并应用诊断性 gates",
    )
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--bootstrap-samples", type=int, default=10_000)
    integrity_audit = subparsers.add_parser(
        "audit-integrity",
        help="只读审计 completed CRD 的 exact tie 与评分一致性",
    )
    integrity_audit.add_argument(
        "--output",
        type=Path,
        required=True,
        help="完整 verified P0-D2H-CRD source run",
    )
    integrity_audit.add_argument(
        "--audit-output",
        type=Path,
        required=True,
        help="独立的 supplementary audit 输出目录",
    )
    fp32_audit = subparsers.add_parser(
        "audit-fp32",
        help="用禁用 TF32 的完整 FP32 forward 只读复核 exact ties",
    )
    fp32_audit.add_argument(
        "--source-output",
        type=Path,
        action="append",
        required=True,
        help="completed CRD source run；可重复传入 NF4 与 BF16",
    )
    fp32_audit.add_argument(
        "--audit-output",
        type=Path,
        required=True,
        help="独立的 FP32 supplementary audit 输出目录",
    )
    conservative = subparsers.add_parser(
        "aggregate-v2",
        help="在独立目录生成保守 exact-tie 上下界诊断",
    )
    conservative.add_argument(
        "--output",
        type=Path,
        required=True,
        help="完整 verified P0-D2H-CRD source run",
    )
    conservative.add_argument(
        "--analysis-output",
        type=Path,
        required=True,
        help="独立的 conservative tie analysis 输出目录",
    )
    conservative.add_argument(
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
        help="exact completed P0-D2H-CAL-FC manifest",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="使用原始 BF16 权重；仅用于独立 precision-retry attempt",
    )


def _request_from_args(args: argparse.Namespace) -> P0D2HCRDRequest:
    return P0D2HCRDRequest(
        output_dir=args.output,
        source_manifest=args.source_manifest,
        use_4bit=not args.no_4bit,
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
    if args.command == "audit-integrity":
        path = audit_scoring_integrity(args.output, args.audit_output)
        print(
            json.dumps(
                {"scoring_integrity_audit_path": str(path)},
                ensure_ascii=False,
            )
        )
        return
    if args.command == "audit-fp32":
        path = audit_fp32_precision(
            tuple(args.source_output),
            args.audit_output,
        )
        print(
            json.dumps(
                {"fp32_precision_audit_path": str(path)},
                ensure_ascii=False,
            )
        )
        return
    if args.command == "aggregate-v2":
        path = aggregate_conservative_ties(
            args.output,
            args.analysis_output,
            args.bootstrap_samples,
        )
        print(
            json.dumps(
                {"conservative_tie_summary_path": str(path)},
                ensure_ascii=False,
            )
        )
        return

    request = _request_from_args(args)
    if args.command == "plan":
        config, _, _, _, bank_audit = resolve_request(request)
        print(
            json.dumps(
                {
                    "config": config.identity_dict(),
                    "expected_unit_count": config.expected_unit_count,
                    "expected_decision_row_count": (
                        config.expected_decision_row_count
                    ),
                    "expected_candidate_sequence_count": (
                        config.expected_candidate_sequence_count
                    ),
                    "bank_audit": {
                        key: value
                        for key, value in bank_audit.items()
                        if key != "records"
                    },
                    "base_only": True,
                    "diagnostic_only": True,
                    "adapter_discovery_requested": False,
                    "training_complexity_review_eligible": False,
                    "automatic_training_started": False,
                    "automatic_narrow_scan_started": False,
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
