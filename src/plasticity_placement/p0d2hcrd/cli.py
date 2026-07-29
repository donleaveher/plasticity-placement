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
    return parser


def _add_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
        help="exact completed P0-D2H-CAL-FC manifest",
    )


def _request_from_args(args: argparse.Namespace) -> P0D2HCRDRequest:
    return P0D2HCRDRequest(
        output_dir=args.output,
        source_manifest=args.source_manifest,
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
