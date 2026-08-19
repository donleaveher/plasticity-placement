from __future__ import annotations

import argparse
import json
from pathlib import Path

from plasticity_placement.p0c.runtime import (
    current_environment_fingerprint,
    current_environment_snapshot,
)
from plasticity_placement.p0d2hfc.analysis import aggregate_experiment
from plasticity_placement.p0d2hfc.config import P0D2HFCRequest
from plasticity_placement.p0d2hfc.runtime import (
    audit_request,
    resolve_request,
    run_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 P0-D2H-CAL-FC base-only forced-choice calibration",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("environment", help="输出当前环境与 fingerprint")
    for command in ("plan", "audit", "run"):
        command_parser = subparsers.add_parser(
            command,
            help={
                "plan": "解析冻结的 2,304-row source matrix",
                "audit": "CPU 审计 prompt 与四候选 token 边界",
                "run": "GPU 运行 full-string forced-choice scoring",
            }[command],
        )
        _add_request_arguments(command_parser)
    aggregate = subparsers.add_parser(
        "aggregate",
        help="聚合 verified forced-choice rows 并应用冻结 gates",
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
        help="完整 verified P0-D2H-CAL manifest",
    )


def _request_from_args(args: argparse.Namespace) -> P0D2HFCRequest:
    return P0D2HFCRequest(
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
        config, _, _, _ = resolve_request(request)
        print(
            json.dumps(
                {
                    "config": config.identity_dict(),
                    "expected_unit_count": config.expected_unit_count,
                    "expected_decision_row_count": (
                        config.expected_decision_row_count
                    ),
                    "candidate_count": (
                        config.expected_decision_row_count * 4
                    ),
                    "base_only": True,
                    "adapter_discovery_requested": False,
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
