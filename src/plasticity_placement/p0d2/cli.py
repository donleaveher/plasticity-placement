from __future__ import annotations

import argparse
import json
from pathlib import Path

from plasticity_placement.p0c.runtime import (
    current_environment_fingerprint,
    current_environment_snapshot,
)
from plasticity_placement.p0d2.analysis import aggregate_experiment
from plasticity_placement.p0d2.config import P0D2Request
from plasticity_placement.p0d2.runtime import resolve_request, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 P0-D2 LoRA 参数预算匹配实验"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("environment", help="输出当前环境与 fingerprint")

    for command in ("plan", "run"):
        stage_parser = subparsers.add_parser(
            command,
            help=f"{'解析' if command == 'plan' else '运行'} P0-D2 冻结矩阵",
        )
        stage_parser.add_argument("--output", type=Path, required=True)
        stage_parser.add_argument("--source-manifest", type=Path, required=True)
        stage_parser.add_argument("--retention-margin", type=float, default=0.05)
        stage_parser.add_argument("--budget-tolerance", type=float, default=0.01)

    aggregate = subparsers.add_parser("aggregate", help="聚合 verified P0-D2 结果")
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser


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

    request = P0D2Request(
        output_dir=args.output,
        source_manifest=args.source_manifest,
        retention_margin=args.retention_margin,
        budget_tolerance=args.budget_tolerance,
    )
    if args.command == "plan":
        config, compiler_hashes, _ = resolve_request(request)
        print(
            json.dumps(
                {
                    "config": config.identity_dict(),
                    "compiler_hashes": compiler_hashes,
                    "expected_unit_count": config.expected_unit_count,
                    "expected_probe_row_count": config.expected_probe_row_count,
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
