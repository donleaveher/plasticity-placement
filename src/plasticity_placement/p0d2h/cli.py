from __future__ import annotations

import argparse
import json
from pathlib import Path

from plasticity_placement.p0c.runtime import (
    current_environment_fingerprint,
    current_environment_snapshot,
)
from plasticity_placement.p0d2h.analysis import aggregate_experiment
from plasticity_placement.p0d2h.config import P0D2HRequest
from plasticity_placement.p0d2h.probes import (
    HARD_CATEGORIES,
    PROBES_PER_LESSON,
)
from plasticity_placement.p0d2h.runtime import resolve_request, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 P0-D2H 只读困难 probe 压力测试"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("environment", help="输出当前环境与 fingerprint")
    for command in ("plan", "run"):
        command_parser = subparsers.add_parser(
            command,
            help=f"{'解析' if command == 'plan' else '运行'} P0-D2H 冻结矩阵",
        )
        command_parser.add_argument("--output", type=Path, required=True)
        command_parser.add_argument(
            "--source-manifest",
            type=Path,
            required=True,
            help="完整 verified P0-D2 budget-match manifest",
        )
        command_parser.add_argument(
            "--resilience-margin",
            type=float,
            default=0.05,
        )
    aggregate = subparsers.add_parser(
        "aggregate",
        help="聚合 verified P0-D2H 结果",
    )
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

    request = P0D2HRequest(
        output_dir=args.output,
        source_manifest=args.source_manifest,
        resilience_margin=args.resilience_margin,
    )
    if args.command == "plan":
        config, _, _ = resolve_request(request)
        print(
            json.dumps(
                {
                    "config": config.identity_dict(),
                    "hard_categories": list(HARD_CATEGORIES),
                    "probes_per_lesson": PROBES_PER_LESSON,
                    "expected_unit_count": config.expected_unit_count,
                    "expected_probe_row_count": (
                        config.expected_probe_row_count
                    ),
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
