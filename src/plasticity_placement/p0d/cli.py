from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

from plasticity_placement.p0c.runtime import (
    current_environment_fingerprint,
    current_environment_snapshot,
)
from plasticity_placement.p0d.analysis import aggregate_experiment
from plasticity_placement.p0d.config import (
    LayerCondition,
    P0DRequest,
    P0DStage,
    band_scan_conditions,
)
from plasticity_placement.p0d.runtime import resolve_request, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 P0-D LoRA layer-locus Colab 实验"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("environment", help="输出当前环境与 fingerprint")

    for command in ("plan", "run"):
        stage_parser = subparsers.add_parser(
            command,
            help=f"{'解析' if command == 'plan' else '运行'} P0-D 条件矩阵",
        )
        stage_parser.add_argument("--output", type=Path, required=True)
        stage_parser.add_argument("--source-manifest", type=Path, required=True)
        stage_parser.add_argument(
            "--stage",
            choices=list(P0DStage),
            default=P0DStage.BAND_SCAN,
        )
        stage_parser.add_argument("--conditions-config", type=Path)
        stage_parser.add_argument("--retention-margin", type=float, default=0.05)

    aggregate = subparsers.add_parser("aggregate", help="聚合 verified P0-D 结果")
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

    request = _request_from_args(args)
    if args.command == "plan":
        config, compiler_hashes, _ = resolve_request(request)
        print(
            json.dumps(
                {
                    "config": config.identity_dict(),
                    "compiler_hashes": compiler_hashes,
                    "expected_unit_count": config.expected_unit_count,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    path = run_experiment(request)
    print(json.dumps({"manifest_path": str(path)}, ensure_ascii=False))


def _request_from_args(args: argparse.Namespace) -> P0DRequest:
    stage = P0DStage(args.stage)
    if args.conditions_config is None:
        if stage is P0DStage.NARROW_SCAN:
            raise SystemExit("narrow_scan requires --conditions-config")
        conditions = band_scan_conditions()
        reference = "full"
        conditions_hash = None
        parent_band_run_id = None
    else:
        conditions_bytes = args.conditions_config.read_bytes()
        payload = json.loads(conditions_bytes)
        if not isinstance(payload, dict) or not isinstance(payload.get("conditions"), list):
            raise SystemExit("conditions config must contain a JSON list named 'conditions'")
        if payload.get("stage") not in {None, stage.value}:
            raise SystemExit("conditions config stage does not match --stage")
        conditions = tuple(
            LayerCondition.from_dict(value) for value in payload["conditions"]
        )
        reference = str(payload.get("reference_condition_id", "full"))
        conditions_hash = sha256(conditions_bytes).hexdigest()
        parent_band_run_id = payload.get("source_band_run_id")
    return P0DRequest(
        output_dir=args.output,
        source_manifest=args.source_manifest,
        stage=stage,
        conditions=conditions,
        reference_condition_id=reference,
        retention_margin=args.retention_margin,
        conditions_config_sha256=conditions_hash,
        parent_band_run_id=(
            str(parent_band_run_id) if parent_band_run_id is not None else None
        ),
    )


if __name__ == "__main__":
    main()
