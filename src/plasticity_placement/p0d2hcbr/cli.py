from __future__ import annotations

import argparse
import json
from pathlib import Path

from plasticity_placement.p0c.runtime import (
    current_environment_fingerprint,
    current_environment_snapshot,
)
from plasticity_placement.p0d2hcbr.aggregate import aggregate_experiment, verify_complete_result
from plasticity_placement.p0d2hcbr.config import (
    CURRICULA,
    PLACEMENTS,
    TRAINING_SEEDS,
    ExperimentSpec,
    unit_id,
)
from plasticity_placement.p0d2hcbr.corrected import plan_corrected_experiment
from plasticity_placement.p0d2hcbr.deviation import (
    adopt_budget_deviation,
    deviation_template,
)
from plasticity_placement.p0d2hcbr.evaluation import EvaluationRequest, evaluate_unit
from plasticity_placement.p0d2hcbr.manifest import Manifest
from plasticity_placement.p0d2hcbr.preflight import plan_experiment
from plasticity_placement.p0d2hcbr.runtime import authorize_experiment, train_unit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 counterbalanced binding remediation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("environment", help="只读输出环境 fingerprint")
    plan = subparsers.add_parser("plan", help="验证 RABX、编译 bank 并冻结 preregistration")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--rabx-output", type=Path, required=True)
    plan.add_argument("--source-code-revision-lock", type=Path, required=True)
    plan.add_argument("--spec", type=Path, required=True)
    corrected = subparsers.add_parser(
        "plan-corrected", help="冻结精确参数匹配的 CBR-v2 并导入六个 full-depth controls"
    )
    corrected.add_argument("--output", type=Path, required=True)
    corrected.add_argument("--source-cbr-output", type=Path, required=True)
    corrected.add_argument("--spec", type=Path, required=True)
    authorize = subparsers.add_parser("authorize", help="采纳外部批准文件")
    authorize.add_argument("--output", type=Path, required=True)
    authorize.add_argument("--authorization", type=Path, required=True)
    deviation = subparsers.add_parser(
        "budget-deviation-template",
        help="只读生成现有 CBR-v1 全量描述性评估的偏差授权模板",
    )
    deviation.add_argument("--output", type=Path, required=True)
    adopt_deviation = subparsers.add_parser(
        "authorize-budget-deviation", help="采纳外部预算偏差评估批准文件"
    )
    adopt_deviation.add_argument("--output", type=Path, required=True)
    adopt_deviation.add_argument("--authorization", type=Path, required=True)
    train = subparsers.add_parser("train", help="训练一个指定 unit 或全部 pending units")
    train.add_argument("--output", type=Path, required=True)
    _add_unit_selector(train)
    evaluate = subparsers.add_parser("evaluate", help="评估一个 unit 或全部未评估 units")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--analysis-root", type=Path, required=True)
    _add_unit_selector(evaluate)
    aggregate = subparsers.add_parser("aggregate", help="聚合全部 12 个 verified evaluations")
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--evaluations-root", type=Path, required=True)
    verify = subparsers.add_parser("verify", help="验证完整 CBR 结果")
    verify.add_argument("--output", type=Path, required=True)
    status = subparsers.add_parser("status", help="只读输出 manifest 状态")
    status.add_argument("--output", type=Path, required=True)
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
    if args.command == "plan":
        path = plan_experiment(
            output_dir=args.output,
            rabx_output=args.rabx_output,
            source_code_revision_lock=args.source_code_revision_lock,
            spec_path=args.spec,
        )
    elif args.command == "plan-corrected":
        path = plan_corrected_experiment(
            output_dir=args.output,
            source_cbr_output=args.source_cbr_output,
            spec_path=args.spec,
        )
    elif args.command == "authorize":
        path = authorize_experiment(args.output, args.authorization)
    elif args.command == "budget-deviation-template":
        print(json.dumps(deviation_template(args.output), ensure_ascii=False, sort_keys=True))
        return
    elif args.command == "authorize-budget-deviation":
        path = adopt_budget_deviation(args.output, args.authorization)
    elif args.command == "train":
        paths = [
            train_unit(args.output, curriculum, placement, seed)
            for curriculum, placement, seed in _selected_units(args)
        ]
        path = paths[-1]
    elif args.command == "evaluate":
        paths = [
            evaluate_unit(
                EvaluationRequest(
                    args.output,
                    curriculum,
                    placement,
                    seed,
                    args.analysis_root / unit_id(curriculum, placement, seed),
                )
            )
            for curriculum, placement, seed in _selected_units(args)
        ]
        path = paths[-1]
    elif args.command == "aggregate":
        path = aggregate_experiment(args.output, args.evaluations_root)
    elif args.command == "verify":
        path = verify_complete_result(args.output)
    elif args.command == "status":
        manifest = Manifest.load(args.output / "manifest.json")
        print(
            json.dumps(
                {
                    "manifest_path": str(manifest.path),
                    "run_id": manifest.payload["run_id"],
                    "state": manifest.state,
                    "training_units": {
                        key: value["state"]
                        for key, value in manifest.payload["training_units"].items()
                    },
                    "evaluation_claims": {
                        key: value["state"]
                        for key, value in manifest.payload["evaluation_claims"].items()
                    },
                    "budget_deviation": manifest.payload.get("budget_deviation"),
                    "mappings_per_adapter_authorized": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    else:
        raise AssertionError(args.command)
    print(json.dumps({"output_path": str(path)}, ensure_ascii=False))


def _add_unit_selector(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--curriculum", choices=CURRICULA)
    parser.add_argument("--placement", choices=PLACEMENTS)
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS)


def _selected_units(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    values = (args.curriculum, args.placement, args.seed)
    if any(value is not None for value in values):
        if any(value is None for value in values):
            raise ValueError("set curriculum, placement, and seed together")
        return [(args.curriculum, args.placement, args.seed)]
    spec = ExperimentSpec()
    return [
        (curriculum, placement, seed)
        for curriculum in spec.curricula
        for placement in spec.placements
        for seed in spec.training.seeds
    ]


if __name__ == "__main__":
    main()
