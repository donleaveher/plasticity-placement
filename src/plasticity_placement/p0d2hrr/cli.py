from __future__ import annotations

import argparse
import json
from pathlib import Path

from plasticity_placement.p0c.runtime import (
    current_environment_fingerprint,
    current_environment_snapshot,
)
from plasticity_placement.p0d2hrr.analysis import aggregate_experiment
from plasticity_placement.p0d2hrr.manifest import PilotManifest
from plasticity_placement.p0d2hrr.paired_audit import (
    PairedAuditRequest,
    audit_paired_results,
)
from plasticity_placement.p0d2hrr.preflight import plan_experiment
from plasticity_placement.p0d2hrr.runtime import (
    authorize_experiment,
    evaluate_experiment,
    train_experiment,
)
from plasticity_placement.p0d2hrr.same_runtime_audit import (
    SameRuntimeAuditRequest,
    run_same_runtime_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行独立预注册的 P0-D2H route-remediation LoRA pilot",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("environment", help="只读输出当前环境与 fingerprint")

    plan = subparsers.add_parser(
        "plan",
        help="验证 frozen source，编译数据/评测审计并冻结 preregistration",
    )
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--source-manifest", type=Path, required=True)
    plan.add_argument("--spec", type=Path, required=True)

    authorize = subparsers.add_parser(
        "authorize",
        help="采纳由实验目录外独立签发、绑定 preregistration hash 的授权",
    )
    authorize.add_argument("--output", type=Path, required=True)
    authorize.add_argument("--authorization", type=Path, required=True)

    train = subparsers.add_parser(
        "train",
        help="在已授权 manifest 下运行唯一一个 frozen LoRA 配置",
    )
    train.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="运行一次 locked forced-choice + CRD source-compatible 评测",
    )
    evaluate.add_argument("--output", type=Path, required=True)

    aggregate = subparsers.add_parser(
        "aggregate",
        help="聚合完整结果并应用原 calibration gate 与 route guardrails",
    )
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--bootstrap-samples", type=int, default=10_000)

    paired_audit = subparsers.add_parser(
        "audit-p0",
        help="只读执行 provenance、base↔adapter 配对与 component-cell 审计",
    )
    paired_audit.add_argument(
        "--output",
        type=Path,
        required=True,
        help="完整 verified route-remediation run",
    )
    paired_audit.add_argument(
        "--base-crd-output",
        type=Path,
        required=True,
        help="完整 verified NF4 base CRD run",
    )
    paired_audit.add_argument(
        "--analysis-output",
        type=Path,
        required=True,
        help="独立的 immutable P0 analysis 输出目录",
    )
    paired_audit.add_argument(
        "--experiment-code-revision-lock",
        type=Path,
        help="原 RR pipeline 的 code_revision.txt；默认从 output 推导",
    )
    paired_audit.add_argument(
        "--base-crd-code-revision-lock",
        type=Path,
        help="原 NF4 CRD pipeline 的 code_revision.txt；标准 layout 下可自动推导",
    )
    paired_audit.add_argument(
        "--historical-preregistration-sha256",
        help="可选的历史控制台 preregistration hash；仅记录，不冒充当前工件",
    )
    paired_audit.add_argument("--bootstrap-samples", type=int, default=10_000)
    paired_audit.add_argument("--bootstrap-seed", type=int, default=20260730)

    same_runtime = subparsers.add_parser(
        "same-runtime-audit",
        help="在同一已加载模型对象内逐 prompt 执行 adapter OFF→ON 配对复评",
    )
    same_runtime.add_argument(
        "--output",
        type=Path,
        required=True,
        help="完整 verified route-remediation run",
    )
    same_runtime.add_argument(
        "--analysis-output",
        type=Path,
        required=True,
        help="新的独立 same-runtime 输出目录",
    )
    same_runtime.add_argument(
        "--experiment-code-revision-lock",
        type=Path,
        help="原 RR pipeline 的 code_revision.txt；默认从 output 推导",
    )
    same_runtime.add_argument("--bootstrap-samples", type=int, default=10_000)
    same_runtime.add_argument("--bootstrap-seed", type=int, default=20260801)
    same_runtime.add_argument(
        "--combined-noninferiority-margin",
        type=float,
        default=0.02,
    )
    same_runtime.add_argument(
        "--sentinel-score-tolerance",
        type=float,
        default=1e-5,
    )

    status = subparsers.add_parser(
        "status",
        help="只读输出当前 manifest 状态",
    )
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
            source_manifest=args.source_manifest,
            spec_path=args.spec,
        )
    elif args.command == "authorize":
        path = authorize_experiment(args.output, args.authorization)
    elif args.command == "train":
        path = train_experiment(args.output)
    elif args.command == "evaluate":
        path = evaluate_experiment(args.output)
    elif args.command == "aggregate":
        path = aggregate_experiment(args.output, args.bootstrap_samples)
    elif args.command == "audit-p0":
        path = audit_paired_results(
            PairedAuditRequest(
                route_remediation_output=args.output,
                base_crd_output=args.base_crd_output,
                analysis_output=args.analysis_output,
                experiment_code_revision_lock=args.experiment_code_revision_lock,
                base_crd_code_revision_lock=args.base_crd_code_revision_lock,
                historical_preregistration_sha256=(args.historical_preregistration_sha256),
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            )
        )
    elif args.command == "same-runtime-audit":
        path = run_same_runtime_audit(
            SameRuntimeAuditRequest(
                route_remediation_output=args.output,
                analysis_output=args.analysis_output,
                experiment_code_revision_lock=args.experiment_code_revision_lock,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
                combined_noninferiority_margin=args.combined_noninferiority_margin,
                sentinel_score_tolerance=args.sentinel_score_tolerance,
            )
        )
    elif args.command == "status":
        manifest = PilotManifest.load(args.output / "manifest.json")
        print(
            json.dumps(
                {
                    "manifest_path": str(manifest.path),
                    "run_id": manifest.payload["run_id"],
                    "state": manifest.state,
                    "errors": manifest.payload.get("errors", []),
                    "training_complexity_review_eligible": (
                        manifest.payload.get("aggregate", {}).get(
                            "training_complexity_review_eligible",
                            False,
                        )
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    else:
        raise AssertionError(args.command)
    print(json.dumps({"output_path": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
