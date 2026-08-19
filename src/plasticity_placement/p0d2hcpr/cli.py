from __future__ import annotations

import argparse
import json
from pathlib import Path

from plasticity_placement.p0c.runtime import (
    current_environment_fingerprint,
    current_environment_snapshot,
)
from plasticity_placement.p0d2hcpr.manifest import Manifest
from plasticity_placement.p0d2hcpr.preflight import plan_experiment
from plasticity_placement.p0d2hcpr.qualification import QualificationRequest, run_qualification
from plasticity_placement.p0d2hcpr.runtime import authorize_experiment, train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 composition-preserving route remediation pilot"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("environment", help="只读输出环境 fingerprint")
    plan = subparsers.add_parser("plan", help="验证失败证据、编译数据并冻结 preregistration")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--source-output", type=Path, required=True, help="verified RR1 output")
    plan.add_argument("--same-runtime-summary", type=Path, required=True)
    plan.add_argument("--source-code-revision-lock", type=Path, required=True)
    plan.add_argument("--spec", type=Path, required=True)
    authorize = subparsers.add_parser("authorize", help="采纳外部批准文件")
    authorize.add_argument("--output", type=Path, required=True)
    authorize.add_argument("--authorization", type=Path, required=True)
    train = subparsers.add_parser("train", help="运行唯一固定训练")
    train.add_argument("--output", type=Path, required=True)
    qualify = subparsers.add_parser("qualify", help="同 runtime OFF→ON locked qualification")
    qualify.add_argument("--output", type=Path, required=True)
    qualify.add_argument("--analysis-output", type=Path, required=True)
    qualify.add_argument(
        "--recovery-authorization",
        type=Path,
        help="外部签发的单次 post-inference classification-failure recovery 授权",
    )
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
            source_output=args.source_output,
            same_runtime_summary=args.same_runtime_summary,
            source_code_revision_lock=args.source_code_revision_lock,
            spec_path=args.spec,
        )
    elif args.command == "authorize":
        path = authorize_experiment(args.output, args.authorization)
    elif args.command == "train":
        path = train_experiment(args.output)
    elif args.command == "qualify":
        path = run_qualification(
            QualificationRequest(
                args.output,
                args.analysis_output,
                recovery_authorization=args.recovery_authorization,
            )
        )
    elif args.command == "status":
        manifest = Manifest.load(args.output / "manifest.json")
        print(
            json.dumps(
                {
                    "manifest_path": str(manifest.path),
                    "run_id": manifest.payload["run_id"],
                    "state": manifest.state,
                    "errors": manifest.payload.get("errors", []),
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


if __name__ == "__main__":
    main()
