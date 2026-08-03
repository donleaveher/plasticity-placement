from __future__ import annotations

import argparse
import json
from pathlib import Path

from plasticity_placement.p0c.runtime import (
    current_environment_fingerprint,
    current_environment_snapshot,
)
from plasticity_placement.p0d2hrr.io import read_json_object
from plasticity_placement.p0d2hrtb.runtime import authorize_audit, plan_audit, run_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 frozen CPR route-transfer bridge audit")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("environment")
    plan = commands.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--cpr-output", type=Path, required=True)
    plan.add_argument("--qualification-summary", type=Path, required=True)
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--output", type=Path, required=True)
    authorize.add_argument("--authorization", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    status = commands.add_parser("status")
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
                sort_keys=True,
            )
        )
        return
    if args.command == "plan":
        path = plan_audit(
            output_dir=args.output,
            cpr_output=args.cpr_output,
            qualification_summary=args.qualification_summary,
        )
    elif args.command == "authorize":
        path = authorize_audit(args.output, args.authorization)
    elif args.command == "run":
        path = run_audit(args.output)
    elif args.command == "status":
        manifest = read_json_object(args.output / "manifest.json", "route-transfer manifest")
        print(
            json.dumps(
                {
                    "run_id": manifest["run_id"],
                    "state": manifest["state"],
                    "errors": manifest.get("errors", []),
                    "training_authorized": False,
                    "mappings_per_adapter_authorized": False,
                },
                sort_keys=True,
            )
        )
        return
    else:
        raise AssertionError(args.command)
    print(json.dumps({"output_path": str(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
