from __future__ import annotations

import argparse
import json
from pathlib import Path

from plasticity_placement.p0d2hrabd.runtime import (
    plan_diagnostic,
    run_diagnostic,
    verify_complete_result,
)
from plasticity_placement.p0d2hrr.io import read_json_object


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 RAB paired error-topology audit")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--rab-output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--output", type=Path, required=True)
    status = commands.add_parser("status")
    status.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "plan":
        path = plan_diagnostic(output_dir=args.output, rab_output=args.rab_output)
    elif args.command == "run":
        path = run_diagnostic(args.output)
    elif args.command == "verify":
        path = verify_complete_result(args.output)
    elif args.command == "status":
        manifest = read_json_object(args.output / "manifest.json", "topology manifest")
        print(
            json.dumps(
                {
                    "run_id": manifest["run_id"],
                    "state": manifest["state"],
                    "errors": manifest.get("errors", []),
                    "historical_rab_decision_changed": False,
                    "inference_authorized": False,
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
