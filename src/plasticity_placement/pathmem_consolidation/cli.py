from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from plasticity_placement.pathmem_consolidation.config import build_g0v2_contract
from plasticity_placement.pathmem_consolidation.manifest import (
    verify_g0v2_bundle,
    verify_parent_g0_v1,
    write_g0v2_bundle,
)
from plasticity_placement.pathmem_consolidation.planner import (
    audit_g1c_plan,
    compile_g1c_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile and verify the CPU-only PathMem G0-v2 R-OPCD plan"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect", help="show the safe contract and plan audit")
    _add_parent(inspect)
    inspect.add_argument("--compact", action="store_true")
    for command in ("prepare", "verify"):
        command_parser = subparsers.add_parser(
            command,
            help={
                "prepare": "write an immutable planning-only G0-v2 bundle",
                "verify": "verify parent, source, artifact, and regeneration hashes",
            }[command],
        )
        _add_parent(command_parser)
        command_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "inspect":
        parent = verify_parent_g0_v1(arguments.g0_v1_manifest)
        contract = build_g0v2_contract(
            parent_manifest_id=parent["manifest_id"],
            parent_manifest_sha256=parent["manifest_sha256"],
        )
        plan = compile_g1c_plan(
            parent_manifest_id=parent["manifest_id"],
            parent_manifest_sha256=parent["manifest_sha256"],
        )
        result = {
            "contract": contract,
            "plan_audit": audit_g1c_plan(plan),
            "safe_default": "no training, no GPU inference, no path contrast",
        }
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                indent=None if arguments.compact else 2,
            )
        )
        return 0
    if arguments.command == "prepare":
        manifest_path = write_g0v2_bundle(arguments.output, arguments.g0_v1_manifest)
        result = {
            "manifest_path": str(manifest_path),
            "verification": verify_g0v2_bundle(
                arguments.output, arguments.g0_v1_manifest
            ),
        }
    else:
        result = verify_g0v2_bundle(arguments.output, arguments.g0_v1_manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("passed", result.get("verification", {}).get("passed")) else 1


def _add_parent(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--g0-v1-manifest", type=Path, required=True)
