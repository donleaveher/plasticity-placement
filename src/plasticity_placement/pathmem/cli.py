from __future__ import annotations

import argparse
import json
from pathlib import Path

from plasticity_placement.pathmem.compiler import audit_compiled_bank, compile_bank
from plasticity_placement.pathmem.manifest import verify_g0_bundle, write_g0_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile and verify the CPU-only PathMem G0 protocol bundle",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect", help="inspect deterministic G0 counts")
    inspect_parser.add_argument("--compact", action="store_true")
    for command in ("prepare", "verify"):
        command_parser = subparsers.add_parser(
            command,
            help={
                "prepare": "write an immutable G0 bundle without training",
                "verify": "verify hashes, regeneration, schemas, and G0 gate",
            }[command],
        )
        command_parser.add_argument("--output", type=Path, required=True)
        command_parser.add_argument("--protocol-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect":
        report = audit_compiled_bank(compile_bank())
        print(
            json.dumps(
                report,
                ensure_ascii=False,
                sort_keys=True,
                indent=None if args.compact else 2,
            )
        )
        return
    if args.command == "prepare":
        manifest_path = write_g0_bundle(args.output, args.protocol_root)
        report = verify_g0_bundle(args.output, args.protocol_root)
        print(
            json.dumps(
                {"manifest_path": str(manifest_path), "verification": report},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return
    report = verify_g0_bundle(args.output, args.protocol_root)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
