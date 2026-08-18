from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from plasticity_placement.pathmem_ropcd_p0.bundle import (
    prepare_plan_bundle,
    verify_plan_bundle,
)
from plasticity_placement.pathmem_ropcd_p0.runtime import (
    adopt_run_authorization,
    aggregate_run,
    authorization_template,
    inspect_execution,
    run_evaluation,
    run_preflight,
    run_training,
    verify_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan and run the authorization-gated R-OPCD-specific P0 smoke"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("inspect", "verify G1-C and print the prospective P0 contract"),
        ("prepare-plan", "write the immutable CPU-only P0 plan bundle"),
        ("verify-plan", "read-only verification of the P0 plan bundle"),
        ("authorization-template", "print the exact human approval template"),
        ("preflight", "run CUDA/BF16, prompt, and privileged-teacher gates"),
        ("aggregate", "compute the frozen P0 engineering gate"),
        ("verify", "read-only verification and aggregate regeneration"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        _add_common(child)
    adopt = subparsers.add_parser(
        "adopt-authorization", help="verify and immutably adopt the P0 approval"
    )
    _add_common(adopt)
    adopt.add_argument("--approval", type=Path, required=True)
    for command, help_text in (
        ("train", "resume a bounded number of lineage-aware R-OPCD units"),
        ("evaluate", "resume a bounded number of frozen P0 panels"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        _add_common(child)
        child.add_argument("--max-units", type=int, default=1)
        child.add_argument("--unit-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    source = {
        "bundle_root": arguments.bundle,
        "parent_manifest": arguments.g0_v1_manifest,
        "g1c_run_root": arguments.g1c_run,
    }
    common = {
        **source,
        "plan_root": arguments.plan,
        "output_root": arguments.output,
    }
    if arguments.command == "inspect":
        result = inspect_execution(**source, output_root=arguments.output)
    elif arguments.command == "prepare-plan":
        result = prepare_plan_bundle(**source, plan_root=arguments.plan)
    elif arguments.command == "verify-plan":
        result = verify_plan_bundle(**source, plan_root=arguments.plan)
    elif arguments.command == "authorization-template":
        result = authorization_template(**common)
    elif arguments.command == "adopt-authorization":
        result = adopt_run_authorization(**common, approval_path=arguments.approval)
    elif arguments.command == "preflight":
        result = run_preflight(**common)
    elif arguments.command == "train":
        result = run_training(
            **common,
            max_units=arguments.max_units,
            target_unit_id=arguments.unit_id,
        )
    elif arguments.command == "evaluate":
        result = run_evaluation(
            **common,
            max_units=arguments.max_units,
            target_unit_id=arguments.unit_id,
        )
    elif arguments.command == "aggregate":
        result = aggregate_run(**common)
    elif arguments.command == "verify":
        result = verify_run(**common)
    else:  # pragma: no cover
        raise AssertionError(arguments.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--g0-v1-manifest", type=Path, required=True)
    parser.add_argument("--g1c-run", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
