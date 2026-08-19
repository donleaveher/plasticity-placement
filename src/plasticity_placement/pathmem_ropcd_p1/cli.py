from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from plasticity_placement.pathmem_ropcd_p1.benchmark import (
    adopt_benchmark_authorization,
    aggregate_benchmark,
    benchmark_authorization_template,
    run_benchmark,
    verify_benchmark,
)
from plasticity_placement.pathmem_ropcd_p1.bundle import (
    prepare_plan_bundle,
    verify_plan_bundle,
)
from plasticity_placement.pathmem_ropcd_p1.runtime import (
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
        description="Plan, benchmark, and run the authorization-gated R-OPCD P1 test"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("inspect", "verify repaired G2 and inspect the prospective CPU plan"),
        ("prepare-plan", "write the immutable CPU-only P1 plan"),
        ("verify-plan", "read-only verification of the P1 plan"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        _add_source(child)
        child.add_argument("--plan", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    for command, help_text in (
        ("benchmark-authorization-template", "print the hardware_dev approval template"),
        ("benchmark-aggregate", "freeze the four-item resource profile"),
        ("benchmark-verify", "regenerate and verify the resource profile"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        _add_benchmark_common(child)
    benchmark_adopt = subparsers.add_parser(
        "benchmark-adopt-authorization", help="immutably adopt hardware_dev approval"
    )
    _add_benchmark_common(benchmark_adopt)
    benchmark_adopt.add_argument("--approval", type=Path, required=True)
    benchmark_run = subparsers.add_parser(
        "benchmark-run", help="resume a bounded number of hardware benchmark units"
    )
    _add_benchmark_common(benchmark_run)
    benchmark_run.add_argument("--max-units", type=int, default=1)
    for command, help_text in (
        ("authorization-template", "print the resource-bound formal P1 approval template"),
        ("preflight", "run CUDA, prompt, and privileged-teacher gates"),
        ("aggregate", "compute the frozen item-bootstrap P1 decision"),
        ("verify", "read-only verification and aggregate regeneration"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        _add_execution_common(child)
    adopt = subparsers.add_parser(
        "adopt-authorization", help="immutably adopt formal P1 approval"
    )
    _add_execution_common(adopt)
    adopt.add_argument("--approval", type=Path, required=True)
    for command, help_text in (
        ("train", "resume a bounded number of lineage-aware P1 units"),
        ("evaluate", "resume a bounded number of frozen P1 panels"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        _add_execution_common(child)
        child.add_argument("--max-units", type=int, default=1)
        child.add_argument("--unit-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    source = {
        "bundle_root": arguments.bundle,
        "parent_manifest": arguments.g0_v1_manifest,
        "g1c_run_root": arguments.g1c_run,
        "p0_plan_root": arguments.p0_plan,
        "p0_run_root": arguments.p0_run,
        "repair_root": arguments.g2_repair,
    }
    plan = {**source, "plan_root": arguments.plan}
    output = {**plan, "output_root": arguments.output}
    formal = (
        {**output, "resource_profile_path": arguments.resource_profile}
        if hasattr(arguments, "resource_profile")
        else None
    )
    if arguments.command == "inspect":
        result = inspect_execution(**source, output_root=arguments.output)
    elif arguments.command == "prepare-plan":
        result = prepare_plan_bundle(**plan)
    elif arguments.command == "verify-plan":
        result = verify_plan_bundle(**plan)
    elif arguments.command == "benchmark-authorization-template":
        result = benchmark_authorization_template(**output)
    elif arguments.command == "benchmark-adopt-authorization":
        result = adopt_benchmark_authorization(**output, approval_path=arguments.approval)
    elif arguments.command == "benchmark-run":
        result = run_benchmark(**output, max_units=arguments.max_units)
    elif arguments.command == "benchmark-aggregate":
        result = aggregate_benchmark(**output)
    elif arguments.command == "benchmark-verify":
        result = verify_benchmark(**output)
    elif arguments.command == "authorization-template":
        assert formal is not None
        result = authorization_template(**formal)
    elif arguments.command == "adopt-authorization":
        assert formal is not None
        result = adopt_run_authorization(**formal, approval_path=arguments.approval)
    elif arguments.command == "preflight":
        assert formal is not None
        result = run_preflight(**formal)
    elif arguments.command == "train":
        assert formal is not None
        result = run_training(
            **formal, max_units=arguments.max_units, target_unit_id=arguments.unit_id
        )
    elif arguments.command == "evaluate":
        assert formal is not None
        result = run_evaluation(
            **formal, max_units=arguments.max_units, target_unit_id=arguments.unit_id
        )
    elif arguments.command == "aggregate":
        assert formal is not None
        result = aggregate_run(**formal)
    elif arguments.command == "verify":
        assert formal is not None
        result = verify_run(**formal)
    else:  # pragma: no cover
        raise AssertionError(arguments.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _add_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--g0-v1-manifest", type=Path, required=True)
    parser.add_argument("--g1c-run", type=Path, required=True)
    parser.add_argument("--p0-plan", type=Path, required=True)
    parser.add_argument("--p0-run", type=Path, required=True)
    parser.add_argument("--g2-repair", type=Path, required=True)


def _add_benchmark_common(parser: argparse.ArgumentParser) -> None:
    _add_source(parser)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def _add_execution_common(parser: argparse.ArgumentParser) -> None:
    _add_benchmark_common(parser)
    parser.add_argument("--resource-profile", type=Path, required=True)
