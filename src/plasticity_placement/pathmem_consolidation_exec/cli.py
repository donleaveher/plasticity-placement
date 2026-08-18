from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from plasticity_placement.pathmem_consolidation_exec.identity import (
    build_authorization_template,
)
from plasticity_placement.pathmem_consolidation_exec.runtime import (
    adopt_run_authorization,
    aggregate_run,
    inspect_execution,
    run_evaluation,
    run_preflight,
    run_training,
    verify_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the authorization-gated PathMem G1-C R-OPCD qualification"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser(
        "inspect", help="verify the planning source and show the frozen execution contract"
    )
    _add_common(inspect)

    template = subparsers.add_parser(
        "authorization-template",
        help="print the exact external human-approval template without writing",
    )
    _add_common(template)

    adopt = subparsers.add_parser(
        "adopt-authorization",
        help="verify and immutably adopt an external G1-C execution approval",
    )
    _add_common(adopt)
    adopt.add_argument("--approval", type=Path, required=True)

    preflight = subparsers.add_parser(
        "preflight", help="run the GPU environment, token, and privileged-teacher gate"
    )
    _add_common(preflight)

    train = subparsers.add_parser(
        "train", help="resume a bounded number of authorized side-memory units"
    )
    _add_common(train)
    _add_unit_selection(train)

    evaluate = subparsers.add_parser(
        "evaluate", help="resume a bounded number of trained-unit evaluations"
    )
    _add_common(evaluate)
    _add_unit_selection(evaluate)

    aggregate = subparsers.add_parser(
        "aggregate", help="compute the frozen G1-C metrics and gate on CPU"
    )
    _add_common(aggregate)

    verify = subparsers.add_parser(
        "verify", help="read-only verification and aggregate regeneration"
    )
    _add_common(verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    common = {
        "bundle_root": arguments.bundle,
        "parent_manifest": arguments.g0_v1_manifest,
        "output_root": arguments.output,
    }
    if arguments.command == "inspect":
        result = inspect_execution(**common)
    elif arguments.command == "authorization-template":
        result = build_authorization_template(**common)
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
    else:  # pragma: no cover - argparse prevents this branch
        raise AssertionError(arguments.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--g0-v1-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def _add_unit_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-units", type=int, default=1)
    parser.add_argument("--unit-id")
