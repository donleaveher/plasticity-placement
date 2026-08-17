from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.compiler import compile_bank
from plasticity_placement.pathmem.io import immutable_json_write
from plasticity_placement.pathmem_exec.config import (
    G1_CONFIGS,
    P0_CONFIG,
    P0_DUPLICATE_NODE_BY_ITEM_INDEX,
)
from plasticity_placement.pathmem_exec.environment import verify_g0_or_raise
from plasticity_placement.pathmem_exec.g1 import run_g1
from plasticity_placement.pathmem_exec.g1_diagnostics import diagnose_g1_run
from plasticity_placement.pathmem_exec.p0 import run_p0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run frozen PathMem G1/P0 stages")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify-g0", help="CPU-only G0 verification")
    _add_g0_paths(verify)

    plan = subparsers.add_parser("show-plan", help="CPU-only execution-plan summary")
    _add_g0_paths(plan)

    g1 = subparsers.add_parser("g1", help="Run formal CUDA G1 qualification")
    _add_g0_paths(g1)
    g1.add_argument("--output", type=Path, required=True)
    g1.add_argument("--recipe", choices=sorted(G1_CONFIGS), default="B")

    diagnose = subparsers.add_parser(
        "diagnose-g1", help="CPU-only post-hoc diagnosis of a completed G1 run"
    )
    diagnose.add_argument("--run-root", type=Path, required=True)
    diagnose.add_argument("--output", type=Path)

    p0 = subparsers.add_parser("p0", help="Run G1-authorized CUDA P0 smoke")
    _add_g0_paths(p0)
    p0.add_argument("--output", type=Path, required=True)
    p0.add_argument("--g1-authorization", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "verify-g0":
        result = verify_g0_or_raise(arguments.g0, arguments.protocol_root)
    elif arguments.command == "show-plan":
        g0 = verify_g0_or_raise(arguments.g0, arguments.protocol_root)
        result = {"g0": g0, "execution_plan": execution_plan_summary()}
    elif arguments.command == "g1":
        result = run_g1(
            output_root=arguments.output,
            g0_dir=arguments.g0,
            protocol_root=arguments.protocol_root,
            config=G1_CONFIGS[arguments.recipe],
        )
    elif arguments.command == "diagnose-g1":
        result = diagnose_g1_run(arguments.run_root)
        if arguments.output is not None:
            immutable_json_write(arguments.output, result, "G1 post-hoc diagnostic")
    elif arguments.command == "p0":
        result = run_p0(
            output_root=arguments.output,
            g0_dir=arguments.g0,
            protocol_root=arguments.protocol_root,
            g1_authorization_path=arguments.g1_authorization,
        )
    else:  # pragma: no cover - argparse prevents this branch
        raise AssertionError(arguments.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def execution_plan_summary() -> dict[str, Any]:
    bank = compile_bank()
    g1_items = [item for item in bank.items if item.split == "interface_dev"]
    p0_items = [item for item in bank.items if item.split == "smoke"]
    duplicate_map = {
        item.item_id: P0_DUPLICATE_NODE_BY_ITEM_INDEX[index] for index, item in enumerate(p0_items)
    }
    return {
        "safe_default": "no training and no GPU inference",
        "authorization_order": ["G0", "G1", "P0"],
        "g1": {
            "split": "interface_dev",
            "item_count": len(g1_items),
            "current_only_anchors_per_item": ["A2", "B2"],
            "trained_artifact_count": len(g1_items) * 2,
            "recipes": {
                recipe_id: config.recipe.to_dict()
                for recipe_id, config in sorted(G1_CONFIGS.items())
            },
            "default_recipe": "B",
        },
        "p0": {
            "split": "smoke",
            "item_count": len(p0_items),
            "core_nodes_per_item": ["A1", "B1", "AB", "BA", "ABA", "BAA", "BAB", "ABB", "A2", "B2"],
            "technical_duplicate_by_item": duplicate_map,
            "trained_artifact_count": len(p0_items) * 11,
            "qualified_recipe_required": P0_CONFIG.recipe.recipe_id,
        },
    }


def _add_g0_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)


if __name__ == "__main__":
    raise SystemExit(main())
