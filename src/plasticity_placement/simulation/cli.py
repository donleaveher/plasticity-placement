from __future__ import annotations

import argparse
from pathlib import Path

from plasticity_placement.simulation.experiment import run_unit
from plasticity_placement.simulation.world import ToolWorld


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 P0 最小载体交叉效应试验")
    parser.add_argument("--units-per-cell", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("artifacts/pilot.jsonl"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.units_per_cell <= 0:
        raise SystemExit("--units-per-cell 必须为正整数")

    world = ToolWorld(horizon=args.horizon)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    recurrence_levels = (0.1, 0.5, 0.9)
    volatility_levels = (0.0, 0.2, 0.5)

    with args.output.open("w", encoding="utf-8") as stream:
        for recurrence_index, recurrence in enumerate(recurrence_levels):
            for volatility_index, volatility in enumerate(volatility_levels):
                cell = recurrence_index * len(volatility_levels) + volatility_index
                for unit_index in range(args.units_per_cell):
                    seed = cell * 100_000 + unit_index
                    unit = world.create_unit(seed, recurrence, volatility)
                    for result in run_unit(unit):
                        stream.write(result.to_json() + "\n")

    print(f"已写入实验结果：{args.output}")


if __name__ == "__main__":
    main()
