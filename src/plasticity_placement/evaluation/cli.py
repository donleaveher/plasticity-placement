from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from plasticity_placement.evaluation.config import EvaluationConfig
from plasticity_placement.evaluation.evaluator import evaluate_lora


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估 LoRA 经验写入、迁移和回滚效果")
    parser.add_argument("--model", required=True, help="基础模型名称或本地路径")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = EvaluationConfig(
        model_name=args.model,
        adapter_path=args.adapter,
        probes_path=args.probes,
        output_dir=args.output,
        max_input_length=args.max_input_length,
        max_new_tokens=args.max_new_tokens,
        use_4bit=args.use_4bit,
        seed=args.seed,
    )
    summary = evaluate_lora(config)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
