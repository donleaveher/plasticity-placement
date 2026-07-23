from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from plasticity_placement.training.config import LayerBand, LoraTrainingConfig
from plasticity_placement.training.lora import train_lora


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练 Agent 经验的 LoRA 参数记忆")
    parser.add_argument("--model", required=True, help="Hugging Face 模型名称或本地路径")
    parser.add_argument("--data", type=Path, required=True, help="prompt/completion JSONL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer-band", choices=list(LayerBand), default=LayerBand.FULL)
    parser.add_argument("--layers", type=int, nargs="*", default=())
    parser.add_argument("--target-modules", nargs="+", default=("q_proj", "v_proj"))
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--resume-adapter", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = LoraTrainingConfig(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        layer_band=LayerBand(args.layer_band),
        explicit_layers=tuple(args.layers),
        target_modules=tuple(args.target_modules),
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        use_4bit=args.use_4bit,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        resume_adapter=args.resume_adapter,
    )
    summary = train_lora(config)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
