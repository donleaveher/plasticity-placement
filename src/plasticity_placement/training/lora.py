from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

from plasticity_placement.training.config import LoraTrainingConfig, select_layers
from plasticity_placement.training.data import CausalLMCollator, CompletionDataset, load_examples


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    output_dir: str
    selected_layers: list[int] | None
    trainable_parameters: int
    total_parameters: int
    optimizer_steps: int
    final_loss: float
    elapsed_seconds: float


def train_lora(config: LoraTrainingConfig) -> TrainingSummary:
    """Train and save a PEFT LoRA adapter for canonical lesson examples."""
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from torch.utils.data import DataLoader
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            get_linear_schedule_with_warmup,
        )
    except ImportError as error:
        raise RuntimeError(
            "training dependencies are missing; run `uv sync --extra train`"
        ) from error

    _set_seed(config.seed, torch)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must provide an EOS or padding token")

    model_kwargs: dict[str, Any] = {"torch_dtype": "auto"}
    if config.use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit training requires a CUDA runtime")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": 0}

    base_model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    num_layers = _infer_num_layers(base_model.config)
    selected_layers = select_layers(config.layer_band, num_layers, config.explicit_layers)

    if config.use_4bit:
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=config.gradient_checkpointing
        )
    elif config.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        base_model.enable_input_require_grads()

    if config.resume_adapter:
        model = PeftModel.from_pretrained(base_model, config.resume_adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=list(config.target_modules),
            layers_to_transform=selected_layers,
            bias="none",
        )
        model = get_peft_model(base_model, peft_config)

    model.config.use_cache = False
    if not config.use_4bit:
        model.to(_device(torch))

    examples = load_examples(config.data_path)
    dataset = CompletionDataset(examples, tokenizer, config.max_length)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=CausalLMCollator(tokenizer.pad_token_id),
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
    updates_per_epoch = math.ceil(len(loader) / config.gradient_accumulation_steps)
    total_steps = updates_per_epoch * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    final_loss = float("nan")
    started = time.perf_counter()
    model.train()
    for _ in range(config.epochs):
        for batch_index, batch in enumerate(loader, start=1):
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            output = model(**batch)
            final_loss = float(output.loss.detach().cpu())
            (output.loss / config.gradient_accumulation_steps).backward()
            is_update = batch_index % config.gradient_accumulation_steps == 0
            is_last = batch_index == len(loader)
            if is_update or is_last:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(config.output_dir)
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_count = sum(parameter.numel() for parameter in model.parameters())
    summary = TrainingSummary(
        output_dir=str(config.output_dir),
        selected_layers=selected_layers,
        trainable_parameters=trainable_count,
        total_parameters=total_count,
        optimizer_steps=optimizer_steps,
        final_loss=final_loss,
        elapsed_seconds=time.perf_counter() - started,
    )
    metadata = {"config": config.to_dict(), "summary": asdict(summary)}
    (config.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _infer_num_layers(model_config: Any) -> int:
    for name in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(model_config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("cannot infer transformer layer count from model config")


def _device(torch: Any) -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
