from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from plasticity_placement.training.config import LoraTrainingConfig, select_layers
from plasticity_placement.training.data import (
    CausalLMCollator,
    CompletionDataset,
    TrainingExample,
    load_examples,
)
from plasticity_placement.training.model_utils import (
    chat_prompt,
    load_tokenizer,
    model_load_kwargs,
    runtime_device,
    set_seed,
)


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    output_dir: str
    selected_layers: list[int] | None
    trainable_parameters: int
    total_parameters: int
    optimizer_steps: int
    final_loss: float
    elapsed_seconds: float
    peak_memory_bytes: int
    adapter_bytes: int
    model_revision: str | None
    training_data_sha256: str
    config_sha256: str
    adapter_sha256: str
    precision: str


def train_lora(config: LoraTrainingConfig) -> TrainingSummary:
    """Train and save a PEFT LoRA adapter for canonical lesson examples."""
    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from torch.utils.data import DataLoader
        from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
    except ImportError as error:
        raise RuntimeError(
            "training dependencies are missing; run `uv sync --extra train`"
        ) from error

    set_seed(config.seed, torch)
    load_kwargs = {"revision": config.model_revision} if config.model_revision else {}
    tokenizer = load_tokenizer(
        config.model_name,
        config.model_revision,
    )
    model_kwargs, precision = model_load_kwargs(torch, use_4bit=config.use_4bit)

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs,
        **load_kwargs,
    )
    if not config.use_4bit:
        precision = str(next(base_model.parameters()).dtype).removeprefix("torch.")
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
        model.to(runtime_device(torch))

    examples = load_examples(config.data_path)
    if config.use_chat_template:
        examples = [
            TrainingExample(
                prompt=chat_prompt(tokenizer, example.prompt),
                completion=example.completion,
            )
            for example in examples
        ]
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
    natural_total_steps = updates_per_epoch * config.epochs
    total_steps = config.max_steps or natural_total_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    optimizer_steps = 0
    final_loss = float("nan")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.train()
    epochs_to_run = (
        math.ceil(total_steps / updates_per_epoch) if config.max_steps else config.epochs
    )
    for _ in range(epochs_to_run):
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
                if optimizer_steps >= total_steps:
                    break
        if optimizer_steps >= total_steps:
            break

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_dir, safe_serialization=True)
    if config.save_tokenizer:
        tokenizer.save_pretrained(config.output_dir)
    adapter_files = _adapter_bundle_files(config.output_dir)
    adapter_bytes = sum(path.stat().st_size for path in adapter_files)
    adapter_sha256 = _files_hash(adapter_files, config.output_dir)
    config_payload = config.to_dict()
    config_sha256 = _json_hash(config_payload)
    training_data_sha256 = sha256(config.data_path.read_bytes()).hexdigest()
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
        peak_memory_bytes=(
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        adapter_bytes=adapter_bytes,
        model_revision=config.model_revision,
        training_data_sha256=training_data_sha256,
        config_sha256=config_sha256,
        adapter_sha256=adapter_sha256,
        precision=precision,
    )
    metadata = {"config": config_payload, "summary": asdict(summary)}
    metadata_path = config.output_dir / "training_metadata.json"
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(metadata_path)
    return summary


def _infer_num_layers(model_config: Any) -> int:
    for name in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(model_config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("cannot infer transformer layer count from model config")


def _adapter_bundle_files(output_dir: Path) -> list[Path]:
    files = sorted(output_dir.glob("adapter_model.*"))
    config_path = output_dir / "adapter_config.json"
    if config_path.exists():
        files.append(config_path)
    if not files or not any(path.name.startswith("adapter_model.") for path in files):
        raise FileNotFoundError(f"missing adapter weights under {output_dir}")
    return sorted(files)


def _files_hash(files: list[Path], root: Path) -> str:
    digest = sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()
