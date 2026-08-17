from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.dag import CoreDAGPlan, DAGNodePlan, resolve_node_identity
from plasticity_placement.pathmem.identity import (
    RuntimeIdentity,
    block_seed,
    root_adapter_seed,
)
from plasticity_placement.pathmem.io import file_hash, immutable_json_write, json_hash
from plasticity_placement.pathmem.schema import EventBlock, OperatorKind
from plasticity_placement.pathmem_exec.artifacts import (
    adapter_bundle_hash,
    write_lineage,
)
from plasticity_placement.pathmem_exec.config import PhaseConfig
from plasticity_placement.training.config import LayerBand, LoraTrainingConfig
from plasticity_placement.training.lora import train_lora
from plasticity_placement.training.model_utils import model_load_kwargs, set_seed

ROOT_ADAPTER_VERSION = "pathmem-root-adapter-v1"
TRAINER_CONTRACT_VERSION = "pathmem-reset-adamw-trainer-v1"


def trainer_config_identity(config: PhaseConfig) -> str:
    return json_hash(
        {
            "trainer_contract": TRAINER_CONTRACT_VERSION,
            "recipe": config.recipe.to_dict(),
            "checkpoint_selection": "fixed_final_step",
            "evaluation_sampling": False,
            "optimizer_state_policy": "reset_at_every_event",
        }
    )


def runtime_identity(
    *,
    config: PhaseConfig,
    item_id: str,
    root_adapter_sha256: str,
    environment: dict[str, Any],
) -> RuntimeIdentity:
    implementation = environment["implementation"]["source_files"]
    trainer_sources = {
        path: digest
        for path, digest in implementation.items()
        if "/training/" in path or path.endswith("pathmem_exec/training.py")
    }
    return RuntimeIdentity(
        experiment_seed=20260813,
        item_id=item_id,
        training_seed=config.training_seeds[0],
        operator=OperatorKind.LORA_ADAMW_RESET,
        base_model_name=config.model.name,
        base_model_revision=config.model.revision,
        root_adapter_sha256=root_adapter_sha256,
        trainer_impl_sha256=json_hash(trainer_sources),
        trainer_config_sha256=trainer_config_identity(config),
        tokenizer_sha256=str(environment["model"]["tokenizer_sha256"]),
        library_lock_sha256=str(environment["library_lock_sha256"]),
        code_and_patch_sha256=str(environment["code_and_patch_sha256"]),
        resolved_precision=str(environment["resolved_precision"]),
    )


def create_root_adapter(
    *,
    config: PhaseConfig,
    item_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Create the item-specific, untrained LoRA root used by every path."""
    metadata_path = output_dir / "pathmem_root.json"
    if output_dir.exists():
        return _verify_root_adapter(config, item_id, output_dir, metadata_path)

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "training dependencies are missing; run `uv sync --extra train`"
        ) from error

    seed = root_adapter_seed(
        20260813,
        item_id,
        config.training_seeds[0],
        OperatorKind.LORA_ADAMW_RESET,
    )
    set_seed(seed, torch)
    load_kwargs, precision = model_load_kwargs(torch, use_4bit=True)
    if precision != "nf4-bfloat16":
        raise RuntimeError(f"root adapter precision changed: {precision}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        **load_kwargs,
    )
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=config.recipe.gradient_checkpointing,
    )
    model = get_peft_model(
        base_model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=config.recipe.rank,
            lora_alpha=config.recipe.alpha,
            lora_dropout=config.recipe.dropout,
            target_modules=list(config.recipe.target_modules),
            layers_to_transform=None,
            bias="none",
        ),
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        model.save_pretrained(staging, safe_serialization=True)
        adapter_sha256 = adapter_bundle_hash(staging)
        payload = {
            "schema_version": ROOT_ADAPTER_VERSION,
            "item_id": item_id,
            "model": config.model.to_dict(),
            "root_adapter_seed": seed,
            "adapter_sha256": adapter_sha256,
            "recipe_sha256": trainer_config_identity(config),
            "resolved_precision": precision,
        }
        immutable_json_write(staging / metadata_path.name, payload, "root adapter metadata")
        staging.rename(output_dir)
    finally:
        del model
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return _verify_root_adapter(config, item_id, output_dir, metadata_path)


def write_event_data(block: EventBlock, path: Path) -> str:
    rows = [
        {
            "example_id": example.example_id,
            "prompt": example.prompt,
            "completion": example.completion,
        }
        for example in block.examples
    ]
    payload = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"event data changed: {path}")
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return file_hash(path)


def train_event_node(
    *,
    config: PhaseConfig,
    attempt_id: str,
    block: EventBlock,
    plan: DAGNodePlan,
    dag: CoreDAGPlan,
    runtime: RuntimeIdentity,
    parent_adapter_dir: Path,
    parent_state_sha256: str,
    root_state_sha256: str,
    data_path: Path,
    output_dir: Path,
    occurrence_id: str | None = None,
) -> dict[str, Any]:
    """Train one event from its exact parent with fresh AdamW/scheduler state."""
    data_sha256 = write_event_data(block, data_path)
    parent_adapter_sha256 = adapter_bundle_hash(parent_adapter_dir)
    if output_dir.exists():
        return _verify_trained_node(
            output_dir=output_dir,
            attempt_id=attempt_id,
            parent_state_sha256=parent_state_sha256,
            root_state_sha256=root_state_sha256,
            training_data_sha256=data_sha256,
        )

    identity_input = resolve_node_identity(
        plan=plan,
        dag=dag,
        runtime=runtime,
        parent_state_sha256=parent_state_sha256,
    )
    if occurrence_id is not None:
        identity_payload = identity_input.to_dict()
        identity_payload.pop("node_id")
        identity_payload["occurrence_id"] = occurrence_id
        node_id = json_hash(identity_payload)
    else:
        identity_payload = identity_input.to_dict()
        node_id = identity_input.node_id

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    train_config = LoraTrainingConfig(
        model_name=config.model.name,
        model_revision=config.model.revision,
        data_path=data_path,
        output_dir=staging,
        layer_band=LayerBand.FULL,
        target_modules=config.recipe.target_modules,
        rank=config.recipe.rank,
        alpha=config.recipe.alpha,
        dropout=config.recipe.dropout,
        learning_rate=config.recipe.learning_rate,
        weight_decay=config.recipe.weight_decay,
        epochs=1,
        batch_size=config.recipe.batch_size,
        gradient_accumulation_steps=config.recipe.gradient_accumulation_steps,
        max_length=config.recipe.training_max_length,
        warmup_ratio=0.0,
        max_steps=config.recipe.optimizer_steps_per_event,
        seed=block_seed(
            runtime.training_seed,
            plan.event_block_sha256,
            plan.block_id.rsplit(":", 1)[-1],
        ),
        use_4bit=True,
        gradient_checkpointing=config.recipe.gradient_checkpointing,
        use_chat_template=True,
        save_tokenizer=False,
        resume_adapter=parent_adapter_dir,
    )
    summary = train_lora(train_config)
    if summary.precision != "nf4-bfloat16":
        raise RuntimeError(f"node precision changed: {summary.precision}")
    if summary.optimizer_steps != config.recipe.optimizer_steps_per_event:
        raise RuntimeError(f"optimizer step count changed: {summary.optimizer_steps}")
    adapter_sha256 = adapter_bundle_hash(staging)
    if adapter_sha256 != summary.adapter_sha256:
        raise RuntimeError("saved adapter hash differs from training summary")
    lineage = {
        "attempt_id": attempt_id,
        "node_id": node_id,
        "node_identity": identity_payload,
        "adapter_sha256": adapter_sha256,
        "parent_adapter_sha256": parent_adapter_sha256,
        "parent_state_sha256": parent_state_sha256,
        "root_state_sha256": root_state_sha256,
        "training_data_sha256": data_sha256,
        "trainer_config_sha256": runtime.trainer_config_sha256,
        "training_summary": asdict(summary),
    }
    write_lineage(staging, lineage)
    staging.rename(output_dir)
    return _verify_trained_node(
        output_dir=output_dir,
        attempt_id=attempt_id,
        parent_state_sha256=parent_state_sha256,
        root_state_sha256=root_state_sha256,
        training_data_sha256=data_sha256,
    )


def _verify_root_adapter(
    config: PhaseConfig,
    item_id: str,
    output_dir: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"root adapter is incomplete: {output_dir}")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": ROOT_ADAPTER_VERSION,
        "item_id": item_id,
        "model": config.model.to_dict(),
        "root_adapter_seed": root_adapter_seed(
            20260813,
            item_id,
            config.training_seeds[0],
            OperatorKind.LORA_ADAMW_RESET,
        ),
        "adapter_sha256": adapter_bundle_hash(output_dir),
        "recipe_sha256": trainer_config_identity(config),
        "resolved_precision": "nf4-bfloat16",
    }
    if payload != expected:
        raise ValueError(f"root adapter identity changed: {output_dir}")
    return payload


def _verify_trained_node(
    *,
    output_dir: Path,
    attempt_id: str,
    parent_state_sha256: str,
    root_state_sha256: str,
    training_data_sha256: str,
) -> dict[str, Any]:
    lineage_path = output_dir / "pathmem_lineage.json"
    if not lineage_path.is_file():
        raise FileNotFoundError(f"trained adapter is incomplete: {output_dir}")
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    observed = {
        "attempt_id": lineage.get("attempt_id"),
        "adapter_sha256": adapter_bundle_hash(output_dir),
        "parent_state_sha256": lineage.get("parent_state_sha256"),
        "root_state_sha256": lineage.get("root_state_sha256"),
        "training_data_sha256": lineage.get("training_data_sha256"),
    }
    expected = {
        "attempt_id": attempt_id,
        "adapter_sha256": lineage.get("adapter_sha256"),
        "parent_state_sha256": parent_state_sha256,
        "root_state_sha256": root_state_sha256,
        "training_data_sha256": training_data_sha256,
    }
    if observed != expected:
        raise ValueError(f"trained adapter lineage changed: {output_dir}")
    return lineage
