from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.io import file_hash, immutable_json_write, json_hash
from plasticity_placement.pathmem_consolidation_exec.config import RopcdExecutionRecipe
from plasticity_placement.pathmem_consolidation_exec.training import (
    TrainingSession,
    _distillation_loss,
    _require_finite_tensor,
    unit_slug,
)
from plasticity_placement.pathmem_exec.artifacts import adapter_bundle_hash
from plasticity_placement.pathmem_ropcd_p0.config import (
    CHECKPOINT_SCHEMA_VERSION,
    P0_ROPCD_RECIPE,
    TRAINING_SCHEMA_VERSION,
)
from plasticity_placement.training.model_utils import set_seed


@dataclass(frozen=True, slots=True)
class TrainingArtifactContract:
    root_schema_version: str
    training_schema_version: str
    checkpoint_schema_version: str
    label: str
    include_training_seed: bool = False


P0_ARTIFACT_CONTRACT = TrainingArtifactContract(
    root_schema_version="pathmem-ropcd-p0-root-v1",
    training_schema_version=TRAINING_SCHEMA_VERSION,
    checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
    label="R-OPCD P0",
)


def prepare_root_adapter(
    session: TrainingSession,
    *,
    root: Path,
    run_id: str,
    item_id: str,
    root_seed: int,
    root_id: str | None = None,
    contract: TrainingArtifactContract = P0_ARTIFACT_CONTRACT,
) -> dict[str, Any]:
    artifact_root_id = item_id if root_id is None else root_id
    if root.exists():
        return verify_root_adapter(
            root,
            run_id=run_id,
            item_id=item_id,
            root_seed=root_seed,
            recipe=session.recipe,
            root_id=root_id,
            contract=contract,
        )
    from peft import LoraConfig, get_peft_model

    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".root.", dir=root.parent))
    set_seed(root_seed, session.torch)
    model = get_peft_model(
        session.base_model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=session.recipe.rank,
            lora_alpha=session.recipe.alpha,
            lora_dropout=session.recipe.dropout,
            target_modules=list(session.recipe.target_modules),
            layers_to_transform=None,
            bias=session.recipe.bias,
            init_lora_weights=True,
        ),
    )
    model.save_pretrained(staging / "adapter", safe_serialization=True)
    session.base_model = model.unload()
    session.base_model.config.use_cache = False
    identity = {
        "schema_version": contract.root_schema_version,
        "run_id": run_id,
        "item_id": item_id,
        "recipe": session.recipe.to_dict(),
        "recipe_sha256": json_hash(session.recipe.to_dict()),
        "root_seed": root_seed,
        "initializer": "peft_init_lora_weights_true",
        "effective_initial_delta": "zero",
        "adapter_sha256": adapter_bundle_hash(staging / "adapter"),
    }
    if root_id is not None:
        identity["root_id"] = artifact_root_id
    immutable_json_write(staging / "root.json", identity, f"{contract.label} root")
    staging.rename(root)
    return verify_root_adapter(
        root,
        run_id=run_id,
        item_id=item_id,
        root_seed=root_seed,
        recipe=session.recipe,
        root_id=root_id,
        contract=contract,
    )


def verify_root_adapter(
    root: Path,
    *,
    run_id: str,
    item_id: str,
    root_seed: int,
    recipe: RopcdExecutionRecipe = P0_ROPCD_RECIPE,
    root_id: str | None = None,
    contract: TrainingArtifactContract = P0_ARTIFACT_CONTRACT,
) -> dict[str, Any]:
    if {path.name for path in root.iterdir()} != {"adapter", "root.json"}:
        raise ValueError(f"{contract.label} root file set changed")
    metadata = json.loads((root / "root.json").read_text(encoding="utf-8"))
    expected = {
        "schema_version": contract.root_schema_version,
        "run_id": run_id,
        "item_id": item_id,
        "recipe": recipe.to_dict(),
        "recipe_sha256": json_hash(recipe.to_dict()),
        "root_seed": root_seed,
        "initializer": "peft_init_lora_weights_true",
        "effective_initial_delta": "zero",
        "adapter_sha256": adapter_bundle_hash(root / "adapter"),
    }
    if root_id is not None:
        expected["root_id"] = root_id
    if metadata != expected:
        raise ValueError(f"{contract.label} root identity changed")
    return {
        **metadata,
        "adapter_dir": str((root / "adapter").resolve()),
        "metadata_sha256": file_hash(root / "root.json"),
    }


def train_p0_unit(
    session: TrainingSession,
    *,
    unit: dict[str, Any],
    parent_adapter_dir: Path,
    root_adapter_sha256: str,
    run_id: str,
    output_root: Path,
    training_seed: int | None = None,
    contract: TrainingArtifactContract = P0_ARTIFACT_CONTRACT,
) -> dict[str, Any]:
    recipe = session.recipe
    artifact_training_seed = recipe.training_seed if training_seed is None else training_seed
    unit_id = str(unit["unit_id"])
    unit_sha256 = json_hash(unit)
    parent_adapter_sha256 = adapter_bundle_hash(parent_adapter_dir)
    final_root = output_root / "units" / unit_slug(unit_id)
    if final_root.exists():
        return verify_p0_unit(
            final_root,
            unit=unit,
            parent_adapter_sha256=parent_adapter_sha256,
            root_adapter_sha256=root_adapter_sha256,
            run_id=run_id,
            recipe=recipe,
            training_seed=artifact_training_seed,
            contract=contract,
        )
    checkpoint_root = output_root / "work" / unit_slug(unit_id) / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint = _latest_checkpoint(
        checkpoint_root,
        unit_sha256=unit_sha256,
        parent_adapter_sha256=parent_adapter_sha256,
        root_adapter_sha256=root_adapter_sha256,
        run_id=run_id,
        recipe=recipe,
        contract=contract,
    )
    model = _load_model(session, checkpoint, parent_adapter_dir)
    optimizer = session.torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=recipe.learning_rate,
        betas=recipe.betas,
        eps=recipe.epsilon,
        weight_decay=recipe.weight_decay,
    )
    scheduler = session.torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    records: list[dict[str, Any]] = []
    start_step = 0
    if checkpoint is not None:
        state = session.torch.load(
            checkpoint / "training_state.pt", map_location="cpu", weights_only=True
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        metadata = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
        start_step = int(metadata["completed_steps"])
        records = json.loads((checkpoint / "records.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    try:
        for step in range(start_step, recipe.optimizer_steps_per_unit):
            pair = unit["rollout_prompt_pairs"][step % recipe.prompt_pairs_per_unit]
            step_seed = _step_seed(
                artifact_training_seed, str(unit["step_seed_namespace"]), step
            )
            set_seed(step_seed, session.torch)
            optimizer.zero_grad(set_to_none=True)
            loss, record = _distillation_loss(
                session,
                model=model,
                unit=unit,
                pair=pair,
                step=step,
                step_seed=step_seed,
            )
            _require_finite_tensor(session.torch, loss, f"{contract.label} distillation loss")
            loss.backward()
            gradient_norm = session.torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                recipe.max_grad_norm,
            )
            _require_finite_tensor(session.torch, gradient_norm, f"{contract.label} gradient norm")
            optimizer.step()
            scheduler.step()
            records.append(
                {
                    **record,
                    "gradient_norm": float(gradient_norm.detach().float().cpu()),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "logical_name": unit["logical_name"],
                    "block_id": unit["block_id"],
                    "step_seed_namespace": unit["step_seed_namespace"],
                    "ordered_exposure_sha256": unit["ordered_exposure_sha256"],
                    "parent_unit_id": unit["parent_unit_id"],
                }
            )
            completed_steps = step + 1
            if (
                completed_steps % recipe.checkpoint_interval_steps == 0
                or completed_steps == recipe.optimizer_steps_per_unit
            ):
                _save_checkpoint(
                    checkpoint_root,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    records=records,
                    completed_steps=completed_steps,
                    unit_sha256=unit_sha256,
                    parent_adapter_sha256=parent_adapter_sha256,
                    root_adapter_sha256=root_adapter_sha256,
                    run_id=run_id,
                    recipe=recipe,
                    torch=session.torch,
                    contract=contract,
                )
        final_checkpoint = _latest_checkpoint(
            checkpoint_root,
            unit_sha256=unit_sha256,
            parent_adapter_sha256=parent_adapter_sha256,
            root_adapter_sha256=root_adapter_sha256,
            run_id=run_id,
            recipe=recipe,
            contract=contract,
        )
        if final_checkpoint is None:
            raise RuntimeError(f"{contract.label} final checkpoint is missing")
        _publish_final_unit(
            final_root,
            final_checkpoint=final_checkpoint,
            records=records,
            unit=unit,
            parent_adapter_sha256=parent_adapter_sha256,
            root_adapter_sha256=root_adapter_sha256,
            run_id=run_id,
            recipe=recipe,
            elapsed_seconds=time.perf_counter() - started,
            training_seed=artifact_training_seed,
            contract=contract,
        )
    finally:
        session.base_model = model.unload()
        session.base_model.config.use_cache = False
        if session.torch.cuda.is_available():
            session.torch.cuda.empty_cache()
    return verify_p0_unit(
        final_root,
        unit=unit,
        parent_adapter_sha256=parent_adapter_sha256,
        root_adapter_sha256=root_adapter_sha256,
        run_id=run_id,
        recipe=recipe,
        training_seed=artifact_training_seed,
        contract=contract,
    )


def verify_p0_unit(
    final_root: Path,
    *,
    unit: dict[str, Any],
    parent_adapter_sha256: str,
    root_adapter_sha256: str,
    run_id: str,
    recipe: RopcdExecutionRecipe = P0_ROPCD_RECIPE,
    training_seed: int | None = None,
    contract: TrainingArtifactContract = P0_ARTIFACT_CONTRACT,
) -> dict[str, Any]:
    artifact_training_seed = recipe.training_seed if training_seed is None else training_seed
    expected_files = {"adapter", "training_metadata.json", "training_records.json"}
    if {path.name for path in final_root.iterdir()} != expected_files:
        raise ValueError(f"{contract.label} trained-unit file set changed")
    metadata_path = final_root / "training_metadata.json"
    records_path = final_root / "training_records.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = json.loads(records_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": contract.training_schema_version,
        "unit_id": unit["unit_id"],
        "unit_sha256": json_hash(unit),
        "run_id": run_id,
        "recipe": recipe.to_dict(),
        "recipe_sha256": json_hash(recipe.to_dict()),
        "parent_unit_id": unit["parent_unit_id"],
        "parent_adapter_sha256": parent_adapter_sha256,
        "root_adapter_sha256": root_adapter_sha256,
        "optimizer_state_lifecycle": "reset_at_semantic_consolidation",
        "completed_steps": recipe.optimizer_steps_per_unit,
        "checkpoint_selection": recipe.checkpoint_selection,
        "adapter_sha256": adapter_bundle_hash(final_root / "adapter"),
        "records_sha256": json_hash(records),
    }
    if contract.include_training_seed:
        expected["training_seed"] = artifact_training_seed
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{contract.label} trained-unit metadata changed: {key}")
    if len(records) != recipe.optimizer_steps_per_unit:
        raise ValueError(f"{contract.label} step record count changed")
    for step, record in enumerate(records):
        if (
            record.get("step") != step
            or record.get("step_seed")
            != _step_seed(artifact_training_seed, str(unit["step_seed_namespace"]), step)
            or record.get("block_id") != unit["block_id"]
            or record.get("ordered_exposure_sha256") != unit["ordered_exposure_sha256"]
        ):
            raise ValueError(f"{contract.label} step exposure record changed")
    return {
        **metadata,
        "adapter_dir": str((final_root / "adapter").resolve()),
        "training_metadata_sha256": file_hash(metadata_path),
        "training_records_sha256": file_hash(records_path),
    }


def _load_model(session: TrainingSession, checkpoint: Path | None, parent: Path) -> Any:
    from peft import PeftModel

    adapter = checkpoint / "adapter" if checkpoint is not None else parent
    model = PeftModel.from_pretrained(session.base_model, adapter, is_trainable=True)
    model.train()
    model.config.use_cache = False
    return model


def _save_checkpoint(
    checkpoint_root: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    records: list[dict[str, Any]],
    completed_steps: int,
    unit_sha256: str,
    parent_adapter_sha256: str,
    root_adapter_sha256: str,
    run_id: str,
    recipe: RopcdExecutionRecipe,
    torch: Any,
    contract: TrainingArtifactContract = P0_ARTIFACT_CONTRACT,
) -> Path:
    final = checkpoint_root / f"step-{completed_steps:04d}"
    if final.exists():
        _verify_checkpoint(
            final,
            unit_sha256=unit_sha256,
            parent_adapter_sha256=parent_adapter_sha256,
            root_adapter_sha256=root_adapter_sha256,
            run_id=run_id,
            recipe=recipe,
            contract=contract,
        )
        return final
    staging = Path(tempfile.mkdtemp(prefix=f".step-{completed_steps:04d}.", dir=checkpoint_root))
    model.save_pretrained(staging / "adapter", safe_serialization=True)
    state_path = staging / "training_state.pt"
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, state_path
    )
    (staging / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    identity = {
        "schema_version": contract.checkpoint_schema_version,
        "run_id": run_id,
        "unit_sha256": unit_sha256,
        "recipe_sha256": json_hash(recipe.to_dict()),
        "parent_adapter_sha256": parent_adapter_sha256,
        "root_adapter_sha256": root_adapter_sha256,
        "completed_steps": completed_steps,
        "adapter_sha256": adapter_bundle_hash(staging / "adapter"),
        "training_state_sha256": file_hash(state_path),
        "records_sha256": file_hash(staging / "records.json"),
    }
    immutable_json_write(staging / "checkpoint.json", identity, f"{contract.label} checkpoint")
    staging.rename(final)
    return final


def _latest_checkpoint(
    checkpoint_root: Path,
    *,
    unit_sha256: str,
    parent_adapter_sha256: str,
    root_adapter_sha256: str,
    run_id: str,
    recipe: RopcdExecutionRecipe,
    contract: TrainingArtifactContract = P0_ARTIFACT_CONTRACT,
) -> Path | None:
    checkpoints = sorted(
        (path for path in checkpoint_root.glob("step-*") if path.is_dir()),
        key=lambda path: path.name,
    )
    for path in checkpoints:
        _verify_checkpoint(
            path,
            unit_sha256=unit_sha256,
            parent_adapter_sha256=parent_adapter_sha256,
            root_adapter_sha256=root_adapter_sha256,
            run_id=run_id,
            recipe=recipe,
            contract=contract,
        )
    return checkpoints[-1] if checkpoints else None


def _verify_checkpoint(
    path: Path,
    *,
    unit_sha256: str,
    parent_adapter_sha256: str,
    root_adapter_sha256: str,
    run_id: str,
    recipe: RopcdExecutionRecipe,
    contract: TrainingArtifactContract = P0_ARTIFACT_CONTRACT,
) -> dict[str, Any]:
    metadata = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    expected = {
        "schema_version": contract.checkpoint_schema_version,
        "run_id": run_id,
        "unit_sha256": unit_sha256,
        "recipe_sha256": json_hash(recipe.to_dict()),
        "parent_adapter_sha256": parent_adapter_sha256,
        "root_adapter_sha256": root_adapter_sha256,
        "completed_steps": int(path.name.removeprefix("step-")),
        "adapter_sha256": adapter_bundle_hash(path / "adapter"),
        "training_state_sha256": file_hash(path / "training_state.pt"),
        "records_sha256": file_hash(path / "records.json"),
    }
    if metadata != expected:
        raise ValueError(f"{contract.label} checkpoint identity changed: {path}")
    records = json.loads((path / "records.json").read_text(encoding="utf-8"))
    if len(records) != expected["completed_steps"]:
        raise ValueError(f"{contract.label} checkpoint record count changed")
    return metadata


def _publish_final_unit(
    final_root: Path,
    *,
    final_checkpoint: Path,
    records: list[dict[str, Any]],
    unit: dict[str, Any],
    parent_adapter_sha256: str,
    root_adapter_sha256: str,
    run_id: str,
    recipe: RopcdExecutionRecipe,
    elapsed_seconds: float,
    training_seed: int,
    contract: TrainingArtifactContract = P0_ARTIFACT_CONTRACT,
) -> None:
    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final_root.name}.", dir=final_root.parent))
    shutil.copytree(final_checkpoint / "adapter", staging / "adapter")
    (staging / "training_records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": contract.training_schema_version,
        "unit_id": unit["unit_id"],
        "unit_sha256": json_hash(unit),
        "run_id": run_id,
        "recipe": recipe.to_dict(),
        "recipe_sha256": json_hash(recipe.to_dict()),
        "parent_unit_id": unit["parent_unit_id"],
        "parent_adapter_sha256": parent_adapter_sha256,
        "root_adapter_sha256": root_adapter_sha256,
        "optimizer_state_lifecycle": "reset_at_semantic_consolidation",
        "completed_steps": recipe.optimizer_steps_per_unit,
        "checkpoint_selection": recipe.checkpoint_selection,
        "adapter_sha256": adapter_bundle_hash(staging / "adapter"),
        "records_sha256": json_hash(records),
        "elapsed_seconds": elapsed_seconds,
        "final_total_loss": float(records[-1]["total_loss"]),
        "final_checkpoint_sha256": file_hash(final_checkpoint / "checkpoint.json"),
    }
    if contract.include_training_seed:
        metadata["training_seed"] = training_seed
    immutable_json_write(
        staging / "training_metadata.json", metadata, f"{contract.label} training"
    )
    staging.rename(final_root)


def _step_seed(training_seed: int, namespace: str, step: int) -> int:
    return int(json_hash([training_seed, namespace, step])[:8], 16)
