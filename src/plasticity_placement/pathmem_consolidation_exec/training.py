from __future__ import annotations

import json
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.io import file_hash, immutable_json_write, json_hash
from plasticity_placement.pathmem_consolidation_exec.config import (
    G1C_EXECUTION_RECIPE,
    TRAINING_SCHEMA_VERSION,
    RopcdExecutionRecipe,
)
from plasticity_placement.pathmem_exec.artifacts import adapter_bundle_hash
from plasticity_placement.training.model_utils import (
    chat_prompt,
    load_tokenizer,
    model_load_kwargs,
    set_seed,
)


@dataclass(slots=True)
class TrainingSession:
    torch: Any
    functional: Any
    tokenizer: Any
    base_model: Any
    recipe: RopcdExecutionRecipe

    @classmethod
    def load(
        cls, recipe: RopcdExecutionRecipe = G1C_EXECUTION_RECIPE
    ) -> TrainingSession:
        try:
            import torch
            import torch.nn.functional as functional
            from peft import prepare_model_for_kbit_training
            from transformers import AutoModelForCausalLM
        except ImportError as error:
            raise RuntimeError("R-OPCD training requires the train dependencies") from error
        set_seed(recipe.training_seed, torch)
        tokenizer = load_tokenizer(recipe.model.name, recipe.model.revision)
        load_kwargs, precision = model_load_kwargs(torch, use_4bit=True)
        if precision != "nf4-bfloat16":
            raise RuntimeError(f"R-OPCD training precision changed: {precision}")
        base_model = AutoModelForCausalLM.from_pretrained(
            recipe.model.name,
            revision=recipe.model.revision,
            **load_kwargs,
        )
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=recipe.gradient_checkpointing,
        )
        base_model.config.use_cache = False
        return cls(
            torch=torch,
            functional=functional,
            tokenizer=tokenizer,
            base_model=base_model,
            recipe=recipe,
        )

    def release(self) -> None:
        del self.base_model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def unit_slug(unit_id: str) -> str:
    readable = unit_id.replace(":", "__").replace("/", "_")
    return f"{readable}__{json_hash(unit_id)[:10]}"


def train_unit(
    session: TrainingSession,
    *,
    unit: dict[str, Any],
    run_id: str,
    output_root: Path,
) -> dict[str, Any]:
    recipe = session.recipe
    unit_id = str(unit["unit_id"])
    unit_sha256 = json_hash(unit)
    final_root = output_root / "units" / unit_slug(unit_id)
    if final_root.exists():
        return verify_trained_unit(final_root, unit=unit, run_id=run_id, recipe=recipe)

    checkpoint_root = output_root / "work" / unit_slug(unit_id) / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint = _latest_checkpoint(
        checkpoint_root,
        unit_sha256=unit_sha256,
        run_id=run_id,
        recipe=recipe,
    )
    model = _load_side_memory_model(session, checkpoint)
    optimizer = session.torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=recipe.learning_rate,
        betas=recipe.betas,
        eps=recipe.epsilon,
        weight_decay=recipe.weight_decay,
    )
    scheduler = session.torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _: 1.0,
    )
    records: list[dict[str, Any]] = []
    start_step = 0
    if checkpoint is not None:
        state = session.torch.load(
            checkpoint / "training_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        metadata = json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))
        start_step = int(metadata["completed_steps"])
        records = json.loads((checkpoint / "records.json").read_text(encoding="utf-8"))
        if len(records) != start_step:
            raise ValueError("R-OPCD checkpoint record count differs from its step")

    started = time.perf_counter()
    try:
        for step in range(start_step, recipe.optimizer_steps_per_unit):
            pair_index = step % recipe.prompt_pairs_per_unit
            pair = unit["rollout_prompt_pairs"][pair_index]
            step_seed = _step_seed(recipe.training_seed, unit_id, step)
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
            _require_finite_tensor(session.torch, loss, "distillation loss")
            loss.backward()
            gradient_norm = session.torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                recipe.max_grad_norm,
            )
            _require_finite_tensor(session.torch, gradient_norm, "gradient norm")
            optimizer.step()
            scheduler.step()
            records.append(
                {
                    **record,
                    "gradient_norm": float(gradient_norm.detach().float().cpu()),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
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
                    run_id=run_id,
                    recipe=recipe,
                    torch=session.torch,
                )

        final_checkpoint = _latest_checkpoint(
            checkpoint_root,
            unit_sha256=unit_sha256,
            run_id=run_id,
            recipe=recipe,
        )
        if final_checkpoint is None:
            raise RuntimeError("R-OPCD final checkpoint is missing")
        final_metadata = json.loads(
            (final_checkpoint / "checkpoint.json").read_text(encoding="utf-8")
        )
        if final_metadata["completed_steps"] != recipe.optimizer_steps_per_unit:
            raise RuntimeError("R-OPCD fixed final checkpoint was not reached")
        _publish_final_unit(
            final_root,
            final_checkpoint=final_checkpoint,
            records=records,
            unit=unit,
            unit_sha256=unit_sha256,
            run_id=run_id,
            recipe=recipe,
            elapsed_seconds=time.perf_counter() - started,
        )
    finally:
        session.base_model = model.unload()
        session.base_model.config.use_cache = False
        if session.torch.cuda.is_available():
            session.torch.cuda.empty_cache()
    return verify_trained_unit(final_root, unit=unit, run_id=run_id, recipe=recipe)


def verify_trained_unit(
    final_root: Path,
    *,
    unit: dict[str, Any],
    run_id: str,
    recipe: RopcdExecutionRecipe = G1C_EXECUTION_RECIPE,
) -> dict[str, Any]:
    expected_files = {"adapter", "training_metadata.json", "training_records.json"}
    observed = {path.name for path in final_root.iterdir()}
    if observed != expected_files:
        raise ValueError(
            f"trained unit file set changed: missing={sorted(expected_files - observed)} "
            f"extra={sorted(observed - expected_files)}"
        )
    metadata = json.loads(
        (final_root / "training_metadata.json").read_text(encoding="utf-8")
    )
    records = json.loads(
        (final_root / "training_records.json").read_text(encoding="utf-8")
    )
    adapter_sha256 = adapter_bundle_hash(final_root / "adapter")
    expected = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "unit_id": unit["unit_id"],
        "unit_sha256": json_hash(unit),
        "run_id": run_id,
        "recipe": recipe.to_dict(),
        "recipe_sha256": json_hash(recipe.to_dict()),
        "completed_steps": recipe.optimizer_steps_per_unit,
        "checkpoint_selection": recipe.checkpoint_selection,
        "adapter_sha256": adapter_sha256,
        "records_sha256": json_hash(records),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"trained unit metadata changed: {key}")
    if len(records) != recipe.optimizer_steps_per_unit:
        raise ValueError("trained unit does not contain the exact step record count")
    if [int(record["step"]) for record in records] != list(
        range(recipe.optimizer_steps_per_unit)
    ):
        raise ValueError("trained unit step sequence changed")
    return {
        **metadata,
        "adapter_dir": str((final_root / "adapter").resolve()),
        "training_metadata_sha256": file_hash(final_root / "training_metadata.json"),
        "training_records_sha256": file_hash(final_root / "training_records.json"),
    }


def _load_side_memory_model(
    session: TrainingSession, checkpoint: Path | None
) -> Any:
    from peft import LoraConfig, PeftModel, get_peft_model

    if checkpoint is not None:
        model = PeftModel.from_pretrained(
            session.base_model,
            checkpoint / "adapter",
            is_trainable=True,
        )
    else:
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
            ),
        )
    model.train()
    model.config.use_cache = False
    return model


def _distillation_loss(
    session: TrainingSession,
    *,
    model: Any,
    unit: dict[str, Any],
    pair: dict[str, Any],
    step: int,
    step_seed: int,
) -> tuple[Any, dict[str, Any]]:
    torch = session.torch
    functional = session.functional
    recipe = session.recipe
    device = next(model.parameters()).device
    student_prompt = chat_prompt(session.tokenizer, str(pair["student_prompt"]))
    teacher_prompt = chat_prompt(
        session.tokenizer, str(pair["privileged_teacher_prompt"])
    )
    student_ids = _token_ids(session.tokenizer, student_prompt)
    teacher_ids = _token_ids(session.tokenizer, teacher_prompt)
    input_ids = torch.tensor([student_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=recipe.rollout_do_sample,
            temperature=recipe.rollout_temperature,
            top_p=recipe.rollout_top_p,
            min_new_tokens=recipe.rollout_min_new_tokens,
            max_new_tokens=recipe.rollout_max_new_tokens,
            pad_token_id=int(session.tokenizer.pad_token_id),
            eos_token_id=int(session.tokenizer.eos_token_id),
        )
    response_ids = [int(value) for value in generated[0, len(student_ids) :].tolist()]
    if not response_ids:
        raise RuntimeError("R-OPCD student rollout produced no response tokens")
    if (
        len(student_ids) + len(response_ids) > recipe.training_max_length
        or len(teacher_ids) + len(response_ids) > recipe.training_max_length
    ):
        raise RuntimeError("R-OPCD on-policy trajectory exceeds the frozen length")
    model.train()

    student_sequence = torch.tensor(
        [student_ids + response_ids], dtype=torch.long, device=device
    )
    student_logits = model(input_ids=student_sequence).logits[
        :, len(student_ids) - 1 : -1, :
    ]
    with model.disable_adapter(), torch.no_grad():
        teacher_sequence = torch.tensor(
            [teacher_ids + response_ids], dtype=torch.long, device=device
        )
        teacher_logits = model(input_ids=teacher_sequence).logits[
            :, len(teacher_ids) - 1 : -1, :
        ]
    trajectory_kl = _forward_kl(
        student_logits,
        teacher_logits,
        temperature=recipe.distillation_temperature,
        functional=functional,
    )
    action_choices = tuple(str(value) for value in pair["action_choices"])
    student_scores = _candidate_sequence_scores(
        session,
        model=model,
        formatted_prompt=student_prompt,
        action_choices=action_choices,
    )
    with model.disable_adapter(), torch.no_grad():
        teacher_scores = _candidate_sequence_scores(
            session,
            model=model,
            formatted_prompt=teacher_prompt,
            action_choices=action_choices,
        )
    temperature = recipe.distillation_temperature
    teacher_log_distribution = functional.log_softmax(
        teacher_scores.float() / temperature, dim=-1
    )
    teacher_distribution = teacher_log_distribution.exp()
    student_log_distribution = functional.log_softmax(
        student_scores.float() / temperature, dim=-1
    )
    candidate_kl = (
        teacher_distribution * (teacher_log_distribution - student_log_distribution)
    ).sum() * (temperature**2)
    total = (
        recipe.trajectory_kl_weight * trajectory_kl
        + recipe.candidate_kl_weight * candidate_kl
    )
    record = {
        "step": step,
        "step_seed": step_seed,
        "pair_index": step % recipe.prompt_pairs_per_unit,
        "probe_id": pair["probe_id"],
        "student_prompt_sha256": json_hash(pair["student_prompt"]),
        "teacher_prompt_sha256": json_hash(pair["privileged_teacher_prompt"]),
        "response_token_ids": response_ids,
        "response_text": session.tokenizer.decode(
            response_ids, skip_special_tokens=False
        ),
        "trajectory_kl": float(trajectory_kl.detach().float().cpu()),
        "candidate_kl": float(candidate_kl.detach().float().cpu()),
        "total_loss": float(total.detach().float().cpu()),
        "student_candidate_scores": [
            float(value) for value in student_scores.detach().float().cpu().tolist()
        ],
        "teacher_candidate_scores": [
            float(value) for value in teacher_scores.detach().float().cpu().tolist()
        ],
        "current_action": unit["current_action"],
        "obsolete_action": unit["obsolete_action"],
    }
    return total, record


def _forward_kl(
    student_logits: Any,
    teacher_logits: Any,
    *,
    temperature: float,
    functional: Any,
) -> Any:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher/student on-policy logit shapes differ")
    teacher_log = functional.log_softmax(teacher_logits.float() / temperature, dim=-1)
    teacher_probability = teacher_log.exp()
    student_log = functional.log_softmax(student_logits.float() / temperature, dim=-1)
    return (
        teacher_probability * (teacher_log - student_log)
    ).sum(dim=-1).mean() * (temperature**2)


def _candidate_sequence_scores(
    session: TrainingSession,
    *,
    model: Any,
    formatted_prompt: str,
    action_choices: tuple[str, str, str, str],
) -> Any:
    torch = session.torch
    device = next(model.parameters()).device
    prompt_ids = _token_ids(session.tokenizer, formatted_prompt)
    candidate_ids = [
        _token_ids(
            session.tokenizer,
            f"{session.recipe.candidate_leading_whitespace}{action}",
        )
        for action in action_choices
    ]
    if any(not values for values in candidate_ids):
        raise ValueError("R-OPCD action tokenization is empty")
    sequences = [prompt_ids + values for values in candidate_ids]
    if max(map(len, sequences)) > session.recipe.training_max_length:
        raise RuntimeError("R-OPCD candidate sequence exceeds the frozen length")
    maximum = max(map(len, sequences))
    pad_id = int(session.tokenizer.pad_token_id)
    input_ids = torch.tensor(
        [sequence + [pad_id] * (maximum - len(sequence)) for sequence in sequences],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.tensor(
        [[1] * len(sequence) + [0] * (maximum - len(sequence)) for sequence in sequences],
        dtype=torch.long,
        device=device,
    )
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    scores: list[Any] = []
    for index, values in enumerate(candidate_ids):
        prediction_logits = logits[
            index,
            len(prompt_ids) - 1 : len(prompt_ids) + len(values) - 1,
            :,
        ]
        targets = torch.tensor(values, dtype=torch.long, device=device)
        token_log_probabilities = session.functional.log_softmax(
            prediction_logits.float(), dim=-1
        ).gather(1, targets.unsqueeze(1)).squeeze(1)
        scores.append(token_log_probabilities.sum())
    return torch.stack(scores)


def _save_checkpoint(
    checkpoint_root: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    records: list[dict[str, Any]],
    completed_steps: int,
    unit_sha256: str,
    run_id: str,
    recipe: RopcdExecutionRecipe,
    torch: Any,
) -> Path:
    final = checkpoint_root / f"step-{completed_steps:04d}"
    if final.exists():
        _verify_checkpoint(
            final,
            unit_sha256=unit_sha256,
            run_id=run_id,
            recipe=recipe,
        )
        return final
    staging = Path(
        tempfile.mkdtemp(prefix=f".step-{completed_steps:04d}.", dir=checkpoint_root)
    )
    adapter_dir = staging / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    state_path = staging / "training_state.pt"
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        state_path,
    )
    (staging / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    identity = {
        "schema_version": "pathmem-g1c-r-opcd-checkpoint-v1",
        "run_id": run_id,
        "unit_sha256": unit_sha256,
        "recipe_sha256": json_hash(recipe.to_dict()),
        "completed_steps": completed_steps,
        "adapter_sha256": adapter_bundle_hash(adapter_dir),
        "training_state_sha256": file_hash(state_path),
        "records_sha256": file_hash(staging / "records.json"),
    }
    immutable_json_write(staging / "checkpoint.json", identity, "R-OPCD checkpoint")
    staging.rename(final)
    return final


def _latest_checkpoint(
    checkpoint_root: Path,
    *,
    unit_sha256: str,
    run_id: str,
    recipe: RopcdExecutionRecipe,
) -> Path | None:
    checkpoints = sorted(
        (path for path in checkpoint_root.glob("step-*") if path.is_dir()),
        key=lambda path: path.name,
    )
    for path in checkpoints:
        _verify_checkpoint(
            path,
            unit_sha256=unit_sha256,
            run_id=run_id,
            recipe=recipe,
        )
    return checkpoints[-1] if checkpoints else None


def _verify_checkpoint(
    path: Path,
    *,
    unit_sha256: str,
    run_id: str,
    recipe: RopcdExecutionRecipe,
) -> dict[str, Any]:
    metadata = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    expected = {
        "schema_version": "pathmem-g1c-r-opcd-checkpoint-v1",
        "run_id": run_id,
        "unit_sha256": unit_sha256,
        "recipe_sha256": json_hash(recipe.to_dict()),
        "completed_steps": int(path.name.removeprefix("step-")),
        "adapter_sha256": adapter_bundle_hash(path / "adapter"),
        "training_state_sha256": file_hash(path / "training_state.pt"),
        "records_sha256": file_hash(path / "records.json"),
    }
    if metadata != expected:
        raise ValueError(f"R-OPCD checkpoint identity changed: {path}")
    records = json.loads((path / "records.json").read_text(encoding="utf-8"))
    if len(records) != expected["completed_steps"]:
        raise ValueError("R-OPCD checkpoint record count changed")
    return metadata


def _publish_final_unit(
    final_root: Path,
    *,
    final_checkpoint: Path,
    records: list[dict[str, Any]],
    unit: dict[str, Any],
    unit_sha256: str,
    run_id: str,
    recipe: RopcdExecutionRecipe,
    elapsed_seconds: float,
) -> None:
    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{final_root.name}.", dir=final_root.parent)
    )
    shutil.copytree(final_checkpoint / "adapter", staging / "adapter")
    (staging / "training_records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "unit_id": unit["unit_id"],
        "unit_sha256": unit_sha256,
        "run_id": run_id,
        "recipe": recipe.to_dict(),
        "recipe_sha256": json_hash(recipe.to_dict()),
        "completed_steps": recipe.optimizer_steps_per_unit,
        "checkpoint_selection": recipe.checkpoint_selection,
        "adapter_sha256": adapter_bundle_hash(staging / "adapter"),
        "records_sha256": json_hash(records),
        "elapsed_seconds": elapsed_seconds,
        "final_total_loss": float(records[-1]["total_loss"]),
        "final_checkpoint_sha256": file_hash(final_checkpoint / "checkpoint.json"),
    }
    immutable_json_write(
        staging / "training_metadata.json", metadata, "R-OPCD training metadata"
    )
    staging.rename(final_root)


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, truncation=False, add_special_tokens=False)
    values = encoded["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _step_seed(training_seed: int, unit_id: str, step: int) -> int:
    return int(json_hash([training_seed, unit_id, step])[:8], 16)


def _require_finite_tensor(torch: Any, value: Any, label: str) -> None:
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise FloatingPointError(f"R-OPCD {label} is non-finite")
