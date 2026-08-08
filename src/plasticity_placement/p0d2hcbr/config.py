from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.p0d2hcrd.config import EXPECTED_MODEL_NAME, EXPECTED_MODEL_REVISION
from plasticity_placement.training.config import LayerBand, LoraTrainingConfig

SPEC_SCHEMA_VERSION = "p0d2hcbr-spec-v1"
CONFIG_SCHEMA_VERSION = "p0d2hcbr-config-v1"
EXPERIMENT_ID = "P0-D2H-COUNTERBALANCED-BINDING-REMEDIATION-v1"
DATA_GENERATOR_VERSION = "p0d2hcbr-training-bank-v1"
HELDOUT_BANK_VERSION = "p0d2hcbr-rab-gen-v1"
AUTHORIZATION_SCOPE = "single_frozen_counterbalanced_binding_remediation_matrix"

CURRICULA = ("coupled", "disentangled")
PLACEMENTS = ("full_depth", "late_matched")
TASKS = ("route_only", "retrieval_only", "combined")
TRAINING_SEEDS = (20260810, 20260811, 20260812)


@dataclass(frozen=True, slots=True)
class DataSpec:
    generator_version: str = DATA_GENERATOR_VERSION
    heldout_bank_version: str = HELDOUT_BANK_VERSION
    train_group_count: int = 48
    dev_group_count: int = 12
    examples_per_group: int = 24
    heldout_group_count: int = 24
    heldout_conditions_per_group: int = 64
    seed: int = 20260810

    def __post_init__(self) -> None:
        expected = (
            DATA_GENERATOR_VERSION,
            HELDOUT_BANK_VERSION,
            48,
            12,
            24,
            24,
            64,
            20260810,
        )
        if (
            self.generator_version,
            self.heldout_bank_version,
            self.train_group_count,
            self.dev_group_count,
            self.examples_per_group,
            self.heldout_group_count,
            self.heldout_conditions_per_group,
            self.seed,
        ) != expected:
            raise ValueError("CBR-v1 data matrix is frozen")

    @property
    def train_example_count(self) -> int:
        return self.train_group_count * self.examples_per_group

    @property
    def dev_example_count(self) -> int:
        return self.dev_group_count * self.examples_per_group

    @property
    def heldout_prompt_count(self) -> int:
        return self.heldout_group_count * self.heldout_conditions_per_group


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    full_depth_rank: int = 8
    full_depth_alpha: int = 16
    late_matched_rank: int = 24
    late_matched_alpha: int = 48
    dropout: float = 0.0
    learning_rate: float = 2e-4
    epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    warmup_ratio: float = 0.03
    max_steps: int = 72
    seeds: tuple[int, ...] = TRAINING_SEEDS
    use_4bit: bool = True
    gradient_checkpointing: bool = True
    use_chat_template: bool = True
    checkpoint_selection: str = "fixed_final_step"
    objective: str = "equal_weight_sft_route_retrieval_combined"

    def __post_init__(self) -> None:
        expected = TrainingSpec.__new_defaults__()
        if self != expected:
            raise ValueError("CBR-v1 training configuration is frozen")

    @classmethod
    def __new_defaults__(cls) -> TrainingSpec:
        value = object.__new__(cls)
        for name, field in cls.__dataclass_fields__.items():
            object.__setattr__(value, name, field.default)
        return value

    def to_lora_config(
        self,
        *,
        placement: str,
        seed: int,
        data_path: Path,
        output_dir: Path,
    ) -> LoraTrainingConfig:
        if placement not in PLACEMENTS or seed not in self.seeds:
            raise ValueError("CBR placement or seed is outside the frozen matrix")
        late = placement == "late_matched"
        return LoraTrainingConfig(
            model_name=EXPECTED_MODEL_NAME,
            model_revision=EXPECTED_MODEL_REVISION,
            data_path=data_path,
            output_dir=output_dir,
            layer_band=LayerBand.LATE if late else LayerBand.FULL,
            target_modules=self.target_modules,
            rank=self.late_matched_rank if late else self.full_depth_rank,
            alpha=self.late_matched_alpha if late else self.full_depth_alpha,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_length=self.max_length,
            warmup_ratio=self.warmup_ratio,
            max_steps=self.max_steps,
            seed=seed,
            use_4bit=self.use_4bit,
            gradient_checkpointing=self.gradient_checkpointing,
            use_chat_template=self.use_chat_template,
        )


@dataclass(frozen=True, slots=True)
class GateSpec:
    binding_min_accuracy: float = 0.75
    factor_equivalence_margin: float = 0.05
    placement_noninferiority_margin: float = 0.02
    combined_noninferiority_margin: float = 0.02
    conditional_route_min_accuracy: float = 0.75
    retrieval_only_min_accuracy: float = 1.0
    guardrail_c_to_w_max: float = 0.02
    parameter_budget_relative_tolerance: float = 0.01
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20260810
    confidence: float = 0.95
    sentinel_score_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if self != GateSpec.__new_defaults__():
            raise ValueError("CBR-v1 decision gates are frozen")

    @classmethod
    def __new_defaults__(cls) -> GateSpec:
        value = object.__new__(cls)
        for name, field in cls.__dataclass_fields__.items():
            object.__setattr__(value, name, field.default)
        return value


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    schema_version: str = SPEC_SCHEMA_VERSION
    experiment_id: str = EXPERIMENT_ID
    model_name: str = EXPECTED_MODEL_NAME
    model_revision: str = EXPECTED_MODEL_REVISION
    data: DataSpec = DataSpec()
    training: TrainingSpec = TrainingSpec()
    gates: GateSpec = GateSpec()
    curricula: tuple[str, ...] = CURRICULA
    placements: tuple[str, ...] = PLACEMENTS
    training_run_limit: int = 12
    locked_evaluation_limit: int = 12
    allows_hyperparameter_search: bool = False
    allows_checkpoint_selection: bool = False
    allows_mappings_per_adapter_scan: bool = False
    allows_full_parameter_training: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != SPEC_SCHEMA_VERSION or self.experiment_id != EXPERIMENT_ID:
            raise ValueError("unsupported CBR-v1 experiment identity")
        if self.model_name != EXPECTED_MODEL_NAME or self.model_revision != EXPECTED_MODEL_REVISION:
            raise ValueError("CBR-v1 model identity changed")
        if self.curricula != CURRICULA or self.placements != PLACEMENTS:
            raise ValueError("CBR-v1 condition matrix changed")
        if self.training_run_limit != 12 or self.locked_evaluation_limit != 12:
            raise ValueError("CBR-v1 requires exactly 12 training and evaluation units")
        if any(
            (
                self.allows_hyperparameter_search,
                self.allows_checkpoint_selection,
                self.allows_mappings_per_adapter_scan,
                self.allows_full_parameter_training,
            )
        ):
            raise ValueError("CBR-v1 authorization scope was expanded")

    @classmethod
    def from_path(cls, path: Path) -> ExperimentSpec:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("CBR spec must be a JSON object")
        data = DataSpec(**dict(payload.pop("data", {})))
        training_payload = dict(payload.pop("training", {}))
        for name in ("target_modules", "seeds"):
            if name in training_payload:
                training_payload[name] = tuple(training_payload[name])
        training = TrainingSpec(**training_payload)
        gates = GateSpec(**dict(payload.pop("gates", {})))
        for name in ("curricula", "placements"):
            if name in payload:
                payload[name] = tuple(payload[name])
        return cls(**payload, data=data, training=training, gates=gates)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["training"]["target_modules"] = list(self.training.target_modules)
        payload["training"]["seeds"] = list(self.training.seeds)
        payload["curricula"] = list(self.curricula)
        payload["placements"] = list(self.placements)
        return payload

    def unit_ids(self) -> tuple[str, ...]:
        return tuple(
            unit_id(curriculum, placement, seed)
            for curriculum in self.curricula
            for placement in self.placements
            for seed in self.training.seeds
        )


def unit_id(curriculum: str, placement: str, seed: int) -> str:
    if curriculum not in CURRICULA or placement not in PLACEMENTS or seed not in TRAINING_SEEDS:
        raise ValueError("condition is outside the frozen CBR-v1 matrix")
    return f"{curriculum}__{placement}__seed-{seed}"
