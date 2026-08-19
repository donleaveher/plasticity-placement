from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.p0d2hcrd.config import EXPECTED_MODEL_NAME, EXPECTED_MODEL_REVISION
from plasticity_placement.training.config import LayerBand, LoraTrainingConfig

SPEC_SCHEMA_VERSION = "p0d2hcpr-spec-v1"
CONFIG_SCHEMA_VERSION = "p0d2hcpr-config-v1"
EXPERIMENT_ID = "P1-COMPOSITION-PRESERVING-ROUTE-REMEDIATION-v1"
DATA_GENERATOR_VERSION = "p0d2hcpr-balanced-composition-bank-v1"
AUTHORIZATION_SCOPE = "single_composition_preserving_remediation_pilot"
TRAINING_SEED = 20260802
TASKS = ("route_only", "retrieval_only", "combined")


@dataclass(frozen=True, slots=True)
class DataSpec:
    generator_version: str = DATA_GENERATOR_VERSION
    train_group_count: int = 48
    dev_group_count: int = 12
    examples_per_group: int = 24
    seed: int = TRAINING_SEED

    def __post_init__(self) -> None:
        if (
            self.generator_version != DATA_GENERATOR_VERSION
            or self.train_group_count != 48
            or self.dev_group_count != 12
            or self.examples_per_group != 24
            or self.seed != TRAINING_SEED
        ):
            raise ValueError("CPR-v1 data matrix is frozen")

    @property
    def train_example_count(self) -> int:
        return self.train_group_count * self.examples_per_group

    @property
    def dev_example_count(self) -> int:
        return self.dev_group_count * self.examples_per_group


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    layer_band: str = LayerBand.FULL.value
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    learning_rate: float = 2e-4
    epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    warmup_ratio: float = 0.03
    max_steps: int = 72
    seed: int = TRAINING_SEED
    use_4bit: bool = True
    gradient_checkpointing: bool = True
    use_chat_template: bool = True
    checkpoint_selection: str = "fixed_final_step"
    objective: str = "equal_weight_sft_route_retrieval_combined"

    def __post_init__(self) -> None:
        expected = (
            "full",
            ("q_proj", "v_proj"),
            8,
            16,
            0.0,
            2e-4,
            1,
            4,
            4,
            512,
            0.03,
            72,
            TRAINING_SEED,
            True,
            True,
            True,
            "fixed_final_step",
            "equal_weight_sft_route_retrieval_combined",
        )
        observed = (
            self.layer_band,
            self.target_modules,
            self.rank,
            self.alpha,
            self.dropout,
            self.learning_rate,
            self.epochs,
            self.batch_size,
            self.gradient_accumulation_steps,
            self.max_length,
            self.warmup_ratio,
            self.max_steps,
            self.seed,
            self.use_4bit,
            self.gradient_checkpointing,
            self.use_chat_template,
            self.checkpoint_selection,
            self.objective,
        )
        if observed != expected:
            raise ValueError("CPR-v1 training configuration is frozen")

    def to_lora_config(self, *, data_path: Path, output_dir: Path) -> LoraTrainingConfig:
        return LoraTrainingConfig(
            model_name=EXPECTED_MODEL_NAME,
            model_revision=EXPECTED_MODEL_REVISION,
            data_path=data_path,
            output_dir=output_dir,
            layer_band=LayerBand(self.layer_band),
            target_modules=self.target_modules,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_length=self.max_length,
            warmup_ratio=self.warmup_ratio,
            max_steps=self.max_steps,
            seed=self.seed,
            use_4bit=self.use_4bit,
            gradient_checkpointing=self.gradient_checkpointing,
            use_chat_template=self.use_chat_template,
        )


@dataclass(frozen=True, slots=True)
class GateSpec:
    conditional_route_min_accuracy: float = 0.75
    route_only_min_accuracy_exclusive: float = 0.625
    route_delta_ci_lower_exclusive: float = 0.0
    retrieval_only_min_accuracy: float = 1.0
    combined_noninferiority_margin: float = 0.02
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = TRAINING_SEED
    sentinel_score_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        observed = (
            self.conditional_route_min_accuracy,
            self.route_only_min_accuracy_exclusive,
            self.route_delta_ci_lower_exclusive,
            self.retrieval_only_min_accuracy,
            self.combined_noninferiority_margin,
            self.bootstrap_samples,
            self.bootstrap_seed,
            self.sentinel_score_tolerance,
        )
        expected = (0.75, 0.625, 0.0, 1.0, 0.02, 10_000, TRAINING_SEED, 1e-5)
        if observed != expected:
            raise ValueError("CPR-v1 qualification gates are frozen")


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    schema_version: str = SPEC_SCHEMA_VERSION
    experiment_id: str = EXPERIMENT_ID
    model_name: str = EXPECTED_MODEL_NAME
    model_revision: str = EXPECTED_MODEL_REVISION
    data: DataSpec = DataSpec()
    training: TrainingSpec = TrainingSpec()
    gates: GateSpec = GateSpec()
    training_run_limit: int = 1
    locked_qualification_limit: int = 1
    allows_hyperparameter_search: bool = False
    allows_mappings_per_adapter_scan: bool = False
    allows_kl_weight_scan: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != SPEC_SCHEMA_VERSION or self.experiment_id != EXPERIMENT_ID:
            raise ValueError("unsupported CPR-v1 experiment identity")
        if self.model_name != EXPECTED_MODEL_NAME or self.model_revision != EXPECTED_MODEL_REVISION:
            raise ValueError("CPR-v1 model identity changed")
        if self.training_run_limit != 1 or self.locked_qualification_limit != 1:
            raise ValueError("CPR-v1 permits one training run and one locked qualification")
        if self.allows_hyperparameter_search or self.allows_mappings_per_adapter_scan:
            raise ValueError("CPR-v1 cannot authorize scans")
        if self.allows_kl_weight_scan:
            raise ValueError("CPR-v1 does not authorize a KL-weight scan")

    @classmethod
    def from_path(cls, path: Path) -> ExperimentSpec:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("CPR spec must be a JSON object")
        data = DataSpec(**payload.pop("data", {}))
        training_payload = payload.pop("training", {})
        if "target_modules" in training_payload:
            training_payload["target_modules"] = tuple(training_payload["target_modules"])
        training = TrainingSpec(**training_payload)
        gates = GateSpec(**payload.pop("gates", {}))
        return cls(**payload, data=data, training=training, gates=gates)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["training"]["target_modules"] = list(self.training.target_modules)
        return payload
