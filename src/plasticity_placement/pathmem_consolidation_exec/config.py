from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

EXECUTION_SCHEMA_VERSION = "pathmem-g1c-r-opcd-execution-v1"
AUTHORIZATION_SCHEMA_VERSION = "pathmem-g1c-r-opcd-authorization-v1"
MANIFEST_SCHEMA_VERSION = "pathmem-g1c-r-opcd-manifest-v1"
PREFLIGHT_SCHEMA_VERSION = "pathmem-g1c-r-opcd-preflight-v1"
TRAINING_SCHEMA_VERSION = "pathmem-g1c-r-opcd-training-v1"
EVALUATION_SCHEMA_VERSION = "pathmem-g1c-r-opcd-evaluation-v1"
SUMMARY_SCHEMA_VERSION = "pathmem-g1c-r-opcd-summary-v1"
RECIPE_ID = "routed_on_policy_context_distillation_v1-recipe-r1"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    revision: str = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    role: str = "g1c_parameter_consolidation_qualification"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RopcdExecutionRecipe:
    recipe_id: str = RECIPE_ID
    model: ModelSpec = ModelSpec()
    quantization: str = "nf4_double_quant"
    compute_dtype: str = "bfloat16"
    side_memory_kind: str = "isolated_full_layer_qv_lora"
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: tuple[str, str] = ("q_proj", "v_proj")
    bias: str = "none"
    optimizer: str = "AdamW"
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    scheduler: str = "constant"
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    optimizer_steps_per_unit: int = 48
    checkpoint_interval_steps: int = 8
    prompt_pairs_per_unit: int = 4
    rollouts_per_step: int = 1
    rollout_do_sample: bool = True
    rollout_temperature: float = 0.8
    rollout_top_p: float = 0.95
    rollout_min_new_tokens: int = 1
    rollout_max_new_tokens: int = 8
    candidate_leading_whitespace: str = ""
    distillation_temperature: float = 1.0
    trajectory_kl_weight: float = 0.5
    candidate_kl_weight: float = 0.5
    training_max_length: int = 512
    evaluation_max_length: int = 512
    gradient_checkpointing: bool = True
    training_seed: int = 41
    panel_seed: int = 20260814
    execution_order_seed: int = 20260817
    checkpoint_format: str = "safetensors_plus_optimizer_state"
    checkpoint_selection: str = "fixed_final_step"

    def __post_init__(self) -> None:
        expected = RopcdExecutionRecipe.__dataclass_fields__
        if self.recipe_id != RECIPE_ID or not expected:
            raise ValueError("unregistered R-OPCD recipe")
        if self.quantization != "nf4_double_quant" or self.compute_dtype != "bfloat16":
            raise ValueError("R-OPCD requires frozen NF4/BF16 precision")
        if (
            self.side_memory_kind != "isolated_full_layer_qv_lora"
            or self.rank != 16
            or self.alpha != 32
            or self.dropout != 0.0
            or self.target_modules != ("q_proj", "v_proj")
            or self.bias != "none"
        ):
            raise ValueError("R-OPCD side-memory parameterization changed")
        if (
            self.optimizer != "AdamW"
            or self.learning_rate != 1e-4
            or self.betas != (0.9, 0.999)
            or self.epsilon != 1e-8
            or self.weight_decay != 0.0
            or self.scheduler != "constant"
            or self.warmup_steps != 0
            or self.max_grad_norm != 1.0
        ):
            raise ValueError("R-OPCD optimizer recipe changed")
        if (
            self.optimizer_steps_per_unit != 48
            or self.checkpoint_interval_steps != 8
            or self.prompt_pairs_per_unit != 4
            or self.rollouts_per_step != 1
        ):
            raise ValueError("R-OPCD exposure or checkpoint cadence changed")
        if self.optimizer_steps_per_unit % self.checkpoint_interval_steps:
            raise ValueError("checkpoint interval must divide the frozen step count")
        if (
            not self.rollout_do_sample
            or self.rollout_temperature != 0.8
            or self.rollout_top_p != 0.95
            or self.rollout_min_new_tokens != 1
            or self.rollout_max_new_tokens != 8
            or self.candidate_leading_whitespace != ""
        ):
            raise ValueError("R-OPCD rollout recipe changed")
        if (
            self.distillation_temperature != 1.0
            or self.trajectory_kl_weight != 0.5
            or self.candidate_kl_weight != 0.5
            or self.trajectory_kl_weight + self.candidate_kl_weight != 1.0
        ):
            raise ValueError("R-OPCD distillation loss changed")
        if self.training_max_length != 512 or self.evaluation_max_length != 512:
            raise ValueError("R-OPCD length contract changed")
        if self.checkpoint_selection != "fixed_final_step":
            raise ValueError("R-OPCD cannot select a checkpoint from outcomes")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_modules"] = list(self.target_modules)
        payload["betas"] = list(self.betas)
        return payload


G1C_EXECUTION_RECIPE = RopcdExecutionRecipe()
