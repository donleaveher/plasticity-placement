from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

EXECUTION_SCHEMA_VERSION = "pathmem-execution-v1"
G1_SCHEMA_VERSION = "pathmem-g1-qualification-v1"
P0_SCHEMA_VERSION = "pathmem-p0-smoke-v1"
G1_AUTHORIZATION_VERSION = "pathmem-g1-authorization-v1"
TRAINING_SEED = 41
PANEL_SEED = 20260814
EXECUTION_ORDER_SEED = 20260815


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    revision: str
    role: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResetLoraRecipe:
    quantization: str = "nf4_double_quant"
    compute_dtype: str = "bfloat16"
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, str] = ("q_proj", "v_proj")
    learning_rate: float = 2e-4
    optimizer: str = "AdamW"
    weight_decay: float = 0.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    optimizer_steps_per_event: int = 16
    training_max_length: int = 256
    evaluation_max_length: int = 512
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        if self.quantization != "nf4_double_quant":
            raise ValueError("PathMem execution requires frozen NF4 double quantization")
        if self.compute_dtype != "bfloat16":
            raise ValueError("PathMem execution requires resolved BF16 compute")
        if self.optimizer != "AdamW" or self.weight_decay != 0.0:
            raise ValueError("PathMem reset-at-event recipe requires AdamW without decay")
        if (
            self.rank != 8
            or self.alpha != 16
            or self.dropout != 0.0
            or self.target_modules != ("q_proj", "v_proj")
        ):
            raise ValueError("PathMem LoRA structure differs from the frozen recipe")
        if (
            self.batch_size != 1
            or self.gradient_accumulation_steps != 1
            or self.optimizer_steps_per_event != 16
        ):
            raise ValueError("PathMem event-update budget differs from the frozen recipe")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_modules"] = list(self.target_modules)
        return payload


@dataclass(frozen=True, slots=True)
class PhaseConfig:
    phase: str
    split: str
    model: ModelSpec
    training_seeds: tuple[int, ...]
    recipe: ResetLoraRecipe = ResetLoraRecipe()
    panel_seed: int = PANEL_SEED
    execution_order_seed: int = EXECUTION_ORDER_SEED

    def __post_init__(self) -> None:
        if self.phase not in {"G1", "P0"}:
            raise ValueError(f"unsupported executable phase: {self.phase}")
        expected_split = {"G1": "interface_dev", "P0": "smoke"}[self.phase]
        if self.split != expected_split:
            raise ValueError(f"{self.phase} requires split={expected_split}")
        if not self.training_seeds or len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("training seeds must be non-empty and unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_schema_version": EXECUTION_SCHEMA_VERSION,
            "phase": self.phase,
            "split": self.split,
            "model": self.model.to_dict(),
            "training_seeds": list(self.training_seeds),
            "recipe": self.recipe.to_dict(),
            "panel_seed": self.panel_seed,
            "execution_order_seed": self.execution_order_seed,
        }


G1_CONFIG = PhaseConfig(
    phase="G1",
    split="interface_dev",
    model=ModelSpec(
        name="Qwen/Qwen2.5-1.5B-Instruct",
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        role="interface_qualification",
    ),
    training_seeds=(TRAINING_SEED,),
)

P0_CONFIG = PhaseConfig(
    phase="P0",
    split="smoke",
    model=ModelSpec(
        name="Qwen/Qwen2.5-0.5B-Instruct",
        revision="7ae557604adf67be50417f59c2c2f167def9a775",
        role="engineering_smoke_only",
    ),
    training_seeds=(TRAINING_SEED,),
)

P0_DUPLICATE_NODE_BY_ITEM_INDEX = ("ABA", "BAA", "BAB", "ABB")
