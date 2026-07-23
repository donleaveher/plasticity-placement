from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class LayerBand(StrEnum):
    FULL = "full"
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"
    EXPLICIT = "explicit"


@dataclass(frozen=True, slots=True)
class LoraTrainingConfig:
    model_name: str
    data_path: Path
    output_dir: Path
    layer_band: LayerBand = LayerBand.FULL
    explicit_layers: tuple[int, ...] = ()
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    warmup_ratio: float = 0.03
    seed: int = 42
    use_4bit: bool = False
    gradient_checkpointing: bool = True
    resume_adapter: Path | None = None

    def __post_init__(self) -> None:
        positive = {
            "rank": self.rank,
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_length": self.max_length,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.layer_band is LayerBand.EXPLICIT and not self.explicit_layers:
            raise ValueError("explicit layer band requires --layers")
        if self.layer_band is not LayerBand.EXPLICIT and self.explicit_layers:
            raise ValueError("--layers can only be used with layer_band=explicit")
        if len(set(self.explicit_layers)) != len(self.explicit_layers):
            raise ValueError("explicit layers must not contain duplicates")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["data_path"] = str(self.data_path)
        result["output_dir"] = str(self.output_dir)
        result["resume_adapter"] = str(self.resume_adapter) if self.resume_adapter else None
        return result


def select_layers(
    band: LayerBand, num_layers: int, explicit: tuple[int, ...] = ()
) -> list[int] | None:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if band is LayerBand.FULL:
        return None
    if band is LayerBand.EXPLICIT:
        if not explicit:
            raise ValueError("explicit layer selection cannot be empty")
        if min(explicit) < 0 or max(explicit) >= num_layers:
            raise ValueError(f"explicit layers must be within [0, {num_layers - 1}]")
        return sorted(explicit)

    boundaries = (0, num_layers // 3, (2 * num_layers) // 3, num_layers)
    index = {LayerBand.EARLY: 0, LayerBand.MIDDLE: 1, LayerBand.LATE: 2}[band]
    start, stop = boundaries[index], boundaries[index + 1]
    if start == stop:
        raise ValueError(f"model has too few layers for {band.value} selection")
    return list(range(start, stop))
