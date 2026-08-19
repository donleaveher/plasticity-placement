from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    model_name: str
    adapter_path: Path
    probes_path: Path
    output_dir: Path
    max_input_length: int = 512
    max_new_tokens: int = 8
    use_4bit: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_input_length <= 0:
            raise ValueError("max_input_length must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["adapter_path"] = str(self.adapter_path)
        result["probes_path"] = str(self.probes_path)
        result["output_dir"] = str(self.output_dir)
        return result
