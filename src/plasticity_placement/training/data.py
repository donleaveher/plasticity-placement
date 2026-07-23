from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class TokenizerLike(Protocol):
    pad_token_id: int | None
    eos_token_id: int | None

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]: ...


@dataclass(frozen=True, slots=True)
class TrainingExample:
    prompt: str
    completion: str


def load_examples(path: Path) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record.get("prompt")
            completion = record.get("completion")
            if not isinstance(prompt, str) or not isinstance(completion, str):
                raise ValueError(f"line {line_number} must contain string prompt and completion")
            if not prompt or not completion:
                raise ValueError(f"line {line_number} contains an empty prompt or completion")
            examples.append(TrainingExample(prompt=prompt, completion=completion))
    if not examples:
        raise ValueError(f"no training examples found in {path}")
    return examples


class CompletionDataset:
    """Tokenize examples while masking prompt tokens from the causal-LM loss."""

    def __init__(self, examples: list[TrainingExample], tokenizer: TokenizerLike, max_length: int):
        self.rows = [self._encode(example, tokenizer, max_length) for example in examples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]

    @staticmethod
    def _encode(
        example: TrainingExample, tokenizer: TokenizerLike, max_length: int
    ) -> dict[str, list[int]]:
        prompt_ids = tokenizer(example.prompt, add_special_tokens=True)["input_ids"]
        completion_ids = tokenizer(example.completion, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            completion_ids = [*completion_ids, tokenizer.eos_token_id]
        input_ids = (prompt_ids + completion_ids)[:max_length]
        prompt_length = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        if not any(label != -100 for label in labels):
            raise ValueError("max_length leaves no completion token for training")
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


class CausalLMCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(row["input_ids"]) for row in rows)
        input_ids, attention_mask, labels = [], [], []
        for row in rows:
            padding = max_length - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(row["attention_mask"] + [0] * padding)
            labels.append(row["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
