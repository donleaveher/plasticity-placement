from pathlib import Path

import pytest

from plasticity_placement.training.data import CompletionDataset, TrainingExample, load_examples


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text: str, **_kwargs: object) -> dict[str, list[int]]:
        return {"input_ids": [ord(character) % 31 + 3 for character in text]}


def test_dataset_masks_prompt_tokens() -> None:
    dataset = CompletionDataset([TrainingExample("ab", "c")], FakeTokenizer(), max_length=8)
    row = dataset[0]
    assert row["labels"][:2] == [-100, -100]
    assert row["labels"][2:] == row["input_ids"][2:]


def test_loader_rejects_invalid_records(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"prompt": "missing completion"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_examples(path)
