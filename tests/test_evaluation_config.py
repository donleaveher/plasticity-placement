from pathlib import Path

import pytest

from plasticity_placement.evaluation.config import EvaluationConfig


def test_evaluation_lengths_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        EvaluationConfig(
            model_name="model",
            adapter_path=tmp_path / "adapter",
            probes_path=tmp_path / "probes.jsonl",
            output_dir=tmp_path / "results",
            max_new_tokens=0,
        )
