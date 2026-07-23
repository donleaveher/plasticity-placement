from pathlib import Path

import pytest

from plasticity_placement.training.config import LayerBand, LoraTrainingConfig, select_layers


def test_layer_bands_partition_model() -> None:
    assert select_layers(LayerBand.EARLY, 12) == [0, 1, 2, 3]
    assert select_layers(LayerBand.MIDDLE, 12) == [4, 5, 6, 7]
    assert select_layers(LayerBand.LATE, 12) == [8, 9, 10, 11]
    assert select_layers(LayerBand.FULL, 12) is None


def test_explicit_layers_are_sorted_and_validated() -> None:
    assert select_layers(LayerBand.EXPLICIT, 12, (7, 4)) == [4, 7]
    with pytest.raises(ValueError):
        select_layers(LayerBand.EXPLICIT, 12, (12,))


def test_explicit_config_requires_layers(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        LoraTrainingConfig(
            model_name="model",
            data_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "adapter",
            layer_band=LayerBand.EXPLICIT,
        )
