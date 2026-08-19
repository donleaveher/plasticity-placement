from pathlib import Path

import pytest

from plasticity_placement.training.config import LayerBand, LoraTrainingConfig, select_layers
from plasticity_placement.training.lora import _adapter_bundle_files
from plasticity_placement.training.model_utils import four_bit_compute_dtype


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


def test_max_steps_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        LoraTrainingConfig(
            model_name="model",
            data_path=tmp_path / "data.jsonl",
            output_dir=tmp_path / "adapter",
            max_steps=0,
        )


def test_adapter_bundle_excludes_tokenizer_files(tmp_path: Path) -> None:
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_bytes(b"large-tokenizer")
    assert {path.name for path in _adapter_bundle_files(tmp_path)} == {
        "adapter_model.safetensors",
        "adapter_config.json",
    }


class _FakeCuda:
    def __init__(self, supports_bf16: bool) -> None:
        self.supports_bf16 = supports_bf16

    def is_bf16_supported(self) -> bool:
        return self.supports_bf16


class _FakeTorch:
    bfloat16 = "bf16"
    float16 = "fp16"

    def __init__(self, supports_bf16: bool) -> None:
        self.cuda = _FakeCuda(supports_bf16)


def test_four_bit_dtype_falls_back_to_fp16() -> None:
    assert four_bit_compute_dtype(_FakeTorch(False)) == ("fp16", "float16")
    assert four_bit_compute_dtype(_FakeTorch(True)) == ("bf16", "bfloat16")
