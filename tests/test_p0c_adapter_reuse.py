from __future__ import annotations

from pathlib import Path

import pytest

from plasticity_placement.p0c.modeling import (
    ModelBundle,
    activate_adapter,
    deactivate_adapter,
)
from plasticity_placement.p0d2h.runtime import _require_cuda_bundle


@pytest.mark.filterwarnings("ignore:fan_in_fan_out is set to False")
@pytest.mark.filterwarnings("ignore:Could not find a config file")
def test_adapter_can_be_replaced_without_reloading_base_model(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    config = transformers.GPT2Config(
        n_layer=1,
        n_head=1,
        n_embd=8,
        n_positions=16,
        vocab_size=32,
        bos_token_id=1,
        eos_token_id=2,
    )
    adapter_source = peft.get_peft_model(
        transformers.GPT2LMHeadModel(config),
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["c_attn"],
            task_type="CAUSAL_LM",
        ),
    )
    adapter_dir = tmp_path / "adapter"
    adapter_source.save_pretrained(adapter_dir)

    base_model = transformers.GPT2LMHeadModel(config)
    base_identity = id(base_model)
    bundle = ModelBundle(
        model=base_model,
        tokenizer=None,
        torch=torch,
        model_revision="test-revision",
        precision="float32",
    )

    activate_adapter(bundle, adapter_dir, adapter_name="active")
    assert id(bundle.model.get_base_model()) == base_identity
    assert set(bundle.model.peft_config) == {"active"}
    deactivate_adapter(bundle, adapter_name="active")
    assert id(bundle.model.base_model.model) == base_identity
    assert bundle.model.peft_config == {}

    activate_adapter(bundle, adapter_dir, adapter_name="active")
    assert id(bundle.model.get_base_model()) == base_identity
    assert set(bundle.model.peft_config) == {"active"}
    deactivate_adapter(bundle, adapter_name="active")


def test_hard_probe_rejects_a_cpu_model_bundle() -> None:
    torch = pytest.importorskip("torch")
    bundle = ModelBundle(
        model=torch.nn.Linear(2, 2),
        tokenizer=None,
        torch=torch,
        model_revision="test-revision",
        precision="float32",
    )
    with pytest.raises(RuntimeError, match="expected a CUDA device"):
        _require_cuda_bundle(bundle)
