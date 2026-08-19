from __future__ import annotations

import random
from typing import Any


def load_tokenizer(
    model_name: str,
    model_revision: str | None,
) -> Any:
    from transformers import AutoTokenizer

    load_kwargs = {"revision": model_revision} if model_revision else {}
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        **load_kwargs,
    )
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
        raise ValueError("tokenizer must provide an EOS or padding token")
    return tokenizer


def model_load_kwargs(
    torch: Any,
    *,
    use_4bit: bool,
) -> tuple[dict[str, Any], str]:
    kwargs: dict[str, Any] = {"torch_dtype": "auto"}
    if not use_4bit:
        return kwargs, "auto"
    if not torch.cuda.is_available():
        raise RuntimeError("4-bit model execution requires a CUDA runtime")

    from transformers import BitsAndBytesConfig

    compute_dtype, dtype_name = four_bit_compute_dtype(torch)
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    kwargs["device_map"] = {"": 0}
    return kwargs, f"nf4-{dtype_name}"


def four_bit_compute_dtype(torch: Any) -> tuple[Any, str]:
    supports_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    return (torch.bfloat16, "bfloat16") if supports_bf16 else (torch.float16, "float16")


def chat_prompt(tokenizer: Any, prompt: str) -> str:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise ValueError("tokenizer does not provide apply_chat_template")
    return str(
        apply_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def runtime_device(torch: Any) -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
