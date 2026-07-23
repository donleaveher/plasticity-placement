from __future__ import annotations

import gc
import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.compiler import ACTIONS, render_external_prompt
from plasticity_placement.p0c.config import P0CConfig
from plasticity_placement.p0c.domain import Arm, P0CProbe
from plasticity_placement.training.model_utils import (
    chat_prompt,
    load_tokenizer,
    model_load_kwargs,
    runtime_device,
    set_seed,
)


@dataclass(frozen=True, slots=True)
class P0CProbeResult:
    run_id: str
    model: str
    model_revision: str | None
    precision: str
    lesson_id: str
    pair_id: str
    lesson_type: str
    training_seed: int | None
    arm: Arm
    probe_id: str
    category: str
    expected_action: str
    predicted_action: str | None
    correct: bool
    invalid: bool
    generated_text: str
    input_tokens: int
    generated_tokens: int
    latency_seconds: float
    prompt_sha256: str
    adapter_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ModelBundle:
    model: Any
    tokenizer: Any
    torch: Any
    model_revision: str | None
    precision: str


def resolve_model_revision(model_name: str, revision: str | None) -> str | None:
    try:
        from transformers import AutoConfig
    except ImportError as error:
        raise RuntimeError(
            "P0-C model dependencies are missing; run `uv sync --extra train --extra colab`"
        ) from error
    load_kwargs = {"revision": revision} if revision else {}
    model_config = AutoConfig.from_pretrained(model_name, **load_kwargs)
    resolved = getattr(model_config, "_commit_hash", None)
    return str(resolved) if resolved else revision


def load_base_model(config: P0CConfig) -> ModelBundle:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "P0-C model dependencies are missing; run `uv sync --extra train --extra colab`"
        ) from error

    set_seed(config.training_seeds[0], torch)
    load_kwargs = {"revision": config.model_revision} if config.model_revision else {}
    tokenizer = load_tokenizer(
        config.model_name,
        config.model_revision,
    )
    model_kwargs, precision = model_load_kwargs(torch, use_4bit=config.use_4bit)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs,
        **load_kwargs,
    )
    if not config.use_4bit:
        model.to(runtime_device(torch))
        precision = str(next(model.parameters()).dtype).removeprefix("torch.")
    model.eval()
    resolved_revision = getattr(model.config, "_commit_hash", None)
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        model_revision=str(resolved_revision) if resolved_revision else config.model_revision,
        precision=precision,
    )


def load_adapter_model(config: P0CConfig, adapter_path: Path) -> ModelBundle:
    try:
        from peft import PeftModel
    except ImportError as error:
        raise RuntimeError(
            "P0-C adapter dependencies are missing; run `uv sync --extra train`"
        ) from error
    bundle = load_base_model(config)
    bundle.model = PeftModel.from_pretrained(
        bundle.model,
        adapter_path,
        is_trainable=False,
    )
    bundle.model.eval()
    return bundle


def evaluate_probes(
    *,
    bundle: ModelBundle,
    probes: Iterable[P0CProbe],
    arm: Arm,
    run_id: str,
    config: P0CConfig,
    training_seed: int | None,
    external_note: str | None = None,
    adapter_sha256: str | None = None,
) -> list[P0CProbeResult]:
    probe_list = list(probes)
    if not probe_list:
        return []
    results: list[P0CProbeResult] = []
    device = next(bundle.model.parameters()).device
    precision = bundle.precision
    _warm_up_generation(
        bundle=bundle,
        probe=probe_list[0],
        config=config,
        external_note=external_note,
        device=device,
    )
    for probe in probe_list:
        prompt = (
            render_external_prompt(probe, external_note)
            if external_note is not None
            else probe.prompt
        )
        formatted = chat_prompt(bundle.tokenizer, prompt)
        encoded = bundle.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_length,
            add_special_tokens=False,
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        input_length = int(encoded["input_ids"].shape[1])
        if bundle.torch.cuda.is_available():
            bundle.torch.cuda.synchronize()
        started = time.perf_counter()
        with bundle.torch.inference_mode():
            generated = bundle.model.generate(
                **encoded,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=bundle.tokenizer.pad_token_id,
                eos_token_id=bundle.tokenizer.eos_token_id,
            )
        if bundle.torch.cuda.is_available():
            bundle.torch.cuda.synchronize()
        latency = time.perf_counter() - started
        generated_ids = generated[0, input_length:]
        generated_text = bundle.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predicted_action = parse_unique_action(generated_text, ACTIONS)
        results.append(
            P0CProbeResult(
                run_id=run_id,
                model=config.model_name,
                model_revision=bundle.model_revision,
                precision=precision,
                lesson_id=probe.lesson_id,
                pair_id=probe.pair_id,
                lesson_type=probe.lesson_type,
                training_seed=training_seed,
                arm=arm,
                probe_id=probe.probe_id,
                category=probe.category,
                expected_action=probe.expected_action,
                predicted_action=predicted_action,
                correct=predicted_action == probe.expected_action,
                invalid=predicted_action is None,
                generated_text=generated_text,
                input_tokens=input_length,
                generated_tokens=int(generated_ids.shape[0]),
                latency_seconds=latency,
                prompt_sha256=sha256(formatted.encode()).hexdigest(),
                adapter_sha256=adapter_sha256,
            )
        )
    return results


def _warm_up_generation(
    *,
    bundle: ModelBundle,
    probe: P0CProbe,
    config: P0CConfig,
    external_note: str | None,
    device: Any,
) -> None:
    prompt = (
        render_external_prompt(probe, external_note) if external_note is not None else probe.prompt
    )
    formatted = chat_prompt(bundle.tokenizer, prompt)
    encoded = bundle.tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_length,
        add_special_tokens=False,
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    with bundle.torch.inference_mode():
        bundle.model.generate(
            **encoded,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            pad_token_id=bundle.tokenizer.pad_token_id,
            eos_token_id=bundle.tokenizer.eos_token_id,
        )
    if bundle.torch.cuda.is_available():
        bundle.torch.cuda.synchronize()


def parse_unique_action(text: str, allowed_actions: tuple[str, ...]) -> str | None:
    normalized = text.strip().casefold()
    for action in allowed_actions:
        if normalized == action.casefold():
            return action
    return None


def write_probe_results(path: Path, results: Iterable[P0CProbeResult]) -> None:
    write_probe_rows(path, (result.to_dict() for result in results))


def write_probe_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def read_probe_results(path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                results.append(json.loads(line))
    return results


def release_model(bundle: ModelBundle) -> None:
    del bundle.model
    gc.collect()
    if bundle.torch.cuda.is_available():
        bundle.torch.cuda.empty_cache()
