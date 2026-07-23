from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.evaluation.config import EvaluationConfig
from plasticity_placement.evaluation.probes import Probe, load_probes, parse_action


@dataclass(frozen=True, slots=True)
class ProbeResult:
    phase: str
    probe_id: str
    category: str
    expected_action: str
    predicted_action: str | None
    correct: bool
    generated_text: str
    latency_seconds: float
    input_tokens: int
    generated_tokens: int


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    output_dir: str
    probe_count: int
    phase_accuracy: dict[str, float]
    category_accuracy: dict[str, dict[str, float]]
    rollback_exact_match_rate: float


def evaluate_lora(config: EvaluationConfig) -> EvaluationSummary:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "evaluation dependencies are missing; run `uv sync --extra train`"
        ) from error

    _set_seed(config.seed, torch)
    probes = load_probes(config.probes_path)
    allowed_actions = tuple(sorted({probe.expected_action for probe in probes}))
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must provide an EOS or padding token")

    model_kwargs: dict[str, Any] = {"torch_dtype": "auto"}
    if config.use_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit evaluation requires a CUDA runtime")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": 0}

    base_model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    if not config.use_4bit:
        base_model.to(_device(torch))
    base_model.eval()

    base_results = _run_phase(
        phase="base",
        model=base_model,
        tokenizer=tokenizer,
        probes=probes,
        allowed_actions=allowed_actions,
        config=config,
        torch=torch,
    )

    model = PeftModel.from_pretrained(base_model, config.adapter_path, is_trainable=False)
    model.eval()
    adapter_results = _run_phase(
        phase="adapter",
        model=model,
        tokenizer=tokenizer,
        probes=probes,
        allowed_actions=allowed_actions,
        config=config,
        torch=torch,
    )
    with model.disable_adapter():
        rollback_results = _run_phase(
            phase="rollback",
            model=model,
            tokenizer=tokenizer,
            probes=probes,
            allowed_actions=allowed_actions,
            config=config,
            torch=torch,
        )

    results = [*base_results, *adapter_results, *rollback_results]
    summary = _summarize(results, base_results, rollback_results, config.output_dir)
    _write_results(config, results, summary)
    return summary


def _run_phase(
    phase: str,
    model: Any,
    tokenizer: Any,
    probes: list[Probe],
    allowed_actions: tuple[str, ...],
    config: EvaluationConfig,
    torch: Any,
) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    device = next(model.parameters()).device
    for probe in probes:
        encoded = tokenizer(
            probe.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_input_length,
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        input_length = encoded["input_ids"].shape[1]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - started
        generated_ids = generated[0, input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predicted_action = parse_action(generated_text, allowed_actions)
        results.append(
            ProbeResult(
                phase=phase,
                probe_id=probe.probe_id,
                category=probe.category,
                expected_action=probe.expected_action,
                predicted_action=predicted_action,
                correct=predicted_action == probe.expected_action,
                generated_text=generated_text,
                latency_seconds=latency,
                input_tokens=input_length,
                generated_tokens=int(generated_ids.shape[0]),
            )
        )
    return results


def _summarize(
    results: list[ProbeResult],
    base_results: list[ProbeResult],
    rollback_results: list[ProbeResult],
    output_dir: Path,
) -> EvaluationSummary:
    phase_groups: dict[str, list[ProbeResult]] = defaultdict(list)
    category_groups: dict[str, dict[str, list[ProbeResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for result in results:
        phase_groups[result.phase].append(result)
        category_groups[result.phase][result.category].append(result)

    phase_accuracy = {
        phase: _accuracy(group) for phase, group in sorted(phase_groups.items())
    }
    category_accuracy = {
        phase: {
            category: _accuracy(group)
            for category, group in sorted(categories.items())
        }
        for phase, categories in sorted(category_groups.items())
    }
    exact_matches = sum(
        base.generated_text == rollback.generated_text
        and base.predicted_action == rollback.predicted_action
        for base, rollback in zip(base_results, rollback_results, strict=True)
    )
    return EvaluationSummary(
        output_dir=str(output_dir),
        probe_count=len(base_results),
        phase_accuracy=phase_accuracy,
        category_accuracy=category_accuracy,
        rollback_exact_match_rate=exact_matches / len(base_results),
    )


def _write_results(
    config: EvaluationConfig,
    results: list[ProbeResult],
    summary: EvaluationSummary,
) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with (config.output_dir / "probe_results.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True) + "\n")
    report = {"config": config.to_dict(), "summary": asdict(summary)}
    (config.output_dir / "evaluation_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _accuracy(results: list[ProbeResult]) -> float:
    return sum(result.correct for result in results) / len(results)


def _device(torch: Any) -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
