from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.identity import ResultIdentityInput
from plasticity_placement.pathmem.io import file_hash, json_hash
from plasticity_placement.pathmem.schema import AccessMode, Probe
from plasticity_placement.pathmem.scoring import (
    SCORING_VERSION,
    audit_candidate_tokenization,
    normalize_candidate_scores,
    score_candidate_batch,
)
from plasticity_placement.pathmem_exec.config import PhaseConfig
from plasticity_placement.training.model_utils import (
    chat_prompt,
    load_tokenizer,
    model_load_kwargs,
)

RENDERER_VERSION = "pathmem-execution-renderer-v1"
LIFECYCLE_TRACE_VERSION = "pathmem-evaluation-lifecycle-v1"


@dataclass(slots=True)
class ModelBundle:
    torch: Any
    tokenizer: Any
    model: Any
    config: PhaseConfig
    loaded_adapters: set[str]


def load_model_bundle(config: PhaseConfig) -> ModelBundle:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "scoring dependencies are missing; run `uv sync --extra train`"
        ) from error
    tokenizer = load_tokenizer(config.model.name, config.model.revision)
    load_kwargs, precision = model_load_kwargs(torch, use_4bit=True)
    if precision != "nf4-bfloat16":
        raise RuntimeError(f"evaluation precision changed: {precision}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        **load_kwargs,
    )
    model.eval()
    return ModelBundle(
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        config=config,
        loaded_adapters=set(),
    )


def activate_adapter(bundle: ModelBundle, adapter_dir: Path, adapter_name: str) -> None:
    safe_name = _adapter_name(adapter_name)
    if not bundle.loaded_adapters:
        from peft import PeftModel

        bundle.model = PeftModel.from_pretrained(
            bundle.model,
            adapter_dir,
            adapter_name=safe_name,
            is_trainable=False,
        )
        bundle.loaded_adapters.add(safe_name)
    elif safe_name not in bundle.loaded_adapters:
        bundle.model.load_adapter(
            adapter_dir,
            adapter_name=safe_name,
            is_trainable=False,
        )
        bundle.loaded_adapters.add(safe_name)
    bundle.model.set_adapter(safe_name)
    bundle.model.eval()


def release_model_bundle(bundle: ModelBundle) -> None:
    del bundle.model
    if bundle.torch.cuda.is_available():
        bundle.torch.cuda.empty_cache()


def answer_copy_prompt(action: str) -> str:
    return (
        "Copy the allowed action shown between the tags. Return exactly that action and "
        "nothing else.\n<allowed_action>"
        f"{action}"
        "</allowed_action>"
    )


def score_probe(
    *,
    bundle: ModelBundle,
    phase: str,
    run_id: str,
    item_id: str,
    probe: Probe,
    raw_prompt: str,
    renderer_kind: str,
    arm: str,
    access_mode: AccessMode,
    node_id: str,
    adapter_sha256: str | None,
    probe_bank_sha256: str,
    external_state_sha256: str,
    lifecycle_trace: dict[str, Any],
    disable_adapter: bool = False,
) -> dict[str, Any]:
    formatted_prompt = chat_prompt(bundle.tokenizer, raw_prompt)
    audit = audit_candidate_tokenization(
        bundle.tokenizer,
        formatted_prompt,
        probe.action_choices,
        evaluation_max_length=bundle.config.recipe.evaluation_max_length,
    )
    if not audit["all_candidates_valid"]:
        raise RuntimeError(f"candidate audit failed for {probe.probe_id}")
    context = bundle.model.disable_adapter() if disable_adapter else nullcontext()
    with context:
        scored = score_candidate_batch(bundle, formatted_prompt, audit)
    candidates = scored["candidates"]
    probabilities: list[float] | None = None
    if scored["error_status"] == "ok":
        distribution = normalize_candidate_scores(
            probe.action_choices,
            tuple(float(candidate["sum_logprob"]) for candidate in candidates),
        )
        probabilities = list(distribution.probabilities)
    lifecycle_sha256 = json_hash({"version": LIFECYCLE_TRACE_VERSION, **lifecycle_trace})
    renderer_sha256 = json_hash(
        {
            "version": RENDERER_VERSION,
            "kind": renderer_kind,
            "raw_prompt": raw_prompt,
            "formatted_prompt": formatted_prompt,
        }
    )
    result_identity = ResultIdentityInput(
        node_id=node_id,
        probe_bank_sha256=probe_bank_sha256,
        renderer_and_prompt_sha256=renderer_sha256,
        external_state_sha256=external_state_sha256,
        access_mode=access_mode.value,
        lifecycle_trace_sha256=lifecycle_sha256,
        scorer_sha256=scorer_identity(),
        evaluation_precision="nf4-bfloat16",
    )
    return {
        "phase": phase,
        "run_id": run_id,
        "model_name": bundle.config.model.name,
        "model_revision": bundle.config.model.revision,
        "evaluation_precision": "nf4-bfloat16",
        "item_id": item_id,
        "probe_id": probe.probe_id,
        "terminal_state": probe.terminal_state,
        "probe_category": probe.category.value,
        "core_endpoint": probe.core_endpoint,
        "arm": arm,
        "renderer_kind": renderer_kind,
        "access_mode": access_mode.value,
        "node_id": node_id,
        "adapter_sha256": adapter_sha256,
        "probe_bank_sha256": probe_bank_sha256,
        "raw_prompt_sha256": json_hash(raw_prompt),
        "formatted_prompt_sha256": audit["prompt_sha256"],
        "external_state_sha256": external_state_sha256,
        "result_identity": result_identity.to_dict(),
        "ordered_actions": list(probe.action_choices),
        "expected_action": probe.expected_action,
        "obsolete_action": probe.obsolete_action,
        "predicted_action": scored["predicted_action"],
        "correct": scored["predicted_action"] == probe.expected_action,
        "obsolete_intrusion": (
            probe.obsolete_action is not None
            and scored["predicted_action"] == probe.obsolete_action
        ),
        "candidate_probabilities": probabilities,
        "candidates": candidates,
        "tie": scored["tie"],
        "non_finite": scored["non_finite"],
        "error_status": scored["error_status"],
        "error_message": scored["error_message"],
        "parser_valid": scored["error_status"] == "ok",
        "candidate_audit": audit,
        "latency_seconds": scored["latency_seconds"],
        "lifecycle_trace": {"version": LIFECYCLE_TRACE_VERSION, **lifecycle_trace},
    }


def candidate_score_vector(row: dict[str, Any]) -> tuple[float, float, float, float]:
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 4:
        raise ValueError("score row does not contain four candidates")
    return tuple(float(candidate["sum_logprob"]) for candidate in candidates)


def maximum_candidate_delta(first: dict[str, Any], second: dict[str, Any]) -> float:
    if first["probe_id"] != second["probe_id"]:
        raise ValueError("candidate delta requires the same probe")
    return max(
        abs(left - right)
        for left, right in zip(
            candidate_score_vector(first),
            candidate_score_vector(second),
            strict=True,
        )
    )


def scorer_identity() -> str:
    package_root = Path(__file__).resolve().parents[1]
    paths = (
        package_root / "pathmem" / "scoring.py",
        package_root / "p0d2hfc" / "scoring.py",
        Path(__file__),
    )
    return json_hash(
        {
            "scoring_version": SCORING_VERSION,
            "files": {path.relative_to(package_root).as_posix(): file_hash(path) for path in paths},
        }
    )


def _adapter_name(value: str) -> str:
    return "pm_" + json_hash(value)[:20]
