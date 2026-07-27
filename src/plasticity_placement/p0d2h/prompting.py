from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from typing import Any

from plasticity_placement.p0c.domain import P0CProbe
from plasticity_placement.p0d2h.probes import OUTPUT_INSTRUCTION_PREFIX
from plasticity_placement.training.model_utils import chat_prompt

EXTERNAL_PROMPT_RENDERER_VERSION = "p0d2h-external-memory-v2"
PLAIN_PROMPT_VARIANT = "hard_probe"
EXTERNAL_PROMPT_VARIANT = "hard_probe_external_memory_v2"


@dataclass(frozen=True, slots=True)
class PromptTokenAudit:
    lesson_id: str
    probe_id: str
    category: str
    prompt_variant: str
    prompt_rendering_version: str
    prompt_sha256: str
    untruncated_input_tokens: int
    retained_input_tokens: int
    truncated_token_count: int
    input_truncated: bool
    output_instruction_preserved: bool
    evaluation_max_length: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def render_evaluation_probe(
    probe: P0CProbe,
    *,
    external_note: str | None,
) -> tuple[P0CProbe, str, str]:
    if external_note is None or not probe.retrieval_relevant:
        return probe, PLAIN_PROMPT_VARIANT, "frozen_hard_probe"
    if OUTPUT_INSTRUCTION_PREFIX not in probe.prompt:
        raise ValueError(f"hard probe is missing the output instruction: {probe.probe_id}")
    challenge, decision = probe.prompt.split(
        OUTPUT_INSTRUCTION_PREFIX,
        maxsplit=1,
    )
    rendered = (
        f"{challenge}\n\n"
        "<verified_memory>\n"
        f"{external_note}\n"
        "</verified_memory>"
        f"{OUTPUT_INSTRUCTION_PREFIX}{decision}"
    )
    return (
        replace(probe, prompt=rendered),
        EXTERNAL_PROMPT_VARIANT,
        EXTERNAL_PROMPT_RENDERER_VERSION,
    )


def audit_prompt(
    tokenizer: Any,
    probe: P0CProbe,
    *,
    prompt_variant: str,
    prompt_rendering_version: str,
    evaluation_max_length: int,
) -> PromptTokenAudit:
    if evaluation_max_length <= 0:
        raise ValueError("evaluation_max_length must be positive")
    formatted = chat_prompt(tokenizer, probe.prompt)
    full_ids = _input_ids(
        tokenizer(
            formatted,
            truncation=False,
            add_special_tokens=False,
        )
    )
    retained_ids = _input_ids(
        tokenizer(
            formatted,
            truncation=True,
            max_length=evaluation_max_length,
            add_special_tokens=False,
        )
    )
    untruncated = len(full_ids)
    retained = len(retained_ids)
    truncated_count = untruncated - retained
    input_truncated = truncated_count > 0
    suffix_size = min(32, len(full_ids))
    retained_suffix = (
        suffix_size > 0
        and len(retained_ids) >= suffix_size
        and retained_ids[-suffix_size:] == full_ids[-suffix_size:]
    )
    instruction_preserved = (
        OUTPUT_INSTRUCTION_PREFIX in probe.prompt
        and probe.prompt.rstrip().endswith("Action:")
        and retained_suffix
    )
    return PromptTokenAudit(
        lesson_id=probe.lesson_id,
        probe_id=probe.probe_id,
        category=probe.category,
        prompt_variant=prompt_variant,
        prompt_rendering_version=prompt_rendering_version,
        prompt_sha256=sha256(formatted.encode()).hexdigest(),
        untruncated_input_tokens=untruncated,
        retained_input_tokens=retained,
        truncated_token_count=truncated_count,
        input_truncated=input_truncated,
        output_instruction_preserved=instruction_preserved,
        evaluation_max_length=evaluation_max_length,
    )


def _input_ids(encoded: Any) -> list[int]:
    values = encoded["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]
