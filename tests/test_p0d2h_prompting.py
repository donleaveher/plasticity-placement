from __future__ import annotations

from plasticity_placement.p0c.compiler import compile_bank
from plasticity_placement.p0d2h.probes import (
    OUTPUT_INSTRUCTION_PREFIX,
    compile_hard_probes,
)
from plasticity_placement.p0d2h.prompting import (
    EXTERNAL_PROMPT_RENDERER_VERSION,
    EXTERNAL_PROMPT_VARIANT,
    PLAIN_PROMPT_VARIANT,
    audit_prompt,
    render_evaluation_probe,
)


class _CharacterTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[0]["content"] + "\n<assistant>"

    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        add_special_tokens: bool,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        assert add_special_tokens is False
        values = [ord(character) for character in text]
        if truncation and max_length is not None:
            values = values[:max_length]
        return {"input_ids": values}


def test_external_memory_is_adjacent_to_the_decision_suffix() -> None:
    item = compile_bank()[0]
    probe = compile_hard_probes(item)[0]
    rendered, variant, version = render_evaluation_probe(
        probe,
        external_note=item.external_note,
    )
    memory_position = rendered.prompt.index("<verified_memory>")
    instruction_position = rendered.prompt.index(OUTPUT_INSTRUCTION_PREFIX)
    assert memory_position > rendered.prompt.index("ARCHIVED")
    assert memory_position < instruction_position
    assert item.external_note in rendered.prompt
    assert variant == EXTERNAL_PROMPT_VARIANT
    assert version == EXTERNAL_PROMPT_RENDERER_VERSION


def test_prompt_token_audit_detects_and_eliminates_right_truncation() -> None:
    probe = compile_hard_probes(compile_bank()[0])[0]
    tokenizer = _CharacterTokenizer()
    truncated = audit_prompt(
        tokenizer,
        probe,
        prompt_variant=PLAIN_PROMPT_VARIANT,
        prompt_rendering_version="frozen_hard_probe",
        evaluation_max_length=64,
    )
    assert truncated.input_truncated is True
    assert truncated.truncated_token_count > 0
    assert truncated.output_instruction_preserved is False

    complete = audit_prompt(
        tokenizer,
        probe,
        prompt_variant=PLAIN_PROMPT_VARIANT,
        prompt_rendering_version="frozen_hard_probe",
        evaluation_max_length=10_000,
    )
    assert complete.input_truncated is False
    assert complete.truncated_token_count == 0
    assert complete.output_instruction_preserved is True
