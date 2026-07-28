from __future__ import annotations

from plasticity_placement.p0c.compiler import compile_bank
from plasticity_placement.p0c.domain import Arm
from plasticity_placement.p0d2h.probes import (
    OUTPUT_INSTRUCTION_PREFIX,
    compile_hard_probes,
)
from plasticity_placement.p0d2h.prompting import (
    EXTERNAL_PROMPT_VARIANT,
    PLAIN_PROMPT_VARIANT,
)
from plasticity_placement.p0d2hc.prompting import (
    ORACLE_PROMPT_VARIANT,
    render_calibration_probe,
)


def test_calibration_prompt_places_oracle_next_to_output_contract() -> None:
    item = compile_bank()[0]
    probe = compile_hard_probes(item)[0]
    rendered, variant, _ = render_calibration_probe(
        probe,
        arm=Arm.ANSWER_COPY_ORACLE,
        external_note=item.external_note,
    )
    oracle_position = rendered.prompt.index("<answer_copy_oracle>")
    instruction_position = rendered.prompt.index(OUTPUT_INSTRUCTION_PREFIX)
    assert oracle_position < instruction_position
    assert rendered.prompt.index(probe.expected_action) > oracle_position
    assert rendered.prompt.rstrip().endswith("Action:")
    assert variant == ORACLE_PROMPT_VARIANT


def test_calibration_plain_and_external_prompts_reuse_frozen_renderers() -> None:
    item = compile_bank()[0]
    probe = compile_hard_probes(item)[0]
    plain = render_calibration_probe(
        probe,
        arm=Arm.NO_WRITE,
        external_note=item.external_note,
    )
    external = render_calibration_probe(
        probe,
        arm=Arm.EXTERNAL,
        external_note=item.external_note,
    )
    assert plain[0] == probe
    assert plain[1] == PLAIN_PROMPT_VARIANT
    assert item.external_note in external[0].prompt
    assert external[1] == EXTERNAL_PROMPT_VARIANT
