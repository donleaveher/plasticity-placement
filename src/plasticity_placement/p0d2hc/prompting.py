from __future__ import annotations

from dataclasses import replace

from plasticity_placement.p0c.domain import Arm, P0CProbe
from plasticity_placement.p0d2h.probes import OUTPUT_INSTRUCTION_PREFIX
from plasticity_placement.p0d2h.prompting import (
    EXTERNAL_PROMPT_VARIANT,
    PLAIN_PROMPT_VARIANT,
    render_evaluation_probe,
)

ORACLE_PROMPT_VARIANT = "hard_probe_answer_copy_oracle_v1"
ORACLE_PROMPT_RENDERER_VERSION = "p0d2hc-answer-copy-v1"


def render_calibration_probe(
    probe: P0CProbe,
    *,
    arm: Arm,
    external_note: str,
) -> tuple[P0CProbe, str, str]:
    if arm is Arm.NO_WRITE:
        return render_evaluation_probe(probe, external_note=None)
    if arm is Arm.EXTERNAL:
        return render_evaluation_probe(
            probe,
            external_note=external_note,
        )
    if arm is not Arm.ANSWER_COPY_ORACLE:
        raise ValueError(f"unsupported calibration arm: {arm}")
    if OUTPUT_INSTRUCTION_PREFIX not in probe.prompt:
        raise ValueError(
            f"hard probe is missing the output instruction: {probe.probe_id}"
        )
    if probe.expected_action not in probe.action_choices:
        raise ValueError(
            f"oracle answer is absent from choices: {probe.probe_id}"
        )
    challenge, decision = probe.prompt.split(
        OUTPUT_INSTRUCTION_PREFIX,
        maxsplit=1,
    )
    rendered = (
        f"{challenge}\n\n"
        "<answer_copy_oracle>\n"
        "The exact action token to output is "
        f"{probe.expected_action}. Copy it exactly and ignore every other "
        "action name.\n"
        "</answer_copy_oracle>"
        f"{OUTPUT_INSTRUCTION_PREFIX}{decision}"
    )
    return (
        replace(probe, prompt=rendered),
        ORACLE_PROMPT_VARIANT,
        ORACLE_PROMPT_RENDERER_VERSION,
    )


def prompt_variant_for_arm(arm: Arm) -> str:
    if arm is Arm.NO_WRITE:
        return PLAIN_PROMPT_VARIANT
    if arm is Arm.EXTERNAL:
        return EXTERNAL_PROMPT_VARIANT
    if arm is Arm.ANSWER_COPY_ORACLE:
        return ORACLE_PROMPT_VARIANT
    raise ValueError(f"unsupported calibration arm: {arm}")
