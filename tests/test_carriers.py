import pytest

from plasticity_placement.simulation.carriers import make_carrier
from plasticity_placement.simulation.domain import AgentState, CarrierKind, Lesson


@pytest.mark.parametrize("kind", list(CarrierKind))
def test_carrier_rollback_restores_behavior(kind: CarrierKind) -> None:
    state = AgentState(default_action="a0", base_policy=())
    lesson = Lesson(context="target", action="a1")
    carrier = make_carrier(kind)
    before = carrier.act("target", state)
    receipt = carrier.write(lesson)
    carrier.rollback(receipt)
    assert carrier.act("target", state) == before


def test_memory_carriers_apply_lesson() -> None:
    state = AgentState(default_action="a0", base_policy=())
    lesson = Lesson(context="target", action="a1")
    for kind in (CarrierKind.EXTERNAL, CarrierKind.PARAMETRIC, CarrierKind.BOTH):
        carrier = make_carrier(kind)
        carrier.write(lesson)
        assert carrier.act("target", state) == "a1"
