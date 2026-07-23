import pytest

from plasticity_placement.simulation.world import ToolWorld


def test_unit_generation_is_deterministic() -> None:
    world = ToolWorld(horizon=10)
    first = world.create_unit(seed=7, recurrence=0.5, volatility=0.2)
    second = world.create_unit(seed=7, recurrence=0.5, volatility=0.2)
    assert first == second


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_invalid_probabilities_are_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        ToolWorld().create_unit(seed=0, recurrence=value, volatility=0.0)
