from plasticity_placement.simulation.domain import CarrierKind
from plasticity_placement.simulation.experiment import run_unit
from plasticity_placement.simulation.world import ToolWorld


def test_branches_share_state_and_panel_hashes() -> None:
    unit = ToolWorld(horizon=20).create_unit(seed=12, recurrence=0.9, volatility=0.2)
    results = run_unit(unit)
    assert {result.carrier for result in results} == set(CarrierKind)
    assert len({result.state_hash for result in results}) == 1
    assert len({result.panel_hash for result in results}) == 1


def test_stable_rule_rewards_written_memory() -> None:
    unit = ToolWorld(horizon=100).create_unit(seed=5, recurrence=1.0, volatility=0.0)
    by_carrier = {result.carrier: result for result in run_unit(unit)}
    assert by_carrier[CarrierKind.EXTERNAL].reward == 1.0
    assert by_carrier[CarrierKind.PARAMETRIC].reward == 1.0
    assert by_carrier[CarrierKind.NONE].reward == 0.0


def test_fully_flipped_rule_rewards_no_write() -> None:
    unit = ToolWorld(horizon=100).create_unit(seed=5, recurrence=1.0, volatility=1.0)
    by_carrier = {result.carrier: result for result in run_unit(unit)}
    assert by_carrier[CarrierKind.NONE].reward == 1.0
    assert by_carrier[CarrierKind.EXTERNAL].reward == 0.0
