from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass

from plasticity_placement.simulation.domain import AgentState, CarrierKind, Lesson


@dataclass(frozen=True, slots=True)
class WriteReceipt:
    snapshot: object


class MemoryCarrier(ABC):
    kind: CarrierKind

    @abstractmethod
    def write(self, lesson: Lesson) -> WriteReceipt: ...

    @abstractmethod
    def act(self, context: str, state: AgentState) -> str: ...

    @abstractmethod
    def rollback(self, receipt: WriteReceipt) -> None: ...

    @property
    @abstractmethod
    def write_cost(self) -> float: ...


class NoWriteCarrier(MemoryCarrier):
    kind = CarrierKind.NONE

    def write(self, lesson: Lesson) -> WriteReceipt:
        return WriteReceipt(snapshot=None)

    def act(self, context: str, state: AgentState) -> str:
        return state.action_for(context)

    def rollback(self, receipt: WriteReceipt) -> None:
        return None

    @property
    def write_cost(self) -> float:
        return 0.0


class ExternalCarrier(MemoryCarrier):
    kind = CarrierKind.EXTERNAL

    def __init__(self) -> None:
        self._notes: dict[str, str] = {}

    def write(self, lesson: Lesson) -> WriteReceipt:
        receipt = WriteReceipt(snapshot=deepcopy(self._notes))
        self._notes[lesson.context] = lesson.action
        return receipt

    def act(self, context: str, state: AgentState) -> str:
        return self._notes.get(context, state.action_for(context))

    def rollback(self, receipt: WriteReceipt) -> None:
        self._notes = deepcopy(receipt.snapshot)  # type: ignore[assignment]

    @property
    def write_cost(self) -> float:
        return 1.0


class SimulatedParametricCarrier(MemoryCarrier):
    """Interface-compatible stand-in for a future LoRA implementation."""

    kind = CarrierKind.PARAMETRIC

    def __init__(self) -> None:
        self._policy_delta: dict[str, str] = {}

    def write(self, lesson: Lesson) -> WriteReceipt:
        receipt = WriteReceipt(snapshot=deepcopy(self._policy_delta))
        self._policy_delta[lesson.context] = lesson.action
        return receipt

    def act(self, context: str, state: AgentState) -> str:
        return self._policy_delta.get(context, state.action_for(context))

    def rollback(self, receipt: WriteReceipt) -> None:
        self._policy_delta = deepcopy(receipt.snapshot)  # type: ignore[assignment]

    @property
    def write_cost(self) -> float:
        return 4.0


class BothCarrier(MemoryCarrier):
    kind = CarrierKind.BOTH

    def __init__(self) -> None:
        self.external = ExternalCarrier()
        self.parametric = SimulatedParametricCarrier()

    def write(self, lesson: Lesson) -> WriteReceipt:
        snapshots = (self.external.write(lesson), self.parametric.write(lesson))
        return WriteReceipt(snapshot=snapshots)

    def act(self, context: str, state: AgentState) -> str:
        parametric_action = self.parametric.act(context, state)
        return self.external._notes.get(context, parametric_action)

    def rollback(self, receipt: WriteReceipt) -> None:
        external_receipt, parametric_receipt = receipt.snapshot  # type: ignore[misc]
        self.external.rollback(external_receipt)
        self.parametric.rollback(parametric_receipt)

    @property
    def write_cost(self) -> float:
        return self.external.write_cost + self.parametric.write_cost


def make_carrier(kind: CarrierKind) -> MemoryCarrier:
    factories = {
        CarrierKind.NONE: NoWriteCarrier,
        CarrierKind.EXTERNAL: ExternalCarrier,
        CarrierKind.PARAMETRIC: SimulatedParametricCarrier,
        CarrierKind.BOTH: BothCarrier,
    }
    return factories[kind]()
