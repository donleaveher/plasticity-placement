from __future__ import annotations

from dataclasses import dataclass

from plasticity_placement.pathmem.io import json_hash
from plasticity_placement.pathmem.reducer import reduce_events
from plasticity_placement.pathmem.schema import AccessMode, EventBlock, MemoryEntry, Probe

BACKEND_INTERFACE_VERSION = "pathmem-backend-interface-v1"


@dataclass(frozen=True, slots=True)
class ExternalTraceRow:
    event_id: str
    event_sha256: str
    ledger_sha256: str
    active_entries: tuple[MemoryEntry, ...]


class VersionedExternalStore:
    """Deterministic latest-valid-write store used as the semantic negative control."""

    def __init__(self, initial_fixture: tuple[MemoryEntry, ...] = ()) -> None:
        self._initial_fixture = initial_fixture
        self._events = []
        self._trace: list[ExternalTraceRow] = []

    def apply(self, block: EventBlock) -> ExternalTraceRow:
        self._events.append(block.event)
        entries = reduce_events(self._initial_fixture, self._events)
        row = ExternalTraceRow(
            event_id=block.event.event_id,
            event_sha256=json_hash(block.event.to_dict()),
            ledger_sha256=json_hash([entry.to_dict() for entry in entries]),
            active_entries=entries,
        )
        self._trace.append(row)
        return row

    @property
    def entries(self) -> tuple[MemoryEntry, ...]:
        return reduce_events(self._initial_fixture, self._events)

    @property
    def trace(self) -> tuple[ExternalTraceRow, ...]:
        return tuple(self._trace)

    def render(self) -> str:
        lines = ["<pathmem_current_state>"]
        lines.extend(f"{entry.key}: {entry.value}" for entry in self.entries)
        lines.append("</pathmem_current_state>")
        return "\n".join(lines)


class ICLHistoryRenderer:
    """Chronological control that intentionally exposes update order."""

    def __init__(self) -> None:
        self._blocks: list[EventBlock] = []

    def apply(self, block: EventBlock) -> None:
        self._blocks.append(block)

    def render(self) -> str:
        lines = ["<pathmem_update_history>"]
        for position, block in enumerate(self._blocks, start=1):
            lines.append(
                f"position={position} key={block.event.key} value={block.event.value} "
                f"event={block.event.kind}"
            )
        lines.append("</pathmem_update_history>")
        return "\n".join(lines)


def render_fresh_session_probe(
    probe: Probe,
    *,
    persistent_state: str,
    access_mode: AccessMode,
) -> str:
    if not probe.prompt:
        raise ValueError("fresh-session probes require a prompt")
    if access_mode is AccessMode.OFF:
        return probe.prompt
    if not persistent_state:
        raise ValueError("memory-on evaluation requires named persistent state")
    return f"{persistent_state}\n\n{probe.prompt}"
