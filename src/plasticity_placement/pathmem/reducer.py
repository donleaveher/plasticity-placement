from __future__ import annotations

from collections.abc import Iterable

from plasticity_placement.pathmem.io import json_hash
from plasticity_placement.pathmem.schema import EventKind, MemoryEntry, MemoryEvent

REDUCER_VERSION = "pathmem-latest-valid-write-reducer-v1"


def reduce_events(
    initial_fixture: Iterable[MemoryEntry],
    events: Iterable[MemoryEvent],
) -> tuple[MemoryEntry, ...]:
    ledger = {entry.key: entry.value for entry in initial_fixture}
    for event in events:
        if event.kind in {EventKind.WRITE, EventKind.REVISE}:
            if event.key is None or event.value is None:
                raise AssertionError("validated write event lost key/value")
            ledger[event.key] = event.value
        elif event.kind is EventKind.DELETE:
            if event.key is None:
                raise AssertionError("validated delete event lost key")
            ledger.pop(event.key, None)
        elif event.kind is EventKind.RESTORE:
            ledger = {entry.key: entry.value for entry in event.snapshot}
        elif event.kind in {EventKind.NOOP, EventKind.IDLE, EventKind.CLOCK_NULL}:
            continue
        else:
            raise ValueError(f"unsupported event kind: {event.kind}")
    return tuple(MemoryEntry(key=key, value=ledger[key]) for key in sorted(ledger))


def endpoint_hash(entries: Iterable[MemoryEntry]) -> str:
    return json_hash([entry.to_dict() for entry in entries])
