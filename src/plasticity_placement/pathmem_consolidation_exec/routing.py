from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.io import json_hash

ROUTER_SCHEMA_VERSION = "pathmem-g1c-exact-key-router-v1"
_KEY_PATTERN = re.compile(r"(?m)\bKey:\s*([A-Za-z0-9_-]+)")


def extract_route_key(prompt: str) -> str | None:
    matches = _KEY_PATTERN.findall(prompt)
    if not matches:
        return None
    unique = set(matches)
    if len(unique) != 1:
        raise ValueError(f"prompt contains ambiguous route keys: {sorted(unique)}")
    return matches[0]


@dataclass(frozen=True, slots=True)
class RouteTarget:
    unit_id: str
    memory_key: str
    adapter_dir: str
    adapter_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "unit_id": self.unit_id,
            "memory_key": self.memory_key,
            "adapter_dir": self.adapter_dir,
            "adapter_sha256": self.adapter_sha256,
        }


class ExactKeyRouter:
    def __init__(self) -> None:
        self._targets: dict[str, RouteTarget] = {}
        self._application_order: list[str] = []

    def apply(self, target: RouteTarget) -> None:
        if not target.unit_id or not target.memory_key or len(target.adapter_sha256) != 64:
            raise ValueError("route targets require complete unit, key, and adapter identity")
        existing = self._targets.get(target.memory_key)
        if existing == target:
            return
        self._targets[target.memory_key] = target
        self._application_order.append(target.unit_id)

    def resolve_key(self, memory_key: str | None) -> RouteTarget | None:
        if memory_key is None:
            return None
        return self._targets.get(memory_key)

    def resolve_prompt(self, prompt: str) -> RouteTarget | None:
        return self.resolve_key(extract_route_key(prompt))

    def snapshot(self) -> dict[str, Any]:
        identity = {
            "schema_version": ROUTER_SCHEMA_VERSION,
            "application_order": list(self._application_order),
            "routes": {
                key: target.to_dict() for key, target in sorted(self._targets.items())
            },
        }
        return {**identity, "router_sha256": json_hash(identity)}


def route_target_from_unit(
    unit: dict[str, Any], *, adapter_dir: Path, adapter_sha256: str
) -> RouteTarget:
    return RouteTarget(
        unit_id=str(unit["unit_id"]),
        memory_key=str(unit["memory_key"]),
        adapter_dir=str(adapter_dir.resolve()),
        adapter_sha256=adapter_sha256,
    )
