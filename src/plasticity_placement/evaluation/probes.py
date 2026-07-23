from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Probe:
    probe_id: str
    prompt: str
    expected_action: str
    category: str


def load_probes(path: Path) -> list[Probe]:
    probes: list[Probe] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            fields = ("id", "prompt", "expected_action", "category")
            if any(not isinstance(record.get(field), str) for field in fields):
                raise ValueError(f"line {line_number} must contain string fields: {fields}")
            probe = Probe(
                probe_id=record["id"],
                prompt=record["prompt"],
                expected_action=record["expected_action"],
                category=record["category"],
            )
            if not all((probe.probe_id, probe.prompt, probe.expected_action, probe.category)):
                raise ValueError(f"line {line_number} contains an empty required field")
            if probe.probe_id in seen_ids:
                raise ValueError(f"duplicate probe id: {probe.probe_id}")
            seen_ids.add(probe.probe_id)
            probes.append(probe)
    if not probes:
        raise ValueError(f"no probes found in {path}")
    return probes


def parse_action(text: str, allowed_actions: tuple[str, ...]) -> str | None:
    """Extract one unambiguous action token from generated text."""
    matches: list[tuple[int, str]] = []
    for action in allowed_actions:
        pattern = rf"(?<![\w]){re.escape(action)}(?![\w])"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            matches.append((match.start(), action))
    if not matches:
        return None
    matches.sort()
    return matches[0][1]
