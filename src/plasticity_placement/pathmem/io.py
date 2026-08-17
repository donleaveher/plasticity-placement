from __future__ import annotations

import json
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def json_hash(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def multiset_hash(values: Iterable[object]) -> str:
    canonical = sorted(canonical_json_bytes(value).decode() for value in values)
    return json_hash(canonical)


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def immutable_write(path: Path, payload: bytes, label: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing {label} changed; use a new G0 output directory")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return sha256(payload).hexdigest()


def atomic_json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def immutable_json_write(path: Path, value: object, label: str) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    return immutable_write(path, payload, label)


def immutable_jsonl_write(path: Path, rows: Iterable[object], label: str) -> str:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    return immutable_write(path, payload, label)


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{label} line {line_number} must be a JSON object")
        rows.append(value)
    return rows
