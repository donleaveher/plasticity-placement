from pathlib import Path

import pytest

from plasticity_placement.evaluation.probes import load_probes, parse_action


def test_parse_action_uses_first_unambiguous_token() -> None:
    assert parse_action("答案是 a1。", ("a0", "a1")) == "a1"
    assert parse_action("先考虑 a0，最终 a1", ("a0", "a1")) == "a0"
    assert parse_action("action-a10", ("a0", "a1")) is None


def test_probe_ids_must_be_unique(tmp_path: Path) -> None:
    path = tmp_path / "probes.jsonl"
    row = '{"id":"same","prompt":"p","expected_action":"a1","category":"lesson"}\n'
    path.write_text(row + row, encoding="utf-8")
    with pytest.raises(ValueError):
        load_probes(path)
