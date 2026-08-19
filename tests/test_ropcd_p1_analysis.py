from __future__ import annotations

from typing import Any

import pytest

from plasticity_placement.pathmem.compiler import compile_bank
from plasticity_placement.pathmem.io import json_hash
from plasticity_placement.pathmem_ropcd_p0 import runtime as p0_runtime
from plasticity_placement.pathmem_ropcd_p1 import runtime
from plasticity_placement.pathmem_ropcd_p1.analysis import summarize_p1
from plasticity_placement.pathmem_ropcd_p1.config import EXPECTED_ROWS, EXPECTED_UNITS
from plasticity_placement.pathmem_ropcd_p1.planner import compile_p1_plan


def _handoff() -> dict[str, Any]:
    identity = {
        "schema_version": "pathmem-ropcd-g2-to-p1-handoff-v1",
        "passed": True,
        "repair_id": "r" * 64,
        "source_run_id": "s" * 64,
        "source_plan_id": "p" * 64,
        "source_matrix_sha256": "m" * 64,
        "g2_summary_sha256": "g" * 64,
        "g2_gate": {"gate": "G2", "passed": True},
        "repair_implementation_sha256": "i" * 64,
        "p1_implementation_review_eligible": True,
        "p1_authorized": False,
        "kill_or_reserve_accessed": False,
    }
    return {**identity, "handoff_id": json_hash(identity)}


def _fake_score_prompt(_session: object, **kwargs: Any) -> dict[str, Any]:
    actions = list(kwargs["action_choices"])
    expected = str(kwargs["expected_action"])
    probabilities = [0.05, 0.05, 0.05, 0.05]
    probabilities[actions.index(expected)] = 0.85
    return {
        "row_id": json_hash(
            [kwargs["unit_id"], kwargs["probe_id"], kwargs["arm"], kwargs["raw_prompt"]]
        ),
        "unit_id": kwargs["unit_id"],
        "item_id": kwargs["item_id"],
        "probe_id": kwargs["probe_id"],
        "probe_category": kwargs["probe_category"],
        "arm": kwargs["arm"],
        "ordered_actions": actions,
        "candidate_probabilities": probabilities,
        "candidate_scores": probabilities,
        "expected_action": expected,
        "obsolete_action": kwargs["obsolete_action"],
        "predicted_action": expected,
        "correct": True,
        "obsolete_intrusion": False,
        "error_status": "ok",
        "tie": False,
        "non_finite": False,
        "parser_valid": True,
        "raw_prompt_sha256": json_hash(kwargs["raw_prompt"]),
        "candidate_audit": {"all_candidates_valid": True},
        **kwargs.get("metadata", {}),
    }


def _matrix(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    plan = compile_p1_plan(_handoff())
    bank = compile_bank()
    probes = {probe.probe_id: probe for probe in bank.probes}
    blocks = {block.block_id: block for block in bank.event_blocks}
    monkeypatch.setattr(p0_runtime, "score_prompt", _fake_score_prompt)
    rows: list[dict[str, Any]] = []
    item_ids = {unit["item_id"] for unit in plan["units"]}
    for probe in bank.probes:
        if probe.item_id in item_ids and probe.core_endpoint:
            rows.append(_fake_score_prompt(object(), **_control_kwargs(probe)))
    for unit in plan["units"]:
        logical = unit["logical_name"]
        if unit["technical_duplicate"]:
            raw = p0_runtime._score_duplicate(object(), unit, probes)
        elif logical in {"A2", "B2"}:
            raw = p0_runtime._score_anchor(object(), unit, probes)
        elif logical in {"ABA", "BAA", "BAB", "ABB"}:
            raw = p0_runtime._score_final_path(object(), unit, probes, blocks)
        else:
            continue
        rows.extend(runtime._tag_rows(raw, unit))
    return rows


def _control_kwargs(probe: Any) -> dict[str, Any]:
    return {
        "raw_prompt": probe.prompt,
        "action_choices": probe.action_choices,
        "expected_action": probe.expected_action,
        "obsolete_action": probe.obsolete_action,
        "unit_id": f"p1r:{probe.item_id}:control:{probe.terminal_state}",
        "item_id": probe.item_id,
        "probe_id": probe.probe_id,
        "probe_category": probe.category.value,
        "arm": "no_memory",
        "disable_adapter": False,
        "route_key": None,
        "route_unit_id": None,
    }


def test_p1_summary_uses_item_clustered_seed_averaging_and_three_way_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _matrix(monkeypatch)
    assert len(rows) == EXPECTED_ROWS
    summary = summarize_p1(rows, verified_units=EXPECTED_UNITS, integrity_checks={})
    assert summary["gate"]["passed"] is True
    assert summary["decision"]["classification"] == "approximate_consistency"
    assert summary["primary"]["D_mean_nats"] == pytest.approx(0.0)
    assert len(summary["primary"]["per_item_seed"]) == 36
    assert len(summary["primary"]["per_item"]) == 12
    assert summary["decision"]["p1b_implementation_review_eligible"] is False
    assert summary["p1b_authorized"] is False
    assert summary["p2_authorized"] is False


def test_p1_summary_rejects_a_missing_randomized_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _matrix(monkeypatch)
    with pytest.raises(
        ValueError, match="endpoint panel|item aggregation|seed aggregation|qualification panel"
    ):
        summarize_p1(rows[:-1], verified_units=EXPECTED_UNITS, integrity_checks={})
