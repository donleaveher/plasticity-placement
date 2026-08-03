from __future__ import annotations

import json
from collections import Counter

import pytest

from plasticity_placement.p0c.compiler import compile_bank
from plasticity_placement.p0d2h.probes import compile_hard_probe_bank
from plasticity_placement.p0d2hcrd.probes import compile_decomposition_bank
from plasticity_placement.p0d2hrtb.analysis import analyze_bridge
from plasticity_placement.p0d2hrtb.config import AuditSpec
from plasticity_placement.p0d2hrtb.probes import (
    _exact_external_slot_lines,
    _external_payload_segment,
    compile_bridge_bank,
    route_anchor_static_sha256,
)
from plasticity_placement.p0d2hrtb.runtime import _publish_raw_scores

SELECTED_IDS = (
    "F_pair01_a",
    "F_pair01_b",
    "F_pair04_a",
    "F_pair04_b",
    "F_pair06_a",
    "F_pair06_b",
    "RF_pair01_a",
    "RF_pair01_b",
    "RF_pair04_a",
    "RF_pair04_b",
    "RF_pair05_a",
    "RF_pair05_b",
    "P_pair01_a",
    "P_pair01_b",
    "P_pair02_a",
    "P_pair02_b",
    "P_pair03_a",
    "P_pair03_b",
    "P_pair04_a",
    "P_pair04_b",
    "P_pair05_a",
    "P_pair05_b",
    "P_pair06_a",
    "P_pair06_b",
)


def _selected():
    by_id = {item.lesson.lesson_id: item for item in compile_bank()}
    return tuple(by_id[lesson_id] for lesson_id in SELECTED_IDS)


def test_bridge_bank_is_deterministic_complete_and_balanced() -> None:
    selected = _selected()
    hard_bank = compile_hard_probe_bank(selected)
    probes, audit = compile_bridge_bank(selected, hard_bank, AuditSpec())
    repeated, repeated_audit = compile_bridge_bank(selected, hard_bank, AuditSpec())

    assert probes == repeated
    assert audit == repeated_audit
    assert audit["all_checks_passed"] is True
    assert len(probes) == 960
    assert set(Counter(probe.cell for probe in probes).values()) == {96}
    assert sum(probe.response_type == "slot" for probe in probes) == 768
    assert sum(probe.response_type == "action" for probe in probes) == 192
    decomposition, _ = compile_decomposition_bank(selected, hard_bank)
    route_anchors = {
        probe.probe_id: probe
        for lesson_probes in decomposition.values()
        for probe in lesson_probes
        if probe.endpoint == "route_only"
    }
    exact = [probe for probe in probes if probe.cell == "slot_map_snake_opaque"]
    assert all(
        probe.prompt == route_anchors[str(probe.route_anchor_probe_id)].prompt
        and probe.expected_candidate
        == route_anchors[str(probe.route_anchor_probe_id)].expected_candidate
        and probe.ordered_candidates
        == route_anchors[str(probe.route_anchor_probe_id)].ordered_candidates
        and probe.route_anchor_static_sha256
        == route_anchor_static_sha256(route_anchors[str(probe.route_anchor_probe_id)])
        for probe in exact
    )
    assert all(
        f"Current marker: {route_anchors[str(probe.route_anchor_probe_id)].current_marker}"
        in probe.prompt
        for probe in probes
        if probe.response_type == "slot"
    )
    current_marker_is_first = Counter()
    for probe in exact:
        routing = probe.prompt.split("\n", maxsplit=1)[0]
        first_marker = routing.split("Routing table: ", maxsplit=1)[1].split(" ->", maxsplit=1)[0]
        current_marker = routing.split("Current marker: ", maxsplit=1)[1].split(".", maxsplit=1)[0]
        current_marker_is_first[current_marker == first_marker] += 1
    assert current_marker_is_first == {True: 48, False: 48}
    external_actions = {
        probe.source_probe_id: probe for probe in probes if probe.cell == "external_action"
    }
    forced_actions = {
        probe.source_probe_id: probe for probe in probes if probe.cell == "forced_slot_action"
    }
    assert all(
        _external_payload_segment(external_actions[source].prompt)
        == _external_payload_segment(forced_actions[source].prompt)
        for source in external_actions
    )
    full_external = [probe for probe in probes if probe.cell == "slot_default_natural_external"]
    assert all(
        item.external_note in probe.prompt
        for item in selected
        for probe in full_external
        if probe.lesson_id == item.lesson.lesson_id
    )
    hard_by_id = {
        probe.probe_id: probe for lesson_probes in hard_bank.values() for probe in lesson_probes
    }
    assert all(
        all(
            line in probe.prompt
            for line in _exact_external_slot_lines(hard_by_id[probe.source_probe_id]).values()
        )
        and "selects slot " in probe.prompt
        for probe in full_external
    )


def test_bridge_spec_rejects_post_hoc_changes() -> None:
    with pytest.raises(ValueError, match="frozen"):
        AuditSpec(material_rescue_threshold=0.05)
    with pytest.raises(ValueError, match="frozen"):
        AuditSpec(decision_count=959)


def test_primary_rescues_use_adapter_effect_difference_in_differences() -> None:
    cells = (
        "slot_map_snake_opaque",
        "slot_map_snake_external",
        "slot_map_natural_opaque",
        "slot_map_natural_external",
        "slot_default_snake_opaque",
        "slot_default_snake_external",
        "slot_default_natural_opaque",
        "slot_default_natural_external",
        "external_action",
        "forced_slot_action",
    )
    pairs = []
    for cell in cells:
        for lesson_index in range(24):
            for variant in range(1, 5):
                full_external = cell == "slot_default_natural_external"
                base_correct = True
                adapter_correct = not full_external
                pairs.append(
                    {
                        "lesson_id": f"lesson-{lesson_index:02d}",
                        "route_variant": variant,
                        "cell": cell,
                        "base_correct": base_correct,
                        "adapter_correct": adapter_correct,
                        "transition": (
                            "correct_to_correct" if adapter_correct else "correct_to_wrong"
                        ),
                        "prediction_changed": not adapter_correct,
                        "signed_expected_margin_delta": 0.0,
                        "base_error_status": "ok",
                        "adapter_error_status": "ok",
                        "base_tie": False,
                        "adapter_tie": False,
                        "base_tie_expected_compatible": False,
                        "adapter_tie_expected_compatible": False,
                    }
                )

    result = analyze_bridge(
        pairs,
        bootstrap_samples=100,
        bootstrap_seed=20260803,
        adjusted_confidence=AuditSpec().primary_two_sided_confidence,
        material_threshold=0.10,
    )

    assert result["supported_barriers"] == ["grammar", "lexicon", "payload"]
    assert result["decision"]["status"] == "one_or_more_transfer_barriers_supported"
    assert all(
        values["estimate"] == 1.0 and values["supported_barrier"] is True
        for values in result["primary_rescue_contrasts"].values()
    )
    assert result["decision"]["mappings_per_adapter_authorized"] is False


def test_primary_factorial_tie_blocks_all_barrier_support() -> None:
    cells = (
        "slot_map_snake_opaque",
        "slot_map_snake_external",
        "slot_map_natural_opaque",
        "slot_map_natural_external",
        "slot_default_snake_opaque",
        "slot_default_snake_external",
        "slot_default_natural_opaque",
        "slot_default_natural_external",
        "external_action",
        "forced_slot_action",
    )
    pairs = []
    for cell in cells:
        for lesson_index in range(24):
            for variant in range(1, 5):
                full_external = cell == "slot_default_natural_external"
                pairs.append(
                    {
                        "lesson_id": f"lesson-{lesson_index:02d}",
                        "route_variant": variant,
                        "cell": cell,
                        "base_correct": True,
                        "adapter_correct": not full_external,
                        "transition": "correct_to_wrong" if full_external else "correct_to_correct",
                        "prediction_changed": full_external,
                        "signed_expected_margin_delta": 0.0,
                        "base_error_status": "ok",
                        "adapter_error_status": "ok",
                        "base_tie": False,
                        "adapter_tie": False,
                        "base_tie_expected_compatible": False,
                        "adapter_tie_expected_compatible": False,
                    }
                )
    pairs[0]["base_error_status"] = "tie"
    pairs[0]["base_tie"] = True
    pairs[0]["base_tie_expected_compatible"] = True

    result = analyze_bridge(
        pairs,
        bootstrap_samples=100,
        bootstrap_seed=20260803,
        adjusted_confidence=AuditSpec().primary_two_sided_confidence,
        material_threshold=0.10,
    )

    assert result["decision"]["status"] == "scoring_integrity_failed"
    assert result["decision"]["primary_factorial_zero_tie_integrity"] is False
    assert result["supported_barriers"] == []
    assert all(
        contrast["supported_barrier"] is False and contrast["scoring_eligible"] is False
        for contrast in result["primary_rescue_contrasts"].values()
    )


def test_raw_scores_are_checkpointed_before_analysis(tmp_path) -> None:
    off = [{"probe_id": "p1", "adapter_state": "adapter_off"}]
    on = [{"probe_id": "p1", "adapter_state": "adapter_on"}]

    hashes = _publish_raw_scores(tmp_path, run_id="rtb-test", off_rows=off, on_rows=on)

    checkpoint = json.loads((tmp_path / "raw_score_checkpoint.json").read_text())
    assert checkpoint["checkpoint_stage"] == "inference_complete_before_analysis"
    assert checkpoint["automatic_analysis_recovery_allowed"] is False
    assert checkpoint["artifacts"] == {
        "adapter_off": hashes["adapter_off"],
        "adapter_on": hashes["adapter_on"],
    }
    assert not (tmp_path / "summary.json").exists()
