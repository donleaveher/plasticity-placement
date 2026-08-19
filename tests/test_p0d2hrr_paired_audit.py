from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from plasticity_placement.p0d2hcrd.scoring import (
    rank_candidate_scores as rank_crd_candidates,
)
from plasticity_placement.p0d2hfc.scoring import (
    rank_candidate_scores as rank_fc_candidates,
)
from plasticity_placement.p0d2hrr.io import json_hash
from plasticity_placement.p0d2hrr.paired_audit import (
    _default_base_crd_revision_lock,
    _render_report,
    _require_crd_audit_matches_bank,
    _require_crd_row_matches_bank,
    _require_independent_output,
    _validate_admissible_statuses,
    _validate_finite_tie_policy,
    build_combined_failure_taxonomy,
    build_crd_pairs,
    build_forced_choice_pairs,
    build_paired_analysis,
    exact_mcnemar,
    paired_composition_margin_interaction,
    paired_summary,
    retrieval_to_combined_margin_shift,
    signed_expected_margin,
)


def _candidate(name: str, token_id: int, score: float) -> dict[str, object]:
    return {
        "candidate": name,
        "continuation": f" {name}",
        "token_ids": [token_id],
        "token_count": 1,
        "sum_logprob": score,
        "mean_logprob": score,
        "token_logprobs": [score],
    }


def _fc_row(
    scores: tuple[float, float, float, float],
    *,
    lesson_id: str = "lesson-1",
    probe_id: str = "probe-1",
    expected: str = "act_a",
) -> dict[str, object]:
    names = ("act_a", "act_b", "act_c", "act_d")
    candidates = [
        {
            **_candidate(name, index + 10, score),
            "action": name,
        }
        for index, (name, score) in enumerate(zip(names, scores, strict=True))
    ]
    for candidate in candidates:
        candidate.pop("candidate")
    outcome = rank_fc_candidates(candidates)
    return {
        "lesson_id": lesson_id,
        "pair_id": f"pair-{lesson_id}",
        "lesson_type": "factual",
        "probe_id": probe_id,
        "category": "conditional_route",
        "arm": "external",
        "expected_action": expected,
        "ordered_allowed_actions": list(names),
        "prompt_variant": "external",
        "prompt_rendering_version": "renderer-v1",
        "prompt_sha256": f"prompt-{lesson_id}-{probe_id}",
        "precision": "nf4-bfloat16",
        "model_name": "model",
        "model_revision": "revision",
        "candidates": candidates,
        **outcome,
        "correct": outcome["predicted_action"] == expected,
    }


def _adapter_fc_row(
    base: dict[str, object],
    scores: tuple[float, float, float, float],
) -> dict[str, object]:
    row = _fc_row(
        scores,
        lesson_id=str(base["lesson_id"]),
        probe_id=str(base["probe_id"]),
        expected=str(base["expected_action"]),
    )
    row["source_row_sha256"] = json_hash(base)
    return row


def _crd_row(
    scores: tuple[float, ...],
    *,
    probe_id: str,
    endpoint: str,
    expected: str,
    ordered: tuple[str, ...],
    lesson_id: str = "lesson-1",
    route_variant: int = 1,
    slot_candidate_order: int | None = None,
    slot_content_order: int | None = None,
    action_panel_position: int | None = None,
) -> dict[str, object]:
    candidates = [
        _candidate(name, index + 20, score)
        for index, (name, score) in enumerate(zip(ordered, scores, strict=True))
    ]
    outcome = rank_crd_candidates(candidates)
    return {
        "lesson_id": lesson_id,
        "pair_id": f"pair-{lesson_id}",
        "lesson_type": "factual",
        "lesson_side": "left",
        "probe_id": probe_id,
        "endpoint": endpoint,
        "expected_candidate": expected,
        "ordered_candidates": list(ordered),
        "route_variant": route_variant,
        "target_slot": "slot_a",
        "current_marker": "marker-a" if endpoint != "retrieval_only" else None,
        "slot_candidate_order": slot_candidate_order,
        "slot_content_order": slot_content_order,
        "action_panel_position": action_panel_position,
        "prompt_sha256": f"prompt-{probe_id}",
        "source_probe_id": "source-probe",
        "source_row_key": "source-row",
        "source_row_sha256": "a" * 64,
        "source_fc_correct": False,
        "precision": "nf4-bfloat16",
        "model_name": "model",
        "model_revision": "revision",
        "candidates": candidates,
        **outcome,
        "correct": outcome["predicted_candidate"] == expected,
    }


def test_forced_choice_pair_records_corrections_regressions_and_margin() -> None:
    base_correct = _fc_row((-0.1, -1.0, -2.0, -3.0), probe_id="p1")
    adapter_wrong = _adapter_fc_row(
        base_correct,
        (-1.0, -0.1, -2.0, -3.0),
    )
    base_wrong = _fc_row((-1.0, -0.1, -2.0, -3.0), probe_id="p2")
    adapter_correct = _adapter_fc_row(
        base_wrong,
        (-0.1, -1.0, -2.0, -3.0),
    )

    pairs = build_forced_choice_pairs(
        [base_correct, base_wrong],
        [adapter_wrong, adapter_correct],
    )

    assert [row["transition"] for row in pairs] == [
        "correct_to_wrong",
        "wrong_to_correct",
    ]
    assert pairs[0]["base_signed_expected_margin"] == pytest.approx(0.9)
    assert pairs[0]["adapter_signed_expected_margin"] == pytest.approx(-0.9)
    assert pairs[0]["signed_expected_margin_delta"] == pytest.approx(-1.8)


def test_forced_choice_pair_rejects_duplicate_or_candidate_drift() -> None:
    base = _fc_row((-0.1, -1.0, -2.0, -3.0))
    adapter = _adapter_fc_row(base, (-0.1, -1.0, -2.0, -3.0))
    with pytest.raises(ValueError, match="duplicate"):
        build_forced_choice_pairs([base, deepcopy(base)], [adapter])

    drifted = deepcopy(adapter)
    drifted["candidates"][0]["token_ids"] = [999]
    with pytest.raises(ValueError, match="candidate token identities differ"):
        build_forced_choice_pairs([base], [drifted])


def test_crd_pair_requires_exact_probe_and_prompt_identity() -> None:
    base = _crd_row(
        (-0.1, -1.0),
        probe_id="route-1",
        endpoint="route_only",
        expected="slot_a",
        ordered=("slot_a", "slot_b"),
        slot_candidate_order=1,
        slot_content_order=1,
    )
    adapter = _crd_row(
        (-0.2, -1.0),
        probe_id="route-1",
        endpoint="route_only",
        expected="slot_a",
        ordered=("slot_a", "slot_b"),
        slot_candidate_order=1,
        slot_content_order=1,
    )
    pairs = build_crd_pairs([base], [adapter])
    assert pairs[0]["transition"] == "correct_to_correct"
    assert pairs[0]["signed_expected_margin_delta"] == pytest.approx(-0.1)

    adapter["prompt_sha256"] = "changed"
    with pytest.raises(ValueError, match="prompt_sha256"):
        build_crd_pairs([base], [adapter])


def test_crd_row_must_match_locked_bank_design_fields() -> None:
    row = _crd_row(
        (-0.1, -1.0),
        probe_id="route-1",
        endpoint="route_only",
        expected="slot_a",
        ordered=("slot_a", "slot_b"),
        slot_candidate_order=1,
        slot_content_order=1,
    )
    prompt = "locked route prompt"
    bank_record = {
        field: row[field]
        for field in (
            "probe_id",
            "lesson_id",
            "pair_id",
            "lesson_type",
            "lesson_side",
            "endpoint",
            "expected_candidate",
            "ordered_candidates",
            "route_variant",
            "target_slot",
            "current_marker",
            "slot_candidate_order",
            "slot_content_order",
            "action_panel_position",
            "source_probe_id",
            "source_row_key",
        )
    }
    bank_record["prompt"] = prompt
    row["bank_record_sha256"] = json_hash(bank_record)
    _require_crd_row_matches_bank(row, bank_record, "row")
    audit = {
        field: bank_record[field]
        for field in (
            "probe_id",
            "lesson_id",
            "endpoint",
            "expected_candidate",
            "ordered_candidates",
            "source_row_key",
            "prompt",
        )
    }
    _require_crd_audit_matches_bank(audit, bank_record, "audit")

    drifted = deepcopy(row)
    drifted["slot_content_order"] = 2
    with pytest.raises(ValueError, match="locked CRD bank"):
        _require_crd_row_matches_bank(drifted, bank_record, "row")
    audit["prompt"] = "changed"
    with pytest.raises(ValueError, match="locked CRD bank"):
        _require_crd_audit_matches_bank(audit, bank_record, "audit")


def test_paired_summary_is_lesson_clustered_and_deterministic() -> None:
    pairs = [
        {
            "lesson_id": "l1",
            "base_correct": True,
            "adapter_correct": False,
            "transition": "correct_to_wrong",
            "prediction_changed": True,
            "signed_expected_margin_delta": -1.0,
            "base_tie": False,
            "adapter_tie": False,
            "base_tie_expected_compatible": False,
            "adapter_tie_expected_compatible": False,
        },
        {
            "lesson_id": "l1",
            "base_correct": True,
            "adapter_correct": True,
            "transition": "correct_to_correct",
            "prediction_changed": False,
            "signed_expected_margin_delta": -0.5,
            "base_tie": False,
            "adapter_tie": False,
            "base_tie_expected_compatible": False,
            "adapter_tie_expected_compatible": False,
        },
        {
            "lesson_id": "l2",
            "base_correct": False,
            "adapter_correct": True,
            "transition": "wrong_to_correct",
            "prediction_changed": True,
            "signed_expected_margin_delta": 1.0,
            "base_tie": True,
            "adapter_tie": False,
            "base_tie_expected_compatible": True,
            "adapter_tie_expected_compatible": False,
        },
        {
            "lesson_id": "l2",
            "base_correct": False,
            "adapter_correct": False,
            "transition": "wrong_to_wrong",
            "prediction_changed": False,
            "signed_expected_margin_delta": 0.5,
            "base_tie": False,
            "adapter_tie": True,
            "base_tie_expected_compatible": False,
            "adapter_tie_expected_compatible": True,
        },
    ]
    first = paired_summary(
        pairs,
        bootstrap_samples=200,
        bootstrap_seed=17,
    )
    second = paired_summary(
        pairs,
        bootstrap_samples=200,
        bootstrap_seed=17,
    )
    assert first == second
    assert first["adapter_minus_base_accuracy"] == 0.0
    assert first["paired_accuracy_difference"]["cluster_count"] == 2
    assert first["transition_counts"] == {
        "correct_to_correct": 1,
        "correct_to_wrong": 1,
        "wrong_to_correct": 1,
        "wrong_to_wrong": 1,
    }
    assert first["tie_sensitivity"]["base"]["all_ties_compatible_accuracy"] == 0.75
    assert first["tie_sensitivity"]["adapter"]["all_ties_compatible_accuracy"] == 0.75


def test_exact_mcnemar_uses_two_sided_binomial_tail() -> None:
    result = exact_mcnemar(1, 3)
    assert result["discordant_count"] == 4
    assert result["two_sided_exact_p"] == pytest.approx(0.625)
    assert exact_mcnemar(0, 0)["two_sided_exact_p"] == 1.0


def test_combined_taxonomy_separates_composition_from_route_association() -> None:
    route_a = _crd_row(
        (-0.1, -2.0),
        probe_id="route-a",
        endpoint="route_only",
        expected="slot_a",
        ordered=("slot_a", "slot_b"),
        slot_candidate_order=1,
        slot_content_order=1,
    )
    route_b = _crd_row(
        (-2.0, -0.1),
        probe_id="route-b",
        endpoint="route_only",
        expected="slot_a",
        ordered=("slot_b", "slot_a"),
        slot_candidate_order=2,
        slot_content_order=1,
    )
    retrieval = _crd_row(
        (-0.1, -2.0, -3.0, -4.0),
        probe_id="retrieval",
        endpoint="retrieval_only",
        expected="act_a",
        ordered=("act_a", "act_b", "act_c", "act_d"),
        action_panel_position=1,
    )
    combined = _crd_row(
        (-2.0, -0.1, -3.0, -4.0),
        probe_id="combined",
        endpoint="combined",
        expected="act_a",
        ordered=("act_a", "act_b", "act_c", "act_d"),
        slot_content_order=1,
        action_panel_position=1,
    )

    summary, failures = build_combined_failure_taxonomy(
        [route_a, route_b, retrieval, combined],
        model_label="adapter",
        bootstrap_samples=100,
        bootstrap_seed=9,
    )

    assert summary["failure_category_counts"] == {"composition_inconsistent": 1}
    assert failures[0]["route_all_correct"] is True
    assert failures[0]["retrieval_correct"] is True
    assert failures[0]["failure_category"] == "composition_inconsistent"


def test_definite_route_association_excludes_mixed_routes_and_combined_ties() -> None:
    route_correct = _crd_row(
        (-0.1, -2.0),
        probe_id="route-correct",
        endpoint="route_only",
        expected="slot_a",
        ordered=("slot_a", "slot_b"),
        slot_candidate_order=1,
        slot_content_order=1,
    )
    route_wrong = _crd_row(
        (-0.1, -2.0),
        probe_id="route-wrong",
        endpoint="route_only",
        expected="slot_a",
        ordered=("slot_b", "slot_a"),
        slot_candidate_order=2,
        slot_content_order=1,
    )
    retrieval = _crd_row(
        (-0.1, -2.0, -3.0, -4.0),
        probe_id="retrieval",
        endpoint="retrieval_only",
        expected="act_a",
        ordered=("act_a", "act_b", "act_c", "act_d"),
        action_panel_position=1,
    )
    combined = _crd_row(
        (-2.0, -0.1, -3.0, -4.0),
        probe_id="combined",
        endpoint="combined",
        expected="act_a",
        ordered=("act_a", "act_b", "act_c", "act_d"),
        slot_content_order=1,
        action_panel_position=1,
    )
    mixed_summary, _ = build_combined_failure_taxonomy(
        [route_correct, route_wrong, retrieval, combined],
        model_label="adapter",
        bootstrap_samples=20,
        bootstrap_seed=4,
    )
    assert mixed_summary["route_ambiguous_count"] == 1
    assert mixed_summary["route_association_excluded_ambiguous_count"] == 1
    assert mixed_summary["route_association_eligible_count"] == 0

    route_second_correct = _crd_row(
        (-2.0, -0.1),
        probe_id="route-second-correct",
        endpoint="route_only",
        expected="slot_a",
        ordered=("slot_b", "slot_a"),
        slot_candidate_order=2,
        slot_content_order=1,
    )
    combined_tie = _crd_row(
        (-0.1, -0.1, -3.0, -4.0),
        probe_id="combined-tie",
        endpoint="combined",
        expected="act_a",
        ordered=("act_a", "act_b", "act_c", "act_d"),
        slot_content_order=1,
        action_panel_position=1,
    )
    tie_summary, _ = build_combined_failure_taxonomy(
        [route_correct, route_second_correct, retrieval, combined_tie],
        model_label="adapter",
        bootstrap_samples=20,
        bootstrap_seed=5,
    )
    assert tie_summary["combined_tie_count"] == 1
    assert tie_summary["route_association_excluded_combined_non_ok_count"] == 1
    assert tie_summary["route_association_eligible_count"] == 0


def test_retrieval_to_combined_margin_shift_pairs_by_position() -> None:
    retrieval = _crd_row(
        (-0.1, -2.0, -3.0, -4.0),
        probe_id="retrieval",
        endpoint="retrieval_only",
        expected="act_a",
        ordered=("act_a", "act_b", "act_c", "act_d"),
        action_panel_position=1,
    )
    combined = _crd_row(
        (-1.5, -0.1, -3.0, -4.0),
        probe_id="combined",
        endpoint="combined",
        expected="act_a",
        ordered=("act_a", "act_b", "act_c", "act_d"),
        slot_content_order=1,
        action_panel_position=1,
    )

    result = retrieval_to_combined_margin_shift(
        [retrieval, combined],
        bootstrap_samples=50,
        bootstrap_seed=3,
    )

    assert result["paired_combined_row_count"] == 1
    assert result["retrieval_correct_combined_wrong_count"] == 1
    assert result["mean_combined_minus_retrieval_signed_margin"] == pytest.approx(-3.3)


def test_paired_composition_margin_interaction_is_difference_in_differences() -> None:
    shared = {
        "lesson_id": "lesson-1",
        "route_variant": 1,
        "action_panel_position": 1,
        "expected_candidate": "act_a",
        "ordered_candidates": ["act_a", "act_b", "act_c", "act_d"],
    }
    retrieval_pair = {
        **shared,
        "probe_id": "retrieval",
        "endpoint": "retrieval_only",
        "signed_expected_margin_delta": -0.2,
    }
    combined_pair = {
        **shared,
        "probe_id": "combined",
        "endpoint": "combined",
        "signed_expected_margin_delta": -1.7,
    }
    result = paired_composition_margin_interaction(
        [retrieval_pair, combined_pair],
        bootstrap_samples=20,
        bootstrap_seed=2,
    )
    assert result["paired_combined_row_count"] == 1
    assert result["mean_interaction"] == pytest.approx(-1.5)


def test_signed_margin_returns_none_for_missing_or_nonfinite_scores() -> None:
    row = _fc_row((-0.1, -1.0, -2.0, -3.0))
    assert signed_expected_margin(row, "expected_action") == pytest.approx(0.9)
    row["candidates"][0]["sum_logprob"] = float("nan")
    assert signed_expected_margin(row, "expected_action") is None


def test_finite_tie_policy_accepts_exact_tie_and_rejects_fatal_scores() -> None:
    exact_tie = _fc_row((-0.1, -0.1, -2.0, -3.0))
    _validate_finite_tie_policy([exact_tie], "finite tie")

    non_finite = deepcopy(exact_tie)
    non_finite["error_status"] = "non_finite"
    non_finite["non_finite"] = True
    with pytest.raises(ValueError, match="only finite ok/tie"):
        _validate_finite_tie_policy([non_finite], "non-finite")


def test_finite_tie_policy_rejects_candidate_token_collision() -> None:
    exact_tie = _fc_row((-0.1, -0.1, -2.0, -3.0))
    exact_tie["candidates"][1]["token_ids"] = exact_tie["candidates"][0]["token_ids"]
    with pytest.raises(ValueError, match="not distinct"):
        _validate_finite_tie_policy([exact_tie], "token collision")


def test_dev_status_policy_does_not_require_candidate_order_field() -> None:
    _validate_admissible_statuses(
        [{"error_status": "ok", "non_finite": False}],
        "dev",
    )


def test_tie_upper_bound_only_counts_expected_candidate_in_top_set() -> None:
    base = _fc_row((-1.0, -0.1, -0.1, -3.0))
    adapter = _adapter_fc_row(base, (-1.0, -0.1, -0.1, -3.0))
    pairs = build_forced_choice_pairs([base], [adapter])
    result = paired_summary(
        pairs,
        bootstrap_samples=20,
        bootstrap_seed=1,
    )
    assert result["tie_sensitivity"]["base"]["tie_count"] == 1
    assert result["tie_sensitivity"]["base"]["expected_compatible_tie_count"] == 0
    assert result["tie_sensitivity"]["base"]["all_ties_compatible_accuracy"] == 0.0


def test_analysis_output_must_be_independent_from_all_transitive_sources(
    tmp_path: Path,
) -> None:
    rr = tmp_path / "rr"
    base = tmp_path / "base"
    frozen_fc = tmp_path / "frozen-fc"
    independent = tmp_path / "analysis"
    _require_independent_output((rr, base, frozen_fc), independent)

    with pytest.raises(ValueError, match="independent"):
        _require_independent_output(
            (rr, base, frozen_fc),
            frozen_fc / "analysis",
        )
    with pytest.raises(ValueError, match="mutually independent"):
        _require_independent_output(
            (rr, rr / "nested-base", frozen_fc),
            independent,
        )


def test_base_crd_revision_lock_default_requires_standard_pipeline_layout(
    tmp_path: Path,
) -> None:
    output = (
        tmp_path
        / "pipelines"
        / "pipeline-crd1"
        / "runs"
        / "provenance-key"
        / "route_decomposition-crd1"
    )
    assert _default_base_crd_revision_lock(output) == (
        tmp_path / "pipelines" / "pipeline-crd1" / "code_revision.txt"
    )
    with pytest.raises(ValueError, match="pass --base-crd-code-revision-lock"):
        _default_base_crd_revision_lock(tmp_path / "base")


def test_complete_analysis_contract_covers_all_frozen_matrix_counts() -> None:
    base_crd: list[dict[str, object]] = []
    adapter_crd: list[dict[str, object]] = []
    for lesson_index in range(24):
        lesson_id = f"lesson-{lesson_index:02d}"
        for route_variant in range(1, 5):
            for content_order in (1, 2):
                for candidate_order in (1, 2):
                    ordered = ("slot_a", "slot_b") if candidate_order == 1 else ("slot_b", "slot_a")
                    scores = tuple(-0.1 if value == "slot_a" else -2.0 for value in ordered)
                    probe_id = (
                        f"{lesson_id}-r{route_variant}-route-so{content_order}-co{candidate_order}"
                    )
                    base = _crd_row(
                        scores,
                        lesson_id=lesson_id,
                        probe_id=probe_id,
                        endpoint="route_only",
                        expected="slot_a",
                        ordered=ordered,
                        route_variant=route_variant,
                        slot_candidate_order=candidate_order,
                        slot_content_order=content_order,
                    )
                    base_crd.append(base)
                    adapter_crd.append(deepcopy(base))
            for action_position in range(1, 5):
                distractors = ["act_b", "act_c", "act_d"]
                action_order = distractors[:]
                action_order.insert(action_position - 1, "act_a")
                ordered_actions = tuple(action_order)
                correct_scores = tuple(
                    -0.1 if value == "act_a" else -2.0 for value in ordered_actions
                )
                retrieval_id = f"{lesson_id}-r{route_variant}-retrieval-ap{action_position}"
                base_retrieval = _crd_row(
                    correct_scores,
                    lesson_id=lesson_id,
                    probe_id=retrieval_id,
                    endpoint="retrieval_only",
                    expected="act_a",
                    ordered=ordered_actions,
                    route_variant=route_variant,
                    action_panel_position=action_position,
                )
                base_crd.append(base_retrieval)
                adapter_crd.append(deepcopy(base_retrieval))
                for content_order in (1, 2):
                    combined_id = (
                        f"{lesson_id}-r{route_variant}-combined-"
                        f"ap{action_position}-so{content_order}"
                    )
                    base_combined = _crd_row(
                        correct_scores,
                        lesson_id=lesson_id,
                        probe_id=combined_id,
                        endpoint="combined",
                        expected="act_a",
                        ordered=ordered_actions,
                        route_variant=route_variant,
                        slot_content_order=content_order,
                        action_panel_position=action_position,
                    )
                    base_crd.append(base_combined)
                    adapter_crd.append(deepcopy(base_combined))

    crd_pairs = build_crd_pairs(base_crd, adapter_crd)
    forced_choice_pairs = [
        {
            "lesson_id": f"lesson-{lesson_index:02d}",
            "category": "conditional_route",
            "arm": "external",
            "expected_candidate": "act_a",
            "base_predicted_candidate": "act_a",
            "adapter_predicted_candidate": "act_a",
            "base_correct": True,
            "adapter_correct": True,
            "base_error_status": "ok",
            "adapter_error_status": "ok",
            "base_tie": False,
            "adapter_tie": False,
            "base_tie_expected_compatible": False,
            "adapter_tie_expected_compatible": False,
            "base_signed_expected_margin": 1.0,
            "adapter_signed_expected_margin": 1.0,
            "signed_expected_margin_delta": 0.0,
            "prediction_changed": False,
            "transition": "correct_to_correct",
        }
        for lesson_index in range(24)
        for _ in range(4)
    ]

    analysis, failures = build_paired_analysis(
        forced_choice_pairs,
        crd_pairs,
        base_crd,
        adapter_crd,
        bootstrap_samples=20,
        bootstrap_seed=5,
    )

    assert len(crd_pairs) == 1_536
    assert analysis["crd"]["by_endpoint"]["route_only"]["decision_count"] == 384
    assert analysis["crd"]["by_endpoint"]["retrieval_only"]["decision_count"] == 384
    assert analysis["crd"]["by_endpoint"]["combined"]["decision_count"] == 768
    assert analysis["component_cells"]["adapter"]["combined_failure_count"] == 0
    assert (
        analysis["component_cells"]["base_to_adapter_composition_margin_interaction"][
            "mean_interaction"
        ]
        == 0.0
    )
    assert failures == []

    report = _render_report(
        {
            "run_id": "audit-run",
            "paper_evidence_ready": False,
            "source": {
                "route_remediation": {
                    "run_id": "rr-run",
                    "preregistration_sha256": "a" * 64,
                },
                "base_crd": {"run_id": "base-run"},
            },
            "historical_preregistration_observation": {
                "status": "historical_observation_differs_from_current_disk",
                "blocks_paper_use_pending_reconciliation": True,
            },
            "analysis": analysis,
        }
    )
    assert "Base-to-adapter paired differences" in report
    assert "Base ties / interval" in report
    assert "Signed expected-candidate margins" in report
    assert "Composition margin interaction" in report
    assert "not causal adapter effects" in report
