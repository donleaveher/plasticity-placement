from __future__ import annotations

import math

import pytest

from plasticity_placement.p0d2hfc.config import CANONICAL_LEADING_WHITESPACE
from plasticity_placement.p0d2hfc.scoring import (
    audit_candidate_tokenization,
    candidate_score_plan,
    rank_candidate_scores,
)


class CharacterTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": [ord(character) for character in text]}

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        assert skip_special_tokens is False
        return "".join(chr(token_id) for token_id in token_ids)


def test_candidate_audit_freezes_empty_leading_whitespace_and_concat() -> None:
    actions = ("act_n7", "act_p3", "act_v9", "act_k2")
    audit = audit_candidate_tokenization(
        CharacterTokenizer(),
        "<assistant>\n",
        actions,
        evaluation_max_length=128,
    )
    assert CANONICAL_LEADING_WHITESPACE == ""
    assert audit["all_candidates_valid"] is True
    assert audit["ordered_allowed_actions"] == list(actions)
    for action, candidate in zip(actions, audit["candidates"], strict=True):
        assert candidate["continuation"] == action
        assert candidate["prompt_prefix_verified"] is True
        assert candidate["standalone_token_ids_match"] is True
        assert candidate["decoded_continuation_matches"] is True
        assert candidate["candidate_score_start_index"] == len("<assistant>\n")


def test_candidate_audit_rejects_full_sequence_overflow() -> None:
    audit = audit_candidate_tokenization(
        CharacterTokenizer(),
        "12345",
        ("a", "b", "c", "d"),
        evaluation_max_length=5,
    )
    assert audit["all_candidates_valid"] is False
    assert all(
        candidate["full_sequence_fits"] is False
        for candidate in audit["candidates"]
    )


def test_candidate_only_logprob_mask_excludes_prompt_tokens() -> None:
    plan = candidate_score_plan(
        [101, 102, 201, 202],
        prompt_token_count=2,
    )
    assert plan == {
        "prompt_token_count": 2,
        "candidate_token_ids": [201, 202],
        "logit_start_index": 1,
        "logit_stop_index": 3,
        "scored_token_count": 2,
    }
    # Logit position 0 predicts a prompt token and is excluded. Positions 1
    # and 2 predict candidate IDs 201 and 202 respectively.


def _candidate(
    action: str,
    total: float,
    average: float,
) -> dict[str, object]:
    return {
        "action": action,
        "continuation": action,
        "token_ids": [1],
        "token_count": 1,
        "sum_logprob": total,
        "mean_logprob": average,
        "token_logprobs": [total],
    }


def test_ranking_keeps_sum_primary_and_mean_diagnostic_separate() -> None:
    outcome = rank_candidate_scores(
        [
            _candidate("act_n7", -1.0, -1.0),
            _candidate("act_p3", -1.5, -0.5),
            _candidate("act_v9", -3.0, -1.5),
            _candidate("act_k2", -4.0, -2.0),
        ]
    )
    assert outcome["predicted_action"] == "act_n7"
    assert outcome["mean_predicted_action"] == "act_p3"
    assert outcome["top1_top2_margin"] == pytest.approx(0.5)
    assert outcome["sum_mean_disagreement"] is True
    assert outcome["error_status"] == "ok"


def test_ranking_makes_tie_and_nonfinite_explicit() -> None:
    tied = rank_candidate_scores(
        [
            _candidate("act_n7", -1.0, -1.0),
            _candidate("act_p3", -1.0, -1.1),
            _candidate("act_v9", -2.0, -2.0),
            _candidate("act_k2", -3.0, -3.0),
        ]
    )
    assert tied["predicted_action"] is None
    assert tied["tie"] is True
    assert tied["error_status"] == "tie"

    nonfinite = rank_candidate_scores(
        [
            _candidate("act_n7", math.nan, -1.0),
            _candidate("act_p3", -1.0, -1.0),
            _candidate("act_v9", -2.0, -2.0),
            _candidate("act_k2", -3.0, -3.0),
        ]
    )
    assert nonfinite["predicted_action"] is None
    assert nonfinite["non_finite"] is True
    assert nonfinite["error_status"] == "non_finite"
    assert nonfinite["candidates"][0]["sum_logprob"] is None
    assert nonfinite["candidates"][0]["score_finite"] is False
