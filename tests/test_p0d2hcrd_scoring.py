from __future__ import annotations

import math

import pytest

from plasticity_placement.p0d2hcrd.scoring import (
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


def _candidate(name: str, total: float, average: float) -> dict[str, object]:
    return {
        "candidate": name,
        "continuation": name,
        "token_ids": [1],
        "token_count": 1,
        "sum_logprob": total,
        "mean_logprob": average,
        "token_logprobs": [total],
    }


@pytest.mark.parametrize(
    "candidates",
    [
        ("slot_a", "slot_b"),
        ("act_n7", "act_p3", "act_v9", "act_k2"),
    ],
)
def test_candidate_audit_accepts_frozen_two_or_four_candidate_sets(
    candidates: tuple[str, ...],
) -> None:
    audit = audit_candidate_tokenization(
        CharacterTokenizer(),
        "<assistant>\n",
        candidates,
        evaluation_max_length=128,
    )
    assert audit["ordered_candidates"] == list(candidates)
    assert audit["all_candidates_valid"] is True
    assert len(audit["candidates"]) == len(candidates)


def test_candidate_mask_excludes_every_prompt_token() -> None:
    assert candidate_score_plan([10, 11, 20, 21], 2) == {
        "prompt_token_count": 2,
        "candidate_token_ids": [20, 21],
        "logit_start_index": 1,
        "logit_stop_index": 3,
        "scored_token_count": 2,
    }


def test_ranking_supports_route_candidates_and_explicit_invalids() -> None:
    outcome = rank_candidate_scores(
        [
            _candidate("slot_a", -1.0, -1.0),
            _candidate("slot_b", -2.0, -0.5),
        ]
    )
    assert outcome["predicted_candidate"] == "slot_a"
    assert outcome["mean_predicted_candidate"] == "slot_b"
    assert outcome["sum_mean_disagreement"] is True
    assert outcome["top1_top2_margin"] == pytest.approx(1.0)

    tied = rank_candidate_scores(
        [
            _candidate("slot_a", -1.0, -1.0),
            _candidate("slot_b", -1.0, -1.2),
        ]
    )
    assert tied["predicted_candidate"] is None
    assert tied["error_status"] == "tie"

    nonfinite = rank_candidate_scores(
        [
            _candidate("slot_a", math.inf, -1.0),
            _candidate("slot_b", -1.0, -1.0),
        ]
    )
    assert nonfinite["predicted_candidate"] is None
    assert nonfinite["error_status"] == "non_finite"
    assert nonfinite["candidates"][0]["sum_logprob"] is None

