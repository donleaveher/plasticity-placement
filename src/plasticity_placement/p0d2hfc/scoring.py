from __future__ import annotations

import math
import time
from hashlib import sha256
from typing import Any

from plasticity_placement.p0d2hfc.config import CANONICAL_LEADING_WHITESPACE


class CandidateTokenMismatch(ValueError):
    """Raised when formal tokenization differs from the frozen CPU audit."""


def audit_candidate_tokenization(
    tokenizer: Any,
    formatted_prompt: str,
    ordered_actions: tuple[str, ...],
    *,
    evaluation_max_length: int,
) -> dict[str, Any]:
    if len(ordered_actions) != 4 or len(set(ordered_actions)) != 4:
        raise ValueError("forced-choice scoring requires four unique actions")
    prompt_ids = _token_ids(tokenizer, formatted_prompt)
    if not prompt_ids:
        raise ValueError("formatted prompt tokenization is empty")
    candidates: list[dict[str, Any]] = []
    for action in ordered_actions:
        continuation = f"{CANONICAL_LEADING_WHITESPACE}{action}"
        standalone_ids = _token_ids(tokenizer, continuation)
        full_ids = _token_ids(tokenizer, formatted_prompt + continuation)
        prefix_verified = full_ids[: len(prompt_ids)] == prompt_ids
        concatenated_ids = full_ids[len(prompt_ids) :]
        standalone_match = concatenated_ids == standalone_ids
        decoded = tokenizer.decode(
            concatenated_ids,
            skip_special_tokens=False,
        )
        decoded_match = decoded == continuation
        full_fits = len(full_ids) <= evaluation_max_length
        if not standalone_ids:
            raise ValueError(f"candidate tokenization is empty: {action}")
        candidates.append(
            {
                "action": action,
                "continuation": continuation,
                "token_ids": standalone_ids,
                "token_count": len(standalone_ids),
                "full_input_token_count": len(full_ids),
                "prompt_prefix_verified": prefix_verified,
                "standalone_token_ids_match": standalone_match,
                "decoded_continuation": decoded,
                "decoded_continuation_matches": decoded_match,
                "full_sequence_fits": full_fits,
                "candidate_score_start_index": len(prompt_ids),
            }
        )
    all_valid = (
        len(prompt_ids) <= evaluation_max_length
        and all(
            candidate["prompt_prefix_verified"]
            and candidate["standalone_token_ids_match"]
            and candidate["decoded_continuation_matches"]
            and candidate["full_sequence_fits"]
            for candidate in candidates
        )
    )
    return {
        "prompt_sha256": sha256(formatted_prompt.encode()).hexdigest(),
        "prompt_token_ids": prompt_ids,
        "untruncated_prompt_token_count": len(prompt_ids),
        "canonical_leading_whitespace": CANONICAL_LEADING_WHITESPACE,
        "ordered_allowed_actions": list(ordered_actions),
        "candidates": candidates,
        "all_candidates_valid": all_valid,
    }


def score_candidate_batch(
    bundle: Any,
    formatted_prompt: str,
    audit: dict[str, Any],
) -> dict[str, Any]:
    prompt_ids = _token_ids(bundle.tokenizer, formatted_prompt)
    if sha256(formatted_prompt.encode()).hexdigest() != audit["prompt_sha256"]:
        raise CandidateTokenMismatch("formatted prompt hash differs from audit")
    if prompt_ids != [int(value) for value in audit["prompt_token_ids"]]:
        raise CandidateTokenMismatch("formatted prompt token IDs differ from audit")

    sequences: list[list[int]] = []
    for candidate in audit["candidates"]:
        continuation = str(candidate["continuation"])
        full_ids = _token_ids(bundle.tokenizer, formatted_prompt + continuation)
        expected_candidate_ids = [int(value) for value in candidate["token_ids"]]
        if (
            full_ids[: len(prompt_ids)] != prompt_ids
            or full_ids[len(prompt_ids) :] != expected_candidate_ids
        ):
            raise CandidateTokenMismatch(
                f"candidate token IDs differ from audit: {candidate['action']}"
            )
        sequences.append(full_ids)

    torch = bundle.torch
    device = next(bundle.model.parameters()).device
    maximum = max(len(sequence) for sequence in sequences)
    pad_id = int(bundle.tokenizer.pad_token_id)
    input_rows = [
        sequence + [pad_id] * (maximum - len(sequence))
        for sequence in sequences
    ]
    attention_rows = [
        [1] * len(sequence) + [0] * (maximum - len(sequence))
        for sequence in sequences
    ]
    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        attention_rows,
        dtype=torch.long,
        device=device,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        logits = bundle.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - started

    scored: list[dict[str, Any]] = []
    for index, (candidate, full_ids) in enumerate(
        zip(audit["candidates"], sequences, strict=True)
    ):
        token_logprobs = candidate_token_logprobs(
            logits[index, : len(full_ids), :],
            full_ids,
            len(prompt_ids),
            torch,
        )
        total = sum(token_logprobs)
        scored.append(
            {
                "action": str(candidate["action"]),
                "continuation": str(candidate["continuation"]),
                "token_ids": [int(value) for value in candidate["token_ids"]],
                "token_count": int(candidate["token_count"]),
                "sum_logprob": total,
                "mean_logprob": total / len(token_logprobs),
                "token_logprobs": token_logprobs,
            }
        )
    return {
        **rank_candidate_scores(scored),
        "latency_seconds": latency,
    }


def candidate_token_logprobs(
    logits: Any,
    full_input_ids: list[int],
    prompt_token_count: int,
    torch: Any,
) -> list[float]:
    plan = candidate_score_plan(full_input_ids, prompt_token_count)
    candidate_ids = plan["candidate_token_ids"]
    prediction_logits = logits[
        plan["logit_start_index"] : plan["logit_stop_index"],
        :,
    ]
    if int(prediction_logits.shape[0]) != len(candidate_ids):
        raise ValueError("candidate-only logit mask length mismatch")
    targets = torch.tensor(
        candidate_ids,
        dtype=torch.long,
        device=prediction_logits.device,
    )
    log_probabilities = torch.log_softmax(
        prediction_logits.float(),
        dim=-1,
    )
    selected = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
    return [float(value) for value in selected.detach().cpu().tolist()]


def candidate_score_plan(
    full_input_ids: list[int],
    prompt_token_count: int,
) -> dict[str, Any]:
    if prompt_token_count <= 0 or prompt_token_count >= len(full_input_ids):
        raise ValueError("candidate-only mask requires prompt and candidate tokens")
    return {
        "prompt_token_count": prompt_token_count,
        "candidate_token_ids": full_input_ids[prompt_token_count:],
        "logit_start_index": prompt_token_count - 1,
        "logit_stop_index": len(full_input_ids) - 1,
        "scored_token_count": len(full_input_ids) - prompt_token_count,
    }


def rank_candidate_scores(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if len(candidates) != 4:
        raise ValueError("exactly four candidate scores are required")
    ranked = [dict(candidate) for candidate in candidates]
    raw_values = [
        candidate.get(field)
        for candidate in ranked
        for field in ("sum_logprob", "mean_logprob")
    ]
    non_finite = any(
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in raw_values
    )
    if non_finite:
        for candidate in ranked:
            score_finite = all(
                isinstance(candidate.get(field), (int, float))
                and math.isfinite(float(candidate[field]))
                for field in ("sum_logprob", "mean_logprob")
            )
            candidate["score_finite"] = score_finite
            if not score_finite:
                candidate["sum_logprob"] = None
                candidate["mean_logprob"] = None
                candidate["token_logprobs"] = [
                    (
                        float(value)
                        if isinstance(value, (int, float))
                        and math.isfinite(float(value))
                        else None
                    )
                    for value in candidate.get("token_logprobs", [])
                ]
            candidate["sum_rank"] = None
            candidate["mean_rank"] = None
        return {
            "candidates": ranked,
            "predicted_action": None,
            "mean_predicted_action": None,
            "top1_top2_margin": None,
            "sum_mean_disagreement": None,
            "tie": False,
            "mean_score_tie": False,
            "non_finite": True,
            "error_status": "non_finite",
            "error_message": "one or more candidate scores are non-finite",
        }

    sum_values = [float(candidate["sum_logprob"]) for candidate in ranked]
    mean_values = [float(candidate["mean_logprob"]) for candidate in ranked]
    for candidate in ranked:
        candidate["score_finite"] = True
    _assign_ranks(ranked, sum_values, "sum_rank")
    _assign_ranks(ranked, mean_values, "mean_rank")
    ordered_sum = sorted(range(4), key=lambda index: sum_values[index], reverse=True)
    ordered_mean = sorted(
        range(4),
        key=lambda index: mean_values[index],
        reverse=True,
    )
    sum_tie = sum_values[ordered_sum[0]] == sum_values[ordered_sum[1]]
    mean_tie = mean_values[ordered_mean[0]] == mean_values[ordered_mean[1]]
    predicted = None if sum_tie else str(ranked[ordered_sum[0]]["action"])
    mean_predicted = None if mean_tie else str(ranked[ordered_mean[0]]["action"])
    return {
        "candidates": ranked,
        "predicted_action": predicted,
        "mean_predicted_action": mean_predicted,
        "top1_top2_margin": (
            sum_values[ordered_sum[0]] - sum_values[ordered_sum[1]]
        ),
        "sum_mean_disagreement": (
            None
            if predicted is None or mean_predicted is None
            else predicted != mean_predicted
        ),
        "tie": sum_tie,
        "mean_score_tie": mean_tie,
        "non_finite": False,
        "error_status": "tie" if sum_tie else "ok",
        "error_message": (
            "multiple candidates share the highest sum_logprob"
            if sum_tie
            else None
        ),
    }


def _assign_ranks(
    candidates: list[dict[str, Any]],
    values: list[float],
    field: str,
) -> None:
    ordered_values = sorted(set(values), reverse=True)
    rank_by_value = {
        value: index + 1 for index, value in enumerate(ordered_values)
    }
    for candidate, value in zip(candidates, values, strict=True):
        candidate[field] = rank_by_value[value]


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        truncation=False,
        add_special_tokens=False,
    )
    values = encoded["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]
