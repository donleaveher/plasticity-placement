from __future__ import annotations

from typing import Any

from plasticity_placement.pathmem.scoring import audit_candidate_tokenization
from plasticity_placement.training.data import CompletionDataset, TrainingExample

TRAINING_SCORING_PARITY_VERSION = "pathmem-training-scoring-parity-v1"


def audit_training_scoring_parity(
    *,
    tokenizer: Any,
    formatted_prompt: str,
    completion: str,
    action_choices: tuple[str, str, str, str],
    training_max_length: int,
    evaluation_max_length: int,
) -> dict[str, Any]:
    """Verify that loss-bearing completion tokens equal formal scoring tokens."""
    prompt_with_special = _token_ids(tokenizer, formatted_prompt, add_special_tokens=True)
    prompt_without_special = _token_ids(tokenizer, formatted_prompt, add_special_tokens=False)
    completion_ids = _token_ids(tokenizer, completion, add_special_tokens=False)
    full_ids = _token_ids(tokenizer, formatted_prompt + completion, add_special_tokens=False)
    boundary_prefix_matches = full_ids[: len(prompt_without_special)] == prompt_without_special
    boundary_completion_ids = full_ids[len(prompt_without_special) :]
    boundary_suffix_matches = boundary_completion_ids == completion_ids

    scoring = audit_candidate_tokenization(
        tokenizer,
        formatted_prompt,
        action_choices,
        evaluation_max_length=evaluation_max_length,
    )
    scoring_by_action = {
        str(candidate["action"]): [int(value) for value in candidate["token_ids"]]
        for candidate in scoring["candidates"]
    }
    candidate_lengths = {
        int(candidate["token_count"]) for candidate in scoring["candidates"]
    }
    expected_loss_token_ids = [*completion_ids]
    if getattr(tokenizer, "eos_token_id", None) is not None:
        expected_loss_token_ids.append(int(tokenizer.eos_token_id))
    training_row = CompletionDataset(
        [TrainingExample(prompt=formatted_prompt, completion=completion)],
        tokenizer,
        training_max_length,
    )[0]
    training_loss_token_ids = [
        int(token_id)
        for token_id, label in zip(
            training_row["input_ids"], training_row["labels"], strict=True
        )
        if label != -100
    ]
    checks = {
        "prompt_special_token_modes_match": prompt_with_special == prompt_without_special,
        "completion_nonempty": bool(completion_ids),
        "prompt_boundary_prefix_matches": boundary_prefix_matches,
        "prompt_boundary_completion_matches": boundary_suffix_matches,
        "loss_action_tokens_equal_scoring_tokens": (
            completion in scoring_by_action and completion_ids == scoring_by_action[completion]
        ),
        "training_dataset_loss_tokens_exact": training_loss_token_ids
        == expected_loss_token_ids,
        "all_candidates_valid": bool(scoring["all_candidates_valid"]),
        "equal_candidate_token_lengths": len(candidate_lengths) == 1,
        "training_sequence_fits": len(prompt_with_special) + len(expected_loss_token_ids)
        <= training_max_length,
    }
    return {
        "version": TRAINING_SCORING_PARITY_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "completion": completion,
        "completion_token_ids": completion_ids,
        "boundary_completion_token_ids": boundary_completion_ids,
        "candidate_token_lengths": sorted(candidate_lengths),
        "training_prompt_token_count": len(prompt_with_special),
        "training_completion_token_count": len(expected_loss_token_ids),
        "training_loss_token_ids": training_loss_token_ids,
        "scoring_audit": scoring,
    }


def _token_ids(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(
        text,
        truncation=False,
        add_special_tokens=add_special_tokens,
    )
    values = encoded["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]
