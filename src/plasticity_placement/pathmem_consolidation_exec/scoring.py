from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.backends import render_fresh_session_probe
from plasticity_placement.pathmem.compiler import compile_bank
from plasticity_placement.pathmem.io import json_hash
from plasticity_placement.pathmem.schema import AccessMode
from plasticity_placement.pathmem.scoring import (
    audit_candidate_tokenization,
    normalize_candidate_scores,
    score_candidate_batch,
)
from plasticity_placement.pathmem_consolidation_exec.config import (
    G1C_EXECUTION_RECIPE,
    RopcdExecutionRecipe,
)
from plasticity_placement.training.model_utils import (
    chat_prompt,
    load_tokenizer,
    model_load_kwargs,
)


@dataclass(slots=True)
class ScoringSession:
    torch: Any
    tokenizer: Any
    model: Any
    recipe: RopcdExecutionRecipe
    active_adapter_dir: Path | None = None

    @classmethod
    def load(
        cls, recipe: RopcdExecutionRecipe = G1C_EXECUTION_RECIPE
    ) -> ScoringSession:
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as error:
            raise RuntimeError("scoring requires the train dependencies") from error
        tokenizer = load_tokenizer(recipe.model.name, recipe.model.revision)
        load_kwargs, precision = model_load_kwargs(torch, use_4bit=True)
        if precision != "nf4-bfloat16":
            raise RuntimeError(f"G1-C scoring precision changed: {precision}")
        model = AutoModelForCausalLM.from_pretrained(
            recipe.model.name,
            revision=recipe.model.revision,
            **load_kwargs,
        )
        model.eval()
        return cls(torch=torch, tokenizer=tokenizer, model=model, recipe=recipe)

    def activate(self, adapter_dir: Path) -> None:
        if self.active_adapter_dir is not None:
            raise RuntimeError("unload the active side-memory module before loading another")
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_dir,
            is_trainable=False,
        )
        self.model.eval()
        self.active_adapter_dir = adapter_dir.resolve()

    def unload(self) -> None:
        if self.active_adapter_dir is None:
            return
        self.model = self.model.unload()
        self.model.eval()
        self.active_adapter_dir = None

    def release(self) -> None:
        del self.model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def score_prompt(
    session: ScoringSession,
    *,
    raw_prompt: str,
    action_choices: tuple[str, str, str, str],
    expected_action: str,
    obsolete_action: str | None,
    unit_id: str,
    item_id: str,
    probe_id: str,
    probe_category: str,
    arm: str,
    disable_adapter: bool = False,
    route_key: str | None = None,
    route_unit_id: str | None = None,
    route_miss: bool = False,
    false_activation: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    formatted_prompt = chat_prompt(session.tokenizer, raw_prompt)
    audit = audit_candidate_tokenization(
        session.tokenizer,
        formatted_prompt,
        action_choices,
        evaluation_max_length=session.recipe.evaluation_max_length,
    )
    if not audit["all_candidates_valid"]:
        raise RuntimeError(f"G1-C candidate audit failed: {probe_id}")
    context = (
        session.model.disable_adapter()
        if disable_adapter and session.active_adapter_dir is not None
        else nullcontext()
    )
    with context:
        scored = score_candidate_batch(session, formatted_prompt, audit)
    candidates = scored["candidates"]
    scores = [
        float(candidate["sum_logprob"])
        if candidate["sum_logprob"] is not None
        else float("nan")
        for candidate in candidates
    ]
    probabilities: list[float] | None = None
    if scored["error_status"] == "ok":
        distribution = normalize_candidate_scores(action_choices, tuple(scores))
        probabilities = list(distribution.probabilities)
    predicted = scored["predicted_action"]
    return {
        "row_id": json_hash(
            {
                "unit_id": unit_id,
                "probe_id": probe_id,
                "arm": arm,
                "raw_prompt": raw_prompt,
                "active_adapter": (
                    str(session.active_adapter_dir) if session.active_adapter_dir else None
                ),
            }
        ),
        "unit_id": unit_id,
        "item_id": item_id,
        "probe_id": probe_id,
        "probe_category": probe_category,
        "arm": arm,
        "raw_prompt_sha256": json_hash(raw_prompt),
        "formatted_prompt_sha256": audit["prompt_sha256"],
        "ordered_actions": list(action_choices),
        "expected_action": expected_action,
        "obsolete_action": obsolete_action,
        "predicted_action": predicted,
        "correct": predicted == expected_action,
        "obsolete_intrusion": obsolete_action is not None and predicted == obsolete_action,
        "parser_valid": scored["error_status"] == "ok",
        "error_status": scored["error_status"],
        "tie": scored["tie"],
        "non_finite": scored["non_finite"],
        "candidate_scores": scores,
        "candidate_probabilities": probabilities,
        "candidates": candidates,
        "candidate_audit": audit,
        "active_adapter_dir": (
            str(session.active_adapter_dir) if session.active_adapter_dir else None
        ),
        "adapter_disabled": disable_adapter,
        "route_key": route_key,
        "route_unit_id": route_unit_id,
        "route_miss": route_miss,
        "false_activation": false_activation,
        "metadata": metadata or {},
    }


def score_teacher_pairs(
    session: ScoringSession, units: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit in sorted(units, key=lambda row: int(row["sequence_index"])):
        for pair in unit["rollout_prompt_pairs"]:
            rows.append(
                score_prompt(
                    session,
                    raw_prompt=str(pair["privileged_teacher_prompt"]),
                    action_choices=tuple(pair["action_choices"]),
                    expected_action=str(unit["current_action"]),
                    obsolete_action=str(unit["obsolete_action"]),
                    unit_id=str(unit["unit_id"]),
                    item_id=str(unit["item_id"]),
                    probe_id=str(pair["probe_id"]),
                    probe_category="qualification",
                    arm="teacher",
                    metadata={"teacher_has_privileged_context": True},
                )
            )
    return rows


def audit_plan_prompts(
    tokenizer: Any,
    units: list[dict[str, Any]],
    *,
    recipe: RopcdExecutionRecipe = G1C_EXECUTION_RECIPE,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    probes = {probe.probe_id: probe for probe in compile_bank().probes}
    for unit in units:
        for pair in unit["rollout_prompt_pairs"]:
            for prompt_kind in ("student_prompt", "privileged_teacher_prompt"):
                formatted = chat_prompt(tokenizer, str(pair[prompt_kind]))
                audit = audit_candidate_tokenization(
                    tokenizer,
                    formatted,
                    tuple(pair["action_choices"]),
                    evaluation_max_length=recipe.evaluation_max_length,
                )
                rows.append(
                    {
                        "unit_id": unit["unit_id"],
                        "probe_id": pair["probe_id"],
                        "prompt_kind": prompt_kind,
                        "prompt_sha256": audit["prompt_sha256"],
                        "prompt_token_count": audit["untruncated_prompt_token_count"],
                        "all_candidates_valid": audit["all_candidates_valid"],
                        "rollout_length_fits": (
                            audit["untruncated_prompt_token_count"]
                            + recipe.rollout_max_new_tokens
                            <= recipe.training_max_length
                        ),
                    }
                )
        evaluation_ids = [
            *unit["core_probe_ids"],
            *unit["near_neighbor_probe_ids"],
            *unit["unrelated_probe_ids"],
        ]
        for probe_id in evaluation_ids:
            probe = probes[str(probe_id)]
            raw_prompt = render_fresh_session_probe(
                probe,
                persistent_state="unused",
                access_mode=AccessMode.OFF,
            )
            formatted = chat_prompt(tokenizer, raw_prompt)
            audit = audit_candidate_tokenization(
                tokenizer,
                formatted,
                probe.action_choices,
                evaluation_max_length=recipe.evaluation_max_length,
            )
            rows.append(
                {
                    "unit_id": unit["unit_id"],
                    "probe_id": probe_id,
                    "prompt_kind": "evaluation_student",
                    "prompt_sha256": audit["prompt_sha256"],
                    "prompt_token_count": audit["untruncated_prompt_token_count"],
                    "all_candidates_valid": audit["all_candidates_valid"],
                    "rollout_length_fits": True,
                }
            )
    passed = len(rows) == 816 and all(
        row["all_candidates_valid"] and row["rollout_length_fits"] for row in rows
    )
    return {
        "passed": passed,
        "audit_count": len(rows),
        "maximum_prompt_tokens": max(
            (int(row["prompt_token_count"]) for row in rows), default=0
        ),
        "rows": rows,
    }
