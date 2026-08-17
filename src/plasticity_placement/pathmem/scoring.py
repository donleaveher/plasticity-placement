from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from plasticity_placement.p0d2hfc.scoring import (
    CandidateTokenMismatch,
    audit_candidate_tokenization,
    score_candidate_batch,
)

SCORING_VERSION = "pathmem-four-candidate-js-v1"
EPSILON_JS_NATS = 0.02
TECHNICAL_TOLERANCE_JS_NATS = 0.002
QUALIFICATION_MARGIN_NATS = math.log(2.0)


@dataclass(frozen=True, slots=True)
class CandidateDistribution:
    actions: tuple[str, str, str, str]
    probabilities: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if len(set(self.actions)) != 4:
            raise ValueError("candidate distributions require four unique actions")
        if any(not math.isfinite(value) or value < 0 for value in self.probabilities):
            raise ValueError("candidate probabilities must be finite and non-negative")
        if not math.isclose(sum(self.probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("candidate probabilities must sum to one")

    def probability(self, action: str) -> float:
        try:
            index = self.actions.index(action)
        except ValueError as error:
            raise KeyError(action) from error
        return self.probabilities[index]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_candidate_scores(
    actions: tuple[str, str, str, str],
    summed_logprobabilities: tuple[float, float, float, float],
) -> CandidateDistribution:
    if len(set(actions)) != 4:
        raise ValueError("normalization requires four unique actions")
    if any(not math.isfinite(value) for value in summed_logprobabilities):
        raise ValueError("candidate scores must be finite")
    maximum = max(summed_logprobabilities)
    weights = tuple(math.exp(value - maximum) for value in summed_logprobabilities)
    denominator = sum(weights)
    probabilities = tuple(value / denominator for value in weights)
    return CandidateDistribution(actions=actions, probabilities=probabilities)


def jensen_shannon_nats(
    first: CandidateDistribution,
    second: CandidateDistribution,
) -> float:
    if first.actions != second.actions:
        raise ValueError("JS distributions must use the same ordered actions")
    midpoint = tuple(
        (left + right) / 2.0
        for left, right in zip(first.probabilities, second.probabilities, strict=True)
    )
    divergence = 0.5 * _kl(first.probabilities, midpoint) + 0.5 * _kl(
        second.probabilities,
        midpoint,
    )
    if divergence < 0 and abs(divergence) < 1e-15:
        return 0.0
    return divergence


def mean_endpoint_defect(
    first: tuple[CandidateDistribution, ...],
    second: tuple[CandidateDistribution, ...],
) -> float:
    if not first or len(first) != len(second):
        raise ValueError("endpoint defect requires equal non-empty probe panels")
    return sum(
        jensen_shannon_nats(left, right)
        for left, right in zip(first, second, strict=True)
    ) / len(first)


def symmetric_item_defect(defect_a: float, defect_b: float) -> float:
    if any(not math.isfinite(value) or value < 0 for value in (defect_a, defect_b)):
        raise ValueError("symmetric endpoint defects must be finite and non-negative")
    return (defect_a + defect_b) / 2.0


def write_qualified(
    distributions: tuple[CandidateDistribution, ...],
    *,
    current_action: str,
    obsolete_action: str,
) -> bool:
    if len(distributions) != 4:
        raise ValueError("write qualification requires the four disjoint probes")
    margins: list[float] = []
    for distribution in distributions:
        current_probability = distribution.probability(current_action)
        obsolete_probability = distribution.probability(obsolete_action)
        top_probability = max(distribution.probabilities)
        if not math.isclose(current_probability, top_probability, abs_tol=1e-15):
            return False
        margins.append(
            math.log(max(current_probability, 1e-300))
            - math.log(max(obsolete_probability, 1e-300))
        )
    return sum(margins) / len(margins) >= QUALIFICATION_MARGIN_NATS


def _kl(probabilities: tuple[float, ...], reference: tuple[float, ...]) -> float:
    total = 0.0
    for probability, baseline in zip(probabilities, reference, strict=True):
        if probability == 0:
            continue
        total += probability * math.log(probability / baseline)
    return total


__all__ = [
    "CandidateDistribution",
    "CandidateTokenMismatch",
    "EPSILON_JS_NATS",
    "SCORING_VERSION",
    "TECHNICAL_TOLERANCE_JS_NATS",
    "audit_candidate_tokenization",
    "jensen_shannon_nats",
    "mean_endpoint_defect",
    "normalize_candidate_scores",
    "score_candidate_batch",
    "symmetric_item_defect",
    "write_qualified",
]
