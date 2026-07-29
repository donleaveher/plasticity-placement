from __future__ import annotations

from typing import Any

from plasticity_placement.p0d2hcrd.precision_audit import (
    _comparison_record,
    _strict_fp32_execution,
)
from plasticity_placement.p0d2hcrd.scoring import rank_candidate_scores


def _candidate(name: str, token_id: int, score: float) -> dict[str, Any]:
    return {
        "candidate": name,
        "continuation": name,
        "token_ids": [token_id],
        "token_count": 1,
        "sum_logprob": score,
        "mean_logprob": score,
        "token_logprobs": [score],
    }


def test_fp32_comparison_records_resolved_source_tie() -> None:
    source_candidates = [
        _candidate("act_k2", 11, -1.0),
        _candidate("act_n7", 22, -1.0),
        _candidate("act_p3", 33, -3.0),
        _candidate("act_v9", 44, -4.0),
    ]
    fp32_candidates = [
        _candidate("act_k2", 11, -1.2),
        _candidate("act_n7", 22, -0.8),
        _candidate("act_p3", 33, -3.1),
        _candidate("act_v9", 44, -4.1),
    ]
    row = {
        "precision": "bfloat16",
        "probe_id": "probe-1",
        "lesson_id": "lesson-1",
        "endpoint": "combined",
        "lesson_type": "procedure_recovery",
        "route_variant": 2,
        "target_slot": "slot_b",
        "slot_content_order": 1,
        "action_panel_position": 3,
        "expected_candidate": "act_n7",
        "ordered_candidates": [
            "act_k2",
            "act_n7",
            "act_p3",
            "act_v9",
        ],
        "candidates": source_candidates,
    }
    tie = {
        "top_sum_candidates": [
            {"candidate": "act_k2"},
            {"candidate": "act_n7"},
        ]
    }
    record = _comparison_record(
        {"manifest": {"run_id": "source-run"}},
        row,
        tie,
        rank_candidate_scores(fp32_candidates),
    )
    assert record["expected_in_source_top_set"] is True
    assert record["fp32"]["error_status"] == "ok"
    assert record["fp32"]["predicted_candidate"] == "act_n7"
    assert record["fp32"]["correct"] is True
    assert record["fp32"]["top1_top2_margin"] > 0
    assert record["candidate_scores"][0]["fp32_minus_source"] < 0


class _Backend:
    def __init__(self, allow_tf32: bool) -> None:
        self.allow_tf32 = allow_tf32


class _Backends:
    def __init__(self) -> None:
        self.cuda = type("CudaBackends", (), {"matmul": _Backend(True)})()
        self.cudnn = _Backend(True)


class _FakeTorch:
    def __init__(self) -> None:
        self.backends = _Backends()
        self.precision = "medium"

    def get_float32_matmul_precision(self) -> str:
        return self.precision

    def set_float32_matmul_precision(self, value: str) -> None:
        self.precision = value


def test_strict_fp32_context_disables_and_restores_tf32() -> None:
    torch = _FakeTorch()
    with _strict_fp32_execution(torch):
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.precision == "highest"
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    assert torch.precision == "medium"
