from __future__ import annotations

from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.compiler import compile_bank
from plasticity_placement.pathmem_consolidation.manifest import verify_parent_g0_v1
from plasticity_placement.pathmem_consolidation.planner import compile_g1c_plan
from plasticity_placement.pathmem_consolidation_exec import runtime
from plasticity_placement.pathmem_consolidation_exec.training import (
    _require_finite_tensor,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PARENT_MANIFEST = (
    REPOSITORY_ROOT
    / "notebooks/pathmem_g1_p0/protocol_snapshot/experiments/g0/manifest.json"
)


class FakeSession:
    def __init__(self) -> None:
        self.active_adapter_dir: Path | None = None

    def activate(self, path: Path) -> None:
        assert self.active_adapter_dir is None
        self.active_adapter_dir = path.resolve()

    def unload(self) -> None:
        self.active_adapter_dir = None


def _plan() -> dict[str, Any]:
    parent = verify_parent_g0_v1(PARENT_MANIFEST)
    return compile_g1c_plan(
        parent_manifest_id=parent["manifest_id"],
        parent_manifest_sha256=parent["manifest_sha256"],
    )


def test_evaluation_replays_exact_retention_and_control_matrix(monkeypatch) -> None:
    plan = _plan()
    units = {str(unit["unit_id"]): unit for unit in plan["units"]}
    trained = {
        unit_id: {
            "adapter_dir": f"/tmp/{index}",
            "adapter_sha256": f"{index + 1:064x}",
        }
        for index, unit_id in enumerate(units)
    }
    probes = {probe.probe_id: probe for probe in compile_bank().probes}

    def fake_score_prompt(_session, **kwargs):
        expected = kwargs["expected_action"]
        return {
            **kwargs,
            "predicted_action": expected,
            "correct": True,
            "obsolete_intrusion": False,
            "parser_valid": True,
            "candidate_scores": [-0.1, -2.0, -3.0, -4.0],
            "candidate_audit": {"all_candidates_valid": True},
            "non_finite": False,
        }

    monkeypatch.setattr(runtime, "score_prompt", fake_score_prompt)
    unit = plan["units"][0]
    rows = runtime._evaluate_unit(
        FakeSession(),
        unit=unit,
        all_units=units,
        trained=trained,
        probes=probes,
    )
    assert len(rows) == 62
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["arm"]] = counts.get(row["arm"], 0) + 1
    assert counts == {
        "access_off": 4,
        "base_control": 12,
        "base_immediate": 4,
        "immediate": 4,
        "restored_base": 4,
        "retained": 4,
        "retained_core": 14,
        "routed_control": 12,
        "wrong_swap": 4,
    }
    retained = [row for row in rows if row["arm"] == "retained"]
    assert len({row["metadata"]["router_sha256"] for row in retained}) == 1
    assert all(row["route_miss"] is False for row in retained)
    controls = [row for row in rows if row["arm"] == "routed_control"]
    assert all(row["false_activation"] is False for row in controls)


def test_forward_kl_is_zero_for_equal_logits_and_backpropagates() -> None:
    import torch
    import torch.nn.functional as functional

    student = torch.tensor([[[1.0, 2.0, 3.0]]], requires_grad=True)
    teacher = student.detach().clone()
    loss = runtime.TrainingSession  # keep runtime import coverage explicit
    assert loss is not None
    from plasticity_placement.pathmem_consolidation_exec.training import _forward_kl

    value = _forward_kl(
        student,
        teacher,
        temperature=1.0,
        functional=functional,
    )
    assert abs(float(value.detach())) < 1e-7
    value.backward()
    assert student.grad is not None


def test_non_finite_training_values_are_rejected() -> None:
    import pytest
    import torch

    _require_finite_tensor(torch, torch.tensor(1.0), "test")
    with pytest.raises(FloatingPointError, match="non-finite"):
        _require_finite_tensor(torch, torch.tensor(float("nan")), "test")
