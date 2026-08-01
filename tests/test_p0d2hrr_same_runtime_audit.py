from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from plasticity_placement.p0d2hrr.same_runtime_audit import (
    SameRuntimeAuditRequest,
    _publish_manifest,
    _require_independent_new_output,
    _stage_artifacts,
    classify_same_runtime_result,
    compare_sentinel_scores,
    score_adapter_off_on,
)


class _FakePeftModel:
    def __init__(self) -> None:
        self.adapter_enabled = True
        self.events: list[str] = []

    @contextmanager
    def disable_adapter(self):
        assert self.adapter_enabled
        self.adapter_enabled = False
        self.events.append("off-enter")
        try:
            yield
        finally:
            self.adapter_enabled = True
            self.events.append("off-exit")


def test_score_adapter_off_on_uses_one_model_and_restores_adapter() -> None:
    model = _FakePeftModel()
    bundle = SimpleNamespace(model=model)

    def scorer(bundle, prompt, audit):
        del prompt, audit
        state = "on" if bundle.model.adapter_enabled else "off"
        bundle.model.events.append(f"score-{state}")
        return {"state": state}

    off, on = score_adapter_off_on(bundle, "prompt", {}, scorer)

    assert off == {"state": "off"}
    assert on == {"state": "on"}
    assert model.adapter_enabled is True
    assert model.events == ["off-enter", "score-off", "off-exit", "score-on"]


def _outcome(score: float, prediction: str = "a") -> dict[str, object]:
    return {
        "candidates": [
            {"action": "a", "sum_logprob": score},
            {"action": "b", "sum_logprob": score - 1.0},
        ],
        "predicted_action": prediction,
        "error_status": "ok",
    }


def test_sentinel_detects_state_replay_drift() -> None:
    assert compare_sentinel_scores(_outcome(-1.0), _outcome(-1.000001), tolerance=1e-5)["passed"]
    failed = compare_sentinel_scores(_outcome(-1.0), _outcome(-1.1), tolerance=1e-5)
    assert failed["passed"] is False
    assert failed["reason"] == "off_state_replay_mismatch"


def _metric(base: float, adapter: float, ci: tuple[float, float]) -> dict[str, object]:
    return {
        "base_accuracy": base,
        "adapter_accuracy": adapter,
        "paired_accuracy_difference": {"ci95": list(ci)},
        "tie_sensitivity": {
            "base": {"tie_count": 0},
            "adapter": {"tie_count": 0},
        },
    }


def _analysis(
    *,
    conditional: float = 0.64,
    route_adapter: float = 0.66,
    route_ci: tuple[float, float] = (0.10, 0.20),
    combined_ci: tuple[float, float] = (-0.09, -0.04),
) -> dict[str, object]:
    return {
        "forced_choice": {"external_conditional_route": _metric(0.64, conditional, (-0.02, 0.02))},
        "crd": {
            "by_endpoint": {
                "route_only": _metric(0.50, route_adapter, route_ci),
                "retrieval_only": _metric(1.0, 1.0, (-0.001, 0.001)),
                "combined": _metric(0.79, 0.73, combined_ci),
            }
        },
    }


def test_classification_confirms_composition_interference() -> None:
    decision = classify_same_runtime_result(
        _analysis(),
        combined_noninferiority_margin=0.02,
    )
    assert decision["status"] == "composition_interference_confirmed"
    assert decision["next_action"] == "design_composition_preserving_remediation"
    assert decision["mappings_per_adapter_authorized"] is False


def test_classification_requires_ci_for_noninferiority_candidate() -> None:
    decision = classify_same_runtime_result(
        _analysis(
            conditional=0.78,
            route_ci=(0.01, 0.08),
            combined_ci=(-0.01, 0.03),
        ),
        combined_noninferiority_margin=0.02,
    )
    assert decision["status"] == "same_runtime_reading_qualification_candidate"
    assert decision["checks"]["adapter_route_only_strictly_above_0_625"] is True
    assert decision["mappings_per_adapter_authorized"] is False


def test_classification_candidate_requires_route_guardrail() -> None:
    decision = classify_same_runtime_result(
        _analysis(
            conditional=0.78,
            route_adapter=0.625,
            route_ci=(0.01, 0.08),
            combined_ci=(-0.01, 0.03),
        ),
        combined_noninferiority_margin=0.02,
    )
    assert decision["status"] == "combined_regression_not_supported"
    assert decision["checks"]["adapter_route_only_strictly_above_0_625"] is False


def test_request_rejects_invalid_statistical_settings() -> None:
    with pytest.raises(ValueError, match="bootstrap_samples"):
        SameRuntimeAuditRequest(
            route_remediation_output=SimpleNamespace(),
            analysis_output=SimpleNamespace(),
            bootstrap_samples=0,
        )
    with pytest.raises(ValueError, match="noninferiority"):
        SameRuntimeAuditRequest(
            route_remediation_output=SimpleNamespace(),
            analysis_output=SimpleNamespace(),
            combined_noninferiority_margin=0.0,
        )
    for invalid in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite"):
            SameRuntimeAuditRequest(
                route_remediation_output=SimpleNamespace(),
                analysis_output=SimpleNamespace(),
                sentinel_score_tolerance=invalid,
            )


def test_output_must_be_new_and_independent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="independent"):
        _require_independent_new_output(source, source / "analysis")

    output = tmp_path / "analysis"
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        _require_independent_new_output(source, output)


def test_manifest_is_separate_commit_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "plasticity_placement.p0d2hrr.same_runtime_audit._render_report",
        lambda summary: f"report for {summary['run_id']}",
    )
    summary = {"run_id": "same-runtime-test"}
    paths, hashes = _stage_artifacts(
        analysis_output=tmp_path / "output",
        summary=summary,
        off_fc=[],
        on_fc=[],
        off_crd=[],
        on_crd=[],
        fc_pairs=[],
        crd_pairs=[],
        failure_records=[],
    )
    assert paths["summary"].is_file()
    assert not paths["manifest"].exists()

    _publish_manifest(
        paths=paths,
        hashes=hashes,
        summary=summary,
        identity={"test": True},
        source_snapshot={"source": "unchanged"},
    )
    assert paths["manifest"].is_file()
