from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from plasticity_placement.p0d2hcbr import aggregate, resume
from plasticity_placement.p0d2hcbr.config import ExperimentSpec, unit_id
from plasticity_placement.p0d2hcbr.manifest import Manifest
from plasticity_placement.p0d2hcbr.resumable_evaluation import (
    EvaluationInputs,
    _build_plan,
    _load_checkpoint_shards,
)
from plasticity_placement.p0d2hrr.io import immutable_json_write


def _manifest_fixture(tmp_path: Path) -> tuple[Path, Manifest, ExperimentSpec, dict[str, str]]:
    output = tmp_path / "training-output"
    output.mkdir()
    spec = ExperimentSpec()
    identity = {
        "preregistration_sha256": "p" * 64,
        "code_sha256": "e" * 64,
    }
    manifest = Manifest.create(
        output / "manifest.json",
        {"run_id": "cbr-run", "config": identity},
        spec.unit_ids(),
    )
    manifest.payload["state"] = "trained"
    manifest.save()
    return output, manifest, spec, identity


def _patch_resume_sources(
    monkeypatch: pytest.MonkeyPatch,
    manifest: Manifest,
    spec: ExperimentSpec,
    identity: dict[str, str],
) -> None:
    monkeypatch.setattr(resume, "load_config", lambda _: (manifest, spec, identity))
    monkeypatch.setattr(
        resume,
        "audit_all_training_units",
        lambda _: {
            "comparisons": {},
            "relative_tolerance": 0.01,
            "all_checks_passed": True,
        },
    )
    monkeypatch.setattr(resume, "current_code_hash", lambda: "r" * 64)


def test_resumable_authorization_archives_zero_artifact_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, manifest, spec, identity = _manifest_fixture(tmp_path)
    _patch_resume_sources(monkeypatch, manifest, spec, identity)
    key = unit_id("disentangled", "full_depth", 20260811)
    old_analysis = tmp_path / "legacy-eval" / key
    claim_document = {
        "schema_version": "p0d2hcbr-evaluation-claim-v1",
        "run_id": "legacy-run",
        "unit_id": key,
        "analysis_output": str(old_analysis.resolve()),
        "adapter_sha256": "a" * 64,
        "allowed_locked_evaluations": 1,
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    claim_path = output / "claims" / f"evaluation-{key}.json"
    claim_hash = immutable_json_write(claim_path, claim_document, "test claim")
    manifest.payload["evaluation_claims"][key] = {
        "state": "claimed",
        "run_id": "legacy-run",
        "analysis_output": str(old_analysis.resolve()),
        "claim_sha256": claim_hash,
        "summary_sha256": None,
    }
    manifest.save()
    analysis_root = tmp_path / "resumable-eval"
    template = resume.resumable_evaluation_template(output, analysis_root)
    assert key in template["eligible_unit_ids"]
    assert template["interrupted_zero_artifact_claims"][key]["artifact_count"] == 0

    approval = {
        **template,
        "decision": "approved",
        "approved_by": "reviewer",
        "approved_at": "2026-08-11T00:00:00+00:00",
    }
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(json.dumps(approval))
    adopted = resume.adopt_resumable_evaluation(
        output, approval_path, analysis_root
    )

    assert adopted.is_file()
    assert key not in manifest.payload["evaluation_claims"]
    archived = manifest.payload["resumable_evaluation"][
        "archived_interrupted_claims"
    ][key]
    assert Path(archived["archived_claim_path"]).is_file()
    assert not claim_path.exists()
    assert resume.validate_resumable_authorization(output)["shard_size"] == 64


def test_resumable_template_rejects_partial_legacy_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, manifest, spec, identity = _manifest_fixture(tmp_path)
    _patch_resume_sources(monkeypatch, manifest, spec, identity)
    key = unit_id("coupled", "late_matched", 20260810)
    old_analysis = tmp_path / "legacy-eval" / key
    old_analysis.mkdir(parents=True)
    (old_analysis / "partial.json").write_text("{}")
    claim_document = {
        "schema_version": "p0d2hcbr-evaluation-claim-v1",
        "run_id": "legacy-run",
        "unit_id": key,
        "analysis_output": str(old_analysis.resolve()),
    }
    claim_path = output / "claims" / f"evaluation-{key}.json"
    claim_hash = immutable_json_write(claim_path, claim_document, "test claim")
    manifest.payload["evaluation_claims"][key] = {
        "state": "claimed",
        "run_id": "legacy-run",
        "analysis_output": str(old_analysis.resolve()),
        "claim_sha256": claim_hash,
        "summary_sha256": None,
    }

    with pytest.raises(ValueError, match="contains artifacts"):
        resume.resumable_evaluation_template(output, tmp_path / "resume")


def _fake_inputs() -> EvaluationInputs:
    def probes(prefix: str, count: int) -> list[SimpleNamespace]:
        return [SimpleNamespace(probe_id=f"{prefix}-{index:04d}") for index in range(count)]

    fc_items = [
        ("lesson", SimpleNamespace(), SimpleNamespace(probe_id=f"fc-{index:04d}"))
        for index in range(96)
    ]
    return EvaluationInputs(
        heldout_probes=probes("heldout", 1_536),
        heldout_audits={},
        rab_probes=probes("rab", 192),
        rab_audits={},
        fc_items=fc_items,
        fc_audits={},
        source_rows={},
        crd_probes=probes("crd", 1_536),
        crd_audits={},
        crd_source_rows={},
        rr_config=None,
        frozen=None,
    )


def test_shard_plan_is_complete_and_adopts_only_valid_checkpoints(tmp_path: Path) -> None:
    analysis_output = tmp_path / "analysis"
    plan = _build_plan(
        analysis_output=analysis_output,
        cbr_run_id="cbr-run",
        key="unit",
        adapter_sha256="a" * 64,
        code_sha256="r" * 64,
        budget_audit={"all_checks_passed": True},
        source_snapshot={"source": "frozen"},
        shard_size=64,
        inputs=_fake_inputs(),
    )
    assert plan["prompt_count"] == 3_360
    assert plan["decision_count"] == 6_720
    assert plan["shard_count"] == 53
    first = plan["shards"][0]
    rows = [
        {"probe_id": probe_id, "adapter_state": state}
        for probe_id in first["item_ids"]
        for state in ("adapter_off", "adapter_on")
    ]
    payload = {
        "schema_version": "p0d2hcbr-resumable-score-shard-v1",
        "run_id": plan["run_id"],
        "shard_id": first["shard_id"],
        "stage": first["stage"],
        "item_ids": list(first["item_ids"]),
        "adapter_sha256": "a" * 64,
        "source_snapshot_sha256": plan["identity"]["source_snapshot_sha256"],
        "pair_order": "adapter_off_then_adapter_on_per_prompt",
        "prompt_count": 64,
        "decision_count": 128,
        "rows": rows,
        "runtime_segment": {
            "single_process_for_shard": True,
            "single_loaded_model_for_shard": True,
            "object_identity_preserved": True,
            "sentinel": {"passed": True},
        },
        "training_authorized": False,
        "mappings_per_adapter_authorized": False,
    }
    immutable_json_write(
        analysis_output / first["relative_path"], payload, "test resumable shard"
    )
    complete, pending = _load_checkpoint_shards(analysis_output, plan)
    assert len(complete) == 1
    assert len(pending) == 52

    payload["item_ids"][0] = "changed"
    (analysis_output / first["relative_path"]).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="integrity differs"):
        _load_checkpoint_shards(analysis_output, plan)


def test_aggregate_uses_each_claims_recorded_evaluation_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, manifest, spec, identity = _manifest_fixture(tmp_path)
    identity["rabx_run_id"] = "rabx-run"
    manifest.payload["config"] = identity
    manifest.payload["state"] = "evaluated"
    manifest.payload["resumable_evaluation"] = {"state": "authorized"}
    expected_roots: dict[str, str] = {}
    for index, key in enumerate(spec.unit_ids()):
        root = tmp_path / ("legacy" if index % 2 == 0 else "resumed") / key
        root.mkdir(parents=True)
        summary = {
            "schema_version": "p0d2hcbr-unit-evaluation-v1",
            "run_id": f"eval-{index}",
            "cbr_run_id": manifest.payload["run_id"],
            "unit": {"unit_id": key},
            "decision": {
                "status": "not_qualified",
                "values": {
                    "heldout_tie_worst_accuracy": 0.5,
                    "guardrail_c_to_w": {"estimate": 0.0},
                },
            },
            "source_artifacts_modified": False,
            "training_authorized": False,
            "mappings_per_adapter_authorized": False,
        }
        summary_path = root / "summary.json"
        summary_hash = immutable_json_write(summary_path, summary, "test summary")
        audit = {
            "schema_version": "p0d2hcbr-unit-evaluation-manifest-v1",
            "run_id": summary["run_id"],
            "artifacts": {
                "summary": {"path": str(summary_path), "sha256": summary_hash}
            },
        }
        audit_path = root / "evaluation_manifest.json"
        audit_hash = immutable_json_write(audit_path, audit, "test evaluation manifest")
        manifest.payload["evaluation_claims"][key] = {
            "state": "complete",
            "analysis_output": str(root.resolve()),
            "summary_sha256": summary_hash,
            "evaluation_manifest_sha256": audit_hash,
        }
        expected_roots[key] = str(root.resolve())
    manifest.save()
    monkeypatch.setattr(
        aggregate,
        "load_config",
        lambda _: (manifest, spec, identity),
    )
    monkeypatch.setattr(
        aggregate,
        "resumable_budget_audit",
        lambda _: {
            "all_checks_passed": True,
            "comparison_scope": "preregistered_parameter_matched",
        },
    )

    result = aggregate.aggregate_experiment(output, tmp_path / "unused-root")
    summary = json.loads(result.read_text())

    assert summary["unit_evaluation_roots"] == expected_roots
    assert manifest.payload["result"]["unit_evaluation_roots"] == expected_roots
    assert aggregate.verify_complete_result(output) == result
