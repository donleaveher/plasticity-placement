from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from plasticity_placement.pathmem_ropcd_p0 import bundle as p0_bundle
from plasticity_placement.pathmem_ropcd_p0 import repair
from plasticity_placement.pathmem_ropcd_p0.cli import build_parser
from plasticity_placement.pathmem_ropcd_p0.config import PLAN_MANIFEST_SCHEMA_VERSION

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPOSITORY_ROOT / "notebooks/pathmem_ropcd_p0_repair/build_notebook.py"
NOTEBOOK_PATH = REPOSITORY_ROOT / (
    "notebooks/pathmem_ropcd_p0_repair/pathmem_ropcd_p0_analysis_repair_colab.ipynb"
)


def _builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ropcd_p0_repair_notebook_builder", BUILDER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repair_permissions_are_cpu_analysis_only() -> None:
    permissions = repair.repair_permissions()
    assert permissions["read_immutable_p0_results_authorized"] is True
    assert permissions["recompute_frozen_p0_aggregate_authorized"] is True
    for field in (
        "source_artifact_mutation_authorized",
        "training_authorized",
        "gpu_inference_authorized",
        "candidate_rescoring_authorized",
        "p1_authorized",
        "kill_or_reserve_access_authorized",
        "learned_router_authorized",
        "rl_controller_authorized",
        "automatic_hyperparameter_search_authorized",
    ):
        assert permissions[field] is False


def test_repair_requires_a_separate_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        repair,
        "_source_snapshot",
        lambda **_: pytest.fail("source should not be opened for an unsafe output"),
    )
    with pytest.raises(ValueError, match="separate output"):
        repair.inspect_analysis_repair(
            bundle_root=tmp_path / "bundle",
            parent_manifest=tmp_path / "parent.json",
            g1c_run_root=tmp_path / "g1c",
            plan_root=tmp_path / "plan",
            output_root=tmp_path / "run",
            repair_output_root=tmp_path / "run",
        )
    with pytest.raises(ValueError, match="separate output"):
        repair.inspect_analysis_repair(
            bundle_root=tmp_path / "bundle",
            parent_manifest=tmp_path / "parent.json",
            g1c_run_root=tmp_path / "g1c",
            plan_root=tmp_path / "plan",
            output_root=tmp_path / "run",
            repair_output_root=tmp_path / "run/repair",
        )


def test_recorded_plan_verifier_does_not_relax_normal_code_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plan_root = tmp_path / "plan"
    plan_root.mkdir()
    handoff = {"handoff_id": "h" * 64, "run_id": "r" * 64}
    plan = {
        "plan_id": "p" * 64,
        "counts": {"units": 44},
        "permissions": {"training_authorized": False},
    }
    (plan_root / "g1c_handoff.json").write_text(json.dumps(handoff), encoding="utf-8")
    (plan_root / "p0_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    recorded_implementation = {"git_revision": "1" * 40, "source_files": {}}
    identity = {
        "schema_version": PLAN_MANIFEST_SCHEMA_VERSION,
        "g1c_handoff_id": handoff["handoff_id"],
        "plan_id": plan["plan_id"],
        "implementation": recorded_implementation,
        "implementation_sha256": repair.json_hash(recorded_implementation),
        "files": {
            "g1c_handoff.json": repair.file_hash(plan_root / "g1c_handoff.json"),
            "p0_plan.json": repair.file_hash(plan_root / "p0_plan.json"),
        },
        "permissions": plan["permissions"],
    }
    manifest = {**identity, "manifest_id": repair.json_hash(identity)}
    (plan_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(p0_bundle, "verify_g1c_handoff", lambda **_: handoff)
    monkeypatch.setattr(p0_bundle, "compile_p0_plan", lambda _: plan)
    monkeypatch.setattr(p0_bundle, "audit_p0_plan", lambda _: {"passed": True})
    monkeypatch.setattr(p0_bundle, "_verify_git_implementation", lambda _: None)
    monkeypatch.setattr(
        p0_bundle,
        "implementation_identity",
        lambda: ({"git_revision": "2" * 40, "source_files": {}}, "i" * 64),
    )
    common = {
        "bundle_root": tmp_path / "bundle",
        "parent_manifest": tmp_path / "parent.json",
        "g1c_run_root": tmp_path / "g1c",
        "plan_root": plan_root,
    }
    assert p0_bundle.verify_recorded_plan_bundle(**common)["passed"] is True
    with pytest.raises(ValueError, match="planning implementation changed"):
        p0_bundle.verify_plan_bundle(**common)


def test_repair_authorization_binds_source_rule_and_implementation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_root = tmp_path / "run"
    output_root.mkdir()
    (output_root / "manifest.json").write_text("manifest\n", encoding="utf-8")
    (output_root / "authorization.json").write_text("authorization\n", encoding="utf-8")
    monkeypatch.setattr(
        repair,
        "implementation_identity",
        lambda: ({"git_revision": "1" * 40, "source_files": {}}, "i" * 64),
    )
    context = SimpleNamespace(
        bundle={"plan": {"plan_id": "p" * 64}},
        manifest=SimpleNamespace(
            payload={
                "run_id": "r" * 64,
                "identity": {
                    "plan_manifest_id": "q" * 64,
                    "authorization_id": "a" * 64,
                    "implementation_sha256": "o" * 64,
                },
            }
        ),
    )
    rows = [
        {
            "row_id": f"row-{index}",
            "arm": "path_qualification" if index < 64 else "other",
            "obsolete_action": None if index < 64 else "obsolete",
        }
        for index in range(2_168)
    ]
    monkeypatch.setattr(
        repair,
        "_source_snapshot",
        lambda **_: (context, rows, {"intact": True}),
    )
    template = repair.analysis_repair_authorization_template(
        bundle_root=tmp_path / "bundle",
        parent_manifest=tmp_path / "parent.json",
        g1c_run_root=tmp_path / "g1c",
        plan_root=tmp_path / "plan",
        output_root=output_root,
        repair_output_root=tmp_path / "repair",
    )
    assert template["source"]["matrix_sha256"] == repair.json_hash(rows)
    assert template["source"]["rows"] == 2_168
    assert template["repair_rule"]["repair_rule_id"] == repair.REPAIR_RULE_ID
    assert template["repair_rule"]["probability_outcomes_used_for_label_recovery"] is False
    assert template["implementation_sha256"] == "i" * 64
    assert template["permissions"] == repair.repair_permissions()


def test_repair_aggregate_and_verify_never_touch_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "manifest.json").write_text("manifest\n", encoding="utf-8")
    (source_root / "authorization.json").write_text("authorization\n", encoding="utf-8")
    marker = source_root / "immutable-marker.txt"
    marker.write_text("unchanged\n", encoding="utf-8")
    repair_root = tmp_path / "repair"
    context = SimpleNamespace(
        manifest=SimpleNamespace(
            payload={
                "run_id": "r" * 64,
                "identity": {
                    "plan_manifest_id": "q" * 64,
                    "authorization_id": "e" * 64,
                    "implementation_sha256": "o" * 64,
                },
            }
        ),
        bundle={"plan": {"plan_id": "p" * 64}},
    )
    rows = [{"row_id": "row-1"}]
    authorization = {
        "authorization_id": "a" * 64,
        "source": repair._source_boundary(source_root),
    }
    summary = {"gate": {"passed": True}, "analysis_repair": {"source_matrix_sha256": "x"}}
    monkeypatch.setattr(repair, "_source_snapshot", lambda **_: (context, rows, {"ok": True}))
    monkeypatch.setattr(
        repair,
        "_authorization_template_from_snapshot",
        lambda **_: {"source": authorization["source"]},
    )
    monkeypatch.setattr(repair, "_verify_repair_authorization", lambda **_: authorization)
    monkeypatch.setattr(repair, "_repaired_summary", lambda *args: summary)
    monkeypatch.setattr(repair, "_repair_report", lambda _: "repair report\n")
    monkeypatch.setattr(
        repair,
        "implementation_identity",
        lambda: ({"git_revision": "1" * 40, "source_files": {}}, "i" * 64),
    )
    common = {
        "bundle_root": tmp_path / "bundle",
        "parent_manifest": tmp_path / "parent.json",
        "g1c_run_root": tmp_path / "g1c",
        "plan_root": tmp_path / "plan",
        "output_root": source_root,
        "repair_output_root": repair_root,
    }
    assert repair.aggregate_analysis_repair(**common) == summary
    verification = repair.verify_analysis_repair(**common)
    assert verification["passed"] is True
    assert marker.read_text(encoding="utf-8") == "unchanged\n"
    assert (repair_root / "aggregate/summary.json").is_file()
    assert (repair_root / "repair_manifest.json").is_file()


def test_cli_exposes_bounded_analysis_repair_lifecycle() -> None:
    parser = build_parser()
    common = [
        "--bundle",
        "bundle",
        "--g0-v1-manifest",
        "parent.json",
        "--g1c-run",
        "g1c",
        "--plan",
        "plan",
        "--output",
        "run",
        "--repair-output",
        "repair",
    ]
    for command in (
        "repair-inspect",
        "repair-authorization-template",
        "repair-aggregate",
        "repair-verify",
    ):
        assert parser.parse_args([command, *common]).command == command
    adopted = parser.parse_args(
        ["repair-adopt-authorization", *common, "--approval", "approval.json"]
    )
    assert adopted.command == "repair-adopt-authorization"


def test_generated_repair_notebook_is_exact_cpu_only_and_parseable() -> None:
    expected = _builder().build_notebook()
    observed = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert observed == expected
    assert observed["metadata"]["accelerator"] == "CPU"
    source = "\n".join(
        "".join(cell["source"]) for cell in observed["cells"] if cell["cell_type"] == "code"
    )
    for cell in observed["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    assert "RUN_INSPECT = True" in source
    for flag in ("RUN_AUTHORIZE", "RUN_AGGREGATE", "RUN_VERIFY"):
        assert f"{flag} = False" in source
    assert "repair-adopt-authorization" in source
    assert "CODE_REVISION_LOCK" in source
    assert "nvidia-smi" not in source
    assert '"training_authorized": False' in source
    assert '"gpu_inference_authorized": False' in source
    assert '"p1_authorized": False' in source
    for forbidden in ("RUN_TRAIN", "RUN_EVALUATE", "RUN_P1", "RUN_KILL", "RUN_RL"):
        assert forbidden not in source
