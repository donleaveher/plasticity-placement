from __future__ import annotations

import json
from pathlib import Path

import pytest

from plasticity_placement.p0d2hcbr import corrected
from plasticity_placement.p0d2hcbr.config import CURRICULA, ExperimentSpec, unit_id
from plasticity_placement.p0d2hcbr.manifest import Manifest
from plasticity_placement.p0d2hcbr.preflight import _preflight_paths, load_config
from plasticity_placement.p0d2hrr.io import file_hash

SPEC_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "p0d2hcbr-counterbalanced-binding-remediation-v2.json"
)


def test_corrected_plan_imports_six_full_controls_and_leaves_six_late_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_manifest = Manifest.create(
        source / "manifest.json",
        {"run_id": "cbr-v1", "config": {}},
        ExperimentSpec().unit_ids(),
    )
    source_manifest.payload["state"] = "complete"
    source_manifest.payload["result"] = {"summary_sha256": "s"}
    for key, entry in source_manifest.payload["training_units"].items():
        curriculum, placement, seed_label = key.split("__")
        seed = int(seed_label.removeprefix("seed-"))
        adapter = source / "adapters" / key
        adapter.mkdir(parents=True)
        config = {"unit": key}
        summary = {
            "adapter_sha256": f"adapter-{key}",
            "training_data_sha256": f"data-{curriculum}",
            "trainable_parameters": 1_089_536 if placement == "full_depth" else 1_167_360,
        }
        metadata = {"config": config, "summary": summary}
        metadata_path = adapter / "training_metadata.json"
        metadata_path.write_text(json.dumps(metadata))
        entry.update(
            {
                "state": "trained",
                "training": {
                    "summary": summary,
                    "training_metadata_sha256": file_hash(metadata_path),
                    "curriculum": curriculum,
                    "placement": placement,
                    "seed": seed,
                    "checkpoint_selection": "fixed_final_step",
                },
            }
        )
    source_manifest.save()
    source_summary = source / "aggregate" / "summary.json"
    source_summary.parent.mkdir()
    source_summary.write_text(
        json.dumps(
            {
                "decision": {"status": "budget_deviation_full_matrix_complete"},
                "placement_comparison_scope": "exploratory_parameter_count_confounded",
            }
        )
    )
    source_paths = _preflight_paths(source)
    source_hashes = {}
    for name, path in source_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        source_hashes[name] = file_hash(path)
    source_code_lock = tmp_path / "source-code.txt"
    source_code_lock.write_text("source-revision\n")
    source_identity = {
        "schema_version": "p0d2hcbr-config-v1",
        "data_hashes": source_hashes,
        "rabx_output": str(tmp_path / "rabx"),
        "rabx_run_id": "rabx",
        "rabc_output": str(tmp_path / "rabc"),
        "rab_output": str(tmp_path / "rab"),
        "source_code_revision_lock": str(source_code_lock),
        "source_code_revision_lock_sha256": file_hash(source_code_lock),
        "source_snapshot": {"source": "frozen"},
    }
    monkeypatch.setattr(corrected, "verify_complete_result", lambda _: source_summary)
    monkeypatch.setattr(
        corrected,
        "load_config",
        lambda _: (source_manifest, ExperimentSpec(), source_identity),
    )
    monkeypatch.setattr(
        corrected,
        "audit_all_training_units",
        lambda _: {"all_checks_passed": False},
    )
    monkeypatch.setattr(corrected, "corrected_source_snapshot", lambda _: {"frozen": True})
    monkeypatch.setattr(corrected, "_adapter_hash", lambda path: f"adapter-{path.name}")
    output = tmp_path / "corrected"

    manifest_path = corrected.plan_corrected_experiment(
        output_dir=output,
        source_cbr_output=source,
        spec_path=SPEC_PATH,
    )

    manifest = Manifest.load(manifest_path)
    imported = {
        key
        for key, value in manifest.payload["training_units"].items()
        if value["state"] == "trained"
    }
    pending = {
        key
        for key, value in manifest.payload["training_units"].items()
        if value["state"] == "pending"
    }
    assert imported == {
        unit_id(curriculum, "full_depth", seed)
        for curriculum in CURRICULA
        for seed in (20260810, 20260811, 20260812)
    }
    assert pending == {
        unit_id(curriculum, "late_matched", seed)
        for curriculum in CURRICULA
        for seed in (20260810, 20260811, 20260812)
    }
    assert manifest.payload["config"]["training_run_limit"] == 6
    assert manifest.payload["training_authorized"] is False
    loaded_manifest, loaded_spec, _ = load_config(output)
    assert loaded_manifest.path == manifest_path
    assert loaded_spec.training.late_explicit_layers == tuple(range(20, 28))
