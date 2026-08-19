from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.backends import render_fresh_session_probe
from plasticity_placement.pathmem.compiler import compile_bank
from plasticity_placement.pathmem.generator import canonical_external_note
from plasticity_placement.pathmem.io import (
    atomic_json_write,
    canonical_json_bytes,
    file_hash,
    immutable_json_write,
    immutable_jsonl_write,
    json_hash,
    read_jsonl,
)
from plasticity_placement.pathmem.schema import AccessMode, Probe
from plasticity_placement.pathmem_consolidation_exec.identity import execution_environment
from plasticity_placement.pathmem_consolidation_exec.scoring import ScoringSession, score_prompt
from plasticity_placement.pathmem_consolidation_exec.training import TrainingSession, unit_slug
from plasticity_placement.pathmem_ropcd_p1.bundle import load_plan_bundle, verify_plan_bundle
from plasticity_placement.pathmem_ropcd_p1.config import (
    BENCHMARK_MANIFEST_SCHEMA_VERSION,
    BENCHMARK_PROFILE_SCHEMA_VERSION,
    EXPECTED_BENCHMARK_UNITS,
    EXPECTED_ROWS,
    EXPECTED_UNITS,
    P1_ROPCD_RECIPE,
)
from plasticity_placement.pathmem_ropcd_p1.identity import (
    adopt_authorization,
    build_benchmark_authorization_template,
    implementation_identity,
    verify_adopted_authorization,
    verify_external_approval,
)
from plasticity_placement.pathmem_ropcd_p1.training import (
    prepare_p1_root,
    train_p1_unit,
    verify_p1_root,
    verify_p1_unit,
)


def benchmark_authorization_template(
    *, plan_root: Path, output_root: Path, **source: Any
) -> dict[str, Any]:
    report = verify_plan_bundle(plan_root=plan_root, **source)
    return build_benchmark_authorization_template(
        plan_root=plan_root, output_root=output_root, plan_report=report
    )


def adopt_benchmark_authorization(
    *, plan_root: Path, output_root: Path, approval_path: Path, **source: Any
) -> dict[str, Any]:
    template = benchmark_authorization_template(
        plan_root=plan_root, output_root=output_root, **source
    )
    approval = verify_external_approval(approval_path, expected_template=template)
    adopt_authorization(output_root, approval)
    manifest = _load_context(plan_root=plan_root, output_root=output_root, source=source)[2]
    return {
        "adopted": True,
        "authorization_id": approval["authorization_id"],
        "benchmark_id": manifest["benchmark_id"],
        "permissions": approval["permissions"],
    }


def run_benchmark(
    *, plan_root: Path, output_root: Path, max_units: int = 1, **source: Any
) -> dict[str, Any]:
    if max_units < 1 or max_units > EXPECTED_BENCHMARK_UNITS:
        raise ValueError("R-OPCD P1 benchmark max_units must be between 1 and 4")
    _, bundle, manifest, _ = _load_context(
        plan_root=plan_root, output_root=output_root, source=source
    )
    pending = [
        unit
        for unit in bundle["plan"]["benchmark_units"]
        if manifest["units"][str(unit["unit_id"])]["state"] != "verified"
    ][:max_units]
    if not pending:
        return _benchmark_progress(manifest)
    environment = execution_environment(P1_ROPCD_RECIPE)
    observed_environment = manifest.get("environment")
    if observed_environment is None:
        manifest["environment"] = environment
        _save_manifest(output_root, manifest)
    elif observed_environment.get("environment_fingerprint") != environment.get(
        "environment_fingerprint"
    ):
        raise RuntimeError("R-OPCD P1 benchmark environment changed")
    load_started = time.perf_counter()
    training = TrainingSession.load(P1_ROPCD_RECIPE)
    manifest.setdefault("overheads", {}).setdefault(
        "training_model_load_seconds", time.perf_counter() - load_started
    )
    _save_manifest(output_root, manifest)
    try:
        for unit in pending:
            unit_id = str(unit["unit_id"])
            state = manifest["units"][unit_id]["state"]
            if state == "planned":
                manifest["units"][unit_id]["state"] = "training"
                manifest["training_started"] = True
                manifest["gpu_inference_started"] = True
                _save_manifest(output_root, manifest)
            if training.torch.cuda.is_available():
                training.torch.cuda.reset_peak_memory_stats()
            root_path = _root_dir(output_root, str(unit["root_id"]))
            root = prepare_p1_root(
                training,
                root=root_path,
                run_id=str(manifest["benchmark_id"]),
                item_id=str(unit["item_id"]),
                root_id=str(unit["root_id"]),
                root_seed=int(unit["root_adapter_seed"]),
            )
            metadata = train_p1_unit(
                training,
                unit=unit,
                parent_adapter_dir=root_path / "adapter",
                root_adapter_sha256=str(root["adapter_sha256"]),
                run_id=str(manifest["benchmark_id"]),
                output_root=output_root,
            )
            manifest["units"][unit_id].update(
                {
                    "state": "trained",
                    "root_adapter_sha256": root["adapter_sha256"],
                    "training_metadata_sha256": metadata["training_metadata_sha256"],
                    "adapter_sha256": metadata["adapter_sha256"],
                    "training_seconds": metadata["elapsed_seconds"],
                    "peak_cuda_allocated_bytes": (
                        int(training.torch.cuda.max_memory_allocated())
                        if training.torch.cuda.is_available()
                        else 0
                    ),
                    "adapter_bytes": _directory_bytes(
                        output_root / "units" / unit_slug(unit_id) / "adapter"
                    ),
                }
            )
            _save_manifest(output_root, manifest)
    finally:
        training.release()
    load_started = time.perf_counter()
    scoring = ScoringSession.load(P1_ROPCD_RECIPE)
    manifest.setdefault("overheads", {}).setdefault(
        "scoring_model_load_seconds", time.perf_counter() - load_started
    )
    _save_manifest(output_root, manifest)
    bank = compile_bank()
    probes = {probe.probe_id: probe for probe in bank.probes}
    items = {item.item_id: item for item in bank.items}
    try:
        for unit in pending:
            unit_id = str(unit["unit_id"])
            scoring.activate(output_root / "units" / unit_slug(unit_id) / "adapter")
            started = time.perf_counter()
            try:
                rows = _score_benchmark_unit(
                    scoring,
                    unit,
                    probes,
                    items[str(unit["item_id"])],
                )
            finally:
                scoring.unload()
            elapsed = time.perf_counter() - started
            result_path = output_root / "results" / f"{unit_slug(unit_id)}.jsonl"
            immutable_jsonl_write(result_path, rows, "R-OPCD P1 benchmark result")
            manifest["units"][unit_id].update(
                {
                    "state": "verified",
                    "result_sha256": file_hash(result_path),
                    "scoring_seconds": elapsed,
                    "scored_prompts": len(rows),
                }
            )
            _save_manifest(output_root, manifest)
    finally:
        scoring.release()
    return _benchmark_progress(manifest)


def aggregate_benchmark(
    *, plan_root: Path, output_root: Path, **source: Any
) -> dict[str, Any]:
    report, bundle, manifest, authorization = _load_context(
        plan_root=plan_root, output_root=output_root, source=source
    )
    if any(unit["state"] != "verified" for unit in manifest["units"].values()):
        raise RuntimeError("R-OPCD P1 hardware benchmark is incomplete")
    profile = _profile(report, bundle, manifest, authorization, output_root)
    immutable_json_write(
        output_root / "resource_profile.json", profile, "R-OPCD P1 resource profile"
    )
    manifest["profile"] = {
        "state": "complete",
        "profile_id": profile["profile_id"],
        "profile_sha256": file_hash(output_root / "resource_profile.json"),
    }
    _save_manifest(output_root, manifest)
    return profile


def verify_benchmark(
    *, plan_root: Path, output_root: Path, **source: Any
) -> dict[str, Any]:
    report, bundle, manifest, authorization = _load_context(
        plan_root=plan_root, output_root=output_root, source=source
    )
    if any(unit["state"] != "verified" for unit in manifest["units"].values()):
        raise RuntimeError("R-OPCD P1 hardware benchmark is incomplete")
    expected = _profile(report, bundle, manifest, authorization, output_root)
    profile_path = output_root / "resource_profile.json"
    observed = json.loads(profile_path.read_text(encoding="utf-8"))
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise ValueError("R-OPCD P1 resource profile regeneration mismatch")
    if manifest.get("profile") != {
        "state": "complete",
        "profile_id": expected["profile_id"],
        "profile_sha256": file_hash(profile_path),
    }:
        raise ValueError("R-OPCD P1 benchmark manifest profile binding changed")
    return {
        "passed": True,
        "profile_id": expected["profile_id"],
        "plan_id": expected["plan_id"],
        "hardware_items": EXPECTED_BENCHMARK_UNITS,
        "projected_p1_wall_seconds": expected["projection"]["p1_wall_seconds"],
        "formal_p1_authorized": False,
    }


def _load_context(
    *, plan_root: Path, output_root: Path, source: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    report = verify_plan_bundle(plan_root=plan_root, **source)
    bundle = load_plan_bundle(plan_root)
    template = build_benchmark_authorization_template(
        plan_root=plan_root, output_root=output_root, plan_report=report
    )
    authorization = verify_adopted_authorization(output_root, expected_template=template)
    implementation, implementation_sha256 = implementation_identity()
    planned = {
        str(unit["unit_id"]): {
            "unit_id": str(unit["unit_id"]),
            "unit_sha256": json_hash(unit),
        }
        for unit in bundle["plan"]["benchmark_units"]
    }
    identity = {
        "schema_version": BENCHMARK_MANIFEST_SCHEMA_VERSION,
        "plan_manifest_id": report["manifest_id"],
        "plan_id": report["plan_id"],
        "authorization_id": authorization["authorization_id"],
        "implementation": implementation,
        "implementation_sha256": implementation_sha256,
        "planned_units": planned,
    }
    benchmark_id = json_hash(identity)
    path = output_root / "benchmark_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity or manifest.get("benchmark_id") != benchmark_id:
            raise ValueError("existing R-OPCD P1 benchmark identity changed")
    else:
        manifest = {
            "identity": identity,
            "benchmark_id": benchmark_id,
            "units": {
                unit_id: {**unit, "state": "planned"}
                for unit_id, unit in planned.items()
            },
            "profile": {"state": "pending"},
            "training_started": False,
            "gpu_inference_started": False,
        }
        _save_manifest(output_root, manifest)
    return report, bundle, manifest, authorization


def _profile(
    report: dict[str, Any],
    bundle: dict[str, Any],
    manifest: dict[str, Any],
    authorization: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    _verify_measurements(bundle, manifest, output_root)
    training_seconds = [float(unit["training_seconds"]) for unit in manifest["units"].values()]
    scoring_seconds_per_prompt = [
        float(unit["scoring_seconds"]) / int(unit["scored_prompts"])
        for unit in manifest["units"].values()
    ]
    train_median = statistics.median(training_seconds)
    scoring_median = statistics.median(scoring_seconds_per_prompt)
    measured_overhead = sum(float(value) for value in manifest["overheads"].values())
    identity = {
        "schema_version": BENCHMARK_PROFILE_SCHEMA_VERSION,
        "verified": True,
        "benchmark_id": manifest["benchmark_id"],
        "authorization_id": authorization["authorization_id"],
        "plan_id": report["plan_id"],
        "g2_handoff_id": report["g2_handoff_id"],
        "implementation_sha256": manifest["identity"]["implementation_sha256"],
        "hardware_split": "hardware_dev",
        "measurements": {
            "units": EXPECTED_BENCHMARK_UNITS,
            "training_seconds": training_seconds,
            "median_training_seconds_per_unit": train_median,
            "median_scoring_seconds_per_prompt": scoring_median,
            "adapter_bytes": [
                int(unit["adapter_bytes"]) for unit in manifest["units"].values()
            ],
            "peak_cuda_allocated_bytes": [
                int(unit["peak_cuda_allocated_bytes"])
                for unit in manifest["units"].values()
            ],
            "model_load_overhead_seconds": manifest["overheads"],
            "environment_fingerprint": manifest["environment"][
                "environment_fingerprint"
            ],
        },
        "projection": {
            "formal_units": EXPECTED_UNITS,
            "formal_rows": EXPECTED_ROWS,
            "measured_load_and_checkpoint_overhead_seconds": measured_overhead,
            "p1_wall_seconds": (
                EXPECTED_UNITS * train_median
                + EXPECTED_ROWS * scoring_median
                + measured_overhead
            ),
            "formula": (
                "units*median_training_seconds + rows*median_scoring_seconds "
                "+ measured_load_and_checkpoint_overhead"
            ),
            "scheduling_claim": "planning estimate only; no throughput or completion guarantee",
        },
        "scientific_outcomes_computed": False,
        "formal_p1_authorized": False,
        "p1b_authorized": False,
        "p2_authorized": False,
        "recipe_sha256": bundle["plan"]["recipe_sha256"],
    }
    return {**identity, "profile_id": json_hash(identity)}


def _score_benchmark_unit(
    session: ScoringSession,
    unit: dict[str, Any],
    probes: dict[str, Probe],
    item: Any,
) -> list[dict[str, Any]]:
    persistent_state = canonical_external_note(item, str(unit["terminal_state"]))
    rows: list[dict[str, Any]] = []
    for probe_id in unit["core_probe_ids"]:
        probe = probes[str(probe_id)]
        both_prompt = render_fresh_session_probe(
            probe, persistent_state=persistent_state, access_mode=AccessMode.ON
        )
        for arm, raw_prompt in (
            ("parametric_path", probe.prompt),
            ("both", both_prompt),
        ):
            row = score_prompt(
                session,
                raw_prompt=raw_prompt,
                action_choices=probe.action_choices,
                expected_action=probe.expected_action,
                obsolete_action=probe.obsolete_action,
                unit_id=str(unit["unit_id"]),
                item_id=str(unit["item_id"]),
                probe_id=probe.probe_id,
                probe_category=probe.category.value,
                arm=arm,
                disable_adapter=False,
                route_key=str(unit["route_key"]),
                route_unit_id=str(unit["unit_id"]),
                metadata={
                    "logical_name": unit["logical_name"],
                    "fresh_session_id": json_hash(
                        [unit["unit_id"], probe.probe_id, arm, "benchmark"]
                    ),
                },
            )
            rows.append(
                {
                    **row,
                    "training_seed": int(unit["training_seed"]),
                    "root_id": str(unit["root_id"]),
                }
            )
    return rows


def _verify_measurements(
    bundle: dict[str, Any], manifest: dict[str, Any], output_root: Path
) -> None:
    for unit in bundle["plan"]["benchmark_units"]:
        unit_id = str(unit["unit_id"])
        recorded = manifest["units"][unit_id]
        root_path = _root_dir(output_root, str(unit["root_id"]))
        root = verify_p1_root(
            root_path,
            run_id=str(manifest["benchmark_id"]),
            item_id=str(unit["item_id"]),
            root_id=str(unit["root_id"]),
            root_seed=int(unit["root_adapter_seed"]),
        )
        metadata = verify_p1_unit(
            output_root / "units" / unit_slug(unit_id),
            unit=unit,
            parent_adapter_sha256=str(root["adapter_sha256"]),
            root_adapter_sha256=str(root["adapter_sha256"]),
            run_id=str(manifest["benchmark_id"]),
        )
        result_path = output_root / "results" / f"{unit_slug(unit_id)}.jsonl"
        rows = read_jsonl(result_path, f"R-OPCD P1 benchmark result {unit_id}")
        if (
            recorded.get("state") != "verified"
            or recorded.get("adapter_sha256") != metadata["adapter_sha256"]
            or recorded.get("training_metadata_sha256")
            != metadata["training_metadata_sha256"]
            or recorded.get("result_sha256") != file_hash(result_path)
            or len(rows) != 28
            or {row.get("arm") for row in rows} != {"parametric_path", "both"}
            or any(row.get("unit_id") != unit_id for row in rows)
        ):
            raise ValueError(f"R-OPCD P1 benchmark artifact binding changed: {unit_id}")


def _save_manifest(output_root: Path, manifest: dict[str, Any]) -> None:
    atomic_json_write(output_root / "benchmark_manifest.json", manifest)


def _benchmark_progress(manifest: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for unit in manifest["units"].values():
        counts[unit["state"]] = counts.get(unit["state"], 0) + 1
    return {
        "benchmark_id": manifest["benchmark_id"],
        "unit_states": counts,
        "training_started": manifest["training_started"],
        "gpu_inference_started": manifest["gpu_inference_started"],
        "scientific_outcomes_computed": False,
    }


def _root_dir(output_root: Path, root_id: str) -> Path:
    return output_root / "roots" / unit_slug(root_id)


def _directory_bytes(path: Path) -> int:
    return sum(candidate.stat().st_size for candidate in path.rglob("*") if candidate.is_file())
