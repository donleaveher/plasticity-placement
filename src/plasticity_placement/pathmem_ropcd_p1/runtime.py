from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.backends import render_fresh_session_probe
from plasticity_placement.pathmem.compiler import compile_bank
from plasticity_placement.pathmem.io import (
    canonical_json_bytes,
    file_hash,
    immutable_json_write,
    immutable_jsonl_write,
    json_hash,
    read_jsonl,
)
from plasticity_placement.pathmem.schema import AccessMode, Probe
from plasticity_placement.pathmem.scoring import audit_candidate_tokenization
from plasticity_placement.pathmem_consolidation_exec.identity import execution_environment
from plasticity_placement.pathmem_consolidation_exec.scoring import (
    ScoringSession,
    score_prompt,
    score_teacher_pairs,
)
from plasticity_placement.pathmem_consolidation_exec.training import TrainingSession, unit_slug
from plasticity_placement.pathmem_ropcd_p0.runtime import (
    _render_states,
    _score_anchor,
    _score_duplicate,
    _score_final_path,
)
from plasticity_placement.pathmem_ropcd_p1.analysis import summarize_p1
from plasticity_placement.pathmem_ropcd_p1.bundle import (
    inspect_plan,
    load_plan_bundle,
    verify_plan_bundle,
)
from plasticity_placement.pathmem_ropcd_p1.config import (
    EXPECTED_ROWS,
    EXPECTED_TEACHER_ROWS,
    EXPECTED_UNITS,
    FINAL_PATHS,
    P1_ROPCD_RECIPE,
    PREFLIGHT_SCHEMA_VERSION,
    SCORED_LOGICAL_NAMES,
    SUMMARY_SCHEMA_VERSION,
)
from plasticity_placement.pathmem_ropcd_p1.identity import (
    adopt_authorization,
    build_execution_authorization_template,
    implementation_identity,
    verify_adopted_authorization,
    verify_external_approval,
)
from plasticity_placement.pathmem_ropcd_p1.manifest import P1ExecutionManifest, UnitState
from plasticity_placement.pathmem_ropcd_p1.training import (
    prepare_p1_root,
    train_p1_unit,
    verify_p1_root,
    verify_p1_unit,
)
from plasticity_placement.training.model_utils import chat_prompt


@dataclass(frozen=True, slots=True)
class RunContext:
    plan_report: dict[str, Any]
    bundle: dict[str, Any]
    authorization: dict[str, Any]
    manifest: P1ExecutionManifest


def inspect_execution(*, output_root: Path, **source: Any) -> dict[str, Any]:
    report = inspect_plan(**source)
    return {
        **report,
        "output_root": str(output_root.resolve()),
        "recipe": P1_ROPCD_RECIPE.to_dict(),
        "safe_default": "inspection_only",
    }


def authorization_template(
    *, plan_root: Path, output_root: Path, resource_profile_path: Path, **source: Any
) -> dict[str, Any]:
    report = verify_plan_bundle(plan_root=plan_root, **source)
    return build_execution_authorization_template(
        plan_root=plan_root,
        output_root=output_root,
        plan_report=report,
        resource_profile_path=resource_profile_path,
    )


def adopt_run_authorization(
    *,
    plan_root: Path,
    output_root: Path,
    resource_profile_path: Path,
    approval_path: Path,
    **source: Any,
) -> dict[str, Any]:
    template = authorization_template(
        plan_root=plan_root,
        output_root=output_root,
        resource_profile_path=resource_profile_path,
        **source,
    )
    approval = verify_external_approval(approval_path, expected_template=template)
    adopt_authorization(output_root, approval)
    context = _load_context(
        plan_root=plan_root,
        output_root=output_root,
        resource_profile_path=resource_profile_path,
        source=source,
    )
    return {
        "adopted": True,
        "authorization_id": context.authorization["authorization_id"],
        "run_id": context.manifest.payload["run_id"],
        "plan_id": context.bundle["plan"]["plan_id"],
        "resource_profile": context.authorization["resource_profile"],
        "permissions": context.authorization["permissions"],
    }


def run_preflight(
    *, plan_root: Path, output_root: Path, resource_profile_path: Path, **source: Any
) -> dict[str, Any]:
    context = _load_context(
        plan_root=plan_root,
        output_root=output_root,
        resource_profile_path=resource_profile_path,
        source=source,
    )
    existing = context.manifest.payload.get("preflight", {})
    if existing.get("state") in {"passed", "teacher_failed"}:
        _verify_preflight_artifacts(output_root, context)
        return json.loads((output_root / "preflight/preflight.json").read_text(encoding="utf-8"))
    plan_before = _plan_snapshot(plan_root)
    environment = execution_environment(P1_ROPCD_RECIPE)
    context.manifest.mark_gpu_inference_started()
    session = ScoringSession.load(P1_ROPCD_RECIPE)
    try:
        prompt_audit = _audit_prompts(session.tokenizer, context.bundle["plan"])
        if prompt_audit["passed"] is not True:
            raise RuntimeError("R-OPCD P1 prompt/token preflight failed")
        teacher_rows = score_teacher_pairs(session, context.bundle["plan"]["units"])
    finally:
        session.release()
    teacher_gate = _teacher_gate(teacher_rows)
    plan_after = _plan_snapshot(plan_root)
    if plan_before != plan_after:
        raise RuntimeError("R-OPCD P1 plan changed during preflight")
    preflight_root = output_root / "preflight"
    environment_path = preflight_root / "environment.json"
    teacher_path = preflight_root / "teacher_rows.jsonl"
    preflight_path = preflight_root / "preflight.json"
    immutable_json_write(environment_path, environment, "R-OPCD P1 environment")
    immutable_jsonl_write(teacher_path, teacher_rows, "R-OPCD P1 teacher rows")
    preflight = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "run_id": context.manifest.payload["run_id"],
        "plan_before": plan_before,
        "plan_after": plan_after,
        "environment_fingerprint": environment["environment_fingerprint"],
        "environment_sha256": file_hash(environment_path),
        "prompt_audit": prompt_audit,
        "teacher_rows_sha256": file_hash(teacher_path),
        "teacher_gate": teacher_gate,
        "training_started": False,
        "gpu_inference_started": True,
        "kill_split_accessed": True,
        "kill_path_contrast_computed": False,
        "p1b_authorized": False,
        "p2_authorized": False,
    }
    immutable_json_write(preflight_path, preflight, "R-OPCD P1 preflight")
    context.manifest.record_preflight(
        {
            "state": "passed" if teacher_gate["passed"] else "teacher_failed",
            "environment_fingerprint": environment["environment_fingerprint"],
            "environment_sha256": file_hash(environment_path),
            "preflight_sha256": file_hash(preflight_path),
            "teacher_rows_sha256": file_hash(teacher_path),
            "teacher_gate": teacher_gate,
        }
    )
    return preflight


def run_training(
    *,
    plan_root: Path,
    output_root: Path,
    resource_profile_path: Path,
    max_units: int = 1,
    target_unit_id: str | None = None,
    **source: Any,
) -> dict[str, Any]:
    _require_max_units(max_units)
    context = _load_context(
        plan_root=plan_root,
        output_root=output_root,
        resource_profile_path=resource_profile_path,
        source=source,
    )
    context.manifest.require_preflight()
    environment = execution_environment(P1_ROPCD_RECIPE)
    context.manifest.require_environment(environment["environment_fingerprint"])
    units = _selected_units(
        context.bundle["plan"]["units"],
        context.manifest,
        eligible={UnitState.PLANNED, UnitState.TRAINING},
        max_units=max_units,
        target_unit_id=target_unit_id,
    )
    if not units:
        return _progress(context.manifest, "training")
    session = TrainingSession.load(P1_ROPCD_RECIPE)
    plan_by_id = {str(unit["unit_id"]): unit for unit in context.bundle["plan"]["units"]}
    try:
        for unit in units:
            unit_id = str(unit["unit_id"])
            root_id = str(unit["root_id"])
            root_path = _root_dir(output_root, root_id)
            root = prepare_p1_root(
                session,
                root=root_path,
                run_id=str(context.manifest.payload["run_id"]),
                item_id=str(unit["item_id"]),
                root_id=root_id,
                root_seed=int(unit["root_adapter_seed"]),
            )
            context.manifest.record_root(
                root_id=root_id,
                value={
                    "root_id": root_id,
                    "item_id": str(unit["item_id"]),
                    "training_seed": int(unit["training_seed"]),
                    "root_adapter_seed": int(unit["root_adapter_seed"]),
                    "state": "verified",
                    "root_path": str(root_path.resolve()),
                    "adapter_sha256": root["adapter_sha256"],
                    "metadata_sha256": root["metadata_sha256"],
                },
            )
            parent_id = unit["parent_unit_id"]
            if parent_id is None:
                parent_adapter = root_path / "adapter"
            else:
                if context.manifest.unit_state(str(parent_id)) not in {
                    UnitState.TRAINED,
                    UnitState.EVALUATING,
                    UnitState.VERIFIED,
                }:
                    raise RuntimeError(f"R-OPCD P1 parent is not trained: {parent_id}")
                _verify_unit_artifact(output_root, context, plan_by_id[str(parent_id)])
                parent_adapter = _unit_root(output_root, str(parent_id)) / "adapter"
            if context.manifest.unit_state(unit_id) is UnitState.PLANNED:
                context.manifest.mark_unit(unit_id, UnitState.TRAINING)
            try:
                metadata = train_p1_unit(
                    session,
                    unit=unit,
                    parent_adapter_dir=parent_adapter,
                    root_adapter_sha256=str(root["adapter_sha256"]),
                    run_id=str(context.manifest.payload["run_id"]),
                    output_root=output_root,
                )
                context.manifest.mark_unit(
                    unit_id,
                    UnitState.TRAINED,
                    adapter_sha256=metadata["adapter_sha256"],
                    parent_adapter_sha256=metadata["parent_adapter_sha256"],
                    training_metadata_sha256=metadata["training_metadata_sha256"],
                    training_records_sha256=metadata["training_records_sha256"],
                )
            except BaseException as error:
                context.manifest.record_error("training", error, unit_id)
                raise
    finally:
        session.release()
    return _progress(context.manifest, "training")


def run_evaluation(
    *,
    plan_root: Path,
    output_root: Path,
    resource_profile_path: Path,
    max_units: int = 1,
    target_unit_id: str | None = None,
    **source: Any,
) -> dict[str, Any]:
    _require_max_units(max_units)
    context = _load_context(
        plan_root=plan_root,
        output_root=output_root,
        resource_profile_path=resource_profile_path,
        source=source,
    )
    context.manifest.require_all_trained()
    environment = execution_environment(P1_ROPCD_RECIPE)
    context.manifest.require_environment(environment["environment_fingerprint"])
    _verify_all_lineages(output_root, context)
    session = ScoringSession.load(P1_ROPCD_RECIPE)
    try:
        _ensure_no_memory_controls(session, output_root, context)
        units = _selected_units(
            context.bundle["plan"]["units"],
            context.manifest,
            eligible={UnitState.TRAINED, UnitState.EVALUATING},
            max_units=max_units,
            target_unit_id=target_unit_id,
        )
        bank = compile_bank()
        probes = {probe.probe_id: probe for probe in bank.probes}
        blocks = {block.block_id: block for block in bank.event_blocks}
        for unit in units:
            unit_id = str(unit["unit_id"])
            logical_name = str(unit["logical_name"])
            if context.manifest.unit_state(unit_id) is UnitState.TRAINED:
                context.manifest.mark_unit(unit_id, UnitState.EVALUATING)
            if logical_name not in SCORED_LOGICAL_NAMES and not unit["technical_duplicate"]:
                context.manifest.mark_unit(unit_id, UnitState.VERIFIED, scoring_not_required=True)
                continue
            result_path = _result_path(output_root, unit_id)
            if result_path.exists():
                rows = read_jsonl(result_path, f"R-OPCD P1 result {unit_id}")
            else:
                session.activate(_unit_root(output_root, unit_id) / "adapter")
                try:
                    if unit["technical_duplicate"]:
                        raw_rows = _score_duplicate(session, unit, probes)
                    elif logical_name in {"A2", "B2"}:
                        raw_rows = _score_anchor(session, unit, probes)
                    else:
                        raw_rows = _score_final_path(session, unit, probes, blocks)
                finally:
                    session.unload()
                rows = _tag_rows(raw_rows, unit)
                immutable_jsonl_write(result_path, rows, "R-OPCD P1 evaluated unit")
            _verify_unit_rows(rows, unit)
            context.manifest.mark_unit(
                unit_id,
                UnitState.VERIFIED,
                result_path=str(result_path.resolve()),
                result_sha256=file_hash(result_path),
                result_rows=len(rows),
            )
    except BaseException as error:
        active = next(
            (
                unit_id
                for unit_id, unit in context.manifest.payload["units"].items()
                if unit["state"] == UnitState.EVALUATING.value
            ),
            None,
        )
        context.manifest.record_error("evaluation", error, active)
        raise
    finally:
        session.release()
    return _progress(context.manifest, "evaluation")


def aggregate_run(
    *, plan_root: Path, output_root: Path, resource_profile_path: Path, **source: Any
) -> dict[str, Any]:
    context = _load_context(
        plan_root=plan_root,
        output_root=output_root,
        resource_profile_path=resource_profile_path,
        source=source,
    )
    context.manifest.require_all_verified()
    rows = _load_rows(output_root, context)
    summary = _summary(rows, _integrity(output_root, context, plan_root, source), context)
    summary_path = output_root / "aggregate/summary.json"
    report_path = output_root / "aggregate/report.md"
    immutable_json_write(summary_path, summary, "R-OPCD P1 summary")
    _immutable_text_write(report_path, _render_report(summary), "R-OPCD P1 report")
    context.manifest.record_aggregate(
        summary_path=summary_path, report_path=report_path, summary=summary
    )
    return summary


def verify_run(
    *, plan_root: Path, output_root: Path, resource_profile_path: Path, **source: Any
) -> dict[str, Any]:
    context = _load_context(
        plan_root=plan_root,
        output_root=output_root,
        resource_profile_path=resource_profile_path,
        source=source,
    )
    context.manifest.require_all_verified()
    rows = _load_rows(output_root, context)
    expected = _summary(rows, _integrity(output_root, context, plan_root, source), context)
    summary_path = output_root / "aggregate/summary.json"
    report_path = output_root / "aggregate/report.md"
    observed = json.loads(summary_path.read_text(encoding="utf-8"))
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise ValueError("R-OPCD P1 aggregate regeneration mismatch")
    if report_path.read_text(encoding="utf-8") != _render_report(expected):
        raise ValueError("R-OPCD P1 report regeneration mismatch")
    aggregate = context.manifest.payload.get("aggregate", {})
    if (
        aggregate.get("summary_sha256") != file_hash(summary_path)
        or aggregate.get("report_sha256") != file_hash(report_path)
        or context.manifest.payload.get("kill_path_contrast_computed") is not True
    ):
        raise ValueError("R-OPCD P1 manifest aggregate binding changed")
    return {
        "passed": True,
        "run_id": context.manifest.payload["run_id"],
        "plan_id": context.bundle["plan"]["plan_id"],
        "g3_gate": expected["gate"],
        "decision": expected["decision"],
        "p1b_implementation_review_eligible": expected["decision"][
            "p1b_implementation_review_eligible"
        ],
        "p1b_authorized": False,
        "p2_authorized": False,
        "reserve_accessed": False,
    }


def _load_context(
    *,
    plan_root: Path,
    output_root: Path,
    resource_profile_path: Path,
    source: dict[str, Any],
) -> RunContext:
    plan_report = verify_plan_bundle(plan_root=plan_root, **source)
    bundle = load_plan_bundle(plan_root)
    template = build_execution_authorization_template(
        plan_root=plan_root,
        output_root=output_root,
        plan_report=plan_report,
        resource_profile_path=resource_profile_path,
    )
    authorization = verify_adopted_authorization(output_root, expected_template=template)
    _, implementation_sha256 = implementation_identity()
    manifest = P1ExecutionManifest.load_or_create(
        output_root / "manifest.json",
        plan_manifest_id=str(plan_report["manifest_id"]),
        plan=bundle["plan"],
        authorization=authorization,
        implementation_sha256=implementation_sha256,
    )
    return RunContext(plan_report, bundle, authorization, manifest)


def _selected_units(
    units: list[dict[str, Any]],
    manifest: P1ExecutionManifest,
    *,
    eligible: set[UnitState],
    max_units: int,
    target_unit_id: str | None,
) -> list[dict[str, Any]]:
    ordered = sorted(units, key=lambda unit: int(unit["sequence_index"]))
    if target_unit_id is not None:
        ordered = [unit for unit in ordered if unit["unit_id"] == target_unit_id]
        if not ordered:
            raise KeyError(f"unknown R-OPCD P1 target unit: {target_unit_id}")
    return [
        unit
        for unit in ordered
        if manifest.unit_state(str(unit["unit_id"])) in eligible
    ][:max_units]


def _ensure_no_memory_controls(
    session: ScoringSession, output_root: Path, context: RunContext
) -> None:
    path = output_root / "results/no_memory.jsonl"
    expected = 336
    if path.exists():
        rows = read_jsonl(path, "R-OPCD P1 no-memory controls")
    else:
        bank = compile_bank()
        item_ids = {str(unit["item_id"]) for unit in context.bundle["plan"]["units"]}
        probes = sorted(
            (probe for probe in bank.probes if probe.item_id in item_ids and probe.core_endpoint),
            key=lambda probe: probe.probe_id,
        )
        rows = [
            _score_control(session, probe)
            for probe in probes
        ]
        immutable_jsonl_write(path, rows, "R-OPCD P1 no-memory controls")
    if len(rows) != expected or any(row.get("arm") != "no_memory" for row in rows):
        raise ValueError("R-OPCD P1 no-memory controls changed")
    context.manifest.record_controls(result_path=path, rows=len(rows))


def _score_control(session: ScoringSession, probe: Probe) -> dict[str, Any]:
    return score_prompt(
        session,
        raw_prompt=probe.prompt,
        action_choices=probe.action_choices,
        expected_action=probe.expected_action,
        obsolete_action=probe.obsolete_action,
        unit_id=f"p1r:{probe.item_id}:control:{probe.terminal_state}",
        item_id=probe.item_id,
        probe_id=probe.probe_id,
        probe_category=probe.category.value,
        arm="no_memory",
        disable_adapter=False,
        route_key=None,
        route_unit_id=None,
        metadata={"logical_name": "N", "fresh_session_id": json_hash([probe.probe_id, "N"])},
    )


def _tag_rows(rows: list[dict[str, Any]], unit: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "training_seed": int(unit["training_seed"]),
            "root_id": str(unit["root_id"]),
        }
        for row in rows
    ]


def _verify_unit_rows(rows: list[dict[str, Any]], unit: dict[str, Any]) -> None:
    logical_name = str(unit["logical_name"])
    expected = 14 if unit["technical_duplicate"] or logical_name in {"A2", "B2"} else 118
    if len(rows) != expected:
        raise ValueError(f"R-OPCD P1 row count changed for {unit['unit_id']}")
    if any(
        row.get("unit_id") != unit["unit_id"]
        or row.get("training_seed") != unit["training_seed"]
        or row.get("root_id") != unit["root_id"]
        for row in rows
    ):
        raise ValueError("R-OPCD P1 rows escaped their item-seed unit")
    row_ids = [row.get("row_id") for row in rows]
    if None in row_ids or len(row_ids) != len(set(row_ids)):
        raise ValueError("R-OPCD P1 unit row identities changed")


def _verify_all_lineages(output_root: Path, context: RunContext) -> bool:
    roots = {
        str(unit["root_id"]): unit for unit in context.bundle["plan"]["units"]
    }
    observed_roots = {
        path.name for path in (output_root / "roots").iterdir() if path.is_dir()
    }
    if observed_roots != {unit_slug(root_id) for root_id in roots}:
        raise ValueError("R-OPCD P1 root directory set changed")
    for root_id, unit in sorted(roots.items()):
        metadata = verify_p1_root(
            _root_dir(output_root, root_id),
            run_id=str(context.manifest.payload["run_id"]),
            item_id=str(unit["item_id"]),
            root_id=root_id,
            root_seed=int(unit["root_adapter_seed"]),
        )
        recorded = context.manifest.payload["roots"][root_id]
        if (
            recorded.get("state") != "verified"
            or recorded.get("adapter_sha256") != metadata["adapter_sha256"]
            or recorded.get("metadata_sha256") != metadata["metadata_sha256"]
        ):
            raise ValueError(f"R-OPCD P1 root binding changed: {root_id}")
    expected_units = {
        unit_slug(str(unit["unit_id"])) for unit in context.bundle["plan"]["units"]
    }
    observed_units = {
        path.name for path in (output_root / "units").iterdir() if path.is_dir()
    }
    if observed_units != expected_units:
        raise ValueError("R-OPCD P1 trained-unit directory set changed")
    for unit in context.bundle["plan"]["units"]:
        _verify_unit_artifact(output_root, context, unit)
    return True


def _verify_unit_artifact(
    output_root: Path, context: RunContext, unit: dict[str, Any]
) -> dict[str, Any]:
    parent_id = unit["parent_unit_id"]
    parent_adapter = (
        _root_dir(output_root, str(unit["root_id"])) / "adapter"
        if parent_id is None
        else _unit_root(output_root, str(parent_id)) / "adapter"
    )
    from plasticity_placement.pathmem_exec.artifacts import adapter_bundle_hash

    root = context.manifest.payload["roots"][str(unit["root_id"])]
    metadata = verify_p1_unit(
        _unit_root(output_root, str(unit["unit_id"])),
        unit=unit,
        parent_adapter_sha256=adapter_bundle_hash(parent_adapter),
        root_adapter_sha256=str(root["adapter_sha256"]),
        run_id=str(context.manifest.payload["run_id"]),
    )
    recorded = context.manifest.payload["units"][str(unit["unit_id"])]["metadata"]
    fields = (
        "adapter_sha256",
        "parent_adapter_sha256",
        "training_metadata_sha256",
        "training_records_sha256",
    )
    if any(metadata[field] != recorded.get(field) for field in fields):
        raise ValueError(f"R-OPCD P1 unit binding changed: {unit['unit_id']}")
    return metadata


def _load_rows(output_root: Path, context: RunContext) -> list[dict[str, Any]]:
    control_path = output_root / "results/no_memory.jsonl"
    controls = context.manifest.payload["control_results"]
    if controls.get("rows") != 336 or controls.get("result_sha256") != file_hash(control_path):
        raise ValueError("R-OPCD P1 no-memory control binding changed")
    rows = read_jsonl(control_path, "R-OPCD P1 no-memory controls")
    for unit in context.bundle["plan"]["units"]:
        if unit["logical_name"] not in SCORED_LOGICAL_NAMES and not unit["technical_duplicate"]:
            continue
        unit_id = str(unit["unit_id"])
        path = _result_path(output_root, unit_id)
        recorded = context.manifest.payload["units"][unit_id]["metadata"]
        if recorded.get("result_sha256") != file_hash(path):
            raise ValueError(f"R-OPCD P1 result binding changed: {unit_id}")
        unit_rows = read_jsonl(path, f"R-OPCD P1 result {unit_id}")
        _verify_unit_rows(unit_rows, unit)
        rows.extend(unit_rows)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"R-OPCD P1 matrix row count changed: {len(rows)}")
    return rows


def _integrity(
    output_root: Path,
    context: RunContext,
    plan_root: Path,
    source: dict[str, Any],
) -> dict[str, bool]:
    report = verify_plan_bundle(plan_root=plan_root, **source)
    preflight = context.manifest.require_preflight()
    plan_units = context.bundle["plan"]["units"]
    by_cell = {
        (str(unit["item_id"]), int(unit["training_seed"]), str(unit["logical_name"])): unit
        for unit in plan_units
    }
    exposure = all(
        by_cell[(item, seed, left)]["ordered_exposure_sha256"]
        == by_cell[(item, seed, right)]["ordered_exposure_sha256"]
        for item, seed, _ in {(key[0], key[1], "") for key in by_cell}
        for left, right in (("ABA", "BAA"), ("BAB", "ABB"))
    )
    duplicates = all(
        unit["ordered_exposure_sha256"]
        == by_cell[(unit["item_id"], unit["training_seed"], unit["duplicate_of"])][
            "ordered_exposure_sha256"
        ]
        and unit["parent_unit_id"]
        == by_cell[(unit["item_id"], unit["training_seed"], unit["duplicate_of"])][
            "parent_unit_id"
        ]
        for unit in plan_units
        if unit["technical_duplicate"]
    )
    return {
        "plan_bundle_reverified": report["passed"] is True,
        "g2_handoff_reverified": report["g2_handoff_id"] == context.bundle["handoff"]["handoff_id"],
        "authorization_bound": context.authorization["authorization_id"]
        == context.manifest.payload["identity"]["authorization_id"],
        "resource_profile_bound": context.authorization["resource_profile"]
        == context.manifest.payload["identity"]["resource_profile"],
        "preflight_hash_bound": file_hash(output_root / "preflight/preflight.json")
        == preflight["preflight_sha256"],
        "environment_hash_bound": file_hash(output_root / "preflight/environment.json")
        == preflight["environment_sha256"],
        "teacher_rows_hash_bound": file_hash(output_root / "preflight/teacher_rows.jsonl")
        == preflight["teacher_rows_sha256"],
        "teacher_gate_passed": preflight["teacher_gate"]["passed"] is True,
        "root_and_lineage_verified": _verify_all_lineages(output_root, context),
        "comparator_event_exposure_matched": exposure,
        "technical_duplicate_parent_event_exposure_matched": duplicates,
        "all_units_verified": all(
            unit["state"] == UnitState.VERIFIED.value
            for unit in context.manifest.payload["units"].values()
        ),
        "kill_only_no_downstream_access": (
            context.manifest.payload["kill_split_accessed"] is True
            and all(
                context.manifest.payload[field] is False
                for field in (
                    "reserve_accessed",
                    "p1b_authorized",
                    "p2_authorized",
                    "learned_router_started",
                    "rl_controller_started",
                    "automatic_hyperparameter_search_started",
                )
            )
        ),
    }


def _summary(
    rows: list[dict[str, Any]], integrity: dict[str, bool], context: RunContext
) -> dict[str, Any]:
    summary = summarize_p1(
        rows,
        verified_units=sum(
            unit["state"] == UnitState.VERIFIED.value
            for unit in context.manifest.payload["units"].values()
        ),
        integrity_checks=integrity,
    )
    return {
        **summary,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_id": context.manifest.payload["run_id"],
        "plan_manifest_id": context.plan_report["manifest_id"],
        "plan_id": context.bundle["plan"]["plan_id"],
        "g2_handoff_id": context.bundle["handoff"]["handoff_id"],
        "g2_repair_id": context.bundle["handoff"]["repair_id"],
        "authorization_id": context.authorization["authorization_id"],
        "resource_profile": context.authorization["resource_profile"],
        "recipe_sha256": context.bundle["plan"]["recipe_sha256"],
    }


def _audit_prompts(tokenizer: Any, plan: dict[str, Any]) -> dict[str, Any]:
    bank = compile_bank()
    probes = {probe.probe_id: probe for probe in bank.probes}
    blocks = {block.block_id: block for block in bank.event_blocks}
    rows: list[dict[str, Any]] = []

    def audit(unit_id: str, probe: Probe, kind: str, raw_prompt: str) -> None:
        result = audit_candidate_tokenization(
            tokenizer,
            chat_prompt(tokenizer, raw_prompt),
            probe.action_choices,
            evaluation_max_length=P1_ROPCD_RECIPE.evaluation_max_length,
        )
        rows.append(
            {
                "unit_id": unit_id,
                "probe_id": probe.probe_id,
                "prompt_kind": kind,
                "prompt_sha256": result["prompt_sha256"],
                "prompt_token_count": result["untruncated_prompt_token_count"],
                "all_candidates_valid": result["all_candidates_valid"],
                "rollout_length_fits": (
                    result["untruncated_prompt_token_count"]
                    + P1_ROPCD_RECIPE.rollout_max_new_tokens
                    <= P1_ROPCD_RECIPE.training_max_length
                    if kind.startswith("training_")
                    else True
                ),
            }
        )

    for unit in plan["units"]:
        for pair in unit["rollout_prompt_pairs"]:
            probe = probes[str(pair["probe_id"])]
            audit(str(unit["unit_id"]), probe, "training_student", str(pair["student_prompt"]))
            audit(
                str(unit["unit_id"]),
                probe,
                "training_teacher",
                str(pair["privileged_teacher_prompt"]),
            )
    item_ids = {str(unit["item_id"]) for unit in plan["units"]}
    for probe in sorted(
        (probe for probe in probes.values() if probe.item_id in item_ids and probe.core_endpoint),
        key=lambda probe: probe.probe_id,
    ):
        audit(f"p1r:{probe.item_id}:control", probe, "evaluation_no_memory", probe.prompt)
    for unit in plan["units"]:
        logical = str(unit["logical_name"])
        if unit["technical_duplicate"] or logical in {"A2", "B2"}:
            for probe_id in unit["core_probe_ids"]:
                probe = probes[str(probe_id)]
                audit(str(unit["unit_id"]), probe, "evaluation_endpoint", probe.prompt)
        elif logical in FINAL_PATHS:
            external_state, icl_state = _render_states(str(unit["item_id"]), logical, blocks)
            for probe_id in unit["core_probe_ids"]:
                probe = probes[str(probe_id)]
                external_prompt = render_fresh_session_probe(
                    probe, persistent_state=external_state, access_mode=AccessMode.ON
                )
                icl_prompt = render_fresh_session_probe(
                    probe, persistent_state=icl_state, access_mode=AccessMode.ON
                )
                for kind, prompt in (
                    ("evaluation_base", probe.prompt),
                    ("evaluation_parametric", probe.prompt),
                    ("evaluation_rescore", probe.prompt),
                    ("evaluation_disabled", probe.prompt),
                    ("evaluation_external", external_prompt),
                    ("evaluation_icl", icl_prompt),
                    ("evaluation_both", external_prompt),
                ):
                    audit(str(unit["unit_id"]), probe, kind, prompt)
            for probe_id in unit["unrelated_probe_ids"]:
                probe = probes[str(probe_id)]
                audit(str(unit["unit_id"]), probe, "evaluation_base_unrelated", probe.prompt)
                audit(
                    str(unit["unit_id"]), probe, "evaluation_parametric_unrelated", probe.prompt
                )
            for probe_id in unit["qualification_probe_ids"]:
                probe = probes[str(probe_id)]
                audit(str(unit["unit_id"]), probe, "evaluation_qualification", probe.prompt)
    expected = EXPECTED_TEACHER_ROWS * 2 + EXPECTED_ROWS
    return {
        "passed": len(rows) == expected
        and all(row["all_candidates_valid"] and row["rollout_length_fits"] for row in rows),
        "audit_count": len(rows),
        "expected_audit_count": expected,
        "maximum_prompt_tokens": max(int(row["prompt_token_count"]) for row in rows),
        "rows": rows,
    }


def _teacher_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = max(len(rows), 1)
    metrics = {
        "current_top1": sum(row.get("correct") is True for row in rows) / denominator,
        "obsolete_intrusion": sum(row.get("obsolete_intrusion") is True for row in rows)
        / denominator,
        "parser_validity": sum(row.get("parser_valid") is True for row in rows) / denominator,
    }
    checks = {
        "exact_teacher_rows": len(rows) == EXPECTED_TEACHER_ROWS,
        "current_top1_at_least_95pct": metrics["current_top1"] >= 0.95,
        "obsolete_intrusion_at_most_5pct": metrics["obsolete_intrusion"] <= 0.05,
        "parser_validity_exactly_100pct": metrics["parser_validity"] == 1.0,
    }
    return {
        "gate": "P1-R-OPCD-teacher",
        "passed": all(checks.values()),
        "metrics": metrics,
        "checks": checks,
        "blockers": [name for name, passed in checks.items() if not passed],
    }


def _verify_preflight_artifacts(output_root: Path, context: RunContext) -> None:
    preflight = context.manifest.payload["preflight"]
    for field, path in {
        "environment_sha256": output_root / "preflight/environment.json",
        "preflight_sha256": output_root / "preflight/preflight.json",
        "teacher_rows_sha256": output_root / "preflight/teacher_rows.jsonl",
    }.items():
        if file_hash(path) != preflight.get(field):
            raise ValueError(f"R-OPCD P1 preflight artifact changed: {field}")


def _plan_snapshot(plan_root: Path) -> dict[str, str]:
    return {
        path.relative_to(plan_root).as_posix(): file_hash(path)
        for path in sorted(plan_root.rglob("*"))
        if path.is_file()
    }


def _unit_root(output_root: Path, unit_id: str) -> Path:
    return output_root / "units" / unit_slug(unit_id)


def _root_dir(output_root: Path, root_id: str) -> Path:
    return output_root / "roots" / unit_slug(root_id)


def _result_path(output_root: Path, unit_id: str) -> Path:
    return output_root / "results" / f"{unit_slug(unit_id)}.jsonl"


def _progress(manifest: P1ExecutionManifest, stage: str) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for unit in manifest.payload["units"].values():
        counts[unit["state"]] = counts.get(unit["state"], 0) + 1
    return {
        "stage": stage,
        "run_id": manifest.payload["run_id"],
        "unit_states": counts,
        "training_started": manifest.payload["training_started"],
        "gpu_inference_started": manifest.payload["gpu_inference_started"],
        "kill_path_contrast_computed": manifest.payload["kill_path_contrast_computed"],
        "p1b_authorized": False,
        "p2_authorized": False,
    }


def _require_max_units(value: int) -> None:
    if value < 1 or value > EXPECTED_UNITS:
        raise ValueError("R-OPCD P1 max_units must be between 1 and 396")


def _immutable_text_write(path: Path, text: str, label: str) -> None:
    payload = text.encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing {label} changed; use a new output")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _render_report(summary: dict[str, Any]) -> str:
    primary = summary["primary"]
    consequence = summary["consequence"]
    return "\n".join(
        [
            "# PathMem R-OPCD P1 investment/kill test",
            "",
            f"- Run ID: `{summary['run_id']}`",
            f"- G3 integrity gate passed: `{summary['gate']['passed']}`",
            f"- Decision: `{summary['decision']['classification']}`",
            f"- Mean D: `{primary['D_mean_nats']:.6f}` nats",
            f"- D 95% CI: `[{primary['D_95_ci_nats'][0]:.6f}, {primary['D_95_ci_nats'][1]:.6f}]`",
            f"- Frozen epsilon_JS: `{primary['epsilon_js_nats']:.6f}` nats",
            f"- Mean C: `{consequence['C_mean']:.6f}`",
            f"- C 95% CI: `[{consequence['C_95_ci'][0]:.6f}, {consequence['C_95_ci'][1]:.6f}]`",
            f"- Consequential at 0.10: `{consequence['consequential']}`",
            "- P1b implementation review eligible: "
            f"`{summary['decision']['p1b_implementation_review_eligible']}`",
            "- P1b authorized: `False`",
            "- P2 authorized: `False`",
            "",
            "This is an investment/kill result for the frozen R-OPCD operator. It does not",
            "automatically authorize P1b, P2, reserve access, learned routing, or RL control.",
            "",
        ]
    )
