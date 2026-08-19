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
    json_hash,
)
from plasticity_placement.pathmem.schema import AccessMode, Probe
from plasticity_placement.pathmem_consolidation_exec.analysis import summarize_g1c
from plasticity_placement.pathmem_consolidation_exec.config import (
    G1C_EXECUTION_RECIPE,
    PREFLIGHT_SCHEMA_VERSION,
    RopcdExecutionRecipe,
)
from plasticity_placement.pathmem_consolidation_exec.identity import (
    adopt_authorization,
    build_authorization_template,
    execution_environment,
    implementation_identity,
    verify_adopted_authorization,
    verify_external_approval,
    verify_source_bundle,
)
from plasticity_placement.pathmem_consolidation_exec.manifest import (
    G1CExecutionManifest,
    UnitState,
)
from plasticity_placement.pathmem_consolidation_exec.routing import (
    ExactKeyRouter,
    route_target_from_unit,
)
from plasticity_placement.pathmem_consolidation_exec.scoring import (
    ScoringSession,
    audit_plan_prompts,
    score_prompt,
    score_teacher_pairs,
)
from plasticity_placement.pathmem_consolidation_exec.training import (
    TrainingSession,
    train_unit,
    unit_slug,
    verify_trained_unit,
)
from plasticity_placement.pathmem_exec.artifacts import (
    immutable_jsonl_write,
    read_jsonl,
)


@dataclass(frozen=True, slots=True)
class RunContext:
    source: dict[str, Any]
    authorization: dict[str, Any]
    manifest: G1CExecutionManifest
    recipe: RopcdExecutionRecipe


def inspect_execution(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    output_root: Path,
    recipe: RopcdExecutionRecipe = G1C_EXECUTION_RECIPE,
) -> dict[str, Any]:
    source = verify_source_bundle(bundle_root, parent_manifest)
    template = build_authorization_template(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        output_root=output_root,
        recipe=recipe,
    )
    return {
        "safe_default": "inspection_only",
        "source": {key: value for key, value in source.items() if key != "plan"},
        "recipe": recipe.to_dict(),
        "recipe_sha256": json_hash(recipe.to_dict()),
        "authorization_template": template,
        "planned_stages": [
            "adopt_authorization",
            "gpu_teacher_preflight",
            "resumable_unit_training",
            "resumable_evaluation",
            "cpu_aggregate",
            "cpu_verify",
        ],
        "training_started": False,
        "gpu_inference_started": False,
        "p0_authorized": False,
        "path_contrast_authorized": False,
    }


def adopt_run_authorization(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    output_root: Path,
    approval_path: Path,
    recipe: RopcdExecutionRecipe = G1C_EXECUTION_RECIPE,
) -> dict[str, Any]:
    source = verify_source_bundle(bundle_root, parent_manifest)
    template = build_authorization_template(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        output_root=output_root,
        recipe=recipe,
    )
    approval = verify_external_approval(approval_path, expected_template=template)
    adopt_authorization(output_root, approval)
    _, implementation_sha256 = implementation_identity()
    manifest = G1CExecutionManifest.load_or_create(
        output_root / "manifest.json",
        source=source,
        authorization=approval,
        recipe=recipe.to_dict(),
        recipe_sha256=json_hash(recipe.to_dict()),
        implementation_sha256=implementation_sha256,
        units=source["plan"]["units"],
    )
    return {
        "authorization_id": approval["authorization_id"],
        "run_id": manifest.payload["run_id"],
        "source_manifest_id": source["manifest_id"],
        "plan_id": source["plan_id"],
        "output_root": str(output_root.resolve()),
        "training_started": manifest.payload["training_started"],
        "gpu_inference_started": manifest.payload["gpu_inference_started"],
    }


def run_preflight(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    output_root: Path,
    recipe: RopcdExecutionRecipe = G1C_EXECUTION_RECIPE,
) -> dict[str, Any]:
    context = _load_context(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        output_root=output_root,
        recipe=recipe,
    )
    existing = context.manifest.payload.get("preflight", {})
    if existing.get("state") in {"passed", "teacher_failed"}:
        _verify_preflight_artifacts(output_root, context)
        return json.loads(
            (output_root / "preflight/preflight.json").read_text(encoding="utf-8")
        )

    source_before = _source_snapshot(bundle_root)
    environment = execution_environment(recipe)
    session = ScoringSession.load(recipe)
    try:
        prompt_audit = audit_plan_prompts(
            session.tokenizer, context.source["plan"]["units"], recipe=recipe
        )
        if prompt_audit["passed"] is not True:
            raise RuntimeError("G1-C prompt/token preflight failed")
        teacher_rows = score_teacher_pairs(session, context.source["plan"]["units"])
    finally:
        session.release()
    from plasticity_placement.pathmem_consolidation_exec.analysis import (
        summarize_teacher_gate,
    )

    teacher_gate = summarize_teacher_gate(teacher_rows)
    source_after = _source_snapshot(bundle_root)
    if source_before != source_after:
        raise RuntimeError("G0-v2 source bundle changed during preflight")
    preflight_root = output_root / "preflight"
    environment_path = preflight_root / "environment.json"
    teacher_path = preflight_root / "teacher_rows.jsonl"
    preflight_path = preflight_root / "preflight.json"
    immutable_json_write(environment_path, environment, "G1-C environment")
    immutable_jsonl_write(teacher_path, teacher_rows, "G1-C teacher rows")
    preflight = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "run_id": context.manifest.payload["run_id"],
        "source_before": source_before,
        "source_after": source_after,
        "environment_fingerprint": environment["environment_fingerprint"],
        "environment_sha256": file_hash(environment_path),
        "prompt_audit": prompt_audit,
        "teacher_rows_sha256": file_hash(teacher_path),
        "teacher_gate": teacher_gate,
        "training_started": False,
        "p0_authorized": False,
        "path_contrast_computed": False,
    }
    immutable_json_write(preflight_path, preflight, "G1-C preflight")
    context.manifest.record_preflight(
        environment_fingerprint=environment["environment_fingerprint"],
        environment_sha256=file_hash(environment_path),
        preflight_sha256=file_hash(preflight_path),
        teacher_rows_sha256=file_hash(teacher_path),
        teacher_gate=teacher_gate,
    )
    return preflight


def run_training(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    output_root: Path,
    max_units: int = 1,
    target_unit_id: str | None = None,
    recipe: RopcdExecutionRecipe = G1C_EXECUTION_RECIPE,
) -> dict[str, Any]:
    if max_units <= 0 or max_units > 24:
        raise ValueError("max_units must be between 1 and 24")
    context = _load_context(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        output_root=output_root,
        recipe=recipe,
    )
    context.manifest.require_preflight()
    environment = execution_environment(recipe)
    context.manifest.require_environment(environment["environment_fingerprint"])
    units = _selected_units(
        context.source["plan"]["units"],
        context.manifest,
        eligible={UnitState.PLANNED, UnitState.TRAINING},
        max_units=max_units,
        target_unit_id=target_unit_id,
    )
    if not units:
        return _progress(context.manifest, "training")
    session = TrainingSession.load(recipe)
    try:
        for unit in units:
            unit_id = str(unit["unit_id"])
            context.manifest.mark_unit(unit_id, UnitState.TRAINING)
            try:
                metadata = train_unit(
                    session,
                    unit=unit,
                    run_id=str(context.manifest.payload["run_id"]),
                    output_root=output_root,
                )
                context.manifest.mark_unit(
                    unit_id,
                    UnitState.TRAINED,
                    adapter_sha256=metadata["adapter_sha256"],
                    adapter_dir=metadata["adapter_dir"],
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
    bundle_root: Path,
    parent_manifest: Path,
    output_root: Path,
    max_units: int = 1,
    target_unit_id: str | None = None,
    recipe: RopcdExecutionRecipe = G1C_EXECUTION_RECIPE,
) -> dict[str, Any]:
    if max_units <= 0 or max_units > 24:
        raise ValueError("max_units must be between 1 and 24")
    context = _load_context(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        output_root=output_root,
        recipe=recipe,
    )
    context.manifest.require_all_trained()
    environment = execution_environment(recipe)
    context.manifest.require_environment(environment["environment_fingerprint"])
    units = _selected_units(
        context.source["plan"]["units"],
        context.manifest,
        eligible={UnitState.TRAINED, UnitState.EVALUATING},
        max_units=max_units,
        target_unit_id=target_unit_id,
    )
    if not units:
        return _progress(context.manifest, "evaluation")
    all_units = {
        str(unit["unit_id"]): unit for unit in context.source["plan"]["units"]
    }
    trained = {
        unit_id: verify_trained_unit(
            output_root / "units" / unit_slug(unit_id),
            unit=unit,
            run_id=str(context.manifest.payload["run_id"]),
            recipe=recipe,
        )
        for unit_id, unit in all_units.items()
    }
    probes = {probe.probe_id: probe for probe in compile_bank().probes}
    session = ScoringSession.load(recipe)
    try:
        for unit in units:
            unit_id = str(unit["unit_id"])
            context.manifest.mark_unit(unit_id, UnitState.EVALUATING)
            try:
                rows = _evaluate_unit(
                    session,
                    unit=unit,
                    all_units=all_units,
                    trained=trained,
                    probes=probes,
                )
                result_path = output_root / "evaluation" / f"{unit_slug(unit_id)}.jsonl"
                result_sha256 = immutable_jsonl_write(
                    result_path, rows, "G1-C unit evaluation"
                )
                context.manifest.mark_unit(
                    unit_id,
                    UnitState.VERIFIED,
                    result_path=str(result_path.resolve()),
                    result_sha256=result_sha256,
                    result_rows=len(rows),
                )
            except BaseException as error:
                context.manifest.record_error("evaluation", error, unit_id)
                if session.active_adapter_dir is not None:
                    session.unload()
                raise
    finally:
        session.release()
    return _progress(context.manifest, "evaluation")


def aggregate_run(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    output_root: Path,
    recipe: RopcdExecutionRecipe = G1C_EXECUTION_RECIPE,
) -> dict[str, Any]:
    context = _load_context(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        output_root=output_root,
        recipe=recipe,
    )
    context.manifest.require_all_verified()
    teacher_rows, evaluation_rows = _load_result_matrix(output_root, context)
    integrity = _integrity_checks(bundle_root, output_root, context, evaluation_rows)
    summary = summarize_g1c(
        teacher_rows,
        evaluation_rows,
        run_id=str(context.manifest.payload["run_id"]),
        source_manifest_id=str(context.source["manifest_id"]),
        plan_id=str(context.source["plan_id"]),
        integrity_checks=integrity,
    )
    aggregate_root = output_root / "aggregate"
    summary_path = aggregate_root / "summary.json"
    report_path = aggregate_root / "report.md"
    immutable_json_write(summary_path, summary, "G1-C aggregate summary")
    _immutable_text_write(report_path, _render_report(summary), "G1-C aggregate report")
    context.manifest.record_aggregate(
        summary_path=summary_path,
        report_path=report_path,
        summary=summary,
    )
    return summary


def verify_run(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    output_root: Path,
    recipe: RopcdExecutionRecipe = G1C_EXECUTION_RECIPE,
) -> dict[str, Any]:
    context = _load_context(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        output_root=output_root,
        recipe=recipe,
    )
    context.manifest.require_all_verified()
    teacher_rows, evaluation_rows = _load_result_matrix(output_root, context)
    integrity = _integrity_checks(bundle_root, output_root, context, evaluation_rows)
    expected_summary = summarize_g1c(
        teacher_rows,
        evaluation_rows,
        run_id=str(context.manifest.payload["run_id"]),
        source_manifest_id=str(context.source["manifest_id"]),
        plan_id=str(context.source["plan_id"]),
        integrity_checks=integrity,
    )
    summary_path = output_root / "aggregate/summary.json"
    observed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if canonical_json_bytes(observed_summary) != canonical_json_bytes(expected_summary):
        raise ValueError("G1-C aggregate regeneration mismatch")
    aggregate = context.manifest.payload.get("aggregate", {})
    report_path = output_root / "aggregate/report.md"
    if report_path.read_text(encoding="utf-8") != _render_report(expected_summary):
        raise ValueError("G1-C aggregate report regeneration mismatch")
    if (
        aggregate.get("state") != "complete"
        or aggregate.get("summary_sha256") != file_hash(summary_path)
        or aggregate.get("report_sha256") != file_hash(report_path)
        or canonical_json_bytes(aggregate.get("gate"))
        != canonical_json_bytes(expected_summary["gate"])
    ):
        raise ValueError("G1-C manifest aggregate binding mismatch")
    if any(
        context.manifest.payload.get(field) is not False
        for field in ("p0_authorized", "path_contrast_computed", "kill_or_reserve_accessed")
    ):
        raise ValueError("G1-C run contains unauthorized downstream state")
    return {
        "passed": True,
        "run_id": context.manifest.payload["run_id"],
        "source_manifest_id": context.source["manifest_id"],
        "plan_id": context.source["plan_id"],
        "g1c_gate": expected_summary["gate"],
        "p0_runner_implementation_review_eligible": expected_summary[
            "p0_runner_implementation_review_eligible"
        ],
        "p0_authorized": False,
        "path_contrast_computed": False,
    }


def _load_context(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    output_root: Path,
    recipe: RopcdExecutionRecipe,
) -> RunContext:
    source = verify_source_bundle(bundle_root, parent_manifest)
    template = build_authorization_template(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        output_root=output_root,
        recipe=recipe,
    )
    authorization = verify_adopted_authorization(
        output_root, expected_template=template
    )
    _, implementation_sha256 = implementation_identity()
    manifest = G1CExecutionManifest.load_or_create(
        output_root / "manifest.json",
        source=source,
        authorization=authorization,
        recipe=recipe.to_dict(),
        recipe_sha256=json_hash(recipe.to_dict()),
        implementation_sha256=implementation_sha256,
        units=source["plan"]["units"],
    )
    return RunContext(
        source=source,
        authorization=authorization,
        manifest=manifest,
        recipe=recipe,
    )


def _selected_units(
    units: list[dict[str, Any]],
    manifest: G1CExecutionManifest,
    *,
    eligible: set[UnitState],
    max_units: int,
    target_unit_id: str | None,
) -> list[dict[str, Any]]:
    ordered = sorted(units, key=lambda unit: int(unit["sequence_index"]))
    if target_unit_id is not None:
        ordered = [unit for unit in ordered if unit["unit_id"] == target_unit_id]
        if not ordered:
            raise KeyError(f"unknown G1-C target unit: {target_unit_id}")
    return [
        unit for unit in ordered if manifest.unit_state(str(unit["unit_id"])) in eligible
    ][:max_units]


def _evaluate_unit(
    session: ScoringSession,
    *,
    unit: dict[str, Any],
    all_units: dict[str, dict[str, Any]],
    trained: dict[str, dict[str, Any]],
    probes: dict[str, Probe],
) -> list[dict[str, Any]]:
    unit_id = str(unit["unit_id"])
    correct = trained[unit_id]
    immediate_pairs = list(unit["rollout_prompt_pairs"])
    rows: list[dict[str, Any]] = []

    for pair in immediate_pairs:
        rows.append(
            _score_pair(
                session,
                unit=unit,
                pair=pair,
                arm="base_immediate",
            )
        )

    router = ExactKeyRouter()
    router.apply(
        route_target_from_unit(
            unit,
            adapter_dir=Path(correct["adapter_dir"]),
            adapter_sha256=str(correct["adapter_sha256"]),
        )
    )
    session.activate(Path(correct["adapter_dir"]))
    for pair in immediate_pairs:
        target = router.resolve_prompt(str(pair["student_prompt"]))
        rows.append(
            _score_pair(
                session,
                unit=unit,
                pair=pair,
                arm="immediate",
                route_target=target,
            )
        )

    for distractor_id in unit["retention_distractor_unit_ids"]:
        distractor = all_units[str(distractor_id)]
        metadata = trained[str(distractor_id)]
        router.apply(
            route_target_from_unit(
                distractor,
                adapter_dir=Path(metadata["adapter_dir"]),
                adapter_sha256=str(metadata["adapter_sha256"]),
            )
        )
    router_snapshot = router.snapshot()
    if len(router_snapshot["application_order"]) != 13:
        raise RuntimeError("retention registry did not apply anchor plus 12 distractors")
    for pair in immediate_pairs:
        target = router.resolve_prompt(str(pair["student_prompt"]))
        rows.append(
            _score_pair(
                session,
                unit=unit,
                pair=pair,
                arm="retained",
                route_target=target,
                metadata={"router_sha256": router_snapshot["router_sha256"]},
            )
        )
    for probe_id in unit["core_probe_ids"]:
        probe = probes[str(probe_id)]
        prompt = _student_probe_prompt(probe)
        target = router.resolve_prompt(prompt)
        rows.append(
            _score_probe(
                session,
                unit=unit,
                probe=probe,
                raw_prompt=prompt,
                arm="retained_core",
                route_target=target,
                metadata={"router_sha256": router_snapshot["router_sha256"]},
            )
        )
    control_ids = [
        *unit["near_neighbor_probe_ids"],
        *unit["unrelated_probe_ids"],
    ]
    for probe_id in control_ids:
        probe = probes[str(probe_id)]
        prompt = _student_probe_prompt(probe)
        rows.append(
            _score_probe(
                session,
                unit=unit,
                probe=probe,
                raw_prompt=prompt,
                arm="base_control",
                disable_adapter=True,
            )
        )
        target = router.resolve_prompt(prompt)
        rows.append(
            _score_probe(
                session,
                unit=unit,
                probe=probe,
                raw_prompt=prompt,
                arm="routed_control",
                disable_adapter=target is None,
                route_target=target,
                false_activation=target is not None,
                metadata={"router_sha256": router_snapshot["router_sha256"]},
            )
        )
    for pair in immediate_pairs:
        rows.append(
            _score_pair(
                session,
                unit=unit,
                pair=pair,
                arm="access_off",
                disable_adapter=True,
            )
        )
    session.unload()
    for pair in immediate_pairs:
        rows.append(
            _score_pair(
                session,
                unit=unit,
                pair=pair,
                arm="restored_base",
            )
        )

    wrong_id = str(unit["wrong_swap_unit_id"])
    session.activate(Path(trained[wrong_id]["adapter_dir"]))
    for pair in immediate_pairs:
        rows.append(
            _score_pair(
                session,
                unit=unit,
                pair=pair,
                arm="wrong_swap",
                metadata={"wrong_swap_unit_id": wrong_id},
            )
        )
    session.unload()
    if len(rows) != 62:
        raise RuntimeError(f"G1-C unit evaluation row count changed: {len(rows)}")
    return rows


def _score_pair(
    session: ScoringSession,
    *,
    unit: dict[str, Any],
    pair: dict[str, Any],
    arm: str,
    disable_adapter: bool = False,
    route_target: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_unit = str(unit["unit_id"])
    route_unit = route_target.unit_id if route_target is not None else None
    return score_prompt(
        session,
        raw_prompt=str(pair["student_prompt"]),
        action_choices=tuple(pair["action_choices"]),
        expected_action=str(unit["current_action"]),
        obsolete_action=str(unit["obsolete_action"]),
        unit_id=expected_unit,
        item_id=str(unit["item_id"]),
        probe_id=str(pair["probe_id"]),
        probe_category="qualification",
        arm=arm,
        disable_adapter=disable_adapter,
        route_key=str(unit["memory_key"]) if route_target is not None else None,
        route_unit_id=route_unit,
        route_miss=(
            arm in {"immediate", "retained"}
            and (route_target is None or route_unit != expected_unit)
        ),
        metadata=metadata,
    )


def _score_probe(
    session: ScoringSession,
    *,
    unit: dict[str, Any],
    probe: Probe,
    raw_prompt: str,
    arm: str,
    disable_adapter: bool = False,
    route_target: Any | None = None,
    false_activation: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected_unit = str(unit["unit_id"])
    route_unit = route_target.unit_id if route_target is not None else None
    return score_prompt(
        session,
        raw_prompt=raw_prompt,
        action_choices=probe.action_choices,
        expected_action=probe.expected_action,
        obsolete_action=probe.obsolete_action,
        unit_id=expected_unit,
        item_id=str(unit["item_id"]),
        probe_id=probe.probe_id,
        probe_category=probe.category.value,
        arm=arm,
        disable_adapter=disable_adapter,
        route_key=route_target.memory_key if route_target is not None else None,
        route_unit_id=route_unit,
        route_miss=(
            arm == "retained_core"
            and (route_target is None or route_unit != expected_unit)
        ),
        false_activation=false_activation,
        metadata=metadata,
    )


def _student_probe_prompt(probe: Probe) -> str:
    return render_fresh_session_probe(
        probe,
        persistent_state="unused",
        access_mode=AccessMode.OFF,
    )


def _load_result_matrix(
    output_root: Path, context: RunContext
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    teacher_path = output_root / "preflight/teacher_rows.jsonl"
    preflight = context.manifest.require_preflight()
    if file_hash(teacher_path) != preflight["teacher_rows_sha256"]:
        raise ValueError("G1-C teacher result hash changed")
    teacher_rows = read_jsonl(teacher_path)
    evaluation_rows: list[dict[str, Any]] = []
    for unit_id, unit in sorted(context.manifest.payload["units"].items()):
        metadata = unit.get("metadata", {})
        result_path = Path(str(metadata["result_path"]))
        if file_hash(result_path) != metadata.get("result_sha256"):
            raise ValueError(f"G1-C evaluation result hash changed: {unit_id}")
        rows = read_jsonl(result_path)
        if len(rows) != metadata.get("result_rows"):
            raise ValueError(f"G1-C evaluation row count changed: {unit_id}")
        evaluation_rows.extend(rows)
    return teacher_rows, evaluation_rows


def _integrity_checks(
    bundle_root: Path,
    output_root: Path,
    context: RunContext,
    evaluation_rows: list[dict[str, Any]],
) -> dict[str, bool]:
    preflight = context.manifest.require_preflight()
    preflight_artifact = json.loads(
        (output_root / "preflight/preflight.json").read_text(encoding="utf-8")
    )
    checks = {
        "source_bundle_unchanged": _source_snapshot(bundle_root)
        == preflight_artifact["source_after"],
        "authorization_bound": context.authorization["authorization_id"]
        == context.manifest.payload["identity"]["authorization_id"],
        "preflight_hash_bound": file_hash(output_root / "preflight/preflight.json")
        == preflight["preflight_sha256"],
        "environment_hash_bound": file_hash(output_root / "preflight/environment.json")
        == preflight["environment_sha256"],
        "teacher_gate_passed": preflight["teacher_gate"]["passed"] is True,
        "prompt_audit_passed": (
            preflight_artifact.get("prompt_audit", {}).get("passed") is True
            and preflight_artifact.get("prompt_audit", {}).get("audit_count") == 816
        ),
        "all_candidate_audits_valid": all(
            row.get("candidate_audit", {}).get("all_candidates_valid") is True
            for row in evaluation_rows
        ),
        "all_evaluation_rows_parser_valid": all(
            row.get("parser_valid") is True for row in evaluation_rows
        ),
        "no_non_finite_scores": all(row.get("non_finite") is False for row in evaluation_rows),
        "all_units_verified": all(
            unit["state"] == UnitState.VERIFIED.value
            for unit in context.manifest.payload["units"].values()
        ),
        "no_p0_authorization": context.manifest.payload["p0_authorized"] is False,
        "no_path_contrast": context.manifest.payload["path_contrast_computed"] is False,
        "no_kill_or_reserve_access": context.manifest.payload["kill_or_reserve_accessed"]
        is False,
    }
    for unit_id, unit in {
        str(value["unit_id"]): value for value in context.source["plan"]["units"]
    }.items():
        try:
            metadata = verify_trained_unit(
                output_root / "units" / unit_slug(unit_id),
                unit=unit,
                run_id=str(context.manifest.payload["run_id"]),
                recipe=context.recipe,
            )
            manifest_metadata = context.manifest.payload["units"][unit_id]["metadata"]
            checks[f"adapter_bound:{unit_id}"] = (
                metadata["adapter_sha256"] == manifest_metadata.get("adapter_sha256")
                and metadata["training_metadata_sha256"]
                == manifest_metadata.get("training_metadata_sha256")
                and metadata["training_records_sha256"]
                == manifest_metadata.get("training_records_sha256")
            )
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            checks[f"adapter_bound:{unit_id}"] = False
    return checks


def _verify_preflight_artifacts(output_root: Path, context: RunContext) -> None:
    preflight = context.manifest.payload["preflight"]
    paths = {
        "environment_sha256": output_root / "preflight/environment.json",
        "preflight_sha256": output_root / "preflight/preflight.json",
        "teacher_rows_sha256": output_root / "preflight/teacher_rows.jsonl",
    }
    for field, path in paths.items():
        if file_hash(path) != preflight.get(field):
            raise ValueError(f"G1-C preflight artifact hash changed: {field}")


def _source_snapshot(bundle_root: Path) -> dict[str, str]:
    paths = sorted(path for path in bundle_root.rglob("*") if path.is_file())
    return {
        path.relative_to(bundle_root).as_posix(): file_hash(path) for path in paths
    }


def _progress(manifest: G1CExecutionManifest, stage: str) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for unit in manifest.payload["units"].values():
        state = str(unit["state"])
        counts[state] = counts.get(state, 0) + 1
    return {
        "stage": stage,
        "run_id": manifest.payload["run_id"],
        "unit_states": dict(sorted(counts.items())),
        "training_started": manifest.payload["training_started"],
        "gpu_inference_started": manifest.payload["gpu_inference_started"],
        "p0_authorized": False,
        "path_contrast_computed": False,
    }


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
    gate = summary["gate"]
    metrics = summary["metrics"]
    lines = [
        "# PathMem G1-C R-OPCD qualification",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Gate passed: `{gate['passed']}`",
        f"- Blockers: `{', '.join(gate['blockers']) or 'none'}`",
        f"- Teacher top-1: `{metrics['teacher_current_top1']:.4f}`",
        f"- Immediate top-1: `{metrics['immediate_current_top1']:.4f}`",
        f"- H12 retained top-1: `{metrics['retention_h12_current_top1']:.4f}`",
        f"- Unrelated regression: `{metrics['unrelated_regression_pp']:.3f} pp`",
        f"- Router miss rate: `{metrics['router_miss_rate']:.4f}`",
        f"- Router false activation rate: `{metrics['router_false_activation_rate']:.4f}`",
        "",
        "Passing this gate permits only separate R-OPCD P0 runner implementation review. ",
        "It does not authorize P0 execution, path contrasts, kill/reserve access, or RL control.",
        "",
    ]
    return "\n".join(lines)
