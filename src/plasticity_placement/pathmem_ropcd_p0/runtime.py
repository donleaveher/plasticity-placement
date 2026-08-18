from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.backends import (
    ICLHistoryRenderer,
    VersionedExternalStore,
    render_fresh_session_probe,
)
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
from plasticity_placement.pathmem_ropcd_p0.analysis import summarize_p0
from plasticity_placement.pathmem_ropcd_p0.bundle import (
    inspect_plan,
    load_plan_bundle,
    verify_plan_bundle,
)
from plasticity_placement.pathmem_ropcd_p0.config import (
    EXPECTED_ROWS,
    EXPECTED_UNITS,
    FINAL_PATHS,
    P0_ROPCD_RECIPE,
    PATH_SUFFIXES,
    PREFLIGHT_SCHEMA_VERSION,
    SCORED_LOGICAL_NAMES,
    SUMMARY_SCHEMA_VERSION,
)
from plasticity_placement.pathmem_ropcd_p0.identity import (
    adopt_authorization,
    build_authorization_template,
    implementation_identity,
    verify_adopted_authorization,
    verify_external_approval,
)
from plasticity_placement.pathmem_ropcd_p0.manifest import P0ExecutionManifest, UnitState
from plasticity_placement.pathmem_ropcd_p0.training import (
    prepare_root_adapter,
    train_p0_unit,
    verify_p0_unit,
    verify_root_adapter,
)
from plasticity_placement.training.model_utils import chat_prompt


@dataclass(frozen=True, slots=True)
class RunContext:
    plan_report: dict[str, Any]
    bundle: dict[str, Any]
    authorization: dict[str, Any]
    manifest: P0ExecutionManifest


def inspect_execution(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    report = inspect_plan(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
    )
    return {
        **report,
        "output_root": str(output_root.resolve()),
        "recipe": P0_ROPCD_RECIPE.to_dict(),
        "safe_default": "inspection_only",
    }


def authorization_template(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    plan_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    report = verify_plan_bundle(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
    )
    return build_authorization_template(
        plan_root=plan_root,
        output_root=output_root,
        plan_report=report,
    )


def adopt_run_authorization(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    plan_root: Path,
    output_root: Path,
    approval_path: Path,
) -> dict[str, Any]:
    template = authorization_template(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
        output_root=output_root,
    )
    approval = verify_external_approval(approval_path, expected_template=template)
    adopt_authorization(output_root, approval)
    context = _load_context(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
        output_root=output_root,
    )
    return {
        "adopted": True,
        "authorization_id": context.authorization["authorization_id"],
        "run_id": context.manifest.payload["run_id"],
        "plan_id": context.bundle["plan"]["plan_id"],
        "permissions": context.authorization["permissions"],
    }


def run_preflight(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    plan_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    context = _load_context(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
        output_root=output_root,
    )
    existing = context.manifest.payload.get("preflight", {})
    if existing.get("state") in {"passed", "teacher_failed"}:
        _verify_preflight_artifacts(output_root, context)
        return json.loads((output_root / "preflight/preflight.json").read_text(encoding="utf-8"))
    plan_before = _plan_snapshot(plan_root)
    environment = execution_environment(P0_ROPCD_RECIPE)
    context.manifest.mark_gpu_inference_started()
    session = ScoringSession.load(P0_ROPCD_RECIPE)
    try:
        prompt_audit = _audit_prompts(session.tokenizer, context.bundle["plan"])
        if prompt_audit["passed"] is not True:
            raise RuntimeError("R-OPCD P0 prompt/token preflight failed")
        teacher_rows = score_teacher_pairs(session, context.bundle["plan"]["units"])
    finally:
        session.release()
    teacher_gate = _teacher_gate(teacher_rows)
    plan_after = _plan_snapshot(plan_root)
    if plan_before != plan_after:
        raise RuntimeError("R-OPCD P0 plan bundle changed during preflight")
    preflight_root = output_root / "preflight"
    environment_path = preflight_root / "environment.json"
    teacher_path = preflight_root / "teacher_rows.jsonl"
    preflight_path = preflight_root / "preflight.json"
    immutable_json_write(environment_path, environment, "R-OPCD P0 environment")
    immutable_jsonl_write(teacher_path, teacher_rows, "R-OPCD P0 teacher rows")
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
        "path_contrast_computed": False,
        "p1_authorized": False,
    }
    immutable_json_write(preflight_path, preflight, "R-OPCD P0 preflight")
    state = "passed" if teacher_gate["passed"] is True else "teacher_failed"
    context.manifest.record_preflight(
        state=state,
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
    g1c_run_root: Path,
    plan_root: Path,
    output_root: Path,
    max_units: int = 1,
    target_unit_id: str | None = None,
) -> dict[str, Any]:
    if max_units < 1 or max_units > EXPECTED_UNITS:
        raise ValueError("R-OPCD P0 max_units must be between 1 and 44")
    context = _load_context(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
        output_root=output_root,
    )
    context.manifest.require_preflight()
    environment = execution_environment(P0_ROPCD_RECIPE)
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
    session = TrainingSession.load(P0_ROPCD_RECIPE)
    try:
        plan_by_id = {str(unit["unit_id"]): unit for unit in context.bundle["plan"]["units"]}
        for unit in units:
            unit_id = str(unit["unit_id"])
            item_id = str(unit["item_id"])
            item_root = _root_dir(output_root, item_id)
            root = prepare_root_adapter(
                session,
                root=item_root,
                run_id=str(context.manifest.payload["run_id"]),
                item_id=item_id,
                root_seed=int(unit["root_adapter_seed"]),
            )
            context.manifest.record_root(
                item_id=item_id,
                root_path=item_root,
                adapter_sha256=str(root["adapter_sha256"]),
                metadata_sha256=str(root["metadata_sha256"]),
            )
            parent_id = unit["parent_unit_id"]
            if parent_id is None:
                parent_adapter = item_root / "adapter"
            else:
                if context.manifest.unit_state(str(parent_id)) not in {
                    UnitState.TRAINED,
                    UnitState.EVALUATING,
                    UnitState.VERIFIED,
                }:
                    raise RuntimeError(f"R-OPCD P0 parent is not trained: {parent_id}")
                parent_unit = plan_by_id[str(parent_id)]
                parent_adapter = _unit_root(output_root, str(parent_id)) / "adapter"
                _verify_unit_artifact(output_root, context, parent_unit)
            if context.manifest.unit_state(unit_id) is UnitState.PLANNED:
                context.manifest.mark_unit(unit_id, UnitState.TRAINING)
            try:
                metadata = train_p0_unit(
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
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    plan_root: Path,
    output_root: Path,
    max_units: int = 1,
    target_unit_id: str | None = None,
) -> dict[str, Any]:
    if max_units < 1 or max_units > EXPECTED_UNITS:
        raise ValueError("R-OPCD P0 max_units must be between 1 and 44")
    context = _load_context(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
        output_root=output_root,
    )
    context.manifest.require_all_trained()
    environment = execution_environment(P0_ROPCD_RECIPE)
    context.manifest.require_environment(environment["environment_fingerprint"])
    _verify_all_lineages(output_root, context)
    session = ScoringSession.load(P0_ROPCD_RECIPE)
    try:
        _ensure_no_memory_controls(session, output_root, context)
        units = _selected_units(
            context.bundle["plan"]["units"],
            context.manifest,
            eligible={UnitState.TRAINED, UnitState.EVALUATING},
            max_units=max_units,
            target_unit_id=target_unit_id,
        )
        probes = {probe.probe_id: probe for probe in compile_bank().probes}
        blocks = {block.block_id: block for block in compile_bank().event_blocks}
        for unit in units:
            unit_id = str(unit["unit_id"])
            logical_name = str(unit["logical_name"])
            if context.manifest.unit_state(unit_id) is UnitState.TRAINED:
                context.manifest.mark_unit(unit_id, UnitState.EVALUATING)
            needs_scoring = logical_name in SCORED_LOGICAL_NAMES or bool(
                unit["technical_duplicate"]
            )
            if not needs_scoring:
                context.manifest.mark_unit(unit_id, UnitState.VERIFIED, scoring_not_required=True)
                continue
            result_path = _result_path(output_root, unit_id)
            if result_path.exists():
                rows = read_jsonl(result_path, f"R-OPCD P0 result {unit_id}")
                _verify_unit_rows(rows, unit)
            else:
                session.activate(_unit_root(output_root, unit_id) / "adapter")
                try:
                    if bool(unit["technical_duplicate"]):
                        rows = _score_duplicate(session, unit, probes)
                    elif logical_name in {"A2", "B2"}:
                        rows = _score_anchor(session, unit, probes)
                    else:
                        rows = _score_final_path(session, unit, probes, blocks)
                finally:
                    session.unload()
                _verify_unit_rows(rows, unit)
                immutable_jsonl_write(result_path, rows, "R-OPCD P0 evaluated unit")
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
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    plan_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    context = _load_context(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
        output_root=output_root,
    )
    context.manifest.require_all_verified()
    rows = _load_rows(output_root, context)
    integrity = _integrity_checks(
        bundle_root, parent_manifest, g1c_run_root, plan_root, output_root, context
    )
    summary = _summary(rows, integrity, context)
    aggregate_root = output_root / "aggregate"
    summary_path = aggregate_root / "summary.json"
    report_path = aggregate_root / "report.md"
    immutable_json_write(summary_path, summary, "R-OPCD P0 summary")
    _immutable_text_write(report_path, _render_report(summary), "R-OPCD P0 report")
    context.manifest.record_aggregate(
        summary_path=summary_path, report_path=report_path, summary=summary
    )
    return summary


def verify_run(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    plan_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    context = _load_context(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
        output_root=output_root,
    )
    context.manifest.require_all_verified()
    rows = _load_rows(output_root, context)
    integrity = _integrity_checks(
        bundle_root, parent_manifest, g1c_run_root, plan_root, output_root, context
    )
    expected = _summary(rows, integrity, context)
    summary_path = output_root / "aggregate/summary.json"
    report_path = output_root / "aggregate/report.md"
    observed = json.loads(summary_path.read_text(encoding="utf-8"))
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise ValueError("R-OPCD P0 aggregate regeneration mismatch")
    if report_path.read_text(encoding="utf-8") != _render_report(expected):
        raise ValueError("R-OPCD P0 report regeneration mismatch")
    aggregate = context.manifest.payload.get("aggregate", {})
    if (
        aggregate.get("state") != "complete"
        or aggregate.get("summary_sha256") != file_hash(summary_path)
        or aggregate.get("report_sha256") != file_hash(report_path)
        or canonical_json_bytes(aggregate.get("gate")) != canonical_json_bytes(expected["gate"])
    ):
        raise ValueError("R-OPCD P0 manifest aggregate binding mismatch")
    if context.manifest.payload.get("path_contrast_computed") is not True:
        raise ValueError("R-OPCD P0 manifest did not record its smoke contrast")
    return {
        "passed": True,
        "run_id": context.manifest.payload["run_id"],
        "plan_id": context.bundle["plan"]["plan_id"],
        "g1c_run_id": context.bundle["handoff"]["run_id"],
        "g2_gate": expected["gate"],
        "p1_implementation_review_eligible": expected["gate"]["passed"],
        "p1_authorized": False,
        "kill_or_reserve_accessed": False,
    }


def _load_context(
    *,
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    plan_root: Path,
    output_root: Path,
) -> RunContext:
    plan_report = verify_plan_bundle(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
    )
    bundle = load_plan_bundle(plan_root)
    template = build_authorization_template(
        plan_root=plan_root, output_root=output_root, plan_report=plan_report
    )
    authorization = verify_adopted_authorization(output_root, expected_template=template)
    _, implementation_sha256 = implementation_identity()
    manifest = P0ExecutionManifest.load_or_create(
        output_root / "manifest.json",
        plan_manifest_id=plan_report["manifest_id"],
        plan=bundle["plan"],
        authorization=authorization,
        implementation_sha256=implementation_sha256,
    )
    return RunContext(
        plan_report=plan_report,
        bundle=bundle,
        authorization=authorization,
        manifest=manifest,
    )


def _selected_units(
    units: list[dict[str, Any]],
    manifest: P0ExecutionManifest,
    *,
    eligible: set[UnitState],
    max_units: int,
    target_unit_id: str | None,
) -> list[dict[str, Any]]:
    ordered = sorted(units, key=lambda unit: int(unit["sequence_index"]))
    if target_unit_id is not None:
        ordered = [unit for unit in ordered if unit["unit_id"] == target_unit_id]
        if not ordered:
            raise KeyError(f"unknown R-OPCD P0 target unit: {target_unit_id}")
    return [unit for unit in ordered if manifest.unit_state(str(unit["unit_id"])) in eligible][
        :max_units
    ]


def _ensure_no_memory_controls(
    session: ScoringSession, output_root: Path, context: RunContext
) -> None:
    path = output_root / "results/no_memory.jsonl"
    recorded = context.manifest.payload.get("control_results", {})
    if path.exists():
        rows = read_jsonl(path, "R-OPCD P0 no-memory controls")
        if len(rows) != 112 or any(row.get("arm") != "no_memory" for row in rows):
            raise ValueError("R-OPCD P0 no-memory controls changed")
        if recorded.get("state") == "complete" and recorded.get("result_sha256") != file_hash(path):
            raise ValueError("R-OPCD P0 no-memory control hash changed")
    else:
        if recorded.get("state") == "complete":
            raise FileNotFoundError("R-OPCD P0 no-memory controls are missing")
        bank = compile_bank()
        smoke_items = {item.item_id for item in bank.items if item.split == "smoke"}
        probes = sorted(
            (
                probe
                for probe in bank.probes
                if probe.item_id in smoke_items and probe.core_endpoint
            ),
            key=lambda probe: probe.probe_id,
        )
        rows = [
            _score(
                session,
                unit_id=f"p0r:{probe.item_id}:control:{probe.terminal_state}",
                logical_name="N",
                probe=probe,
                raw_prompt=probe.prompt,
                arm="no_memory",
                disable_adapter=False,
                route_key=None,
                route_unit_id=None,
            )
            for probe in probes
        ]
        immutable_jsonl_write(path, rows, "R-OPCD P0 no-memory controls")
    context.manifest.record_controls(result_path=path, rows=len(rows))


def _score_anchor(
    session: ScoringSession, unit: dict[str, Any], probes: dict[str, Probe]
) -> list[dict[str, Any]]:
    return [
        _score(
            session,
            unit_id=str(unit["unit_id"]),
            logical_name=str(unit["logical_name"]),
            probe=probes[str(probe_id)],
            raw_prompt=probes[str(probe_id)].prompt,
            arm="p_latest",
            disable_adapter=False,
            route_key=str(unit["route_key"]),
            route_unit_id=str(unit["unit_id"]),
        )
        for probe_id in unit["core_probe_ids"]
    ]


def _score_duplicate(
    session: ScoringSession, unit: dict[str, Any], probes: dict[str, Probe]
) -> list[dict[str, Any]]:
    return [
        {
            **_score(
                session,
                unit_id=str(unit["unit_id"]),
                logical_name=str(unit["logical_name"]),
                probe=probes[str(probe_id)],
                raw_prompt=probes[str(probe_id)].prompt,
                arm="technical_duplicate",
                disable_adapter=False,
                route_key=str(unit["route_key"]),
                route_unit_id=str(unit["unit_id"]),
            ),
            "duplicate_of": unit["duplicate_of"],
        }
        for probe_id in unit["core_probe_ids"]
    ]


def _score_final_path(
    session: ScoringSession,
    unit: dict[str, Any],
    probes: dict[str, Probe],
    blocks: dict[str, Any],
) -> list[dict[str, Any]]:
    logical_name = str(unit["logical_name"])
    external_state, icl_state = _render_states(str(unit["item_id"]), logical_name, blocks)
    rows: list[dict[str, Any]] = []
    for probe_id in unit["core_probe_ids"]:
        probe = probes[str(probe_id)]
        external_prompt = render_fresh_session_probe(
            probe, persistent_state=external_state, access_mode=AccessMode.ON
        )
        icl_prompt = render_fresh_session_probe(
            probe, persistent_state=icl_state, access_mode=AccessMode.ON
        )
        specs = (
            ("base_before", probe.prompt, True),
            ("parametric_path", probe.prompt, False),
            ("parametric_rescore", probe.prompt, False),
            ("adapter_disabled", probe.prompt, True),
            ("external_latest", external_prompt, True),
            ("icl_history", icl_prompt, True),
            ("both", external_prompt, False),
        )
        for arm, raw_prompt, disabled in specs:
            rows.append(
                _score(
                    session,
                    unit_id=str(unit["unit_id"]),
                    logical_name=logical_name,
                    probe=probe,
                    raw_prompt=raw_prompt,
                    arm=arm,
                    disable_adapter=disabled,
                    route_key=str(unit["route_key"]) if not disabled else None,
                    route_unit_id=str(unit["unit_id"]) if not disabled else None,
                )
            )
    for probe_id in unit["unrelated_probe_ids"]:
        probe = probes[str(probe_id)]
        for arm, disabled in (("base_unrelated", True), ("parametric_unrelated", False)):
            rows.append(
                _score(
                    session,
                    unit_id=str(unit["unit_id"]),
                    logical_name=logical_name,
                    probe=probe,
                    raw_prompt=probe.prompt,
                    arm=arm,
                    disable_adapter=disabled,
                    route_key=str(unit["route_key"]) if not disabled else None,
                    route_unit_id=str(unit["unit_id"]) if not disabled else None,
                )
            )
    for probe_id in unit["qualification_probe_ids"]:
        probe = probes[str(probe_id)]
        rows.append(
            _score(
                session,
                unit_id=str(unit["unit_id"]),
                logical_name=logical_name,
                probe=probe,
                raw_prompt=probe.prompt,
                arm="path_qualification",
                disable_adapter=False,
                route_key=str(unit["route_key"]),
                route_unit_id=str(unit["unit_id"]),
            )
        )
    return rows


def _score(
    session: ScoringSession,
    *,
    unit_id: str,
    logical_name: str,
    probe: Probe,
    raw_prompt: str,
    arm: str,
    disable_adapter: bool,
    route_key: str | None,
    route_unit_id: str | None,
) -> dict[str, Any]:
    row = score_prompt(
        session,
        raw_prompt=raw_prompt,
        action_choices=probe.action_choices,
        expected_action=probe.expected_action,
        obsolete_action=probe.obsolete_action,
        unit_id=unit_id,
        item_id=probe.item_id,
        probe_id=probe.probe_id,
        probe_category=probe.category.value,
        arm=arm,
        disable_adapter=disable_adapter,
        route_key=route_key,
        route_unit_id=route_unit_id,
        metadata={
            "logical_name": logical_name,
            "fresh_session_id": json_hash([unit_id, probe.probe_id, arm]),
        },
    )
    return {
        **row,
        "logical_name": logical_name,
        "terminal_state": probe.terminal_state,
    }


def _render_states(item_id: str, logical_name: str, blocks: dict[str, Any]) -> tuple[str, str]:
    external = VersionedExternalStore()
    icl = ICLHistoryRenderer()
    for suffix in PATH_SUFFIXES[logical_name]:
        block = blocks[f"{item_id}:{suffix}"]
        external.apply(block)
        icl.apply(block)
    return external.render(), icl.render()


def _verify_unit_rows(rows: list[dict[str, Any]], unit: dict[str, Any]) -> None:
    logical_name = str(unit["logical_name"])
    expected = 14 if bool(unit["technical_duplicate"]) or logical_name in {"A2", "B2"} else 118
    if len(rows) != expected:
        raise ValueError(f"R-OPCD P0 row count changed for {unit['unit_id']}")
    if any(row.get("unit_id") != unit["unit_id"] for row in rows):
        raise ValueError("R-OPCD P0 evaluated rows escaped their unit")
    row_ids = [row.get("row_id") for row in rows]
    if None in row_ids or len(row_ids) != len(set(row_ids)):
        raise ValueError("R-OPCD P0 unit row identities changed")


def _verify_all_lineages(output_root: Path, context: RunContext) -> bool:
    item_roots = {
        str(unit["item_id"]): int(unit["root_adapter_seed"])
        for unit in context.bundle["plan"]["units"]
    }
    root_parent = output_root / "roots"
    observed_roots = {path.name for path in root_parent.iterdir() if path.is_dir()}
    if observed_roots != set(item_roots):
        raise ValueError("R-OPCD P0 item-root directory set changed")
    for item_id, root_seed in sorted(item_roots.items()):
        root = verify_root_adapter(
            _root_dir(output_root, item_id),
            run_id=str(context.manifest.payload["run_id"]),
            item_id=item_id,
            root_seed=root_seed,
            recipe=P0_ROPCD_RECIPE,
        )
        recorded_root = context.manifest.payload.get("roots", {}).get(item_id, {})
        if (
            recorded_root.get("state") != "verified"
            or recorded_root.get("root_adapter_seed") != root_seed
            or recorded_root.get("adapter_sha256") != root["adapter_sha256"]
            or recorded_root.get("metadata_sha256") != root["metadata_sha256"]
        ):
            raise ValueError(f"R-OPCD P0 root manifest binding changed: {item_id}")
    expected_dirs = {unit_slug(str(unit["unit_id"])) for unit in context.bundle["plan"]["units"]}
    unit_root = output_root / "units"
    observed_dirs = {path.name for path in unit_root.iterdir() if path.is_dir()}
    if observed_dirs != expected_dirs:
        raise ValueError("R-OPCD P0 trained-unit directory set changed")
    for unit in sorted(context.bundle["plan"]["units"], key=lambda row: row["sequence_index"]):
        _verify_unit_artifact(output_root, context, unit)
    return True


def _verify_unit_artifact(
    output_root: Path, context: RunContext, unit: dict[str, Any]
) -> dict[str, Any]:
    parent_id = unit["parent_unit_id"]
    parent_adapter = (
        _root_dir(output_root, str(unit["item_id"])) / "adapter"
        if parent_id is None
        else _unit_root(output_root, str(parent_id)) / "adapter"
    )
    root = context.manifest.payload["roots"][str(unit["item_id"])]
    metadata = verify_p0_unit(
        _unit_root(output_root, str(unit["unit_id"])),
        unit=unit,
        parent_adapter_sha256=_adapter_hash(parent_adapter),
        root_adapter_sha256=str(root["adapter_sha256"]),
        run_id=str(context.manifest.payload["run_id"]),
        recipe=P0_ROPCD_RECIPE,
    )
    manifest_metadata = context.manifest.payload["units"][str(unit["unit_id"])]["metadata"]
    if any(
        metadata[field] != manifest_metadata.get(field)
        for field in (
            "adapter_sha256",
            "parent_adapter_sha256",
            "training_metadata_sha256",
            "training_records_sha256",
        )
    ):
        raise ValueError(f"R-OPCD P0 unit manifest binding changed: {unit['unit_id']}")
    return metadata


def _load_rows(output_root: Path, context: RunContext) -> list[dict[str, Any]]:
    controls = context.manifest.payload.get("control_results", {})
    control_path = output_root / "results/no_memory.jsonl"
    if (
        controls.get("state") != "complete"
        or controls.get("result_sha256") != file_hash(control_path)
        or controls.get("rows") != 112
    ):
        raise ValueError("R-OPCD P0 no-memory control binding changed")
    rows = read_jsonl(control_path, "R-OPCD P0 no-memory controls")
    for unit in context.bundle["plan"]["units"]:
        logical_name = str(unit["logical_name"])
        if logical_name not in SCORED_LOGICAL_NAMES and not bool(unit["technical_duplicate"]):
            continue
        unit_id = str(unit["unit_id"])
        metadata = context.manifest.payload["units"][unit_id]["metadata"]
        path = _result_path(output_root, unit_id)
        if metadata.get("result_path") != str(path.resolve()) or metadata.get(
            "result_sha256"
        ) != file_hash(path):
            raise ValueError(f"R-OPCD P0 result binding changed: {unit_id}")
        unit_rows = read_jsonl(path, f"R-OPCD P0 result {unit_id}")
        _verify_unit_rows(unit_rows, unit)
        rows.extend(unit_rows)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"R-OPCD P0 matrix row count changed: {len(rows)}")
    return rows


def _integrity_checks(
    bundle_root: Path,
    parent_manifest: Path,
    g1c_run_root: Path,
    plan_root: Path,
    output_root: Path,
    context: RunContext,
) -> dict[str, bool]:
    plan_report = verify_plan_bundle(
        bundle_root=bundle_root,
        parent_manifest=parent_manifest,
        g1c_run_root=g1c_run_root,
        plan_root=plan_root,
    )
    preflight = context.manifest.require_preflight()
    preflight_artifact = json.loads(
        (output_root / "preflight/preflight.json").read_text(encoding="utf-8")
    )
    lineage = _verify_all_lineages(output_root, context)
    plan_units = context.bundle["plan"]["units"]
    by_item_logical = {
        (str(unit["item_id"]), str(unit["logical_name"])): unit for unit in plan_units
    }
    exposure_matches = all(
        by_item_logical[(item_id, left)]["ordered_exposure_sha256"]
        == by_item_logical[(item_id, right)]["ordered_exposure_sha256"]
        for item_id in sorted({str(unit["item_id"]) for unit in plan_units})
        for left, right in (("ABA", "BAA"), ("BAB", "ABB"))
    )
    duplicate_matches = all(
        unit["ordered_exposure_sha256"]
        == by_item_logical[(str(unit["item_id"]), str(unit["duplicate_of"]))][
            "ordered_exposure_sha256"
        ]
        and unit["parent_unit_id"]
        == by_item_logical[(str(unit["item_id"]), str(unit["duplicate_of"]))]["parent_unit_id"]
        for unit in plan_units
        if unit["technical_duplicate"]
    )
    return {
        "plan_bundle_reverified": plan_report["passed"] is True,
        "g1c_handoff_reverified": plan_report["g1c_handoff_id"]
        == context.bundle["handoff"]["handoff_id"],
        "authorization_bound": context.authorization["authorization_id"]
        == context.manifest.payload["identity"]["authorization_id"],
        "preflight_hash_bound": file_hash(output_root / "preflight/preflight.json")
        == preflight["preflight_sha256"],
        "environment_hash_bound": file_hash(output_root / "preflight/environment.json")
        == preflight["environment_sha256"],
        "teacher_rows_hash_bound": file_hash(output_root / "preflight/teacher_rows.jsonl")
        == preflight["teacher_rows_sha256"],
        "teacher_gate_passed": preflight["teacher_gate"]["passed"] is True,
        "prompt_audit_passed": preflight_artifact["prompt_audit"]["passed"] is True,
        "root_and_lineage_verified": lineage,
        "comparator_event_exposure_matched": exposure_matches,
        "technical_duplicate_parent_event_exposure_matched": duplicate_matches,
        "all_units_verified": all(
            unit["state"] == UnitState.VERIFIED.value
            for unit in context.manifest.payload["units"].values()
        ),
        "no_unplanned_downstream_state": all(
            context.manifest.payload.get(field) is False
            for field in (
                "p1_authorized",
                "kill_or_reserve_accessed",
                "learned_router_started",
                "rl_controller_started",
                "automatic_hyperparameter_search_started",
            )
        ),
    }


def _summary(
    rows: list[dict[str, Any]], integrity: dict[str, bool], context: RunContext
) -> dict[str, Any]:
    summary = summarize_p0(
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
        "g1c_handoff_id": context.bundle["handoff"]["handoff_id"],
        "g1c_run_id": context.bundle["handoff"]["run_id"],
        "authorization_id": context.authorization["authorization_id"],
        "recipe_sha256": context.bundle["plan"]["recipe_sha256"],
    }


def _audit_prompts(tokenizer: Any, plan: dict[str, Any]) -> dict[str, Any]:
    probes = {probe.probe_id: probe for probe in compile_bank().probes}
    blocks = {block.block_id: block for block in compile_bank().event_blocks}
    rows: list[dict[str, Any]] = []

    def audit(unit_id: str, probe_id: str, kind: str, raw_prompt: str, choices: list[str]) -> None:
        result = audit_candidate_tokenization(
            tokenizer,
            chat_prompt(tokenizer, raw_prompt),
            tuple(choices),
            evaluation_max_length=P0_ROPCD_RECIPE.evaluation_max_length,
        )
        rows.append(
            {
                "unit_id": unit_id,
                "probe_id": probe_id,
                "prompt_kind": kind,
                "prompt_sha256": result["prompt_sha256"],
                "prompt_token_count": result["untruncated_prompt_token_count"],
                "all_candidates_valid": result["all_candidates_valid"],
                "rollout_length_fits": (
                    result["untruncated_prompt_token_count"]
                    + P0_ROPCD_RECIPE.rollout_max_new_tokens
                    <= P0_ROPCD_RECIPE.training_max_length
                    if kind.startswith("training_")
                    else True
                ),
            }
        )

    for unit in plan["units"]:
        for pair in unit["rollout_prompt_pairs"]:
            audit(
                str(unit["unit_id"]),
                str(pair["probe_id"]),
                "training_student",
                str(pair["student_prompt"]),
                pair["action_choices"],
            )
            audit(
                str(unit["unit_id"]),
                str(pair["probe_id"]),
                "training_teacher",
                str(pair["privileged_teacher_prompt"]),
                pair["action_choices"],
            )
    smoke_items = sorted({str(unit["item_id"]) for unit in plan["units"]})
    for probe in sorted(
        (
            probe
            for probe in probes.values()
            if probe.item_id in smoke_items and probe.core_endpoint
        ),
        key=lambda probe: probe.probe_id,
    ):
        audit(
            f"p0r:{probe.item_id}:control:{probe.terminal_state}",
            probe.probe_id,
            "evaluation_no_memory",
            probe.prompt,
            list(probe.action_choices),
        )
    for unit in plan["units"]:
        logical_name = str(unit["logical_name"])
        if bool(unit["technical_duplicate"]):
            for probe_id in unit["core_probe_ids"]:
                probe = probes[str(probe_id)]
                audit(
                    str(unit["unit_id"]),
                    probe.probe_id,
                    "evaluation_duplicate",
                    probe.prompt,
                    list(probe.action_choices),
                )
        elif logical_name in {"A2", "B2"}:
            for probe_id in unit["core_probe_ids"]:
                probe = probes[str(probe_id)]
                audit(
                    str(unit["unit_id"]),
                    probe.probe_id,
                    "evaluation_latest",
                    probe.prompt,
                    list(probe.action_choices),
                )
        elif logical_name in FINAL_PATHS:
            external_state, icl_state = _render_states(str(unit["item_id"]), logical_name, blocks)
            for probe_id in unit["core_probe_ids"]:
                probe = probes[str(probe_id)]
                external_prompt = render_fresh_session_probe(
                    probe, persistent_state=external_state, access_mode=AccessMode.ON
                )
                icl_prompt = render_fresh_session_probe(
                    probe, persistent_state=icl_state, access_mode=AccessMode.ON
                )
                for kind, raw_prompt in (
                    ("evaluation_base_before", probe.prompt),
                    ("evaluation_parametric", probe.prompt),
                    ("evaluation_rescore", probe.prompt),
                    ("evaluation_disabled", probe.prompt),
                    ("evaluation_external", external_prompt),
                    ("evaluation_icl", icl_prompt),
                    ("evaluation_both", external_prompt),
                ):
                    audit(
                        str(unit["unit_id"]),
                        probe.probe_id,
                        kind,
                        raw_prompt,
                        list(probe.action_choices),
                    )
            for probe_id in unit["unrelated_probe_ids"]:
                probe = probes[str(probe_id)]
                for kind in ("evaluation_base_unrelated", "evaluation_parametric_unrelated"):
                    audit(
                        str(unit["unit_id"]),
                        probe.probe_id,
                        kind,
                        probe.prompt,
                        list(probe.action_choices),
                    )
            for probe_id in unit["qualification_probe_ids"]:
                probe = probes[str(probe_id)]
                audit(
                    str(unit["unit_id"]),
                    probe.probe_id,
                    "evaluation_path_qualification",
                    probe.prompt,
                    list(probe.action_choices),
                )
    expected_count = 352 + EXPECTED_ROWS
    return {
        "passed": len(rows) == expected_count
        and all(row["all_candidates_valid"] and row["rollout_length_fits"] for row in rows),
        "audit_count": len(rows),
        "expected_audit_count": expected_count,
        "maximum_prompt_tokens": max(int(row["prompt_token_count"]) for row in rows),
        "rows": rows,
    }


def _teacher_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = EXPECTED_UNITS * 4
    current_top1 = sum(row.get("correct") is True for row in rows) / max(len(rows), 1)
    obsolete = sum(row.get("obsolete_intrusion") is True for row in rows) / max(len(rows), 1)
    parser = sum(row.get("parser_valid") is True for row in rows) / max(len(rows), 1)
    checks = {
        "exact_176_rows": len(rows) == expected,
        "current_top1_at_least_95pct": current_top1 >= 0.95,
        "obsolete_intrusion_at_most_5pct": obsolete <= 0.05,
        "parser_validity_exactly_100pct": parser == 1.0,
    }
    return {
        "gate": "P0-R-OPCD-teacher",
        "passed": all(checks.values()),
        "metrics": {
            "current_top1": current_top1,
            "obsolete_intrusion": obsolete,
            "parser_validity": parser,
        },
        "checks": checks,
        "blockers": [name for name, passed in checks.items() if not passed],
    }


def _verify_preflight_artifacts(output_root: Path, context: RunContext) -> None:
    preflight = context.manifest.payload["preflight"]
    paths = {
        "environment_sha256": output_root / "preflight/environment.json",
        "preflight_sha256": output_root / "preflight/preflight.json",
        "teacher_rows_sha256": output_root / "preflight/teacher_rows.jsonl",
    }
    for field, path in paths.items():
        if file_hash(path) != preflight.get(field):
            raise ValueError(f"R-OPCD P0 preflight artifact changed: {field}")


def _plan_snapshot(plan_root: Path) -> dict[str, str]:
    return {
        path.relative_to(plan_root).as_posix(): file_hash(path)
        for path in sorted(plan_root.rglob("*"))
        if path.is_file()
    }


def _unit_root(output_root: Path, unit_id: str) -> Path:
    return output_root / "units" / unit_slug(unit_id)


def _root_dir(output_root: Path, item_id: str) -> Path:
    return output_root / "roots" / item_id


def _result_path(output_root: Path, unit_id: str) -> Path:
    return output_root / "results" / f"{unit_slug(unit_id)}.jsonl"


def _adapter_hash(path: Path) -> str:
    from plasticity_placement.pathmem_exec.artifacts import adapter_bundle_hash

    return adapter_bundle_hash(path)


def _progress(manifest: P0ExecutionManifest, stage: str) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for unit in manifest.payload["units"].values():
        counts[unit["state"]] = counts.get(unit["state"], 0) + 1
    return {
        "stage": stage,
        "run_id": manifest.payload["run_id"],
        "unit_states": counts,
        "training_started": manifest.payload["training_started"],
        "gpu_inference_started": manifest.payload["gpu_inference_started"],
        "path_contrast_computed": manifest.payload["path_contrast_computed"],
        "p1_authorized": False,
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
    metrics = summary["metrics"]
    gate = summary["gate"]
    return "\n".join(
        [
            "# PathMem R-OPCD-specific P0 engineering smoke",
            "",
            f"- Run ID: `{summary['run_id']}`",
            f"- G2 gate passed: `{gate['passed']}`",
            f"- Blockers: `{', '.join(gate['blockers']) or 'none'}`",
            "- Parametric endpoint defect: "
            f"`{metrics['parametric_item_mean_defect_nats']:.6f}` nats",
            f"- External defect upper 95%: `{metrics['external_defect_upper_95_nats']:.6f}` nats",
            f"- ICL endpoint defect: `{metrics['icl_item_mean_defect_nats']:.6f}` nats",
            f"- Both endpoint defect: `{metrics['both_item_mean_defect_nats']:.6f}` nats",
            f"- Path write qualification rate: `{metrics['path_write_qualification_rate']:.4f}`",
            f"- Joint path pass rate: `{metrics['joint_path_pass_rate']:.4f}`",
            f"- Unrelated regression: `{metrics['unrelated_regression_pp']:.3f}` pp",
            "",
            "This is engineering-smoke evidence only. It does not establish a "
            "scientific path result,",
            "change epsilon_JS, authorize P1, expose kill/reserve data, or enable "
            "learned/RL routing.",
            "",
        ]
    )
