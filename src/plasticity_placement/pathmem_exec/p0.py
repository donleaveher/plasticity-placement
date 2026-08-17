from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.backends import (
    ICLHistoryRenderer,
    VersionedExternalStore,
    render_fresh_session_probe,
)
from plasticity_placement.pathmem.compiler import compile_bank
from plasticity_placement.pathmem.dag import CoreDAGPlan, DAGNodePlan, build_core_dag_plan
from plasticity_placement.pathmem.io import file_hash, immutable_json_write, json_hash
from plasticity_placement.pathmem.schema import AccessMode, Probe, SemanticItem
from plasticity_placement.pathmem.scoring import audit_candidate_tokenization
from plasticity_placement.pathmem_exec.analysis import summarize_p0
from plasticity_placement.pathmem_exec.artifacts import (
    AttemptState,
    PhaseManifest,
    adapter_bundle_hash,
    immutable_jsonl_write,
    read_jsonl,
    verify_g1_authorization,
)
from plasticity_placement.pathmem_exec.config import (
    P0_CONFIG,
    P0_DUPLICATE_NODE_BY_ITEM_INDEX,
    P0_SCHEMA_VERSION,
)
from plasticity_placement.pathmem_exec.environment import (
    execution_environment,
    resolve_and_verify_model,
    verify_g0_or_raise,
)
from plasticity_placement.pathmem_exec.scoring import (
    activate_adapter,
    load_model_bundle,
    release_model_bundle,
    score_probe,
)
from plasticity_placement.pathmem_exec.training import (
    create_root_adapter,
    runtime_identity,
    train_event_node,
)
from plasticity_placement.training.model_utils import chat_prompt

PATH_SUFFIXES = {
    "ABA": ("A1", "B1", "A2"),
    "BAA": ("B1", "A1", "A2"),
    "BAB": ("B1", "A1", "B2"),
    "ABB": ("A1", "B1", "B2"),
}
FINAL_PATHS = tuple(PATH_SUFFIXES)
SCORED_CORE_NODES = {*FINAL_PATHS, "A2", "B2"}


def run_p0(
    *,
    output_root: Path,
    g0_dir: Path,
    protocol_root: Path,
    g1_authorization_path: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    g0 = verify_g0_or_raise(g0_dir, protocol_root)
    authorization = verify_g1_authorization(
        g1_authorization_path,
        expected_g0_manifest_id=str(g0["manifest_id"]),
    )
    environment = execution_environment(P0_CONFIG)
    bank = compile_bank()
    items = tuple(item for item in bank.items if item.split == P0_CONFIG.split)
    if len(items) != 4:
        raise RuntimeError(f"P0 split changed: {len(items)} items")
    blocks_by_item = _group_by_item(
        bank.event_blocks, lambda block: block.block_id.split(":", 1)[0]
    )
    probes_by_item = _group_by_item(bank.probes, lambda probe: probe.item_id)
    families_by_item = _group_by_item(bank.history_families, lambda family: family.item_id)

    tokenizer, _ = resolve_and_verify_model(P0_CONFIG)
    preflight = _preflight(items, blocks_by_item, probes_by_item, tokenizer, environment)
    immutable_json_write(output_root / "preflight.json", preflight, "P0 preflight")

    runtimes: dict[str, Any] = {}
    dags: dict[str, CoreDAGPlan] = {}
    roots: dict[str, Path] = {}
    plans: dict[str, DAGNodePlan] = {}
    planned: dict[str, dict[str, Any]] = {}
    execution_order: list[str] = []
    for item_index, item in enumerate(items):
        root_dir = output_root / "roots" / item.item_id
        root = create_root_adapter(config=P0_CONFIG, item_id=item.item_id, output_dir=root_dir)
        runtime = runtime_identity(
            config=P0_CONFIG,
            item_id=item.item_id,
            root_adapter_sha256=str(root["adapter_sha256"]),
            environment=environment,
        )
        dag = build_core_dag_plan(
            tuple(families_by_item[item.item_id]),
            tuple(blocks_by_item[item.item_id]),
            runtime,
        )
        runtimes[item.item_id] = runtime
        dags[item.item_id] = dag
        roots[item.item_id] = root_dir
        for plan in dag.nodes:
            attempt_id = _attempt_id(item.item_id, plan.logical_name)
            parent_attempt_id = (
                None
                if plan.parent_logical_name == "ROOT"
                else _attempt_id(item.item_id, plan.parent_logical_name)
            )
            planned[attempt_id] = {
                "item_id": item.item_id,
                "logical_name": plan.logical_name,
                "plan_id": plan.plan_id,
                "parent_attempt_id": parent_attempt_id,
                "technical_duplicate": False,
            }
            plans[attempt_id] = plan
            execution_order.append(attempt_id)
        duplicate_of = P0_DUPLICATE_NODE_BY_ITEM_INDEX[item_index]
        original_plan = next(node for node in dag.nodes if node.logical_name == duplicate_of)
        duplicate_id = _attempt_id(item.item_id, f"DUP-{duplicate_of}")
        duplicate_parent = _attempt_id(item.item_id, original_plan.parent_logical_name)
        planned[duplicate_id] = {
            "item_id": item.item_id,
            "logical_name": f"DUP-{duplicate_of}",
            "duplicate_of": duplicate_of,
            "plan_id": json_hash(
                {"original_plan_id": original_plan.plan_id, "duplicate": "technical-1"}
            ),
            "parent_attempt_id": duplicate_parent,
            "technical_duplicate": True,
        }
        plans[duplicate_id] = original_plan
        execution_order.append(duplicate_id)

    if len(planned) != 44:
        raise RuntimeError(f"P0 planned artifact count changed: {len(planned)}")
    identity = {
        "schema_version": P0_SCHEMA_VERSION,
        "phase_config": P0_CONFIG.to_dict(),
        "g1_authorization_id": authorization["authorization_id"],
        "environment_fingerprint": environment["environment_fingerprint"],
        "preflight_sha256": file_hash(output_root / "preflight.json"),
    }
    manifest = PhaseManifest.load_or_create(
        output_root / "manifest.json",
        phase="P0",
        g0_manifest_id=str(g0["manifest_id"]),
        identity=identity,
        planned_units=planned,
    )

    for attempt_id in execution_order:
        state = manifest.state(attempt_id)
        if state is AttemptState.FAILED:
            raise RuntimeError(f"P0 attempt is terminally failed; use a new output: {attempt_id}")
        if state in {AttemptState.TRAINED, AttemptState.SCORING, AttemptState.VERIFIED}:
            continue
        unit = planned[attempt_id]
        item_id = str(unit["item_id"])
        runtime = runtimes[item_id]
        plan = plans[attempt_id]
        block = next(block for block in blocks_by_item[item_id] if block.block_id == plan.block_id)
        parent_attempt_id = unit["parent_attempt_id"]
        if parent_attempt_id is None:
            parent_adapter_dir = roots[item_id]
            parent_state_sha256 = runtime.root_state_sha256
        else:
            parent_adapter_dir = _adapter_dir(output_root, str(parent_attempt_id))
            parent_state_sha256 = str(_read_lineage(parent_adapter_dir)["node_id"])
        manifest.mark(attempt_id, AttemptState.TRAINING)
        try:
            lineage = train_event_node(
                config=P0_CONFIG,
                attempt_id=attempt_id,
                block=block,
                plan=plan,
                dag=dags[item_id],
                runtime=runtime,
                parent_adapter_dir=parent_adapter_dir,
                parent_state_sha256=parent_state_sha256,
                root_state_sha256=runtime.root_state_sha256,
                data_path=output_root / "data" / f"{block.block_id.replace(':', '__')}.jsonl",
                output_dir=_adapter_dir(output_root, attempt_id),
                occurrence_id=(
                    f"{plan.block_id.rsplit(':', 1)[-1]}#technical-duplicate-1"
                    if bool(unit["technical_duplicate"])
                    else None
                ),
            )
            manifest.mark(
                attempt_id,
                AttemptState.TRAINED,
                node_id=lineage["node_id"],
                adapter_sha256=lineage["adapter_sha256"],
            )
        except BaseException as error:
            manifest.record_error(attempt_id, error)
            manifest.mark(attempt_id, AttemptState.FAILED)
            raise

    control_path = output_root / "results" / "no_memory.jsonl"
    bundle = load_model_bundle(P0_CONFIG)
    all_rows: list[dict[str, Any]] = []
    try:
        if control_path.exists():
            all_rows.extend(read_jsonl(control_path))
        else:
            controls = _score_no_memory(
                bundle=bundle,
                items=items,
                probes_by_item=probes_by_item,
                run_id=str(manifest.payload["run_id"]),
            )
            immutable_jsonl_write(control_path, controls, "P0 no-memory controls")
            all_rows.extend(controls)

        for attempt_id in execution_order:
            unit = planned[attempt_id]
            logical_name = str(unit["logical_name"])
            needs_scoring = logical_name in SCORED_CORE_NODES or bool(unit["technical_duplicate"])
            result_path = output_root / "results" / f"{attempt_id.replace('::', '__')}.jsonl"
            state = manifest.state(attempt_id)
            if state is AttemptState.VERIFIED:
                if needs_scoring:
                    all_rows.extend(read_jsonl(result_path))
                continue
            if state is AttemptState.TRAINED:
                manifest.mark(attempt_id, AttemptState.SCORING)
            if not needs_scoring:
                manifest.mark(attempt_id, AttemptState.VERIFIED, scoring_not_required=True)
                continue
            if result_path.exists():
                rows = read_jsonl(result_path)
            else:
                item_id = str(unit["item_id"])
                item = next(candidate for candidate in items if candidate.item_id == item_id)
                lineage = _read_lineage(_adapter_dir(output_root, attempt_id))
                activate_adapter(bundle, _adapter_dir(output_root, attempt_id), attempt_id)
                if bool(unit["technical_duplicate"]):
                    rows = _score_duplicate(
                        bundle=bundle,
                        item=item,
                        probes=tuple(probes_by_item[item_id]),
                        attempt_id=attempt_id,
                        duplicate_of=str(unit["duplicate_of"]),
                        node_id=str(lineage["node_id"]),
                        adapter_sha256=str(lineage["adapter_sha256"]),
                        run_id=str(manifest.payload["run_id"]),
                    )
                elif logical_name in {"A2", "B2"}:
                    rows = _score_latest_anchor(
                        bundle=bundle,
                        item=item,
                        probes=tuple(probes_by_item[item_id]),
                        logical_name=logical_name,
                        attempt_id=attempt_id,
                        node_id=str(lineage["node_id"]),
                        adapter_sha256=str(lineage["adapter_sha256"]),
                        run_id=str(manifest.payload["run_id"]),
                    )
                else:
                    rows = _score_final_path(
                        bundle=bundle,
                        item=item,
                        probes=tuple(probes_by_item[item_id]),
                        blocks=tuple(blocks_by_item[item_id]),
                        logical_name=logical_name,
                        attempt_id=attempt_id,
                        node_id=str(lineage["node_id"]),
                        adapter_sha256=str(lineage["adapter_sha256"]),
                        run_id=str(manifest.payload["run_id"]),
                    )
                immutable_jsonl_write(result_path, rows, "P0 scored unit")
            manifest.mark(
                attempt_id,
                AttemptState.VERIFIED,
                result_sha256=file_hash(result_path),
            )
            all_rows.extend(rows)
    except BaseException as error:
        active = next(
            (
                attempt_id
                for attempt_id in execution_order
                if manifest.state(attempt_id) is AttemptState.SCORING
            ),
            None,
        )
        if active is not None:
            manifest.record_error(active, error)
            manifest.mark(active, AttemptState.FAILED)
        raise
    finally:
        release_model_bundle(bundle)

    manifest.require_all_verified()
    lineage_verified = _verify_lineages(
        output_root=output_root,
        planned=planned,
        runtimes=runtimes,
    )
    summary = summarize_p0(
        all_rows,
        verified_artifacts=sum(
            manifest.state(attempt_id) is AttemptState.VERIFIED for attempt_id in planned
        ),
        planned_artifacts=len(planned),
        lineage_verified=lineage_verified,
        exposure_verified=bool(preflight["exposure_verified"]),
    )
    summary.update(
        {
            "schema_version": P0_SCHEMA_VERSION,
            "run_id": manifest.payload["run_id"],
            "g0_manifest_id": g0["manifest_id"],
            "g1_authorization_id": authorization["authorization_id"],
            "environment_fingerprint": environment["environment_fingerprint"],
        }
    )
    immutable_json_write(output_root / "summary.json", summary, "P0 summary")
    return summary


def _preflight(
    items: tuple[SemanticItem, ...],
    blocks_by_item: dict[str, list[Any]],
    probes_by_item: dict[str, list[Probe]],
    tokenizer: Any,
    environment: dict[str, Any],
) -> dict[str, Any]:
    candidate_audits: list[bool] = []
    training_audits: list[bool] = []
    external_hashes: dict[tuple[str, str, str], set[str]] = {}
    for item in items:
        blocks = tuple(blocks_by_item[item.item_id])
        for block in blocks:
            for example in block.examples:
                prompt_ids = tokenizer(
                    chat_prompt(tokenizer, example.prompt), add_special_tokens=False
                )["input_ids"]
                completion_ids = tokenizer(example.completion, add_special_tokens=False)[
                    "input_ids"
                ]
                training_audits.append(
                    bool(completion_ids)
                    and len(prompt_ids) + len(completion_ids) + 1
                    <= P0_CONFIG.recipe.training_max_length
                )
        for path_name in FINAL_PATHS:
            external_state, icl_state = _render_states(blocks, path_name)
            terminal_state = path_name[-1]
            for probe in _core_probes(tuple(probes_by_item[item.item_id]), terminal_state):
                rendered = (
                    probe.prompt,
                    render_fresh_session_probe(
                        probe, persistent_state=external_state, access_mode=AccessMode.ON
                    ),
                    render_fresh_session_probe(
                        probe, persistent_state=icl_state, access_mode=AccessMode.ON
                    ),
                )
                for raw_prompt in rendered:
                    audit = audit_candidate_tokenization(
                        tokenizer,
                        chat_prompt(tokenizer, raw_prompt),
                        probe.action_choices,
                        evaluation_max_length=P0_CONFIG.recipe.evaluation_max_length,
                    )
                    candidate_audits.append(bool(audit["all_candidates_valid"]))
                key = (item.item_id, terminal_state, probe.probe_id)
                external_hashes.setdefault(key, set()).add(
                    json_hash(
                        render_fresh_session_probe(
                            probe,
                            persistent_state=external_state,
                            access_mode=AccessMode.ON,
                        )
                    )
                )
    prompts_identical = bool(external_hashes) and all(
        len(hashes) == 1 for hashes in external_hashes.values()
    )
    passed = all(candidate_audits) and all(training_audits) and prompts_identical
    if not passed:
        raise RuntimeError("P0 tokenizer/exposure preflight failed")
    return {
        "phase": "P0",
        "passed": True,
        "training_started": False,
        "gpu_scoring_started": False,
        "candidate_prompt_audit_count": len(candidate_audits),
        "training_example_audit_count": len(training_audits),
        "external_prompts_byte_identical": prompts_identical,
        "exposure_verified": True,
        "environment": environment,
    }


def _score_no_memory(
    *,
    bundle: Any,
    items: tuple[SemanticItem, ...],
    probes_by_item: dict[str, list[Probe]],
    run_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    probe_bank = _probe_bank_hash(probes_by_item)
    for item in items:
        for probe in _core_probes(tuple(probes_by_item[item.item_id]), None):
            attempt_id = f"{item.item_id}::no-memory::{probe.terminal_state}"
            row = score_probe(
                bundle=bundle,
                phase="P0",
                run_id=run_id,
                item_id=item.item_id,
                probe=probe,
                raw_prompt=probe.prompt,
                renderer_kind="no_memory",
                arm="no_memory",
                access_mode=AccessMode.OFF,
                node_id=json_hash({"item": item.item_id, "control": "no_memory"}),
                adapter_sha256=None,
                probe_bank_sha256=probe_bank,
                external_state_sha256=json_hash({"state": "none"}),
                lifecycle_trace={"attempt_id": attempt_id, "probe_id": probe.probe_id},
            )
            rows.append({**row, "attempt_id": attempt_id, "logical_name": "N"})
    return rows


def _score_latest_anchor(
    *,
    bundle: Any,
    item: SemanticItem,
    probes: tuple[Probe, ...],
    logical_name: str,
    attempt_id: str,
    node_id: str,
    adapter_sha256: str,
    run_id: str,
) -> list[dict[str, Any]]:
    state = logical_name[0]
    return [
        {
            **_score_arm(
                bundle=bundle,
                item=item,
                probe=probe,
                raw_prompt=probe.prompt,
                arm="p_latest",
                renderer="parametric_latest",
                access=AccessMode.ON,
                disabled=False,
                attempt_id=attempt_id,
                node_id=node_id,
                adapter_sha256=adapter_sha256,
                run_id=run_id,
                external_state=json_hash({"state": "none"}),
                logical_name=logical_name,
                probe_bank_sha256=json_hash([row.to_dict() for row in probes]),
            ),
        }
        for probe in _core_probes(probes, state)
    ]


def _score_duplicate(
    *,
    bundle: Any,
    item: SemanticItem,
    probes: tuple[Probe, ...],
    attempt_id: str,
    duplicate_of: str,
    node_id: str,
    adapter_sha256: str,
    run_id: str,
) -> list[dict[str, Any]]:
    return [
        {
            **_score_arm(
                bundle=bundle,
                item=item,
                probe=probe,
                raw_prompt=probe.prompt,
                arm="technical_duplicate",
                renderer="parametric_probe",
                access=AccessMode.ON,
                disabled=False,
                attempt_id=attempt_id,
                node_id=node_id,
                adapter_sha256=adapter_sha256,
                run_id=run_id,
                external_state=json_hash({"state": "none"}),
                logical_name=f"DUP-{duplicate_of}",
                probe_bank_sha256=json_hash([row.to_dict() for row in probes]),
            ),
            "duplicate_of": duplicate_of,
        }
        for probe in _core_probes(probes, duplicate_of[-1])
    ]


def _score_final_path(
    *,
    bundle: Any,
    item: SemanticItem,
    probes: tuple[Probe, ...],
    blocks: tuple[Any, ...],
    logical_name: str,
    attempt_id: str,
    node_id: str,
    adapter_sha256: str,
    run_id: str,
) -> list[dict[str, Any]]:
    external_state, icl_state = _render_states(blocks, logical_name)
    probe_bank = json_hash([row.to_dict() for row in probes])
    rows: list[dict[str, Any]] = []
    for probe in _core_probes(probes, logical_name[-1]):
        external_prompt = render_fresh_session_probe(
            probe, persistent_state=external_state, access_mode=AccessMode.ON
        )
        icl_prompt = render_fresh_session_probe(
            probe, persistent_state=icl_state, access_mode=AccessMode.ON
        )
        specs = (
            (
                "base_before",
                probe.prompt,
                "parametric_probe",
                AccessMode.OFF,
                True,
                json_hash({"state": "none"}),
            ),
            (
                "parametric_path",
                probe.prompt,
                "parametric_probe",
                AccessMode.ON,
                False,
                json_hash({"state": "none"}),
            ),
            (
                "parametric_rescore",
                probe.prompt,
                "parametric_probe",
                AccessMode.ON,
                False,
                json_hash({"state": "none"}),
            ),
            (
                "adapter_disabled",
                probe.prompt,
                "parametric_probe",
                AccessMode.OFF,
                True,
                json_hash({"state": "none"}),
            ),
            (
                "external_latest",
                external_prompt,
                "external_latest",
                AccessMode.ON,
                True,
                json_hash(external_state),
            ),
            ("icl_history", icl_prompt, "icl_history", AccessMode.ON, True, json_hash(icl_state)),
            (
                "both",
                external_prompt,
                "external_plus_parametric",
                AccessMode.ON,
                False,
                json_hash(external_state),
            ),
        )
        for arm, raw_prompt, renderer, access, disabled, external_hash in specs:
            rows.append(
                _score_arm(
                    bundle=bundle,
                    item=item,
                    probe=probe,
                    raw_prompt=raw_prompt,
                    arm=arm,
                    renderer=renderer,
                    access=access,
                    disabled=disabled,
                    attempt_id=attempt_id,
                    node_id=node_id,
                    adapter_sha256=(
                        None if arm in {"external_latest", "icl_history"} else adapter_sha256
                    ),
                    run_id=run_id,
                    external_state=external_hash,
                    logical_name=logical_name,
                    probe_bank_sha256=probe_bank,
                )
            )
    return rows


def _score_arm(
    *,
    bundle: Any,
    item: SemanticItem,
    probe: Probe,
    raw_prompt: str,
    arm: str,
    renderer: str,
    access: AccessMode,
    disabled: bool,
    attempt_id: str,
    node_id: str,
    adapter_sha256: str | None,
    run_id: str,
    external_state: str,
    logical_name: str,
    probe_bank_sha256: str,
) -> dict[str, Any]:
    row = score_probe(
        bundle=bundle,
        phase="P0",
        run_id=run_id,
        item_id=item.item_id,
        probe=probe,
        raw_prompt=raw_prompt,
        renderer_kind=renderer,
        arm=arm,
        access_mode=access,
        node_id=node_id,
        adapter_sha256=adapter_sha256,
        probe_bank_sha256=probe_bank_sha256,
        external_state_sha256=external_state,
        lifecycle_trace={
            "attempt_id": attempt_id,
            "logical_name": logical_name,
            "probe_id": probe.probe_id,
            "arm": arm,
            "fresh_session_id": json_hash([attempt_id, logical_name, probe.probe_id, arm]),
        },
        disable_adapter=disabled,
    )
    return {**row, "attempt_id": attempt_id, "logical_name": logical_name}


def _render_states(blocks: tuple[Any, ...], path_name: str) -> tuple[str, str]:
    by_suffix = {block.block_id.rsplit(":", 1)[-1]: block for block in blocks}
    external = VersionedExternalStore()
    icl = ICLHistoryRenderer()
    for suffix in PATH_SUFFIXES[path_name]:
        external.apply(by_suffix[suffix])
        icl.apply(by_suffix[suffix])
    return external.render(), icl.render()


def _core_probes(probes: tuple[Probe, ...], state: str | None) -> tuple[Probe, ...]:
    selected = tuple(
        probe
        for probe in probes
        if probe.core_endpoint and (state is None or probe.terminal_state == state)
    )
    expected = 28 if state is None else 14
    if len(selected) != expected:
        raise RuntimeError(f"core panel changed: {len(selected)} != {expected}")
    return selected


def _verify_lineages(
    *, output_root: Path, planned: dict[str, dict[str, Any]], runtimes: dict[str, Any]
) -> bool:
    adapter_root = output_root / "adapters"
    observed_dirs = {
        path.name
        for path in adapter_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    expected_dirs = {attempt_id.replace("::", "__") for attempt_id in planned}
    if observed_dirs != expected_dirs:
        return False
    for attempt_id, unit in planned.items():
        lineage = _read_lineage(_adapter_dir(output_root, attempt_id))
        runtime = runtimes[str(unit["item_id"])]
        parent_attempt_id = unit["parent_attempt_id"]
        parent_state = (
            runtime.root_state_sha256
            if parent_attempt_id is None
            else str(_read_lineage(_adapter_dir(output_root, str(parent_attempt_id)))["node_id"])
        )
        parent_adapter = (
            output_root / "roots" / str(unit["item_id"])
            if parent_attempt_id is None
            else _adapter_dir(output_root, str(parent_attempt_id))
        )
        summary = lineage.get("training_summary", {})
        identity = lineage.get("node_identity", {})
        checks = (
            lineage.get("attempt_id") == attempt_id,
            lineage.get("adapter_sha256")
            == adapter_bundle_hash(_adapter_dir(output_root, attempt_id)),
            lineage.get("parent_adapter_sha256") == adapter_bundle_hash(parent_adapter),
            lineage.get("parent_state_sha256") == parent_state,
            lineage.get("root_state_sha256") == runtime.root_state_sha256,
            lineage.get("trainer_config_sha256") == runtime.trainer_config_sha256,
            lineage.get("node_id") == json_hash(identity),
            identity.get("tokenizer_sha256") == runtime.tokenizer_sha256,
            identity.get("library_lock_sha256") == runtime.library_lock_sha256,
            identity.get("code_and_patch_sha256") == runtime.code_and_patch_sha256,
            summary.get("optimizer_steps") == P0_CONFIG.recipe.optimizer_steps_per_event,
            summary.get("precision") == "nf4-bfloat16",
            summary.get("model_revision") == P0_CONFIG.model.revision,
        )
        if not all(checks):
            return False
    return True


def _group_by_item(values: tuple[Any, ...], key: Any) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for value in values:
        grouped.setdefault(str(key(value)), []).append(value)
    return grouped


def _probe_bank_hash(probes_by_item: dict[str, list[Probe]]) -> str:
    return json_hash(
        [
            probe.to_dict()
            for item_id in sorted(probes_by_item)
            if item_id.startswith("pmv1-smoke")
            for probe in probes_by_item[item_id]
        ]
    )


def _attempt_id(item_id: str, logical_name: str) -> str:
    return f"{item_id}::seed-{P0_CONFIG.training_seeds[0]}::{logical_name}"


def _adapter_dir(output_root: Path, attempt_id: str) -> Path:
    return output_root / "adapters" / attempt_id.replace("::", "__")


def _read_lineage(adapter_dir: Path) -> dict[str, Any]:
    return json.loads((adapter_dir / "pathmem_lineage.json").read_text(encoding="utf-8"))
