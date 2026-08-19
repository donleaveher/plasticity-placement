from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from plasticity_placement.pathmem.compiler import compile_bank
from plasticity_placement.pathmem.dag import CoreDAGPlan, DAGNodePlan, build_core_dag_plan
from plasticity_placement.pathmem.generator import canonical_external_note
from plasticity_placement.pathmem.io import (
    file_hash,
    immutable_json_write,
    json_hash,
)
from plasticity_placement.pathmem.schema import AccessMode, Probe, ProbeCategory, SemanticItem
from plasticity_placement.pathmem.scoring import audit_candidate_tokenization
from plasticity_placement.pathmem_exec.analysis import summarize_g1
from plasticity_placement.pathmem_exec.artifacts import (
    AttemptState,
    PhaseManifest,
    immutable_jsonl_write,
    read_jsonl,
    write_g1_authorization,
)
from plasticity_placement.pathmem_exec.config import G1_CONFIG, G1_SCHEMA_VERSION, PhaseConfig
from plasticity_placement.pathmem_exec.environment import (
    execution_environment,
    resolve_and_verify_model,
    verify_g0_or_raise,
)
from plasticity_placement.pathmem_exec.scoring import (
    activate_adapter,
    answer_copy_prompt,
    load_model_bundle,
    release_model_bundle,
    score_probe,
)
from plasticity_placement.pathmem_exec.training import (
    create_root_adapter,
    runtime_identity,
    train_event_node,
)
from plasticity_placement.pathmem_exec.training_audit import audit_training_scoring_parity
from plasticity_placement.training.model_utils import chat_prompt


def run_g1(
    *,
    output_root: Path,
    g0_dir: Path,
    protocol_root: Path,
    config: PhaseConfig = G1_CONFIG,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    g0 = verify_g0_or_raise(g0_dir, protocol_root)
    if config.phase != "G1":
        raise ValueError("G1 execution requires a G1 phase config")
    environment = execution_environment(config)
    bank = compile_bank()
    items = tuple(item for item in bank.items if item.split == config.split)
    if len(items) != 12:
        raise RuntimeError(f"G1 split changed: {len(items)} items")
    blocks_by_item = _group_by_item(
        bank.event_blocks, lambda block: block.block_id.split(":", 1)[0]
    )
    probes_by_item = _group_by_item(bank.probes, lambda probe: probe.item_id)
    families_by_item = _group_by_item(bank.history_families, lambda family: family.item_id)

    tokenizer, _ = resolve_and_verify_model(config)
    preflight = _preflight(
        items,
        blocks_by_item,
        probes_by_item,
        tokenizer,
        environment,
        config,
    )
    immutable_json_write(output_root / "preflight.json", preflight, "G1 preflight")

    runtimes: dict[str, Any] = {}
    dags: dict[str, CoreDAGPlan] = {}
    roots: dict[str, Path] = {}
    planned: dict[str, dict[str, Any]] = {}
    plans: dict[str, DAGNodePlan] = {}
    for item in items:
        root_dir = output_root / "roots" / item.item_id
        root = create_root_adapter(config=config, item_id=item.item_id, output_dir=root_dir)
        runtime = runtime_identity(
            config=config,
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
        for logical_name in ("A2", "B2"):
            plan = next(node for node in dag.nodes if node.logical_name == logical_name)
            attempt_id = _attempt_id(item.item_id, logical_name, config)
            planned[attempt_id] = {
                "item_id": item.item_id,
                "terminal_state": logical_name[0],
                "logical_name": logical_name,
                "plan_id": plan.plan_id,
                "parent_logical_name": "ROOT",
            }
            plans[attempt_id] = plan

    identity = {
        "schema_version": G1_SCHEMA_VERSION,
        "phase_config": config.to_dict(),
        "environment_fingerprint": environment["environment_fingerprint"],
        "preflight_sha256": file_hash(output_root / "preflight.json"),
    }
    manifest = PhaseManifest.load_or_create(
        output_root / "manifest.json",
        phase="G1",
        g0_manifest_id=str(g0["manifest_id"]),
        identity=identity,
        planned_units=planned,
    )

    for attempt_id, unit in sorted(planned.items()):
        if manifest.state(attempt_id) in {
            AttemptState.TRAINED,
            AttemptState.SCORING,
            AttemptState.VERIFIED,
        }:
            continue
        item_id = str(unit["item_id"])
        runtime = runtimes[item_id]
        plan = plans[attempt_id]
        block = next(block for block in blocks_by_item[item_id] if block.block_id == plan.block_id)
        manifest.mark(attempt_id, AttemptState.TRAINING)
        try:
            lineage = train_event_node(
                config=config,
                attempt_id=attempt_id,
                block=block,
                plan=plan,
                dag=dags[item_id],
                runtime=runtime,
                parent_adapter_dir=roots[item_id],
                parent_state_sha256=runtime.root_state_sha256,
                root_state_sha256=runtime.root_state_sha256,
                data_path=output_root / "data" / f"{block.block_id.replace(':', '__')}.jsonl",
                output_dir=_adapter_dir(output_root, attempt_id),
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

    verified_lineages: dict[str, dict[str, Any]] = {}
    for attempt_id, unit in sorted(planned.items()):
        item_id = str(unit["item_id"])
        runtime = runtimes[item_id]
        plan = plans[attempt_id]
        block = next(block for block in blocks_by_item[item_id] if block.block_id == plan.block_id)
        lineage = train_event_node(
            config=config,
            attempt_id=attempt_id,
            block=block,
            plan=plan,
            dag=dags[item_id],
            runtime=runtime,
            parent_adapter_dir=roots[item_id],
            parent_state_sha256=runtime.root_state_sha256,
            root_state_sha256=runtime.root_state_sha256,
            data_path=output_root / "data" / f"{block.block_id.replace(':', '__')}.jsonl",
            output_dir=_adapter_dir(output_root, attempt_id),
        )
        manifest_metadata = manifest.payload["units"][attempt_id].get("metadata", {})
        if (
            manifest_metadata.get("node_id") != lineage["node_id"]
            or manifest_metadata.get("adapter_sha256") != lineage["adapter_sha256"]
        ):
            raise ValueError(f"G1 manifest/lineage mismatch: {attempt_id}")
        verified_lineages[attempt_id] = lineage

    controls_path = output_root / "results" / "base_controls.jsonl"
    bundle = load_model_bundle(config)
    try:
        existing_controls_sha256 = manifest.payload.get("control_results_sha256")
        if controls_path.exists() and existing_controls_sha256 is not None:
            if file_hash(controls_path) != existing_controls_sha256:
                raise ValueError("G1 base-control result hash changed")
            control_rows = read_jsonl(controls_path)
        else:
            control_rows = _score_controls(
                bundle=bundle,
                items=items,
                probes_by_item=probes_by_item,
                run_id=str(manifest.payload["run_id"]),
            )
            immutable_jsonl_write(controls_path, control_rows, "G1 base controls")
        controls_sha256 = file_hash(controls_path)
        if existing_controls_sha256 not in {None, controls_sha256}:
            raise ValueError("G1 base-control result hash changed")
        if existing_controls_sha256 is None:
            manifest.payload["control_results_sha256"] = controls_sha256
            manifest.save()

        all_rows = list(control_rows)
        for attempt_id, unit in sorted(planned.items()):
            result_path = output_root / "results" / f"{attempt_id.replace('::', '__')}.jsonl"
            if manifest.state(attempt_id) is AttemptState.VERIFIED:
                expected_result_sha256 = manifest.payload["units"][attempt_id]["metadata"].get(
                    "result_sha256"
                )
                if file_hash(result_path) != expected_result_sha256:
                    raise ValueError(f"G1 verified result hash changed: {attempt_id}")
                all_rows.extend(read_jsonl(result_path))
                continue
            if manifest.state(attempt_id) is AttemptState.TRAINED:
                manifest.mark(attempt_id, AttemptState.SCORING)
            item_id = str(unit["item_id"])
            terminal_state = str(unit["terminal_state"])
            lineage = verified_lineages[attempt_id]
            activate_adapter(bundle, _adapter_dir(output_root, attempt_id), attempt_id)
            try:
                rows = _score_anchor(
                    bundle=bundle,
                    item=next(item for item in items if item.item_id == item_id),
                    probes=tuple(probes_by_item[item_id]),
                    terminal_state=terminal_state,
                    attempt_id=attempt_id,
                    node_id=str(lineage["node_id"]),
                    adapter_sha256=str(lineage["adapter_sha256"]),
                    run_id=str(manifest.payload["run_id"]),
                )
                result_sha256 = immutable_jsonl_write(result_path, rows, "G1 anchor results")
                manifest.mark(
                    attempt_id,
                    AttemptState.VERIFIED,
                    result_sha256=result_sha256,
                )
                all_rows.extend(rows)
            except BaseException as error:
                manifest.record_error(attempt_id, error)
                manifest.mark(attempt_id, AttemptState.FAILED)
                raise
    finally:
        release_model_bundle(bundle)

    manifest.require_all_verified()
    summary = summarize_g1(
        all_rows,
        verified_attempts=len(planned),
        additional_integrity={
            "training_scoring_token_parity": bool(preflight["training_scoring_token_parity"]),
            "event_optimizer_step_exposure_exact": bool(
                preflight["event_optimizer_step_exposure_matches"]
            ),
            "artifact_lineage_hashes_verified": True,
        },
    )
    summary.update(
        {
            "schema_version": G1_SCHEMA_VERSION,
            "run_id": manifest.payload["run_id"],
            "g0_manifest_id": g0["manifest_id"],
            "environment_fingerprint": environment["environment_fingerprint"],
            "recipe_id": config.recipe.recipe_id,
            "recipe_sha256": json_hash(config.recipe.to_dict()),
        }
    )
    summary_path = output_root / "summary.json"
    immutable_json_write(summary_path, summary, "G1 summary")
    if summary["authorization_ready"]:
        authorization = write_g1_authorization(
            output_root / "g1_authorization.json",
            g0_manifest_id=str(g0["manifest_id"]),
            run_id=str(manifest.payload["run_id"]),
            summary_path=summary_path,
            gate=summary["gate"],
            integrity_checks=summary["integrity_checks"],
            qualified_recipe_id=config.recipe.recipe_id,
            qualified_recipe_sha256=json_hash(config.recipe.to_dict()),
        )
        summary = {**summary, "authorization": authorization}
    return summary


def _preflight(
    items: tuple[SemanticItem, ...],
    blocks_by_item: dict[str, list[Any]],
    probes_by_item: dict[str, list[Probe]],
    tokenizer: Any,
    environment: dict[str, Any],
    config: PhaseConfig,
) -> dict[str, Any]:
    prompt_audits: list[dict[str, Any]] = []
    training_audits: list[dict[str, Any]] = []
    for item in items:
        for block in blocks_by_item[item.item_id]:
            if block.state_label not in {"A", "B"} or not block.block_id.endswith("2"):
                continue
            for example in block.examples:
                formatted = chat_prompt(tokenizer, example.prompt)
                audit = audit_training_scoring_parity(
                    tokenizer=tokenizer,
                    formatted_prompt=formatted,
                    completion=example.completion,
                    action_choices=item.action_choices,
                    training_max_length=config.recipe.training_max_length,
                    evaluation_max_length=config.recipe.evaluation_max_length,
                )
                training_audits.append(
                    {
                        "example_id": example.example_id,
                        "passed": audit["passed"],
                        "checks": audit["checks"],
                        "completion_token_ids": audit["completion_token_ids"],
                    }
                )
        for probe in _qualification_probes(tuple(probes_by_item[item.item_id]), None):
            current = _current_action(item, probe.terminal_state)
            obsolete = _obsolete_action(item, probe.terminal_state)
            qualified_probe = replace(probe, obsolete_action=obsolete)
            note = canonical_external_note(item, probe.terminal_state)
            prompts = (
                ("answer_copy", answer_copy_prompt(current)),
                ("external_latest", f"{note}\n\n{probe.prompt}"),
                ("parametric", probe.prompt),
            )
            for kind, raw_prompt in prompts:
                audit = audit_candidate_tokenization(
                    tokenizer,
                    chat_prompt(tokenizer, raw_prompt),
                    qualified_probe.action_choices,
                    evaluation_max_length=config.recipe.evaluation_max_length,
                )
                prompt_audits.append(
                    {
                        "probe_id": probe.probe_id,
                        "kind": kind,
                        "all_candidates_valid": audit["all_candidates_valid"],
                        "equal_candidate_token_lengths": (
                            len(
                                {
                                    int(candidate["token_count"])
                                    for candidate in audit["candidates"]
                                }
                            )
                            == 1
                        ),
                    }
                )
    passed = all(
        row["all_candidates_valid"] and row["equal_candidate_token_lengths"]
        for row in prompt_audits
    ) and all(row["passed"] for row in training_audits)
    step_exposure_matches = all(
        block.optimizer_steps == config.recipe.optimizer_steps_per_event
        for item in items
        for block in blocks_by_item[item.item_id]
        if block.state_label in {"A", "B"} and block.block_id.endswith("2")
    )
    passed = passed and step_exposure_matches
    if not passed:
        raise RuntimeError("G1 tokenizer/truncation preflight failed")
    return {
        "phase": "G1",
        "passed": passed,
        "training_started": False,
        "gpu_scoring_started": False,
        "prompt_audit_count": len(prompt_audits),
        "training_example_audit_count": len(training_audits),
        "training_scoring_token_parity": all(row["passed"] for row in training_audits),
        "equal_candidate_token_lengths": all(
            row["equal_candidate_token_lengths"] for row in prompt_audits
        ),
        "event_optimizer_step_exposure_matches": step_exposure_matches,
        "recipe_id": config.recipe.recipe_id,
        "environment": environment,
    }


def _score_controls(
    *,
    bundle: Any,
    items: tuple[SemanticItem, ...],
    probes_by_item: dict[str, list[Probe]],
    run_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    probe_bank_sha256 = _probe_bank_hash(probes_by_item)
    for item in items:
        for probe in _qualification_probes(tuple(probes_by_item[item.item_id]), None):
            current = _current_action(item, probe.terminal_state)
            obsolete = _obsolete_action(item, probe.terminal_state)
            probe = replace(probe, obsolete_action=obsolete)
            control_id = f"{item.item_id}::{probe.terminal_state}::{probe.probe_id}"
            node_id = json_hash({"phase": "G1", "control": "base", "item": item.item_id})
            common = {
                "bundle": bundle,
                "phase": "G1",
                "run_id": run_id,
                "item_id": item.item_id,
                "probe": probe,
                "access_mode": AccessMode.ON,
                "node_id": node_id,
                "adapter_sha256": None,
                "probe_bank_sha256": probe_bank_sha256,
            }
            answer = score_probe(
                **common,
                raw_prompt=answer_copy_prompt(current),
                renderer_kind="answer_copy",
                arm="answer_copy",
                external_state_sha256=json_hash({"state": "none"}),
                lifecycle_trace={"arm": "answer_copy", "probe_id": probe.probe_id},
            )
            note = canonical_external_note(item, probe.terminal_state)
            external = score_probe(
                **common,
                raw_prompt=f"{note}\n\n{probe.prompt}",
                renderer_kind="canonical_external_note",
                arm="external_latest",
                external_state_sha256=json_hash(note),
                lifecycle_trace={"arm": "external_latest", "probe_id": probe.probe_id},
            )
            rows.extend(
                ({**answer, "attempt_id": control_id}, {**external, "attempt_id": control_id})
            )
    return rows


def _score_anchor(
    *,
    bundle: Any,
    item: SemanticItem,
    probes: tuple[Probe, ...],
    terminal_state: str,
    attempt_id: str,
    node_id: str,
    adapter_sha256: str,
    run_id: str,
) -> list[dict[str, Any]]:
    selected = [
        probe
        for probe in probes
        if probe.terminal_state == terminal_state
        and probe.category in {ProbeCategory.QUALIFICATION, ProbeCategory.UNRELATED}
    ]
    probe_bank_sha256 = json_hash(
        [probe.to_dict() for probe in sorted(probes, key=lambda row: row.probe_id)]
    )
    rows: list[dict[str, Any]] = []
    for probe in selected:
        common = {
            "bundle": bundle,
            "phase": "G1",
            "run_id": run_id,
            "item_id": item.item_id,
            "probe": probe,
            "raw_prompt": probe.prompt,
            "renderer_kind": "parametric_probe",
            "node_id": node_id,
            "adapter_sha256": adapter_sha256,
            "probe_bank_sha256": probe_bank_sha256,
            "external_state_sha256": json_hash({"state": "none"}),
        }
        is_qualification = probe.category is ProbeCategory.QUALIFICATION
        arms = (
            (
                ("bare_base", AccessMode.OFF, True),
                ("parametric_current", AccessMode.ON, False),
                ("parametric_rescore", AccessMode.ON, False),
                ("adapter_disabled", AccessMode.OFF, True),
            )
            if is_qualification
            else (
                ("base_unrelated", AccessMode.OFF, True),
                ("parametric_unrelated", AccessMode.ON, False),
            )
        )
        for arm, access_mode, disabled in arms:
            row = score_probe(
                **common,
                arm=arm,
                access_mode=access_mode,
                disable_adapter=disabled,
                lifecycle_trace={
                    "attempt_id": attempt_id,
                    "probe_id": probe.probe_id,
                    "arm": arm,
                    "fresh_session_id": json_hash([attempt_id, probe.probe_id, arm]),
                },
            )
            rows.append({**row, "attempt_id": attempt_id})
    return rows


def _qualification_probes(probes: tuple[Probe, ...], state: str | None) -> tuple[Probe, ...]:
    return tuple(
        probe
        for probe in probes
        if probe.category is ProbeCategory.QUALIFICATION
        and (state is None or probe.terminal_state == state)
    )


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
            for probe in sorted(probes_by_item[item_id], key=lambda row: row.probe_id)
            if item_id.startswith("pmv1-interface_dev")
        ]
    )


def _current_action(item: SemanticItem, state: str) -> str:
    return item.state_a_action if state == "A" else item.state_b_action


def _obsolete_action(item: SemanticItem, state: str) -> str:
    return item.state_b_action if state == "A" else item.state_a_action


def _attempt_id(item_id: str, logical_name: str, config: PhaseConfig) -> str:
    return f"{item_id}::seed-{config.training_seeds[0]}::{logical_name}"


def _adapter_dir(output_root: Path, attempt_id: str) -> Path:
    return output_root / "adapters" / attempt_id.replace("::", "__")
