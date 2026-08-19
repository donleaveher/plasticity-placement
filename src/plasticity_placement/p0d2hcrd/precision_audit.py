from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any

from plasticity_placement.p0c.modeling import ModelBundle, release_model
from plasticity_placement.p0c.runtime import (
    current_code_hash,
    current_environment_snapshot,
)
from plasticity_placement.p0d2hcrd.config import (
    EXPECTED_MODEL_NAME,
    EXPECTED_MODEL_REVISION,
    P0D2HCRDRequest,
)
from plasticity_placement.p0d2hcrd.integrity_audit import (
    build_scoring_integrity_audit,
    load_verified_rows,
    raw_tree_hash,
)
from plasticity_placement.p0d2hcrd.runtime import resolve_request
from plasticity_placement.p0d2hcrd.scoring import (
    audit_candidate_tokenization,
    score_candidate_batch,
)
from plasticity_placement.training.model_utils import (
    chat_prompt,
    load_tokenizer,
    set_seed,
)

FP32_AUDIT_VERSION = "p0d2hcrd-fp32-anomaly-audit-v1"


def audit_fp32_precision(
    source_output_dirs: tuple[Path, ...],
    audit_output_dir: Path,
) -> Path:
    """Rescore structurally valid exact ties with a strict FP32 CUDA forward."""
    sources = _load_sources(source_output_dirs, audit_output_dir)
    config, bank, _, _, _ = resolve_request(
        P0D2HCRDRequest(
            output_dir=sources[0]["path"],
            source_manifest=Path(str(sources[0]["manifest"]["source_manifest_path"])),
            use_4bit=bool(sources[0]["manifest"]["config"]["model"]["use_4bit"]),
        )
    )
    _verify_common_experiment(sources, config.decomposition_bank_sha256)
    probe_index = {probe.probe_id: probe for probes in bank.values() for probe in probes}

    bundle = _load_strict_fp32_model()
    try:
        records = _rescore_ties(
            sources,
            probe_index,
            bundle,
            evaluation_max_length=config.evaluation_max_length,
        )
        environment = current_environment_snapshot()
    finally:
        release_model(bundle)

    _verify_sources_unchanged(sources)
    source_identities = [source["identity"] for source in sources]
    identity = {
        "schema_version": FP32_AUDIT_VERSION,
        "analysis_code_sha256": current_code_hash(),
        "sources": source_identities,
        "model_name": EXPECTED_MODEL_NAME,
        "model_revision": EXPECTED_MODEL_REVISION,
        "evaluation_precision": "float32",
        "tf32_allowed": False,
        "autocast_used": False,
        "candidate_scoring": "candidate_token_sum_logprob",
    }
    run_id = "p0d2hcrd-fp32-audit-" + _json_hash(identity)[:10]
    resolved_count = sum(record["fp32"]["error_status"] == "ok" for record in records)
    persistent_count = sum(record["fp32"]["error_status"] == "tie" for record in records)
    error_count = len(records) - resolved_count - persistent_count
    classification = (
        "all_exact_ties_resolved_in_strict_fp32"
        if resolved_count == len(records)
        else "exact_ties_persist_in_strict_fp32"
        if persistent_count and not error_count
        else "strict_fp32_rescore_error_observed"
    )
    payload = {
        **identity,
        "run_id": run_id,
        "analysis_status": "post_hoc_mechanistic_audit",
        "classification": classification,
        "source_gate_status_changed": False,
        "next_stage_eligibility_changed": False,
        "source_tie_record_count": len(records),
        "unique_probe_count": len({str(record["probe_id"]) for record in records}),
        "resolved_tie_count": resolved_count,
        "persistent_tie_count": persistent_count,
        "rescore_error_count": error_count,
        "strict_fp32_checks": {
            "parameter_dtype": "float32",
            "logits_dtype": "float32",
            "cuda_required": True,
            "tf32_allowed": False,
            "autocast_used": False,
        },
        "environment": environment,
        "training_complexity_review_eligible": False,
        "automatic_training_started": False,
        "automatic_narrow_scan_started": False,
        "artifact_paths": {
            "summary_json": str(audit_output_dir / "fp32_precision_audit.json"),
            "report_markdown": str(audit_output_dir / "fp32_precision_audit.md"),
            "records_jsonl": str(audit_output_dir / "fp32_precision_records.jsonl"),
        },
    }
    audit_output_dir.mkdir(parents=True, exist_ok=True)
    result_path = audit_output_dir / "fp32_precision_audit.json"
    _write_immutable(
        result_path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    _write_immutable(
        audit_output_dir / "fp32_precision_records.jsonl",
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        ).encode(),
    )
    _write_immutable(
        audit_output_dir / "fp32_precision_audit.md",
        _render_markdown(payload, records).encode(),
    )
    return result_path


def _load_sources(
    source_output_dirs: tuple[Path, ...],
    audit_output_dir: Path,
) -> list[dict[str, Any]]:
    if not source_output_dirs:
        raise ValueError("FP32 audit requires at least one completed CRD source")
    resolved = [path.resolve() for path in source_output_dirs]
    if len(resolved) != len(set(resolved)):
        raise ValueError("FP32 audit source outputs must be unique")
    sources: list[dict[str, Any]] = []
    for source_output_dir in source_output_dirs:
        _require_independent_output(source_output_dir, audit_output_dir)
        manifest_path = source_output_dir / "manifest.json"
        summary_path = source_output_dir / "results" / "aggregate" / "summary.json"
        manifest_bytes = manifest_path.read_bytes()
        summary_bytes = summary_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        summary = json.loads(summary_bytes)
        rows, raw_sha256 = load_verified_rows(source_output_dir, manifest)
        report, anomaly_records = build_scoring_integrity_audit(rows)
        if (
            report["classification"] != "distinct_candidate_exact_score_ties_observed"
            or not report["structural_checks_passed"]
            or not anomaly_records
            or summary.get("run_id") != manifest.get("run_id")
            or summary.get("decision_row_count") != len(rows)
        ):
            raise ValueError(
                "FP32 audit requires a structurally valid completed CRD "
                "source containing only finite exact-score ties"
            )
        row_index = {str(row["probe_id"]): row for row in rows}
        precisions = sorted(
            {str(unit.get("evaluation_precision")) for unit in manifest["units"].values()}
        )
        identity = {
            "source_output_dir": str(source_output_dir),
            "source_run_id": manifest["run_id"],
            "source_manifest_sha256": sha256(manifest_bytes).hexdigest(),
            "source_summary_sha256": sha256(summary_bytes).hexdigest(),
            "source_raw_results_sha256": raw_sha256,
            "source_experiment_code_sha256": manifest["config"]["code_sha256"],
            "source_diagnostic_status": summary["diagnostic_gate"]["status"],
            "source_use_4bit": manifest["config"]["model"]["use_4bit"],
            "source_evaluation_precisions": precisions,
            "source_tie_count": len(anomaly_records),
        }
        sources.append(
            {
                "path": source_output_dir,
                "manifest_path": manifest_path,
                "summary_path": summary_path,
                "manifest_bytes": manifest_bytes,
                "summary_bytes": summary_bytes,
                "manifest": manifest,
                "rows": row_index,
                "raw_sha256": raw_sha256,
                "ties": anomaly_records,
                "identity": identity,
            }
        )
    return sources


def _verify_common_experiment(
    sources: list[dict[str, Any]],
    reconstructed_bank_sha256: str,
) -> None:
    first = sources[0]["manifest"]["config"]
    stable_fields = (
        "source_manifest_sha256",
        "source_run_id",
        "source_summary_sha256",
        "source_candidate_token_audit_sha256",
        "source_raw_results_sha256",
        "selected_lesson_ids",
        "decomposition_bank_sha256",
        "prompt_renderers",
        "evaluation_max_length",
        "candidate_scoring_version",
        "primary_score_definition",
    )
    for source in sources:
        manifest = source["manifest"]
        config = manifest["config"]
        if any(config.get(field) != first.get(field) for field in stable_fields):
            raise ValueError("FP32 audit sources do not share one frozen CRD bank")
        if (
            config.get("decomposition_bank_sha256") != reconstructed_bank_sha256
            or manifest.get("model", {}).get("model_name") != EXPECTED_MODEL_NAME
            or manifest.get("model", {}).get("model_revision") != EXPECTED_MODEL_REVISION
        ):
            raise ValueError("FP32 audit model or reconstructed bank changed")


def _load_strict_fp32_model() -> ModelBundle:
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise RuntimeError(
            "FP32 audit dependencies are missing; run `uv sync --extra train --extra colab`"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("strict FP32 anomaly audit requires a CUDA GPU")
    set_seed(0, torch)
    tokenizer = load_tokenizer(EXPECTED_MODEL_NAME, EXPECTED_MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        EXPECTED_MODEL_NAME,
        revision=EXPECTED_MODEL_REVISION,
        torch_dtype=torch.float32,
    )
    model.to(torch.device("cuda"))
    model.eval()
    bundle = ModelBundle(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        model_revision=EXPECTED_MODEL_REVISION,
        precision="float32",
    )
    parameter_dtypes = {parameter.dtype for parameter in model.parameters()}
    if parameter_dtypes != {torch.float32}:
        release_model(bundle)
        raise RuntimeError(f"strict FP32 model has unexpected parameter dtypes: {parameter_dtypes}")
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision is not None and str(resolved_revision) != EXPECTED_MODEL_REVISION:
        release_model(bundle)
        raise RuntimeError("strict FP32 model revision differs from the frozen revision")
    bundle.model_revision = (
        str(resolved_revision) if resolved_revision is not None else EXPECTED_MODEL_REVISION
    )
    return bundle


def _rescore_ties(
    sources: list[dict[str, Any]],
    probe_index: dict[str, Any],
    bundle: ModelBundle,
    *,
    evaluation_max_length: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    with _strict_fp32_execution(bundle.torch):
        for source in sources:
            for tie in source["ties"]:
                probe_id = str(tie["probe_id"])
                try:
                    probe = probe_index[probe_id]
                    row = source["rows"][probe_id]
                except KeyError as error:
                    raise ValueError(f"cannot reconstruct tied CRD probe: {probe_id}") from error
                formatted_prompt = chat_prompt(bundle.tokenizer, probe.prompt)
                reconstruction = audit_candidate_tokenization(
                    bundle.tokenizer,
                    formatted_prompt,
                    tuple(probe.ordered_candidates),
                    evaluation_max_length=evaluation_max_length,
                )
                _verify_reconstruction(row, probe, reconstruction)
                cache_key = (probe_id, str(row["prompt_sha256"]))
                if cache_key not in cache:
                    cache[cache_key] = score_candidate_batch(
                        bundle,
                        formatted_prompt,
                        reconstruction,
                        expected_logits_dtype=bundle.torch.float32,
                    )
                outcome = cache[cache_key]
                records.append(_comparison_record(source, row, tie, outcome))
    return records


@contextmanager
def _strict_fp32_execution(torch: Any) -> Iterator[None]:
    matmul_backend = torch.backends.cuda.matmul
    cudnn_backend = torch.backends.cudnn
    previous_matmul_tf32 = bool(matmul_backend.allow_tf32)
    previous_cudnn_tf32 = bool(cudnn_backend.allow_tf32)
    get_precision = getattr(torch, "get_float32_matmul_precision", None)
    previous_precision = get_precision() if callable(get_precision) else None
    matmul_backend.allow_tf32 = False
    cudnn_backend.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        matmul_backend.allow_tf32 = previous_matmul_tf32
        cudnn_backend.allow_tf32 = previous_cudnn_tf32
        if previous_precision is not None and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(previous_precision)


def _verify_reconstruction(
    row: dict[str, Any],
    probe: Any,
    reconstruction: dict[str, Any],
) -> None:
    persisted_tokens = {
        str(candidate["candidate"]): [int(value) for value in candidate["token_ids"]]
        for candidate in row["candidates"]
    }
    reconstructed_tokens = {
        str(candidate["candidate"]): [int(value) for value in candidate["token_ids"]]
        for candidate in reconstruction["candidates"]
    }
    if (
        reconstruction.get("all_candidates_valid") is not True
        or reconstruction["prompt_sha256"] != row["prompt_sha256"]
        or list(probe.ordered_candidates) != row["ordered_candidates"]
        or reconstructed_tokens != persisted_tokens
        or probe.expected_candidate != row["expected_candidate"]
    ):
        raise ValueError(f"FP32 audit prompt/candidate reconstruction changed: {row['probe_id']}")


def _comparison_record(
    source: dict[str, Any],
    row: dict[str, Any],
    tie: dict[str, Any],
    outcome: dict[str, Any],
) -> dict[str, Any]:
    original_scores = {
        str(candidate["candidate"]): float(candidate["sum_logprob"])
        for candidate in row["candidates"]
    }
    fp32_scores = {
        str(candidate["candidate"]): float(candidate["sum_logprob"])
        for candidate in outcome["candidates"]
    }
    expected = str(row["expected_candidate"])
    return {
        "schema_version": "p0d2hcrd-fp32-comparison-record-v1",
        "source_run_id": source["manifest"]["run_id"],
        "source_precision": row["precision"],
        "probe_id": row["probe_id"],
        "lesson_id": row["lesson_id"],
        "endpoint": row["endpoint"],
        "lesson_type": row["lesson_type"],
        "route_variant": row["route_variant"],
        "target_slot": row["target_slot"],
        "slot_content_order": row["slot_content_order"],
        "action_panel_position": row["action_panel_position"],
        "expected_candidate": expected,
        "source_tied_candidates": [
            candidate["candidate"] for candidate in tie["top_sum_candidates"]
        ],
        "expected_in_source_top_set": expected
        in {str(candidate["candidate"]) for candidate in tie["top_sum_candidates"]},
        "candidate_scores": [
            {
                "candidate": candidate,
                "source_sum_logprob": original_scores[candidate],
                "fp32_sum_logprob": fp32_scores[candidate],
                "fp32_minus_source": (fp32_scores[candidate] - original_scores[candidate]),
            }
            for candidate in row["ordered_candidates"]
        ],
        "fp32": {
            "precision": "float32",
            "predicted_candidate": outcome["predicted_candidate"],
            "mean_predicted_candidate": outcome["mean_predicted_candidate"],
            "top1_top2_margin": outcome["top1_top2_margin"],
            "tie": outcome["tie"],
            "mean_score_tie": outcome["mean_score_tie"],
            "non_finite": outcome["non_finite"],
            "error_status": outcome["error_status"],
            "correct": outcome["predicted_candidate"] == expected,
            "expected_in_top_set": (
                outcome["predicted_candidate"] == expected
                if outcome["error_status"] == "ok"
                else expected in _top_candidate_names(outcome["candidates"])
            ),
        },
    }


def _top_candidate_names(candidates: list[dict[str, Any]]) -> set[str]:
    finite = [
        candidate
        for candidate in candidates
        if isinstance(candidate.get("sum_logprob"), (int, float))
    ]
    if not finite:
        return set()
    maximum = max(float(candidate["sum_logprob"]) for candidate in finite)
    return {
        str(candidate["candidate"])
        for candidate in finite
        if float(candidate["sum_logprob"]) == maximum
    }


def _verify_sources_unchanged(sources: list[dict[str, Any]]) -> None:
    for source in sources:
        if (
            source["manifest_path"].read_bytes() != source["manifest_bytes"]
            or source["summary_path"].read_bytes() != source["summary_bytes"]
            or raw_tree_hash(source["path"]) != source["raw_sha256"]
        ):
            raise RuntimeError("source CRD artifacts changed during FP32 audit")


def _require_independent_output(source: Path, output: Path) -> None:
    source_resolved = source.resolve()
    output_resolved = output.resolve()
    if (
        source_resolved == output_resolved
        or output_resolved.is_relative_to(source_resolved)
        or source_resolved.is_relative_to(output_resolved)
    ):
        raise ValueError("FP32 audit output must be independent from every source CRD run")


def _render_markdown(
    payload: dict[str, Any],
    records: list[dict[str, Any]],
) -> str:
    lines = [
        "# P0-D2H-CRD strict-FP32 anomaly audit",
        "",
        f"- Classification: `{payload['classification']}`",
        f"- Source ties rescored: {payload['source_tie_record_count']}",
        f"- Resolved in FP32: {payload['resolved_tie_count']}",
        f"- Persisting in FP32: {payload['persistent_tie_count']}",
        "- TF32 allowed: `false`",
        "- Autocast used: `false`",
        "- Source gate status changed: `false`",
        "- Next-stage eligibility changed: `false`",
        "",
        "| Source precision | Probe | Source top set | FP32 prediction | "
        "FP32 margin | FP32 status |",
        "|---|---|---|---|---:|---|",
    ]
    for record in records:
        fp32 = record["fp32"]
        lines.append(
            f"| `{record['source_precision']}` | `{record['probe_id']}` | "
            f"`{' / '.join(record['source_tied_candidates'])}` | "
            f"`{fp32['predicted_candidate']}` | "
            f"{float(fp32['top1_top2_margin'] or 0.0):.8f} | "
            f"`{fp32['error_status']}` |"
        )
    lines.extend(
        [
            "",
            "This is a post-hoc mechanistic precision audit. It does not "
            "modify either source run, its frozen zero-tie gate, or training "
            "eligibility.",
            "",
        ]
    )
    return "\n".join(lines)


def _json_hash(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing FP32 audit artifact changed: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
