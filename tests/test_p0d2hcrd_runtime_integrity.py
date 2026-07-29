from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path

import pytest

from plasticity_placement.p0c.compiler import compile_bank
from plasticity_placement.p0c.modeling import write_probe_rows
from plasticity_placement.p0d2h.probes import compile_hard_probe_bank
from plasticity_placement.p0d2hcrd.analysis import aggregate_experiment
from plasticity_placement.p0d2hcrd.config import (
    CRDModel,
    ResolvedP0D2HCRDConfig,
)
from plasticity_placement.p0d2hcrd.manifest import P0D2HCRDManifest
from plasticity_placement.p0d2hcrd.probes import (
    PROMPT_RENDERERS,
    compile_decomposition_bank,
)
from plasticity_placement.p0d2hcrd.runtime import (
    _decision_row,
    _file_hash,
    _load_bank_audit,
    _load_candidate_token_audit,
    _prepare_candidate_token_audit,
    _verify_rows,
    _write_bank_audit,
)
from plasticity_placement.p0d2hcrd.scoring import rank_candidate_scores

SELECTED_IDS = (
    "F_pair01_a",
    "F_pair01_b",
    "F_pair04_a",
    "F_pair04_b",
    "F_pair06_a",
    "F_pair06_b",
    "RF_pair01_a",
    "RF_pair01_b",
    "RF_pair04_a",
    "RF_pair04_b",
    "RF_pair05_a",
    "RF_pair05_b",
    "P_pair01_a",
    "P_pair01_b",
    "P_pair02_a",
    "P_pair02_b",
    "P_pair03_a",
    "P_pair03_b",
    "P_pair04_a",
    "P_pair04_b",
    "P_pair05_a",
    "P_pair05_b",
    "P_pair06_a",
    "P_pair06_b",
)


def selected_lessons():
    by_id = {item.lesson.lesson_id: item for item in compile_bank()}
    return tuple(by_id[lesson_id] for lesson_id in SELECTED_IDS)


class PieceTokenizer:
    chat_template = "test-chat-template"
    pad_token_id = 0

    def __init__(self) -> None:
        self.piece_to_id: dict[str, int] = {}
        self.id_to_piece: dict[int, str] = {}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<user>{messages[0]['content']}</user><assistant>\n"

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        pieces = re.findall(r"\s+|[A-Za-z0-9_<>-]+|[^\w\s]", text)
        token_ids: list[int] = []
        for piece in pieces:
            if piece not in self.piece_to_id:
                token_id = len(self.piece_to_id) + 1
                self.piece_to_id[piece] = token_id
                self.id_to_piece[token_id] = piece
            token_ids.append(self.piece_to_id[piece])
        return {"input_ids": token_ids}

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        assert skip_special_tokens is False
        return "".join(self.id_to_piece[token_id] for token_id in token_ids)


def _json_hash(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _config(tmp_path: Path, bank_sha256: str) -> ResolvedP0D2HCRDConfig:
    return ResolvedP0D2HCRDConfig(
        output_dir=tmp_path / "out",
        source_output_dir=tmp_path / "source",
        source_calibration_output_dir=tmp_path / "calibration",
        source_hard_probe_output_dir=tmp_path / "hard",
        source_compiled_output_dir=tmp_path / "compiled",
        code_sha256="code",
        source_manifest_sha256="source-manifest",
        source_run_id="source-run",
        source_summary_sha256="source-summary",
        source_candidate_token_audit_sha256="source-audit",
        source_raw_results_sha256="source-raw",
        source_calibration_manifest_sha256="calibration-manifest",
        source_calibration_raw_results_sha256="calibration-raw",
        source_hard_probe_manifest_sha256="hard-manifest",
        hard_probe_hashes={"hard_probes_sha256": "hard-probes"},
        compiler_hashes={"lessons_sha256": "lessons"},
        selected_lesson_ids=SELECTED_IDS,
        decomposition_bank_sha256=bank_sha256,
        model=CRDModel(
            model_id="scale_canary",
            role="scale_canary",
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            model_revision=(
                "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
            ),
            use_4bit=True,
        ),
        prompt_renderers=dict(PROMPT_RENDERERS),
    )


def _source_rows():
    result = {}
    for item in selected_lessons():
        lesson_id = item.lesson.lesson_id
        for variant in range(1, 5):
            probe_id = f"{lesson_id}_conditional_route_{variant:02d}"
            key = f"scale_canary::{lesson_id}::external::{probe_id}"
            row = {
                "lesson_id": lesson_id,
                "probe_id": probe_id,
                "category": "conditional_route",
                "arm": "external",
                "expected_action": item.lesson.desired_action,
                "predicted_action": item.lesson.desired_action,
                "correct": True,
                "top1_top2_margin": 1.0,
            }
            result[key] = {
                "row": row,
                "source_result_path": f"/source/{lesson_id}.jsonl",
                "source_raw_file_sha256": f"file-{lesson_id}",
                "source_row_sha256": _json_hash(row),
            }
    return result


def _passing_outcome(
    audit: dict,
    expected_candidate: str,
) -> dict:
    candidates = []
    for index, candidate in enumerate(audit["candidates"]):
        token_count = int(candidate["token_count"])
        total = -0.1 if candidate["candidate"] == expected_candidate else -2.0 - index
        candidates.append(
            {
                "candidate": candidate["candidate"],
                "continuation": candidate["continuation"],
                "token_ids": candidate["token_ids"],
                "token_count": token_count,
                "sum_logprob": total,
                "mean_logprob": total / token_count,
                "token_logprobs": [total / token_count] * token_count,
            }
        )
    return {
        **rank_candidate_scores(candidates),
        "latency_seconds": 0.01,
    }


def test_cpu_audit_and_raw_row_verification_cover_complete_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = selected_lessons()
    bank, bank_audit = compile_decomposition_bank(
        selected,
        compile_hard_probe_bank(selected),
    )
    config = _config(tmp_path, str(bank_audit["bank_sha256"]))
    _, bank_hash = _write_bank_audit(config, bank_audit)
    monkeypatch.setattr(
        "plasticity_placement.p0d2hcrd.runtime.load_tokenizer",
        lambda *_: PieceTokenizer(),
    )
    source_rows = _source_rows()
    _, audit_hash, audits, report = _prepare_candidate_token_audit(
        config,
        bank,
        source_rows,
        bank_hash,
    )
    assert report["decision_count"] == 1_536
    assert report["candidate_count"] == 5_376
    assert report["failed_decision_count"] == 0
    assert report["all_checks_passed"] is True
    loaded_bank_hash, _ = _load_bank_audit(config)
    loaded_audit_hash, loaded_audits, _ = _load_candidate_token_audit(
        config,
        loaded_bank_hash,
    )
    assert loaded_bank_hash == bank_hash
    assert loaded_audit_hash == audit_hash
    assert loaded_audits == audits

    session_id = "p0d2hcrd-session-test"
    fingerprint = "environment-test"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "source_sessions.json").write_text(
        json.dumps(
            [
                {
                    "session_id": session_id,
                    "environment_fingerprint": fingerprint,
                }
            ]
        ),
        encoding="utf-8",
    )
    lesson_id = SELECTED_IDS[0]
    rows = [
        _decision_row(
            config,
            "run-test",
            "nf4-bfloat16",
            probe,
            audits[probe.probe_id],
            source_rows[probe.source_row_key],
            _passing_outcome(
                audits[probe.probe_id],
                probe.expected_candidate,
            ),
            session_id,
            fingerprint,
        )
        for probe in bank[lesson_id]
    ]
    path = (
        config.output_dir
        / "results"
        / "raw"
        / "scale_canary"
        / f"{lesson_id}.jsonl"
    )
    write_probe_rows(path, rows)
    assert (
        _verify_rows(
            config,
            "run-test",
            path,
            bank[lesson_id],
            source_rows,
            audits,
            expected_precision="nf4-bfloat16",
        )
        == "nf4-bfloat16"
    )

    tampered = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    tampered["source_row_sha256"] = "changed"
    rows[0] = tampered
    write_probe_rows(path, rows)
    with pytest.raises(ValueError, match="provenance mismatch"):
        _verify_rows(
            config,
            "run-test",
            path,
            bank[lesson_id],
            source_rows,
            audits,
            expected_precision="nf4-bfloat16",
        )


def test_complete_fake_backend_aggregates_diagnostic_only_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = selected_lessons()
    bank, bank_audit = compile_decomposition_bank(
        selected,
        compile_hard_probe_bank(selected),
    )
    config = _config(tmp_path, str(bank_audit["bank_sha256"]))
    _, bank_hash = _write_bank_audit(config, bank_audit)
    monkeypatch.setattr(
        "plasticity_placement.p0d2hcrd.runtime.load_tokenizer",
        lambda *_: PieceTokenizer(),
    )
    source_rows = _source_rows()
    _, audit_hash, audits, _ = _prepare_candidate_token_audit(
        config,
        bank,
        source_rows,
        bank_hash,
    )
    session_id = "p0d2hcrd-session-aggregate"
    fingerprint = "environment-aggregate"
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "source_sessions.json").write_text(
        json.dumps(
            [
                {
                    "session_id": session_id,
                    "environment_fingerprint": fingerprint,
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = P0D2HCRDManifest.load_or_create(
        config.output_dir / "manifest.json",
        run_id="run-aggregate",
        config=config.identity_dict(),
        source_manifest_path="/source/manifest.json",
        bank_audit_sha256=bank_hash,
        candidate_token_audit_sha256=audit_hash,
    )
    for lesson_id in SELECTED_IDS:
        rows = [
            _decision_row(
                config,
                "run-aggregate",
                "nf4-bfloat16",
                probe,
                audits[probe.probe_id],
                source_rows[probe.source_row_key],
                _passing_outcome(
                    audits[probe.probe_id],
                    probe.expected_candidate,
                ),
                session_id,
                fingerprint,
            )
            for probe in bank[lesson_id]
        ]
        path = (
            config.output_dir
            / "results"
            / "raw"
            / "scale_canary"
            / f"{lesson_id}.jsonl"
        )
        write_probe_rows(path, rows)
        manifest.mark_unit(
            lesson_id,
            "evaluating",
        )
        manifest.mark_unit(
            lesson_id,
            "verified",
            result_path=str(path),
            evaluation_precision="nf4-bfloat16",
            result_sha256=_file_hash(path),
        )
    monkeypatch.setattr(
        "plasticity_placement.p0d2hcrd.analysis.resolve_request",
        lambda request: (
            config,
            bank,
            selected,
            source_rows,
            bank_audit,
        ),
    )
    summary_path = aggregate_experiment(
        config.output_dir,
        bootstrap_samples=100,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["decision_row_count"] == 1_536
    assert summary["candidate_sequence_count"] == 5_376
    assert summary["diagnostic_gate"]["status"] == (
        "no_component_bottleneck_detected"
    )
    assert all(
        summary["endpoint_summary"][endpoint]["accuracy"] == 1.0
        for endpoint in ("route_only", "retrieval_only", "combined")
    )
    assert summary["linked_source_conditional_route"]["accuracy"] == 1.0
    assert summary["gates"]["training_complexity_review_eligible"] is False
    assert summary["gates"]["automatic_training_started"] is False
    assert summary["gates"]["automatic_narrow_scan_started"] is False
