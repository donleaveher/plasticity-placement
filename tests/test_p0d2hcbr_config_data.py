from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from plasticity_placement.p0d2hcbr.config import CorrectedExperimentSpec, ExperimentSpec
from plasticity_placement.p0d2hcbr.data import compile_heldout_bank, compile_training_banks
from plasticity_placement.p0d2hcbr.preflight import _source_overlap_audit

SPEC_PATH = (
    Path(__file__).parents[1] / "configs" / "p0d2hcbr-counterbalanced-binding-remediation-v1.json"
)
CORRECTED_SPEC_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "p0d2hcbr-counterbalanced-binding-remediation-v2.json"
)


def test_cbr_spec_freezes_twelve_parameter_matched_units(tmp_path: Path) -> None:
    spec = ExperimentSpec.from_path(SPEC_PATH)

    assert len(spec.unit_ids()) == 12
    assert spec.training_run_limit == 12
    assert spec.locked_evaluation_limit == 12
    assert spec.allows_full_parameter_training is False
    full = spec.training.to_lora_config(
        placement="full_depth",
        seed=spec.training.seeds[0],
        data_path=tmp_path / "train.jsonl",
        output_dir=tmp_path / "full",
    )
    late = spec.training.to_lora_config(
        placement="late_matched",
        seed=spec.training.seeds[0],
        data_path=tmp_path / "train.jsonl",
        output_dir=tmp_path / "late",
    )
    assert (full.layer_band.value, full.rank, full.alpha) == ("full", 8, 16)
    assert (late.layer_band.value, late.rank, late.alpha) == ("late", 24, 48)
    with pytest.raises(ValueError, match="frozen"):
        type(spec.data)(heldout_group_count=23)


def test_training_banks_match_exposure_and_disentangle_labels() -> None:
    spec = ExperimentSpec().data
    train, dev, audit = compile_training_banks(spec)

    assert audit["all_checks_passed"] is True
    assert {name: len(rows) for name, rows in train.items()} == {
        "coupled": 1_152,
        "disentangled": 1_152,
    }
    assert {name: len(rows) for name, rows in dev.items()} == {
        "coupled": 288,
        "disentangled": 288,
    }
    coupled = [row for row in train["coupled"] if row.task == "combined"]
    disentangled = [row for row in train["disentangled"] if row.task == "combined"]
    assert set((row.receipt, row.selected_slot_label) for row in coupled) == {
        ("A", "A"),
        ("B", "B"),
    }
    assert set((row.receipt, row.selected_slot_label) for row in disentangled) == {
        (receipt, slot) for receipt in ("A", "B") for slot in ("A", "B")
    }


def test_corrected_spec_exactly_matches_full_and_late_nominal_budgets(
    tmp_path: Path,
) -> None:
    spec = CorrectedExperimentSpec.from_path(CORRECTED_SPEC_PATH)
    late = spec.training.to_lora_config(
        placement="late_matched",
        seed=spec.training.seeds[0],
        data_path=tmp_path / "train.jsonl",
        output_dir=tmp_path / "late",
    )

    assert spec.training_run_limit == 6
    assert spec.imported_full_depth_unit_count == 6
    assert spec.training.num_hidden_layers * spec.training.full_depth_rank == 224
    assert len(spec.training.late_explicit_layers) * spec.training.late_matched_rank == 224
    assert late.layer_band.value == "explicit"
    assert late.explicit_layers == tuple(range(20, 28))
    assert (late.rank, late.alpha) == (28, 56)


def test_heldout_bank_is_complete_independent_factorial() -> None:
    probes, audit = compile_heldout_bank(ExperimentSpec().data)

    assert len(probes) == 1_536
    assert audit["all_checks_passed"] is True
    assert Counter(
        (
            row.receipt,
            row.selected_slot_label,
            row.selected_display_position,
            row.expected_candidate_position,
        )
        for row in probes
    ) == {
        (receipt, slot, display, candidate): 48
        for receipt in ("A", "B")
        for slot in ("A", "B")
        for display in (0, 1)
        for candidate in range(4)
    }


def test_source_overlap_audit_rejects_exact_historical_prompt(tmp_path: Path) -> None:
    spec = ExperimentSpec().data
    train, dev, _ = compile_training_banks(spec)
    heldout, _ = compile_heldout_bank(spec)
    rabx = tmp_path / "rabx" / "preflight"
    rab = tmp_path / "rab" / "preflight"
    rabx.mkdir(parents=True)
    rab.mkdir(parents=True)
    (rabx / "label_disentanglement_probes.jsonl").write_text(
        json.dumps({"prompt": "historical prompt"}) + "\n"
    )
    (rab / "binding_probes.jsonl").write_text(
        json.dumps({"prompt": "another historical prompt"}) + "\n"
    )
    rows = [
        row
        for curriculum in ("coupled", "disentangled")
        for row in train[curriculum] + dev[curriculum]
    ]

    audit = _source_overlap_audit(tmp_path / "rabx", tmp_path / "rab", rows, heldout)

    assert audit["all_checks_passed"] is True
    contaminated = [*rows]
    contaminated[0] = type(contaminated[0])(
        **{**contaminated[0].to_dict(), "prompt": "historical prompt"}
    )
    with pytest.raises(ValueError, match="source-overlap"):
        _source_overlap_audit(tmp_path / "rabx", tmp_path / "rab", contaminated, heldout)
