from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from plasticity_placement.p0d2hcbr.config import ExperimentSpec
from plasticity_placement.p0d2hcbr.data import compile_heldout_bank, compile_training_banks

SPEC_PATH = (
    Path(__file__).parents[1] / "configs" / "p0d2hcbr-counterbalanced-binding-remediation-v1.json"
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
