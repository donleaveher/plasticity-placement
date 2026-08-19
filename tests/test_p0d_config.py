from pathlib import Path

import pytest

from plasticity_placement.p0d.config import (
    LayerCondition,
    P0DRequest,
    P0DStage,
    band_scan_conditions,
    resolve_conditions,
)
from plasticity_placement.training.config import LayerBand


def test_band_resolution_records_exact_indices_for_uneven_depth() -> None:
    resolved = resolve_conditions(band_scan_conditions(), num_hidden_layers=10)
    by_id = {condition.condition_id: condition.selected_layers for condition in resolved}
    assert by_id == {
        "full": tuple(range(10)),
        "early": (0, 1, 2),
        "middle": (3, 4, 5),
        "late": (6, 7, 8, 9),
    }


def test_explicit_condition_rejects_duplicate_or_out_of_range_layers() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        LayerCondition("bad", LayerBand.EXPLICIT, (2, 2))
    condition = LayerCondition("layer-10", LayerBand.EXPLICIT, (10,))
    with pytest.raises(ValueError, match="within"):
        resolve_conditions((condition,), num_hidden_layers=10)


def test_band_scan_matrix_is_frozen(tmp_path: Path) -> None:
    changed = (
        LayerCondition("full", LayerBand.FULL),
        LayerCondition("middle", LayerBand.MIDDLE),
    )
    with pytest.raises(ValueError, match="frozen"):
        P0DRequest(
            output_dir=tmp_path / "out",
            source_manifest=tmp_path / "source.json",
            conditions=changed,
        )


def test_narrow_scan_requires_explicit_comparisons(tmp_path: Path) -> None:
    conditions = (
        LayerCondition("full", LayerBand.FULL),
        LayerCondition("layer-11", LayerBand.EXPLICIT, (11,)),
        LayerCondition("window-11-12", LayerBand.EXPLICIT, (11, 12)),
    )
    request = P0DRequest(
        output_dir=tmp_path / "out",
        source_manifest=tmp_path / "source.json",
        stage=P0DStage.NARROW_SCAN,
        conditions=conditions,
        conditions_config_sha256="conditions-hash",
        parent_band_run_id="p0d-band-run",
    )
    assert request.reference_condition_id == "full"
    with pytest.raises(ValueError, match="explicit"):
        P0DRequest(
            output_dir=tmp_path / "out",
            source_manifest=tmp_path / "source.json",
            stage=P0DStage.NARROW_SCAN,
            conditions=band_scan_conditions(),
            conditions_config_sha256="conditions-hash",
            parent_band_run_id="p0d-band-run",
        )
