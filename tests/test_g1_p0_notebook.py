from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _builder() -> ModuleType:
    path = Path(__file__).parents[1] / "notebooks/pathmem_g1_p0/build_notebook.py"
    spec = importlib.util.spec_from_file_location("pathmem_g1_p0_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_notebook_matches_builder_and_starts_with_colab_url() -> None:
    builder = _builder()
    notebook_path = Path(builder.NOTEBOOK_PATH)
    generated = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert generated == builder.build_notebook()
    first_source = "".join(generated["cells"][0]["source"])
    assert first_source.startswith("[Open this notebook in Google Colab](https://colab")
    assert builder.COLAB_URL in first_source


def test_notebook_has_one_safe_control_cell_and_ordered_lifecycle() -> None:
    notebook = _builder().build_notebook()
    controls = [
        cell for cell in notebook["cells"] if "user-controls" in cell["metadata"].get("tags", [])
    ]
    assert len(controls) == 1
    source = "".join(controls[0]["source"])
    assert "RUN_G1 = False" in source
    assert "RUN_P0 = False" in source
    assert "USE_GOOGLE_DRIVE = False" in source
    assert 'APPROVER = ""' in source

    tags = [cell["metadata"]["tags"][0] for cell in notebook["cells"]]
    assert tags.index("user-controls") < tags.index("setup")
    assert tags.index("setup") < tags.index("identity-and-plan")
    assert tags.index("identity-and-plan") < tags.index("lifecycle-execution")


def test_notebook_control_assignments_occur_only_in_control_cell() -> None:
    notebook = _builder().build_notebook()
    names = ("RUN_G1 =", "RUN_P0 =", "USE_GOOGLE_DRIVE =", "APPROVER =", "RUN_LABEL =")
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        is_control = "user-controls" in cell["metadata"].get("tags", [])
        assert is_control or not any(name in source for name in names)


def test_embedded_protocol_snapshot_verifies_with_frozen_g0() -> None:
    repository = Path(__file__).parents[1]
    protocol = repository / "notebooks/pathmem_g1_p0/protocol_snapshot"
    from plasticity_placement.pathmem.manifest import verify_g0_bundle

    report = verify_g0_bundle(protocol / "experiments/g0", protocol)
    assert report["passed"] is True
