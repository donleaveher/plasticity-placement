from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

NOTEBOOK_DIR = (
    Path(__file__).resolve().parents[1] / "notebooks" / "p0d2h_route_decomposition_precision_audit"
)


def _load_generator() -> ModuleType:
    path = NOTEBOOK_DIR / "build_p0d2hcrd_precision_audit_notebook.py"
    spec = importlib.util.spec_from_file_location(
        "build_p0d2hcrd_precision_audit_notebook",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_precision_audit_notebook_matches_generator_and_parses() -> None:
    generator = _load_generator()
    path = NOTEBOOK_DIR / "p0d2h_route_decomposition_precision_audit_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook == generator.build_notebook()
    assert notebook["metadata"]["colab"]["name"] == path.name
    assert notebook["metadata"]["accelerator"] == "GPU"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(
                "".join(cell["source"]),
                filename=f"{path.name}:cell-{index}",
            )


def test_precision_audit_notebook_freezes_read_only_workflow() -> None:
    notebook = json.loads(
        (NOTEBOOK_DIR / "p0d2h_route_decomposition_precision_audit_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    code = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "p0d2hcrd-route-decomposition-8527b64f36" in code
    assert "p0d2hcrd-route-decomposition-fd812a9314" in code
    assert "'plasticity-p0d2hcrd', 'audit-fp32'" in code
    assert "'plasticity-p0d2hcrd', 'aggregate-v2'" in code
    assert "--source-output', str(NF4_STAGE_DIR)" in code
    assert "--source-output', str(BF16_STAGE_DIR)" in code
    assert "source_tie_record_count') != 5" in code
    assert "gate.get('status') != 'route_bottleneck_supported'" in code
    assert "training_started': False" in code
    assert "'plasticity-p0d2hcrd', 'run'" not in code
    assert "'plasticity-train-lora'" not in code
    assert "source_hashes(NF4_STAGE_DIR) != NF4_HASHES" in code
    assert "source_hashes(BF16_STAGE_DIR) != BF16_HASHES" in code

    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    assert "full FP32 CUDA forward with TF32 and autocast disabled" in markdown
    assert "does not modify either frozen zero-tie gate" in markdown
