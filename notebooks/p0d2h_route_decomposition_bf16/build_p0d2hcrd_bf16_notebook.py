from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from textwrap import dedent
from types import ModuleType
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_route_decomposition_bf16_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/"
    "notebooks/p0d2h_route_decomposition_bf16/"
    f"{NOTEBOOK_NAME}"
)
BASE_GENERATOR_PATH = (
    Path(__file__).resolve().parents[1] / "p0d2h_route_decomposition" / "build_p0d2hcrd_notebook.py"
)


def _load_base_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_p0d2hcrd_notebook_for_bf16",
        BASE_GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CRD notebook generator: {BASE_GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace(
    notebook: dict[str, Any],
    old: str,
    new: str,
    *,
    expected_count: int = 1,
) -> None:
    observed = 0
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        count = source.count(old)
        if count:
            observed += count
            cell["source"] = source.replace(old, new).splitlines(keepends=True)
    if observed != expected_count:
        raise ValueError(
            f"BF16 notebook replacement count changed: {old!r} "
            f"expected={expected_count} observed={observed}"
        )


def _markdown(source: str) -> list[str]:
    return dedent(source).strip().splitlines(keepends=True)


def build_notebook() -> dict[str, Any]:
    base = _load_base_generator().build_notebook()
    notebook = json.loads(json.dumps(base))
    notebook["metadata"]["colab"]["name"] = NOTEBOOK_NAME
    notebook["cells"][0]["source"] = _markdown(
        f"""
        # P0-D2H-CRD BF16 Precision Retry

        [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

        This prospective precision retry keeps the exact model revision,
        forced-choice source, 24 lessons, 1,536 decisions, prompts, candidates,
        sum-logprob scoring, and zero-tie gate from P0-D2H-CRD. The only
        experimental change is loading the original BF16 weights instead of
        NF4-quantized weights.

        Run in order. CPU source, bank, and candidate-token audits precede the
        only GPU model load. Use an L4, A100, or another CUDA GPU with BF16
        support. No adapter, training, scan, RLVR, or automatic next stage is
        present.
        """
    )

    replacements = (
        (
            "REPO_DIR = Path('/content/plasticity-placement')",
            "REPO_DIR = Path('/content/plasticity-placement-crd-bf16')",
        ),
        (
            "# Independent route/retrieval decomposition destination.",
            "# Independent full-BF16 precision-retry destination.",
        ),
        (
            "CRD_PIPELINE_ATTEMPT = 'pipeline-crd1'",
            "CRD_PIPELINE_ATTEMPT = 'pipeline-crd-bf16-1'",
        ),
        ("CRD_ATTEMPT = 'crd1'", "CRD_ATTEMPT = 'crd-bf16-1'"),
        (
            "/hard-probe-route-decomposition/v1/pipelines",
            "/hard-probe-route-decomposition-bf16/v1/pipelines",
        ),
        (
            "'pipeline_version': 'p0d2hcrd-counterbalanced-bank-v1',",
            "'pipeline_version': 'p0d2hcrd-bf16-precision-retry-v1',",
        ),
        (
            "'model_revision': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',",
            (
                "'model_revision': "
                "'989aa7980e4cf806f80c7fef2b1adb7bc71aa306',\n"
                "    'use_4bit': False,\n"
                "    'expected_evaluation_precision': 'bfloat16',\n"
                "    'comparison_source_run_id': "
                "'p0d2hcrd-route-decomposition-8527b64f36',"
            ),
        ),
        (
            "        command.extend(['--source-manifest', str(SOURCE_MANIFEST)])\n"
            "    return command",
            "        command.extend(['--source-manifest', str(SOURCE_MANIFEST)])\n"
            "        command.append('--no-4bit')\n"
            "    return command",
        ),
        (
            "    or plan['expected_candidate_sequence_count'] != 5376\n"
            "    or plan['base_only'] is not True",
            "    or plan['expected_candidate_sequence_count'] != 5376\n"
            "    or plan['config']['model'].get('use_4bit') is not False\n"
            "    or plan['base_only'] is not True",
        ),
        (
            "    print('FORMAL CRD GPU SCORING STARTING: first model CUDA load occurs here')",
            "    import torch\n"
            "    if not torch.cuda.is_bf16_supported():\n"
            "        raise RuntimeError('Select a CUDA GPU with BF16 support')\n"
            "    print('FORMAL CRD BF16 SCORING STARTING: first model CUDA load occurs here')",
        ),
        (
            "run_checked('p0d2hcrd-formal-run', crd_command('run'))",
            "run_checked('p0d2hcrd-bf16-formal-run', crd_command('run'))",
        ),
        (
            "        or manifest.get('errors')\n"
            "    ):\n"
            "        raise RuntimeError('CRD manifest is not complete and verified')",
            "        or manifest.get('errors')\n"
            "        or manifest.get('model', {}).get('use_4bit') is not False\n"
            "        or any(\n"
            "            unit.get('evaluation_precision') != 'bfloat16'\n"
            "            for unit in manifest.get('units', {}).values()\n"
            "        )\n"
            "    ):\n"
            "        raise RuntimeError(\n"
            "            'BF16 CRD manifest is not complete, verified, and bfloat16'\n"
            "        )",
        ),
        (
            "    decision = json.loads(decision_path.read_text())\n"
            "    display(Markdown(table_path.read_text()))",
            "    decision = json.loads(decision_path.read_text())\n"
            "    if summary.get('model', {}).get('use_4bit') is not False:\n"
            "        raise RuntimeError('Aggregate did not preserve BF16 model identity')\n"
            "    display(Markdown(table_path.read_text()))",
        ),
        (
            "        'endpoint_summary': summary['endpoint_summary'],",
            "        'model': summary['model'],\n"
            "        'endpoint_summary': summary['endpoint_summary'],",
        ),
        (
            "print('Endpoint candidate total: 5376 candidate sequences')",
            "print('Endpoint candidate total: 5376 candidate sequences')\n"
            "print('Precision intervention: NF4 -> original BF16 weights')",
        ),
        (
            "f'CRD code lock mismatch: {locked_revision} != {CODE_REVISION}. '",
            "f'CRD BF16 code lock mismatch: {locked_revision} != {CODE_REVISION}. '",
        ),
        (
            "## 6. Formal route/retrieval/combined scoring (GPU)",
            "## 6. Formal BF16 route/retrieval/combined scoring (GPU)",
        ),
        (
            "print('RECOVERY MANIFEST VERIFIED: 24 units, 1536 rows')",
            "print('BF16 RECOVERY MANIFEST VERIFIED: 24 units, 1536 rows')",
        ),
    )
    for old, new in replacements:
        _replace(notebook, old, new)
    return notebook


def main() -> None:
    output = OUTPUT_DIR / NOTEBOOK_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
