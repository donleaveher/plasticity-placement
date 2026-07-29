from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_route_decomposition_precision_audit_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/"
    "notebooks/p0d2h_route_decomposition_precision_audit/"
    f"{NOTEBOOK_NAME}"
)


def _markdown(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


CONFIGURATION_SOURCE = f"""
from google.colab import drive
drive.mount('/content/drive')

import json
import re
import shlex
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

from IPython.display import Markdown, display

REPO_URL = 'https://github.com/donleaveher/plasticity-placement.git'
BRANCH = '{BRANCH}'
REQUESTED_CODE_REVISION = None
REPO_DIR = Path('/content/plasticity-placement-crd-precision-analysis')

NF4_RUN_ID = 'p0d2hcrd-route-decomposition-8527b64f36'
NF4_PIPELINE_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/'
    'hard-probe-route-decomposition/v1/pipelines/pipeline-crd1'
)
NF4_STAGE_SUFFIX = 'route_decomposition-crd1'

BF16_RUN_ID = 'p0d2hcrd-route-decomposition-fd812a9314'
BF16_PIPELINE_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/'
    'hard-probe-route-decomposition-bf16/v1/pipelines/'
    'pipeline-crd-bf16-1'
)
BF16_STAGE_SUFFIX = 'route_decomposition-crd-bf16-1'

ANALYSIS_PIPELINE_ATTEMPT = 'pipeline-pa1'
ANALYSIS_ATTEMPT = 'pa1'
ANALYSIS_PIPELINE_ROOT = (
    Path(
        '/content/drive/MyDrive/plasticity-p0d/'
        'hard-probe-route-decomposition-precision-analysis/'
        'v1/pipelines'
    )
    / ANALYSIS_PIPELINE_ATTEMPT
)

for name, value in {{
    'ANALYSIS_PIPELINE_ATTEMPT': ANALYSIS_PIPELINE_ATTEMPT,
    'ANALYSIS_ATTEMPT': ANALYSIS_ATTEMPT,
}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
print('GPU strict-FP32 anomaly audit + CPU conservative tie analysis')
print('Frozen source ties: NF4=3, BF16=2')
print('No source gate or training eligibility will be changed')
"""


CHECKOUT_SOURCE = """
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', 'uv'],
    check=True,
)
if REPO_DIR.exists() and not (REPO_DIR / '.git').exists():
    raise RuntimeError(f'{REPO_DIR} exists but is not a Git repository')
if not REPO_DIR.exists():
    subprocess.run(
        ['git', 'clone', '--branch', BRANCH, REPO_URL, str(REPO_DIR)],
        check=True,
    )
dirty = subprocess.run(
    ['git', '-C', str(REPO_DIR), 'status', '--porcelain'],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if dirty:
    raise RuntimeError(f'Precision-audit checkout has local changes:\\n{dirty}')

ANALYSIS_PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
CODE_REVISION_LOCK = ANALYSIS_PIPELINE_ROOT / 'code_revision.txt'
locked_revision = (
    CODE_REVISION_LOCK.read_text().strip()
    if CODE_REVISION_LOCK.exists()
    else None
)
subprocess.run(
    ['git', '-C', str(REPO_DIR), 'fetch', 'origin', BRANCH],
    check=True,
)
revision_ref = REQUESTED_CODE_REVISION or locked_revision or f'origin/{BRANCH}'
CODE_REVISION = subprocess.run(
    ['git', '-C', str(REPO_DIR), 'rev-parse', f'{revision_ref}^{{commit}}'],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if locked_revision and CODE_REVISION != locked_revision:
    raise RuntimeError(
        f'Precision-audit code lock mismatch: {locked_revision} != '
        f'{CODE_REVISION}. Use a new ANALYSIS_PIPELINE_ATTEMPT.'
    )
if not locked_revision:
    temporary = CODE_REVISION_LOCK.with_suffix('.txt.tmp')
    temporary.write_text(CODE_REVISION + '\\n')
    temporary.replace(CODE_REVISION_LOCK)
subprocess.run(
    ['git', '-C', str(REPO_DIR), 'checkout', '--detach', CODE_REVISION],
    check=True,
)
subprocess.run(
    ['uv', 'sync', '--extra', 'train', '--extra', 'colab'],
    cwd=REPO_DIR,
    check=True,
)
print('Checked out independent precision-analysis code:', CODE_REVISION)
"""


SOURCE_SOURCE = """
def find_source(root, suffix, run_id, use_4bit, tie_count):
    matches = []
    for manifest_path in sorted(root.glob(f'runs/*/{suffix}/manifest.json')):
        summary_path = (
            manifest_path.parent / 'results' / 'aggregate' / 'summary.json'
        )
        if not summary_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        summary = json.loads(summary_path.read_text())
        observed_ties = sum(
            round(
                values['tie_rate'] * values['decision_count']
            )
            for values in summary.get('endpoint_summary', {}).values()
        )
        if (
            manifest.get('run_id') == run_id
            and len(manifest.get('units', {})) == 24
            and all(
                unit.get('state') == 'verified'
                for unit in manifest.get('units', {}).values()
            )
            and not manifest.get('errors')
            and manifest.get('model', {}).get('use_4bit') is use_4bit
            and summary.get('run_id') == run_id
            and summary.get('decision_row_count') == 1536
            and summary.get('candidate_sequence_count') == 5376
            and summary.get('diagnostic_gate', {}).get('status')
            == 'scoring_integrity_failed'
            and observed_ties == tie_count
        ):
            matches.append(manifest_path.parent)
    if len(matches) != 1:
        raise RuntimeError(
            f'Expected one completed source {run_id}, found {matches}'
        )
    return matches[0]

NF4_STAGE_DIR = find_source(
    NF4_PIPELINE_ROOT, NF4_STAGE_SUFFIX, NF4_RUN_ID, True, 3
)
BF16_STAGE_DIR = find_source(
    BF16_PIPELINE_ROOT, BF16_STAGE_SUFFIX, BF16_RUN_ID, False, 2
)

def source_hashes(stage_dir):
    manifest = stage_dir / 'manifest.json'
    summary = stage_dir / 'results' / 'aggregate' / 'summary.json'
    digest = sha256()
    files = sorted((stage_dir / 'results' / 'raw').rglob('*.jsonl'))
    if len(files) != 24:
        raise RuntimeError(f'Expected 24 raw files: {stage_dir}')
    for path in files:
        digest.update(str(path.relative_to(stage_dir)).encode())
        digest.update(path.read_bytes())
    return {
        'manifest': sha256(manifest.read_bytes()).hexdigest(),
        'summary': sha256(summary.read_bytes()).hexdigest(),
        'raw': digest.hexdigest(),
    }

NF4_HASHES = source_hashes(NF4_STAGE_DIR)
BF16_HASHES = source_hashes(BF16_STAGE_DIR)
COMBINED_SOURCE_HASH = sha256(
    json.dumps(
        {'nf4': NF4_HASHES, 'bf16': BF16_HASHES},
        sort_keys=True,
    ).encode()
).hexdigest()
PROVENANCE_KEY = (
    f'code-{CODE_REVISION[:10]}_source-{COMBINED_SOURCE_HASH[:10]}'
)
PROVENANCE_ROOT = ANALYSIS_PIPELINE_ROOT / 'runs' / PROVENANCE_KEY
FP32_AUDIT_DIR = (
    PROVENANCE_ROOT / ('fp32_precision_audit-' + ANALYSIS_ATTEMPT)
)
CONSERVATIVE_DIR = (
    PROVENANCE_ROOT / ('conservative_tie_analysis-' + ANALYSIS_ATTEMPT)
)
print(json.dumps({
    'nf4_source': str(NF4_STAGE_DIR),
    'bf16_source': str(BF16_STAGE_DIR),
    'fp32_audit_output': str(FP32_AUDIT_DIR),
    'conservative_output': str(CONSERVATIVE_DIR),
}, indent=2))
"""


FP32_RUN_SOURCE = """
import torch

if not torch.cuda.is_available():
    raise RuntimeError('Select a CUDA GPU runtime')
subprocess.run(['nvidia-smi'], check=True)

command = [
    'uv', 'run', 'plasticity-p0d2hcrd', 'audit-fp32',
    '--source-output', str(NF4_STAGE_DIR),
    '--source-output', str(BF16_STAGE_DIR),
    '--audit-output', str(FP32_AUDIT_DIR),
]
print('$', shlex.join(command))
subprocess.run(command, cwd=REPO_DIR, check=True)

if source_hashes(NF4_STAGE_DIR) != NF4_HASHES:
    raise RuntimeError('NF4 source changed during FP32 audit')
if source_hashes(BF16_STAGE_DIR) != BF16_HASHES:
    raise RuntimeError('BF16 source changed during FP32 audit')
print('READ-ONLY SOURCE CHECK PASSED')
"""


CONSERVATIVE_RUN_SOURCE = """
command = [
    'uv', 'run', 'plasticity-p0d2hcrd', 'aggregate-v2',
    '--output', str(BF16_STAGE_DIR),
    '--analysis-output', str(CONSERVATIVE_DIR),
    '--bootstrap-samples', '10000',
]
print('$', shlex.join(command))
subprocess.run(command, cwd=REPO_DIR, check=True)
if source_hashes(BF16_STAGE_DIR) != BF16_HASHES:
    raise RuntimeError('BF16 source changed during conservative analysis')
"""


RESULTS_SOURCE = """
fp32_path = FP32_AUDIT_DIR / 'fp32_precision_audit.json'
fp32_markdown = FP32_AUDIT_DIR / 'fp32_precision_audit.md'
fp32 = json.loads(fp32_path.read_text())
if (
    fp32.get('analysis_status') != 'post_hoc_mechanistic_audit'
    or fp32.get('source_tie_record_count') != 5
    or fp32.get('rescore_error_count') != 0
    or fp32.get('strict_fp32_checks', {}).get('parameter_dtype')
    != 'float32'
    or fp32.get('strict_fp32_checks', {}).get('logits_dtype')
    != 'float32'
    or fp32.get('strict_fp32_checks', {}).get('tf32_allowed') is not False
    or fp32.get('source_gate_status_changed') is not False
    or fp32.get('next_stage_eligibility_changed') is not False
):
    raise RuntimeError('Strict-FP32 audit contract failed')
display(Markdown(fp32_markdown.read_text()))

conservative_path = CONSERVATIVE_DIR / 'conservative_tie_summary.json'
conservative_markdown = CONSERVATIVE_DIR / 'conservative_tie_report.md'
conservative = json.loads(conservative_path.read_text())
gate = conservative['diagnostic_gate']
if (
    conservative.get('source_gate_status_changed') is not False
    or conservative.get('source_next_stage_eligibility_changed') is not False
    or conservative.get('tie_summary', {}).get('finite_exact_tie_count') != 2
    or conservative.get('tie_summary', {}).get('fatal_error_count') != 0
    or gate.get('status') != 'route_bottleneck_supported'
    or gate.get('bound_statuses', {}).get('match') is not True
    or gate.get('training_complexity_review_eligible') is not False
    or gate.get('automatic_training_started') is not False
    or gate.get('automatic_narrow_scan_started') is not False
):
    raise RuntimeError('Conservative tie-analysis contract failed')
display(Markdown(conservative_markdown.read_text()))
print(json.dumps({
    'fp32_classification': fp32['classification'],
    'resolved_ties': fp32['resolved_tie_count'],
    'persistent_ties': fp32['persistent_tie_count'],
    'conservative_status': gate['status'],
    'bound_statuses': gate['bound_statuses'],
    'fp32_summary': str(fp32_path),
    'conservative_summary': str(conservative_path),
    'training_started': False,
}, ensure_ascii=False, indent=2))
"""


def build_notebook() -> dict[str, Any]:
    cells = [
        _markdown(
            f"""
            # P0-D2H-CRD Strict-FP32 Audit and Conservative Tie Analysis

            [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

            This notebook keeps the completed NF4 and BF16 CRD runs immutable.
            It rescoring only their five structurally valid exact ties using a
            full FP32 CUDA forward with TF32 and autocast disabled, then applies
            the conservative lower/upper tie policy to the BF16 result in an
            independent analysis namespace.

            It does not modify either frozen zero-tie gate and contains no
            adapter, training, narrow-scan, or RLVR execution.
            """
        ),
        _markdown("## 1. Configuration and Drive mount"),
        _code(CONFIGURATION_SOURCE),
        _markdown("## 2. Independent code checkout and immutable code lock"),
        _code(CHECKOUT_SOURCE),
        _markdown("## 3. Resolve and hash the completed NF4/BF16 sources"),
        _code(SOURCE_SOURCE),
        _markdown("## 4. Strict FP32 rescore of all five exact ties (GPU)"),
        _code(FP32_RUN_SOURCE),
        _markdown("## 5. Conservative tie-interval analysis (CPU)"),
        _code(CONSERVATIVE_RUN_SOURCE),
        _markdown("## 6. Verify and display supplementary results"),
        _code(RESULTS_SOURCE),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": NOTEBOOK_NAME,
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


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
