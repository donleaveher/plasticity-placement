from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
EXPERIMENT_CODE_REVISION = "f4a71efe6bc717d345843b025c95fceeb6071954"
RESUMABLE_CODE_REVISION = "99462a4a7742b82f0cee395d4da891d1ed6e48db"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_cbr_resumable_evaluation_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/"
    "p0d2h_cbr_resumable_evaluation/"
    f"{NOTEBOOK_NAME}"
)


def markdown(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


CONFIG = f"""
from google.colab import drive
drive.mount('/content/drive')

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from IPython.display import Markdown, display

REPO_URL = 'https://github.com/donleaveher/plasticity-placement.git'
BRANCH = '{BRANCH}'
EXPERIMENT_CODE_REVISION = '{EXPERIMENT_CODE_REVISION}'
REQUESTED_RESUMABLE_CODE_REVISION = '{RESUMABLE_CODE_REVISION}'
REPO_DIR = Path('/content/plasticity-placement-cbr-resumable')

CBR_PIPELINE_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/counterbalanced-binding-remediation/'
    'v1/pipelines/pipeline-cbr1'
)
CBR_IDENTITY_DIR = 'code-f4a71efe6b_rabx-ba5d970807_settings-8a5a6390ae'
CBR_OUTPUT = CBR_PIPELINE_ROOT / 'runs' / CBR_IDENTITY_DIR / (
    'counterbalanced-binding-cbr1'
)
CBR_CODE_REVISION_LOCK = CBR_PIPELINE_ROOT / 'code_revision.txt'
RESUMABLE_CODE_REVISION_LOCK = CBR_PIPELINE_ROOT / 'resumable_code_revision.txt'

RESUME_ATTEMPT = 'resume1'
if not re.fullmatch(r'[A-Za-z0-9._-]+', RESUME_ATTEMPT):
    raise ValueError(f'RESUME_ATTEMPT contains unsafe path characters: {{RESUME_ATTEMPT!r}}')
RESUME_EVALUATIONS_ROOT = (
    CBR_PIPELINE_ROOT / 'evaluations' / CBR_IDENTITY_DIR / RESUME_ATTEMPT
)
RESUME_APPROVAL_PATH = (
    CBR_PIPELINE_ROOT / 'approvals' / CBR_IDENTITY_DIR
    / f'authorization-{{RESUME_ATTEMPT}}.json'
)

print('CBR output:', CBR_OUTPUT)
print('Resumable evaluations:', RESUME_EVALUATIONS_ROOT)
print('External approval:', RESUME_APPROVAL_PATH)
"""


CONTROLS = """
# EDIT ONLY THIS CELL BETWEEN PASSES. Enable at most one RUN_* gate.
RUN_INSPECT_RESUME = True
RUN_AUTHORIZE_RESUME = False
RUN_EVALUATE_RESUMABLE = False
RUN_AGGREGATE = False
RUN_VERIFY = False

APPROVER = ''  # Human/responsible-party identifier; never a password or token.
SHARD_SIZE = 64  # Frozen when authorization is adopted; allowed range is 8..256.

# Leave all three blank/None to process all authorized unfinished units.
# Set all three to resume only one unit.
TARGET_CURRICULUM = ''  # coupled or disentangled
TARGET_PLACEMENT = ''   # full_depth or late_matched
TARGET_SEED = None      # 20260810, 20260811, or 20260812

stages = {
    'inspect_resume': RUN_INSPECT_RESUME,
    'authorize_resume': RUN_AUTHORIZE_RESUME,
    'evaluate_resumable': RUN_EVALUATE_RESUMABLE,
    'aggregate': RUN_AGGREGATE,
    'verify': RUN_VERIFY,
}
if sum(bool(value) for value in stages.values()) > 1:
    raise ValueError(f'Enable at most one stage per pass: {stages}')
if RUN_AUTHORIZE_RESUME and not APPROVER.strip():
    raise ValueError('Set APPROVER for the authorization pass')
if not 8 <= SHARD_SIZE <= 256:
    raise ValueError('SHARD_SIZE must be between 8 and 256')
target_values = (TARGET_CURRICULUM, TARGET_PLACEMENT, TARGET_SEED)
if any(value not in ('', None) for value in target_values) and any(
    value in ('', None) for value in target_values
):
    raise ValueError('Set all three TARGET_* values together')
if TARGET_CURRICULUM and TARGET_CURRICULUM not in {'coupled', 'disentangled'}:
    raise ValueError('Unsupported TARGET_CURRICULUM')
if TARGET_PLACEMENT and TARGET_PLACEMENT not in {'full_depth', 'late_matched'}:
    raise ValueError('Unsupported TARGET_PLACEMENT')
if TARGET_SEED is not None and TARGET_SEED not in {20260810, 20260811, 20260812}:
    raise ValueError('Unsupported TARGET_SEED')
print('Selected stage:', stages)
"""


LIFECYCLE = """
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
if REPO_DIR.exists() and not (REPO_DIR / '.git').exists():
    raise RuntimeError(f'{REPO_DIR} exists but is not a Git repository')
if not REPO_DIR.exists():
    subprocess.run(['git', 'clone', '--branch', BRANCH, REPO_URL, str(REPO_DIR)], check=True)
dirty = subprocess.run(
    ['git', '-C', str(REPO_DIR), 'status', '--porcelain'],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if dirty:
    raise RuntimeError(f'Checkout has local changes:\\n{dirty}')
subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', 'origin', BRANCH], check=True)
resolved_revision = subprocess.run(
    ['git', '-C', str(REPO_DIR), 'rev-parse',
     f'{REQUESTED_RESUMABLE_CODE_REVISION}^{{commit}}'],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if resolved_revision != REQUESTED_RESUMABLE_CODE_REVISION:
    raise RuntimeError('Requested resumable implementation revision changed')
if not CBR_CODE_REVISION_LOCK.is_file():
    raise FileNotFoundError(f'Missing original CBR code lock: {CBR_CODE_REVISION_LOCK}')
if CBR_CODE_REVISION_LOCK.read_text().strip() != EXPERIMENT_CODE_REVISION:
    raise RuntimeError('Existing CBR experiment code lock differs')
locked_resume = (
    RESUMABLE_CODE_REVISION_LOCK.read_text().strip()
    if RESUMABLE_CODE_REVISION_LOCK.exists()
    else None
)
if locked_resume and locked_resume != resolved_revision:
    raise RuntimeError('Resumable code lock differs; use a new RESUME_ATTEMPT')
if not locked_resume:
    RESUMABLE_CODE_REVISION_LOCK.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESUMABLE_CODE_REVISION_LOCK.with_suffix('.txt.tmp')
    temporary.write_text(resolved_revision + '\\n')
    temporary.replace(RESUMABLE_CODE_REVISION_LOCK)
subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', '--detach', resolved_revision], check=True)
subprocess.run(
    ['uv', 'sync', '--extra', 'train', '--extra', 'colab'], cwd=REPO_DIR, check=True
)
if stages['evaluate_resumable']:
    subprocess.run(['nvidia-smi'], check=True)
if not (CBR_OUTPUT / 'manifest.json').is_file():
    raise FileNotFoundError(f'Missing existing CBR output: {CBR_OUTPUT}')

def run_capture(command):
    completed = subprocess.run(
        command, cwd=REPO_DIR, capture_output=True, text=True
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f'Command failed with exit code {completed.returncode}: {command}')
    return completed

def run_streaming(command):
    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    process = subprocess.Popen(
        command,
        cwd=REPO_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end='', flush=True)
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f'Command failed with exit code {returncode}: {command}')

def selected_keys(manifest):
    if TARGET_CURRICULUM:
        return [f'{TARGET_CURRICULUM}__{TARGET_PLACEMENT}__seed-{TARGET_SEED}']
    resume = manifest.get('resumable_evaluation')
    if not isinstance(resume, dict):
        return []
    return list(resume['eligible_unit_ids'])

print('Original experiment revision:', CBR_CODE_REVISION_LOCK.read_text().strip())
print('Resumable code revision:', resolved_revision)

if RUN_INSPECT_RESUME:
    manifest = json.loads((CBR_OUTPUT / 'manifest.json').read_text())
    if isinstance(manifest.get('resumable_evaluation'), dict):
        print(json.dumps(manifest['resumable_evaluation'], indent=2, ensure_ascii=False))
    else:
        run_capture([
            'uv', 'run', 'plasticity-p0d2hcbr', 'resumable-evaluation-template',
            '--output', str(CBR_OUTPUT),
            '--analysis-root', str(RESUME_EVALUATIONS_ROOT),
            '--shard-size', str(SHARD_SIZE),
        ])

if RUN_AUTHORIZE_RESUME:
    manifest = json.loads((CBR_OUTPUT / 'manifest.json').read_text())
    if isinstance(manifest.get('resumable_evaluation'), dict):
        print('Resumable evaluation authorization already adopted')
    else:
        result = run_capture([
            'uv', 'run', 'plasticity-p0d2hcbr', 'resumable-evaluation-template',
            '--output', str(CBR_OUTPUT),
            '--analysis-root', str(RESUME_EVALUATIONS_ROOT),
            '--shard-size', str(SHARD_SIZE),
        ])
        template = json.loads(result.stdout.splitlines()[-1])
        if RESUME_APPROVAL_PATH.exists():
            approval = json.loads(RESUME_APPROVAL_PATH.read_text())
        else:
            approval = {
                **template,
                'decision': 'approved',
                'approved_by': APPROVER.strip(),
                'approved_at': datetime.now(timezone.utc).isoformat(),
            }
            RESUME_APPROVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary = RESUME_APPROVAL_PATH.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(approval, indent=2, sort_keys=True) + '\\n')
            temporary.replace(RESUME_APPROVAL_PATH)
        run_capture([
            'uv', 'run', 'plasticity-p0d2hcbr', 'authorize-resumable-evaluation',
            '--output', str(CBR_OUTPUT),
            '--analysis-root', str(RESUME_EVALUATIONS_ROOT),
            '--authorization', str(RESUME_APPROVAL_PATH),
            '--shard-size', str(SHARD_SIZE),
        ])

if RUN_EVALUATE_RESUMABLE:
    manifest = json.loads((CBR_OUTPUT / 'manifest.json').read_text())
    if not isinstance(manifest.get('resumable_evaluation'), dict):
        raise RuntimeError('Authorize resumable evaluation before scoring')
    for key in selected_keys(manifest):
        manifest = json.loads((CBR_OUTPUT / 'manifest.json').read_text())
        claim = manifest.get('evaluation_claims', {}).get(key)
        if claim and claim.get('state') == 'complete':
            print('Evaluation already complete:', key)
            continue
        curriculum, placement, seed_label = key.split('__')
        run_streaming([
            'uv', 'run', 'plasticity-p0d2hcbr', 'evaluate-resumable',
            '--output', str(CBR_OUTPUT),
            '--curriculum', curriculum,
            '--placement', placement,
            '--seed', seed_label.removeprefix('seed-'),
        ])

if RUN_AGGREGATE:
    manifest = json.loads((CBR_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] == 'evaluated':
        run_streaming([
            'uv', 'run', 'plasticity-p0d2hcbr', 'aggregate',
            '--output', str(CBR_OUTPUT),
            '--evaluations-root', str(RESUME_EVALUATIONS_ROOT),
        ])
    elif manifest['state'] == 'complete':
        print('CBR aggregate already complete')
    else:
        raise RuntimeError(f'CBR aggregation requires evaluated: {manifest["state"]}')

if RUN_VERIFY:
    run_streaming([
        'uv', 'run', 'plasticity-p0d2hcbr', 'verify', '--output', str(CBR_OUTPUT)
    ])

manifest = json.loads((CBR_OUTPUT / 'manifest.json').read_text())
print('CBR state:', manifest['state'])
print('Evaluations:', {
    key: value['state'] for key, value in manifest['evaluation_claims'].items()
})
resume = manifest.get('resumable_evaluation', {})
for key, value in resume.get('units', {}).items():
    print(
        key,
        value.get('state'),
        f"{{len(value.get('completed_shards', {{}}))}}/{{value.get('shard_count')}} shards",
    )
"""


RESULTS = """
summary_path = CBR_OUTPUT / 'aggregate' / 'summary.json'
report_path = CBR_OUTPUT / 'aggregate' / 'report.md'
if summary_path.is_file():
    summary = json.loads(summary_path.read_text())
    display(Markdown(report_path.read_text()))
    print(json.dumps({
        'summary_path': str(summary_path),
        'decision': summary['decision']['status'],
        'placement_comparison_scope': summary['placement_comparison_scope'],
        'qualified_arms': summary['decision']['qualified_arms'],
        'additional_training_authorized': summary['additional_training_authorized'],
        'mappings_per_adapter_authorized': summary['mappings_per_adapter_authorized'],
    }, indent=2))
else:
    print('No aggregate result yet; rerun the evaluation gate to resume pending shards.')
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            markdown(
                f"""
                # P0-D2H-CBR resumable evaluation

                [Open this notebook in Google Colab]({COLAB_URL})

                This inference-only workflow reuses the existing trained CBR adapters. It writes
                an immutable checkpoint after every prompt shard and automatically validates and
                skips completed shards after a Colab interruption. It never retrains an adapter.

                The safe default only inspects the external authorization template. Edit only the
                controls cell, enable one stage per pass, and keep `SHARD_SIZE` unchanged after
                authorization.
                """
            ),
            code(CONFIG),
            code(CONTROLS),
            markdown(
                """
                Run the lifecycle cell in this order: `inspect → authorize → evaluate → aggregate
                → verify`. If Colab disconnects during evaluation, reconnect, rerun the setup and
                controls cells with `RUN_EVALUATE_RESUMABLE=True`, then rerun the lifecycle cell.
                Valid completed shards are reused automatically.
                """
            ),
            code(LIFECYCLE),
            code(RESULTS),
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / NOTEBOOK_NAME
    output.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n")
    print(output)


if __name__ == "__main__":
    main()
