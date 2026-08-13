from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
SOURCE_CBR_CODE_REVISION = "f4a71efe6bc717d345843b025c95fceeb6071954"
SOURCE_CBR_RECOVERY_CODE_REVISION = "04437d33de7af53d6bdd408665a3be941c440965"
CORRECTED_CODE_REVISION = "99462a4a7742b82f0cee395d4da891d1ed6e48db"
RESUMABLE_CODE_REVISION = "99462a4a7742b82f0cee395d4da891d1ed6e48db"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_cbr_v2_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/p0d2h_cbr_v2/{NOTEBOOK_NAME}"
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
from hashlib import sha256
from pathlib import Path

from IPython.display import Markdown, display

REPO_URL = 'https://github.com/donleaveher/plasticity-placement.git'
BRANCH = '{BRANCH}'
SOURCE_CBR_CODE_REVISION = '{SOURCE_CBR_CODE_REVISION}'
SOURCE_CBR_RECOVERY_CODE_REVISION = '{SOURCE_CBR_RECOVERY_CODE_REVISION}'
REQUESTED_CORRECTED_CODE_REVISION = '{CORRECTED_CODE_REVISION}'
REQUESTED_RESUMABLE_CODE_REVISION = '{RESUMABLE_CODE_REVISION}'
REPO_DIR = Path('/content/plasticity-placement-cbr-v2')

CBR1_PIPELINE_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/counterbalanced-binding-remediation/'
    'v1/pipelines/pipeline-cbr1'
)
CBR1_IDENTITY_DIR = 'code-f4a71efe6b_rabx-ba5d970807_settings-8a5a6390ae'
CBR1_OUTPUT = CBR1_PIPELINE_ROOT / 'runs' / CBR1_IDENTITY_DIR / (
    'counterbalanced-binding-cbr1'
)
CBR1_CODE_REVISION_LOCK = CBR1_PIPELINE_ROOT / 'code_revision.txt'
CBR1_RECOVERY_CODE_REVISION_LOCK = (
    CBR1_PIPELINE_ROOT / 'budget_recovery_code_revision.txt'
)

CBR2_PIPELINE_ATTEMPT = 'pipeline-cbr2r1'
CBR2_EXPERIMENT_ATTEMPT = 'cbr2'
CBR2_RESUME_ATTEMPT = 'resume1'
CBR2_PIPELINE_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/counterbalanced-binding-remediation/'
    'v2/pipelines'
) / CBR2_PIPELINE_ATTEMPT
CBR2_CODE_REVISION_LOCK = CBR2_PIPELINE_ROOT / 'code_revision.txt'
CBR2_RESUMABLE_CODE_REVISION_LOCK = CBR2_PIPELINE_ROOT / (
    f'resumable_code_revision-{{CBR2_RESUME_ATTEMPT}}.txt'
)

for name, value in {{
    'CBR2_PIPELINE_ATTEMPT': CBR2_PIPELINE_ATTEMPT,
    'CBR2_EXPERIMENT_ATTEMPT': CBR2_EXPERIMENT_ATTEMPT,
    'CBR2_RESUME_ATTEMPT': CBR2_RESUME_ATTEMPT,
}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')

print('Verified CBR-v1 source:', CBR1_OUTPUT)
print('CBR-v2 pipeline:', CBR2_PIPELINE_ROOT)
"""


CONTROLS = """
# EDIT ONLY THIS CELL BETWEEN PASSES. Enable exactly one RUN_* gate.
RUN_PLAN_V2 = True
RUN_AUTHORIZE_V2 = False
RUN_TRAIN_V2 = False
RUN_STATUS_V2 = False
RUN_INSPECT_RESUME_V2 = False
RUN_AUTHORIZE_RESUME_V2 = False
RUN_EVALUATE_RESUMABLE_V2 = False
RUN_AGGREGATE_V2 = False
RUN_VERIFY_V2 = False

APPROVER = ''  # Human/responsible-party identifier; never a password or token.
SHARD_SIZE = 64  # Frozen when resumable evaluation authorization is adopted.

# Leave all three blank/None to process every pending training/evaluation unit.
# Set all three to train or evaluate one exact unit per pass.
TARGET_CURRICULUM = ''  # coupled or disentangled
TARGET_PLACEMENT = ''   # late_matched for training; either placement for evaluation
TARGET_SEED = None      # 20260810, 20260811, or 20260812

stages = {
    'plan_v2': RUN_PLAN_V2,
    'authorize_v2': RUN_AUTHORIZE_V2,
    'train_v2': RUN_TRAIN_V2,
    'status_v2': RUN_STATUS_V2,
    'inspect_resume_v2': RUN_INSPECT_RESUME_V2,
    'authorize_resume_v2': RUN_AUTHORIZE_RESUME_V2,
    'evaluate_resumable_v2': RUN_EVALUATE_RESUMABLE_V2,
    'aggregate_v2': RUN_AGGREGATE_V2,
    'verify_v2': RUN_VERIFY_V2,
}
if sum(bool(value) for value in stages.values()) != 1:
    raise ValueError(f'Enable exactly one stage per pass: {stages}')
if (RUN_AUTHORIZE_V2 or RUN_AUTHORIZE_RESUME_V2) and not APPROVER.strip():
    raise ValueError('Set APPROVER for an authorization pass')
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
if RUN_TRAIN_V2 and TARGET_PLACEMENT not in {'', 'late_matched'}:
    raise ValueError('CBR-v2 imports full_depth; training may target only late_matched')
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

def resolve_commit(requested_revision):
    resolved = subprocess.run(
        ['git', '-C', str(REPO_DIR), 'rev-parse', f'{requested_revision}^{{commit}}'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != requested_revision:
        raise RuntimeError(f'Requested code revision changed: {requested_revision}')
    return resolved

corrected_revision = resolve_commit(REQUESTED_CORRECTED_CODE_REVISION)
resumable_revision = resolve_commit(REQUESTED_RESUMABLE_CODE_REVISION)
corrected_stage_selected = any((RUN_PLAN_V2, RUN_AUTHORIZE_V2, RUN_TRAIN_V2))
resumable_stage_selected = any((
    RUN_INSPECT_RESUME_V2,
    RUN_AUTHORIZE_RESUME_V2,
    RUN_EVALUATE_RESUMABLE_V2,
    RUN_AGGREGATE_V2,
    RUN_VERIFY_V2,
))
active_revision = corrected_revision if corrected_stage_selected else resumable_revision
subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', '--detach', active_revision], check=True)
subprocess.run(
    ['uv', 'sync', '--extra', 'train', '--extra', 'colab'], cwd=REPO_DIR, check=True
)

if not CBR1_CODE_REVISION_LOCK.is_file():
    raise FileNotFoundError(f'Missing CBR-v1 code lock: {CBR1_CODE_REVISION_LOCK}')
if CBR1_CODE_REVISION_LOCK.read_text().strip() != SOURCE_CBR_CODE_REVISION:
    raise RuntimeError('CBR-v1 experiment code lock differs')
if not CBR1_RECOVERY_CODE_REVISION_LOCK.is_file():
    raise FileNotFoundError(
        'Missing CBR-v1 recovery code lock; finish CBR-v1 verify before CBR-v2'
    )
if (
    CBR1_RECOVERY_CODE_REVISION_LOCK.read_text().strip()
    != SOURCE_CBR_RECOVERY_CODE_REVISION
):
    raise RuntimeError('CBR-v1 recovery code lock differs')
if not (CBR1_OUTPUT / 'aggregate' / 'summary.json').is_file():
    raise FileNotFoundError('CBR-v1 aggregate/summary.json is missing; do not start CBR-v2')
if not (CBR1_OUTPUT / 'manifest.json').is_file():
    raise FileNotFoundError(f'Missing completed CBR-v1 output: {CBR1_OUTPUT}')

corrected_spec = REPO_DIR / 'configs/p0d2hcbr-counterbalanced-binding-remediation-v2.json'
source_prereg_hash = sha256((CBR1_OUTPUT / 'preregistration.json').read_bytes()).hexdigest()
corrected_spec_hash = sha256(corrected_spec.read_bytes()).hexdigest()
cbr2_identity_dir = (
    f'code-{corrected_revision[:10]}_cbr1-{source_prereg_hash[:10]}_'
    f'settings-{corrected_spec_hash[:10]}'
)
CBR2_OUTPUT = CBR2_PIPELINE_ROOT / 'runs' / cbr2_identity_dir / (
    'counterbalanced-binding-corrected-' + CBR2_EXPERIMENT_ATTEMPT
)
CBR2_APPROVAL_PATH = CBR2_PIPELINE_ROOT / 'approvals' / cbr2_identity_dir / (
    'authorization-' + CBR2_EXPERIMENT_ATTEMPT + '.json'
)
CBR2_RESUME_EVALUATIONS_ROOT = (
    CBR2_PIPELINE_ROOT / 'evaluations' / cbr2_identity_dir / CBR2_RESUME_ATTEMPT
)
CBR2_RESUME_APPROVAL_PATH = CBR2_PIPELINE_ROOT / 'approvals' / cbr2_identity_dir / (
    'authorization-' + CBR2_RESUME_ATTEMPT + '.json'
)

def run_capture(command):
    completed = subprocess.run(command, cwd=REPO_DIR, capture_output=True, text=True)
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

def write_lock(path, revision):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(revision + '\\n')
    temporary.replace(path)

def selected_training_keys(manifest):
    if TARGET_CURRICULUM:
        return [f'{TARGET_CURRICULUM}__{TARGET_PLACEMENT}__seed-{TARGET_SEED}']
    return sorted(
        key for key in manifest['training_units'] if '__late_matched__' in key
    )

def selected_resume_keys(manifest):
    if TARGET_CURRICULUM:
        return [f'{TARGET_CURRICULUM}__{TARGET_PLACEMENT}__seed-{TARGET_SEED}']
    resume = manifest.get('resumable_evaluation')
    if not isinstance(resume, dict):
        return []
    return list(resume['eligible_unit_ids'])

def print_v2_status():
    manifest_path = CBR2_OUTPUT / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('CBR-v2 is not planned yet')
    manifest = json.loads(manifest_path.read_text())
    print('CBR-v2 state:', manifest['state'])
    for key, value in sorted(manifest['training_units'].items()):
        print(key, '→', value['state'])
    evaluations = manifest.get('evaluation_claims', {})
    if evaluations:
        print('Evaluation claims:')
        for key, value in sorted(evaluations.items()):
            print(key, '→', value['state'])
    resume = manifest.get('resumable_evaluation')
    if isinstance(resume, dict):
        print('Resumable evaluation:', resume.get('state'))
        for key, value in sorted(resume.get('units', {}).items()):
            print(
                key,
                value.get('state'),
                f"{len(value.get('completed_shards', {}))}/{value.get('shard_count')} shards",
            )
    return manifest

CBR2_PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
locked_corrected_revision = (
    CBR2_CODE_REVISION_LOCK.read_text().strip()
    if CBR2_CODE_REVISION_LOCK.exists()
    else None
)
if locked_corrected_revision and locked_corrected_revision != corrected_revision:
    raise RuntimeError('CBR-v2 code lock mismatch; use a new CBR2_PIPELINE_ATTEMPT')
if not locked_corrected_revision:
    if not RUN_PLAN_V2:
        raise FileNotFoundError('Run the CBR-v2 plan pass before later CBR-v2 stages')
    write_lock(CBR2_CODE_REVISION_LOCK, corrected_revision)

if resumable_stage_selected:
    locked_resumable_revision = (
        CBR2_RESUMABLE_CODE_REVISION_LOCK.read_text().strip()
        if CBR2_RESUMABLE_CODE_REVISION_LOCK.exists()
        else None
    )
    if locked_resumable_revision and locked_resumable_revision != resumable_revision:
        raise RuntimeError('Resumable code lock differs; use a new CBR2_RESUME_ATTEMPT')
    if not locked_resumable_revision:
        write_lock(CBR2_RESUMABLE_CODE_REVISION_LOCK, resumable_revision)

if RUN_TRAIN_V2 or RUN_EVALUATE_RESUMABLE_V2:
    subprocess.run(['nvidia-smi'], check=True)
    environment = run_capture([
        'uv', 'run', 'plasticity-p0d2hcbr', 'environment'
    ])
    environment_payload = json.loads(environment.stdout.splitlines()[-1])
    if environment_payload['environment']['cuda_available'] is not True:
        raise RuntimeError('CBR-v2 training/evaluation requires a CUDA runtime')
    print('Environment fingerprint:', environment_payload['fingerprint'])

print('Corrected plan/train revision:', corrected_revision)
print('Resumable evaluation revision:', resumable_revision)
print('Active revision:', active_revision)
print('CBR-v2 output:', CBR2_OUTPUT)
print('CBR-v2 training approval:', CBR2_APPROVAL_PATH)
print('CBR-v2 resumable evaluations:', CBR2_RESUME_EVALUATIONS_ROOT)
print('CBR-v2 resumable approval:', CBR2_RESUME_APPROVAL_PATH)

if RUN_PLAN_V2:
    if (CBR2_OUTPUT / 'manifest.json').is_file():
        print('CBR-v2 is already planned; showing the existing immutable status')
        print_v2_status()
    else:
        run_capture([
            'uv', 'run', 'plasticity-p0d2hcbr', 'plan-corrected',
            '--output', str(CBR2_OUTPUT),
            '--source-cbr-output', str(CBR1_OUTPUT),
            '--spec', str(corrected_spec),
        ])

if RUN_AUTHORIZE_V2:
    manifest = json.loads((CBR2_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] == 'planned':
        template = json.loads((CBR2_OUTPUT / 'authorization.template.json').read_text())
        if CBR2_APPROVAL_PATH.exists():
            approval = json.loads(CBR2_APPROVAL_PATH.read_text())
        else:
            approval = {
                **template,
                'decision': 'approved',
                'approved_by': APPROVER.strip(),
                'approved_at': datetime.now(timezone.utc).isoformat(),
            }
            CBR2_APPROVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary = CBR2_APPROVAL_PATH.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(approval, indent=2, sort_keys=True) + '\\n')
            temporary.replace(CBR2_APPROVAL_PATH)
        run_capture([
            'uv', 'run', 'plasticity-p0d2hcbr', 'authorize',
            '--output', str(CBR2_OUTPUT),
            '--authorization', str(CBR2_APPROVAL_PATH),
        ])
    elif manifest['state'] in {'authorized', 'training', 'trained', 'evaluated', 'complete'}:
        print('CBR-v2 authorization already adopted; state:', manifest['state'])
    else:
        raise RuntimeError(f'CBR-v2 authorization is invalid from: {manifest["state"]}')

if RUN_TRAIN_V2:
    manifest = json.loads((CBR2_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] in {'trained', 'evaluated', 'complete'}:
        print('All six corrected late_matched units are already trained')
    elif manifest['state'] not in {'authorized', 'training'}:
        raise RuntimeError(f'CBR-v2 training is invalid from: {manifest["state"]}')
    else:
        for key in selected_training_keys(manifest):
            manifest = json.loads((CBR2_OUTPUT / 'manifest.json').read_text())
            if manifest['training_units'][key]['state'] == 'trained':
                print('Training unit already complete:', key)
                continue
            curriculum, placement, seed_label = key.split('__')
            run_streaming([
                'uv', 'run', 'plasticity-p0d2hcbr', 'train',
                '--output', str(CBR2_OUTPUT),
                '--curriculum', curriculum,
                '--placement', placement,
                '--seed', seed_label.removeprefix('seed-'),
            ])

if RUN_STATUS_V2:
    print_v2_status()

if RUN_INSPECT_RESUME_V2:
    manifest = json.loads((CBR2_OUTPUT / 'manifest.json').read_text())
    if isinstance(manifest.get('resumable_evaluation'), dict):
        print(json.dumps(manifest['resumable_evaluation'], indent=2, ensure_ascii=False))
    else:
        run_capture([
            'uv', 'run', 'plasticity-p0d2hcbr', 'resumable-evaluation-template',
            '--output', str(CBR2_OUTPUT),
            '--analysis-root', str(CBR2_RESUME_EVALUATIONS_ROOT),
            '--shard-size', str(SHARD_SIZE),
        ])

if RUN_AUTHORIZE_RESUME_V2:
    manifest = json.loads((CBR2_OUTPUT / 'manifest.json').read_text())
    if isinstance(manifest.get('resumable_evaluation'), dict):
        print('CBR-v2 resumable evaluation authorization already adopted')
    else:
        result = run_capture([
            'uv', 'run', 'plasticity-p0d2hcbr', 'resumable-evaluation-template',
            '--output', str(CBR2_OUTPUT),
            '--analysis-root', str(CBR2_RESUME_EVALUATIONS_ROOT),
            '--shard-size', str(SHARD_SIZE),
        ])
        template = json.loads(result.stdout.splitlines()[-1])
        if CBR2_RESUME_APPROVAL_PATH.exists():
            approval = json.loads(CBR2_RESUME_APPROVAL_PATH.read_text())
        else:
            approval = {
                **template,
                'decision': 'approved',
                'approved_by': APPROVER.strip(),
                'approved_at': datetime.now(timezone.utc).isoformat(),
            }
            CBR2_RESUME_APPROVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary = CBR2_RESUME_APPROVAL_PATH.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(approval, indent=2, sort_keys=True) + '\\n')
            temporary.replace(CBR2_RESUME_APPROVAL_PATH)
        run_capture([
            'uv', 'run', 'plasticity-p0d2hcbr', 'authorize-resumable-evaluation',
            '--output', str(CBR2_OUTPUT),
            '--analysis-root', str(CBR2_RESUME_EVALUATIONS_ROOT),
            '--authorization', str(CBR2_RESUME_APPROVAL_PATH),
            '--shard-size', str(SHARD_SIZE),
        ])

if RUN_EVALUATE_RESUMABLE_V2:
    manifest = json.loads((CBR2_OUTPUT / 'manifest.json').read_text())
    if not isinstance(manifest.get('resumable_evaluation'), dict):
        raise RuntimeError('Authorize CBR-v2 resumable evaluation before scoring')
    for key in selected_resume_keys(manifest):
        manifest = json.loads((CBR2_OUTPUT / 'manifest.json').read_text())
        claim = manifest.get('evaluation_claims', {}).get(key)
        if claim and claim.get('state') == 'complete':
            print('Evaluation already complete:', key)
            continue
        curriculum, placement, seed_label = key.split('__')
        run_streaming([
            'uv', 'run', 'plasticity-p0d2hcbr', 'evaluate-resumable',
            '--output', str(CBR2_OUTPUT),
            '--curriculum', curriculum,
            '--placement', placement,
            '--seed', seed_label.removeprefix('seed-'),
        ])

if RUN_AGGREGATE_V2:
    manifest = json.loads((CBR2_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] == 'evaluated':
        run_streaming([
            'uv', 'run', 'plasticity-p0d2hcbr', 'aggregate',
            '--output', str(CBR2_OUTPUT),
            '--evaluations-root', str(CBR2_RESUME_EVALUATIONS_ROOT),
        ])
    elif manifest['state'] == 'complete':
        print('CBR-v2 aggregate already complete')
    else:
        raise RuntimeError(f'CBR-v2 aggregation requires evaluated: {manifest["state"]}')

if RUN_VERIFY_V2:
    run_streaming([
        'uv', 'run', 'plasticity-p0d2hcbr', 'verify', '--output', str(CBR2_OUTPUT)
    ])

if (CBR2_OUTPUT / 'manifest.json').is_file() and not RUN_STATUS_V2:
    print_v2_status()
"""


RESULTS = """
summary_path = CBR2_OUTPUT / 'aggregate' / 'summary.json'
report_path = CBR2_OUTPUT / 'aggregate' / 'report.md'
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
    }, indent=2, ensure_ascii=False))
else:
    print(
        'No CBR-v2 aggregate result yet. Train all 12 units, then use the resumable '
        'evaluation gates; do not use the legacy RUN_EVALUATE_V2 gate.'
    )
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            markdown(
                f"""
                # P0-D2H-CBR-v2 corrected matched-budget experiment

                [Open this notebook directly in Google Colab]({COLAB_URL})

                This standalone workflow starts only after CBR-v1 has passed `verify` and
                `aggregate/summary.json` exists. It imports the six immutable full-depth controls,
                trains only six corrected `late_matched` adapters on explicit layers 20–27 at
                rank 28, and evaluates all 12 units with immutable resumable prompt shards.

                Edit only the dedicated controls cell and run exactly one gate per pass. The
                default gate is the read/plan-safe `RUN_PLAN_V2=True`. The legacy monolithic v2
                evaluation gate is intentionally absent.
                """
            ),
            code(CONFIG),
            code(CONTROLS),
            markdown(
                """
                Run the lifecycle cell in this order:

                1. `RUN_PLAN_V2=True`.
                2. `RUN_AUTHORIZE_V2=True` with `APPROVER="tourbillion"`.
                3. `RUN_TRAIN_V2=True`; leave `TARGET_*` blank for all six corrected late units,
                   or select one `late_matched` unit per pass.
                4. `RUN_STATUS_V2=True`; continue only when all 12 units and the v2 manifest are
                   `trained`.
                5. `RUN_INSPECT_RESUME_V2=True`.
                6. `RUN_AUTHORIZE_RESUME_V2=True` with the same responsible-party identifier.
                7. `RUN_EVALUATE_RESUMABLE_V2=True`; reconnect and rerun this same gate after a
                   disconnect. Valid completed shards are verified and skipped automatically.
                8. `RUN_AGGREGATE_V2=True` after all 12 evaluation claims are complete.
                9. `RUN_VERIFY_V2=True`; success returns `aggregate/summary.json`.

                Keep `SHARD_SIZE=64` after resumable authorization. None of these gates authorizes
                further training, checkpoint selection, hyperparameter search, or 1/4/8 mappings.
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
    print(COLAB_URL)


if __name__ == "__main__":
    main()
