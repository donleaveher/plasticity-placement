from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
CODE_REVISION = "f4a71efe6bc717d345843b025c95fceeb6071954"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_counterbalanced_binding_remediation_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/"
    "p0d2h_counterbalanced_binding_remediation/"
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
import re
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from IPython.display import Markdown, display

REPO_URL = 'https://github.com/donleaveher/plasticity-placement.git'
BRANCH = '{BRANCH}'
REQUESTED_CODE_REVISION = '{CODE_REVISION}'
REPO_DIR = Path('/content/plasticity-placement-cbr')

RABX_PIPELINE_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/rab-label-disentanglement/v1/'
    'pipelines/pipeline-rabx1'
)
RABX_OUTPUT = (
    RABX_PIPELINE_ROOT / 'runs/code-58eaa8c4dd_rabc-9de375f8ec/'
    'rab-label-disentanglement-rabx1'
)
RABX_CODE_REVISION_LOCK = RABX_PIPELINE_ROOT / 'code_revision.txt'

PIPELINE_ATTEMPT = 'pipeline-cbr1'
EXPERIMENT_ATTEMPT = 'cbr1'
EVALUATION_ATTEMPT = 'eval1'
PIPELINE_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/'
    'counterbalanced-binding-remediation/v1/pipelines'
) / PIPELINE_ATTEMPT

for name, value in {{
    'PIPELINE_ATTEMPT': PIPELINE_ATTEMPT,
    'EXPERIMENT_ATTEMPT': EXPERIMENT_ATTEMPT,
    'EVALUATION_ATTEMPT': EVALUATION_ATTEMPT,
}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
print('Frozen RABX source:', RABX_OUTPUT)
print('New CBR pipeline:', PIPELINE_ROOT)
"""


CONTROLS = """
# EDIT ONLY THIS CELL BETWEEN PASSES.
RUN_PLAN = True
RUN_AUTHORIZE = False
RUN_TRAIN = False
RUN_EVALUATE = False
RUN_AGGREGATE = False
RUN_VERIFY = False
APPROVER = ''  # Human/responsible-party identifier; never a password or token.

# Leave all three blank/None to run every pending unit in the selected stage.
TARGET_CURRICULUM = ''  # coupled or disentangled
TARGET_PLACEMENT = ''   # full_depth or late_matched
TARGET_SEED = None      # 20260810, 20260811, or 20260812

stages = {
    'plan': RUN_PLAN,
    'authorize': RUN_AUTHORIZE,
    'train': RUN_TRAIN,
    'evaluate': RUN_EVALUATE,
    'aggregate': RUN_AGGREGATE,
    'verify': RUN_VERIFY,
}
if sum(bool(value) for value in stages.values()) > 1:
    raise ValueError(f'Enable at most one stage per pass: {stages}')
if RUN_AUTHORIZE and not APPROVER.strip():
    raise ValueError('Set APPROVER before authorization')
target_values = (TARGET_CURRICULUM, TARGET_PLACEMENT, TARGET_SEED)
if any(value not in ('', None) for value in target_values) and any(
    value in ('', None) for value in target_values
):
    raise ValueError('Set TARGET_CURRICULUM, TARGET_PLACEMENT, and TARGET_SEED together')
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

PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
CODE_REVISION_LOCK = PIPELINE_ROOT / 'code_revision.txt'
locked_revision = CODE_REVISION_LOCK.read_text().strip() if CODE_REVISION_LOCK.exists() else None
subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', 'origin', BRANCH], check=True)
revision_ref = locked_revision or REQUESTED_CODE_REVISION
resolved_revision = subprocess.run(
    ['git', '-C', str(REPO_DIR), 'rev-parse', f'{revision_ref}^{{commit}}'],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if resolved_revision != REQUESTED_CODE_REVISION:
    raise RuntimeError('Requested CBR implementation revision changed')
if locked_revision and locked_revision != resolved_revision:
    raise RuntimeError('CBR code lock mismatch; use a new PIPELINE_ATTEMPT')
if not locked_revision:
    temporary = CODE_REVISION_LOCK.with_suffix('.txt.tmp')
    temporary.write_text(resolved_revision + '\\n')
    temporary.replace(CODE_REVISION_LOCK)
subprocess.run(
    ['git', '-C', str(REPO_DIR), 'checkout', '--detach', resolved_revision], check=True
)
subprocess.run(
    ['uv', 'sync', '--extra', 'train', '--extra', 'colab'], cwd=REPO_DIR, check=True
)
subprocess.run(['nvidia-smi'], check=True)

environment = subprocess.run(
    ['uv', 'run', 'plasticity-p0d2hcbr', 'environment'],
    cwd=REPO_DIR,
    check=True,
    capture_output=True,
    text=True,
)
environment_payload = json.loads(environment.stdout.splitlines()[-1])
if environment_payload['environment']['cuda_available'] is not True:
    raise RuntimeError('CBR training and evaluation require a CUDA runtime')

required = [
    RABX_OUTPUT / 'manifest.json',
    RABX_OUTPUT / 'summary.json',
    RABX_OUTPUT / 'audit_manifest.json',
    RABX_CODE_REVISION_LOCK,
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f'Missing frozen RABX artifacts: {missing}')
rabx_summary = json.loads((RABX_OUTPUT / 'summary.json').read_text())
if (
    rabx_summary.get('analysis_status') != 'label_disentanglement_complete'
    or rabx_summary.get('analysis', {}).get('causal_attribution', {}).get('status')
    != 'multiple_mechanisms'
    or rabx_summary.get('training_authorized') is not False
    or rabx_summary.get('mappings_per_adapter_authorized') is not False
):
    raise RuntimeError('Frozen RABX source is not the reviewed completed result')

spec_path = REPO_DIR / 'configs/p0d2hcbr-counterbalanced-binding-remediation-v1.json'
spec_hash = sha256(spec_path.read_bytes()).hexdigest()
rabx_hash = sha256((RABX_OUTPUT / 'summary.json').read_bytes()).hexdigest()
identity_dir = (
    f'code-{resolved_revision[:10]}_rabx-{rabx_hash[:10]}_settings-{spec_hash[:10]}'
)
CBR_OUTPUT = PIPELINE_ROOT / 'runs' / identity_dir / (
    'counterbalanced-binding-' + EXPERIMENT_ATTEMPT
)
EVALUATIONS_ROOT = PIPELINE_ROOT / 'evaluations' / identity_dir / EVALUATION_ATTEMPT
APPROVAL_PATH = PIPELINE_ROOT / 'approvals' / identity_dir / (
    'authorization-' + EXPERIMENT_ATTEMPT + '.json'
)

def run(command):
    completed = subprocess.run(command, cwd=REPO_DIR, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f'Command failed with exit code {completed.returncode}: {command}')
    return completed

def unit_arguments():
    if not TARGET_CURRICULUM:
        return []
    return [
        '--curriculum', TARGET_CURRICULUM,
        '--placement', TARGET_PLACEMENT,
        '--seed', str(TARGET_SEED),
    ]

print('Code revision:', resolved_revision)
print('Environment fingerprint:', environment_payload['fingerprint'])
print('CBR output:', CBR_OUTPUT)
print('Independent evaluations:', EVALUATIONS_ROOT)
print('External approval:', APPROVAL_PATH)

if RUN_PLAN:
    if (CBR_OUTPUT / 'manifest.json').is_file():
        print('CBR plan already exists; validating status')
        run(['uv', 'run', 'plasticity-p0d2hcbr', 'status', '--output', str(CBR_OUTPUT)])
    else:
        run([
            'uv', 'run', 'plasticity-p0d2hcbr', 'plan',
            '--output', str(CBR_OUTPUT),
            '--rabx-output', str(RABX_OUTPUT),
            '--source-code-revision-lock', str(RABX_CODE_REVISION_LOCK),
            '--spec', str(spec_path),
        ])

if not (CBR_OUTPUT / 'manifest.json').is_file():
    raise FileNotFoundError('Run the plan pass before any later stage')

if RUN_AUTHORIZE:
    manifest = json.loads((CBR_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] == 'planned':
        template = json.loads((CBR_OUTPUT / 'authorization.template.json').read_text())
        if APPROVAL_PATH.exists():
            approval = json.loads(APPROVAL_PATH.read_text())
        else:
            approval = {
                **template,
                'decision': 'approved',
                'approved_by': APPROVER.strip(),
                'approved_at': datetime.now(timezone.utc).isoformat(),
            }
            APPROVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary = APPROVAL_PATH.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(approval, indent=2, sort_keys=True) + '\\n')
            temporary.replace(APPROVAL_PATH)
        run([
            'uv', 'run', 'plasticity-p0d2hcbr', 'authorize',
            '--output', str(CBR_OUTPUT), '--authorization', str(APPROVAL_PATH),
        ])
    else:
        print('Authorization already adopted; current state:', manifest['state'])

if RUN_TRAIN:
    manifest = json.loads((CBR_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] in {'authorized', 'training'}:
        run([
            'uv', 'run', 'plasticity-p0d2hcbr', 'train',
            '--output', str(CBR_OUTPUT), *unit_arguments(),
        ])
    elif manifest['state'] in {'trained', 'evaluated', 'complete'}:
        print('Requested training units are already complete and immutable')
    else:
        raise RuntimeError(f'Training is invalid from state: {manifest["state"]}')

if RUN_EVALUATE:
    manifest = json.loads((CBR_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] not in {'trained', 'evaluated'}:
        raise RuntimeError(f'Evaluation requires trained/evaluated state: {manifest["state"]}')
    if TARGET_CURRICULUM:
        targets = [
            f'{TARGET_CURRICULUM}__{TARGET_PLACEMENT}__seed-{TARGET_SEED}'
        ]
    else:
        targets = sorted(manifest['training_units'])
    for key in targets:
        manifest = json.loads((CBR_OUTPUT / 'manifest.json').read_text())
        claim = manifest.get('evaluation_claims', {}).get(key)
        if claim and claim.get('state') == 'complete':
            print('Evaluation already complete:', key)
            continue
        if claim:
            raise RuntimeError(f'Incomplete immutable evaluation claim exists: {key}')
        curriculum, placement, seed_label = key.split('__')
        seed = seed_label.removeprefix('seed-')
        run([
            'uv', 'run', 'plasticity-p0d2hcbr', 'evaluate',
            '--output', str(CBR_OUTPUT),
            '--analysis-root', str(EVALUATIONS_ROOT),
            '--curriculum', curriculum,
            '--placement', placement,
            '--seed', seed,
        ])

if RUN_AGGREGATE:
    manifest = json.loads((CBR_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] == 'evaluated':
        run([
            'uv', 'run', 'plasticity-p0d2hcbr', 'aggregate',
            '--output', str(CBR_OUTPUT),
            '--evaluations-root', str(EVALUATIONS_ROOT),
        ])
    elif manifest['state'] == 'complete':
        print('Aggregate already complete; not rewriting immutable results')
    else:
        raise RuntimeError(f'Aggregation requires evaluated state: {manifest["state"]}')

if RUN_VERIFY:
    run(['uv', 'run', 'plasticity-p0d2hcbr', 'verify', '--output', str(CBR_OUTPUT)])

manifest = json.loads((CBR_OUTPUT / 'manifest.json').read_text())
print('Current state:', manifest['state'])
print('Training unit states:', {
    key: value['state'] for key, value in manifest['training_units'].items()
})
print('Evaluation claim states:', {
    key: value['state'] for key, value in manifest['evaluation_claims'].items()
})
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
        'qualified_arms': summary['decision']['qualified_arms'],
        'additional_training_authorized': summary['additional_training_authorized'],
        'mappings_per_adapter_authorized': summary['mappings_per_adapter_authorized'],
    }, indent=2))
else:
    print('No aggregate result yet. Complete plan, authorize, train, evaluate, and aggregate.')
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            markdown(
                f"""
                # P0-D2H-CBR-v1: counterbalanced binding remediation

                [Open this notebook in Google Colab]({COLAB_URL})

                This notebook runs the frozen 2 curricula × 2 LoRA placements × 3 seeds
                remediation matrix and 12 independent same-runtime evaluations. The default pass
                only plans and audits the banks. Edit only the dedicated controls cell and enable
                one stage per pass. Full-depth means LoRA across all layers, not full-parameter
                fine-tuning. No result automatically authorizes additional training or 1/4/8.
                """
            ),
            code(CONFIG),
            code(CONTROLS),
            markdown(
                """
                Run the lifecycle cell after setting the controls. After the plan pass, inspect
                `preregistration.json`, both curriculum banks, RAB-GEN bank/token audits, and the
                external authorization template before approving training.
                """
            ),
            code(LIFECYCLE),
            code(RESULTS),
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / NOTEBOOK_NAME
    path.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n")
    print(path)
    print(COLAB_URL)


if __name__ == "__main__":
    main()
