from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
EXPERIMENT_CODE_REVISION = "f4a71efe6bc717d345843b025c95fceeb6071954"
RECOVERY_CODE_REVISION = "04437d33de7af53d6bdd408665a3be941c440965"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_cbr_budget_recovery_corrected_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/"
    "p0d2h_cbr_budget_recovery_corrected/"
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
EXPERIMENT_CODE_REVISION = '{EXPERIMENT_CODE_REVISION}'
REQUESTED_RECOVERY_CODE_REVISION = '{RECOVERY_CODE_REVISION}'
REPO_DIR = Path('/content/plasticity-placement-cbr-budget-recovery')

CBR1_PIPELINE_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/counterbalanced-binding-remediation/'
    'v1/pipelines/pipeline-cbr1'
)
CBR1_IDENTITY_DIR = (
    'code-f4a71efe6b_rabx-ba5d970807_settings-8a5a6390ae'
)
CBR1_OUTPUT = CBR1_PIPELINE_ROOT / 'runs' / CBR1_IDENTITY_DIR / (
    'counterbalanced-binding-cbr1'
)
CBR1_EVALUATIONS_ROOT = CBR1_PIPELINE_ROOT / 'evaluations' / CBR1_IDENTITY_DIR / 'eval1'
CBR1_CODE_REVISION_LOCK = CBR1_PIPELINE_ROOT / 'code_revision.txt'
RECOVERY_CODE_REVISION_LOCK = CBR1_PIPELINE_ROOT / 'budget_recovery_code_revision.txt'
DEVIATION_APPROVAL_PATH = CBR1_PIPELINE_ROOT / 'approvals' / CBR1_IDENTITY_DIR / (
    'budget-deviation-cbr1.json'
)

CBR2_PIPELINE_ATTEMPT = 'pipeline-cbr2'
CBR2_EXPERIMENT_ATTEMPT = 'cbr2'
CBR2_EVALUATION_ATTEMPT = 'eval1'
CBR2_PIPELINE_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/counterbalanced-binding-remediation/'
    'v2/pipelines'
) / CBR2_PIPELINE_ATTEMPT
CBR2_CODE_REVISION_LOCK = CBR2_PIPELINE_ROOT / 'code_revision.txt'

for name, value in {{
    'CBR2_PIPELINE_ATTEMPT': CBR2_PIPELINE_ATTEMPT,
    'CBR2_EXPERIMENT_ATTEMPT': CBR2_EXPERIMENT_ATTEMPT,
    'CBR2_EVALUATION_ATTEMPT': CBR2_EVALUATION_ATTEMPT,
}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
print('Existing CBR-v1:', CBR1_OUTPUT)
print('New corrected CBR-v2 pipeline:', CBR2_PIPELINE_ROOT)
"""


CONTROLS = """
# EDIT ONLY THIS CELL BETWEEN PASSES. Enable at most one RUN_* gate.
RUN_INSPECT_DEVIATION = True
RUN_AUTHORIZE_DEVIATION = False
RUN_EVALUATE_V1 = False
RUN_AGGREGATE_V1 = False
RUN_VERIFY_V1 = False
RUN_PLAN_V2 = False
RUN_AUTHORIZE_V2 = False
RUN_TRAIN_V2 = False
RUN_EVALUATE_V2 = False
RUN_AGGREGATE_V2 = False
RUN_VERIFY_V2 = False
APPROVER = ''  # Human/responsible-party identifier; never a password or token.

# Leave all three blank/None to process every pending unit in an evaluation stage.
# For safer Colab resume, set all three and process one unit per pass.
TARGET_CURRICULUM = ''  # coupled or disentangled
TARGET_PLACEMENT = ''   # full_depth or late_matched
TARGET_SEED = None      # 20260810, 20260811, or 20260812

stages = {
    'inspect_deviation': RUN_INSPECT_DEVIATION,
    'authorize_deviation': RUN_AUTHORIZE_DEVIATION,
    'evaluate_v1': RUN_EVALUATE_V1,
    'aggregate_v1': RUN_AGGREGATE_V1,
    'verify_v1': RUN_VERIFY_V1,
    'plan_v2': RUN_PLAN_V2,
    'authorize_v2': RUN_AUTHORIZE_V2,
    'train_v2': RUN_TRAIN_V2,
    'evaluate_v2': RUN_EVALUATE_V2,
    'aggregate_v2': RUN_AGGREGATE_V2,
    'verify_v2': RUN_VERIFY_V2,
}
if sum(bool(value) for value in stages.values()) > 1:
    raise ValueError(f'Enable at most one stage per pass: {stages}')
if (RUN_AUTHORIZE_DEVIATION or RUN_AUTHORIZE_V2) and not APPROVER.strip():
    raise ValueError('Set APPROVER for an authorization pass')
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
    raise ValueError('CBR-v2 only trains pending late_matched units; full_depth is imported')
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
resolved_recovery_revision = subprocess.run(
    ['git', '-C', str(REPO_DIR), 'rev-parse', f'{REQUESTED_RECOVERY_CODE_REVISION}^{{commit}}'],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if resolved_recovery_revision != REQUESTED_RECOVERY_CODE_REVISION:
    raise RuntimeError('Requested recovery implementation revision changed')

if not CBR1_CODE_REVISION_LOCK.is_file():
    raise FileNotFoundError(f'Missing original CBR-v1 code lock: {CBR1_CODE_REVISION_LOCK}')
experiment_revision = CBR1_CODE_REVISION_LOCK.read_text().strip()
if experiment_revision != EXPERIMENT_CODE_REVISION:
    raise RuntimeError('Existing CBR-v1 experiment code lock differs')
locked_recovery = (
    RECOVERY_CODE_REVISION_LOCK.read_text().strip()
    if RECOVERY_CODE_REVISION_LOCK.exists()
    else None
)
if locked_recovery and locked_recovery != resolved_recovery_revision:
    raise RuntimeError('Recovery code lock mismatch; inspect before changing attempts')
if not locked_recovery:
    RECOVERY_CODE_REVISION_LOCK.parent.mkdir(parents=True, exist_ok=True)
    temporary = RECOVERY_CODE_REVISION_LOCK.with_suffix('.txt.tmp')
    temporary.write_text(resolved_recovery_revision + '\\n')
    temporary.replace(RECOVERY_CODE_REVISION_LOCK)
subprocess.run(
    ['git', '-C', str(REPO_DIR), 'checkout', '--detach', resolved_recovery_revision], check=True
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
    raise RuntimeError('CBR evaluation and corrected training require a CUDA runtime')
if not (CBR1_OUTPUT / 'manifest.json').is_file():
    raise FileNotFoundError(f'Missing existing CBR-v1 output: {CBR1_OUTPUT}')

corrected_spec = REPO_DIR / 'configs/p0d2hcbr-counterbalanced-binding-remediation-v2.json'
source_prereg_hash = sha256((CBR1_OUTPUT / 'preregistration.json').read_bytes()).hexdigest()
corrected_spec_hash = sha256(corrected_spec.read_bytes()).hexdigest()
cbr2_identity_dir = (
    f'code-{resolved_recovery_revision[:10]}_cbr1-{source_prereg_hash[:10]}_'
    f'settings-{corrected_spec_hash[:10]}'
)
CBR2_OUTPUT = CBR2_PIPELINE_ROOT / 'runs' / cbr2_identity_dir / (
    'counterbalanced-binding-corrected-' + CBR2_EXPERIMENT_ATTEMPT
)
CBR2_EVALUATIONS_ROOT = CBR2_PIPELINE_ROOT / 'evaluations' / cbr2_identity_dir / (
    CBR2_EVALUATION_ATTEMPT
)
CBR2_APPROVAL_PATH = CBR2_PIPELINE_ROOT / 'approvals' / cbr2_identity_dir / (
    'authorization-' + CBR2_EXPERIMENT_ATTEMPT + '.json'
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

def selected_keys(manifest):
    if TARGET_CURRICULUM:
        return [f'{TARGET_CURRICULUM}__{TARGET_PLACEMENT}__seed-{TARGET_SEED}']
    return sorted(manifest['training_units'])

def evaluate_pending(output, evaluations_root):
    manifest = json.loads((output / 'manifest.json').read_text())
    for key in selected_keys(manifest):
        manifest = json.loads((output / 'manifest.json').read_text())
        claim = manifest.get('evaluation_claims', {}).get(key)
        if claim and claim.get('state') == 'complete':
            print('Evaluation already complete:', key)
            continue
        if claim:
            raise RuntimeError(f'Incomplete immutable evaluation claim exists: {key}')
        curriculum, placement, seed_label = key.split('__')
        run([
            'uv', 'run', 'plasticity-p0d2hcbr', 'evaluate',
            '--output', str(output),
            '--analysis-root', str(evaluations_root),
            '--curriculum', curriculum,
            '--placement', placement,
            '--seed', seed_label.removeprefix('seed-'),
        ])

print('Original experiment revision:', experiment_revision)
print('Recovery/corrected code revision:', resolved_recovery_revision)
print('Environment fingerprint:', environment_payload['fingerprint'])
print('CBR-v1 output:', CBR1_OUTPUT)
print('CBR-v1 evaluations:', CBR1_EVALUATIONS_ROOT)
print('Deviation approval:', DEVIATION_APPROVAL_PATH)
print('CBR-v2 output:', CBR2_OUTPUT)
print('CBR-v2 evaluations:', CBR2_EVALUATIONS_ROOT)
print('CBR-v2 approval:', CBR2_APPROVAL_PATH)

v2_stage_selected = any((
    RUN_PLAN_V2,
    RUN_AUTHORIZE_V2,
    RUN_TRAIN_V2,
    RUN_EVALUATE_V2,
    RUN_AGGREGATE_V2,
    RUN_VERIFY_V2,
))
if v2_stage_selected:
    CBR2_PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
    locked_v2_revision = (
        CBR2_CODE_REVISION_LOCK.read_text().strip()
        if CBR2_CODE_REVISION_LOCK.exists()
        else None
    )
    if locked_v2_revision and locked_v2_revision != resolved_recovery_revision:
        raise RuntimeError('CBR-v2 code lock mismatch; use a new CBR2_PIPELINE_ATTEMPT')
    if not locked_v2_revision:
        if not RUN_PLAN_V2:
            raise FileNotFoundError('Run the CBR-v2 plan pass before later CBR-v2 stages')
        temporary = CBR2_CODE_REVISION_LOCK.with_suffix('.txt.tmp')
        temporary.write_text(resolved_recovery_revision + '\\n')
        temporary.replace(CBR2_CODE_REVISION_LOCK)

if RUN_INSPECT_DEVIATION:
    run([
        'uv', 'run', 'plasticity-p0d2hcbr', 'budget-deviation-template',
        '--output', str(CBR1_OUTPUT),
    ])

if RUN_AUTHORIZE_DEVIATION:
    manifest = json.loads((CBR1_OUTPUT / 'manifest.json').read_text())
    if manifest.get('budget_deviation'):
        print('Budget-deviation authorization already adopted')
    else:
        result = run([
            'uv', 'run', 'plasticity-p0d2hcbr', 'budget-deviation-template',
            '--output', str(CBR1_OUTPUT),
        ])
        template = json.loads(result.stdout.splitlines()[-1])
        if DEVIATION_APPROVAL_PATH.exists():
            approval = json.loads(DEVIATION_APPROVAL_PATH.read_text())
        else:
            approval = {
                **template,
                'decision': 'approved',
                'approved_by': APPROVER.strip(),
                'approved_at': datetime.now(timezone.utc).isoformat(),
            }
            DEVIATION_APPROVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary = DEVIATION_APPROVAL_PATH.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(approval, indent=2, sort_keys=True) + '\\n')
            temporary.replace(DEVIATION_APPROVAL_PATH)
        run([
            'uv', 'run', 'plasticity-p0d2hcbr', 'authorize-budget-deviation',
            '--output', str(CBR1_OUTPUT),
            '--authorization', str(DEVIATION_APPROVAL_PATH),
        ])

if RUN_EVALUATE_V1:
    manifest = json.loads((CBR1_OUTPUT / 'manifest.json').read_text())
    if not manifest.get('budget_deviation'):
        raise RuntimeError('Authorize the CBR-v1 budget deviation before evaluation')
    if manifest['state'] not in {'trained', 'evaluated'}:
        raise RuntimeError(f'CBR-v1 evaluation requires trained/evaluated: {manifest["state"]}')
    evaluate_pending(CBR1_OUTPUT, CBR1_EVALUATIONS_ROOT)

if RUN_AGGREGATE_V1:
    manifest = json.loads((CBR1_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] == 'evaluated':
        run([
            'uv', 'run', 'plasticity-p0d2hcbr', 'aggregate',
            '--output', str(CBR1_OUTPUT),
            '--evaluations-root', str(CBR1_EVALUATIONS_ROOT),
        ])
    elif manifest['state'] == 'complete':
        print('CBR-v1 aggregate already complete')
    else:
        raise RuntimeError(f'CBR-v1 aggregation requires evaluated: {manifest["state"]}')

if RUN_VERIFY_V1:
    run(['uv', 'run', 'plasticity-p0d2hcbr', 'verify', '--output', str(CBR1_OUTPUT)])

if RUN_PLAN_V2:
    if (CBR2_OUTPUT / 'manifest.json').is_file():
        run(['uv', 'run', 'plasticity-p0d2hcbr', 'status', '--output', str(CBR2_OUTPUT)])
    else:
        run([
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
        run([
            'uv', 'run', 'plasticity-p0d2hcbr', 'authorize',
            '--output', str(CBR2_OUTPUT), '--authorization', str(CBR2_APPROVAL_PATH),
        ])
    else:
        print('CBR-v2 authorization already adopted; state:', manifest['state'])

if RUN_TRAIN_V2:
    manifest = json.loads((CBR2_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] not in {'authorized', 'training'}:
        if manifest['state'] in {'trained', 'evaluated', 'complete'}:
            print('All six corrected late units are already trained')
        else:
            raise RuntimeError(f'CBR-v2 training is invalid from: {manifest["state"]}')
    else:
        keys = selected_keys(manifest)
        if not TARGET_CURRICULUM:
            keys = [key for key in keys if '__late_matched__' in key]
        for key in keys:
            manifest = json.loads((CBR2_OUTPUT / 'manifest.json').read_text())
            if manifest['training_units'][key]['state'] == 'trained':
                print('Training unit already complete:', key)
                continue
            curriculum, placement, seed_label = key.split('__')
            run([
                'uv', 'run', 'plasticity-p0d2hcbr', 'train',
                '--output', str(CBR2_OUTPUT),
                '--curriculum', curriculum,
                '--placement', placement,
                '--seed', seed_label.removeprefix('seed-'),
            ])

if RUN_EVALUATE_V2:
    manifest = json.loads((CBR2_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] not in {'trained', 'evaluated'}:
        raise RuntimeError(f'CBR-v2 evaluation requires trained/evaluated: {manifest["state"]}')
    evaluate_pending(CBR2_OUTPUT, CBR2_EVALUATIONS_ROOT)

if RUN_AGGREGATE_V2:
    manifest = json.loads((CBR2_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] == 'evaluated':
        run([
            'uv', 'run', 'plasticity-p0d2hcbr', 'aggregate',
            '--output', str(CBR2_OUTPUT),
            '--evaluations-root', str(CBR2_EVALUATIONS_ROOT),
        ])
    elif manifest['state'] == 'complete':
        print('CBR-v2 aggregate already complete')
    else:
        raise RuntimeError(f'CBR-v2 aggregation requires evaluated: {manifest["state"]}')

if RUN_VERIFY_V2:
    run(['uv', 'run', 'plasticity-p0d2hcbr', 'verify', '--output', str(CBR2_OUTPUT)])

for label, output in (('CBR-v1', CBR1_OUTPUT), ('CBR-v2', CBR2_OUTPUT)):
    if (output / 'manifest.json').is_file():
        manifest = json.loads((output / 'manifest.json').read_text())
        print(label, 'state:', manifest['state'])
        print(label, 'training:', {
            key: value['state'] for key, value in manifest['training_units'].items()
        })
        print(label, 'evaluations:', {
            key: value['state'] for key, value in manifest['evaluation_claims'].items()
        })
"""


RESULTS = """
result_outputs = (
    ('CBR-v1 deviation analysis', CBR1_OUTPUT),
    ('CBR-v2 corrected', CBR2_OUTPUT),
)
for label, output in result_outputs:
    summary_path = output / 'aggregate' / 'summary.json'
    report_path = output / 'aggregate' / 'report.md'
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        display(Markdown(f'## {label}'))
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
        print(label, ': no aggregate result yet')
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            markdown(
                f"""
                # P0-D2H-CBR budget recovery and corrected placement

                [Open this notebook in Google Colab]({COLAB_URL})

                Phase 1 performs an externally authorized, fully descriptive evaluation of the
                12 existing CBR-v1 adapters while marking every cross-placement contrast as
                parameter-count-confounded. Phase 2 creates a new CBR-v2 preregistration, imports
                six immutable full-depth controls, trains six corrected late adapters on explicit
                layers 20–27 at rank 28, and evaluates the exact matched-budget matrix.

                The default pass only inspects the deviation authorization template. Edit only
                the dedicated controls cell and enable one stage per pass. Neither phase
                authorizes 1/4/8 mappings or any further training.
                """
            ),
            code(CONFIG),
            code(CONTROLS),
            markdown(
                """
                Run the lifecycle cell after changing the controls. Complete the CBR-v1 sequence
                `inspect → authorize deviation → evaluate → aggregate → verify` before beginning
                `plan v2 → authorize v2 → train six late units → evaluate → aggregate → verify`.
                One-unit training/evaluation passes are safer against Colab disconnects.
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
