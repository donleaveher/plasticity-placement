from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_composition_preserving_remediation_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/"
    "p0d2h_composition_preserving_remediation/"
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
REQUESTED_CODE_REVISION = None
REPO_DIR = Path('/content/plasticity-placement-cpr')

# Verified RR1 and completed same-runtime failure evidence.
RR_PIPELINE_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/route-remediation-lora/v1/'
    'pipelines/pipeline-rr1'
)
RR_OUTPUT = RR_PIPELINE_ROOT / 'runs/route-remediation-rr1'
RR_CODE_REVISION_LOCK = RR_PIPELINE_ROOT / 'code_revision.txt'
SAME_RUNTIME_SUMMARY = Path(
    '/content/drive/MyDrive/plasticity-p0d/route-remediation-same-runtime/v1/'
    'pipelines/pipeline-sr1/runs/'
    'code-14fa450281_rr-58c0c6b5a6_settings-1914344695/'
    'same-runtime-sr1/summary.json'
)

# New CPR-v1 namespace. Change either attempt name to start a genuinely new run.
CPR_PIPELINE_ATTEMPT = 'pipeline-cpr1'
CPR_ATTEMPT = 'cpr1'
QUALIFICATION_ATTEMPT = 'q1'
CPR_PIPELINE_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/'
    'composition-preserving-remediation/v1/pipelines'
) / CPR_PIPELINE_ATTEMPT

# Safe defaults: plan only. Enable later stages one at a time after inspecting output.
RUN_PLAN = True
RUN_AUTHORIZE = False
RUN_TRAIN = False
RUN_QUALIFY = False
APPROVER = ''

for name, value in {{
    'CPR_PIPELINE_ATTEMPT': CPR_PIPELINE_ATTEMPT,
    'CPR_ATTEMPT': CPR_ATTEMPT,
    'QUALIFICATION_ATTEMPT': QUALIFICATION_ATTEMPT,
}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
if sum(bool(value) for value in (RUN_AUTHORIZE, RUN_TRAIN, RUN_QUALIFY)) > 1:
    raise ValueError('Enable at most one mutating stage per notebook pass')
print('RR1 source:', RR_OUTPUT)
print('Same-runtime evidence:', SAME_RUNTIME_SUMMARY)
print('CPR pipeline:', CPR_PIPELINE_ROOT)
"""


CHECKOUT = """
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

CPR_PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
CODE_REVISION_LOCK = CPR_PIPELINE_ROOT / 'code_revision.txt'
locked_revision = CODE_REVISION_LOCK.read_text().strip() if CODE_REVISION_LOCK.exists() else None
subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', 'origin', BRANCH], check=True)
revision_ref = REQUESTED_CODE_REVISION or locked_revision or f'origin/{BRANCH}'
CODE_REVISION = subprocess.run(
    ['git', '-C', str(REPO_DIR), 'rev-parse', f'{revision_ref}^{{commit}}'],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if locked_revision and locked_revision != CODE_REVISION:
    raise RuntimeError('CPR code lock mismatch; use a new CPR_PIPELINE_ATTEMPT')
if not locked_revision:
    temporary = CODE_REVISION_LOCK.with_suffix('.txt.tmp')
    temporary.write_text(CODE_REVISION + '\\n')
    temporary.replace(CODE_REVISION_LOCK)
subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', '--detach', CODE_REVISION], check=True)
subprocess.run(['uv', 'sync', '--extra', 'train', '--extra', 'colab'], cwd=REPO_DIR, check=True)
subprocess.run(['nvidia-smi'], check=True)

environment = subprocess.run(
    ['uv', 'run', 'plasticity-p0d2hcpr', 'environment'],
    cwd=REPO_DIR,
    check=True,
    capture_output=True,
    text=True,
)
environment_payload = json.loads(environment.stdout.splitlines()[-1])
if environment_payload['environment']['cuda_available'] is not True:
    raise RuntimeError('CPR training and qualification require a CUDA runtime')
print('Code revision:', CODE_REVISION)
print('Environment fingerprint:', environment_payload['fingerprint'])
"""


IDENTITY = """
required = [
    RR_OUTPUT / 'manifest.json',
    RR_CODE_REVISION_LOCK,
    SAME_RUNTIME_SUMMARY,
    SAME_RUNTIME_SUMMARY.parent / 'audit_manifest.json',
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f'Missing source artifacts: {missing}')
rr_manifest = json.loads((RR_OUTPUT / 'manifest.json').read_text())
same_runtime = json.loads(SAME_RUNTIME_SUMMARY.read_text())
if (
    rr_manifest.get('schema_version') != 'p0d2hrr-manifest-v1'
    or rr_manifest.get('state') != 'verified'
):
    raise RuntimeError('RR1 source is not a verified route-remediation run')
if same_runtime.get('schema_version') != 'p0d2hrr-same-runtime-audit-v1':
    raise RuntimeError('Same-runtime source schema changed')
if same_runtime.get('decision', {}).get('mappings_per_adapter_authorized') is not False:
    raise RuntimeError('Unexpected 1/4/8 authorization in source evidence')
if same_runtime.get('runtime_identity', {}).get('sentinel', {}).get('passed') is not True:
    raise RuntimeError('Source OFF replay sentinel did not pass')

spec_path = REPO_DIR / 'configs/p0d2hcpr-composition-preserving-remediation-v1.json'
spec_hash = sha256(spec_path.read_bytes()).hexdigest()
same_runtime_hash = sha256(SAME_RUNTIME_SUMMARY.read_bytes()).hexdigest()
identity_dir = (
    f'code-{CODE_REVISION[:10]}_source-{same_runtime_hash[:10]}_settings-{spec_hash[:10]}'
)
CPR_OUTPUT = CPR_PIPELINE_ROOT / 'runs' / identity_dir / ('composition-remediation-' + CPR_ATTEMPT)
QUALIFICATION_OUTPUT = (
    CPR_PIPELINE_ROOT / 'qualifications' / identity_dir
    / ('same-runtime-' + QUALIFICATION_ATTEMPT)
)
APPROVAL_PATH = (
    CPR_PIPELINE_ROOT / 'approvals' / identity_dir
    / f'authorization-{CPR_ATTEMPT}.json'
)
print('CPR output:', CPR_OUTPUT)
print('Qualification output:', QUALIFICATION_OUTPUT)
"""


PLAN = """
def run(command):
    completed = subprocess.run(command, cwd=REPO_DIR, check=True, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout)
    return completed

if RUN_PLAN:
    run([
        'uv', 'run', 'plasticity-p0d2hcpr', 'plan',
        '--output', str(CPR_OUTPUT),
        '--source-output', str(RR_OUTPUT),
        '--same-runtime-summary', str(SAME_RUNTIME_SUMMARY),
        '--source-code-revision-lock', str(RR_CODE_REVISION_LOCK),
        '--spec', str(spec_path),
    ])
manifest = json.loads((CPR_OUTPUT / 'manifest.json').read_text())
audit = json.loads((CPR_OUTPUT / 'preflight/data_audit.json').read_text())
if audit.get('all_checks_passed') is not True:
    raise RuntimeError('CPR data preflight failed')
print('State:', manifest['state'])
print('Run ID:', manifest['run_id'])
print('Data balance:', json.dumps(audit['balance'], indent=2))
"""


AUTHORIZE = """
if RUN_AUTHORIZE:
    if not APPROVER.strip():
        raise ValueError('Set APPROVER before enabling RUN_AUTHORIZE')
    manifest = json.loads((CPR_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] == 'planned':
        template = json.loads((CPR_OUTPUT / 'authorization.template.json').read_text())
        template.update({
            'decision': 'approved',
            'approved_by': APPROVER.strip(),
            'approved_at': datetime.now(timezone.utc).isoformat(),
        })
        APPROVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if APPROVAL_PATH.exists() and json.loads(APPROVAL_PATH.read_text()) != template:
            raise RuntimeError('Existing external approval differs; use a new attempt')
        if not APPROVAL_PATH.exists():
            temporary = APPROVAL_PATH.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(template, indent=2, sort_keys=True) + '\\n')
            temporary.replace(APPROVAL_PATH)
        run([
            'uv', 'run', 'plasticity-p0d2hcpr', 'authorize',
            '--output', str(CPR_OUTPUT), '--authorization', str(APPROVAL_PATH),
        ])
    else:
        print('Authorization skipped; current state:', manifest['state'])
"""


TRAIN = """
if RUN_TRAIN:
    manifest = json.loads((CPR_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] == 'authorized':
        run(['uv', 'run', 'plasticity-p0d2hcpr', 'train', '--output', str(CPR_OUTPUT)])
    elif manifest['state'] != 'trained':
        raise RuntimeError(f'Training requires authorized state, found {manifest["state"]}')
    else:
        print('Training already complete; adapter is immutable')
"""


QUALIFY = """
if RUN_QUALIFY:
    if QUALIFICATION_OUTPUT.exists():
        if not (QUALIFICATION_OUTPUT / 'qualification_manifest.json').is_file():
            raise RuntimeError('Incomplete qualification output exists; use a new attempt')
        print('Qualification already complete; not rerunning locked panel')
    else:
        run([
            'uv', 'run', 'plasticity-p0d2hcpr', 'qualify',
            '--output', str(CPR_OUTPUT),
            '--analysis-output', str(QUALIFICATION_OUTPUT),
        ])
"""


RESULTS = """
if (QUALIFICATION_OUTPUT / 'summary.json').is_file():
    summary = json.loads((QUALIFICATION_OUTPUT / 'summary.json').read_text())
    display(Markdown((QUALIFICATION_OUTPUT / 'report.md').read_text()))
    print(json.dumps({
        'summary_path': str(QUALIFICATION_OUTPUT / 'summary.json'),
        'decision': summary['decision']['status'],
        'checks': summary['decision']['checks'],
        'historical_gate_changed': summary['historical_gate_changed'],
        'mappings_per_adapter_authorized': summary['mappings_per_adapter_authorized'],
    }, indent=2))
else:
    print('No qualification result yet. Run plan, authorize, train, and qualify in that order.')
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            markdown(
                f"""
                # P0-D2H-CPR-v1: composition-preserving remediation

                [Open this notebook in Google Colab]({COLAB_URL})

                This notebook runs one preregistered fresh LoRA with equal route-only,
                retrieval-only, and combined rehearsal, followed by one locked same-runtime
                OFF→ON qualification. Safe defaults execute preflight only. Enable one later
                stage per pass. No outcome automatically authorizes 1/4/8.
                """
            ),
            code(CONFIG),
            code(CHECKOUT),
            code(IDENTITY),
            code(PLAN),
            markdown(
                """
                Inspect the preregistration and data audit before authorization. Authorization
                is a separate explicit action bound to the exact preregistration hash. Set a
                real `APPROVER`, enable only `RUN_AUTHORIZE`, and rerun the notebook.
                """
            ),
            code(AUTHORIZE),
            code(TRAIN),
            code(QUALIFY),
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
