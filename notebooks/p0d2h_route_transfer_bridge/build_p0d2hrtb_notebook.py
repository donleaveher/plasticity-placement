from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_route_transfer_bridge_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/p0d2h_route_transfer_bridge/{NOTEBOOK_NAME}"
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
REQUESTED_CODE_REVISION = 'dd004f54e39598e4b67ae525bb46a38c293378b1'
REPO_DIR = Path('/content/plasticity-placement-route-transfer')

CPR_OUTPUT = Path(
    '/content/drive/MyDrive/plasticity-p0d/composition-preserving-remediation/v1/'
    'pipelines/pipeline-cpr2/runs/'
    'code-5a65f32042_source-8add2ee848_settings-9d697268a0/'
    'composition-remediation-cpr2'
)
QUALIFICATION_SUMMARY = Path(
    '/content/drive/MyDrive/plasticity-p0d/'
    'composition-preserving-remediation-qualification-recovery/v1/'
    'pipelines/pipeline-cpr2-qfix1/qualifications/'
    'code-b4dcbdf952_claim-4bf494dab9/same-runtime-q2/summary.json'
)
PIPELINE_ATTEMPT = 'pipeline-rtb1'
AUDIT_ATTEMPT = 'rtb1'
PIPELINE_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/route-transfer-bridge/v1/pipelines'
) / PIPELINE_ATTEMPT

# Safe defaults: preregister only. Enable at most one later stage per pass.
RUN_PLAN = True
RUN_AUTHORIZE = False
RUN_AUDIT = False
APPROVER = ''  # Example: 'Your Name'

for name, value in {{'PIPELINE_ATTEMPT': PIPELINE_ATTEMPT, 'AUDIT_ATTEMPT': AUDIT_ATTEMPT}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
if sum(bool(value) for value in (RUN_AUTHORIZE, RUN_AUDIT)) > 1:
    raise ValueError('Enable at most one post-plan stage per notebook pass')
print('Frozen CPR adapter:', CPR_OUTPUT)
print('Closed q2 evidence:', QUALIFICATION_SUMMARY)
print('Bridge pipeline:', PIPELINE_ROOT)
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

PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
CODE_REVISION_LOCK = PIPELINE_ROOT / 'code_revision.txt'
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
    raise RuntimeError('Route-transfer code lock mismatch; use a new pipeline attempt')
if not locked_revision:
    temporary = CODE_REVISION_LOCK.with_suffix('.txt.tmp')
    temporary.write_text(CODE_REVISION + '\\n')
    temporary.replace(CODE_REVISION_LOCK)
subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', '--detach', CODE_REVISION], check=True)
subprocess.run(['uv', 'sync', '--extra', 'train', '--extra', 'colab'], cwd=REPO_DIR, check=True)
subprocess.run(['nvidia-smi'], check=True)

environment = subprocess.run(
    ['uv', 'run', 'plasticity-p0d2hrtb', 'environment'],
    cwd=REPO_DIR,
    check=True,
    capture_output=True,
    text=True,
)
environment_payload = json.loads(environment.stdout.splitlines()[-1])
if environment_payload['environment']['cuda_available'] is not True:
    raise RuntimeError('Route-transfer audit requires a CUDA runtime')
print('Code revision:', CODE_REVISION)
print('Environment fingerprint:', environment_payload['fingerprint'])
"""


IDENTITY = """
required = [
    CPR_OUTPUT / 'manifest.json',
    CPR_OUTPUT / 'adapter' / 'training_metadata.json',
    QUALIFICATION_SUMMARY,
    QUALIFICATION_SUMMARY.parent / 'qualification_manifest.json',
    QUALIFICATION_SUMMARY.parent / 'paired_records.jsonl',
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f'Missing frozen bridge source artifacts: {missing}')
qualification_hash = sha256(QUALIFICATION_SUMMARY.read_bytes()).hexdigest()
identity = f'code-{CODE_REVISION[:10]}_q2-{qualification_hash[:10]}'
AUDIT_OUTPUT = PIPELINE_ROOT / 'runs' / identity / ('route-transfer-' + AUDIT_ATTEMPT)
APPROVAL_PATH = PIPELINE_ROOT / 'approvals' / identity / f'authorization-{AUDIT_ATTEMPT}.json'
print('Audit output:', AUDIT_OUTPUT)
print('External approval:', APPROVAL_PATH)
"""


LIFECYCLE = """
def run(command):
    completed = subprocess.run(command, cwd=REPO_DIR, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f'Command failed with exit code {completed.returncode}: {command}')
    return completed

if RUN_PLAN:
    if (AUDIT_OUTPUT / 'manifest.json').is_file():
        print('Preregistration already exists; not regenerating the bridge bank')
    else:
        run([
            'uv', 'run', 'plasticity-p0d2hrtb', 'plan',
            '--output', str(AUDIT_OUTPUT),
            '--cpr-output', str(CPR_OUTPUT),
            '--qualification-summary', str(QUALIFICATION_SUMMARY),
        ])

if RUN_AUTHORIZE:
    if not APPROVER.strip():
        raise ValueError('Set APPROVER before enabling RUN_AUTHORIZE')
    manifest = json.loads((AUDIT_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] == 'planned':
        template = json.loads((AUDIT_OUTPUT / 'authorization.template.json').read_text())
        fixed = {
            **template,
            'decision': 'approved',
            'approved_by': APPROVER.strip(),
        }
        fixed.pop('approved_at')
        APPROVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if APPROVAL_PATH.exists():
            approval = json.loads(APPROVAL_PATH.read_text())
            differences = {
                key: {'expected': value, 'observed': approval.get(key)}
                for key, value in fixed.items()
                if approval.get(key) != value
            }
            if differences:
                raise RuntimeError(f'Existing approval differs: {differences}')
        else:
            approval = {**fixed, 'approved_at': datetime.now(timezone.utc).isoformat()}
            temporary = APPROVAL_PATH.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(approval, indent=2, sort_keys=True) + '\\n')
            temporary.replace(APPROVAL_PATH)
        run([
            'uv', 'run', 'plasticity-p0d2hrtb', 'authorize',
            '--output', str(AUDIT_OUTPUT), '--authorization', str(APPROVAL_PATH),
        ])
    else:
        print('Authorization skipped; current state:', manifest['state'])

if RUN_AUDIT:
    manifest = json.loads((AUDIT_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] == 'authorized':
        run(['uv', 'run', 'plasticity-p0d2hrtb', 'run', '--output', str(AUDIT_OUTPUT)])
    elif manifest['state'] == 'complete':
        print('Bridge audit already complete; not rerunning')
    else:
        raise RuntimeError(f'Bridge audit requires authorized state, found {manifest["state"]}')
"""


RESULTS = """
manifest_path = AUDIT_OUTPUT / 'manifest.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
if manifest and manifest['state'] == 'complete':
    summary_path = AUDIT_OUTPUT / 'summary.json'
    report_path = AUDIT_OUTPUT / 'report.md'
    audit_manifest_path = AUDIT_OUTPUT / 'audit_manifest.json'
    if not all(path.is_file() for path in (summary_path, report_path, audit_manifest_path)):
        raise RuntimeError('Complete bridge manifest is missing published artifacts')
    summary_digest = sha256(summary_path.read_bytes()).hexdigest()
    audit_manifest = json.loads(audit_manifest_path.read_text())
    expected_digests = {
        manifest['result']['summary_sha256'],
        audit_manifest['artifacts']['summary'],
    }
    if expected_digests != {summary_digest}:
        raise RuntimeError('Bridge summary hash does not match complete manifests')
    summary = json.loads(summary_path.read_text())
    display(Markdown(report_path.read_text()))
    print(json.dumps({
        'summary_path': str(summary_path),
        'decision': summary['analysis']['decision']['status'],
        'supported_barriers': summary['analysis']['supported_barriers'],
        'primary_rescue_contrasts': summary['analysis']['primary_rescue_contrasts'],
        'historical_gate_changed': summary['historical_gate_changed'],
        'training_authorized': summary['training_authorized'],
        'mappings_per_adapter_authorized': summary['mappings_per_adapter_authorized'],
    }, indent=2))
elif manifest:
    print('No displayable result; bridge manifest state:', manifest['state'])
else:
    print('No bridge plan yet. Inspect, authorize, and run in separate passes.')
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            markdown(
                f"""
                # P0-D2H route-transfer bridge audit

                [Open this notebook in Google Colab]({COLAB_URL})

                This post-failure, inference-only audit tests why the frozen cpr2 adapter reaches
                99.22% on route-only but fails external conditional routing. It preregisters a
                2×2×2 prompt bridge anchored to exact CRD route-only rows and two action endpoints,
                then scores adapter OFF→ON in one loaded runtime. Raw rows are checkpointed before
                analysis. It never trains, retries CPR qualification, or authorizes 1/4/8.
                """
            ),
            code(CONFIG),
            code("\n\n".join((CHECKOUT, IDENTITY, LIFECYCLE))),
            markdown(
                """
                Run the combined setup/audit cell with safe defaults and inspect
                `preregistration.json`, `preflight/bank_audit.json`, and
                `authorization.template.json`. Then set a real `APPROVER` in the configuration
                cell, enable only `RUN_AUTHORIZE`, rerun the configuration and combined cell.
                Finally disable it, enable only `RUN_AUDIT`, and rerun both cells to execute the
                single frozen audit.
                """
            ),
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
