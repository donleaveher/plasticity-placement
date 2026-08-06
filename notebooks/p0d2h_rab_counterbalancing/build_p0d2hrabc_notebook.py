from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_rab_counterbalancing_colab.ipynb"
CODE_REVISION = "8fc2c1bb0c762690cbff19aaa9238781a50a5d1d"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/p0d2h_rab_counterbalancing/{NOTEBOOK_NAME}"
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
CODE_REVISION = {CODE_REVISION!r}
REPO_DIR = Path('/content/plasticity-placement-rab-counterbalancing')

RAB_OUTPUT = Path(
    '/content/drive/MyDrive/plasticity-p0d/receipt-action-binding/v1/'
    'pipelines/pipeline-rab1/runs/'
    'code-25f72d5f53_rsh-4d405f751a/receipt-action-binding-rab1'
)
PIPELINE_ATTEMPT = 'pipeline-rabc1'
AUDIT_ATTEMPT = 'rabc1'
PIPELINE_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/rab-counterbalancing/v1/pipelines'
) / PIPELINE_ATTEMPT

for name, value in {{'PIPELINE_ATTEMPT': PIPELINE_ATTEMPT, 'AUDIT_ATTEMPT': AUDIT_ATTEMPT}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
print('Frozen completed RAB source:', RAB_OUTPUT)
print('Counterbalancing pipeline:', PIPELINE_ROOT)
"""


CONTROLS = """
# EDIT ONLY THIS CELL BETWEEN PASSES.
RUN_PLAN = True
RUN_AUTHORIZE = False
RUN_AUDIT = False
APPROVER = ''  # Human/responsible-party identifier; never a password or token.

stages = {'plan': RUN_PLAN, 'authorize': RUN_AUTHORIZE, 'audit': RUN_AUDIT}
if sum(bool(value) for value in stages.values()) > 1:
    raise ValueError(f'Enable at most one stage per pass: {stages}')
if RUN_AUTHORIZE and not APPROVER.strip():
    raise ValueError('Set APPROVER to a human identifier before authorization')
print('Selected controls:', stages)
"""


SETUP = """
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
revision_ref = CODE_REVISION or locked_revision or f'origin/{BRANCH}'
RESOLVED_CODE_REVISION = subprocess.run(
    ['git', '-C', str(REPO_DIR), 'rev-parse', f'{revision_ref}^{{commit}}'],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if locked_revision and locked_revision != RESOLVED_CODE_REVISION:
    raise RuntimeError('Counterbalancing code lock mismatch; use a new pipeline attempt')
if not locked_revision:
    temporary = CODE_REVISION_LOCK.with_suffix('.txt.tmp')
    temporary.write_text(RESOLVED_CODE_REVISION + '\\n')
    temporary.replace(CODE_REVISION_LOCK)
subprocess.run(
    ['git', '-C', str(REPO_DIR), 'checkout', '--detach', RESOLVED_CODE_REVISION], check=True
)
subprocess.run(
    ['uv', 'sync', '--extra', 'train', '--extra', 'colab'], cwd=REPO_DIR, check=True
)
if RUN_AUDIT:
    subprocess.run(['nvidia-smi'], check=True)

environment = subprocess.run(
    ['uv', 'run', 'plasticity-p0d2hrabc', 'environment'],
    cwd=REPO_DIR,
    check=True,
    capture_output=True,
    text=True,
)
environment_payload = json.loads(environment.stdout.splitlines()[-1])
if RUN_AUDIT and environment_payload['environment']['cuda_available'] is not True:
    raise RuntimeError('RAB counterbalancing audit requires a CUDA runtime')
print('Code revision:', RESOLVED_CODE_REVISION)
print('Environment fingerprint:', environment_payload['fingerprint'])
"""


IDENTITY = """
required = (
    RAB_OUTPUT / 'manifest.json',
    RAB_OUTPUT / 'preregistration.json',
    RAB_OUTPUT / 'summary.json',
    RAB_OUTPUT / 'paired_records.jsonl',
    RAB_OUTPUT / 'results' / 'adapter_off.jsonl',
    RAB_OUTPUT / 'results' / 'adapter_on.jsonl',
    RAB_OUTPUT / 'raw_score_checkpoint.json',
    RAB_OUTPUT / 'audit_manifest.json',
    RAB_OUTPUT / 'preflight' / 'binding_probes.jsonl',
    RAB_OUTPUT / 'preflight' / 'bank_audit.json',
    RAB_OUTPUT / 'preflight' / 'token_audit.json',
)
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f'Missing frozen RAB artifacts: {missing}')
rab_hash = sha256((RAB_OUTPUT / 'summary.json').read_bytes()).hexdigest()
identity = f'code-{RESOLVED_CODE_REVISION[:10]}_rab-{rab_hash[:10]}'
AUDIT_OUTPUT = PIPELINE_ROOT / 'runs' / identity / (
    'rab-counterbalancing-' + AUDIT_ATTEMPT
)
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

def write_or_validate_approval(path, fixed):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        approval = json.loads(path.read_text())
        differences = {
            key: {'expected': value, 'observed': approval.get(key)}
            for key, value in fixed.items()
            if approval.get(key) != value
        }
        if differences:
            raise RuntimeError(f'Existing approval differs: {differences}')
        return approval
    approval = {**fixed, 'approved_at': datetime.now(timezone.utc).isoformat()}
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(approval, indent=2, sort_keys=True) + '\\n')
    temporary.replace(path)
    return approval

if RUN_PLAN:
    if (AUDIT_OUTPUT / 'manifest.json').is_file():
        print('Preregistration already exists; not rebuilding counterbalanced bank')
    else:
        run([
            'uv', 'run', 'plasticity-p0d2hrabc', 'plan',
            '--output', str(AUDIT_OUTPUT), '--rab-output', str(RAB_OUTPUT),
        ])
    bank_audit = AUDIT_OUTPUT / 'preflight' / 'bank_audit.json'
    if bank_audit.is_file():
        payload = json.loads(bank_audit.read_text())
        print(json.dumps(
            {
                'prompt_condition_count': payload['prompt_condition_count'],
                'prompt_text_count': payload['prompt_text_count'],
                'prompt_text_multiplicity_counts': payload[
                    'prompt_text_multiplicity_counts'
                ],
                'all_checks_passed': payload['all_checks_passed'],
            },
            indent=2,
        ))

if RUN_AUTHORIZE:
    manifest = json.loads((AUDIT_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] == 'planned':
        template = json.loads((AUDIT_OUTPUT / 'authorization.template.json').read_text())
        fixed = {**template, 'decision': 'approved', 'approved_by': APPROVER.strip()}
        fixed.pop('approved_at')
        write_or_validate_approval(APPROVAL_PATH, fixed)
    elif manifest['state'] == 'authorized':
        if not APPROVAL_PATH.is_file():
            raise RuntimeError('Authorized manifest is missing its external approval')
        print('Authorization already adopted; validating idempotent completion')
    else:
        raise RuntimeError(f'Authorization is invalid from state: {manifest["state"]}')
    run([
        'uv', 'run', 'plasticity-p0d2hrabc', 'authorize',
        '--output', str(AUDIT_OUTPUT), '--authorization', str(APPROVAL_PATH),
    ])

if RUN_AUDIT:
    manifest = json.loads((AUDIT_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] == 'authorized':
        run([
            'uv', 'run', 'plasticity-p0d2hrabc', 'run',
            '--output', str(AUDIT_OUTPUT),
        ])
    elif manifest['state'] == 'complete':
        print('RAB counterbalancing audit already complete; not rerunning')
    else:
        raise RuntimeError(f'Counterbalancing audit requires authorized state: {manifest["state"]}')
"""


RESULTS = """
manifest_path = AUDIT_OUTPUT / 'manifest.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
if manifest and manifest['state'] == 'complete':
    run(['uv', 'run', 'plasticity-p0d2hrabc', 'verify', '--output', str(AUDIT_OUTPUT)])
    summary_path = AUDIT_OUTPUT / 'summary.json'
    report_path = AUDIT_OUTPUT / 'report.md'
    audit_manifest_path = AUDIT_OUTPUT / 'audit_manifest.json'
    required_paths = (
        summary_path,
        report_path,
        audit_manifest_path,
        AUDIT_OUTPUT / 'paired_records.jsonl',
        AUDIT_OUTPUT / 'results' / 'adapter_off.jsonl',
        AUDIT_OUTPUT / 'results' / 'adapter_on.jsonl',
        AUDIT_OUTPUT / 'raw_score_checkpoint.json',
    )
    if not all(path.is_file() for path in required_paths):
        raise RuntimeError('Complete counterbalancing manifest is missing published artifacts')
    summary_digest = sha256(summary_path.read_bytes()).hexdigest()
    audit_manifest_digest = sha256(audit_manifest_path.read_bytes()).hexdigest()
    audit_manifest = json.loads(audit_manifest_path.read_text())
    if {manifest['result']['summary_sha256'], audit_manifest['artifacts']['summary']} != {
        summary_digest
    }:
        raise RuntimeError('Counterbalancing summary hash does not match complete manifests')
    if manifest['result']['audit_manifest_sha256'] != audit_manifest_digest:
        raise RuntimeError('Counterbalancing audit-manifest hash does not match complete manifest')
    summary = json.loads(summary_path.read_text())
    display(Markdown(report_path.read_text()))
    print(json.dumps(
        {
            'summary_path': str(summary_path),
            'analysis_status': summary['analysis']['analysis_status'],
            'causal_attribution': summary['analysis']['causal_attribution'],
            'main_effects': summary['analysis']['main_effects'],
            'interactions': summary['analysis']['interactions'],
            'tie_diagnostics': summary['analysis']['tie_diagnostics'],
            'historical_rab_decision_changed': summary[
                'historical_rab_decision_changed'
            ],
            'training_authorized': summary['training_authorized'],
            'mappings_per_adapter_authorized': summary[
                'mappings_per_adapter_authorized'
            ],
        },
        indent=2,
    ))
elif manifest:
    print('No displayable result; counterbalancing manifest state:', manifest['state'])
else:
    print('No counterbalancing plan yet. Plan, inspect, authorize, and run in separate passes.')
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            markdown(
                f"""
                # P0-D2H frozen RAB counterbalancing audit

                [Open this notebook in Google Colab]({COLAB_URL})

                This audit keeps the exact completed RAB units and frozen adapter, then crosses
                receipt/mapping, slot display order, and four action-panel rotations. It scores
                every condition OFF→ON in one runtime, propagates ties without using candidate
                order, and separates joint receipt/slot-label, serial-position,
                candidate-position, and interaction signatures. It never trains, reclassifies
                RAB, or authorizes 1/4/8.
                """
            ),
            code(CONFIG),
            code(CONTROLS),
            code("\n\n".join((SETUP, IDENTITY, LIFECYCLE))),
            markdown(
                """
                Edit only the controls cell. First keep the safe defaults and run the
                configuration, controls, and combined lifecycle cell. Inspect
                `preregistration.json`, `preflight/bank_audit.json`,
                `preflight/token_audit.json`, `preflight/analysis_plan.json`, and
                `authorization.template.json`.

                Next set `RUN_PLAN = False`, set a real human `APPROVER`, enable only
                `RUN_AUTHORIZE`, and rerun the controls and combined lifecycle cell. Finally
                disable authorization, enable only `RUN_AUDIT`, and rerun those two cells. Run
                the result cell only after completion.
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
