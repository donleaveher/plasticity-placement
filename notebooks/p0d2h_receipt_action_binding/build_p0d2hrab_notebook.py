from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_receipt_action_binding_colab.ipynb"
CODE_REVISION = "25f72d5f53010b32a3b4ebb569dbd1863b7aeaa0"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/p0d2h_receipt_action_binding/{NOTEBOOK_NAME}"
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
REPO_DIR = Path('/content/plasticity-placement-receipt-action-binding')

RSH_OUTPUT = Path(
    '/content/drive/MyDrive/plasticity-p0d/route-state-handoff/v1/'
    'pipelines/pipeline-rsh1/runs/'
    'code-b536c34c2b_rtb-d3cc84c2ed/route-state-handoff-rsh1'
)
PIPELINE_ATTEMPT = 'pipeline-rab1'
AUDIT_ATTEMPT = 'rab1'
PIPELINE_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/receipt-action-binding/v1/pipelines'
) / PIPELINE_ATTEMPT

for name, value in {{'PIPELINE_ATTEMPT': PIPELINE_ATTEMPT, 'AUDIT_ATTEMPT': AUDIT_ATTEMPT}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
print('Frozen RSH source:', RSH_OUTPUT)
print('Receipt/action pipeline:', PIPELINE_ROOT)
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
    raise RuntimeError('Receipt/action code lock mismatch; use a new pipeline attempt')
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
    ['uv', 'run', 'plasticity-p0d2hrab', 'environment'],
    cwd=REPO_DIR,
    check=True,
    capture_output=True,
    text=True,
)
environment_payload = json.loads(environment.stdout.splitlines()[-1])
if RUN_AUDIT and environment_payload['environment']['cuda_available'] is not True:
    raise RuntimeError('Receipt/action binding audit requires a CUDA runtime')
print('Code revision:', RESOLVED_CODE_REVISION)
print('Environment fingerprint:', environment_payload['fingerprint'])
"""


IDENTITY = """
required = [
    RSH_OUTPUT / 'manifest.json',
    RSH_OUTPUT / 'summary.json',
    RSH_OUTPUT / 'paired_records.jsonl',
    RSH_OUTPUT / 'audit_manifest.json',
    RSH_OUTPUT / 'preflight' / 'handoff_probes.jsonl',
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f'Missing frozen RSH artifacts: {missing}')
rsh_hash = sha256((RSH_OUTPUT / 'summary.json').read_bytes()).hexdigest()
identity = f'code-{RESOLVED_CODE_REVISION[:10]}_rsh-{rsh_hash[:10]}'
AUDIT_OUTPUT = PIPELINE_ROOT / 'runs' / identity / ('receipt-action-binding-' + AUDIT_ATTEMPT)
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
        print('Preregistration already exists; not rebuilding receipt/action bank')
    else:
        run([
            'uv', 'run', 'plasticity-p0d2hrab', 'plan',
            '--output', str(AUDIT_OUTPUT), '--rsh-output', str(RSH_OUTPUT),
        ])
    quadrant_report = AUDIT_OUTPUT / 'preflight' / 'rsh_quadrant_diagnostic.md'
    if quadrant_report.is_file():
        display(Markdown(quadrant_report.read_text()))

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
        'uv', 'run', 'plasticity-p0d2hrab', 'authorize',
        '--output', str(AUDIT_OUTPUT), '--authorization', str(APPROVAL_PATH),
    ])

if RUN_AUDIT:
    manifest = json.loads((AUDIT_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] == 'authorized':
        run(['uv', 'run', 'plasticity-p0d2hrab', 'run', '--output', str(AUDIT_OUTPUT)])
    elif manifest['state'] == 'complete':
        print('Receipt/action audit already complete; not rerunning')
    else:
        raise RuntimeError(
            f'Receipt/action audit requires authorized state, found {manifest["state"]}'
        )
"""


RESULTS = """
manifest_path = AUDIT_OUTPUT / 'manifest.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
if manifest and manifest['state'] == 'complete':
    run(['uv', 'run', 'plasticity-p0d2hrab', 'verify', '--output', str(AUDIT_OUTPUT)])
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
        raise RuntimeError('Complete receipt/action manifest is missing published artifacts')
    summary_digest = sha256(summary_path.read_bytes()).hexdigest()
    audit_manifest_digest = sha256(audit_manifest_path.read_bytes()).hexdigest()
    audit_manifest = json.loads(audit_manifest_path.read_text())
    if {
        manifest['result']['summary_sha256'],
        audit_manifest['artifacts']['summary'],
    } != {summary_digest}:
        raise RuntimeError('Receipt/action summary hash does not match complete manifests')
    if manifest['result']['audit_manifest_sha256'] != audit_manifest_digest:
        raise RuntimeError('Receipt/action audit-manifest hash does not match complete manifest')
    summary = json.loads(summary_path.read_text())
    display(Markdown(report_path.read_text()))
    print(json.dumps({
        'summary_path': str(summary_path),
        'decision': summary['analysis']['decision']['status'],
        'binding_accuracy': summary['analysis']['adapter_on_conservative_binding_accuracy'],
        'causal_specificity': summary['analysis']['adapter_on_conservative_specificity'],
        'orientation_accuracy': summary['analysis']['orientation_conservative_accuracy'],
        'noninferiority': summary['analysis']['adapter_on_minus_off_noninferiority'],
        'historical_rsh_decision_changed': summary['historical_rsh_decision_changed'],
        'training_authorized': summary['training_authorized'],
        'mappings_per_adapter_authorized': summary['mappings_per_adapter_authorized'],
    }, indent=2))
elif manifest:
    print('No displayable result; receipt/action manifest state:', manifest['state'])
else:
    print('No receipt/action plan yet. Plan, inspect, authorize, and run in separate passes.')
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            markdown(
                f"""
                # P0-D2H receipt/action binding audit

                [Open this notebook in Google Colab]({COLAB_URL})

                Planning first produces the CPU-only four-quadrant diagnostic from the completed
                RSH paired records. The frozen GPU audit then places two valid verified mappings
                in slots A/B and crosses signed receipt A/B with canonical/swapped binding. Every
                prompt is scored OFF→ON in one runtime. It never trains, reclassifies RSH, or
                authorizes 1/4/8.
                """
            ),
            code(CONFIG),
            code(CONTROLS),
            code("\n\n".join((SETUP, IDENTITY, LIFECYCLE))),
            markdown(
                """
                Edit only the controls cell. First keep safe defaults and run the configuration,
                controls, and combined cell. Inspect `preregistration.json`, the displayed CPU
                diagnostic, `preflight/rsh_quadrant_diagnostic.json`,
                `preflight/bank_audit.json`, `preflight/token_audit.json`, and
                `authorization.template.json`.

                Next set `RUN_PLAN = False`, set a real `APPROVER`, enable only
                `RUN_AUTHORIZE`, and rerun the controls and combined cell. Finally disable
                authorization, enable only `RUN_AUDIT`, and rerun those two cells. Run the result
                cell only after completion.
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
