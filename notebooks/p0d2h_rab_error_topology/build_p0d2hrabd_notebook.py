from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_rab_error_topology_colab.ipynb"
CODE_REVISION = "d850d82e81ef5b39bcb833f16f847c30274c16ba"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/p0d2h_rab_error_topology/{NOTEBOOK_NAME}"
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
from hashlib import sha256
from pathlib import Path

from IPython.display import Markdown, display

REPO_URL = 'https://github.com/donleaveher/plasticity-placement.git'
BRANCH = '{BRANCH}'
CODE_REVISION = {CODE_REVISION!r}
REPO_DIR = Path('/content/plasticity-placement-rab-error-topology')

RAB_OUTPUT = Path(
    '/content/drive/MyDrive/plasticity-p0d/receipt-action-binding/v1/'
    'pipelines/pipeline-rab1/runs/'
    'code-25f72d5f53_rsh-4d405f751a/receipt-action-binding-rab1'
)
PIPELINE_ATTEMPT = 'pipeline-rabd1'
DIAGNOSTIC_ATTEMPT = 'rabd1'
PIPELINE_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/rab-error-topology/v1/pipelines'
) / PIPELINE_ATTEMPT

for name, value in {{
    'PIPELINE_ATTEMPT': PIPELINE_ATTEMPT,
    'DIAGNOSTIC_ATTEMPT': DIAGNOSTIC_ATTEMPT,
}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
print('Frozen completed RAB source:', RAB_OUTPUT)
print('Diagnostic pipeline:', PIPELINE_ROOT)
"""


CONTROLS = """
# EDIT ONLY THIS CELL BETWEEN PASSES.
RUN_PLAN = True
RUN_DIAGNOSTIC = False

stages = {'plan': RUN_PLAN, 'diagnostic': RUN_DIAGNOSTIC}
if sum(bool(value) for value in stages.values()) > 1:
    raise ValueError(f'Enable at most one stage per pass: {stages}')
print('Selected controls:', stages)
"""


SETUP_AND_RUN = """
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
    raise RuntimeError('RAB diagnostic code lock mismatch; use a new pipeline attempt')
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

required_source = (
    RAB_OUTPUT / 'manifest.json',
    RAB_OUTPUT / 'preregistration.json',
    RAB_OUTPUT / 'summary.json',
    RAB_OUTPUT / 'paired_records.jsonl',
    RAB_OUTPUT / 'results' / 'adapter_off.jsonl',
    RAB_OUTPUT / 'results' / 'adapter_on.jsonl',
    RAB_OUTPUT / 'audit_manifest.json',
    RAB_OUTPUT / 'preflight' / 'binding_probes.jsonl',
)
missing = [str(path) for path in required_source if not path.is_file()]
if missing:
    raise FileNotFoundError(f'Missing frozen RAB artifacts: {missing}')
rab_hash = sha256((RAB_OUTPUT / 'summary.json').read_bytes()).hexdigest()
identity = f'code-{RESOLVED_CODE_REVISION[:10]}_rab-{rab_hash[:10]}'
DIAGNOSTIC_OUTPUT = PIPELINE_ROOT / 'runs' / identity / (
    'rab-error-topology-' + DIAGNOSTIC_ATTEMPT
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

manifest_path = DIAGNOSTIC_OUTPUT / 'manifest.json'
if RUN_PLAN:
    if manifest_path.is_file():
        print('Frozen diagnostic plan already exists; not regenerating it')
    else:
        run([
            'uv', 'run', 'plasticity-p0d2hrabd', 'plan',
            '--output', str(DIAGNOSTIC_OUTPUT), '--rab-output', str(RAB_OUTPUT),
        ])

if RUN_DIAGNOSTIC:
    if not manifest_path.is_file():
        raise RuntimeError('Run the planning pass before the diagnostic pass')
    manifest = json.loads(manifest_path.read_text())
    if manifest['state'] == 'planned':
        run(['uv', 'run', 'plasticity-p0d2hrabd', 'run', '--output', str(DIAGNOSTIC_OUTPUT)])
    elif manifest['state'] == 'complete':
        print('RAB error-topology diagnostic already complete; not rerunning')
    else:
        raise RuntimeError(f'Diagnostic requires planned state, found {manifest["state"]}')

print('Code revision:', RESOLVED_CODE_REVISION)
print('Diagnostic output:', DIAGNOSTIC_OUTPUT)
"""


RESULTS = """
manifest_path = DIAGNOSTIC_OUTPUT / 'manifest.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
if manifest and manifest['state'] == 'complete':
    run(['uv', 'run', 'plasticity-p0d2hrabd', 'verify', '--output', str(DIAGNOSTIC_OUTPUT)])
    summary_path = DIAGNOSTIC_OUTPUT / 'summary.json'
    report_path = DIAGNOSTIC_OUTPUT / 'report.md'
    audit_manifest_path = DIAGNOSTIC_OUTPUT / 'audit_manifest.json'
    required = (
        summary_path,
        report_path,
        audit_manifest_path,
        DIAGNOSTIC_OUTPUT / 'cell_records.jsonl',
        DIAGNOSTIC_OUTPUT / 'unit_records.jsonl',
        DIAGNOSTIC_OUTPUT / 'factor_slices.json',
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError('Complete diagnostic manifest is missing published artifacts')
    summary_digest = sha256(summary_path.read_bytes()).hexdigest()
    audit_digest = sha256(audit_manifest_path.read_bytes()).hexdigest()
    audit_manifest = json.loads(audit_manifest_path.read_text())
    if {
        manifest['result']['summary_sha256'],
        audit_manifest['artifacts']['summary'],
    } != {summary_digest}:
        raise RuntimeError('Diagnostic summary hash does not match complete manifests')
    if manifest['result']['audit_manifest_sha256'] != audit_digest:
        raise RuntimeError('Diagnostic audit-manifest hash does not match complete manifest')
    summary = json.loads(summary_path.read_text())
    display(Markdown(report_path.read_text()))
    print(json.dumps({
        'summary_path': str(summary_path),
        'analysis_status': summary['analysis']['analysis_status'],
        'correctness_transitions': summary['analysis']['correctness_transitions'],
        'unit_taxonomy': summary['analysis']['unit_taxonomy'],
        'margin_summary': summary['analysis']['margin_summary'],
        'historical_rab_decision_changed': summary['historical_rab_decision_changed'],
        'inference_authorized': summary['inference_authorized'],
        'training_authorized': summary['training_authorized'],
        'mappings_per_adapter_authorized': summary['mappings_per_adapter_authorized'],
    }, indent=2))
elif manifest:
    print('No displayable result; diagnostic manifest state:', manifest['state'])
else:
    print('No diagnostic plan yet. Run planning and analysis in separate passes.')
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            markdown(
                f"""
                # P0-D2H RAB paired error-topology diagnostic

                [Open this notebook in Google Colab]({COLAB_URL})

                This CPU-only diagnostic reads the exact completed RAB OFF/ON score records and
                reconstructs cell transitions, four-cell unit error categories,
                selected-minus-counterfactual margins, factor slices, and action bias. It does
                not load a model, perform inference, train, reclassify RAB, or authorize 1/4/8.
                """
            ),
            code(CONFIG),
            code(CONTROLS),
            code(SETUP_AND_RUN),
            markdown(
                """
                First keep the safe defaults and run through the combined setup/planning cell.
                Inspect `preregistration.json` and `preflight/analysis_plan.json`. Then edit only
                the controls cell: set `RUN_PLAN = False` and `RUN_DIAGNOSTIC = True`, rerun the
                controls and combined cell, and finally run the result cell. No approver or
                authorization file is required because this pass is deterministic post-hoc CPU
                analysis of already-published scores.
                """
            ),
            code(RESULTS),
        ],
        "metadata": {
            "colab": {"provenance": []},
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
