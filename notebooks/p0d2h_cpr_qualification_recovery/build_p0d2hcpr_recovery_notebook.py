from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_cpr_qualification_recovery_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/"
    "p0d2h_cpr_qualification_recovery/"
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
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from IPython.display import Markdown, display

REPO_URL = 'https://github.com/donleaveher/plasticity-placement.git'
BRANCH = '{BRANCH}'
REQUESTED_CODE_REVISION = None
REPO_DIR = Path('/content/plasticity-placement-cpr-qfix')

# Exact immutable cpr2 adapter whose q1 inference completed before classification failed.
CPR_OUTPUT = Path(
    '/content/drive/MyDrive/plasticity-p0d/composition-preserving-remediation/v1/'
    'pipelines/pipeline-cpr2/runs/'
    'code-5a65f32042_source-8add2ee848_settings-9d697268a0/'
    'composition-remediation-cpr2'
)
RECOVERY_PIPELINE_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/'
    'composition-preserving-remediation-qualification-recovery/v1/'
    'pipelines/pipeline-cpr2-qfix1'
)

# Safe default: inspect only. To authorize q2, set your name and flip this once.
RUN_RECOVERY = False
APPROVER = ''  # Example: 'Your Name'

print('Frozen CPR output:', CPR_OUTPUT)
print('Recovery namespace:', RECOVERY_PIPELINE_ROOT)
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

RECOVERY_PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
CODE_REVISION_LOCK = RECOVERY_PIPELINE_ROOT / 'analysis_code_revision.txt'
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
    raise RuntimeError('Recovery analysis code lock mismatch')
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
    raise RuntimeError('Qualification recovery requires a CUDA runtime')
ANALYSIS_CODE_SHA256 = environment_payload['environment']['code_sha256']
print('Recovery code revision:', CODE_REVISION)
print('Recovery analysis code SHA-256:', ANALYSIS_CODE_SHA256)
"""


PREFLIGHT = """
required = [
    CPR_OUTPUT / 'manifest.json',
    CPR_OUTPUT / 'qualification_claim.json',
    CPR_OUTPUT / 'adapter' / 'training_metadata.json',
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f'Missing frozen cpr2 artifacts: {missing}')

manifest = json.loads((CPR_OUTPUT / 'manifest.json').read_text())
claim_path = CPR_OUTPUT / 'qualification_claim.json'
claim = json.loads(claim_path.read_text())
if manifest.get('state') != 'trained':
    raise RuntimeError(f'Expected trained cpr2 manifest, found {manifest.get("state")!r}')
if claim.get('schema_version') != 'p0d2hcpr-qualification-claim-v1':
    raise RuntimeError('Original q1 claim schema changed')
if claim.get('adapter_sha256') != manifest['training']['summary']['adapter_sha256']:
    raise RuntimeError('Original q1 claim does not bind the trained adapter')

ORIGINAL_OUTPUT = Path(claim['analysis_output']).resolve()
if ORIGINAL_OUTPUT.exists():
    raise RuntimeError(
        f'Original q1 result artifacts exist at {ORIGINAL_OUTPUT}; recovery is forbidden'
    )
ORIGINAL_CLAIM_SHA256 = sha256(claim_path.read_bytes()).hexdigest()
identity = f'code-{CODE_REVISION[:10]}_claim-{ORIGINAL_CLAIM_SHA256[:10]}'
RECOVERY_OUTPUT = RECOVERY_PIPELINE_ROOT / 'qualifications' / identity / 'same-runtime-q2'
APPROVAL_PATH = RECOVERY_PIPELINE_ROOT / 'approvals' / identity / 'authorization-q2.json'

existing_recovery_claim = CPR_OUTPUT / 'qualification_recovery_claim.json'
if existing_recovery_claim.exists() and not (RECOVERY_OUTPUT / 'summary.json').is_file():
    raise RuntimeError('A q2 recovery claim already exists but its expected summary is absent')

print('Original failed output:', ORIGINAL_OUTPUT)
print('Original claim SHA-256:', ORIGINAL_CLAIM_SHA256)
print('Recovery output:', RECOVERY_OUTPUT)
print('Recovery approval:', APPROVAL_PATH)
"""


RECOVER = """
def run(command):
    completed = subprocess.run(command, cwd=REPO_DIR, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f'Command failed with exit code {completed.returncode}: {command}'
        )
    return completed

if RUN_RECOVERY:
    if not APPROVER.strip():
        raise ValueError('Set APPROVER to your name before enabling RUN_RECOVERY')
    if RECOVERY_OUTPUT.exists():
        if not (RECOVERY_OUTPUT / 'qualification_manifest.json').is_file():
            raise RuntimeError('Incomplete q2 output exists; do not retry or delete it')
        print('q2 qualification recovery already complete; not rerunning')
    else:
        authorization_fields = {
            'schema_version': 'p0d2hcpr-qualification-recovery-authorization-v1',
            'decision': 'approved',
            'scope': 'single_post_inference_classification_failure_recovery',
            'failure_signature': 'KeyError:accuracy_interval_after_complete_scoring',
            'original_claim_sha256': ORIGINAL_CLAIM_SHA256,
            'original_analysis_output': str(ORIGINAL_OUTPUT),
            'recovery_analysis_output': str(RECOVERY_OUTPUT.resolve()),
            'analysis_code_sha256': ANALYSIS_CODE_SHA256,
            'allowed_recovery_runs': 1,
            'allows_training': False,
            'allows_mappings_per_adapter_scan': False,
            'approved_by': APPROVER.strip(),
        }
        APPROVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if APPROVAL_PATH.exists():
            authorization = json.loads(APPROVAL_PATH.read_text())
            differences = {
                key: {'expected': value, 'observed': authorization.get(key)}
                for key, value in authorization_fields.items()
                if authorization.get(key) != value
            }
            if differences:
                raise RuntimeError(f'Existing recovery approval differs: {differences}')
        else:
            authorization = {
                **authorization_fields,
                'approved_at': datetime.now(timezone.utc).isoformat(),
            }
            temporary = APPROVAL_PATH.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(authorization, indent=2, sort_keys=True) + '\\n')
            temporary.replace(APPROVAL_PATH)
        run([
            'uv', 'run', 'plasticity-p0d2hcpr', 'qualify',
            '--output', str(CPR_OUTPUT),
            '--analysis-output', str(RECOVERY_OUTPUT),
            '--recovery-authorization', str(APPROVAL_PATH),
        ])
else:
    print('Inspection-only pass. Set APPROVER and RUN_RECOVERY=True for the one q2 run.')
"""


RESULTS = """
if (RECOVERY_OUTPUT / 'summary.json').is_file():
    summary = json.loads((RECOVERY_OUTPUT / 'summary.json').read_text())
    display(Markdown((RECOVERY_OUTPUT / 'report.md').read_text()))
    print(json.dumps({
        'summary_path': str(RECOVERY_OUTPUT / 'summary.json'),
        'decision': summary['decision']['status'],
        'checks': summary['decision']['checks'],
        'qualification_recovery': summary['source']['qualification_recovery'],
        'historical_gate_changed': summary['historical_gate_changed'],
        'mappings_per_adapter_authorized': summary['mappings_per_adapter_authorized'],
    }, indent=2))
else:
    print('No q2 result yet. This notebook never trains and never authorizes 1/4/8.')
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            markdown(
                f"""
                # P0-D2H-CPR q1 classification-failure recovery

                [Open this notebook in Google Colab]({COLAB_URL})

                q1 completed all 96 forced-choice and 1,536 CRD OFF→ON scores, then failed
                before publishing any result because the classifier expected an obsolete tie
                field. This notebook performs one explicitly authorized q2 qualification with
                the same immutable cpr2 adapter and locked panel. It does not train, tune, or
                run 1/4/8.
                """
            ),
            code(CONFIG),
            code(CHECKOUT),
            code(PREFLIGHT),
            markdown(
                """
                Inspect the paths and hashes above. To approve the single infrastructure
                recovery, set `APPROVER` to your name and `RUN_RECOVERY = True` in the first
                cell, then run all cells. The authorization is bound to q1's immutable claim,
                q2's output path, and the fixed analysis-code hash.
                """
            ),
            code(RECOVER),
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
