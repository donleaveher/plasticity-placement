from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
EXPERIMENT_CODE_REVISION = "58eaa8c4dde431e393925662b6c1e913bba02611"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_rab_label_disentanglement_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/p0d2h_rab_label_disentanglement/{NOTEBOOK_NAME}"
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
RECOVERY_CODE_REVISION = None
EXPERIMENT_REPO_DIR = Path('/content/plasticity-placement-rab-label-disentanglement')
RECOVERY_REPO_DIR = Path('/content/plasticity-placement-rab-label-recovery')

RABC_OUTPUT = Path(
    '/content/drive/MyDrive/plasticity-p0d/rab-counterbalancing/v1/pipelines/'
    'pipeline-rabc1/runs/code-8fc2c1bb0c_rab-9368c8c29c/'
    'rab-counterbalancing-rabc1'
)
PIPELINE_ATTEMPT = 'pipeline-rabx1'
AUDIT_ATTEMPT = 'rabx1'
PIPELINE_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/rab-label-disentanglement/v1/pipelines'
) / PIPELINE_ATTEMPT

for name, value in {{'PIPELINE_ATTEMPT': PIPELINE_ATTEMPT, 'AUDIT_ATTEMPT': AUDIT_ATTEMPT}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
print('Frozen completed RABC source:', RABC_OUTPUT)
print('Label-disentanglement pipeline:', PIPELINE_ROOT)
"""


CONTROLS = """
# EDIT ONLY THIS CELL BETWEEN PASSES.
RUN_PLAN = True
RUN_AUTHORIZE = False
RUN_RECOVERY_INSPECTION = False
RUN_RECOVER_ZERO_ARTIFACT = False
RUN_AUDIT = False
APPROVER = ''  # Human/responsible-party identifier; never a password or token.
ORIGINAL_RUNTIME_TERMINATED = False  # True only after confirming the old Colab process stopped.

stages = {
    'plan': RUN_PLAN,
    'authorize': RUN_AUTHORIZE,
    'recovery_inspection': RUN_RECOVERY_INSPECTION,
    'recover_zero_artifact': RUN_RECOVER_ZERO_ARTIFACT,
    'audit': RUN_AUDIT,
}
if sum(bool(value) for value in stages.values()) > 1:
    raise ValueError(f'Enable at most one stage per pass: {stages}')
if (RUN_AUTHORIZE or RUN_RECOVER_ZERO_ARTIFACT) and not APPROVER.strip():
    raise ValueError('Set APPROVER to a human identifier before authorization or recovery')
if RUN_RECOVER_ZERO_ARTIFACT and ORIGINAL_RUNTIME_TERMINATED is not True:
    raise ValueError('Confirm ORIGINAL_RUNTIME_TERMINATED before recovery')
print('Selected controls:', stages)
"""


SETUP = """
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)

def prepare_checkout(repo_dir, requested_revision, revision_lock, label):
    if repo_dir.exists() and not (repo_dir / '.git').exists():
        raise RuntimeError(f'{repo_dir} exists but is not a Git repository')
    if not repo_dir.exists():
        subprocess.run(['git', 'clone', '--branch', BRANCH, REPO_URL, str(repo_dir)], check=True)
    dirty = subprocess.run(
        ['git', '-C', str(repo_dir), 'status', '--porcelain'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f'{label} checkout has local changes:\\n{dirty}')
    subprocess.run(['git', '-C', str(repo_dir), 'fetch', 'origin', BRANCH], check=True)
    locked_revision = revision_lock.read_text().strip() if revision_lock.exists() else None
    revision_ref = requested_revision or locked_revision or f'origin/{BRANCH}'
    revision = subprocess.run(
        ['git', '-C', str(repo_dir), 'rev-parse', f'{revision_ref}^{{commit}}'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if locked_revision and locked_revision != revision:
        raise RuntimeError(f'{label} code lock mismatch; use a new pipeline attempt')
    if not locked_revision:
        temporary = revision_lock.with_suffix(revision_lock.suffix + '.tmp')
        temporary.write_text(revision + '\\n')
        temporary.replace(revision_lock)
    subprocess.run(['git', '-C', str(repo_dir), 'checkout', '--detach', revision], check=True)
    return revision

PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
EXPERIMENT_CODE_REVISION_LOCK = PIPELINE_ROOT / 'code_revision.txt'
RECOVERY_CODE_REVISION_LOCK = PIPELINE_ROOT / 'recovery_code_revision.txt'
RESOLVED_EXPERIMENT_CODE_REVISION = prepare_checkout(
    EXPERIMENT_REPO_DIR,
    EXPERIMENT_CODE_REVISION,
    EXPERIMENT_CODE_REVISION_LOCK,
    'experiment',
)
RESOLVED_RECOVERY_CODE_REVISION = prepare_checkout(
    RECOVERY_REPO_DIR,
    RECOVERY_CODE_REVISION,
    RECOVERY_CODE_REVISION_LOCK,
    'recovery',
)
subprocess.run(
    ['uv', 'sync', '--extra', 'train', '--extra', 'colab'],
    cwd=EXPERIMENT_REPO_DIR,
    check=True,
)
subprocess.run(['uv', 'sync'], cwd=RECOVERY_REPO_DIR, check=True)
if RUN_AUDIT:
    subprocess.run(['nvidia-smi'], check=True)

environment = subprocess.run(
    ['uv', 'run', 'plasticity-p0d2hrabx', 'environment'],
    cwd=EXPERIMENT_REPO_DIR,
    check=True,
    capture_output=True,
    text=True,
)
environment_payload = json.loads(environment.stdout.splitlines()[-1])
if RUN_AUDIT and environment_payload['environment']['cuda_available'] is not True:
    raise RuntimeError('RAB label-disentanglement audit requires a CUDA runtime')
print('Experiment code revision:', RESOLVED_EXPERIMENT_CODE_REVISION)
print('Recovery code revision:', RESOLVED_RECOVERY_CODE_REVISION)
print('Environment fingerprint:', environment_payload['fingerprint'])
"""


IDENTITY = """
required = (
    RABC_OUTPUT / 'manifest.json',
    RABC_OUTPUT / 'preregistration.json',
    RABC_OUTPUT / 'summary.json',
    RABC_OUTPUT / 'paired_records.jsonl',
    RABC_OUTPUT / 'results' / 'adapter_off.jsonl',
    RABC_OUTPUT / 'results' / 'adapter_on.jsonl',
    RABC_OUTPUT / 'raw_score_checkpoint.json',
    RABC_OUTPUT / 'audit_manifest.json',
    RABC_OUTPUT / 'preflight' / 'counterbalanced_probes.jsonl',
    RABC_OUTPUT / 'preflight' / 'bank_audit.json',
    RABC_OUTPUT / 'preflight' / 'token_audit.json',
    RABC_OUTPUT / 'preflight' / 'analysis_plan.json',
)
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f'Missing frozen RABC artifacts: {missing}')
rabc_hash = sha256((RABC_OUTPUT / 'summary.json').read_bytes()).hexdigest()
identity = f'code-{RESOLVED_EXPERIMENT_CODE_REVISION[:10]}_rabc-{rabc_hash[:10]}'
AUDIT_OUTPUT = PIPELINE_ROOT / 'runs' / identity / (
    'rab-label-disentanglement-' + AUDIT_ATTEMPT
)
APPROVAL_PATH = PIPELINE_ROOT / 'approvals' / identity / f'authorization-{AUDIT_ATTEMPT}.json'
RECOVERY_APPROVAL_PATH = (
    PIPELINE_ROOT / 'approvals' / identity / f'zero-artifact-recovery-{AUDIT_ATTEMPT}.json'
)
RECOVERY_TEMPLATE_PATH = (
    PIPELINE_ROOT / 'approvals' / identity
    / f'zero-artifact-recovery-template-{AUDIT_ATTEMPT}.json'
)
print('Audit output:', AUDIT_OUTPUT)
print('External approval:', APPROVAL_PATH)
print('External recovery approval:', RECOVERY_APPROVAL_PATH)
"""


LIFECYCLE = """
def run(command, *, cwd):
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f'Command failed with exit code {completed.returncode}: {command}')
    return completed

def last_json_line(completed):
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError('Command produced no JSON output')
    return json.loads(lines[-1])

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
        print('Preregistration already exists; not rebuilding label bank')
    else:
        run([
            'uv', 'run', 'plasticity-p0d2hrabx', 'plan',
            '--output', str(AUDIT_OUTPUT), '--rabc-output', str(RABC_OUTPUT),
        ], cwd=EXPERIMENT_REPO_DIR)
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
                'factor_counts': payload['factor_counts'],
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
        'uv', 'run', 'plasticity-p0d2hrabx', 'authorize',
        '--output', str(AUDIT_OUTPUT), '--authorization', str(APPROVAL_PATH),
    ], cwd=EXPERIMENT_REPO_DIR)

if RUN_RECOVERY_INSPECTION:
    manifest = json.loads((AUDIT_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] != 'running':
        raise RuntimeError(
            f'Zero-artifact inspection requires running state: {manifest["state"]}'
        )
    result = run([
        'uv', 'run', 'plasticity-p0d2hrabx', 'zero-artifact-recovery-template',
        '--output', str(AUDIT_OUTPUT),
        '--experiment-code-revision-lock', str(EXPERIMENT_CODE_REVISION_LOCK),
    ], cwd=RECOVERY_REPO_DIR)
    template = last_json_line(result)
    RECOVERY_TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RECOVERY_TEMPLATE_PATH.with_suffix(RECOVERY_TEMPLATE_PATH.suffix + '.tmp')
    temporary.write_text(json.dumps(template, indent=2, sort_keys=True) + '\\n')
    temporary.replace(RECOVERY_TEMPLATE_PATH)
    print('Recovery eligibility verified; inspect template:', RECOVERY_TEMPLATE_PATH)

if RUN_RECOVER_ZERO_ARTIFACT:
    manifest = json.loads((AUDIT_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] == 'running':
        if not RECOVERY_TEMPLATE_PATH.is_file():
            raise RuntimeError('Run a separate recovery-inspection pass first')
        inspected_template = json.loads(RECOVERY_TEMPLATE_PATH.read_text())
        result = run([
            'uv', 'run', 'plasticity-p0d2hrabx', 'zero-artifact-recovery-template',
            '--output', str(AUDIT_OUTPUT),
            '--experiment-code-revision-lock', str(EXPERIMENT_CODE_REVISION_LOCK),
        ], cwd=RECOVERY_REPO_DIR)
        current_template = last_json_line(result)
        if current_template != inspected_template:
            raise RuntimeError(
                'Recovery evidence changed after inspection; run a fresh inspection pass'
            )
        fixed = {
            **current_template,
            'decision': 'approved',
            'approved_by': APPROVER.strip(),
            'original_runtime_confirmed_terminated': ORIGINAL_RUNTIME_TERMINATED,
        }
        fixed.pop('approved_at')
        write_or_validate_approval(RECOVERY_APPROVAL_PATH, fixed)
    elif manifest['state'] == 'authorized' and isinstance(manifest.get('recovery'), dict):
        if not RECOVERY_APPROVAL_PATH.is_file():
            raise RuntimeError('Recovered manifest is missing its external approval')
        print('Recovery already adopted; validating idempotent completion')
    else:
        raise RuntimeError(
            f'Zero-artifact recovery is not valid from state: {manifest["state"]}'
        )
    run([
        'uv', 'run', 'plasticity-p0d2hrabx', 'recover-zero-artifact',
        '--output', str(AUDIT_OUTPUT),
        '--authorization', str(RECOVERY_APPROVAL_PATH),
        '--experiment-code-revision-lock', str(EXPERIMENT_CODE_REVISION_LOCK),
    ], cwd=RECOVERY_REPO_DIR)

if RUN_AUDIT:
    manifest = json.loads((AUDIT_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] == 'authorized':
        run([
            'uv', 'run', 'plasticity-p0d2hrabx', 'run', '--output', str(AUDIT_OUTPUT),
        ], cwd=EXPERIMENT_REPO_DIR)
    elif manifest['state'] == 'complete':
        print('RAB label-disentanglement audit already complete; not rerunning')
    else:
        raise RuntimeError(f'Label audit requires authorized state: {manifest["state"]}')
"""


RESULTS = """
manifest_path = AUDIT_OUTPUT / 'manifest.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
if manifest and manifest['state'] == 'complete':
    run(
        ['uv', 'run', 'plasticity-p0d2hrabx', 'verify', '--output', str(AUDIT_OUTPUT)],
        cwd=EXPERIMENT_REPO_DIR,
    )
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
        raise RuntimeError('Complete label manifest is missing published artifacts')
    summary_digest = sha256(summary_path.read_bytes()).hexdigest()
    audit_manifest_digest = sha256(audit_manifest_path.read_bytes()).hexdigest()
    audit_manifest = json.loads(audit_manifest_path.read_text())
    if {manifest['result']['summary_sha256'], audit_manifest['artifacts']['summary']} != {
        summary_digest
    }:
        raise RuntimeError('Label summary hash does not match complete manifests')
    if manifest['result']['audit_manifest_sha256'] != audit_manifest_digest:
        raise RuntimeError('Label audit-manifest hash does not match complete manifest')
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
            'historical_rabc_attribution_changed': summary[
                'historical_rabc_attribution_changed'
            ],
            'training_authorized': summary['training_authorized'],
            'mappings_per_adapter_authorized': summary[
                'mappings_per_adapter_authorized'
            ],
        },
        indent=2,
    ))
elif manifest:
    print('No displayable result; label manifest state:', manifest['state'])
else:
    print('No label plan yet. Plan, inspect, authorize, and run in separate passes.')
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            markdown(
                f"""
                # P0-D2H frozen RAB receipt/slot-label disentanglement audit

                [Open this notebook in Google Colab]({COLAB_URL})

                This audit adds a visible canonical/crossed receipt-to-slot codebook. Receipt
                token A/B and selected Slot A/B are therefore independently crossed, while slot
                display order and four action-panel rotations remain controlled. Every prompt is
                scored OFF→ON in one runtime. The audit never trains, reclassifies RAB/RABC, or
                authorizes 1/4/8. A separately governed zero-artifact recovery is available for
                a stale running manifest left by a terminated Colab session.
                """
            ),
            code(CONFIG),
            code(CONTROLS),
            code("\n\n".join((SETUP, IDENTITY, LIFECYCLE))),
            markdown(
                """
                Edit only the controls cell. First keep the safe defaults and run the setup and
                combined lifecycle cell. Inspect `preflight/source_audit.json`,
                `preflight/bank_audit.json`, `preflight/token_audit.json`,
                `preflight/analysis_plan.json`, `preregistration.json`, and
                `authorization.template.json`.

                Next disable planning, set a real human `APPROVER`, and enable only
                `RUN_AUTHORIZE`. Finally disable authorization and enable only `RUN_AUDIT`.
                Rerun the controls and combined lifecycle cells for each pass. Run the result
                cell only after completion.

                If a terminated Colab session leaves the manifest at running, do not edit or
                delete it. After it has been stale for six hours, enable only
                RUN_RECOVERY_INSPECTION and inspect the external template. In a separate pass,
                set ORIGINAL_RUNTIME_TERMINATED = True, set APPROVER, and enable only
                RUN_RECOVER_ZERO_ARTIFACT. After recovery returns the same run to authorized,
                enable only RUN_AUDIT. Run the result cell only after completion.
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
