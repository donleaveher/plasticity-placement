from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_route_transfer_bridge_colab.ipynb"
EXPERIMENT_CODE_REVISION = "dd004f54e39598e4b67ae525bb46a38c293378b1"
RECOVERY_CODE_REVISION = "8678cc8c2510e50015fb451f558d47901f918716"
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
EXPERIMENT_CODE_REVISION = '{EXPERIMENT_CODE_REVISION}'
RECOVERY_CODE_REVISION = {RECOVERY_CODE_REVISION!r}
EXPERIMENT_REPO_DIR = Path('/content/plasticity-placement-route-transfer')
RECOVERY_REPO_DIR = Path('/content/plasticity-placement-route-transfer-recovery')

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

for name, value in {{'PIPELINE_ATTEMPT': PIPELINE_ATTEMPT, 'AUDIT_ATTEMPT': AUDIT_ATTEMPT}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
print('Frozen CPR adapter:', CPR_OUTPUT)
print('Closed q2 evidence:', QUALIFICATION_SUMMARY)
print('Bridge pipeline:', PIPELINE_ROOT)
"""


CONTROLS = """
# EDIT ONLY THIS CELL BETWEEN PASSES.
# Safe default: inspect/plan only. Enable exactly one state-changing stage at a time.
RUN_PLAN = True
RUN_AUTHORIZE = False
RUN_RECOVER_ZERO_ARTIFACT = False
RUN_AUDIT = False
APPROVER = ''  # Human/responsible-party identifier; never a password or token.
ORIGINAL_RUNTIME_TERMINATED = False  # Set True only after confirming the old Colab is stopped.

stages = {
    'plan': RUN_PLAN,
    'authorize': RUN_AUTHORIZE,
    'recover_zero_artifact': RUN_RECOVER_ZERO_ARTIFACT,
    'audit': RUN_AUDIT,
}
if sum(bool(value) for value in stages.values()) > 1:
    raise ValueError(f'Enable at most one stage per pass: {stages}')
if (RUN_AUTHORIZE or RUN_RECOVER_ZERO_ARTIFACT) and not APPROVER.strip():
    raise ValueError('Set APPROVER to a human identifier for authorization or recovery')
if RUN_RECOVER_ZERO_ARTIFACT and ORIGINAL_RUNTIME_TERMINATED is not True:
    raise ValueError('Confirm ORIGINAL_RUNTIME_TERMINATED before recovery')
print('Selected controls:', stages)
"""


CHECKOUT = """
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
subprocess.run(['nvidia-smi'], check=True)

environment = subprocess.run(
    ['uv', 'run', 'plasticity-p0d2hrtb', 'environment'],
    cwd=EXPERIMENT_REPO_DIR,
    check=True,
    capture_output=True,
    text=True,
)
environment_payload = json.loads(environment.stdout.splitlines()[-1])
if environment_payload['environment']['cuda_available'] is not True:
    raise RuntimeError('Route-transfer audit requires a CUDA runtime')
print('Experiment code revision:', RESOLVED_EXPERIMENT_CODE_REVISION)
print('Recovery code revision:', RESOLVED_RECOVERY_CODE_REVISION)
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
identity = f'code-{RESOLVED_EXPERIMENT_CODE_REVISION[:10]}_q2-{qualification_hash[:10]}'
AUDIT_OUTPUT = PIPELINE_ROOT / 'runs' / identity / ('route-transfer-' + AUDIT_ATTEMPT)
APPROVAL_PATH = PIPELINE_ROOT / 'approvals' / identity / f'authorization-{AUDIT_ATTEMPT}.json'
RECOVERY_APPROVAL_PATH = (
    PIPELINE_ROOT / 'approvals' / identity / f'zero-artifact-recovery-{AUDIT_ATTEMPT}.json'
)
RECOVERY_TEMPLATE_PATH = (
    PIPELINE_ROOT / 'approvals' / identity / f'zero-artifact-recovery-template-{AUDIT_ATTEMPT}.json'
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

def write_or_validate_external_json(path, fixed, timestamp_key):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        value = json.loads(path.read_text())
        differences = {
            key: {'expected': expected, 'observed': value.get(key)}
            for key, expected in fixed.items()
            if value.get(key) != expected
        }
        if differences:
            raise RuntimeError(f'Existing external JSON differs at {path}: {differences}')
        return value
    value = {**fixed, timestamp_key: datetime.now(timezone.utc).isoformat()}
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\\n')
    temporary.replace(path)
    return value

if RUN_PLAN:
    if (AUDIT_OUTPUT / 'manifest.json').is_file():
        print('Preregistration already exists; not regenerating the bridge bank')
    else:
        run([
            'uv', 'run', 'plasticity-p0d2hrtb', 'plan',
            '--output', str(AUDIT_OUTPUT),
            '--cpr-output', str(CPR_OUTPUT),
            '--qualification-summary', str(QUALIFICATION_SUMMARY),
        ], cwd=EXPERIMENT_REPO_DIR)

if RUN_AUTHORIZE:
    manifest = json.loads((AUDIT_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] == 'planned':
        template = json.loads((AUDIT_OUTPUT / 'authorization.template.json').read_text())
        fixed = {
            **template,
            'decision': 'approved',
            'approved_by': APPROVER.strip(),
        }
        fixed.pop('approved_at')
        write_or_validate_external_json(APPROVAL_PATH, fixed, 'approved_at')
        run([
            'uv', 'run', 'plasticity-p0d2hrtb', 'authorize',
            '--output', str(AUDIT_OUTPUT), '--authorization', str(APPROVAL_PATH),
        ], cwd=EXPERIMENT_REPO_DIR)
    else:
        print('Authorization skipped; current state:', manifest['state'])

manifest_path = AUDIT_OUTPUT / 'manifest.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
if manifest and manifest['state'] == 'running':
    recovery_template_result = run([
        'uv', 'run', 'plasticity-p0d2hrtb', 'zero-artifact-recovery-template',
        '--output', str(AUDIT_OUTPUT),
        '--experiment-code-revision-lock', str(EXPERIMENT_CODE_REVISION_LOCK),
    ], cwd=RECOVERY_REPO_DIR)
    observed_recovery_template = last_json_line(recovery_template_result)
    if RUN_RECOVER_ZERO_ARTIFACT:
        if not RECOVERY_TEMPLATE_PATH.is_file():
            raise RuntimeError('Run a separate safe inspection pass before recovery')
        recovery_template = json.loads(RECOVERY_TEMPLATE_PATH.read_text())
        if recovery_template != observed_recovery_template:
            raise RuntimeError(
                'Recovery evidence changed after inspection; run a fresh inspection pass'
            )
        print('Recovery evidence matches the separately inspected template')
    else:
        recovery_template = observed_recovery_template
        RECOVERY_TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = RECOVERY_TEMPLATE_PATH.with_suffix(RECOVERY_TEMPLATE_PATH.suffix + '.tmp')
        temporary.write_text(json.dumps(recovery_template, indent=2, sort_keys=True) + '\\n')
        temporary.replace(RECOVERY_TEMPLATE_PATH)
        print('Recovery eligibility verified; inspect template:', RECOVERY_TEMPLATE_PATH)

if RUN_RECOVER_ZERO_ARTIFACT:
    if manifest and manifest['state'] == 'running':
        recovery_template = json.loads(RECOVERY_TEMPLATE_PATH.read_text())
        fixed = {
            **recovery_template,
            'decision': 'approved',
            'approved_by': APPROVER.strip(),
            'original_runtime_confirmed_terminated': ORIGINAL_RUNTIME_TERMINATED,
        }
        fixed.pop('approved_at')
        write_or_validate_external_json(RECOVERY_APPROVAL_PATH, fixed, 'approved_at')
    elif manifest and manifest['state'] == 'authorized' and isinstance(
        manifest.get('recovery'), dict
    ):
        if not RECOVERY_APPROVAL_PATH.is_file():
            raise RuntimeError('Recovered manifest is missing its external approval')
        print('Recovery already adopted; validating idempotent completion')
    else:
        state = manifest['state'] if manifest else 'missing'
        raise RuntimeError(f'Zero-artifact recovery is not valid from state: {state}')
    run([
        'uv', 'run', 'plasticity-p0d2hrtb', 'recover-zero-artifact',
        '--output', str(AUDIT_OUTPUT),
        '--authorization', str(RECOVERY_APPROVAL_PATH),
        '--experiment-code-revision-lock', str(EXPERIMENT_CODE_REVISION_LOCK),
    ], cwd=RECOVERY_REPO_DIR)

if RUN_AUDIT:
    manifest = json.loads((AUDIT_OUTPUT / 'manifest.json').read_text())
    if manifest['state'] == 'authorized':
        run(
            ['uv', 'run', 'plasticity-p0d2hrtb', 'run', '--output', str(AUDIT_OUTPUT)],
            cwd=EXPERIMENT_REPO_DIR,
        )
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
            code(CONTROLS),
            code("\n\n".join((CHECKOUT, IDENTITY, LIFECYCLE))),
            markdown(
                """
                The controls are isolated in their own cell so every pass edits and reruns only
                that cell plus the combined setup/lifecycle cell. For a new attempt, run
                `RUN_PLAN`, inspect `preregistration.json`, `preflight/bank_audit.json`, and
                `authorization.template.json`, then authorize and audit in separate passes.

                For the existing stale `running` attempt, the safe-default pass performs only a
                read-only eligibility inspection and writes an external recovery template. Inspect
                that template, set a real `APPROVER`, confirm
                `ORIGINAL_RUNTIME_TERMINATED = True`, enable only
                `RUN_RECOVER_ZERO_ARTIFACT`, and rerun the controls and combined cells. The recovery
                changes the same manifest back to `authorized`; it does not score. Then disable
                recovery, enable only `RUN_AUDIT`, and rerun those two cells to execute exactly one
                identical frozen retry. Never enable recovery and audit in the same pass.
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
