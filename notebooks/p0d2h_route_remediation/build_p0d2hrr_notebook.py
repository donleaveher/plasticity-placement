from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_route_remediation_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/p0d2h_route_remediation/"
    f"{NOTEBOOK_NAME}"
)


def _markdown(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def _code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


CONFIGURATION_SOURCE = f"""
from google.colab import drive
drive.mount('/content/drive')

import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from IPython.display import Markdown, display

REPO_URL = 'https://github.com/donleaveher/plasticity-placement.git'
BRANCH = '{BRANCH}'
REQUESTED_CODE_REVISION = None
REPO_DIR = Path('/content/plasticity-placement')

# Exact completed P0-D2H-CAL-FC source, always read-only.
SOURCE_PIPELINE_ATTEMPT = 'pipeline-f1'
SOURCE_FORCED_CHOICE_ATTEMPT = 'f1'
SOURCE_PIPELINE_ROOT = (
    Path('/content/drive/MyDrive/plasticity-p0d/hard-probe-forced-choice/v1/pipelines')
    / SOURCE_PIPELINE_ATTEMPT
)

# Independent route-remediation destination.
RR_PIPELINE_ATTEMPT = 'pipeline-rr1'
RR_ATTEMPT = 'rr1'
RR_PIPELINE_ROOT = (
    Path('/content/drive/MyDrive/plasticity-p0d/route-remediation-lora/v1/pipelines')
    / RR_PIPELINE_ATTEMPT
)
RR_OUTPUT = RR_PIPELINE_ROOT / 'runs' / ('route-remediation-' + RR_ATTEMPT)

# An independent reviewer places the approved file here. The notebook never
# creates or edits an approved authorization.
AUTHORIZATION_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/authorizations/'
    'p0d2h-route-remediation/v1'
)
AUTHORIZATION_PATH = AUTHORIZATION_ROOT / (
    RR_PIPELINE_ATTEMPT + '-' + RR_ATTEMPT + '.approved.json'
)

BOOTSTRAP_SAMPLES = 10000

# Safe defaults: only compile/audit/freeze the preregistration.
RUN_PLAN = True
ADOPT_AUTHORIZATION = False
RUN_TRAINING = False
RUN_LOCKED_EVALUATION = False
RUN_AGGREGATE = False

for name, value in {{
    'SOURCE_PIPELINE_ATTEMPT': SOURCE_PIPELINE_ATTEMPT,
    'SOURCE_FORCED_CHOICE_ATTEMPT': SOURCE_FORCED_CHOICE_ATTEMPT,
    'RR_PIPELINE_ATTEMPT': RR_PIPELINE_ATTEMPT,
    'RR_ATTEMPT': RR_ATTEMPT,
}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
if BOOTSTRAP_SAMPLES <= 0:
    raise ValueError('BOOTSTRAP_SAMPLES must be positive')

requested_stage_count = sum(
    (
        ADOPT_AUTHORIZATION,
        RUN_TRAINING,
        RUN_LOCKED_EVALUATION,
        RUN_AGGREGATE,
    )
)
if requested_stage_count > 1:
    raise ValueError(
        'Enable at most one post-plan stage per notebook pass. '
        'Authorization, training, evaluation, and aggregation are separate gates.'
    )

print('Route-remediation output:', RR_OUTPUT)
print('Independent authorization path:', AUTHORIZATION_PATH)
print(
    'Requested post-plan stage:',
    next(
        (
            name
            for name, enabled in (
                ('authorize', ADOPT_AUTHORIZATION),
                ('train', RUN_TRAINING),
                ('evaluate', RUN_LOCKED_EVALUATION),
                ('aggregate', RUN_AGGREGATE),
            )
            if enabled
        ),
        'none (preflight only)',
    ),
)
print('Frozen matrix: 480 train + 96 dev + 1152 forced-choice + 1536 CRD')
"""


CHECKOUT_SOURCE = """
subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', 'uv'],
    check=True,
)
if REPO_DIR.exists() and not (REPO_DIR / '.git').exists():
    raise RuntimeError(f'{REPO_DIR} exists but is not a Git repository')
if not REPO_DIR.exists():
    subprocess.run(
        ['git', 'clone', '--branch', BRANCH, REPO_URL, str(REPO_DIR)],
        check=True,
    )
dirty = subprocess.run(
    ['git', '-C', str(REPO_DIR), 'status', '--porcelain'],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if dirty:
    raise RuntimeError(f'Repository has local changes:\\n{dirty}')

RR_PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
CODE_REVISION_LOCK = RR_PIPELINE_ROOT / 'code_revision.txt'
locked_revision = (
    CODE_REVISION_LOCK.read_text().strip()
    if CODE_REVISION_LOCK.exists()
    else None
)
subprocess.run(
    ['git', '-C', str(REPO_DIR), 'fetch', 'origin', BRANCH],
    check=True,
)
revision_ref = REQUESTED_CODE_REVISION or locked_revision or f'origin/{BRANCH}'
CODE_REVISION = subprocess.run(
    ['git', '-C', str(REPO_DIR), 'rev-parse', f'{revision_ref}^{{commit}}'],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if locked_revision and CODE_REVISION != locked_revision:
    raise RuntimeError(
        f'RR code lock mismatch: {locked_revision} != {CODE_REVISION}. '
        'Use a new RR_PIPELINE_ATTEMPT.'
    )
if not locked_revision:
    temporary = CODE_REVISION_LOCK.with_suffix('.txt.tmp')
    temporary.write_text(CODE_REVISION + '\\n')
    temporary.replace(CODE_REVISION_LOCK)
subprocess.run(
    ['git', '-C', str(REPO_DIR), 'checkout', '--detach', CODE_REVISION],
    check=True,
)
subprocess.run(
    ['uv', 'sync', '--extra', 'train', '--extra', 'colab'],
    cwd=REPO_DIR,
    check=True,
)

LOG_DIR = RR_PIPELINE_ROOT / 'logs' / ('route-remediation-' + RR_ATTEMPT)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def run_json(command):
    completed = subprocess.run(
        command,
        cwd=REPO_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise RuntimeError(
            f'Command failed ({completed.returncode}): {shlex.join(command)}'
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f'Command returned no JSON: {shlex.join(command)}')
    return json.loads(lines[-1])

def run_checked(label, command):
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    safe_label = re.sub(r'[^A-Za-z0-9._-]+', '-', label)
    log_path = LOG_DIR / f'{timestamp}-{safe_label}.log'
    child_environment = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    print(f'\\n[{label}] $ {shlex.join(command)}')
    print(f'[{label}] log: {log_path}')
    started = time.monotonic()
    with log_path.open('w', encoding='utf-8') as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_DIR,
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end='')
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    elapsed = time.monotonic() - started
    print(f'[{label}] exit={return_code} elapsed={elapsed / 60:.1f} min')
    if return_code != 0:
        raise RuntimeError(
            f'{label} failed with exit code {return_code}; log={log_path}'
        )

environment_context = run_json(
    ['uv', 'run', 'plasticity-p0d2hrr', 'environment']
)
environment = environment_context['environment']
print(json.dumps({
    'code_revision': CODE_REVISION,
    'code_sha256': environment['code_sha256'],
    'environment_fingerprint': environment_context['fingerprint'],
    'cuda_available': environment['cuda_available'],
    'cuda_version': environment['cuda_version'],
    'gpu': environment['gpu'],
    'packages': environment['packages'],
}, ensure_ascii=False, indent=2))
print('No model has been loaded; checkout/install is preflight.')
"""


SOURCE_SOURCE = """
source_matches = sorted(
    SOURCE_PIPELINE_ROOT.glob(
        'runs/*/forced_choice-'
        + SOURCE_FORCED_CHOICE_ATTEMPT
        + '/manifest.json'
    )
)
if len(source_matches) != 1:
    raise RuntimeError(
        'Expected exactly one completed P0-D2H-CAL-FC manifest under '
        f'{SOURCE_PIPELINE_ROOT}, found {source_matches}'
    )
SOURCE_MANIFEST = source_matches[0]
SOURCE_STAGE_DIR = SOURCE_MANIFEST.parent
source_summary_path = (
    SOURCE_STAGE_DIR / 'results' / 'aggregate' / 'summary.json'
)
source_audit_path = (
    SOURCE_STAGE_DIR / 'preflight' / 'candidate_token_audit.json'
)

EXPECTED_SOURCE = {
    'run_id': 'p0d2hfc-forced-choice-f604efe6bf',
    'manifest_sha256': (
        '89ba71ea0fde052089329e630d23c1a86652bc47a129de1921786c8e0b4dd214'
    ),
    'summary_sha256': (
        '24b1fe51bf400e5c1e0b15aa68d67b0db913591cce082aeac13b9396854d22e1'
    ),
    'candidate_audit_sha256': (
        'd140cfb2ac0c57b4819d3793b59f5f7c767fe428336be856fc9299460731b230'
    ),
    'raw_tree_sha256': (
        'dfe9b9014966f484e988a2f6d9f8e18d4ee7a1b5f06d671ae2e973cef8a0b24a'
    ),
}

def raw_tree_hash():
    digest = sha256()
    files = sorted((SOURCE_STAGE_DIR / 'results' / 'raw').rglob('*.jsonl'))
    if len(files) != 48:
        raise RuntimeError(f'Expected 48 source raw files, found {len(files)}')
    for path in files:
        digest.update(str(path.relative_to(SOURCE_STAGE_DIR)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()

def source_snapshot():
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    return {
        'run_id': source_manifest.get('run_id'),
        'manifest_sha256': sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(),
        'summary_sha256': sha256(source_summary_path.read_bytes()).hexdigest(),
        'candidate_audit_sha256': sha256(source_audit_path.read_bytes()).hexdigest(),
        'raw_tree_sha256': raw_tree_hash(),
    }

SOURCE_SNAPSHOT_BEFORE = source_snapshot()
if SOURCE_SNAPSHOT_BEFORE != EXPECTED_SOURCE:
    raise RuntimeError(
        'Exact P0-D2H-CAL-FC source identity mismatch:\\n'
        + json.dumps(SOURCE_SNAPSHOT_BEFORE, indent=2)
    )
source_manifest = json.loads(SOURCE_MANIFEST.read_text())
if (
    source_manifest.get('schema_version') != 'p0d2hfc-manifest-v1'
    or len(source_manifest.get('units', {})) != 48
    or any(
        unit.get('state') != 'verified'
        for unit in source_manifest.get('units', {}).values()
    )
    or source_manifest.get('errors')
):
    raise RuntimeError('P0-D2H-CAL-FC source is not complete and verified')

SPEC_PATH = REPO_DIR / 'configs' / 'p0d2hrr-route-remediation-pilot-v1.json'
if not SPEC_PATH.is_file():
    raise FileNotFoundError(SPEC_PATH)
if (
    RR_OUTPUT.resolve() == SOURCE_STAGE_DIR.resolve()
    or RR_OUTPUT.resolve().is_relative_to(SOURCE_STAGE_DIR.resolve())
    or SOURCE_STAGE_DIR.resolve().is_relative_to(RR_OUTPUT.resolve())
):
    raise RuntimeError('RR output must be independent from the frozen source')
if (
    AUTHORIZATION_PATH.resolve() == RR_OUTPUT.resolve()
    or AUTHORIZATION_PATH.resolve().is_relative_to(RR_OUTPUT.resolve())
):
    raise RuntimeError('Authorization path must be outside the RR output')

def assert_source_unchanged():
    observed = source_snapshot()
    if observed != SOURCE_SNAPSHOT_BEFORE:
        raise RuntimeError(
            'Frozen P0-D2H-CAL-FC source changed during this notebook:\\n'
            + json.dumps(observed, indent=2)
        )
    return observed

def rr_command(action):
    assert_source_unchanged()
    command = [
        'uv', 'run', 'plasticity-p0d2hrr', action,
        '--output', str(RR_OUTPUT),
    ]
    if action == 'plan':
        command.extend([
            '--source-manifest', str(SOURCE_MANIFEST),
            '--spec', str(SPEC_PATH),
        ])
    elif action == 'authorize':
        command.extend(['--authorization', str(AUTHORIZATION_PATH)])
    elif action == 'aggregate':
        command.extend(['--bootstrap-samples', str(BOOTSTRAP_SAMPLES)])
    return command

def load_rr_manifest():
    path = RR_OUTPUT / 'manifest.json'
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if payload.get('schema_version') != 'p0d2hrr-manifest-v1':
        raise RuntimeError(f'Unexpected RR manifest: {path}')
    return payload

print(json.dumps(SOURCE_SNAPSHOT_BEFORE, ensure_ascii=False, indent=2))
print('Frozen source is complete, verified, and hash-locked.')
"""


PLAN_SOURCE = """
if RUN_PLAN:
    print('CPU PREREGISTRATION/TOKEN AUDIT STARTING; no model CUDA load')
    run_checked('p0d2hrr-plan', rr_command('plan'))
else:
    print('PLAN SKIPPED: RUN_PLAN=False; requiring an existing manifest')

manifest = load_rr_manifest()
preregistration_path = RR_OUTPUT / 'preregistration.json'
authorization_template_path = RR_OUTPUT / 'authorization.template.json'
required_preflight = {
    'data': RR_OUTPUT / 'preflight' / 'route_data_audit.json',
    'dev': RR_OUTPUT / 'preflight' / 'dev_token_audit.json',
    'forced_choice': (
        RR_OUTPUT / 'preflight' / 'forced_choice_token_audit.json'
    ),
    'crd_bank': RR_OUTPUT / 'preflight' / 'crd_bank_audit.json',
    'crd': RR_OUTPUT / 'preflight' / 'crd_token_audit.json',
}
for path in (
    preregistration_path,
    authorization_template_path,
    *required_preflight.values(),
):
    if not path.is_file():
        raise FileNotFoundError(path)

preregistration = json.loads(preregistration_path.read_text())
authorization_template = json.loads(authorization_template_path.read_text())
audits = {
    name: json.loads(path.read_text())
    for name, path in required_preflight.items()
}
if (
    preregistration.get('training_complexity_review_eligible') is not False
    or preregistration.get('automatic_training_started') is not False
    or preregistration.get('automatic_narrow_scan_started') is not False
    or preregistration.get('automatic_rlvr_started') is not False
):
    raise RuntimeError('Preregistration improperly authorizes a later stage')
if (
    authorization_template.get('decision') != 'pending'
    or authorization_template.get('preregistration_sha256')
    != preregistration.get('preregistration_sha256')
):
    raise RuntimeError('Authorization template is not pending or hash-bound')
if (
    audits['data'].get('all_checks_passed') is not True
    or audits['data'].get('train_example_count') != 480
    or audits['data'].get('dev_example_count') != 96
    or audits['dev'].get('all_checks_passed') is not True
    or audits['dev'].get('decision_count') != 96
    or audits['forced_choice'].get('all_checks_passed') is not True
    or audits['forced_choice'].get('decision_count') != 1152
    or audits['crd_bank'].get('all_checks_passed') is not True
    or audits['crd'].get('all_checks_passed') is not True
    or audits['crd'].get('decision_count') != 1536
):
    raise RuntimeError('Route-remediation preflight audit failed')

print(json.dumps({
    'run_id': manifest['run_id'],
    'state': manifest['state'],
    'preregistration_sha256': preregistration['preregistration_sha256'],
    'model': preregistration['spec']['model_name'],
    'model_revision': preregistration['spec']['model_revision'],
    'source_precision': preregistration['source_evaluation_precision'],
    'training': preregistration['spec']['training'],
    'gates': preregistration['spec']['gates'],
    'data_audit': {
        'train_examples': audits['data']['train_example_count'],
        'dev_examples': audits['data']['dev_example_count'],
        'checks': audits['data']['checks'],
    },
    'token_audits': {
        name: {
            key: report.get(key)
            for key in (
                'decision_count',
                'candidate_count',
                'failed_decision_count',
                'input_truncated_count',
                'all_checks_passed',
            )
        }
        for name, report in audits.items()
        if name != 'data'
    },
}, ensure_ascii=False, indent=2))
print('PREREGISTRATION COMPLETE. Training remains unauthorized.')
"""


AUTHORIZATION_REVIEW_SOURCE = """
manifest = load_rr_manifest()
preregistration = json.loads(
    (RR_OUTPUT / 'preregistration.json').read_text()
)
template = json.loads(
    (RR_OUTPUT / 'authorization.template.json').read_text()
)
if template.get('decision') != 'pending':
    raise RuntimeError('The generated authorization template must remain pending')
if (
    template.get('preregistration_sha256')
    != preregistration.get('preregistration_sha256')
):
    raise RuntimeError('Authorization template is not bound to this preregistration')

print('Current state:', manifest['state'])
print('Preregistration SHA-256:', preregistration['preregistration_sha256'])
print('Pending template:', RR_OUTPUT / 'authorization.template.json')
print('Expected independent approval:', AUTHORIZATION_PATH)
if AUTHORIZATION_PATH.exists():
    candidate = json.loads(AUTHORIZATION_PATH.read_text())
    print(json.dumps({
        'schema_version': candidate.get('schema_version'),
        'decision': candidate.get('decision'),
        'scope': candidate.get('scope'),
        'preregistration_sha256': candidate.get('preregistration_sha256'),
        'approved_by': candidate.get('approved_by'),
        'approved_at': candidate.get('approved_at'),
    }, ensure_ascii=False, indent=2))
else:
    print(
        'No approval found. An independent reviewer must author the file '
        'outside RR_OUTPUT; this notebook will not create it.'
    )
"""


AUTHORIZE_SOURCE = """
if ADOPT_AUTHORIZATION:
    if not AUTHORIZATION_PATH.is_file():
        raise FileNotFoundError(
            'Independent approval file is missing: ' + str(AUTHORIZATION_PATH)
        )
    if load_rr_manifest()['state'] != 'planned':
        raise RuntimeError(
            'Authorization adoption requires state=planned; current state='
            + load_rr_manifest()['state']
        )
    run_checked('p0d2hrr-authorize', rr_command('authorize'))
    manifest = load_rr_manifest()
    if manifest['state'] != 'authorized':
        raise RuntimeError(f'Authorization did not advance state: {manifest}')
    print(json.dumps(manifest['authorization'], ensure_ascii=False, indent=2))
else:
    print('AUTHORIZATION NOT ADOPTED: ADOPT_AUTHORIZATION=False')
    print('Current state:', load_rr_manifest()['state'])
"""


TRAIN_SOURCE = """
if RUN_TRAINING:
    manifest = load_rr_manifest()
    if manifest['state'] != 'authorized':
        raise PermissionError(
            'Training requires state=authorized; current state='
            + manifest['state']
        )
    if environment.get('cuda_available') is not True:
        raise RuntimeError('Select a GPU runtime before training')
    print('FORMAL ROUTE-REMEDIATION TRAINING STARTING: first model CUDA load')
    subprocess.run(['nvidia-smi'], check=True)
    run_checked('p0d2hrr-train', rr_command('train'))
    subprocess.run(['nvidia-smi'], check=True)
    manifest = load_rr_manifest()
    if manifest['state'] != 'trained':
        raise RuntimeError(f'Training did not complete: {manifest}')
    print(json.dumps(manifest['training'], ensure_ascii=False, indent=2))
else:
    print('TRAINING SKIPPED: RUN_TRAINING=False')
    print('Current state:', load_rr_manifest()['state'])
"""


EVALUATE_SOURCE = """
if RUN_LOCKED_EVALUATION:
    manifest = load_rr_manifest()
    if manifest['state'] != 'trained':
        raise PermissionError(
            'Locked evaluation requires state=trained; current state='
            + manifest['state']
        )
    if environment.get('cuda_available') is not True:
        raise RuntimeError('Select a GPU runtime before locked evaluation')
    print(
        'LOCKED GPU EVALUATION STARTING: '
        '96 dev + 1152 forced-choice + 1536 CRD decisions'
    )
    subprocess.run(['nvidia-smi'], check=True)
    run_checked('p0d2hrr-evaluate', rr_command('evaluate'))
    subprocess.run(['nvidia-smi'], check=True)
    manifest = load_rr_manifest()
    if manifest['state'] != 'evaluated':
        raise RuntimeError(f'Locked evaluation did not complete: {manifest}')
    print(json.dumps(manifest['evaluation'], ensure_ascii=False, indent=2))
else:
    print('LOCKED EVALUATION SKIPPED: RUN_LOCKED_EVALUATION=False')
    print('Current state:', load_rr_manifest()['state'])
"""


AGGREGATE_SOURCE = """
if RUN_AGGREGATE:
    manifest = load_rr_manifest()
    if manifest['state'] != 'evaluated':
        raise RuntimeError(
            'Aggregation requires state=evaluated; current state='
            + manifest['state']
        )
    run_checked('p0d2hrr-aggregate', rr_command('aggregate'))
    manifest = load_rr_manifest()
    if manifest['state'] != 'verified':
        raise RuntimeError(f'Aggregation did not verify the run: {manifest}')
else:
    print('AGGREGATION SKIPPED: RUN_AGGREGATE=False')
    print('Current state:', load_rr_manifest()['state'])
"""


RESULTS_SOURCE = """
status = run_json(rr_command('status'))
print(json.dumps(status, ensure_ascii=False, indent=2))

summary_path = RR_OUTPUT / 'results' / 'aggregate' / 'summary.json'
table_path = RR_OUTPUT / 'results' / 'aggregate' / 'main_table.md'
decision_path = (
    RR_OUTPUT / 'results' / 'aggregate' / 'next_stage_decision.json'
)
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    decision = json.loads(decision_path.read_text())
    display(Markdown(table_path.read_text()))
    if (
        decision.get('automatic_training_started') is not False
        or decision.get('automatic_narrow_scan_started') is not False
        or decision.get('automatic_rlvr_started') is not False
    ):
        raise RuntimeError('The no-automatic-next-stage boundary was violated')
    print(json.dumps({
        'dev': summary['dev'],
        'official_model_gate': (
            summary['forced_choice']['official_model_gate']
        ),
        'endpoint_summary': summary['crd']['endpoint_summary'],
        'route_remediation_gate': summary['route_remediation_gate'],
        'next_stage_decision': decision,
        'summary_path': str(summary_path),
    }, ensure_ascii=False, indent=2))
else:
    print(
        'No verified aggregate yet. Keep stages separate and enable only the '
        'next lifecycle flag in a later pass.'
    )
"""


SOURCE_FINAL_SOURCE = """
SOURCE_SNAPSHOT_AFTER = assert_source_unchanged()
if SOURCE_SNAPSHOT_AFTER != SOURCE_SNAPSHOT_BEFORE:
    raise RuntimeError('Frozen source changed')
print(json.dumps({
    'source_unchanged': True,
    'source_snapshot': SOURCE_SNAPSHOT_AFTER,
    'rr_output': str(RR_OUTPUT),
    'manifest_state': load_rr_manifest()['state'],
    'automatic_next_stage_started': False,
}, ensure_ascii=False, indent=2))
"""


def build_notebook() -> dict[str, Any]:
    cells = [
        _markdown(
            f"""
            # P0-D2H-RR: Route-Remediation LoRA Pilot

            [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

            This notebook implements the separately preregistered routing
            remediation pilot for the frozen 1.5B scale canary. It trains one
            fixed rank-8 LoRA adapter on a synthetic route-only/route-and-copy
            bank, then reruns the complete forced-choice and CRD panels.

            Run cells in order. The default pass performs only code locking,
            source verification, data compilation, and token audits. It does
            **not** authorize or start training. Authorization adoption,
            training, locked evaluation, and aggregation are separate gated
            passes; enable at most one corresponding flag each time.

            A successful aggregate permits only human review of a future
            1/4/8 training-complexity experiment. There is no automatic scan,
            narrow scan, RLVR, or next-stage launch.
            """
        ),
        _markdown("## 1. Frozen configuration, Drive namespaces, and safe stage flags"),
        _code(CONFIGURATION_SOURCE),
        _markdown("## 2. Checkout code-locked revision and install dependencies"),
        _code(CHECKOUT_SOURCE),
        _markdown("## 3. Locate and hash-freeze the exact read-only source"),
        _code(SOURCE_SOURCE),
        _markdown("## 4. Compile and audit the preregistration (CPU preflight)"),
        _code(PLAN_SOURCE),
        _markdown(
            """
            ## 5. Independent authorization handoff

            The generated file under `RR_OUTPUT` is a pending template, not an
            approval. A named reviewer must create the approved JSON at
            `AUTHORIZATION_PATH`, which is outside the experiment output and
            bound to the displayed preregistration SHA-256.
            """
        ),
        _code(AUTHORIZATION_REVIEW_SOURCE),
        _markdown("## 6. Adopt one externally authored authorization"),
        _code(AUTHORIZE_SOURCE),
        _markdown("## 7. Run the single frozen LoRA training job (GPU)"),
        _code(TRAIN_SOURCE),
        _markdown("## 8. Run the one locked external evaluation (GPU)"),
        _code(EVALUATE_SOURCE),
        _markdown("## 9. Aggregate the complete matrices and apply gates"),
        _code(AGGREGATE_SOURCE),
        _markdown("## 10. Review results; no automatic next stage"),
        _code(RESULTS_SOURCE),
        _markdown("## 11. Verify that the frozen source remained unchanged"),
        _code(SOURCE_FINAL_SOURCE),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": NOTEBOOK_NAME, "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    output = OUTPUT_DIR / NOTEBOOK_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
