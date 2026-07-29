from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_route_decomposition_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/p0d2h_route_decomposition/"
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

# Exact completed P0-D2H-CAL-FC source, read-only.
SOURCE_PIPELINE_ATTEMPT = 'pipeline-f1'
SOURCE_FORCED_CHOICE_ATTEMPT = 'f1'
SOURCE_PIPELINE_ROOT = (
    Path('/content/drive/MyDrive/plasticity-p0d/hard-probe-forced-choice/v1/pipelines')
    / SOURCE_PIPELINE_ATTEMPT
)

# Independent route/retrieval decomposition destination.
CRD_PIPELINE_ATTEMPT = 'pipeline-crd1'
CRD_ATTEMPT = 'crd1'
CRD_PIPELINE_ROOT = (
    Path('/content/drive/MyDrive/plasticity-p0d/hard-probe-route-decomposition/v1/pipelines')
    / CRD_PIPELINE_ATTEMPT
)

BOOTSTRAP_SAMPLES = 10000
RUN_FORMAL_SCORING = True

for name, value in {{
    'SOURCE_PIPELINE_ATTEMPT': SOURCE_PIPELINE_ATTEMPT,
    'SOURCE_FORCED_CHOICE_ATTEMPT': SOURCE_FORCED_CHOICE_ATTEMPT,
    'CRD_PIPELINE_ATTEMPT': CRD_PIPELINE_ATTEMPT,
    'CRD_ATTEMPT': CRD_ATTEMPT,
}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
if BOOTSTRAP_SAMPLES <= 0:
    raise ValueError('BOOTSTRAP_SAMPLES must be positive')
print(
    'Execution mode:',
    'FORMAL GPU SCORING' if RUN_FORMAL_SCORING else 'CPU PREFLIGHT ONLY',
)
print('Frozen matrix: 1 model × 24 lessons × 64 decisions = 1536 rows')
print('Endpoint candidate total: 5376 candidate sequences')
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

CRD_PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
CODE_REVISION_LOCK = CRD_PIPELINE_ROOT / 'code_revision.txt'
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
        f'CRD code lock mismatch: {locked_revision} != {CODE_REVISION}. '
        'Use a new CRD_PIPELINE_ATTEMPT.'
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

environment_context = run_json(
    ['uv', 'run', 'plasticity-p0d2hcrd', 'environment']
)
environment = environment_context['environment']
print(json.dumps({
    'cuda_available': environment['cuda_available'],
    'cuda_version': environment['cuda_version'],
    'gpu': environment['gpu'],
    'packages': environment['packages'],
}, ensure_ascii=False, indent=2))
print('No model has been loaded; checkout/install is preflight.')
print('Checked out P0-D2H-CRD code:', CODE_REVISION)
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
source_manifest = json.loads(SOURCE_MANIFEST.read_text())
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

observed_source = {
    'run_id': source_manifest.get('run_id'),
    'manifest_sha256': sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(),
    'summary_sha256': sha256(source_summary_path.read_bytes()).hexdigest(),
    'candidate_audit_sha256': sha256(source_audit_path.read_bytes()).hexdigest(),
    'raw_tree_sha256': raw_tree_hash(),
}
if observed_source != EXPECTED_SOURCE:
    raise RuntimeError(
        'Exact P0-D2H-CAL-FC source identity mismatch:\\n'
        + json.dumps(observed_source, indent=2)
    )
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

CODE_HASH = environment['code_sha256']
PROVENANCE_KEY = (
    f'code-{CODE_HASH[:10]}_source-{EXPECTED_SOURCE["manifest_sha256"][:10]}'
)
PROVENANCE_ROOT = CRD_PIPELINE_ROOT / 'runs' / PROVENANCE_KEY
STAGE_DIR = PROVENANCE_ROOT / ('route_decomposition-' + CRD_ATTEMPT)
LOG_DIR = PROVENANCE_ROOT / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

frozen_context = {
    'schema_version': 1,
    'pipeline_version': 'p0d2hcrd-counterbalanced-bank-v1',
    'code_sha256': CODE_HASH,
    'source': EXPECTED_SOURCE,
    'model_id': 'scale_canary',
    'model_name': 'Qwen/Qwen2.5-1.5B-Instruct',
    'model_revision': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
    'evaluation_max_length': 512,
    'primary_score_definition': 'candidate_token_sum_logprob',
    'rows': 1536,
    'candidate_sequences': 5376,
    'gate_thresholds': {
        'route_only': 0.90,
        'retrieval_only': 0.90,
        'combined': 0.75,
        'tie_error_nonfinite_rate': 0.0,
    },
    'training_complexity_review_eligible': False,
    'automatic_training_started': False,
    'automatic_narrow_scan_started': False,
}
STAGE_DIR.mkdir(parents=True, exist_ok=True)
context_path = STAGE_DIR / 'notebook_context.json'
if context_path.exists():
    if json.loads(context_path.read_text()) != frozen_context:
        raise RuntimeError(f'Frozen CRD context mismatch: {context_path}')
else:
    temporary = context_path.with_suffix('.json.tmp')
    temporary.write_text(
        json.dumps(frozen_context, indent=2, sort_keys=True) + '\\n'
    )
    temporary.replace(context_path)

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

def crd_command(action):
    if sha256(SOURCE_MANIFEST.read_bytes()).hexdigest() != (
        EXPECTED_SOURCE['manifest_sha256']
    ):
        raise RuntimeError('P0-D2H-CAL-FC source manifest changed after freeze')
    if raw_tree_hash() != EXPECTED_SOURCE['raw_tree_sha256']:
        raise RuntimeError('P0-D2H-CAL-FC source raw rows changed after freeze')
    command = [
        'uv', 'run', 'plasticity-p0d2hcrd', action,
        '--output', str(STAGE_DIR),
    ]
    if action in {'plan', 'audit', 'run'}:
        command.extend(['--source-manifest', str(SOURCE_MANIFEST)])
    return command
"""


PREFLIGHT_SOURCE = """
plan = run_json(crd_command('plan'))
if (
    plan['expected_unit_count'] != 24
    or plan['expected_decision_row_count'] != 1536
    or plan['expected_candidate_sequence_count'] != 5376
    or plan['base_only'] is not True
    or plan['diagnostic_only'] is not True
    or plan['adapter_discovery_requested'] is not False
    or plan['training_complexity_review_eligible'] is not False
    or plan['automatic_training_started'] is not False
    or plan['automatic_narrow_scan_started'] is not False
):
    raise RuntimeError(f'Unexpected frozen CRD plan: {plan}')
print(json.dumps({
    'model': plan['config']['model'],
    'endpoints': plan['config']['endpoints'],
    'rows_per_lesson': plan['config']['rows_per_lesson'],
    'candidate_counts': plan['config']['candidates_per_endpoint'],
    'gate_thresholds': plan['config']['gate_thresholds'],
    'bank_audit': plan['bank_audit'],
}, ensure_ascii=False, indent=2))
print('CPU METADATA/BANK PREFLIGHT COMPLETE; no model loaded')
"""


AUDIT_SOURCE = """
print('FULL CRD CANDIDATE-TOKEN AUDIT STARTING ON CPU')
audit_result = run_json(crd_command('audit'))
audit = audit_result['audit']
if (
    audit.get('all_checks_passed') is not True
    or audit.get('decision_count') != 1536
    or audit.get('candidate_count') != 5376
    or audit.get('failed_decision_count') != 0
    or audit.get('input_truncated_count') != 0
):
    raise RuntimeError(f'CRD candidate-token audit failed: {audit}')
print(json.dumps(audit, ensure_ascii=False, indent=2))
print('CPU TOKEN AUDIT COMPLETE; 0 models loaded to CUDA')
"""


RUN_SOURCE = """
if RUN_FORMAL_SCORING:
    if environment.get('cuda_available') is not True:
        raise RuntimeError('Select a GPU runtime before formal scoring')
    print('FORMAL CRD GPU SCORING STARTING: first model CUDA load occurs here')
    subprocess.run(['nvidia-smi'], check=True)
    run_checked('p0d2hcrd-formal-run', crd_command('run'))
    subprocess.run(['nvidia-smi'], check=True)
else:
    print('FORMAL SCORING SKIPPED: RUN_FORMAL_SCORING=False')
"""


VERIFY_SOURCE = """
manifest_path = STAGE_DIR / 'manifest.json'
if not manifest_path.exists():
    if RUN_FORMAL_SCORING:
        raise FileNotFoundError(manifest_path)
    print('No manifest expected in preflight-only mode')
else:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get('schema_version') != 'p0d2hcrd-manifest-v1'
        or len(manifest.get('units', {})) != 24
        or any(
            unit.get('state') != 'verified'
            for unit in manifest.get('units', {}).values()
        )
        or manifest.get('errors')
    ):
        raise RuntimeError('CRD manifest is not complete and verified')
    raw_files = sorted((STAGE_DIR / 'results' / 'raw').rglob('*.jsonl'))
    row_count = sum(
        len([line for line in path.read_text().splitlines() if line.strip()])
        for path in raw_files
    )
    if len(raw_files) != 24 or row_count != 1536:
        raise RuntimeError(
            f'Raw CRD matrix mismatch: files={len(raw_files)} rows={row_count}'
        )
    print('RECOVERY MANIFEST VERIFIED: 24 units, 1536 rows')
"""


AGGREGATE_SOURCE = """
if RUN_FORMAL_SCORING:
    run_checked(
        'p0d2hcrd-aggregate',
        crd_command('aggregate')
        + ['--bootstrap-samples', str(BOOTSTRAP_SAMPLES)],
    )
else:
    print('AGGREGATE SKIPPED: no formal rows')
"""


RESULTS_SOURCE = """
summary_path = STAGE_DIR / 'results' / 'aggregate' / 'summary.json'
table_path = STAGE_DIR / 'results' / 'aggregate' / 'main_table.md'
decision_path = (
    STAGE_DIR / 'results' / 'aggregate' / 'next_stage_decision.json'
)
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    decision = json.loads(decision_path.read_text())
    display(Markdown(table_path.read_text()))
    if (
        decision.get('training_complexity_review_eligible') is not False
        or decision.get('automatic_training_started') is not False
        or decision.get('automatic_narrow_scan_started') is not False
    ):
        raise RuntimeError('CRD diagnostic boundary was violated')
    print(json.dumps({
        'endpoint_summary': summary['endpoint_summary'],
        'contrasts': summary['contrasts'],
        'diagnostic_gate': summary['diagnostic_gate'],
        'next_stage_decision': decision,
        'summary_path': str(summary_path),
    }, ensure_ascii=False, indent=2))
else:
    print('No aggregate yet. Run the formal and aggregate cells first.')
"""


def build_notebook() -> dict[str, Any]:
    cells = [
        _markdown(
            f"""
            # P0-D2H-CRD: Route/Retrieval Decomposition

            [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

            This independent base-only diagnostic separates explicit
            marker-to-slot routing from selected-record retrieval and retains a
            combined endpoint. It reads the exact completed P0-D2H-CAL-FC tree
            and its transitive sources without modifying them.

            Run in order. Cells through the full 1,536-prompt candidate-token
            audit are CPU preflight. The first model CUDA load occurs only in
            cell 6. No adapter, training, scan, RLVR, or automatic next stage is
            present.
            """
        ),
        _markdown("## 1. Frozen configuration and independent Drive namespace"),
        _code(CONFIGURATION_SOURCE),
        _markdown("## 2. Checkout locked code and install dependencies (preflight)"),
        _code(CHECKOUT_SOURCE),
        _markdown("## 3. Locate and hash-freeze the exact forced-choice source"),
        _code(SOURCE_SOURCE),
        _markdown("## 4. CPU source/bank metadata preflight"),
        _code(PREFLIGHT_SOURCE),
        _markdown("## 5. Full prompt/candidate-token audit (CPU)"),
        _code(AUDIT_SOURCE),
        _markdown("## 6. Formal route/retrieval/combined scoring (GPU)"),
        _code(RUN_SOURCE),
        _markdown("## 7. Verify atomic recovery and the complete matrix"),
        _code(VERIFY_SOURCE),
        _markdown("## 8. Aggregate lesson-clustered diagnostic metrics"),
        _code(AGGREGATE_SOURCE),
        _markdown("## 9. Review diagnostic results; no automatic next stage"),
        _code(RESULTS_SOURCE),
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
