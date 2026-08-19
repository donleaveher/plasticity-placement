from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2_budget_match_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/p0d2_budget_match/{NOTEBOOK_NAME}"
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
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from IPython.display import display

REPO_URL = 'https://github.com/donleaveher/plasticity-placement.git'
BRANCH = '{BRANCH}'
REQUESTED_CODE_REVISION = None
REPO_DIR = Path('/content/plasticity-placement')

P0C_PIPELINE_ATTEMPT = 'pipeline-a1'
P0C_CONFIRMATORY_ATTEMPT = 'a1'
P0C_PIPELINE_ROOT = (
    Path('/content/drive/MyDrive/plasticity-p0c/v4/pipelines')
    / P0C_PIPELINE_ATTEMPT
)

P0D2_PIPELINE_ATTEMPT = 'pipeline-a1'
BUDGET_MATCH_ATTEMPT = 'a1'
P0D2_PIPELINE_ROOT = (
    Path('/content/drive/MyDrive/plasticity-p0d/budget-match/v1/pipelines')
    / P0D2_PIPELINE_ATTEMPT
)

RETENTION_MARGIN = 0.05
BUDGET_TOLERANCE = 0.01
RUN_BUDGET_MATCH = False

for name, value in {{
    'P0C_PIPELINE_ATTEMPT': P0C_PIPELINE_ATTEMPT,
    'P0C_CONFIRMATORY_ATTEMPT': P0C_CONFIRMATORY_ATTEMPT,
    'P0D2_PIPELINE_ATTEMPT': P0D2_PIPELINE_ATTEMPT,
    'BUDGET_MATCH_ATTEMPT': BUDGET_MATCH_ATTEMPT,
}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
if not 0.0 <= RETENTION_MARGIN <= 1.0:
    raise ValueError('RETENTION_MARGIN must be between 0 and 1')
if not 0.0 <= BUDGET_TOLERANCE <= 0.1:
    raise ValueError('BUDGET_TOLERANCE must be between 0 and 0.1')
"""


CHECKOUT_SOURCE = """
subprocess.run(['nvidia-smi'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)

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
    raise RuntimeError(
        'The Colab checkout contains local changes; use a fresh runtime.\\n' + dirty
    )

P0D2_PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
CODE_REVISION_LOCK = P0D2_PIPELINE_ROOT / 'code-revision.txt'
locked_revision = (
    CODE_REVISION_LOCK.read_text().strip()
    if CODE_REVISION_LOCK.exists()
    else None
)
subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', 'origin', BRANCH], check=True)
revision_ref = REQUESTED_CODE_REVISION or locked_revision or f'origin/{BRANCH}'
CODE_REVISION = subprocess.run(
    ['git', '-C', str(REPO_DIR), 'rev-parse', f'{revision_ref}^{{commit}}'],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if locked_revision and CODE_REVISION != locked_revision:
    raise RuntimeError(
        f'P0-D2 code lock mismatch: {locked_revision} != {CODE_REVISION}. '
        'Use a new P0D2_PIPELINE_ATTEMPT.'
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
print('Checked out P0-D2 code:', CODE_REVISION)
"""


PROVENANCE_SOURCE = """
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

source_matches = sorted(
    P0C_PIPELINE_ROOT.glob(
        'runs/*/confirmatory-'
        + P0C_CONFIRMATORY_ATTEMPT
        + '/manifest.json'
    )
)
if len(source_matches) != 1:
    raise RuntimeError(
        'Expected exactly one P0-C Confirmatory manifest under '
        f'{P0C_PIPELINE_ROOT}, found {source_matches}'
    )
SOURCE_P0C_MANIFEST = source_matches[0]
source_manifest = json.loads(SOURCE_P0C_MANIFEST.read_text())
source_lessons = source_manifest.get('selected_lessons', [])
source_units = source_manifest.get('units', {})
source_config = source_manifest.get('config', {})
SOURCE_MODEL_REVISION = source_config.get('model_revision')
if (
    source_config.get('tier') != 'confirmatory'
    or len(source_lessons) != 24
    or len(source_units) != 72
    or set(source_manifest.get('base_arms', {}).values()) != {'verified'}
    or any(
        unit.get('state') != 'verified'
        or float(unit.get('rollback_exact_match_rate', 0.0)) != 1.0
        for unit in source_units.values()
    )
):
    raise RuntimeError('P0-C source manifest is not a complete verified Confirmatory run')
SOURCE_MANIFEST_HASH = sha256(SOURCE_P0C_MANIFEST.read_bytes()).hexdigest()
SELECTED_LESSON_HASH = sha256(
    json.dumps(source_lessons, separators=(',', ':')).encode()
).hexdigest()

environment_context = run_json(['uv', 'run', 'plasticity-p0d2', 'environment'])
environment = environment_context['environment']
ENVIRONMENT_FINGERPRINT = environment_context['fingerprint']
CODE_HASH = environment['code_sha256']

FROZEN_MATRIX_ID = 'p0d2-seven-condition-budget-match-v1'
MATRIX_HASH = sha256(FROZEN_MATRIX_ID.encode()).hexdigest()
PROVENANCE_KEY = (
    f'code-{CODE_HASH[:10]}_source-{SOURCE_MANIFEST_HASH[:10]}'
)
PROVENANCE_ROOT = P0D2_PIPELINE_ROOT / 'runs' / PROVENANCE_KEY
STAGE_DIR = PROVENANCE_ROOT / ('budget_match-' + BUDGET_MATCH_ATTEMPT)
LOG_DIR = PROVENANCE_ROOT / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

frozen_context = {
    'schema_version': 1,
    'pipeline_version': 'p0d2-budget-match-v1',
    'code_sha256': CODE_HASH,
    'source_manifest_sha256': SOURCE_MANIFEST_HASH,
    'source_model_revision': SOURCE_MODEL_REVISION,
    'selected_lesson_sha256': SELECTED_LESSON_HASH,
    'matrix_sha256': MATRIX_HASH,
    'retention_margin': RETENTION_MARGIN,
    'budget_tolerance': BUDGET_TOLERANCE,
    'stage': 'budget_match',
}
context_path = STAGE_DIR / 'notebook_context.json'
STAGE_DIR.mkdir(parents=True, exist_ok=True)
if context_path.exists():
    if json.loads(context_path.read_text()) != frozen_context:
        raise RuntimeError(
            f'Frozen P0-D2 stage context mismatch: {context_path}'
        )
else:
    temporary = context_path.with_suffix('.json.tmp')
    temporary.write_text(
        json.dumps(frozen_context, indent=2, sort_keys=True) + '\\n'
    )
    temporary.replace(context_path)

sessions_path = STAGE_DIR / 'source_sessions.json'
sessions = json.loads(sessions_path.read_text()) if sessions_path.exists() else []
sessions.append({
    'recorded_at': datetime.now(timezone.utc).isoformat(),
    'git_revision': CODE_REVISION,
    'branch': BRANCH,
    'stage': 'budget_match',
    'environment_fingerprint': ENVIRONMENT_FINGERPRINT,
    'environment': environment,
    'source_p0c_manifest': str(SOURCE_P0C_MANIFEST),
})
temporary = sessions_path.with_suffix('.json.tmp')
temporary.write_text(json.dumps(sessions, indent=2, sort_keys=True) + '\\n')
temporary.replace(sessions_path)

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
        try:
            for line in process.stdout:
                print(line, end='')
                log_file.write(line)
                log_file.flush()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return_code = process.wait()
    elapsed = time.monotonic() - started
    print(f'[{label}] exit={return_code} elapsed={elapsed / 60:.1f} min')
    if return_code != 0:
        raise RuntimeError(
            f'{label} failed with exit code {return_code}; log={log_path}'
        )

def p0d2_command(action):
    if sha256(SOURCE_P0C_MANIFEST.read_bytes()).hexdigest() != SOURCE_MANIFEST_HASH:
        raise RuntimeError('P0-C source manifest changed after provenance freeze')
    command = [
        'uv', 'run', 'plasticity-p0d2', action,
        '--output', str(STAGE_DIR),
    ]
    if action in {'plan', 'run'}:
        command.extend([
            '--source-manifest', str(SOURCE_P0C_MANIFEST),
            '--retention-margin', str(RETENTION_MARGIN),
            '--budget-tolerance', str(BUDGET_TOLERANCE),
        ])
    return command

def manifest_status():
    path = STAGE_DIR / 'manifest.json'
    if not path.exists():
        return {'exists': False, 'stage_dir': str(STAGE_DIR)}
    manifest = json.loads(path.read_text())
    return {
        'exists': True,
        'run_id': manifest.get('run_id'),
        'stage_dir': str(STAGE_DIR),
        'lesson_count': len(manifest.get('selected_lessons', [])),
        'conditions': list(manifest.get('conditions', {})),
        'base_states': dict(Counter(manifest.get('base_arms', {}).values())),
        'unit_states': dict(Counter(
            unit.get('state') for unit in manifest.get('units', {}).values()
        )),
        'errors': manifest.get('errors', [])[-10:],
    }

def verify_manifest():
    path = STAGE_DIR / 'manifest.json'
    if not path.exists():
        raise FileNotFoundError(f'Missing P0-D2 manifest: {path}')
    manifest = json.loads(path.read_text())
    units = manifest.get('units', {})
    if len(manifest.get('selected_lessons', [])) != 24:
        raise RuntimeError('P0-D2 manifest does not contain 24 lessons')
    if len(manifest.get('conditions', {})) != 7:
        raise RuntimeError('P0-D2 manifest does not contain 7 conditions')
    if len(units) != 504:
        raise RuntimeError(f'Expected 504 adapter units, got {len(units)}')
    if set(manifest.get('base_arms', {}).values()) != {'verified'}:
        raise RuntimeError('P0-D2 base arms are incomplete')
    bad = {
        key: unit.get('state')
        for key, unit in units.items()
        if unit.get('state') != 'verified'
        or float(unit.get('rollback_exact_match_rate', 0.0)) != 1.0
    }
    matched_bad = {
        key: unit.get('budget_validation')
        for key, unit in units.items()
        if str(unit.get('condition_id', '')).endswith('-matched')
        and (
            unit.get('budget_validation', {}).get('within_tolerance') is not True
            or float(
                unit.get('budget_validation', {}).get('relative_error', 1.0)
            ) > BUDGET_TOLERANCE
        )
    }
    if bad or matched_bad or manifest.get('errors'):
        raise RuntimeError(
            f'P0-D2 invalid units={bad}, budget={matched_bad}, '
            f'errors={manifest.get("errors", [])[-5:]}'
        )
    return manifest

print(json.dumps({
    'code_revision': CODE_REVISION,
    'code_sha256': CODE_HASH,
    'source_manifest': str(SOURCE_P0C_MANIFEST),
    'source_manifest_sha256': SOURCE_MANIFEST_HASH,
    'source_model_revision': SOURCE_MODEL_REVISION,
    'source_base_rank': source_config.get('rank'),
    'source_base_alpha': source_config.get('alpha'),
    'selected_lesson_sha256': SELECTED_LESSON_HASH,
    'matrix_sha256': MATRIX_HASH,
    'budget_tolerance': BUDGET_TOLERANCE,
    'environment_fingerprint': ENVIRONMENT_FINGERPRINT,
    'stage_dir': str(STAGE_DIR),
}, indent=2))
"""


RUN_SOURCE = """
plan = run_json(p0d2_command('plan'))
expected_ids = {
    'full-base',
    'early-base',
    'middle-base',
    'late-base',
    'early-matched',
    'middle-matched',
    'late-matched',
}
conditions = plan['config']['conditions']
if plan['expected_unit_count'] != 504:
    raise RuntimeError(
        f"P0-D2 expected 504 units, got {plan['expected_unit_count']}"
    )
if plan['expected_probe_row_count'] != 27456:
    raise RuntimeError(
        'P0-D2 expected 27,456 rows, got '
        f"{plan['expected_probe_row_count']}"
    )
if {condition['condition_id'] for condition in conditions} != expected_ids:
    raise RuntimeError('P0-D2 frozen condition IDs changed')
matched = [
    condition
    for condition in conditions
    if condition['condition_id'].endswith('-matched')
]
if any(
    condition['nominal_budget_relative_error'] > BUDGET_TOLERANCE
    for condition in matched
):
    raise RuntimeError('P0-D2 nominal parameter budget preflight failed')
display(conditions)

if RUN_BUDGET_MATCH:
    run_checked('p0d2-budget-match-run', p0d2_command('run'))
    verify_manifest()
    run_checked(
        'p0d2-budget-match-aggregate',
        [
            'uv', 'run', 'plasticity-p0d2', 'aggregate',
            '--output', str(STAGE_DIR),
            '--bootstrap-samples', '10000',
        ],
    )

print(json.dumps(manifest_status(), ensure_ascii=False, indent=2))
summary_path = STAGE_DIR / 'results' / 'aggregate' / 'summary.json'
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    if (
        summary['adapter_unit_count'] != 504
        or summary['probe_row_count'] != 27456
    ):
        raise RuntimeError('Unexpected P0-D2 result dimensions')
    if not summary['gates']['run_valid']:
        raise RuntimeError('P0-D2 aggregate did not pass the run-valid gate')
    display(summary['condition_summary'])
    display(summary['contrast_families'])
    display(summary['contrasts'])
    display(summary['budget_validation'])
    display(summary['gates'])
    print(
        'Next-stage decision file:',
        STAGE_DIR / 'results' / 'aggregate' / 'next_stage_candidates.json',
    )
else:
    print('Dry preflight complete. Set RUN_BUDGET_MATCH = True to start GPU work.')
"""


def build_notebook() -> dict[str, Any]:
    cells = [
        _markdown(
            f"""
            <a href="{COLAB_URL}" target="_parent"><img
            src="https://colab.research.google.com/assets/colab-badge.svg"
            alt="Open In Colab"/></a>

            # P0-D2: LoRA Parameter-Budget Match

            Frozen seven-condition matrix on the verified P0-C Confirmatory cohort.
            This notebook does not modify P0-C or P0-D1 outputs and does not
            automatically start a narrow scan.
            """
        ),
        _markdown(
            """
            ## 1. Configuration

            Select a GPU runtime. Point the P0-C attempt labels at the complete
            verified Confirmatory run. Keep `RUN_BUDGET_MATCH = False` for the
            first preflight.
            """
        ),
        _code(CONFIGURATION_SOURCE),
        _markdown("## 2. Checkout and install frozen code"),
        _code(CHECKOUT_SOURCE),
        _markdown("## 3. Resolve source provenance and recovery directory"),
        _code(PROVENANCE_SOURCE),
        _markdown("## 4. Preflight, run, verify, and aggregate"),
        _code(RUN_SOURCE),
        _markdown(
            """
            ## Recovery policy

            A normal disconnect resumes verified units in the same attempt when
            code, source manifest, matrix, tolerances, and environment match.
            Preserve an attempt containing a `failed` unit and increment
            `BUDGET_MATCH_ATTEMPT`. Never edit manifests, adapters, raw rows,
            aggregate files, or frozen context in place.

            P0-D2 results remain `TBD` until the complete verified aggregate exists.
            """
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": NOTEBOOK_NAME,
                "provenance": [],
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / NOTEBOOK_NAME
    path.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
