from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2_hard_probe_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/p0d2_hard_probe/{NOTEBOOK_NAME}"
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

P0D2_PIPELINE_ATTEMPT = 'pipeline-a1'
P0D2_BUDGET_MATCH_ATTEMPT = 'a1'
P0D2_PIPELINE_ROOT = (
    Path('/content/drive/MyDrive/plasticity-p0d/budget-match/v1/pipelines')
    / P0D2_PIPELINE_ATTEMPT
)

P0D2H_PIPELINE_ATTEMPT = 'pipeline-a1'
HARD_PROBE_ATTEMPT = 'a1'
P0D2H_PIPELINE_ROOT = (
    Path('/content/drive/MyDrive/plasticity-p0d/hard-probe/v1/pipelines')
    / P0D2H_PIPELINE_ATTEMPT
)

RESILIENCE_MARGIN = 0.05
RUN_HARD_PROBE = False

for name, value in {{
    'P0D2_PIPELINE_ATTEMPT': P0D2_PIPELINE_ATTEMPT,
    'P0D2_BUDGET_MATCH_ATTEMPT': P0D2_BUDGET_MATCH_ATTEMPT,
    'P0D2H_PIPELINE_ATTEMPT': P0D2H_PIPELINE_ATTEMPT,
    'HARD_PROBE_ATTEMPT': HARD_PROBE_ATTEMPT,
}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
if not 0.0 <= RESILIENCE_MARGIN <= 1.0:
    raise ValueError('RESILIENCE_MARGIN must be between 0 and 1')
"""


CHECKOUT_SOURCE = """
subprocess.run(['nvidia-smi'], check=True)
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

P0D2H_PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
CODE_REVISION_LOCK = P0D2H_PIPELINE_ROOT / 'code_revision.txt'
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
        f'P0-D2H code lock mismatch: {locked_revision} != {CODE_REVISION}. '
        'Use a new P0D2H_PIPELINE_ATTEMPT.'
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
print('Checked out P0-D2H code:', CODE_REVISION)
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
    P0D2_PIPELINE_ROOT.glob(
        'runs/*/budget_match-'
        + P0D2_BUDGET_MATCH_ATTEMPT
        + '/manifest.json'
    )
)
if len(source_matches) != 1:
    raise RuntimeError(
        'Expected exactly one P0-D2 budget-match manifest under '
        f'{P0D2_PIPELINE_ROOT}, found {source_matches}'
    )
SOURCE_P0D2_MANIFEST = source_matches[0]
source_manifest = json.loads(SOURCE_P0D2_MANIFEST.read_text())
source_units = source_manifest.get('units', {})
source_config = source_manifest.get('config', {})
source_base = source_manifest.get('base_arms', {})
source_summary_path = (
    SOURCE_P0D2_MANIFEST.parent / 'results' / 'aggregate' / 'summary.json'
)
if not source_summary_path.exists():
    raise FileNotFoundError(f'Missing P0-D2 summary: {source_summary_path}')
source_summary = json.loads(source_summary_path.read_text())
if (
    source_manifest.get('schema_version') != 'p0d2-manifest-v1'
    or source_config.get('stage') != 'budget_match'
    or len(source_manifest.get('selected_lessons', [])) != 24
    or len(source_manifest.get('conditions', {})) != 7
    or len(source_units) != 504
    or set(source_base.values()) != {'verified'}
    or any(
        unit.get('state') != 'verified'
        or float(unit.get('rollback_exact_match_rate', 0.0)) != 1.0
        for unit in source_units.values()
    )
    or source_summary.get('run_id') != source_manifest.get('run_id')
    or source_summary.get('gates', {}).get('run_valid') is not True
    or 'late-matched'
    not in source_summary.get('gates', {}).get('eligible_locus_conditions', [])
):
    raise RuntimeError(
        'Source is not a complete verified P0-D2 run with late-matched eligible'
    )

SOURCE_MANIFEST_HASH = sha256(SOURCE_P0D2_MANIFEST.read_bytes()).hexdigest()
SOURCE_SUMMARY_HASH = sha256(source_summary_path.read_bytes()).hexdigest()
environment_context = run_json(
    ['uv', 'run', 'plasticity-p0d2h', 'environment']
)
environment = environment_context['environment']
ENVIRONMENT_FINGERPRINT = environment_context['fingerprint']
CODE_HASH = environment['code_sha256']

PROVENANCE_KEY = (
    f'code-{CODE_HASH[:10]}_source-{SOURCE_MANIFEST_HASH[:10]}'
)
PROVENANCE_ROOT = P0D2H_PIPELINE_ROOT / 'runs' / PROVENANCE_KEY
STAGE_DIR = PROVENANCE_ROOT / ('hard_probe-' + HARD_PROBE_ATTEMPT)
LOG_DIR = PROVENANCE_ROOT / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

frozen_context = {
    'schema_version': 1,
    'pipeline_version': 'p0d2h-hard-probe-v1',
    'code_sha256': CODE_HASH,
    'source_manifest_sha256': SOURCE_MANIFEST_HASH,
    'source_summary_sha256': SOURCE_SUMMARY_HASH,
    'source_run_id': source_manifest['run_id'],
    'resilience_margin': RESILIENCE_MARGIN,
    'stage': 'hard_probe',
}
STAGE_DIR.mkdir(parents=True, exist_ok=True)
context_path = STAGE_DIR / 'notebook_context.json'
if context_path.exists():
    if json.loads(context_path.read_text()) != frozen_context:
        raise RuntimeError(f'Frozen P0-D2H context mismatch: {context_path}')
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
    'stage': 'hard_probe',
    'environment_fingerprint': ENVIRONMENT_FINGERPRINT,
    'environment': environment,
    'source_p0d2_manifest': str(SOURCE_P0D2_MANIFEST),
})
temporary = sessions_path.with_suffix('.json.tmp')
temporary.write_text(
    json.dumps(sessions, indent=2, sort_keys=True) + '\\n'
)
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

def p0d2h_command(action):
    if sha256(SOURCE_P0D2_MANIFEST.read_bytes()).hexdigest() != SOURCE_MANIFEST_HASH:
        raise RuntimeError('P0-D2 source manifest changed after freeze')
    if sha256(source_summary_path.read_bytes()).hexdigest() != SOURCE_SUMMARY_HASH:
        raise RuntimeError('P0-D2 source summary changed after freeze')
    command = [
        'uv', 'run', 'plasticity-p0d2h', action,
        '--output', str(STAGE_DIR),
    ]
    if action in {'plan', 'run'}:
        command.extend([
            '--source-manifest', str(SOURCE_P0D2_MANIFEST),
            '--resilience-margin', str(RESILIENCE_MARGIN),
        ])
    return command
"""


RUN_SOURCE = """
plan = run_json(p0d2h_command('plan'))
expected_conditions = [
    'full-base',
    'early-matched',
    'middle-matched',
    'late-matched',
]
if (
    [condition['condition_id'] for condition in plan['config']['conditions']]
    != expected_conditions
    or plan['hard_categories'] != [
        'binding_decoys',
        'conflict_stack',
        'conditional_route',
        'long_context',
    ]
    or plan['probes_per_lesson'] != 16
    or plan['expected_unit_count'] != 288
    or plan['expected_probe_row_count'] != 5376
):
    raise RuntimeError('Unexpected P0-D2H preflight plan')
display(plan)

def manifest_status():
    path = STAGE_DIR / 'manifest.json'
    if not path.exists():
        return {'exists': False, 'stage_dir': str(STAGE_DIR)}
    manifest = json.loads(path.read_text())
    return {
        'exists': True,
        'run_id': manifest.get('run_id'),
        'stage_dir': str(STAGE_DIR),
        'base_states': dict(Counter(
            value.get('state')
            for value in manifest.get('base_arms', {}).values()
        )),
        'unit_states': dict(Counter(
            value.get('state')
            for value in manifest.get('units', {}).values()
        )),
        'errors': manifest.get('errors', [])[-10:],
    }

if RUN_HARD_PROBE:
    run_checked('p0d2h-hard-probe', p0d2h_command('run'))
    manifest = json.loads((STAGE_DIR / 'manifest.json').read_text())
    if (
        len(manifest.get('selected_lessons', [])) != 24
        or len(manifest.get('conditions', {})) != 4
        or len(manifest.get('units', {})) != 288
        or any(
            value.get('state') != 'verified'
            for value in manifest.get('base_arms', {}).values()
        )
        or any(
            value.get('state') != 'verified'
            for value in manifest.get('units', {}).values()
        )
        or manifest.get('errors')
    ):
        raise RuntimeError('P0-D2H manifest is incomplete or invalid')
    run_checked(
        'p0d2h-hard-probe-aggregate',
        [
            'uv', 'run', 'plasticity-p0d2h', 'aggregate',
            '--output', str(STAGE_DIR),
            '--bootstrap-samples', '10000',
        ],
    )

print(json.dumps(manifest_status(), ensure_ascii=False, indent=2))
summary_path = STAGE_DIR / 'results' / 'aggregate' / 'summary.json'
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    if (
        summary['adapter_unit_count'] != 288
        or summary['probe_row_count'] != 5376
        or not summary['gates']['run_valid']
    ):
        raise RuntimeError('Unexpected P0-D2H aggregate dimensions or gate')
    display(summary['condition_summary'])
    display(summary['degradation_vs_original_tg'])
    display(summary['contrasts'])
    display(summary['gates'])
    print(
        'Next-stage decision:',
        STAGE_DIR / 'results' / 'aggregate' / 'next_stage_decision.json',
    )
else:
    print('Dry preflight complete. Set RUN_HARD_PROBE = True to evaluate.')
"""


def build_notebook() -> dict[str, Any]:
    cells = [
        _markdown(
            f"""
            <a href="{COLAB_URL}" target="_parent"><img
            src="https://colab.research.google.com/assets/colab-badge.svg"
            alt="Open In Colab"/></a>

            # P0-D2H: Read-Only Hard-Probe Stress Test

            Evaluate verified P0-D2 adapters on four frozen difficulty dimensions.
            This notebook never retrains adapters, modifies P0-D2 outputs, starts a
            narrow scan, or starts a multi-mapping experiment.
            """
        ),
        _markdown(
            """
            ## 1. Configuration

            Select a GPU runtime. Point the P0-D2 attempt labels at the complete
            verified budget-match run. Keep `RUN_HARD_PROBE = False` for the first
            preflight.
            """
        ),
        _code(CONFIGURATION_SOURCE),
        _markdown("## 2. Checkout and install frozen code"),
        _code(CHECKOUT_SOURCE),
        _markdown("## 3. Resolve source provenance and recovery directory"),
        _code(PROVENANCE_SOURCE),
        _markdown("## 4. Preflight, evaluate, verify, and aggregate"),
        _code(RUN_SOURCE),
        _markdown(
            """
            ## Recovery and interpretation

            A normal disconnect resumes verified units in the same attempt when
            code, source hashes, probe bank, and environment match. Preserve any
            attempt containing a `failed` unit and increment `HARD_PROBE_ATTEMPT`.

            P0-D2H increases evaluation difficulty only. A reviewed result may
            motivate a separate multi-mapping training experiment, but this notebook
            cannot start one automatically. All results remain `TBD` until the
            complete verified aggregate exists.
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
