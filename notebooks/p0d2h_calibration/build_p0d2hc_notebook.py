from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_calibration_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/p0d2h_calibration/"
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

# Read-only source: the complete verified P0-D2H-R run.
P0D2H_PIPELINE_ATTEMPT = 'pipeline-r1'
HARD_PROBE_ATTEMPT = 'r1'
P0D2H_PIPELINE_ROOT = (
    Path('/content/drive/MyDrive/plasticity-p0d/hard-probe/v1/pipelines')
    / P0D2H_PIPELINE_ATTEMPT
)

# Independent destination: never place this under the P0-D2H-R directory.
P0D2HC_PIPELINE_ATTEMPT = 'pipeline-c1'
CALIBRATION_ATTEMPT = 'c1'
P0D2HC_PIPELINE_ROOT = (
    Path('/content/drive/MyDrive/plasticity-p0d/hard-probe-calibration/v1/pipelines')
    / P0D2HC_PIPELINE_ATTEMPT
)

RUN_SCALE_CANARY = True
CANARY_MODEL_NAME = 'Qwen/Qwen2.5-1.5B-Instruct'
CANARY_MODEL_REVISION = None
USE_4BIT = True
EVALUATION_MAX_LENGTH = 512
ORACLE_MIN_ACCURACY = 0.90
ORACLE_MIN_CATEGORY_ACCURACY = 0.80
ORACLE_MAX_INVALID_RATE = 0.01
EXTERNAL_MIN_ACCURACY = 0.75
EXTERNAL_MAX_INVALID_RATE = 0.05
RUN_FORMAL_EVALUATION = True

for name, value in {{
    'P0D2H_PIPELINE_ATTEMPT': P0D2H_PIPELINE_ATTEMPT,
    'HARD_PROBE_ATTEMPT': HARD_PROBE_ATTEMPT,
    'P0D2HC_PIPELINE_ATTEMPT': P0D2HC_PIPELINE_ATTEMPT,
    'CALIBRATION_ATTEMPT': CALIBRATION_ATTEMPT,
}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
if RUN_SCALE_CANARY and not CANARY_MODEL_NAME:
    raise ValueError('RUN_SCALE_CANARY requires CANARY_MODEL_NAME')
print(
    'Execution mode:',
    'FORMAL GPU RUN' if RUN_FORMAL_EVALUATION else 'PREFLIGHT ONLY',
)
print(
    'Expected rows:',
    2304 if RUN_SCALE_CANARY else 1152,
    '(base-only; no LoRA training or adapter loading)',
)
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

P0D2HC_PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
CODE_REVISION_LOCK = P0D2HC_PIPELINE_ROOT / 'code_revision.txt'
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
        f'P0-D2H-CAL code lock mismatch: {locked_revision} != '
        f'{CODE_REVISION}. Use a new P0D2HC_PIPELINE_ATTEMPT.'
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
    ['uv', 'run', 'plasticity-p0d2hc', 'environment']
)
environment = environment_context['environment']
if RUN_FORMAL_EVALUATION and environment.get('cuda_available') is not True:
    raise RuntimeError(
        'PyTorch cannot access CUDA. Select a GPU runtime, restart, and rerun.'
    )
print(json.dumps({{
    'cuda_available': environment['cuda_available'],
    'cuda_version': environment['cuda_version'],
    'gpu': environment['gpu'],
    'packages': environment['packages'],
}}, ensure_ascii=False, indent=2))
print('Checked out P0-D2H-CAL code:', CODE_REVISION)
"""


PROVENANCE_SOURCE = """
source_matches = sorted(
    P0D2H_PIPELINE_ROOT.glob(
        'runs/*/hard_probe-' + HARD_PROBE_ATTEMPT + '/manifest.json'
    )
)
if len(source_matches) != 1:
    raise RuntimeError(
        'Expected exactly one P0-D2H-R manifest under '
        f'{P0D2H_PIPELINE_ROOT}, found {source_matches}'
    )
SOURCE_P0D2H_MANIFEST = source_matches[0]
source_manifest = json.loads(SOURCE_P0D2H_MANIFEST.read_text())
source_summary_path = (
    SOURCE_P0D2H_MANIFEST.parent
    / 'results' / 'aggregate' / 'summary.json'
)
source_audit_path = (
    SOURCE_P0D2H_MANIFEST.parent
    / 'preflight' / 'prompt_token_audit.json'
)
source_summary = json.loads(source_summary_path.read_text())
source_audit = json.loads(source_audit_path.read_text())
if (
    source_manifest.get('schema_version') != 'p0d2h-manifest-v1'
    or source_manifest.get('config', {}).get('schema_version')
    != 'p0d2h-config-v2'
    or len(source_manifest.get('selected_lessons', [])) != 24
    or len(source_manifest.get('units', {})) != 288
    or any(
        unit.get('state') != 'verified'
        for unit in source_manifest.get('units', {}).values()
    )
    or len(source_manifest.get('base_arms', {})) != 24
    or any(
        unit.get('state') != 'verified'
        for unit in source_manifest.get('base_arms', {}).values()
    )
    or source_summary.get('schema_version') != 'p0d2h-summary-v2'
    or source_summary.get('probe_row_count') != 5376
    or source_summary.get('gates', {}).get('run_valid') is not True
    or source_audit.get('all_prompts_fit') is not True
):
    raise RuntimeError('P0-D2H-R source is not complete and verified')

SOURCE_MANIFEST_HASH = sha256(
    SOURCE_P0D2H_MANIFEST.read_bytes()
).hexdigest()
SOURCE_SUMMARY_HASH = sha256(source_summary_path.read_bytes()).hexdigest()
CODE_HASH = environment['code_sha256']
PROVENANCE_KEY = (
    f'code-{CODE_HASH[:10]}_source-{SOURCE_MANIFEST_HASH[:10]}'
)
PROVENANCE_ROOT = P0D2HC_PIPELINE_ROOT / 'runs' / PROVENANCE_KEY
STAGE_DIR = PROVENANCE_ROOT / (
    'oracle_calibration-' + CALIBRATION_ATTEMPT
)
LOG_DIR = PROVENANCE_ROOT / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

frozen_context = {
    'schema_version': 1,
    'pipeline_version': 'p0d2hc-base-only-calibration-v1',
    'code_sha256': CODE_HASH,
    'source_manifest_sha256': SOURCE_MANIFEST_HASH,
    'source_summary_sha256': SOURCE_SUMMARY_HASH,
    'source_run_id': source_manifest['run_id'],
    'run_scale_canary': RUN_SCALE_CANARY,
    'canary_model_name': (
        CANARY_MODEL_NAME if RUN_SCALE_CANARY else None
    ),
    'canary_model_revision_requested': (
        CANARY_MODEL_REVISION if RUN_SCALE_CANARY else None
    ),
    'use_4bit': USE_4BIT,
    'evaluation_max_length': EVALUATION_MAX_LENGTH,
    'oracle_min_accuracy': ORACLE_MIN_ACCURACY,
    'oracle_min_category_accuracy': ORACLE_MIN_CATEGORY_ACCURACY,
    'oracle_max_invalid_rate': ORACLE_MAX_INVALID_RATE,
    'external_min_accuracy': EXTERNAL_MIN_ACCURACY,
    'external_max_invalid_rate': EXTERNAL_MAX_INVALID_RATE,
}
STAGE_DIR.mkdir(parents=True, exist_ok=True)
context_path = STAGE_DIR / 'notebook_context.json'
if context_path.exists():
    if json.loads(context_path.read_text()) != frozen_context:
        raise RuntimeError(
            f'Frozen P0-D2H-CAL context mismatch: {context_path}'
        )
else:
    temporary = context_path.with_suffix('.json.tmp')
    temporary.write_text(
        json.dumps(frozen_context, indent=2, sort_keys=True) + '\\n'
    )
    temporary.replace(context_path)

sessions_path = STAGE_DIR / 'source_sessions.json'
sessions = (
    json.loads(sessions_path.read_text())
    if sessions_path.exists()
    else []
)
sessions.append({
    'recorded_at': datetime.now(timezone.utc).isoformat(),
    'git_revision': CODE_REVISION,
    'branch': BRANCH,
    'stage': 'hard_probe_calibration',
    'environment_fingerprint': environment_context['fingerprint'],
    'source_p0d2h_manifest': str(SOURCE_P0D2H_MANIFEST),
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

def p0d2hc_command(action):
    if sha256(
        SOURCE_P0D2H_MANIFEST.read_bytes()
    ).hexdigest() != SOURCE_MANIFEST_HASH:
        raise RuntimeError('P0-D2H-R source manifest changed after freeze')
    if sha256(
        source_summary_path.read_bytes()
    ).hexdigest() != SOURCE_SUMMARY_HASH:
        raise RuntimeError('P0-D2H-R source summary changed after freeze')
    command = [
        'uv', 'run', 'plasticity-p0d2hc', action,
        '--output', str(STAGE_DIR),
    ]
    if action in {'plan', 'audit', 'run'}:
        command.extend([
            '--source-manifest', str(SOURCE_P0D2H_MANIFEST),
            '--evaluation-max-length', str(EVALUATION_MAX_LENGTH),
            '--oracle-min-accuracy', str(ORACLE_MIN_ACCURACY),
            '--oracle-min-category-accuracy',
            str(ORACLE_MIN_CATEGORY_ACCURACY),
            '--oracle-max-invalid-rate', str(ORACLE_MAX_INVALID_RATE),
            '--external-min-accuracy', str(EXTERNAL_MIN_ACCURACY),
            '--external-max-invalid-rate',
            str(EXTERNAL_MAX_INVALID_RATE),
        ])
        if RUN_SCALE_CANARY:
            command.extend([
                '--canary-model-name', CANARY_MODEL_NAME,
            ])
            if CANARY_MODEL_REVISION:
                command.extend([
                    '--canary-model-revision', CANARY_MODEL_REVISION,
                ])
        if not USE_4BIT:
            command.append('--no-4bit')
    return command
"""


PREFLIGHT_SOURCE = """
plan = run_json(p0d2hc_command('plan'))
expected_models = 2 if RUN_SCALE_CANARY else 1
expected_units = expected_models * 24
expected_rows = expected_models * 1152
if (
    plan['calibration_arms']
    != ['no_write', 'external', 'answer_copy_oracle']
    or plan['expected_unit_count'] != expected_units
    or plan['expected_probe_row_count'] != expected_rows
    or plan['training_requested'] is not False
    or plan['narrow_scan_requested'] is not False
):
    raise RuntimeError(f'Unexpected frozen plan: {plan}')
print(json.dumps({
    'models': plan['config']['models'],
    'arms': plan['calibration_arms'],
    'hard_categories': plan['hard_categories'],
    'expected_units': expected_units,
    'expected_rows': expected_rows,
}, ensure_ascii=False, indent=2))
print('PREFLIGHT COMPLETE')
"""


AUDIT_SOURCE = """
print(
    'TOKEN AUDIT STARTING (CPU tokenizer work; 0 GPU memory is expected here)'
)
audit_result = run_json(p0d2hc_command('audit'))
audit = audit_result['audit']
if (
    audit.get('all_prompts_fit') is not True
    or audit.get('input_truncated_count') != 0
    or audit.get('output_instruction_missing_count') != 0
):
    raise RuntimeError(f'Prompt-token audit failed: {audit}')
print(json.dumps(audit, ensure_ascii=False, indent=2))
print('TOKEN AUDIT COMPLETE')
"""


RUN_SOURCE = """
if RUN_FORMAL_EVALUATION:
    print('FORMAL GPU RUN STARTING')
    subprocess.run(['nvidia-smi'], check=True)
    run_checked('p0d2hc-formal-run', p0d2hc_command('run'))
    subprocess.run(['nvidia-smi'], check=True)
else:
    print('FORMAL RUN SKIPPED: RUN_FORMAL_EVALUATION=False')
"""


VERIFY_SOURCE = """
manifest_path = STAGE_DIR / 'manifest.json'
if not manifest_path.exists():
    if RUN_FORMAL_EVALUATION:
        raise FileNotFoundError(manifest_path)
    print('No manifest expected in preflight-only mode')
else:
    manifest = json.loads(manifest_path.read_text())
    expected_units = (2 if RUN_SCALE_CANARY else 1) * 24
    if (
        manifest.get('schema_version') != 'p0d2hc-manifest-v1'
        or len(manifest.get('units', {})) != expected_units
        or any(
            unit.get('state') != 'verified'
            for unit in manifest.get('units', {}).values()
        )
        or manifest.get('errors')
    ):
        raise RuntimeError('Calibration manifest is not complete and verified')
    print(
        'RECOVERY MANIFEST VERIFIED:',
        len(manifest['units']),
        'lesson-model units',
    )
"""


AGGREGATE_SOURCE = """
if RUN_FORMAL_EVALUATION:
    print('AGGREGATE STARTING')
    run_checked(
        'p0d2hc-aggregate',
        p0d2hc_command('aggregate')
        + ['--bootstrap-samples', '10000'],
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
    print(json.dumps({
        'model_gates': summary['model_gates'],
        'cross_scale_gate': summary['cross_scale_gate'],
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
            # P0-D2H-CAL: Base-Only Oracle Calibration

            [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

            This notebook evaluates the frozen verified P0-D2H-R hard-probe
            cohort with three base-only arms: `no_write`, `external`, and
            `answer_copy_oracle`. It performs no LoRA training, loads no
            adapters, and never starts a layer scan or a 1/4/8-mapping run.

            Run cells in order. Setup and token audit are mostly CPU work;
            GPU memory is expected to rise in the formal-run cell.
            """
        ),
        _markdown("## 1. Frozen configuration and independent Drive namespace"),
        _code(CONFIGURATION_SOURCE),
        _markdown("## 2. Checkout locked code and verify the GPU runtime"),
        _code(CHECKOUT_SOURCE),
        _markdown("## 3. Locate and freeze the verified P0-D2H-R source"),
        _code(PROVENANCE_SOURCE),
        _markdown("## 4. Metadata preflight"),
        _code(PREFLIGHT_SOURCE),
        _markdown("## 5. Prompt-token audit (CPU)"),
        _code(AUDIT_SOURCE),
        _markdown("## 6. Formal base-only evaluation (GPU)"),
        _code(RUN_SOURCE),
        _markdown("## 7. Verify the resumable manifest"),
        _code(VERIFY_SOURCE),
        _markdown("## 8. Aggregate and apply frozen gates"),
        _code(AGGREGATE_SOURCE),
        _markdown("## 9. Review results and next-stage eligibility"),
        _code(RESULTS_SOURCE),
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
