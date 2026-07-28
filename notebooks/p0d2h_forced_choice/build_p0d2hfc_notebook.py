from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_forced_choice_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/p0d2h_forced_choice/"
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

# Complete verified P0-D2H-CAL source, read-only.
SOURCE_PIPELINE_ATTEMPT = 'pipeline-c1'
SOURCE_CALIBRATION_ATTEMPT = 'c1'
SOURCE_PIPELINE_ROOT = (
    Path('/content/drive/MyDrive/plasticity-p0d/hard-probe-calibration/v1/pipelines')
    / SOURCE_PIPELINE_ATTEMPT
)

# Independent forced-choice destination.
FORCED_CHOICE_PIPELINE_ATTEMPT = 'pipeline-f1'
FORCED_CHOICE_ATTEMPT = 'f1'
FORCED_CHOICE_PIPELINE_ROOT = (
    Path('/content/drive/MyDrive/plasticity-p0d/hard-probe-forced-choice/v1/pipelines')
    / FORCED_CHOICE_PIPELINE_ATTEMPT
)

BOOTSTRAP_SAMPLES = 10000
RUN_FORMAL_SCORING = True

for name, value in {{
    'SOURCE_PIPELINE_ATTEMPT': SOURCE_PIPELINE_ATTEMPT,
    'SOURCE_CALIBRATION_ATTEMPT': SOURCE_CALIBRATION_ATTEMPT,
    'FORCED_CHOICE_PIPELINE_ATTEMPT': FORCED_CHOICE_PIPELINE_ATTEMPT,
    'FORCED_CHOICE_ATTEMPT': FORCED_CHOICE_ATTEMPT,
}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
if BOOTSTRAP_SAMPLES <= 0:
    raise ValueError('BOOTSTRAP_SAMPLES must be positive')
print(
    'Execution mode:',
    'FORMAL GPU SCORING' if RUN_FORMAL_SCORING else 'CPU PREFLIGHT ONLY',
)
print('Frozen matrix: 2 models × 24 lessons × 3 arms × 16 probes = 2304 rows')
print('Four candidates per row = 9216 candidate sequences')
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

FORCED_CHOICE_PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
CODE_REVISION_LOCK = FORCED_CHOICE_PIPELINE_ROOT / 'code_revision.txt'
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
        f'Forced-choice code lock mismatch: {locked_revision} != '
        f'{CODE_REVISION}. Use a new FORCED_CHOICE_PIPELINE_ATTEMPT.'
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
    ['uv', 'run', 'plasticity-p0d2hfc', 'environment']
)
environment = environment_context['environment']
print(json.dumps({
    'cuda_available': environment['cuda_available'],
    'cuda_version': environment['cuda_version'],
    'gpu': environment['gpu'],
    'packages': environment['packages'],
}, ensure_ascii=False, indent=2))
print('No model has been loaded; checkout/install cell is preflight.')
print('Checked out P0-D2H-CAL-FC code:', CODE_REVISION)
"""


SOURCE_SOURCE = """
source_matches = sorted(
    SOURCE_PIPELINE_ROOT.glob(
        'runs/*/oracle_calibration-'
        + SOURCE_CALIBRATION_ATTEMPT
        + '/manifest.json'
    )
)
if len(source_matches) != 1:
    raise RuntimeError(
        'Expected exactly one complete P0-D2H-CAL source manifest under '
        f'{SOURCE_PIPELINE_ROOT}, found {source_matches}'
    )
SOURCE_MANIFEST = source_matches[0]
SOURCE_STAGE_DIR = SOURCE_MANIFEST.parent
source_manifest = json.loads(SOURCE_MANIFEST.read_text())
source_summary_path = (
    SOURCE_STAGE_DIR / 'results' / 'aggregate' / 'summary.json'
)
source_summary = json.loads(source_summary_path.read_text())
source_audit_path = (
    SOURCE_STAGE_DIR / 'preflight' / 'prompt_token_audit.json'
)
source_audit = json.loads(source_audit_path.read_text())
if (
    source_manifest.get('schema_version') != 'p0d2hc-manifest-v1'
    or len(source_manifest.get('models', {})) != 2
    or len(source_manifest.get('units', {})) != 48
    or any(
        unit.get('state') != 'verified'
        for unit in source_manifest.get('units', {}).values()
    )
    or source_manifest.get('errors')
    or source_summary.get('schema_version') != 'p0d2hc-summary-v1'
    or source_summary.get('probe_row_count') != 2304
    or source_summary.get('gates', {}).get('run_valid') is not True
    or source_audit.get('all_prompts_fit') is not True
):
    raise RuntimeError('P0-D2H-CAL source is not complete and run-valid')

def raw_tree_hash():
    digest = sha256()
    files = sorted((SOURCE_STAGE_DIR / 'results' / 'raw').rglob('*.jsonl'))
    if len(files) != 48:
        raise RuntimeError(f'Expected 48 raw source files, found {len(files)}')
    for path in files:
        digest.update(str(path.relative_to(SOURCE_STAGE_DIR)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()

SOURCE_MANIFEST_HASH = sha256(SOURCE_MANIFEST.read_bytes()).hexdigest()
SOURCE_SUMMARY_HASH = sha256(source_summary_path.read_bytes()).hexdigest()
SOURCE_RAW_HASH = raw_tree_hash()
CODE_HASH = environment['code_sha256']
PROVENANCE_KEY = (
    f'code-{CODE_HASH[:10]}_source-{SOURCE_MANIFEST_HASH[:10]}'
)
PROVENANCE_ROOT = FORCED_CHOICE_PIPELINE_ROOT / 'runs' / PROVENANCE_KEY
STAGE_DIR = PROVENANCE_ROOT / ('forced_choice-' + FORCED_CHOICE_ATTEMPT)
LOG_DIR = PROVENANCE_ROOT / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

frozen_context = {
    'schema_version': 1,
    'pipeline_version': 'p0d2hfc-full-string-sum-logprob-v1',
    'code_sha256': CODE_HASH,
    'source_manifest_sha256': SOURCE_MANIFEST_HASH,
    'source_summary_sha256': SOURCE_SUMMARY_HASH,
    'source_raw_results_sha256': SOURCE_RAW_HASH,
    'source_run_id': source_manifest['run_id'],
    'evaluation_max_length': 512,
    'canonical_leading_whitespace': '',
    'primary_score_definition': 'candidate_token_sum_logprob',
    'gate_thresholds': {
        'oracle_overall': 0.95,
        'oracle_every_category': 0.90,
        'external_overall': 0.85,
        'external_every_category': 0.75,
        'no_write_overall_max': 0.35,
        'tie_error_nonfinite_rate': 0.0,
        'external_minus_no_write_ci_lower_strictly_above': 0.0,
    },
}
STAGE_DIR.mkdir(parents=True, exist_ok=True)
context_path = STAGE_DIR / 'notebook_context.json'
if context_path.exists():
    if json.loads(context_path.read_text()) != frozen_context:
        raise RuntimeError(f'Frozen forced-choice context mismatch: {context_path}')
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

def p0d2hfc_command(action):
    if sha256(SOURCE_MANIFEST.read_bytes()).hexdigest() != SOURCE_MANIFEST_HASH:
        raise RuntimeError('P0-D2H-CAL source manifest changed after freeze')
    if sha256(source_summary_path.read_bytes()).hexdigest() != SOURCE_SUMMARY_HASH:
        raise RuntimeError('P0-D2H-CAL source summary changed after freeze')
    if raw_tree_hash() != SOURCE_RAW_HASH:
        raise RuntimeError('P0-D2H-CAL source raw rows changed after freeze')
    command = [
        'uv', 'run', 'plasticity-p0d2hfc', action,
        '--output', str(STAGE_DIR),
    ]
    if action in {'plan', 'audit', 'run'}:
        command.extend(['--source-manifest', str(SOURCE_MANIFEST)])
    return command
"""


PREFLIGHT_SOURCE = """
plan = run_json(p0d2hfc_command('plan'))
if (
    plan['expected_unit_count'] != 48
    or plan['expected_decision_row_count'] != 2304
    or plan['candidate_count'] != 9216
    or plan['base_only'] is not True
    or plan['adapter_discovery_requested'] is not False
    or plan['automatic_training_started'] is not False
    or plan['automatic_narrow_scan_started'] is not False
):
    raise RuntimeError(f'Unexpected frozen forced-choice plan: {plan}')
print(json.dumps({
    'models': plan['config']['models'],
    'arms': plan['config']['calibration_arms'],
    'hard_categories': plan['config']['hard_categories'],
    'primary_score': plan['config']['primary_score_definition'],
    'gate_thresholds': plan['config']['gate_thresholds'],
    'expected_rows': plan['expected_decision_row_count'],
    'candidate_count': plan['candidate_count'],
}, ensure_ascii=False, indent=2))
print('CPU METADATA PREFLIGHT COMPLETE; no model loaded')
"""


AUDIT_SOURCE = """
print('FULL CANDIDATE-TOKEN AUDIT STARTING ON CPU')
audit_result = run_json(p0d2hfc_command('audit'))
audit = audit_result['audit']
if (
    audit.get('all_checks_passed') is not True
    or audit.get('decision_count') != 2304
    or audit.get('candidate_count') != 9216
    or audit.get('failed_decision_count') != 0
    or audit.get('input_truncated_count') != 0
):
    raise RuntimeError(f'Candidate-token audit failed: {audit}')
print(json.dumps(audit, ensure_ascii=False, indent=2))
print('CPU TOKEN AUDIT COMPLETE; 0 models loaded to CUDA')
"""


RUN_SOURCE = """
if RUN_FORMAL_SCORING:
    if environment.get('cuda_available') is not True:
        raise RuntimeError('Select a GPU runtime before formal scoring')
    print('FORMAL GPU SCORING STARTING: first model CUDA load occurs here')
    subprocess.run(['nvidia-smi'], check=True)
    run_checked('p0d2hfc-formal-run', p0d2hfc_command('run'))
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
        manifest.get('schema_version') != 'p0d2hfc-manifest-v1'
        or len(manifest.get('units', {})) != 48
        or any(
            unit.get('state') != 'verified'
            for unit in manifest.get('units', {}).values()
        )
        or manifest.get('errors')
    ):
        raise RuntimeError('Forced-choice manifest is not complete and verified')
    raw_files = sorted((STAGE_DIR / 'results' / 'raw').rglob('*.jsonl'))
    row_count = sum(
        len([line for line in path.read_text().splitlines() if line.strip()])
        for path in raw_files
    )
    if len(raw_files) != 48 or row_count != 2304:
        raise RuntimeError(
            f'Raw matrix mismatch: files={len(raw_files)} rows={row_count}'
        )
    print('RECOVERY MANIFEST VERIFIED: 48 units, 2304 rows')
"""


AGGREGATE_SOURCE = """
if RUN_FORMAL_SCORING:
    run_checked(
        'p0d2hfc-aggregate',
        p0d2hfc_command('aggregate')
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
    print(json.dumps({
        'model_gates': summary['model_gates'],
        'cross_scale_gate': summary['cross_scale_gate'],
        'next_stage_decision': decision,
        'automatic_training_started': False,
        'automatic_narrow_scan_started': False,
        'summary_path': str(summary_path),
    }, ensure_ascii=False, indent=2))
else:
    print('No aggregate yet. Run the formal and aggregate cells first.')
"""


def build_notebook() -> dict[str, Any]:
    cells = [
        _markdown(
            f"""
            # P0-D2H-CAL-FC: Format-Stable Forced Choice

            [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

            This independent notebook scores the four frozen action strings by
            candidate-only conditional log-likelihood. It reads the complete
            P0-D2H-CAL source without changing it and performs no adapter
            discovery/loading, LoRA training, narrow scan, RLVR, or automatic
            1/4/8-mapping action.

            Run in order. Cells through the full candidate-token audit are CPU
            preflight. The first model CUDA load occurs only in cell 6.
            """
        ),
        _markdown("## 1. Frozen configuration and independent Drive namespace"),
        _code(CONFIGURATION_SOURCE),
        _markdown("## 2. Checkout locked code and install dependencies (preflight)"),
        _code(CHECKOUT_SOURCE),
        _markdown("## 3. Locate and hash-freeze the complete P0-D2H-CAL source"),
        _code(SOURCE_SOURCE),
        _markdown("## 4. CPU metadata preflight"),
        _code(PREFLIGHT_SOURCE),
        _markdown("## 5. Full prompt/candidate-token audit (CPU)"),
        _code(AUDIT_SOURCE),
        _markdown("## 6. Formal full-string candidate scoring (GPU)"),
        _code(RUN_SOURCE),
        _markdown("## 7. Verify atomic recovery and the complete matrix"),
        _code(VERIFY_SOURCE),
        _markdown("## 8. Aggregate lesson-clustered metrics and frozen gates"),
        _code(AGGREGATE_SOURCE),
        _markdown("## 9. Review results; no automatic next stage"),
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
