from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_route_remediation_same_runtime_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/"
    "p0d2h_route_remediation_same_runtime/"
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
import re
import shlex
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

from IPython.display import Markdown, display

REPO_URL = 'https://github.com/donleaveher/plasticity-placement.git'
BRANCH = '{BRANCH}'
REQUESTED_ANALYSIS_CODE_REVISION = None
REPO_DIR = Path('/content/plasticity-placement-rr-same-runtime')

# Completed, verified route-remediation source.
RR_PIPELINE_ATTEMPT = 'pipeline-rr1'
RR_ATTEMPT = 'rr1'
EXPECTED_RR_RUN_ID = 'p0d2hrr-902b15bc17'
RR_PIPELINE_ROOT = (
    Path('/content/drive/MyDrive/plasticity-p0d/route-remediation-lora/v1/pipelines')
    / RR_PIPELINE_ATTEMPT
)
RR_OUTPUT = RR_PIPELINE_ROOT / 'runs' / ('route-remediation-' + RR_ATTEMPT)
RR_CODE_REVISION_LOCK = RR_PIPELINE_ROOT / 'code_revision.txt'

# New, independent inference-only output.
ANALYSIS_PIPELINE_ATTEMPT = 'pipeline-sr1'
ANALYSIS_ATTEMPT = 'sr1'
ANALYSIS_PIPELINE_ROOT = (
    Path(
        '/content/drive/MyDrive/plasticity-p0d/'
        'route-remediation-same-runtime/v1/pipelines'
    )
    / ANALYSIS_PIPELINE_ATTEMPT
)

RUN_AUDIT = True
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20260801
COMBINED_NONINFERIORITY_MARGIN = 0.02
SENTINEL_SCORE_TOLERANCE = 1e-5

for name, value in {{
    'RR_PIPELINE_ATTEMPT': RR_PIPELINE_ATTEMPT,
    'RR_ATTEMPT': RR_ATTEMPT,
    'ANALYSIS_PIPELINE_ATTEMPT': ANALYSIS_PIPELINE_ATTEMPT,
    'ANALYSIS_ATTEMPT': ANALYSIS_ATTEMPT,
}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
if BOOTSTRAP_SAMPLES <= 0 or BOOTSTRAP_SEED < 0:
    raise ValueError('Invalid bootstrap configuration')
if not 0 < COMBINED_NONINFERIORITY_MARGIN < 0.5:
    raise ValueError('Invalid combined non-inferiority margin')

print('RR source:', RR_OUTPUT)
print('Analysis root:', ANALYSIS_PIPELINE_ROOT)
print('Frozen matrix: 96 external conditional-route + 1,536 CRD pairs')
print('Manipulation: adapter OFF -> ON per prompt in one loaded model object')
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
    raise RuntimeError(f'Analysis checkout has local changes:\\n{dirty}')

ANALYSIS_PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
ANALYSIS_CODE_REVISION_LOCK = ANALYSIS_PIPELINE_ROOT / 'code_revision.txt'
locked_revision = (
    ANALYSIS_CODE_REVISION_LOCK.read_text().strip()
    if ANALYSIS_CODE_REVISION_LOCK.exists()
    else None
)
subprocess.run(
    ['git', '-C', str(REPO_DIR), 'fetch', 'origin', BRANCH],
    check=True,
)
revision_ref = REQUESTED_ANALYSIS_CODE_REVISION or locked_revision or f'origin/{BRANCH}'
ANALYSIS_CODE_REVISION = subprocess.run(
    ['git', '-C', str(REPO_DIR), 'rev-parse', f'{revision_ref}^{{commit}}'],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if locked_revision and ANALYSIS_CODE_REVISION != locked_revision:
    raise RuntimeError(
        f'Analysis code lock mismatch: {locked_revision} != '
        f'{ANALYSIS_CODE_REVISION}. Use a new ANALYSIS_PIPELINE_ATTEMPT.'
    )
if not locked_revision:
    temporary = ANALYSIS_CODE_REVISION_LOCK.with_suffix('.txt.tmp')
    temporary.write_text(ANALYSIS_CODE_REVISION + '\\n')
    temporary.replace(ANALYSIS_CODE_REVISION_LOCK)
subprocess.run(
    ['git', '-C', str(REPO_DIR), 'checkout', '--detach', ANALYSIS_CODE_REVISION],
    check=True,
)
subprocess.run(
    ['uv', 'sync', '--extra', 'train', '--extra', 'colab'],
    cwd=REPO_DIR,
    check=True,
)
subprocess.run(['nvidia-smi'], check=True)

def run_json(command):
    completed = subprocess.run(
        command,
        cwd=REPO_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f'Command returned no JSON: {shlex.join(command)}')
    return json.loads(lines[-1])

environment_context = run_json(
    ['uv', 'run', 'plasticity-p0d2hrr', 'environment']
)
if environment_context['environment']['cuda_available'] is not True:
    raise RuntimeError('Select a Colab GPU runtime before running this notebook')
display({
    'analysis_code_revision': ANALYSIS_CODE_REVISION,
    'analysis_code_sha256': environment_context['environment']['code_sha256'],
    'gpu': environment_context['environment']['gpu'],
    'cuda_version': environment_context['environment']['cuda_version'],
})
"""


SOURCE_DISCOVERY = """
RR_MANIFEST = RR_OUTPUT / 'manifest.json'
if not RR_MANIFEST.is_file():
    raise FileNotFoundError(f'Missing route-remediation manifest: {RR_MANIFEST}')
rr_manifest = json.loads(RR_MANIFEST.read_text())
if (
    rr_manifest.get('schema_version') != 'p0d2hrr-manifest-v1'
    or rr_manifest.get('run_id') != EXPECTED_RR_RUN_ID
    or rr_manifest.get('state') != 'verified'
    or rr_manifest.get('errors') != []
):
    raise RuntimeError('Route-remediation source is not the expected verified run')
if not RR_CODE_REVISION_LOCK.is_file():
    raise FileNotFoundError(f'Missing original RR code lock: {RR_CODE_REVISION_LOCK}')

rr_manifest_sha256 = sha256(RR_MANIFEST.read_bytes()).hexdigest()
rr_adapter_sha256 = rr_manifest.get('training', {}).get('summary', {}).get(
    'adapter_sha256'
)
if not isinstance(rr_adapter_sha256, str) or not re.fullmatch(
    r'[0-9a-f]{64}', rr_adapter_sha256
):
    raise RuntimeError('Route-remediation adapter hash is missing or invalid')
SETTINGS = {
    'bootstrap_samples': BOOTSTRAP_SAMPLES,
    'bootstrap_seed': BOOTSTRAP_SEED,
    'combined_noninferiority_margin': COMBINED_NONINFERIORITY_MARGIN,
    'sentinel_score_tolerance': SENTINEL_SCORE_TOLERANCE,
}
settings_sha256 = sha256(
    json.dumps(SETTINGS, sort_keys=True, separators=(',', ':')).encode()
).hexdigest()
PROVENANCE_KEY = (
    f'code-{ANALYSIS_CODE_REVISION[:10]}_rr-{rr_manifest_sha256[:10]}_'
    f'settings-{settings_sha256[:10]}'
)
ANALYSIS_OUTPUT = (
    ANALYSIS_PIPELINE_ROOT / 'runs' / PROVENANCE_KEY
    / ('same-runtime-' + ANALYSIS_ATTEMPT)
)
SUMMARY_PATH = ANALYSIS_OUTPUT / 'summary.json'
AUDIT_MANIFEST_PATH = ANALYSIS_OUTPUT / 'audit_manifest.json'
if ANALYSIS_OUTPUT.exists() and not AUDIT_MANIFEST_PATH.is_file():
    raise RuntimeError(
        f'Uncommitted partial output exists at {ANALYSIS_OUTPUT}; inspect it and use a new '
        'ANALYSIS_ATTEMPT rather than overwriting.'
    )

EXPECTED_IDENTITY = {
    'schema_version': 'p0d2hrr-same-runtime-audit-v1',
    'analysis_code_sha256': environment_context['environment']['code_sha256'],
    'route_remediation_run_id': rr_manifest['run_id'],
    'route_remediation_manifest_sha256': rr_manifest_sha256,
    'adapter_sha256': rr_adapter_sha256,
    **SETTINGS,
    'matrix': {
        'forced_choice': 'external_conditional_route_only',
        'crd': 'complete_route_retrieval_combined',
        'pair_order': 'adapter_off_then_adapter_on_per_prompt',
    },
}

print(json.dumps({
    'rr_run_id': rr_manifest['run_id'],
    'rr_manifest_sha256': rr_manifest_sha256,
    'rr_code_revision': RR_CODE_REVISION_LOCK.read_text().strip(),
    'analysis_code_revision': ANALYSIS_CODE_REVISION,
    'settings_sha256': settings_sha256,
    'analysis_output': str(ANALYSIS_OUTPUT),
    'existing_complete_result': AUDIT_MANIFEST_PATH.is_file(),
}, ensure_ascii=False, indent=2))
"""


RUN_SOURCE = """
def run_streaming_json(command):
    print('$', shlex.join(command), flush=True)
    process = subprocess.Popen(
        command,
        cwd=REPO_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end='', flush=True)
        if line.strip():
            lines.append(line.strip())
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f'Command failed ({return_code}): {shlex.join(command)}')
    if not lines:
        raise RuntimeError('Same-runtime command returned no output')
    return json.loads(lines[-1])

def canonical_hash(value):
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ).encode()
    ).hexdigest()

def directory_hash(root):
    files = sorted(path for path in root.rglob('*') if path.is_file())
    if not files:
        raise RuntimeError(f'Cannot snapshot empty source tree: {root}')
    digest = sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {'path': str(root), 'file_count': len(files), 'sha256': digest.hexdigest()}

def current_source_snapshot():
    frozen_fc_output = Path(rr_manifest['source_manifest_path']).resolve().parent
    return {
        'route_remediation': directory_hash(RR_OUTPUT.resolve()),
        'frozen_forced_choice': directory_hash(frozen_fc_output),
        'experiment_code_revision_lock': {
            'path': str(RR_CODE_REVISION_LOCK.resolve()),
            'sha256': sha256(RR_CODE_REVISION_LOCK.read_bytes()).hexdigest(),
        },
    }

def verify_complete_output():
    if not AUDIT_MANIFEST_PATH.is_file():
        raise RuntimeError('Missing audit_manifest.json commit marker')
    manifest = json.loads(AUDIT_MANIFEST_PATH.read_text())
    expected_run_id = 'p0d2hrr-same-runtime-' + canonical_hash(EXPECTED_IDENTITY)[:10]
    if (
        manifest.get('schema_version')
        != 'p0d2hrr-same-runtime-audit-manifest-v1'
        or manifest.get('run_id') != expected_run_id
        or manifest.get('identity') != EXPECTED_IDENTITY
        or manifest.get('source_artifacts_modified') is not False
        or manifest.get('source_gate_status_changed') is not False
        or manifest.get('training_authorized') is not False
        or manifest.get('mappings_per_adapter_authorized') is not False
    ):
        raise RuntimeError('Audit manifest identity or boundaries differ')
    artifacts = manifest.get('artifacts')
    expected_artifacts = {
        'summary': SUMMARY_PATH,
        'report': ANALYSIS_OUTPUT / 'report.md',
        'off_forced_choice': ANALYSIS_OUTPUT / 'results' / 'adapter_off_fc.jsonl',
        'on_forced_choice': ANALYSIS_OUTPUT / 'results' / 'adapter_on_fc.jsonl',
        'off_crd': ANALYSIS_OUTPUT / 'results' / 'adapter_off_crd.jsonl',
        'on_crd': ANALYSIS_OUTPUT / 'results' / 'adapter_on_crd.jsonl',
        'paired_records': ANALYSIS_OUTPUT / 'paired_records.jsonl',
        'combined_failures': ANALYSIS_OUTPUT / 'combined_failure_records.jsonl',
    }
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_artifacts):
        raise RuntimeError('Audit manifest artifact inventory differs')
    for name, record in artifacts.items():
        artifact_path = Path(record['path']).resolve()
        if artifact_path != expected_artifacts[name].resolve():
            raise RuntimeError(f'Artifact path differs: {name}')
        if (
            not artifact_path.is_file()
            or sha256(artifact_path.read_bytes()).hexdigest() != record['sha256']
        ):
            raise RuntimeError(f'Artifact hash differs: {name}')
    summary = json.loads(SUMMARY_PATH.read_text())
    snapshot = current_source_snapshot()
    if (
        summary.get('run_id') != expected_run_id
        or summary.get('source_snapshot_before') != snapshot
        or summary.get('source_snapshot_after') != snapshot
        or manifest.get('source_snapshot_sha256') != canonical_hash(snapshot)
    ):
        raise RuntimeError('Current source or committed source snapshot differs')
    return summary

if AUDIT_MANIFEST_PATH.is_file():
    print('Complete immutable result already exists; reusing it.')
elif RUN_AUDIT:
    result = run_streaming_json([
        'uv', 'run', 'plasticity-p0d2hrr', 'same-runtime-audit',
        '--output', str(RR_OUTPUT),
        '--analysis-output', str(ANALYSIS_OUTPUT),
        '--experiment-code-revision-lock', str(RR_CODE_REVISION_LOCK),
        '--bootstrap-samples', str(BOOTSTRAP_SAMPLES),
        '--bootstrap-seed', str(BOOTSTRAP_SEED),
        '--combined-noninferiority-margin',
        str(COMBINED_NONINFERIORITY_MARGIN),
        '--sentinel-score-tolerance', str(SENTINEL_SCORE_TOLERANCE),
    ])
    if Path(result['output_path']) != SUMMARY_PATH:
        raise RuntimeError(f'Unexpected summary path: {result}')
else:
    raise RuntimeError('RUN_AUDIT=False and no complete result exists')

summary = verify_complete_output()
if (
    summary.get('schema_version') != 'p0d2hrr-same-runtime-audit-v1'
    or summary.get('analysis_status') != 'same_runtime_adapter_off_on_complete'
    or summary.get('source_artifacts_modified') is not False
    or summary.get('source_gate_status_changed') is not False
    or summary.get('training_authorized') is not False
    or summary.get('mappings_per_adapter_authorized') is not False
    or summary.get('runtime_identity', {}).get('single_loaded_base_model') is not True
    or summary.get('runtime_identity', {}).get('sentinel', {}).get('passed') is not True
):
    raise RuntimeError('Same-runtime output invariants failed')

display(Markdown((ANALYSIS_OUTPUT / 'report.md').read_text()))
display({
    'summary_path': str(SUMMARY_PATH),
    'audit_manifest': str(ANALYSIS_OUTPUT / 'audit_manifest.json'),
    'paired_records': str(ANALYSIS_OUTPUT / 'paired_records.jsonl'),
    'decision': summary['decision']['status'],
    'next_action': summary['decision']['next_action'],
    'historical_gate_changed': summary['decision']['historical_gate_changed'],
    'mappings_per_adapter_authorized': (
        summary['decision']['mappings_per_adapter_authorized']
    ),
})
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            _markdown(
                f"""
                # Same-runtime adapter OFF/ON qualification audit

                [Open this notebook in Colab]({COLAB_URL})

                This is the final qualification experiment before any 1/4/8
                mappings-per-adapter work. It loads the verified 1.5B base model
                once, attaches the fixed route-remediation adapter once, and
                scores every locked prompt as adjacent adapter OFF → ON pairs.

                It runs 96 external conditional-route pairs and the complete
                1,536-row CRD panel. It performs no training and cannot authorize
                the 1/4/8 experiment.
                """
            ),
            _markdown(
                """
                ## 1. Colab runtime

                Select a GPU runtime, then use **Runtime → Run all**. The Drive
                source run is read-only; all new artifacts go to an independent
                provenance-keyed directory.
                """
            ),
            _code(CONFIGURATION_SOURCE),
            _markdown(
                """
                ## 2. Lock and install analysis code

                The notebook locks an exact Git commit before installing the
                training/Colab extras. Reusing a pipeline attempt requires the
                same commit.
                """
            ),
            _code(CHECKOUT_SOURCE),
            _markdown(
                """
                ## 3. Verify the frozen route-remediation source

                The source must be the exact completed verified run. A partial
                destination is never overwritten.
                """
            ),
            _code(SOURCE_DISCOVERY),
            _markdown(
                """
                ## 4. Run the paired GPU audit

                The evaluator validates all source provenance, then uses one
                loaded model object. Each prompt is scored OFF and immediately ON.
                Progress is streamed during the long CRD panel.
                """
            ),
            _code(RUN_SOURCE),
            _markdown(
                """
                ## 5. Decision boundary

                A confirmed combined regression leads only to designing a new
                composition-preserving remediation. A non-regression result still
                requires review and, after any remediation, a complete original
                gate rerun. This notebook never starts training or 1/4/8.
                """
            ),
        ],
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
            "language_info": {"name": "python", "version": "3.12"},
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
