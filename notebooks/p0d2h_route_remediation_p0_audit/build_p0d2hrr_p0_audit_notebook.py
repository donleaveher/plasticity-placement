from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_route_remediation_p0_audit_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/"
    "p0d2h_route_remediation_p0_audit/"
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
REPO_DIR = Path('/content/plasticity-placement-rr-p0-audit')

# Completed verified route-remediation source. This notebook never edits it.
RR_PIPELINE_ATTEMPT = 'pipeline-rr1'
RR_ATTEMPT = 'rr1'
EXPECTED_RR_RUN_ID = 'p0d2hrr-902b15bc17'
HISTORICAL_PREREGISTRATION_SHA256 = (
    'ae456ee26d9ac12c1ba35c0372574d7ae7f276eabba9800ae9af3360f6868986'
)
RR_PIPELINE_ROOT = (
    Path('/content/drive/MyDrive/plasticity-p0d/route-remediation-lora/v1/pipelines')
    / RR_PIPELINE_ATTEMPT
)
RR_OUTPUT = RR_PIPELINE_ROOT / 'runs' / ('route-remediation-' + RR_ATTEMPT)
RR_CODE_REVISION_LOCK = RR_PIPELINE_ROOT / 'code_revision.txt'

# Exact NF4 base comparator. The BF16 retry is not an admissible comparator.
BASE_CRD_PIPELINE_ATTEMPT = 'pipeline-crd1'
BASE_CRD_ATTEMPT = 'crd1'
EXPECTED_BASE_CRD_RUN_ID = 'p0d2hcrd-route-decomposition-8527b64f36'
BASE_CRD_PIPELINE_ROOT = (
    Path(
        '/content/drive/MyDrive/plasticity-p0d/'
        'hard-probe-route-decomposition/v1/pipelines'
    )
    / BASE_CRD_PIPELINE_ATTEMPT
)
BASE_CRD_CODE_REVISION_LOCK = BASE_CRD_PIPELINE_ROOT / 'code_revision.txt'

# Independent, immutable analysis destination.
ANALYSIS_PIPELINE_ATTEMPT = 'pipeline-p0a1'
ANALYSIS_ATTEMPT = 'p0a1'
ANALYSIS_PIPELINE_ROOT = (
    Path(
        '/content/drive/MyDrive/plasticity-p0d/'
        'route-remediation-followup-analysis/v1/pipelines'
    )
    / ANALYSIS_PIPELINE_ATTEMPT
)
BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20260730

for name, value in {{
    'RR_PIPELINE_ATTEMPT': RR_PIPELINE_ATTEMPT,
    'RR_ATTEMPT': RR_ATTEMPT,
    'BASE_CRD_PIPELINE_ATTEMPT': BASE_CRD_PIPELINE_ATTEMPT,
    'BASE_CRD_ATTEMPT': BASE_CRD_ATTEMPT,
    'ANALYSIS_PIPELINE_ATTEMPT': ANALYSIS_PIPELINE_ATTEMPT,
    'ANALYSIS_ATTEMPT': ANALYSIS_ATTEMPT,
}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
if BOOTSTRAP_SAMPLES <= 0 or BOOTSTRAP_SEED < 0:
    raise ValueError('Invalid deterministic bootstrap configuration')

print('RR source:', RR_OUTPUT)
print('NF4 CRD root:', BASE_CRD_PIPELINE_ROOT)
print('Analysis root:', ANALYSIS_PIPELINE_ROOT)
print('Mode: CPU-only, read-only source audit; no model loading or training.')
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
    raise RuntimeError(f'Independent analysis checkout has local changes:\\n{dirty}')

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
revision_ref = (
    REQUESTED_ANALYSIS_CODE_REVISION
    or locked_revision
    or f'origin/{BRANCH}'
)
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
subprocess.run(['uv', 'sync'], cwd=REPO_DIR, check=True)

def run_json(command):
    print('$', shlex.join(command))
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
    ['uv', 'run', 'plasticity-p0d2hrr', 'environment']
)
print(json.dumps({
    'analysis_code_revision': ANALYSIS_CODE_REVISION,
    'analysis_code_sha256': environment_context['environment']['code_sha256'],
    'cuda_available': environment_context['environment']['cuda_available'],
}, ensure_ascii=False, indent=2))
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
    raise FileNotFoundError(
        f'Missing original RR code revision lock: {RR_CODE_REVISION_LOCK}'
    )
if not BASE_CRD_CODE_REVISION_LOCK.is_file():
    raise FileNotFoundError(
        f'Missing original NF4 CRD code revision lock: '
        f'{BASE_CRD_CODE_REVISION_LOCK}'
    )
SOURCE_FC_MANIFEST = Path(rr_manifest['source_manifest_path']).resolve()
SOURCE_FC_OUTPUT = SOURCE_FC_MANIFEST.parent

base_matches = []
for candidate in sorted(
    BASE_CRD_PIPELINE_ROOT.glob(
        'runs/*/route_decomposition-' + BASE_CRD_ATTEMPT + '/manifest.json'
    )
):
    candidate_manifest = json.loads(candidate.read_text())
    model = candidate_manifest.get('model', {})
    units = candidate_manifest.get('units', {})
    if (
        candidate_manifest.get('schema_version') == 'p0d2hcrd-manifest-v1'
        and candidate_manifest.get('run_id') == EXPECTED_BASE_CRD_RUN_ID
        and candidate_manifest.get('errors') == []
        and model.get('model_id') == 'scale_canary'
        and model.get('use_4bit') is True
        and len(units) == 24
        and all(unit.get('state') == 'verified' for unit in units.values())
    ):
        base_matches.append(candidate)
if len(base_matches) != 1:
    raise RuntimeError(
        'Expected exactly one verified NF4 base CRD run, found '
        f'{base_matches}'
    )
BASE_CRD_MANIFEST = base_matches[0]
BASE_CRD_OUTPUT = BASE_CRD_MANIFEST.parent

def directory_hash(root):
    files = sorted(path for path in root.rglob('*') if path.is_file())
    if not files:
        raise RuntimeError(f'Cannot snapshot empty source directory: {root}')
    digest = sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {'file_count': len(files), 'sha256': digest.hexdigest()}

def source_snapshot():
    return {
        'rr_manifest': sha256(RR_MANIFEST.read_bytes()).hexdigest(),
        'rr_preregistration': sha256(
            (RR_OUTPUT / 'preregistration.json').read_bytes()
        ).hexdigest(),
        'rr_authorization': sha256(
            (RR_OUTPUT / 'authorization.json').read_bytes()
        ).hexdigest(),
        'rr_preflight': directory_hash(RR_OUTPUT / 'preflight'),
        'rr_adapter': directory_hash(RR_OUTPUT / 'adapter'),
        'rr_results': directory_hash(RR_OUTPUT / 'results'),
        'rr_code_revision_lock': sha256(
            RR_CODE_REVISION_LOCK.read_bytes()
        ).hexdigest(),
        'base_manifest': sha256(BASE_CRD_MANIFEST.read_bytes()).hexdigest(),
        'base_preflight': directory_hash(BASE_CRD_OUTPUT / 'preflight'),
        'base_results': directory_hash(BASE_CRD_OUTPUT / 'results'),
        'base_code_revision_lock': sha256(
            BASE_CRD_CODE_REVISION_LOCK.read_bytes()
        ).hexdigest(),
        'frozen_fc_manifest': sha256(
            SOURCE_FC_MANIFEST.read_bytes()
        ).hexdigest(),
        'frozen_fc_preflight': directory_hash(SOURCE_FC_OUTPUT / 'preflight'),
        'frozen_fc_results': directory_hash(SOURCE_FC_OUTPUT / 'results'),
    }

SOURCE_SNAPSHOT_BEFORE = source_snapshot()
rr_manifest_sha256 = SOURCE_SNAPSHOT_BEFORE['rr_manifest']
base_manifest_sha256 = SOURCE_SNAPSHOT_BEFORE['base_manifest']
PROVENANCE_KEY = (
    f'code-{ANALYSIS_CODE_REVISION[:10]}_'
    f'rr-{rr_manifest_sha256[:10]}_base-{base_manifest_sha256[:10]}'
)
ANALYSIS_OUTPUT = (
    ANALYSIS_PIPELINE_ROOT / 'runs' / PROVENANCE_KEY
    / ('paired_audit-' + ANALYSIS_ATTEMPT)
)
sources = (
    RR_OUTPUT.resolve(),
    BASE_CRD_OUTPUT.resolve(),
    SOURCE_FC_OUTPUT.resolve(),
)
for index, source in enumerate(sources):
    for other in sources[index + 1:]:
        if (
            source == other
            or source.is_relative_to(other)
            or other.is_relative_to(source)
        ):
            raise RuntimeError('Frozen audit sources are not mutually independent')
    destination = ANALYSIS_OUTPUT.resolve()
    if (
        destination == source
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise RuntimeError('Analysis output is not independent from its sources')

print(json.dumps({
    'rr_output': str(RR_OUTPUT),
    'rr_run_id': rr_manifest['run_id'],
    'base_crd_output': str(BASE_CRD_OUTPUT),
    'base_crd_run_id': EXPECTED_BASE_CRD_RUN_ID,
    'analysis_output': str(ANALYSIS_OUTPUT),
    'bootstrap_samples': BOOTSTRAP_SAMPLES,
    'bootstrap_seed': BOOTSTRAP_SEED,
}, ensure_ascii=False, indent=2))
"""


RUN_AUDIT = """
result = run_json([
    'uv', 'run', 'plasticity-p0d2hrr', 'audit-p0',
    '--output', str(RR_OUTPUT),
    '--base-crd-output', str(BASE_CRD_OUTPUT),
    '--analysis-output', str(ANALYSIS_OUTPUT),
    '--experiment-code-revision-lock', str(RR_CODE_REVISION_LOCK),
    '--base-crd-code-revision-lock', str(BASE_CRD_CODE_REVISION_LOCK),
    '--historical-preregistration-sha256',
    HISTORICAL_PREREGISTRATION_SHA256,
    '--bootstrap-samples', str(BOOTSTRAP_SAMPLES),
    '--bootstrap-seed', str(BOOTSTRAP_SEED),
])
SUMMARY_PATH = Path(result['output_path'])
if SUMMARY_PATH != ANALYSIS_OUTPUT / 'summary.json':
    raise RuntimeError(f'Unexpected paired-audit summary path: {SUMMARY_PATH}')

SOURCE_SNAPSHOT_AFTER = source_snapshot()
if SOURCE_SNAPSHOT_AFTER != SOURCE_SNAPSHOT_BEFORE:
    raise RuntimeError('Source artifacts changed during read-only audit')

summary = json.loads(SUMMARY_PATH.read_text())
if (
    summary.get('schema_version') != 'p0d2hrr-paired-audit-v1'
    or summary.get('analysis_status') != 'post_hoc_read_only_p0'
    or summary.get('current_artifact_graph_checks_passed') is not True
    or summary.get('historical_attempt_uniqueness_verified') is not False
    or summary.get('source_artifacts_modified') is not False
    or summary.get('source_gate_status_changed') is not False
    or summary.get('next_stage_eligibility_changed') is not False
    or summary.get('training_authorized') is not False
):
    raise RuntimeError('Paired-audit invariants failed')

display(Markdown((ANALYSIS_OUTPUT / 'report.md').read_text()))
display({
    'summary_path': str(SUMMARY_PATH),
    'audit_manifest': str(ANALYSIS_OUTPUT / 'audit_manifest.json'),
    'paired_records': str(ANALYSIS_OUTPUT / 'paired_records.jsonl'),
    'combined_failure_records': str(
        ANALYSIS_OUTPUT / 'combined_failure_records.jsonl'
    ),
    'source_unchanged': SOURCE_SNAPSHOT_AFTER == SOURCE_SNAPSHOT_BEFORE,
    'gate_changed': summary['source_gate_status_changed'],
    'training_authorized': summary['training_authorized'],
    'paper_evidence_ready': summary['paper_evidence_ready'],
    'historical_preregistration_observation': (
        summary['historical_preregistration_observation']
    ),
})
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            _markdown(
                f"""
                # P0 route-remediation provenance and paired audit

                [Open this notebook in Colab]({COLAB_URL})

                This is the first post-remediation experiment: a CPU-only,
                read-only supplementary audit. It verifies preregistration,
                authorization, code, adapter, raw-result, and base-comparator
                provenance; then emits base→adapter transitions, signed-margin
                shifts, lesson-clustered intervals, exact-tie bounds, and matched
                component-cell associations.

                It changes no source artifact, frozen gate, authorization, or
                training eligibility. It does not load a model.
                """
            ),
            _code(CONFIGURATION_SOURCE),
            _markdown(
                """
                ## Independent analysis checkout

                The analysis uses its own checkout and code-revision lock. The
                completed route-remediation checkout and Drive artifacts remain
                untouched.
                """
            ),
            _code(CHECKOUT_SOURCE),
            _markdown(
                """
                ## Resolve and snapshot both frozen sources

                The comparator must be the original NF4 CRD run. The BF16 retry
                is deliberately rejected because it changes the precision and
                weight representation.
                """
            ),
            _code(SOURCE_DISCOVERY),
            _markdown(
                """
                ## Run the immutable P0 audit

                This cell performs no forward pass. It revalidates all stored
                rankings, joins every forced-choice and CRD row, runs the fixed
                lesson-cluster bootstrap, verifies the source snapshot again,
                and displays the generated report.
                """
            ),
            _code(RUN_AUDIT),
            _markdown(
                """
                ## Decision boundary

                These outputs are diagnostic. Do not start another LoRA, a
                1/4/8 scan, narrow scan, or RLVR from this notebook. The next
                protocol depends on which paired mechanism survives this audit.
                """
            ),
        ],
        "metadata": {
            "accelerator": "CPU",
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
        "nbformat_minor": 0,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / NOTEBOOK_NAME
    output.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
