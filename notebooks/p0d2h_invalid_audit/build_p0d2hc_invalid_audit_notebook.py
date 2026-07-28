from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_invalid_output_audit_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/p0d2h_invalid_audit/"
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
REQUESTED_CODE_REVISION = None
REPO_DIR = Path('/content/plasticity-placement')

SOURCE_PIPELINE_ATTEMPT = 'pipeline-c1'
SOURCE_CALIBRATION_ATTEMPT = 'c1'
SOURCE_PIPELINE_ROOT = (
    Path('/content/drive/MyDrive/plasticity-p0d/hard-probe-calibration/v1/pipelines')
    / SOURCE_PIPELINE_ATTEMPT
)

AUDIT_PIPELINE_ATTEMPT = 'pipeline-a1'
AUDIT_ATTEMPT = 'a1'
AUDIT_PIPELINE_ROOT = (
    Path(
        '/content/drive/MyDrive/plasticity-p0d/hard-probe-calibration-analysis/v1/pipelines'
    )
    / AUDIT_PIPELINE_ATTEMPT
)
BOOTSTRAP_SAMPLES = 10000

for name, value in {{
    'SOURCE_PIPELINE_ATTEMPT': SOURCE_PIPELINE_ATTEMPT,
    'SOURCE_CALIBRATION_ATTEMPT': SOURCE_CALIBRATION_ATTEMPT,
    'AUDIT_PIPELINE_ATTEMPT': AUDIT_PIPELINE_ATTEMPT,
    'AUDIT_ATTEMPT': AUDIT_ATTEMPT,
}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
if BOOTSTRAP_SAMPLES <= 0:
    raise ValueError('BOOTSTRAP_SAMPLES must be positive')
print('CPU-only post-hoc audit; no inference or adapter loading.')
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

AUDIT_PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
CODE_REVISION_LOCK = AUDIT_PIPELINE_ROOT / 'code_revision.txt'
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
        f'Invalid-audit code lock mismatch: {locked_revision} != '
        f'{CODE_REVISION}. Use a new AUDIT_PIPELINE_ATTEMPT.'
    )
if not locked_revision:
    temporary = CODE_REVISION_LOCK.with_suffix('.txt.tmp')
    temporary.write_text(CODE_REVISION + '\\n')
    temporary.replace(CODE_REVISION_LOCK)
subprocess.run(
    ['git', '-C', str(REPO_DIR), 'checkout', '--detach', CODE_REVISION],
    check=True,
)
subprocess.run(['uv', 'sync'], cwd=REPO_DIR, check=True)
print('Checked out audit code:', CODE_REVISION)
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
        'Expected exactly one P0-D2H-CAL source manifest under '
        f'{SOURCE_PIPELINE_ROOT}, found {source_matches}'
    )
SOURCE_MANIFEST = source_matches[0]
SOURCE_STAGE_DIR = SOURCE_MANIFEST.parent
source_manifest = json.loads(SOURCE_MANIFEST.read_text())
source_summary_path = (
    SOURCE_STAGE_DIR / 'results' / 'aggregate' / 'summary.json'
)
source_summary = json.loads(source_summary_path.read_text())
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
    or source_summary.get('run_id') != source_manifest.get('run_id')
    or source_summary.get('probe_row_count') != 2304
    or source_summary.get('gates', {}).get('run_valid') is not True
):
    raise RuntimeError('P0-D2H-CAL source is not complete and run-valid')

def raw_tree_hash():
    digest = sha256()
    files = sorted((SOURCE_STAGE_DIR / 'results' / 'raw').rglob('*.jsonl'))
    if len(files) != 48:
        raise RuntimeError(f'Expected 48 raw files, found {len(files)}')
    for path in files:
        digest.update(str(path.relative_to(SOURCE_STAGE_DIR)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()

SOURCE_MANIFEST_HASH = sha256(SOURCE_MANIFEST.read_bytes()).hexdigest()
SOURCE_RAW_HASH = raw_tree_hash()
PROVENANCE_KEY = (
    f'code-{CODE_REVISION[:10]}_source-{SOURCE_MANIFEST_HASH[:10]}'
)
AUDIT_STAGE_DIR = (
    AUDIT_PIPELINE_ROOT / 'runs' / PROVENANCE_KEY
    / ('invalid_audit-' + AUDIT_ATTEMPT)
)
print(json.dumps({
    'source_manifest': str(SOURCE_MANIFEST),
    'source_run_id': source_manifest['run_id'],
    'source_manifest_sha256': SOURCE_MANIFEST_HASH,
    'source_raw_results_sha256': SOURCE_RAW_HASH,
    'audit_output': str(AUDIT_STAGE_DIR),
}, ensure_ascii=False, indent=2))
"""


RUN_SOURCE = """
command = [
    'uv', 'run', 'plasticity-p0d2hc', 'audit-invalid',
    '--output', str(SOURCE_STAGE_DIR),
    '--audit-output', str(AUDIT_STAGE_DIR),
    '--bootstrap-samples', str(BOOTSTRAP_SAMPLES),
]
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
        f'Invalid-output audit failed ({completed.returncode})'
    )
print(completed.stdout)
if sha256(SOURCE_MANIFEST.read_bytes()).hexdigest() != SOURCE_MANIFEST_HASH:
    raise RuntimeError('Source manifest changed during read-only audit')
if raw_tree_hash() != SOURCE_RAW_HASH:
    raise RuntimeError('Source raw rows changed during read-only audit')
print('READ-ONLY SOURCE CHECK PASSED')
"""


RESULTS_SOURCE = """
summary_path = AUDIT_STAGE_DIR / 'invalid_output_audit.json'
markdown_path = AUDIT_STAGE_DIR / 'invalid_output_audit.md'
records_path = AUDIT_STAGE_DIR / 'invalid_output_records.jsonl'
audit = json.loads(summary_path.read_text())
if (
    audit.get('analysis_status') != 'post_hoc_supplementary'
    or audit.get('strict_gate_status_changed') is not False
    or audit.get('next_stage_eligibility_changed') is not False
    or audit.get('raw_rows_modified') is not False
):
    raise RuntimeError('Invalid audit status/provenance mismatch')
display(Markdown(markdown_path.read_text()))
print(json.dumps({
    'run_id': audit['run_id'],
    'row_count': audit['row_count'],
    'invalid_record_count': audit['invalid_record_count'],
    'overall_by_model_arm': audit['overall_by_model_arm'],
    'summary_path': str(summary_path),
    'invalid_records_path': str(records_path),
}, ensure_ascii=False, indent=2))
"""


def build_notebook() -> dict[str, Any]:
    cells = [
        _markdown(
            f"""
            # P0-D2H-CAL Invalid-Output Audit

            [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

            CPU-only, read-only supplementary analysis of the completed
            P0-D2H-CAL `pipeline-c1` raw rows. It classifies strict invalid
            outputs and reports conservative semantic recovery. It performs no
            inference, changes no frozen gate, and writes to an independent
            analysis namespace.
            """
        ),
        _markdown("## 1. Configuration and independent analysis namespace"),
        _code(CONFIGURATION_SOURCE),
        _markdown("## 2. Checkout and lock analysis code"),
        _code(CHECKOUT_SOURCE),
        _markdown("## 3. Verify and freeze the P0-D2H-CAL source"),
        _code(SOURCE_SOURCE),
        _markdown("## 4. Run the read-only invalid-output audit"),
        _code(RUN_SOURCE),
        _markdown("## 5. Review semantic recovery and taxonomy"),
        _code(RESULTS_SOURCE),
    ]
    return {
        "cells": cells,
        "metadata": {
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
