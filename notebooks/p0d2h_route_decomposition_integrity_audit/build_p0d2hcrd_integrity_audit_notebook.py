from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_route_decomposition_integrity_audit_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/"
    "notebooks/p0d2h_route_decomposition_integrity_audit/"
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
REPO_DIR = Path('/content/plasticity-placement-crd-integrity-audit')

SOURCE_PIPELINE_ATTEMPT = 'pipeline-crd1'
SOURCE_CRD_ATTEMPT = 'crd1'
EXPECTED_SOURCE_RUN_ID = 'p0d2hcrd-route-decomposition-8527b64f36'
EXPECTED_DECISION_COUNT = 1536
EXPECTED_CANDIDATE_COUNT = 5376
EXPECTED_TIE_COUNT = 3
SOURCE_PIPELINE_ROOT = (
    Path('/content/drive/MyDrive/plasticity-p0d/hard-probe-route-decomposition/v1/pipelines')
    / SOURCE_PIPELINE_ATTEMPT
)

AUDIT_PIPELINE_ATTEMPT = 'pipeline-a1'
AUDIT_ATTEMPT = 'a1'
AUDIT_PIPELINE_ROOT = (
    Path(
        '/content/drive/MyDrive/plasticity-p0d/hard-probe-route-decomposition-analysis/v1/pipelines'
    )
    / AUDIT_PIPELINE_ATTEMPT
)

for name, value in {{
    'SOURCE_PIPELINE_ATTEMPT': SOURCE_PIPELINE_ATTEMPT,
    'SOURCE_CRD_ATTEMPT': SOURCE_CRD_ATTEMPT,
    'AUDIT_PIPELINE_ATTEMPT': AUDIT_PIPELINE_ATTEMPT,
    'AUDIT_ATTEMPT': AUDIT_ATTEMPT,
}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
print('CPU-only, read-only CRD scoring-integrity audit.')
print('Expected source:', EXPECTED_SOURCE_RUN_ID)
print('Expected anomalies: 3 exact tie rows')
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
    raise RuntimeError(f'Audit checkout has local changes:\\n{dirty}')

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
        f'CRD integrity-audit code lock mismatch: {locked_revision} != '
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
print('Checked out independent audit code:', CODE_REVISION)
print('Formal CRD checkout remains untouched: /content/plasticity-placement')
"""


SOURCE_SOURCE = """
complete_matches = []
for candidate in sorted(
    SOURCE_PIPELINE_ROOT.glob(
        'runs/*/route_decomposition-'
        + SOURCE_CRD_ATTEMPT
        + '/manifest.json'
    )
):
    candidate_manifest = json.loads(candidate.read_text())
    candidate_summary_path = (
        candidate.parent / 'results' / 'aggregate' / 'summary.json'
    )
    if not candidate_summary_path.exists():
        continue
    candidate_summary = json.loads(candidate_summary_path.read_text())
    if (
        candidate_manifest.get('schema_version') == 'p0d2hcrd-manifest-v1'
        and candidate_manifest.get('run_id') == EXPECTED_SOURCE_RUN_ID
        and len(candidate_manifest.get('units', {})) == 24
        and all(
            unit.get('state') == 'verified'
            for unit in candidate_manifest.get('units', {}).values()
        )
        and not candidate_manifest.get('errors')
        and candidate_summary.get('schema_version') == 'p0d2hcrd-summary-v1'
        and candidate_summary.get('run_id') == EXPECTED_SOURCE_RUN_ID
        and candidate_summary.get('decision_row_count')
        == EXPECTED_DECISION_COUNT
        and candidate_summary.get('candidate_sequence_count')
        == EXPECTED_CANDIDATE_COUNT
        and candidate_summary.get('gates', {}).get('run_valid') is True
        and candidate_summary.get('diagnostic_gate', {}).get('status')
        == 'scoring_integrity_failed'
    ):
        complete_matches.append(candidate)
if len(complete_matches) != 1:
    raise RuntimeError(
        'Expected exactly one completed target CRD source, found '
        f'{complete_matches}'
    )

SOURCE_MANIFEST = complete_matches[0]
SOURCE_STAGE_DIR = SOURCE_MANIFEST.parent
SOURCE_SUMMARY = SOURCE_STAGE_DIR / 'results' / 'aggregate' / 'summary.json'
source_manifest = json.loads(SOURCE_MANIFEST.read_text())

def raw_tree_hash():
    digest = sha256()
    files = sorted((SOURCE_STAGE_DIR / 'results' / 'raw').rglob('*.jsonl'))
    if len(files) != 24:
        raise RuntimeError(f'Expected 24 CRD raw files, found {len(files)}')
    row_count = 0
    for path in files:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        row_count += len(lines)
        digest.update(str(path.relative_to(SOURCE_STAGE_DIR)).encode())
        digest.update(path.read_bytes())
    if row_count != EXPECTED_DECISION_COUNT:
        raise RuntimeError(f'Expected 1536 CRD rows, found {row_count}')
    return digest.hexdigest()

SOURCE_MANIFEST_HASH = sha256(SOURCE_MANIFEST.read_bytes()).hexdigest()
SOURCE_SUMMARY_HASH = sha256(SOURCE_SUMMARY.read_bytes()).hexdigest()
SOURCE_RAW_HASH = raw_tree_hash()
PROVENANCE_KEY = (
    f'code-{CODE_REVISION[:10]}_source-{SOURCE_MANIFEST_HASH[:10]}'
)
AUDIT_STAGE_DIR = (
    AUDIT_PIPELINE_ROOT / 'runs' / PROVENANCE_KEY
    / ('scoring_integrity_audit-' + AUDIT_ATTEMPT)
)
print(json.dumps({
    'source_stage': str(SOURCE_STAGE_DIR),
    'source_run_id': source_manifest['run_id'],
    'source_manifest_sha256': SOURCE_MANIFEST_HASH,
    'source_summary_sha256': SOURCE_SUMMARY_HASH,
    'source_raw_results_sha256': SOURCE_RAW_HASH,
    'audit_output': str(AUDIT_STAGE_DIR),
}, ensure_ascii=False, indent=2))
"""


RUN_SOURCE = """
command = [
    'uv', 'run', 'plasticity-p0d2hcrd', 'audit-integrity',
    '--output', str(SOURCE_STAGE_DIR),
    '--audit-output', str(AUDIT_STAGE_DIR),
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
        f'CRD scoring-integrity audit failed ({completed.returncode})'
    )
print(completed.stdout)
if sha256(SOURCE_MANIFEST.read_bytes()).hexdigest() != SOURCE_MANIFEST_HASH:
    raise RuntimeError('Source manifest changed during read-only audit')
if sha256(SOURCE_SUMMARY.read_bytes()).hexdigest() != SOURCE_SUMMARY_HASH:
    raise RuntimeError('Source summary changed during read-only audit')
if raw_tree_hash() != SOURCE_RAW_HASH:
    raise RuntimeError('Source raw rows changed during read-only audit')
print('READ-ONLY SOURCE CHECK PASSED')
"""


RESULTS_SOURCE = """
summary_path = AUDIT_STAGE_DIR / 'scoring_integrity_audit.json'
markdown_path = AUDIT_STAGE_DIR / 'scoring_integrity_audit.md'
records_path = AUDIT_STAGE_DIR / 'scoring_integrity_records.jsonl'
audit = json.loads(summary_path.read_text())
records = [
    json.loads(line)
    for line in records_path.read_text().splitlines()
    if line.strip()
]
if (
    audit.get('analysis_status') != 'post_hoc_supplementary'
    or audit.get('source_gate_status_changed') is not False
    or audit.get('next_stage_eligibility_changed') is not False
    or audit.get('raw_rows_modified') is not False
    or audit.get('manifest_modified') is not False
    or audit.get('row_count') != EXPECTED_DECISION_COUNT
    or audit.get('tie_record_count') != EXPECTED_TIE_COUNT
    or audit.get('anomaly_record_count') != len(records)
    or audit.get('source', {}).get('source_run_id')
    != EXPECTED_SOURCE_RUN_ID
    or audit.get('source', {}).get('source_diagnostic_status')
    != 'scoring_integrity_failed'
):
    raise RuntimeError('CRD integrity-audit status/provenance mismatch')

display(Markdown(markdown_path.read_text()))
print(json.dumps({
    'run_id': audit['run_id'],
    'classification': audit['classification'],
    'structural_checks_passed': audit['structural_checks_passed'],
    'status_counts': audit['status_counts'],
    'endpoint_sensitivity': audit['endpoint_sensitivity'],
    'tie_factor_counts': audit['tie_factor_counts'],
    'summary_path': str(summary_path),
    'anomaly_records_path': str(records_path),
}, ensure_ascii=False, indent=2))
print('FULL ANOMALY EVIDENCE')
print(json.dumps(records, ensure_ascii=False, indent=2))
"""


def build_notebook() -> dict[str, Any]:
    cells = [
        _markdown(
            f"""
            # P0-D2H-CRD Scoring-Integrity Audit

            [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_URL})

            CPU-only, read-only supplementary audit of the completed
            P0-D2H-CRD `pipeline-crd1` result. It independently recomputes
            candidate, token, score, rank, and status invariants for the three
            exact tie rows. It changes no source artifact, frozen gate, or
            next-stage eligibility and uses a checkout separate from the formal
            CRD notebook.
            """
        ),
        _markdown("## 1. Configuration and independent analysis namespace"),
        _code(CONFIGURATION_SOURCE),
        _markdown("## 2. Create and lock an independent audit checkout"),
        _code(CHECKOUT_SOURCE),
        _markdown("## 3. Verify and freeze the completed CRD source"),
        _code(SOURCE_SOURCE),
        _markdown("## 4. Run the read-only scoring-integrity audit"),
        _code(RUN_SOURCE),
        _markdown("## 5. Review classification and full anomaly evidence"),
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
