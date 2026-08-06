from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "p0d2h_rab_error_localization_colab.ipynb"
CODE_REVISION = "bb24ed8517b3562cc030a4a48e71577237c3ff9d"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/p0d2h_rab_error_localization/{NOTEBOOK_NAME}"
)


def markdown(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


CONFIG = f"""
from google.colab import drive
drive.mount('/content/drive')

import json
import re
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

from IPython.display import Markdown, display

REPO_URL = 'https://github.com/donleaveher/plasticity-placement.git'
BRANCH = '{BRANCH}'
CODE_REVISION = {CODE_REVISION!r}
REPO_DIR = Path('/content/plasticity-placement-rab-error-localization')

RABD_OUTPUT = Path(
    '/content/drive/MyDrive/plasticity-p0d/rab-error-topology/v1/'
    'pipelines/pipeline-rabd1/runs/'
    'code-d850d82e81_rab-9368c8c29c/rab-error-topology-rabd1'
)
PIPELINE_ATTEMPT = 'pipeline-rabdl1'
LOCALIZATION_ATTEMPT = 'rabdl1'
PIPELINE_ROOT = Path(
    '/content/drive/MyDrive/plasticity-p0d/rab-error-localization/v1/pipelines'
) / PIPELINE_ATTEMPT

for name, value in {{
    'PIPELINE_ATTEMPT': PIPELINE_ATTEMPT,
    'LOCALIZATION_ATTEMPT': LOCALIZATION_ATTEMPT,
}}.items():
    if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
        raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
print('Frozen completed RABD source:', RABD_OUTPUT)
print('Localization pipeline:', PIPELINE_ROOT)
"""


CONTROLS = """
# EDIT ONLY THIS CELL BETWEEN PASSES.
RUN_PLAN = True
RUN_LOCALIZATION = False

stages = {'plan': RUN_PLAN, 'localization': RUN_LOCALIZATION}
if sum(bool(value) for value in stages.values()) > 1:
    raise ValueError(f'Enable at most one stage per pass: {stages}')
print('Selected controls:', stages)
"""


SETUP_AND_RUN = """
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'], check=True)
if REPO_DIR.exists() and not (REPO_DIR / '.git').exists():
    raise RuntimeError(f'{REPO_DIR} exists but is not a Git repository')
if not REPO_DIR.exists():
    subprocess.run(['git', 'clone', '--branch', BRANCH, REPO_URL, str(REPO_DIR)], check=True)
dirty = subprocess.run(
    ['git', '-C', str(REPO_DIR), 'status', '--porcelain'],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if dirty:
    raise RuntimeError(f'Checkout has local changes:\\n{dirty}')

PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
CODE_REVISION_LOCK = PIPELINE_ROOT / 'code_revision.txt'
locked_revision = CODE_REVISION_LOCK.read_text().strip() if CODE_REVISION_LOCK.exists() else None
subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', 'origin', BRANCH], check=True)
revision_ref = CODE_REVISION or locked_revision or f'origin/{BRANCH}'
RESOLVED_CODE_REVISION = subprocess.run(
    ['git', '-C', str(REPO_DIR), 'rev-parse', f'{revision_ref}^{{commit}}'],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if locked_revision and locked_revision != RESOLVED_CODE_REVISION:
    raise RuntimeError('RAB localization code lock mismatch; use a new pipeline attempt')
if not locked_revision:
    temporary = CODE_REVISION_LOCK.with_suffix('.txt.tmp')
    temporary.write_text(RESOLVED_CODE_REVISION + '\\n')
    temporary.replace(CODE_REVISION_LOCK)
subprocess.run(
    ['git', '-C', str(REPO_DIR), 'checkout', '--detach', RESOLVED_CODE_REVISION], check=True
)
subprocess.run(
    ['uv', 'sync', '--extra', 'train', '--extra', 'colab'], cwd=REPO_DIR, check=True
)

required_source = (
    RABD_OUTPUT / 'manifest.json',
    RABD_OUTPUT / 'preregistration.json',
    RABD_OUTPUT / 'summary.json',
    RABD_OUTPUT / 'report.md',
    RABD_OUTPUT / 'cell_records.jsonl',
    RABD_OUTPUT / 'unit_records.jsonl',
    RABD_OUTPUT / 'factor_slices.json',
    RABD_OUTPUT / 'audit_manifest.json',
    RABD_OUTPUT / 'preflight' / 'analysis_plan.json',
)
missing = [str(path) for path in required_source if not path.is_file()]
if missing:
    raise FileNotFoundError(f'Missing frozen RABD artifacts: {missing}')
rabd_hash = sha256((RABD_OUTPUT / 'summary.json').read_bytes()).hexdigest()
identity = f'code-{RESOLVED_CODE_REVISION[:10]}_rabd-{rabd_hash[:10]}'
LOCALIZATION_OUTPUT = PIPELINE_ROOT / 'runs' / identity / (
    'rab-error-localization-' + LOCALIZATION_ATTEMPT
)

def run(command):
    completed = subprocess.run(command, cwd=REPO_DIR, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f'Command failed with exit code {completed.returncode}: {command}')
    return completed

manifest_path = LOCALIZATION_OUTPUT / 'manifest.json'
if RUN_PLAN:
    if manifest_path.is_file():
        print('Frozen localization plan already exists; not regenerating it')
    else:
        run([
            'uv', 'run', 'plasticity-p0d2hrabdl', 'plan',
            '--output', str(LOCALIZATION_OUTPUT), '--rabd-output', str(RABD_OUTPUT),
        ])

if RUN_LOCALIZATION:
    if not manifest_path.is_file():
        raise RuntimeError('Run the planning pass before the localization pass')
    manifest = json.loads(manifest_path.read_text())
    if manifest['state'] == 'planned':
        run([
            'uv', 'run', 'plasticity-p0d2hrabdl', 'run',
            '--output', str(LOCALIZATION_OUTPUT),
        ])
    elif manifest['state'] == 'complete':
        print('RAB error localization already complete; not rerunning')
    else:
        raise RuntimeError(f'Localization requires planned state, found {manifest["state"]}')

print('Code revision:', RESOLVED_CODE_REVISION)
print('Localization output:', LOCALIZATION_OUTPUT)
"""


RESULTS = """
manifest_path = LOCALIZATION_OUTPUT / 'manifest.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
if manifest and manifest['state'] == 'complete':
    run(['uv', 'run', 'plasticity-p0d2hrabdl', 'verify', '--output', str(LOCALIZATION_OUTPUT)])
    summary_path = LOCALIZATION_OUTPUT / 'summary.json'
    report_path = LOCALIZATION_OUTPUT / 'report.md'
    audit_manifest_path = LOCALIZATION_OUTPUT / 'audit_manifest.json'
    required = (
        summary_path,
        report_path,
        audit_manifest_path,
        LOCALIZATION_OUTPUT / 'cell_transition_localization.json',
        LOCALIZATION_OUTPUT / 'taxonomy_migrations.json',
        LOCALIZATION_OUTPUT / 'margin_transition_profiles.json',
        LOCALIZATION_OUTPUT / 'unit_hotspots.jsonl',
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError('Complete localization manifest is missing published artifacts')
    summary_digest = sha256(summary_path.read_bytes()).hexdigest()
    audit_digest = sha256(audit_manifest_path.read_bytes()).hexdigest()
    audit_manifest = json.loads(audit_manifest_path.read_text())
    observed = {path.name: sha256(path.read_bytes()).hexdigest() for path in required[3:]}
    expected = {
        'cell_transition_localization.json': audit_manifest['artifacts'][
            'cell_transition_localization'
        ],
        'taxonomy_migrations.json': audit_manifest['artifacts']['taxonomy_migrations'],
        'margin_transition_profiles.json': audit_manifest['artifacts'][
            'margin_transition_profiles'
        ],
        'unit_hotspots.jsonl': audit_manifest['artifacts']['unit_hotspots'],
    }
    if observed != expected:
        raise RuntimeError(f'Localization artifact hashes differ: {observed} != {expected}')
    if {
        manifest['result']['summary_sha256'],
        audit_manifest['artifacts']['summary'],
    } != {summary_digest}:
        raise RuntimeError('Localization summary hash does not match complete manifests')
    if manifest['result']['audit_manifest_sha256'] != audit_digest:
        raise RuntimeError('Localization audit-manifest hash does not match complete manifest')
    summary = json.loads(summary_path.read_text())
    display(Markdown(report_path.read_text()))
    print(json.dumps({
        'summary_path': str(summary_path),
        'analysis_status': summary['analysis']['analysis_status'],
        'transition_totals': summary['analysis']['transition_totals'],
        'top_adverse_levels': summary['analysis']['top_adverse_levels'],
        'taxonomy_focus': summary['analysis']['taxonomy_focus'],
        'margin_transition_profiles': summary['analysis']['margin_transition_profiles'],
        'historical_rab_decision_changed': summary['historical_rab_decision_changed'],
        'historical_rabd_status_changed': summary['historical_rabd_status_changed'],
        'inference_authorized': summary['inference_authorized'],
        'training_authorized': summary['training_authorized'],
        'mappings_per_adapter_authorized': summary['mappings_per_adapter_authorized'],
    }, indent=2))
elif manifest:
    print('No displayable result; localization manifest state:', manifest['state'])
else:
    print('No localization plan yet. Run planning and localization in separate passes.')
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            markdown(
                f"""
                # P0-D2H RAB error-localization reader

                [Open this notebook in Google Colab]({COLAB_URL})

                This CPU-only reader localizes the exact completed RAB topology result across
                factors, taxonomy migrations, correctness-transition margins, and all 48 unit
                hotspots. It does not load a model, run inference, train, reclassify a historical
                result, or authorize 1/4/8.
                """
            ),
            code(CONFIG),
            code(CONTROLS),
            code(SETUP_AND_RUN),
            markdown(
                """
                First keep the safe defaults and run through the combined setup/planning cell.
                Inspect `preregistration.json` and `preflight/analysis_plan.json`. Then edit only
                the controls cell: set `RUN_PLAN = False` and `RUN_LOCALIZATION = True`, rerun the
                controls and combined cell, and finally run the result cell. No approver or
                authorization artifact is required for this deterministic post-hoc CPU reader.
                """
            ),
            code(RESULTS),
        ],
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / NOTEBOOK_NAME
    path.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n")
    print(path)
    print(COLAB_URL)


if __name__ == "__main__":
    main()
