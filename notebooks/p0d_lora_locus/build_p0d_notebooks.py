from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent, indent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
STAGES = ("band_scan", "narrow_scan")


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


def _configuration_source(stage: str) -> str:
    run_flag = f"RUN_{stage.upper()}"
    band_attempt = "BAND_SCAN_ATTEMPT = 'a1'" if stage == "narrow_scan" else ""
    band_validation = (
        "'BAND_SCAN_ATTEMPT': BAND_SCAN_ATTEMPT,"
        if stage == "narrow_scan"
        else ""
    )
    return f"""
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

    P0D_PIPELINE_ATTEMPT = 'pipeline-a1'
    {stage.upper()}_ATTEMPT = 'a1'
    {band_attempt}
    P0D_PIPELINE_ROOT = (
        Path('/content/drive/MyDrive/plasticity-p0d/lora-locus/v1/pipelines')
        / P0D_PIPELINE_ATTEMPT
    )
    NARROW_CONDITIONS_CONFIG = (
        P0D_PIPELINE_ROOT / 'frozen-narrow-conditions.json'
    )

    {run_flag} = False

    for name, value in {{
        'P0C_PIPELINE_ATTEMPT': P0C_PIPELINE_ATTEMPT,
        'P0C_CONFIRMATORY_ATTEMPT': P0C_CONFIRMATORY_ATTEMPT,
        'P0D_PIPELINE_ATTEMPT': P0D_PIPELINE_ATTEMPT,
        '{stage.upper()}_ATTEMPT': {stage.upper()}_ATTEMPT,
        {band_validation}
    }}.items():
        if not re.fullmatch(r'[A-Za-z0-9._-]+', value):
            raise ValueError(f'{{name}} contains unsafe path characters: {{value!r}}')
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

P0D_PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
CODE_REVISION_LOCK = P0D_PIPELINE_ROOT / 'code-revision.txt'
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
        f'P0-D code lock mismatch: {locked_revision} != {CODE_REVISION}. '
        'Use a new P0D_PIPELINE_ATTEMPT.'
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
print('Checked out P0-D code:', CODE_REVISION)
"""


def _provenance_source(stage: str) -> str:
    conditions_source = (
        """
        CONDITIONS_CONFIG = None
        CONDITIONS_HASH = 'frozen-band-scan-v1'
        """
        if stage == "band_scan"
        else """
        if not NARROW_CONDITIONS_CONFIG.exists():
            raise FileNotFoundError(
                'Freeze the reviewed narrow-scan JSON before running: '
                f'{NARROW_CONDITIONS_CONFIG}'
            )
        CONDITIONS_CONFIG = NARROW_CONDITIONS_CONFIG
        CONDITIONS_HASH = sha256(CONDITIONS_CONFIG.read_bytes()).hexdigest()
        """
    )
    conditions_block = indent(dedent(conditions_source).strip(), "    ")
    return f"""
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
                f'Command failed ({{completed.returncode}}): {{shlex.join(command)}}'
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(f'Command returned no JSON: {{shlex.join(command)}}')
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
            f'{{P0C_PIPELINE_ROOT}}, found {{source_matches}}'
        )
    SOURCE_P0C_MANIFEST = source_matches[0]
    source_manifest = json.loads(SOURCE_P0C_MANIFEST.read_text())
    source_lessons = source_manifest.get('selected_lessons', [])
    source_units = source_manifest.get('units', {{}})
    SOURCE_MODEL_REVISION = source_manifest.get('config', {{}}).get('model_revision')
    if (
        source_manifest.get('config', {{}}).get('tier') != 'confirmatory'
        or len(source_lessons) != 24
        or len(source_units) != 72
        or set(source_manifest.get('base_arms', {{}}).values()) != {{'verified'}}
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

    environment_context = run_json(['uv', 'run', 'plasticity-p0d', 'environment'])
    environment = environment_context['environment']
    ENVIRONMENT_FINGERPRINT = environment_context['fingerprint']
    CODE_HASH = environment['code_sha256']

{conditions_block}

    PROVENANCE_KEY = (
        f'code-{{CODE_HASH[:10]}}_source-{{SOURCE_MANIFEST_HASH[:10]}}'
    )
    PROVENANCE_ROOT = P0D_PIPELINE_ROOT / 'runs' / PROVENANCE_KEY
    STAGE_DIR = PROVENANCE_ROOT / ('{stage}-' + {stage.upper()}_ATTEMPT)
    LOG_DIR = PROVENANCE_ROOT / 'logs'
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    frozen_context = {{
        'schema_version': 1,
        'pipeline_version': 'p0d-lora-locus-v1',
        'code_sha256': CODE_HASH,
        'source_manifest_sha256': SOURCE_MANIFEST_HASH,
        'source_model_revision': SOURCE_MODEL_REVISION,
        'selected_lesson_sha256': SELECTED_LESSON_HASH,
        'stage': '{stage}',
        'conditions_sha256': CONDITIONS_HASH,
    }}
    context_path = STAGE_DIR / 'notebook_context.json'
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    if context_path.exists():
        if json.loads(context_path.read_text()) != frozen_context:
            raise RuntimeError(
                f'Frozen P0-D stage context mismatch: {{context_path}}'
            )
    else:
        temporary = context_path.with_suffix('.json.tmp')
        temporary.write_text(
            json.dumps(frozen_context, indent=2, sort_keys=True) + '\\n'
        )
        temporary.replace(context_path)

    sessions_path = STAGE_DIR / 'source_sessions.json'
    sessions = json.loads(sessions_path.read_text()) if sessions_path.exists() else []
    sessions.append({{
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'git_revision': CODE_REVISION,
        'branch': BRANCH,
        'stage': '{stage}',
        'environment_fingerprint': ENVIRONMENT_FINGERPRINT,
        'environment': environment,
        'source_p0c_manifest': str(SOURCE_P0C_MANIFEST),
        'conditions_config': str(CONDITIONS_CONFIG) if CONDITIONS_CONFIG else None,
    }})
    temporary = sessions_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(sessions, indent=2, sort_keys=True) + '\\n')
    temporary.replace(sessions_path)

    def run_checked(label, command):
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        safe_label = re.sub(r'[^A-Za-z0-9._-]+', '-', label)
        log_path = LOG_DIR / f'{{timestamp}}-{{safe_label}}.log'
        child_environment = {{**os.environ, 'PYTHONUNBUFFERED': '1'}}
        print(f'\\n[{{label}}] $ {{shlex.join(command)}}')
        print(f'[{{label}}] log: {{log_path}}')
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
        print(f'[{{label}}] exit={{return_code}} elapsed={{elapsed / 60:.1f}} min')
        if return_code != 0:
            raise RuntimeError(
                f'{{label}} failed with exit code {{return_code}}; log={{log_path}}'
            )

    def p0d_command(action):
        if sha256(SOURCE_P0C_MANIFEST.read_bytes()).hexdigest() != SOURCE_MANIFEST_HASH:
            raise RuntimeError('P0-C source manifest changed after provenance freeze')
        if (
            CONDITIONS_CONFIG is not None
            and sha256(CONDITIONS_CONFIG.read_bytes()).hexdigest() != CONDITIONS_HASH
        ):
            raise RuntimeError('P0-D conditions config changed after provenance freeze')
        command = [
            'uv', 'run', 'plasticity-p0d', action,
            '--output', str(STAGE_DIR),
        ]
        if action in {{'plan', 'run'}}:
            command.extend([
                '--source-manifest', str(SOURCE_P0C_MANIFEST),
                '--stage', '{stage}',
            ])
            if CONDITIONS_CONFIG is not None:
                command.extend(['--conditions-config', str(CONDITIONS_CONFIG)])
        return command

    def manifest_status():
        path = STAGE_DIR / 'manifest.json'
        if not path.exists():
            return {{'exists': False, 'stage_dir': str(STAGE_DIR)}}
        manifest = json.loads(path.read_text())
        return {{
            'exists': True,
            'run_id': manifest.get('run_id'),
            'stage_dir': str(STAGE_DIR),
            'lesson_count': len(manifest.get('selected_lessons', [])),
            'conditions': list(manifest.get('conditions', {{}})),
            'base_states': dict(Counter(manifest.get('base_arms', {{}}).values())),
            'unit_states': dict(Counter(
                unit.get('state') for unit in manifest.get('units', {{}}).values()
            )),
            'errors': manifest.get('errors', [])[-10:],
        }}

    def verify_manifest(expected_units):
        path = STAGE_DIR / 'manifest.json'
        if not path.exists():
            raise FileNotFoundError(f'Missing P0-D manifest: {{path}}')
        manifest = json.loads(path.read_text())
        units = manifest.get('units', {{}})
        if len(manifest.get('selected_lessons', [])) != 24:
            raise RuntimeError('P0-D manifest does not contain 24 lessons')
        if len(units) != expected_units:
            raise RuntimeError(
                f'Expected {{expected_units}} adapter units, got {{len(units)}}'
            )
        if set(manifest.get('base_arms', {{}}).values()) != {{'verified'}}:
            raise RuntimeError('P0-D base arms are incomplete')
        bad = {{
            key: unit.get('state')
            for key, unit in units.items()
            if unit.get('state') != 'verified'
            or float(unit.get('rollback_exact_match_rate', 0.0)) != 1.0
        }}
        if bad or manifest.get('errors'):
            raise RuntimeError(
                f'P0-D units invalid={{bad}}, errors={{manifest.get("errors", [])[-5:]}}'
            )
        return manifest

    print(json.dumps({{
        'code_revision': CODE_REVISION,
        'code_sha256': CODE_HASH,
        'source_manifest': str(SOURCE_P0C_MANIFEST),
        'source_manifest_sha256': SOURCE_MANIFEST_HASH,
        'source_model_revision': SOURCE_MODEL_REVISION,
        'selected_lesson_sha256': SELECTED_LESSON_HASH,
        'conditions_sha256': CONDITIONS_HASH,
        'environment_fingerprint': ENVIRONMENT_FINGERPRINT,
        'stage_dir': str(STAGE_DIR),
    }}, indent=2))
    """


BAND_SOURCE = """
plan = run_json(p0d_command('plan'))
if plan['expected_unit_count'] != 288:
    raise RuntimeError(f"Band scan expected 288 units, got {plan['expected_unit_count']}")
display(plan['config']['conditions'])

if RUN_BAND_SCAN:
    run_checked('p0d-band-scan-run', p0d_command('run'))
    verify_manifest(expected_units=288)
    run_checked(
        'p0d-band-scan-aggregate',
        [
            'uv', 'run', 'plasticity-p0d', 'aggregate',
            '--output', str(STAGE_DIR),
            '--bootstrap-samples', '10000',
        ],
    )

print(json.dumps(manifest_status(), ensure_ascii=False, indent=2))
summary_path = STAGE_DIR / 'results' / 'aggregate' / 'summary.json'
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    if summary['adapter_unit_count'] != 288 or summary['probe_row_count'] != 16224:
        raise RuntimeError('Unexpected P0-D band-scan result dimensions')
    display(summary['condition_summary'])
    display(summary['contrasts'])
    display(summary['gates'])
    print(
        'Next-stage candidate file:',
        STAGE_DIR / 'results' / 'aggregate' / 'next_stage_candidates.json',
    )
"""


NARROW_SOURCE = """
band_dir = PROVENANCE_ROOT / ('band_scan-' + BAND_SCAN_ATTEMPT)
band_summary_path = band_dir / 'results' / 'aggregate' / 'summary.json'
band_candidates_path = (
    band_dir / 'results' / 'aggregate' / 'next_stage_candidates.json'
)
if not band_summary_path.exists() or not band_candidates_path.exists():
    raise FileNotFoundError(
        'Narrow scan requires a reviewed completed band scan under '
        f'{band_dir}'
    )
band_summary = json.loads(band_summary_path.read_text())
candidate_status = json.loads(band_candidates_path.read_text())
if candidate_status.get('status') != 'manual_freeze_required':
    raise RuntimeError('Band scan did not pass the narrow-scan entry gate')

narrow_payload = json.loads(NARROW_CONDITIONS_CONFIG.read_text())
if narrow_payload.get('source_band_run_id') != band_summary.get('run_id'):
    raise RuntimeError(
        'Frozen narrow conditions do not reference the reviewed band-scan run'
    )
plan = run_json(p0d_command('plan'))
expected_units = int(plan['expected_unit_count'])
if expected_units != 72 * len(plan['config']['conditions']):
    raise RuntimeError('Narrow-scan unit count does not match conditions × 24 × 3')
display(plan['config']['conditions'])

if RUN_NARROW_SCAN:
    run_checked('p0d-narrow-scan-run', p0d_command('run'))
    verify_manifest(expected_units=expected_units)
    run_checked(
        'p0d-narrow-scan-aggregate',
        [
            'uv', 'run', 'plasticity-p0d', 'aggregate',
            '--output', str(STAGE_DIR),
            '--bootstrap-samples', '10000',
        ],
    )

print(json.dumps(manifest_status(), ensure_ascii=False, indent=2))
summary_path = STAGE_DIR / 'results' / 'aggregate' / 'summary.json'
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    display(summary['condition_summary'])
    display(summary['contrasts'])
    display(summary['gates'])
"""


def build_notebook(stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(stage)
    filename = f"p0d_{stage}_colab.ipynb"
    badge = (
        "https://colab.research.google.com/github/donleaveher/"
        "plasticity-placement/blob/"
        f"{BRANCH.replace('/', '%2F')}/notebooks/p0d_lora_locus/{filename}"
    )
    title = "Band Scan" if stage == "band_scan" else "Narrow Scan"
    dependency = (
        "Requires the complete verified P0-C v4 Confirmatory manifest."
        if stage == "band_scan"
        else (
            "Requires a completed band scan that passed its entry gate and a manually "
            "frozen explicit-layer condition JSON."
        )
    )
    cells = [
        _markdown(
            f"""
            <a href="{badge}" target="_parent"><img
            src="https://colab.research.google.com/assets/colab-badge.svg"
            alt="Open In Colab"/></a>

            # P0-D LoRA Layer Locus — {title}

            Standalone notebook for **{stage} only**. {dependency}
            P0-C inputs are read-only; all P0-D outputs use the independent
            `plasticity-p0d/lora-locus/v1` Drive namespace.
            """
        ),
        _markdown(
            """
            ## 1. Configuration

            Select a GPU runtime. Point the P0-C attempt labels at the verified
            Confirmatory run. Increment only this P0-D stage attempt after an immutable
            failure or environment change.
            """
        ),
        _code(_configuration_source(stage)),
        _markdown("## 2. Checkout and install the frozen P0-D code"),
        _code(CHECKOUT_SOURCE),
        _markdown("## 3. Resolve source provenance, stage directory, and recovery"),
        _code(_provenance_source(stage)),
        _markdown(f"## 4. Run {title}"),
        _code(BAND_SOURCE if stage == "band_scan" else NARROW_SOURCE),
        _markdown(
            """
            ## Recovery policy

            A normal disconnect can resume verified units in the same stage attempt when
            code, source manifest, condition matrix, and environment match. Preserve any
            attempt containing a `failed` unit and increment this stage's attempt label.
            Never edit manifests, raw rows, adapters, aggregate files, or frozen
            conditions in place.
            """
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": filename, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for stage in STAGES:
        path = OUTPUT_DIR / f"p0d_{stage}_colab.ipynb"
        path.write_text(
            json.dumps(build_notebook(stage), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
