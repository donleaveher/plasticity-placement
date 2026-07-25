from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
STAGES = ("smoke", "calibration", "pilot", "confirmatory")


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
    required = STAGES[: STAGES.index(stage) + 1]
    attempts = "\n    ".join(f"{name.upper()}_ATTEMPT = 'a1'" for name in required)
    validation = "\n".join(
        f"'{name.upper()}_ATTEMPT': {name.upper()}_ATTEMPT," for name in required
    )
    validation = "\n    ".join(validation.splitlines())
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
    from collections import Counter, defaultdict
    from datetime import datetime, timezone
    from hashlib import sha256
    from pathlib import Path

    from IPython.display import display

    REPO_URL = 'https://github.com/donleaveher/plasticity-placement.git'
    BRANCH = '{BRANCH}'
    REQUESTED_CODE_REVISION = None
    REPO_DIR = Path('/content/plasticity-placement')

    MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
    REQUESTED_MODEL_REVISION = None
    USE_4BIT = True
    MAX_LENGTH = 256
    MAX_NEW_TOKENS = 8

    PIPELINE_ATTEMPT = 'pipeline-a1'
    {attempts}

    PIPELINE_ROOT = (
        Path('/content/drive/MyDrive/plasticity-p0c/v4/pipelines')
        / PIPELINE_ATTEMPT
    )
    DRIVE_BASE = PIPELINE_ROOT / 'runs'
    CODE_REVISION_LOCK = PIPELINE_ROOT / 'code-revision.txt'
    MODEL_REVISION_LOCK = PIPELINE_ROOT / 'model-revision.txt'
    for name, value in {{
        'PIPELINE_ATTEMPT': PIPELINE_ATTEMPT,
    {validation}
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
        'The Colab repository has local changes; start a fresh runtime or save them '
        f'before continuing:\\n{dirty}'
    )

PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
locked_code_revision = (
    CODE_REVISION_LOCK.read_text().strip()
    if CODE_REVISION_LOCK.exists()
    else None
)
subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', 'origin', BRANCH], check=True)
revision_ref = REQUESTED_CODE_REVISION or locked_code_revision or f'origin/{BRANCH}'
CODE_REVISION = subprocess.run(
    ['git', '-C', str(REPO_DIR), 'rev-parse', f'{revision_ref}^{{commit}}'],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if locked_code_revision and CODE_REVISION != locked_code_revision:
    raise RuntimeError(
        f'Pipeline code lock mismatch: {locked_code_revision} != {CODE_REVISION}. '
        'Use a new PIPELINE_ATTEMPT for a different code revision.'
    )
if not locked_code_revision:
    revision_tmp = CODE_REVISION_LOCK.with_suffix('.txt.tmp')
    revision_tmp.write_text(CODE_REVISION + '\\n')
    revision_tmp.replace(CODE_REVISION_LOCK)
subprocess.run(
    ['git', '-C', str(REPO_DIR), 'checkout', '--detach', CODE_REVISION],
    check=True,
)
subprocess.run(
    ['uv', 'sync', '--extra', 'train', '--extra', 'colab'],
    cwd=REPO_DIR,
    check=True,
)
print('Checked out:', CODE_REVISION)
"""


def _workflow_source(stage: str) -> str:
    required = STAGES[: STAGES.index(stage) + 1]
    attempts = ",\n    ".join(f"'{name}': {name.upper()}_ATTEMPT" for name in required)
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
                f'Command failed with exit code {{completed.returncode}}: '
                f'{{shlex.join(command)}}'
            )
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(f'Command returned no JSON: {{shlex.join(command)}}')
        return json.loads(lines[-1])

    locked_model_revision = (
        MODEL_REVISION_LOCK.read_text().strip()
        if MODEL_REVISION_LOCK.exists()
        else None
    )
    requested_model_revision = REQUESTED_MODEL_REVISION or locked_model_revision
    resolve_command = ['uv', 'run', 'plasticity-p0c', 'resolve-model', '--model', MODEL_NAME]
    if requested_model_revision is not None:
        resolve_command.extend(['--model-revision', requested_model_revision])
    MODEL_REVISION = run_json(resolve_command)['resolved_revision']
    if not MODEL_REVISION:
        raise RuntimeError('Could not resolve an immutable model revision')
    if locked_model_revision and MODEL_REVISION != locked_model_revision:
        raise RuntimeError(
            f'Pipeline model lock mismatch: {{locked_model_revision}} != {{MODEL_REVISION}}. '
            'Use a new PIPELINE_ATTEMPT for a different model revision.'
        )
    if not locked_model_revision:
        model_revision_tmp = MODEL_REVISION_LOCK.with_suffix('.txt.tmp')
        model_revision_tmp.write_text(MODEL_REVISION + '\\n')
        model_revision_tmp.replace(MODEL_REVISION_LOCK)

    environment_context = run_json(['uv', 'run', 'plasticity-p0c', 'environment'])
    environment = environment_context['environment']
    ENVIRONMENT_FINGERPRINT = environment_context['fingerprint']
    CODE_HASH = environment['code_sha256']

    experiment_settings = {{
        'pipeline_version': 'p0c-v4',
        'model_name': MODEL_NAME,
        'model_revision': MODEL_REVISION,
        'use_4bit': USE_4BIT,
        'max_length': MAX_LENGTH,
        'max_new_tokens': MAX_NEW_TOKENS,
    }}
    settings_payload = json.dumps(
        experiment_settings,
        sort_keys=True,
        separators=(',', ':'),
    )
    SETTINGS_FINGERPRINT = sha256(settings_payload.encode()).hexdigest()[:10]
    PROVENANCE_KEY = f'code-{{CODE_HASH[:10]}}_cfg-{{SETTINGS_FINGERPRINT}}'
    PROVENANCE_ROOT = DRIVE_BASE / PROVENANCE_KEY
    STAGE_ATTEMPTS = {{
    {attempts}
    }}
    STAGE_DIRS = {{
        name: PROVENANCE_ROOT / f'{{name}}-{{attempt}}'
        for name, attempt in STAGE_ATTEMPTS.items()
    }}
    LOG_DIR = PROVENANCE_ROOT / 'logs'
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    critical_environment = {{
        key: environment.get(key)
        for key in (
            'python', 'packages', 'cuda_available', 'cuda_version', 'gpu', 'code_sha256'
        )
    }}
    frozen_context = {{
        'schema_version': 2,
        'pipeline_version': 'p0c-v4',
        'code_sha256': CODE_HASH,
        'experiment_settings': experiment_settings,
    }}
    context_path = PROVENANCE_ROOT / 'run_context.json'
    if context_path.exists():
        if json.loads(context_path.read_text()) != frozen_context:
            raise RuntimeError(f'Frozen run context mismatch: {{context_path}}')
    else:
        context_tmp = context_path.with_suffix('.json.tmp')
        context_tmp.write_text(json.dumps(frozen_context, indent=2, sort_keys=True) + '\\n')
        context_tmp.replace(context_path)

    sessions_path = PROVENANCE_ROOT / 'source_sessions.json'
    sessions = json.loads(sessions_path.read_text()) if sessions_path.exists() else []
    sessions.append({{
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'git_revision': CODE_REVISION,
        'branch': BRANCH,
        'notebook_stage': '{stage}',
        'environment_fingerprint': ENVIRONMENT_FINGERPRINT,
        'critical_environment': critical_environment,
    }})
    sessions_tmp = sessions_path.with_suffix('.json.tmp')
    sessions_tmp.write_text(json.dumps(sessions, indent=2, sort_keys=True) + '\\n')
    sessions_tmp.replace(sessions_path)

    def run_checked(label, command):
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        safe_label = re.sub(r'[^A-Za-z0-9._-]+', '-', label)
        log_path = LOG_DIR / f'{{timestamp}}-{{safe_label}}.log'
        child_environment = {{**os.environ, 'PYTHONUNBUFFERED': '1'}}
        started = time.monotonic()
        print(f'\\n[{{label}}] $ {{shlex.join(command)}}')
        print(f'[{{label}}] log: {{log_path}}')
        with log_path.open('w', encoding='utf-8') as log_file:
            log_file.write(f'$ {{shlex.join(command)}}\\n')
            log_file.flush()
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
                f'{{label}} failed with exit code {{return_code}}. Full log: {{log_path}}'
            )
        return log_path

    def read_json(path):
        return json.loads(Path(path).read_text())

    def manifest_status(output_dir):
        manifest_path = Path(output_dir) / 'manifest.json'
        if not manifest_path.exists():
            return {{'exists': False, 'message': 'No manifest yet.'}}
        manifest = read_json(manifest_path)
        return {{
            'exists': True,
            'run_id': manifest.get('run_id'),
            'selected_lessons': manifest.get('selected_lessons', []),
            'base_states': dict(Counter(manifest.get('base_arms', {{}}).values())),
            'unit_states': dict(Counter(
                unit.get('state') for unit in manifest.get('units', {{}}).values()
            )),
            'errors': manifest.get('errors', [])[-10:],
        }}

    def verify_manifest(output_dir, expected_lessons, expected_units):
        manifest_path = Path(output_dir) / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f'Missing manifest: {{manifest_path}}')
        manifest = read_json(manifest_path)
        selected = manifest.get('selected_lessons', [])
        base_arms = manifest.get('base_arms', {{}})
        units = manifest.get('units', {{}})
        if len(selected) != expected_lessons:
            raise RuntimeError(
                f'Expected {{expected_lessons}} lessons, observed {{len(selected)}}'
            )
        if len(base_arms) != expected_lessons or set(base_arms.values()) != {{'verified'}}:
            raise RuntimeError(f'Base arms incomplete: {{Counter(base_arms.values())}}')
        if len(units) != expected_units:
            raise RuntimeError(f'Expected {{expected_units}} units, observed {{len(units)}}')
        bad_units = {{
            name: unit.get('state')
            for name, unit in units.items()
            if unit.get('state') != 'verified'
            or float(unit.get('rollback_exact_match_rate', 0.0)) != 1.0
        }}
        if bad_units or manifest.get('errors'):
            raise RuntimeError(
                f'Invalid units={{bad_units}}, errors={{manifest.get("errors", [])[-5:]}}'
            )
        print(
            f'Verified manifest: lessons={{len(selected)}}, '
            f'units={{len(units)}}, rollback=1.0'
        )
        return manifest

    def screening_pair_status(output_dir):
        output_dir = Path(output_dir)
        screening_path = output_dir / 'results' / 'screening.jsonl'
        lessons_path = output_dir / 'compiled' / 'lessons.jsonl'
        if not screening_path.exists() or not lessons_path.exists():
            return {{'available': False}}
        lessons = {{
            row['lesson_id']: row
            for row in (
                json.loads(line) for line in lessons_path.read_text().splitlines()
            )
        }}
        groups = defaultdict(list)
        for line in screening_path.read_text().splitlines():
            row = json.loads(line)
            groups[row['lesson_id']].append(row)
        eligible = set()
        for lesson_id, rows in groups.items():
            accuracy = sum(bool(row['correct']) for row in rows) / len(rows)
            invalid_rate = sum(bool(row['invalid']) for row in rows) / len(rows)
            if accuracy <= 0.5 and invalid_rate <= 0.5:
                eligible.add(lesson_id)
        by_pair = defaultdict(list)
        for lesson_id in eligible:
            lesson = lessons[lesson_id]
            if lesson['split'] in {{'confirmatory', 'reserve'}}:
                by_pair[lesson['pair_id']].append(lesson)
        result = {{'available': True}}
        for lesson_type in ('fact_mapping', 'procedure_recovery'):
            result[lesson_type] = sorted(
                pair_id
                for pair_id, pair in by_pair.items()
                if len(pair) == 2
                and {{item['lesson_type'] for item in pair}} == {{lesson_type}}
            )
        return result

    def model_cli_args():
        arguments = [
            '--model', MODEL_NAME,
            '--model-revision', MODEL_REVISION,
            '--max-length', str(MAX_LENGTH),
            '--max-new-tokens', str(MAX_NEW_TOKENS),
        ]
        if USE_4BIT:
            arguments.append('--use-4bit')
        return arguments

    print('Git revision:', CODE_REVISION)
    print('Formal code hash:', CODE_HASH)
    print('Model revision:', MODEL_REVISION)
    print('GPU:', environment.get('gpu'))
    print('Pipeline root:', PIPELINE_ROOT)
    print('Provenance root:', PROVENANCE_ROOT)
    print('Stage directories:', STAGE_DIRS)
    """


STAGE_SOURCES = {
    "smoke": """
    RUN_SMOKE = False
    SMOKE_DIR = STAGE_DIRS['smoke']

    if RUN_SMOKE:
        run_checked('smoke-v4-run', [
            'uv', 'run', 'plasticity-p0c', 'run',
            '--output', str(SMOKE_DIR),
            '--tier', 'smoke',
            *model_cli_args(),
            '--rank', '8',
            '--alpha', '16',
            '--learning-rate', '2e-4',
            '--max-steps', '16',
        ])
        verify_manifest(SMOKE_DIR, expected_lessons=2, expected_units=2)
        run_checked('smoke-v4-aggregate', [
            'uv', 'run', 'plasticity-p0c', 'aggregate',
            '--output', str(SMOKE_DIR),
            '--bootstrap-samples', '1000',
        ])

    print(json.dumps(manifest_status(SMOKE_DIR), ensure_ascii=False, indent=2))
    summary_path = SMOKE_DIR / 'results' / 'aggregate' / 'summary.json'
    if summary_path.exists():
        summary = read_json(summary_path)
        display(summary['arm_summary'])
        display(summary['contrasts'])
    """,
    "calibration": """
    RUN_CALIBRATION = False
    SMOKE_DIR = STAGE_DIRS['smoke']
    CALIBRATION_DIR = STAGE_DIRS['calibration']
    CALIBRATION_REPORT = CALIBRATION_DIR / 'calibration_report.json'

    if RUN_CALIBRATION:
        verify_manifest(SMOKE_DIR, expected_lessons=2, expected_units=2)
        run_checked('calibration-v4', [
            'uv', 'run', 'plasticity-p0c', 'calibrate',
            '--output', str(CALIBRATION_DIR),
            '--model', MODEL_NAME,
            '--model-revision', MODEL_REVISION,
            '--max-length', str(MAX_LENGTH),
            '--max-new-tokens', str(MAX_NEW_TOKENS),
            '--seed', '42',
            '--bootstrap-samples', '1000',
            *(['--use-4bit'] if USE_4BIT else []),
        ])

    if CALIBRATION_REPORT.exists():
        report = read_json(CALIBRATION_REPORT)
        print('Stage 1:', Counter(
            row.get('status') for row in report.get('stage1_results', [])
        ))
        print('Final:', Counter(
            row.get('status') for row in report.get('final_results', [])
        ))
        print('Selected config:')
        print(json.dumps(report.get('selected_config'), indent=2))
        diagnostic_rows = []
        for result in report.get('final_results', []):
            metrics = result.get('combined') or {}
            failed_gates = []
            if metrics:
                if metrics['target_gain'] < 0.40:
                    failed_gates.append('target_gain<0.40')
                if metrics['p_exact'] < 0.75:
                    failed_gates.append('p_exact<0.75')
                if metrics['p_interference'] > 0.05:
                    failed_gates.append('interference>0.05')
                if metrics['invalid_increase'] > 0.05:
                    failed_gates.append('invalid_increase>0.05')
            diagnostic_rows.append({
                'candidate_id': result.get('candidate_id'),
                'status': result.get('status'),
                'qualified': result.get('qualified', False),
                'target_gain': metrics.get('target_gain'),
                'p_exact': metrics.get('p_exact'),
                'p_interference': metrics.get('p_interference'),
                'invalid_increase': metrics.get('invalid_increase'),
                'failed_gates': ', '.join(failed_gates)
                or ('PASS' if metrics else 'no metrics'),
                'stage2_error': result.get('stage2_error'),
            })
        if diagnostic_rows:
            import pandas as pd
            display(pd.DataFrame(diagnostic_rows))
    else:
        print('No v4 calibration report yet.')
    """,
    "pilot": """
    RUN_PILOT = False
    SMOKE_DIR = STAGE_DIRS['smoke']
    CALIBRATION_DIR = STAGE_DIRS['calibration']
    PILOT_DIR = STAGE_DIRS['pilot']
    CALIBRATION_REPORT = CALIBRATION_DIR / 'calibration_report.json'

    if RUN_PILOT:
        verify_manifest(SMOKE_DIR, expected_lessons=2, expected_units=2)
        if not CALIBRATION_REPORT.exists():
            raise FileNotFoundError(f'Missing calibration report: {CALIBRATION_REPORT}')
        report = read_json(CALIBRATION_REPORT)
        if not isinstance(report.get('selected_config'), dict) or not report['selected_config']:
            raise RuntimeError('Calibration has no qualified selected_config')
        run_checked('pilot-v4-run', [
            'uv', 'run', 'plasticity-p0c', 'run',
            '--output', str(PILOT_DIR),
            '--tier', 'pilot',
            *model_cli_args(),
            '--calibration-config', str(CALIBRATION_REPORT),
        ])
        verify_manifest(PILOT_DIR, expected_lessons=8, expected_units=8)
        run_checked('pilot-v4-aggregate', [
            'uv', 'run', 'plasticity-p0c', 'aggregate',
            '--output', str(PILOT_DIR),
            '--bootstrap-samples', '10000',
        ])

    print(json.dumps(manifest_status(PILOT_DIR), ensure_ascii=False, indent=2))
    print('Screening:', screening_pair_status(PILOT_DIR))
    summary_path = PILOT_DIR / 'results' / 'aggregate' / 'summary.json'
    if summary_path.exists():
        summary = read_json(summary_path)
        display(summary['arm_summary'])
        display(summary['contrasts'])
        lesson_contrasts = summary['lesson_contrasts']
        positive_pn = sum(
            row['target_difference'] > 0
            for row in lesson_contrasts if row['contrast'] == 'P-N'
        )
        nontrivial_pe = sum(
            abs(row['target_difference']) >= 0.10
            or abs(row['interference_difference']) >= 0.10
            for row in lesson_contrasts if row['contrast'] == 'P-E'
        )
        print('P>N lessons:', positive_pn, '/ 8 (requires >= 6)')
        print('Nontrivial P/E lessons:', nontrivial_pe, '/ 8 (requires >= 2)')
        print('Review pair/action diagnostics before declaring v4 Pilot GO.')
    """,
    "confirmatory": """
    RUN_CONFIRMATORY = False
    PILOT_GO_CONFIRMED = False
    CALIBRATION_DIR = STAGE_DIRS['calibration']
    PILOT_DIR = STAGE_DIRS['pilot']
    CONFIRMATORY_DIR = STAGE_DIRS['confirmatory']
    CALIBRATION_REPORT = CALIBRATION_DIR / 'calibration_report.json'
    PILOT_SUMMARY = PILOT_DIR / 'results' / 'aggregate' / 'summary.json'

    if RUN_CONFIRMATORY:
        verify_manifest(PILOT_DIR, expected_lessons=8, expected_units=8)
        if not PILOT_SUMMARY.exists():
            raise FileNotFoundError(f'Missing Pilot summary: {PILOT_SUMMARY}')
        if not PILOT_GO_CONFIRMED:
            raise RuntimeError(
                'Review the v4 Pilot gates, then set PILOT_GO_CONFIRMED = True'
            )
        if not CALIBRATION_REPORT.exists():
            raise FileNotFoundError(f'Missing calibration report: {CALIBRATION_REPORT}')
        report = read_json(CALIBRATION_REPORT)
        if not isinstance(report.get('selected_config'), dict) or not report['selected_config']:
            raise RuntimeError('Calibration has no qualified selected_config')
        run_checked('confirmatory-v4-run', [
            'uv', 'run', 'plasticity-p0c', 'run',
            '--output', str(CONFIRMATORY_DIR),
            '--tier', 'confirmatory',
            *model_cli_args(),
            '--seeds', '41', '42', '43',
            '--calibration-config', str(CALIBRATION_REPORT),
        ])
        verify_manifest(CONFIRMATORY_DIR, expected_lessons=24, expected_units=72)
        run_checked('confirmatory-v4-aggregate', [
            'uv', 'run', 'plasticity-p0c', 'aggregate',
            '--output', str(CONFIRMATORY_DIR),
            '--bootstrap-samples', '10000',
        ])

    print(json.dumps(manifest_status(CONFIRMATORY_DIR), ensure_ascii=False, indent=2))
    print('Screening:', screening_pair_status(CONFIRMATORY_DIR))
    summary_path = CONFIRMATORY_DIR / 'results' / 'aggregate' / 'summary.json'
    if summary_path.exists():
        summary = read_json(summary_path)
        if summary['selected_lesson_count'] != 24 or summary['probe_row_count'] != 6864:
            raise RuntimeError('Unexpected Confirmatory result dimensions')
        display(summary['arm_summary'])
        display(summary['contrasts'])
        display(summary['seed_stability'])
        display(summary['pair_action_diagnostics'])
    """,
}


def build_notebook(stage: str) -> dict[str, Any]:
    title = stage.capitalize()
    filename = f"p0c_{stage}_v4_colab.ipynb"
    badge = (
        "https://colab.research.google.com/github/donleaveher/"
        "plasticity-placement/blob/"
        f"{BRANCH.replace('/', '%2F')}/notebooks/p0c_v4/{filename}"
    )
    dependencies = {
        "smoke": "No upstream experiment.",
        "calibration": "Requires a verified Smoke v4 attempt.",
        "pilot": "Requires verified Smoke v4 and a successful Calibration v4 report.",
        "confirmatory": "Requires a successful Calibration v4 report and reviewed Pilot v4 GO.",
    }
    cells = [
        _markdown(
            f"""
            <a href="{badge}" target="_parent"><img
            src="https://colab.research.google.com/assets/colab-badge.svg"
            alt="Open In Colab"/></a>

            # P0-C v4 — {title}

            Standalone v4 notebook for **{stage} only**. {dependencies[stage]}
            It uses the compiler-v4 bank, immutable code/model revisions, environment
            per-session environment fingerprints, phase-attempt directories, streamed
            Drive logs, and strict manifest verification. Different stages may use
            different Colab GPU runtimes.

            Do not use this notebook to run another experiment stage.
            """
        ),
        _markdown(
            """
            ## 1. Configuration

            Select a GPU runtime. Increment only this notebook's stage attempt after an
            immutable failure; keep older attempts for audit. All upstream attempt labels
            must point to completed runs under the same provenance root.
            """
        ),
        _code(_configuration_source(stage)),
        _markdown("## 2. Checkout and install the frozen experiment code"),
        _code(CHECKOUT_SOURCE),
        _markdown("## 3. Resolve provenance, directories, logging, and recovery helpers"),
        _code(_workflow_source(stage)),
        _markdown(f"## 4. Run {title} v4"),
        _code(STAGE_SOURCES[stage]),
        _markdown(
            """
            ## Recovery policy

            A normal disconnect may resume the same attempt when provenance matches.
            Preserve any attempt containing an immutable `failed` unit and increment only
            that stage's attempt label. Never edit manifests, calibration reports,
            compiled artifacts, or screening results by hand.
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
        path = OUTPUT_DIR / f"p0c_{stage}_v4_colab.ipynb"
        path.write_text(
            json.dumps(build_notebook(stage), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
