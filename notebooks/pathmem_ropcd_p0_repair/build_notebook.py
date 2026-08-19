from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "pathmem_ropcd_p0_analysis_repair_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/pathmem_ropcd_p0_repair/{NOTEBOOK_NAME}"
)


def markdown(source: str, *, tag: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {"tags": [tag]},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str, *, tag: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [tag]},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


CONTROLS = """
# @title Analysis-repair controls — enable exactly one stage
RUN_INSPECT = True  # @param {type:"boolean"}
RUN_AUTHORIZE = False  # @param {type:"boolean"}
RUN_AGGREGATE = False  # @param {type:"boolean"}
RUN_VERIFY = False  # @param {type:"boolean"}
USE_GOOGLE_DRIVE = True  # @param {type:"boolean"}
APPROVER = ""  # @param {type:"string"}
G1C_RUN_LABEL = "g1c-r-opcd-r1"  # @param {type:"string"}
SOURCE_RUN_LABEL = "ropcd-p0-r1"  # @param {type:"string"}
REPAIR_LABEL = "ropcd-p0-r1-null-obsolete-v1"  # @param {type:"string"}
"""


SETUP = f"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from IPython.display import JSON, display

stages = {{
    "repair-inspect": RUN_INSPECT,
    "repair-authorize": RUN_AUTHORIZE,
    "repair-aggregate": RUN_AGGREGATE,
    "repair-verify": RUN_VERIFY,
}}
if sum(bool(enabled) for enabled in stages.values()) != 1:
    raise ValueError(f"Enable exactly one analysis-repair stage: {{stages}}")
SELECTED_STAGE = next(name for name, enabled in stages.items() if enabled)
if RUN_AUTHORIZE and not APPROVER.strip():
    raise ValueError("APPROVER must identify the responsible human")
for label_name, label in (
    ("G1C_RUN_LABEL", G1C_RUN_LABEL),
    ("SOURCE_RUN_LABEL", SOURCE_RUN_LABEL),
    ("REPAIR_LABEL", REPAIR_LABEL),
):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", label):
        raise ValueError(f"{{label_name}} contains an unsupported path character")

REPOSITORY_URL = "https://github.com/donleaveher/plasticity-placement.git"
REPOSITORY_BRANCH = "{BRANCH}"
REPOSITORY_ROOT = Path("/content/plasticity-placement-ropcd-p0-repair")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
if REPOSITORY_ROOT.exists() and not (REPOSITORY_ROOT / ".git").is_dir():
    raise RuntimeError(f"{{REPOSITORY_ROOT}} exists but is not a Git repository")
if not REPOSITORY_ROOT.exists():
    subprocess.run(
        [
            "git", "clone", "--branch", REPOSITORY_BRANCH, "--single-branch",
            REPOSITORY_URL, str(REPOSITORY_ROOT),
        ],
        check=True,
    )
observed_remote = subprocess.run(
    ["git", "-C", str(REPOSITORY_ROOT), "remote", "get-url", "origin"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if observed_remote != REPOSITORY_URL:
    raise RuntimeError(f"Unexpected checkout remote: {{observed_remote}}")
dirty = subprocess.run(
    ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if dirty:
    raise RuntimeError(f"Repair checkout has local changes:\\n{{dirty}}")
subprocess.run(
    ["git", "-C", str(REPOSITORY_ROOT), "fetch", "origin", REPOSITORY_BRANCH],
    check=True,
)
"""


PATHS_AND_REVISION = """
if USE_GOOGLE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    STORAGE_ROOT = Path("/content/drive/MyDrive/pathmem")
else:
    STORAGE_ROOT = Path("/content/pathmem")

SOURCE_BUNDLE = STORAGE_ROOT / "g0-v2-r-opcd-plans/v1/g0-v2-r-opcd-plan-v1/bundle"
G1C_RUN_ROOT = STORAGE_ROOT / "g1c-r-opcd-executions/v1" / G1C_RUN_LABEL
PLAN_ROOT = STORAGE_ROOT / "ropcd-p0-plans/v1" / SOURCE_RUN_LABEL / "bundle"
SOURCE_RUN_ROOT = STORAGE_ROOT / "ropcd-p0-executions/v1" / SOURCE_RUN_LABEL
REPAIR_ROOT = STORAGE_ROOT / "ropcd-p0-analysis-repairs/v1" / REPAIR_LABEL
APPROVAL_PATH = STORAGE_ROOT / "ropcd-p0-analysis-repair-approvals/v1" / f"{REPAIR_LABEL}.json"
CODE_REVISION_LOCK = REPAIR_ROOT / "code_revision.txt"

locked_revision = (
    CODE_REVISION_LOCK.read_text(encoding="utf-8").strip()
    if CODE_REVISION_LOCK.is_file()
    else None
)
if SELECTED_STAGE in {"repair-aggregate", "repair-verify"} and not locked_revision:
    raise FileNotFoundError("Run the repair authorization stage before aggregation")
revision_ref = locked_revision or f"origin/{REPOSITORY_BRANCH}"
CHECKOUT_REVISION = subprocess.run(
    ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", f"{revision_ref}^{{commit}}"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
subprocess.run(
    ["git", "-C", str(REPOSITORY_ROOT), "checkout", "--detach", CHECKOUT_REVISION],
    check=True,
)
subprocess.run(
    ["uv", "sync", "--frozen", "--no-dev"],
    cwd=REPOSITORY_ROOT,
    check=True,
)

PARENT_MANIFEST = (
    REPOSITORY_ROOT
    / "notebooks/pathmem_g1_p0/protocol_snapshot/experiments/g0/manifest.json"
)
for label, path in (
    ("G0-v2 bundle", SOURCE_BUNDLE),
    ("G1-C run", G1C_RUN_ROOT),
    ("P0 plan", PLAN_ROOT),
    ("P0 source run", SOURCE_RUN_ROOT),
):
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label}: {path}")
if not PARENT_MANIFEST.is_file():
    raise FileNotFoundError(f"Missing frozen G0-v1 manifest: {PARENT_MANIFEST}")

BASE_COMMAND = [
    "uv", "run", "--no-sync", "python", "-m",
    "plasticity_placement.pathmem_ropcd_p0",
]
COMMON_ARGUMENTS = [
    "--bundle", str(SOURCE_BUNDLE),
    "--g0-v1-manifest", str(PARENT_MANIFEST),
    "--g1c-run", str(G1C_RUN_ROOT),
    "--plan", str(PLAN_ROOT),
    "--output", str(SOURCE_RUN_ROOT),
    "--repair-output", str(REPAIR_ROOT),
]
display(
    JSON(
        {
            "selected_stage": SELECTED_STAGE,
            "checkout_revision": CHECKOUT_REVISION,
            "source_run_root": str(SOURCE_RUN_ROOT),
            "repair_root": str(REPAIR_ROOT),
            "approval_path": str(APPROVAL_PATH),
            "cpu_only": True,
            "source_mutation_authorized": False,
            "training_authorized": False,
            "gpu_inference_authorized": False,
            "p1_authorized": False,
        }
    )
)
"""


LIFECYCLE = """
def run_json(command: list[str]) -> dict:
    print("$", shlex.join(command))
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: "
            f"{shlex.join(command)}"
        )
    return json.loads(completed.stdout)


if RUN_INSPECT:
    result = run_json(BASE_COMMAND + ["repair-inspect", *COMMON_ARGUMENTS])
elif RUN_AUTHORIZE:
    REPAIR_ROOT.mkdir(parents=True, exist_ok=True)
    if CODE_REVISION_LOCK.exists():
        if CODE_REVISION_LOCK.read_text(encoding="utf-8").strip() != CHECKOUT_REVISION:
            raise RuntimeError("Repair revision changed; choose a fresh REPAIR_LABEL")
    else:
        temporary_lock = CODE_REVISION_LOCK.with_suffix(".txt.tmp")
        temporary_lock.write_text(CHECKOUT_REVISION + "\\n", encoding="utf-8")
        temporary_lock.replace(CODE_REVISION_LOCK)
    template = run_json(
        BASE_COMMAND + ["repair-authorization-template", *COMMON_ARGUMENTS]
    )
    if APPROVAL_PATH.is_file():
        approval = json.loads(APPROVAL_PATH.read_text(encoding="utf-8"))
    else:
        APPROVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        approval = {
            **template,
            "decision": "approved",
            "approved_by": APPROVER.strip(),
            "approved_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary_approval = APPROVAL_PATH.with_suffix(".json.tmp")
        temporary_approval.write_text(
            json.dumps(approval, indent=2, sort_keys=True) + "\\n",
            encoding="utf-8",
        )
        temporary_approval.replace(APPROVAL_PATH)
    result = run_json(
        BASE_COMMAND
        + [
            "repair-adopt-authorization",
            *COMMON_ARGUMENTS,
            "--approval",
            str(APPROVAL_PATH),
        ]
    )
elif RUN_AGGREGATE:
    result = run_json(BASE_COMMAND + ["repair-aggregate", *COMMON_ARGUMENTS])
else:
    result = run_json(BASE_COMMAND + ["repair-verify", *COMMON_ARGUMENTS])

display(JSON(result))
"""


RESULTS = """
print(f"Immutable source run: {SOURCE_RUN_ROOT}")
print(f"Separate analysis repair: {REPAIR_ROOT}")
print("No adapters, checkpoints, result rows, or source manifest were modified.")
print("This workflow never loads a model or authorizes training, GPU inference, or P1.")
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            markdown(
                f"""
                [Open this notebook in Google Colab]({COLAB_URL})

                # R-OPCD P0 qualification-label analysis repair

                This CPU-only workflow repairs the aggregate-time interpretation of
                `obsolete_action=null`. It reads and revalidates the immutable completed
                P0 run, writes only to a separate repair namespace, and never retrains or
                rescores a prompt.
                """,
                tag="colab-url",
            ),
            markdown(
                """
                Run four separate passes by editing only the next cell:

                1. `RUN_INSPECT=True`;
                2. `RUN_AUTHORIZE=True` with a responsible `APPROVER`;
                3. `RUN_AGGREGATE=True`;
                4. `RUN_VERIFY=True`.

                Keep the source labels unchanged. A new `REPAIR_LABEL` is required if the
                repair code or approval changes.
                """,
                tag="instructions",
            ),
            code(CONTROLS, tag="controls"),
            code(SETUP, tag="setup"),
            code(PATHS_AND_REVISION, tag="paths-and-revision"),
            code(LIFECYCLE, tag="lifecycle"),
            code(RESULTS, tag="results"),
        ],
        "metadata": {
            "accelerator": "CPU",
            "colab": {"name": NOTEBOOK_NAME, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / NOTEBOOK_NAME
    payload = json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n"
    output.write_text(payload, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
