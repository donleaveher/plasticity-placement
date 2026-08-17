from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "pathmem_g1c_consolidation_colab.ipynb"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/"
    "pathmem_g1c_consolidation/"
    f"{NOTEBOOK_NAME}"
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
# @title User controls — edit only this cell between passes
RUN_INSPECT = True  # @param {type:"boolean"}
RUN_PREPARE = False  # @param {type:"boolean"}
RUN_VERIFY = False  # @param {type:"boolean"}
USE_GOOGLE_DRIVE = False  # @param {type:"boolean"}
RUN_LABEL = "g0-v2-r-opcd-plan-v1"  # @param {type:"string"}
"""


SETUP = f"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from IPython.display import JSON, display

stages = {{
    "inspect": RUN_INSPECT,
    "prepare": RUN_PREPARE,
    "verify": RUN_VERIFY,
}}
if sum(bool(enabled) for enabled in stages.values()) != 1:
    raise ValueError(f"Enable exactly one lifecycle stage: {{stages}}")
if not re.fullmatch(r"[A-Za-z0-9._-]+", RUN_LABEL):
    raise ValueError("RUN_LABEL contains an unsupported path character")

REPOSITORY_URL = "https://github.com/donleaveher/plasticity-placement.git"
REPOSITORY_BRANCH = "{BRANCH}"
REPOSITORY_ROOT = Path("/content/plasticity-placement-g1c-plan")

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "uv"],
    check=True,
)
if REPOSITORY_ROOT.exists() and not (REPOSITORY_ROOT / ".git").is_dir():
    raise RuntimeError(f"{{REPOSITORY_ROOT}} exists but is not a Git repository")
if not REPOSITORY_ROOT.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            REPOSITORY_BRANCH,
            "--single-branch",
            REPOSITORY_URL,
            str(REPOSITORY_ROOT),
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
    raise RuntimeError(f"Colab checkout has local changes:\\n{{dirty}}")
subprocess.run(
    ["git", "-C", str(REPOSITORY_ROOT), "fetch", "origin", REPOSITORY_BRANCH],
    check=True,
)
"""


IDENTITY_AND_PATHS = """
if USE_GOOGLE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    OUTPUT_BASE = Path("/content/drive/MyDrive/pathmem/g0-v2-r-opcd-plans/v1")
else:
    OUTPUT_BASE = Path("/content/pathmem/g0-v2-r-opcd-plans/v1")

RUN_ROOT = OUTPUT_BASE / RUN_LABEL
BUNDLE_ROOT = RUN_ROOT / "bundle"
CODE_REVISION_LOCK = RUN_ROOT / "code_revision.txt"
locked_revision = (
    CODE_REVISION_LOCK.read_text(encoding="utf-8").strip()
    if CODE_REVISION_LOCK.is_file()
    else None
)
if RUN_VERIFY and not locked_revision:
    raise FileNotFoundError(
        f"Verification requires the revision lock written by prepare: {CODE_REVISION_LOCK}"
    )
revision_ref = locked_revision or f"origin/{REPOSITORY_BRANCH}"
CHECKOUT_REVISION = subprocess.run(
    ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", f"{revision_ref}^{{commit}}"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if locked_revision and CHECKOUT_REVISION != locked_revision:
    raise RuntimeError("Recorded G0-v2 planning revision no longer resolves exactly")
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
if not PARENT_MANIFEST.is_file():
    raise FileNotFoundError(f"Missing frozen G0-v1 manifest: {PARENT_MANIFEST}")
BASE_COMMAND = [
    "uv",
    "run",
    "--no-sync",
    "python",
    "-m",
    "plasticity_placement.pathmem_consolidation",
]
print(
    json.dumps(
        {
            "selected_stage": next(name for name, enabled in stages.items() if enabled),
            "checkout_revision": CHECKOUT_REVISION,
            "parent_g0_v1_manifest": str(PARENT_MANIFEST),
            "bundle_root": str(BUNDLE_ROOT),
            "cpu_only": True,
        },
        indent=2,
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
            f"Command failed with exit code {completed.returncode}: {shlex.join(command)}"
        )
    return json.loads(completed.stdout)


parent_arguments = ["--g0-v1-manifest", str(PARENT_MANIFEST)]
if RUN_INSPECT:
    result = run_json(BASE_COMMAND + ["inspect", *parent_arguments, "--compact"])
elif RUN_PREPARE:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if CODE_REVISION_LOCK.exists():
        if CODE_REVISION_LOCK.read_text(encoding="utf-8").strip() != CHECKOUT_REVISION:
            raise RuntimeError("Planning revision lock changed; choose a new RUN_LABEL")
    else:
        temporary = CODE_REVISION_LOCK.with_suffix(".txt.tmp")
        temporary.write_text(CHECKOUT_REVISION + "\\n", encoding="utf-8")
        temporary.replace(CODE_REVISION_LOCK)
    result = run_json(
        BASE_COMMAND + ["prepare", *parent_arguments, "--output", str(BUNDLE_ROOT)]
    )
else:
    result = run_json(
        BASE_COMMAND + ["verify", *parent_arguments, "--output", str(BUNDLE_ROOT)]
    )

display(JSON(result))
"""


RESULTS = """
manifest_path = BUNDLE_ROOT / "manifest.json"
if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = json.loads((BUNDLE_ROOT / "g1c_plan.json").read_text(encoding="utf-8"))
    display(
        JSON(
            {
                "manifest_id": manifest["manifest_id"],
                "plan_id": plan["plan_id"],
                "counts": plan["counts"],
                "authorization": plan["authorization"],
                "training_started": manifest["training_started"],
                "gpu_inference_started": manifest["gpu_inference_started"],
                "path_contrast_computed": manifest["path_contrast_computed"],
            }
        )
    )
    print(f"Immutable planning bundle: {BUNDLE_ROOT}")
else:
    print("Inspection complete; no bundle or execution artifact was written.")
print("This notebook never trains, loads a model, runs GPU inference, or authorizes P0.")
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            markdown(
                f"""
                [Open this notebook in Google Colab]({COLAB_URL})

                # PathMem G0-v2 / G1-C R-OPCD planning bundle

                This CPU-only notebook packages the first R-OPCD implementation slice for
                Colab. It verifies the frozen G0-v1 parent, compiles the 24-unit
                `interface_dev` consolidation plan, and can persist or re-verify the immutable
                three-file planning bundle. It never loads a model, trains a side memory, uses
                CUDA, authorizes G1-C/P0, opens kill/reserve, or computes path contrasts.
                """,
                tag="colab-url",
            ),
            markdown(
                """
                ## Run passes

                Edit **only the next cell** between passes, then run all cells from the top.

                1. **Inspect (default):** leave the controls unchanged. This validates the frozen
                   parent and prints the contract/plan audit without creating an output bundle.
                2. **Prepare:** set `RUN_INSPECT=False` and `RUN_PREPARE=True`. Optionally enable
                   Drive persistence. This writes the immutable contract, plan, and manifest.
                3. **Verify:** keep the same `RUN_LABEL` and storage choice, set only
                   `RUN_VERIFY=True`, and rerun all cells. The checkout is detached at the exact
                   revision recorded by the prepare pass before every hash is checked.

                `RUN_LABEL` is a filesystem label, not an approval. None of these controls can
                authorize training or the later GPU runner.
                """,
                tag="instructions",
            ),
            code(CONTROLS, tag="user-controls"),
            code(SETUP, tag="setup"),
            code(IDENTITY_AND_PATHS, tag="identity-and-paths"),
            code(LIFECYCLE, tag="lifecycle"),
            code(RESULTS, tag="results"),
        ],
        "metadata": {
            "accelerator": "CPU",
            "colab": {"name": NOTEBOOK_NAME},
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / NOTEBOOK_NAME
    path.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
