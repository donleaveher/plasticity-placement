from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "pathmem_ropcd_p0_execution_colab.ipynb"
TARGET_UNIT_PATTERN = (
    r"p0r:pmv1-smoke-0[1-4]:seed-41:"
    r"(?:A1|B1|AB|BA|ABA|BAA|BAB|ABB|A2|B2|DUP-(?:ABA|BAA|BAB|ABB))"
)
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/pathmem_ropcd_p0_execution/{NOTEBOOK_NAME}"
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
RUN_PREPARE_PLAN = False  # @param {type:"boolean"}
RUN_VERIFY_PLAN = False  # @param {type:"boolean"}
RUN_AUTHORIZE = False  # @param {type:"boolean"}
RUN_PREFLIGHT = False  # @param {type:"boolean"}
RUN_TRAIN = False  # @param {type:"boolean"}
RUN_EVALUATE = False  # @param {type:"boolean"}
RUN_AGGREGATE = False  # @param {type:"boolean"}
RUN_VERIFY = False  # @param {type:"boolean"}
USE_GOOGLE_DRIVE = True  # @param {type:"boolean"}
APPROVER = ""  # @param {type:"string"}
G1C_RUN_LABEL = "g1c-r-opcd-r1"  # @param {type:"string"}
RUN_LABEL = "ropcd-p0-r1"  # @param {type:"string"}
MAX_UNITS = 1  # @param {type:"integer"}
TARGET_UNIT_ID = ""  # @param {type:"string"}
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
    "inspect": RUN_INSPECT,
    "prepare-plan": RUN_PREPARE_PLAN,
    "verify-plan": RUN_VERIFY_PLAN,
    "authorize": RUN_AUTHORIZE,
    "preflight": RUN_PREFLIGHT,
    "train": RUN_TRAIN,
    "evaluate": RUN_EVALUATE,
    "aggregate": RUN_AGGREGATE,
    "verify": RUN_VERIFY,
}}
if sum(bool(enabled) for enabled in stages.values()) != 1:
    raise ValueError(f"Enable exactly one lifecycle stage: {{stages}}")
SELECTED_STAGE = next(name for name, enabled in stages.items() if enabled)
if RUN_AUTHORIZE and not APPROVER.strip():
    raise ValueError("APPROVER must identify the responsible human")
for label_name, label in (("G1C_RUN_LABEL", G1C_RUN_LABEL), ("RUN_LABEL", RUN_LABEL)):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", label):
        raise ValueError(f"{{label_name}} contains an unsupported path character")
if MAX_UNITS < 1 or MAX_UNITS > 44:
    raise ValueError("MAX_UNITS must be between 1 and 44")
if TARGET_UNIT_ID and not re.fullmatch(r"{TARGET_UNIT_PATTERN}", TARGET_UNIT_ID):
    raise ValueError("TARGET_UNIT_ID is not a valid frozen R-OPCD P0 unit ID")

REPOSITORY_URL = "https://github.com/donleaveher/plasticity-placement.git"
REPOSITORY_BRANCH = "{BRANCH}"
REPOSITORY_ROOT = Path("/content/plasticity-placement-ropcd-p0")

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
    STORAGE_ROOT = Path("/content/drive/MyDrive/pathmem")
else:
    STORAGE_ROOT = Path("/content/pathmem")

SOURCE_BUNDLE = (
    STORAGE_ROOT / "g0-v2-r-opcd-plans/v1/g0-v2-r-opcd-plan-v1/bundle"
)
G1C_RUN_ROOT = STORAGE_ROOT / "g1c-r-opcd-executions/v1" / G1C_RUN_LABEL
PLAN_ROOT = STORAGE_ROOT / "ropcd-p0-plans/v1" / RUN_LABEL / "bundle"
RUN_ROOT = STORAGE_ROOT / "ropcd-p0-executions/v1" / RUN_LABEL
APPROVAL_PATH = STORAGE_ROOT / "ropcd-p0-approvals/v1" / f"{RUN_LABEL}.json"
CODE_REVISION_LOCK = RUN_ROOT / "code_revision.txt"

locked_revision = (
    CODE_REVISION_LOCK.read_text(encoding="utf-8").strip()
    if CODE_REVISION_LOCK.is_file()
    else None
)
pre_authorization_stages = {"inspect", "prepare-plan", "verify-plan", "authorize"}
if SELECTED_STAGE not in pre_authorization_stages and not locked_revision:
    raise FileNotFoundError(
        f"Run the authorization pass before {SELECTED_STAGE}: {CODE_REVISION_LOCK}"
    )
revision_ref = locked_revision or f"origin/{REPOSITORY_BRANCH}"
CHECKOUT_REVISION = subprocess.run(
    ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", f"{revision_ref}^{{commit}}"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if locked_revision and CHECKOUT_REVISION != locked_revision:
    raise RuntimeError("Recorded P0 execution revision no longer resolves exactly")
subprocess.run(
    ["git", "-C", str(REPOSITORY_ROOT), "checkout", "--detach", CHECKOUT_REVISION],
    check=True,
)

gpu_stage = SELECTED_STAGE in {"preflight", "train", "evaluate"}
sync_command = ["uv", "sync", "--frozen", "--no-dev"]
if gpu_stage:
    sync_command += ["--extra", "train", "--extra", "colab"]
subprocess.run(sync_command, cwd=REPOSITORY_ROOT, check=True)
if gpu_stage:
    subprocess.run(["nvidia-smi"], check=True)

PARENT_MANIFEST = (
    REPOSITORY_ROOT
    / "notebooks/pathmem_g1_p0/protocol_snapshot/experiments/g0/manifest.json"
)
if not SOURCE_BUNDLE.is_dir():
    raise FileNotFoundError(f"Missing immutable G0-v2 bundle: {SOURCE_BUNDLE}")
if not G1C_RUN_ROOT.is_dir():
    raise FileNotFoundError(f"Missing verified G1-C run: {G1C_RUN_ROOT}")
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
    "--output", str(RUN_ROOT),
]
print(
    json.dumps(
        {
            "selected_stage": SELECTED_STAGE,
            "checkout_revision": CHECKOUT_REVISION,
            "g1c_run_root": str(G1C_RUN_ROOT),
            "plan_root": str(PLAN_ROOT),
            "run_root": str(RUN_ROOT),
            "approval_path": str(APPROVAL_PATH),
            "gpu_stage": gpu_stage,
            "p1_authorized": False,
            "kill_or_reserve_access_authorized": False,
            "learned_router_authorized": False,
            "rl_controller_authorized": False,
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
            f"Command failed with exit code {completed.returncode}: "
            f"{shlex.join(command)}"
        )
    return json.loads(completed.stdout)


if RUN_INSPECT:
    result = run_json(BASE_COMMAND + ["inspect", *COMMON_ARGUMENTS])
elif RUN_PREPARE_PLAN:
    result = run_json(BASE_COMMAND + ["prepare-plan", *COMMON_ARGUMENTS])
elif RUN_VERIFY_PLAN:
    result = run_json(BASE_COMMAND + ["verify-plan", *COMMON_ARGUMENTS])
elif RUN_AUTHORIZE:
    if not PLAN_ROOT.is_dir():
        raise FileNotFoundError("Run prepare-plan and verify-plan before authorization")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if CODE_REVISION_LOCK.exists():
        if CODE_REVISION_LOCK.read_text(encoding="utf-8").strip() != CHECKOUT_REVISION:
            raise RuntimeError("P0 execution revision changed; choose a new RUN_LABEL")
    else:
        temporary_lock = CODE_REVISION_LOCK.with_suffix(".txt.tmp")
        temporary_lock.write_text(CHECKOUT_REVISION + "\\n", encoding="utf-8")
        temporary_lock.replace(CODE_REVISION_LOCK)
    template = run_json(BASE_COMMAND + ["authorization-template", *COMMON_ARGUMENTS])
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
        + ["adopt-authorization", *COMMON_ARGUMENTS, "--approval", str(APPROVAL_PATH)]
    )
elif RUN_PREFLIGHT:
    result = run_json(BASE_COMMAND + ["preflight", *COMMON_ARGUMENTS])
elif RUN_TRAIN:
    selection = ["--max-units", str(MAX_UNITS)]
    if TARGET_UNIT_ID:
        selection += ["--unit-id", TARGET_UNIT_ID]
    result = run_json(BASE_COMMAND + ["train", *COMMON_ARGUMENTS, *selection])
elif RUN_EVALUATE:
    selection = ["--max-units", str(MAX_UNITS)]
    if TARGET_UNIT_ID:
        selection += ["--unit-id", TARGET_UNIT_ID]
    result = run_json(BASE_COMMAND + ["evaluate", *COMMON_ARGUMENTS, *selection])
elif RUN_AGGREGATE:
    result = run_json(BASE_COMMAND + ["aggregate", *COMMON_ARGUMENTS])
else:
    result = run_json(BASE_COMMAND + ["verify", *COMMON_ARGUMENTS])

display(JSON(result))
"""


RESULTS = """
manifest_path = RUN_ROOT / "manifest.json"
if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unit_states = {}
    for unit in manifest.get("units", {}).values():
        unit_states[unit["state"]] = unit_states.get(unit["state"], 0) + 1
    display(
        JSON(
            {
                "run_id": manifest.get("run_id"),
                "selected_stage": SELECTED_STAGE,
                "unit_states": unit_states,
                "preflight": manifest.get("preflight"),
                "aggregate": manifest.get("aggregate"),
                "training_started": manifest.get("training_started", False),
                "gpu_inference_started": manifest.get("gpu_inference_started", False),
                "path_contrast_computed": manifest.get("path_contrast_computed", False),
                "p1_authorized": manifest.get("p1_authorized", False),
            }
        )
    )
print(f"Plan root: {PLAN_ROOT}")
print(f"Run root: {RUN_ROOT}")
print("This workflow is limited to the four-item R-OPCD P0 engineering smoke.")
print("It never authorizes P1, kill/reserve access, learned routing, RL, or HPO.")
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            markdown(
                f"""
                [Open this notebook in Google Colab]({COLAB_URL})

                # PathMem R-OPCD-specific P0 engineering smoke

                This notebook transitively verifies the immutable passing G1-C run, compiles
                a separate four-item P0 plan, and executes only after a new exact human
                authorization. P1, kill/reserve access, learned routing, RL, and HPO remain off.
                """,
                tag="colab-url",
            ),
            markdown(
                """
                ## Required passes

                Edit **only the next cell**, enable exactly one stage, and run all cells from
                the top for every pass.

                1. **Inspect (CPU):** transitively verify G1-C and print the prospective contract.
                2. **Prepare plan (CPU):** write the immutable 44-unit, 2,168-row plan bundle.
                3. **Verify plan (CPU):** regenerate and hash-check the plan before approval.
                4. **Authorize (CPU):** set `APPROVER`; lock code and adopt the exact P0 approval.
                5. **Preflight (GPU):** audit native BF16, all 2,520 prompt uses,
                   and 176 teacher rows.
                6. **Train (GPU):** repeat until all 44 lineage-aware side-memory units are trained.
                7. **Evaluate (GPU):** after training completes, repeat until all 44 units verify.
                8. **Aggregate (CPU):** compute the frozen engineering matrix and G2 gate.
                9. **Verify (CPU):** revalidate the full source, lineage, rows, summary, and report.

                Keep `G1C_RUN_LABEL`, `RUN_LABEL`, Drive setting, and GPU type fixed. Leave
                `TARGET_UNIT_ID` empty except when recovering one exact planned unit.
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
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "name": NOTEBOOK_NAME},
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
