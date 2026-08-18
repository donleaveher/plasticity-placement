from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "pathmem_g1c_execution_colab.ipynb"
TARGET_UNIT_PATTERN = r"g1c:pmv1-interface_dev-(?:0[1-9]|1[0-2]):[AB]"
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/pathmem_g1c_execution/{NOTEBOOK_NAME}"
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
RUN_AUTHORIZE = False  # @param {type:"boolean"}
RUN_PREFLIGHT = False  # @param {type:"boolean"}
RUN_TRAIN = False  # @param {type:"boolean"}
RUN_EVALUATE = False  # @param {type:"boolean"}
RUN_AGGREGATE = False  # @param {type:"boolean"}
RUN_VERIFY = False  # @param {type:"boolean"}
USE_GOOGLE_DRIVE = False  # @param {type:"boolean"}
APPROVER = ""  # @param {type:"string"}
RUN_LABEL = "g1c-r-opcd-r1"  # @param {type:"string"}
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
if not re.fullmatch(r"[A-Za-z0-9._-]+", RUN_LABEL):
    raise ValueError("RUN_LABEL contains an unsupported path character")
if MAX_UNITS < 1:
    raise ValueError("MAX_UNITS must be positive")
if TARGET_UNIT_ID and not re.fullmatch(r"{TARGET_UNIT_PATTERN}", TARGET_UNIT_ID):
    raise ValueError(
        "TARGET_UNIT_ID must match g1c:pmv1-interface_dev-01:A through "
        "g1c:pmv1-interface_dev-12:B"
    )

REPOSITORY_URL = "https://github.com/donleaveher/plasticity-placement.git"
REPOSITORY_BRANCH = "{BRANCH}"
REPOSITORY_ROOT = Path("/content/plasticity-placement-g1c-execution")

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
    STORAGE_ROOT
    / "g0-v2-r-opcd-plans/v1/g0-v2-r-opcd-plan-v1/bundle"
)
RUN_ROOT = STORAGE_ROOT / "g1c-r-opcd-executions/v1" / RUN_LABEL
APPROVAL_PATH = STORAGE_ROOT / "g1c-r-opcd-approvals/v1" / f"{RUN_LABEL}.json"
CODE_REVISION_LOCK = RUN_ROOT / "code_revision.txt"

locked_revision = (
    CODE_REVISION_LOCK.read_text(encoding="utf-8").strip()
    if CODE_REVISION_LOCK.is_file()
    else None
)
if SELECTED_STAGE not in {"inspect", "authorize"} and not locked_revision:
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
    raise RuntimeError("Recorded execution revision no longer resolves exactly")
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
    raise FileNotFoundError(
        f"Missing immutable planning bundle: {SOURCE_BUNDLE}. "
        "Prepare it with the CPU planning notebook first."
    )
if not PARENT_MANIFEST.is_file():
    raise FileNotFoundError(f"Missing frozen G0-v1 manifest: {PARENT_MANIFEST}")

BASE_COMMAND = [
    "uv", "run", "--no-sync", "python", "-m",
    "plasticity_placement.pathmem_consolidation_exec",
]
COMMON_ARGUMENTS = [
    "--bundle", str(SOURCE_BUNDLE),
    "--g0-v1-manifest", str(PARENT_MANIFEST),
    "--output", str(RUN_ROOT),
]
print(
    json.dumps(
        {
            "selected_stage": SELECTED_STAGE,
            "checkout_revision": CHECKOUT_REVISION,
            "source_bundle": str(SOURCE_BUNDLE),
            "run_root": str(RUN_ROOT),
            "approval_path": str(APPROVAL_PATH),
            "gpu_stage": gpu_stage,
            "p0_authorized": False,
            "path_contrast_authorized": False,
            "kill_or_reserve_access_authorized": False,
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
elif RUN_AUTHORIZE:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if CODE_REVISION_LOCK.exists():
        if CODE_REVISION_LOCK.read_text(encoding="utf-8").strip() != CHECKOUT_REVISION:
            raise RuntimeError("Execution revision lock changed; choose a new RUN_LABEL")
    else:
        temporary_lock = CODE_REVISION_LOCK.with_suffix(".txt.tmp")
        temporary_lock.write_text(CHECKOUT_REVISION + "\\n", encoding="utf-8")
        temporary_lock.replace(CODE_REVISION_LOCK)
    template = run_json(
        BASE_COMMAND + ["authorization-template", *COMMON_ARGUMENTS]
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
    for unit in manifest.get("units", []):
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
                "p0_started": manifest.get("p0_started", False),
                "path_contrast_computed": manifest.get(
                    "path_contrast_computed", False
                ),
            }
        )
    )
print(f"Run root: {RUN_ROOT}")
print("This workflow authorizes only G1-C R-OPCD qualification.")
print("It never authorizes or runs P0, path contrasts, kill/reserve, RL, or HPO.")
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            markdown(
                f"""
                [Open this notebook in Google Colab]({COLAB_URL})

                # PathMem G1-C R-OPCD CUDA qualification

                This notebook executes the frozen G1-C parameter-consolidation recipe against
                the immutable G0-v2 planning bundle. It is resumable at unit and checkpoint
                boundaries, binds execution to an exact Git revision and human approval, and
                keeps P0, path contrast, kill/reserve, RL, and hyperparameter search disabled.
                """,
                tag="colab-url",
            ),
            markdown(
                """
                ## Required passes

                Edit **only the next cell**, enable exactly one stage, and run all cells from
                the top for every pass.

                1. **Inspect (default, CPU):** verify the source bundle and print the frozen
                   recipe without writing an execution artifact.
                2. **Authorize (CPU):** set `APPROVER` to the responsible human and enable only
                   `RUN_AUTHORIZE`. This locks the code revision and adopts the exact approval.
                3. **Preflight (GPU):** audit CUDA/BF16, dependency and model identities, all
                   prompts, and the 96 privileged-teacher assignments.
                4. **Train (GPU):** resume up to `MAX_UNITS` isolated LoRA units per pass.
                5. **Evaluate (GPU):** after all 24 units are trained, resume up to
                   `MAX_UNITS` exact routing/locality/rollback panels per pass.
                6. **Aggregate (CPU):** regenerate the frozen 1,488-row G1-C matrix and gate.
                7. **Verify (CPU):** independently hash-check authorization, environment,
                   adapters, rows, summary, report, and regenerated metrics.

                Use a Colab runtime with an NVIDIA GPU that supports native BF16 for passes
                3–5. Keep the same Drive setting and `RUN_LABEL` throughout. `TARGET_UNIT_ID`
                is optional and is only for deterministic recovery of one planned unit.
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
