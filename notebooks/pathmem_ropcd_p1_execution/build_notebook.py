from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

BRANCH = "agent/add-lora-evaluation"
OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_NAME = "pathmem_ropcd_p1_execution_colab.ipynb"
TARGET_UNIT_PATTERN = (
    r"p1r:pmv1-kill-(?:0[1-9]|1[0-2]):seed-(?:41|42|43):"
    r"(?:A1|B1|AB|BA|ABA|BAA|BAB|ABB|A2|B2|DUP-(?:ABA|BAA|BAB|ABB))"
)
COLAB_URL = (
    "https://colab.research.google.com/github/donleaveher/"
    "plasticity-placement/blob/"
    f"{BRANCH.replace('/', '%2F')}/notebooks/pathmem_ropcd_p1_execution/{NOTEBOOK_NAME}"
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
# @title P1 controls — edit only this cell between passes
RUN_INSPECT = True  # @param {type:"boolean"}
RUN_PREPARE_PLAN = False  # @param {type:"boolean"}
RUN_VERIFY_PLAN = False  # @param {type:"boolean"}
RUN_BENCHMARK_AUTHORIZE = False  # @param {type:"boolean"}
RUN_BENCHMARK = False  # @param {type:"boolean"}
RUN_BENCHMARK_AGGREGATE = False  # @param {type:"boolean"}
RUN_BENCHMARK_VERIFY = False  # @param {type:"boolean"}
RUN_AUTHORIZE = False  # @param {type:"boolean"}
RUN_PREFLIGHT = False  # @param {type:"boolean"}
RUN_TRAIN = False  # @param {type:"boolean"}
RUN_EVALUATE = False  # @param {type:"boolean"}
RUN_AGGREGATE = False  # @param {type:"boolean"}
RUN_VERIFY = False  # @param {type:"boolean"}
USE_GOOGLE_DRIVE = True  # @param {type:"boolean"}
APPROVER = ""  # @param {type:"string"}
G1C_RUN_LABEL = "g1c-r-opcd-r1"  # @param {type:"string"}
P0_RUN_LABEL = "ropcd-p0-r1"  # @param {type:"string"}
G2_REPAIR_LABEL = "ropcd-p0-r1-null-obsolete-v1"  # @param {type:"string"}
RUN_LABEL = "ropcd-p1-r1"  # @param {type:"string"}
MAX_BENCHMARK_UNITS = 1  # @param {type:"integer"}
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
    "benchmark-authorize": RUN_BENCHMARK_AUTHORIZE,
    "benchmark-run": RUN_BENCHMARK,
    "benchmark-aggregate": RUN_BENCHMARK_AGGREGATE,
    "benchmark-verify": RUN_BENCHMARK_VERIFY,
    "authorize": RUN_AUTHORIZE,
    "preflight": RUN_PREFLIGHT,
    "train": RUN_TRAIN,
    "evaluate": RUN_EVALUATE,
    "aggregate": RUN_AGGREGATE,
    "verify": RUN_VERIFY,
}}
if sum(bool(enabled) for enabled in stages.values()) != 1:
    raise ValueError(f"Enable exactly one P1 lifecycle stage: {{stages}}")
SELECTED_STAGE = next(name for name, enabled in stages.items() if enabled)
if (RUN_BENCHMARK_AUTHORIZE or RUN_AUTHORIZE) and not APPROVER.strip():
    raise ValueError("APPROVER must identify the responsible human")
for label_name, label in (
    ("G1C_RUN_LABEL", G1C_RUN_LABEL),
    ("P0_RUN_LABEL", P0_RUN_LABEL),
    ("G2_REPAIR_LABEL", G2_REPAIR_LABEL),
    ("RUN_LABEL", RUN_LABEL),
):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", label):
        raise ValueError(f"{{label_name}} contains an unsupported path character")
if MAX_BENCHMARK_UNITS < 1 or MAX_BENCHMARK_UNITS > 4:
    raise ValueError("MAX_BENCHMARK_UNITS must be between 1 and 4")
if MAX_UNITS < 1 or MAX_UNITS > 396:
    raise ValueError("MAX_UNITS must be between 1 and 396")
if TARGET_UNIT_ID and not re.fullmatch(r"{TARGET_UNIT_PATTERN}", TARGET_UNIT_ID):
    raise ValueError("TARGET_UNIT_ID is not a valid frozen R-OPCD P1 unit ID")

REPOSITORY_URL = "https://github.com/donleaveher/plasticity-placement.git"
REPOSITORY_BRANCH = "{BRANCH}"
REPOSITORY_ROOT = Path("/content/plasticity-placement-ropcd-p1")

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
    raise RuntimeError(f"P1 checkout has local changes:\\n{{dirty}}")
subprocess.run(
    ["git", "-C", str(REPOSITORY_ROOT), "fetch", "origin", REPOSITORY_BRANCH],
    check=True,
)
"""


PATHS_AND_IDENTITY = """
if USE_GOOGLE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    STORAGE_ROOT = Path("/content/drive/MyDrive/pathmem")
else:
    STORAGE_ROOT = Path("/content/pathmem")

SOURCE_BUNDLE = STORAGE_ROOT / "g0-v2-r-opcd-plans/v1/g0-v2-r-opcd-plan-v1/bundle"
G1C_RUN_ROOT = STORAGE_ROOT / "g1c-r-opcd-executions/v1" / G1C_RUN_LABEL
P0_PLAN_ROOT = STORAGE_ROOT / "ropcd-p0-plans/v1" / P0_RUN_LABEL / "bundle"
P0_RUN_ROOT = STORAGE_ROOT / "ropcd-p0-executions/v1" / P0_RUN_LABEL
G2_REPAIR_ROOT = STORAGE_ROOT / "ropcd-p0-analysis-repairs/v1" / G2_REPAIR_LABEL
PLAN_ROOT = STORAGE_ROOT / "ropcd-p1-plans/v1" / RUN_LABEL / "bundle"
BENCHMARK_ROOT = STORAGE_ROOT / "ropcd-p1-benchmarks/v1" / RUN_LABEL
RESOURCE_PROFILE = BENCHMARK_ROOT / "resource_profile.json"
RUN_ROOT = STORAGE_ROOT / "ropcd-p1-executions/v1" / RUN_LABEL
BENCHMARK_APPROVAL = STORAGE_ROOT / "ropcd-p1-benchmark-approvals/v1" / f"{RUN_LABEL}.json"
RUN_APPROVAL = STORAGE_ROOT / "ropcd-p1-approvals/v1" / f"{RUN_LABEL}.json"
CODE_REVISION_LOCK = STORAGE_ROOT / "ropcd-p1-code-locks/v1" / f"{RUN_LABEL}.txt"

locked_revision = (
    CODE_REVISION_LOCK.read_text(encoding="utf-8").strip()
    if CODE_REVISION_LOCK.is_file()
    else None
)
if SELECTED_STAGE not in {"inspect", "prepare-plan"} and not locked_revision:
    raise FileNotFoundError("Run prepare-plan before later P1 stages")
revision_ref = locked_revision or f"origin/{REPOSITORY_BRANCH}"
CHECKOUT_REVISION = subprocess.run(
    ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", f"{revision_ref}^{{commit}}"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if locked_revision and CHECKOUT_REVISION != locked_revision:
    raise RuntimeError("Recorded P1 revision no longer resolves exactly")
subprocess.run(
    ["git", "-C", str(REPOSITORY_ROOT), "checkout", "--detach", CHECKOUT_REVISION],
    check=True,
)

gpu_stage = SELECTED_STAGE in {"benchmark-run", "preflight", "train", "evaluate"}
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
required_sources = {
    "G0-v2 bundle": SOURCE_BUNDLE,
    "G1-C run": G1C_RUN_ROOT,
    "P0 plan": P0_PLAN_ROOT,
    "P0 run": P0_RUN_ROOT,
    "G2 repair": G2_REPAIR_ROOT,
}
for label, path in required_sources.items():
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {{label}}: {{path}}")
if not PARENT_MANIFEST.is_file():
    raise FileNotFoundError(f"Missing frozen G0-v1 manifest: {{PARENT_MANIFEST}}")

BASE_COMMAND = [
    "uv", "run", "--no-sync", "python", "-m",
    "plasticity_placement.pathmem_ropcd_p1",
]
SOURCE_ARGUMENTS = [
    "--bundle", str(SOURCE_BUNDLE),
    "--g0-v1-manifest", str(PARENT_MANIFEST),
    "--g1c-run", str(G1C_RUN_ROOT),
    "--p0-plan", str(P0_PLAN_ROOT),
    "--p0-run", str(P0_RUN_ROOT),
    "--g2-repair", str(G2_REPAIR_ROOT),
]
PLAN_ARGUMENTS = [*SOURCE_ARGUMENTS, "--plan", str(PLAN_ROOT)]
INSPECT_ARGUMENTS = [*PLAN_ARGUMENTS, "--output", str(RUN_ROOT)]
BENCHMARK_ARGUMENTS = [*PLAN_ARGUMENTS, "--output", str(BENCHMARK_ROOT)]
RUN_ARGUMENTS = [
    *PLAN_ARGUMENTS,
    "--output", str(RUN_ROOT),
    "--resource-profile", str(RESOURCE_PROFILE),
]
display(
    JSON(
        {
            "selected_stage": SELECTED_STAGE,
            "checkout_revision": CHECKOUT_REVISION,
            "plan_root": str(PLAN_ROOT),
            "benchmark_root": str(BENCHMARK_ROOT),
            "resource_profile": str(RESOURCE_PROFILE),
            "run_root": str(RUN_ROOT),
            "gpu_stage": gpu_stage,
            "reserve_access_authorized": False,
            "p1b_authorized": False,
            "p2_authorized": False,
            "learned_router_authorized": False,
            "rl_controller_authorized": False,
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


def approve(template_command: list[str], adopt_command: list[str], path: Path) -> dict:
    template = run_json(template_command)
    if path.is_file():
        approval = json.loads(path.read_text(encoding="utf-8"))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        approval = {
            **template,
            "decision": "approved",
            "approved_by": APPROVER.strip(),
            "approved_at": datetime.now(timezone.utc).isoformat(),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(approval, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
        )
        temporary.replace(path)
    return run_json([*adopt_command, "--approval", str(path)])


if RUN_INSPECT:
    result = run_json(BASE_COMMAND + ["inspect", *INSPECT_ARGUMENTS])
elif RUN_PREPARE_PLAN:
    result = run_json(BASE_COMMAND + ["prepare-plan", *INSPECT_ARGUMENTS])
    CODE_REVISION_LOCK.parent.mkdir(parents=True, exist_ok=True)
    if CODE_REVISION_LOCK.is_file():
        if CODE_REVISION_LOCK.read_text(encoding="utf-8").strip() != CHECKOUT_REVISION:
            raise RuntimeError("P1 revision lock changed; choose a new RUN_LABEL")
    else:
        temporary = CODE_REVISION_LOCK.with_suffix(".txt.tmp")
        temporary.write_text(CHECKOUT_REVISION + "\\n", encoding="utf-8")
        temporary.replace(CODE_REVISION_LOCK)
elif RUN_VERIFY_PLAN:
    result = run_json(BASE_COMMAND + ["verify-plan", *INSPECT_ARGUMENTS])
elif RUN_BENCHMARK_AUTHORIZE:
    result = approve(
        BASE_COMMAND + ["benchmark-authorization-template", *BENCHMARK_ARGUMENTS],
        BASE_COMMAND + ["benchmark-adopt-authorization", *BENCHMARK_ARGUMENTS],
        BENCHMARK_APPROVAL,
    )
elif RUN_BENCHMARK:
    result = run_json(
        BASE_COMMAND
        + ["benchmark-run", *BENCHMARK_ARGUMENTS, "--max-units", str(MAX_BENCHMARK_UNITS)]
    )
elif RUN_BENCHMARK_AGGREGATE:
    result = run_json(BASE_COMMAND + ["benchmark-aggregate", *BENCHMARK_ARGUMENTS])
elif RUN_BENCHMARK_VERIFY:
    result = run_json(BASE_COMMAND + ["benchmark-verify", *BENCHMARK_ARGUMENTS])
elif RUN_AUTHORIZE:
    if not RESOURCE_PROFILE.is_file():
        raise FileNotFoundError("Run and verify the hardware benchmark before P1 authorization")
    result = approve(
        BASE_COMMAND + ["authorization-template", *RUN_ARGUMENTS],
        BASE_COMMAND + ["adopt-authorization", *RUN_ARGUMENTS],
        RUN_APPROVAL,
    )
elif RUN_PREFLIGHT:
    result = run_json(BASE_COMMAND + ["preflight", *RUN_ARGUMENTS])
elif RUN_TRAIN:
    selection = ["--max-units", str(MAX_UNITS)]
    if TARGET_UNIT_ID:
        selection += ["--unit-id", TARGET_UNIT_ID]
    result = run_json(BASE_COMMAND + ["train", *RUN_ARGUMENTS, *selection])
elif RUN_EVALUATE:
    selection = ["--max-units", str(MAX_UNITS)]
    if TARGET_UNIT_ID:
        selection += ["--unit-id", TARGET_UNIT_ID]
    result = run_json(BASE_COMMAND + ["evaluate", *RUN_ARGUMENTS, *selection])
elif RUN_AGGREGATE:
    result = run_json(BASE_COMMAND + ["aggregate", *RUN_ARGUMENTS])
else:
    result = run_json(BASE_COMMAND + ["verify", *RUN_ARGUMENTS])

display(JSON(result))
"""


RESULTS = """
for label, path in (
    ("benchmark", BENCHMARK_ROOT / "benchmark_manifest.json"),
    ("formal P1", RUN_ROOT / "manifest.json"),
):
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        states = {}
        for unit in manifest.get("units", {}).values():
            states[unit["state"]] = states.get(unit["state"], 0) + 1
        display(
            JSON(
                {
                    "artifact": label,
                    "run_or_benchmark_id": manifest.get(
                        "run_id", manifest.get("benchmark_id")
                    ),
                    "unit_states": states,
                    "preflight": manifest.get("preflight"),
                    "profile": manifest.get("profile"),
                    "aggregate": manifest.get("aggregate"),
                    "training_started": manifest.get("training_started", False),
                    "gpu_inference_started": manifest.get(
                        "gpu_inference_started", False
                    ),
                    "kill_path_contrast_computed": manifest.get(
                        "kill_path_contrast_computed", False
                    ),
                    "p1b_authorized": manifest.get("p1b_authorized", False),
                    "p2_authorized": manifest.get("p2_authorized", False),
                }
            )
        )
print(f"Plan root: {PLAN_ROOT}")
print(f"Benchmark root: {BENCHMARK_ROOT}")
print(f"Formal P1 root: {RUN_ROOT}")
print("P1b, P2, reserve access, learned routing, RL, and HPO remain unauthorized.")
"""


def build_notebook() -> dict[str, Any]:
    return {
        "cells": [
            markdown(
                f"""
                [Open this notebook in Google Colab]({COLAB_URL})

                # PathMem R-OPCD P1 investment/kill test

                This notebook verifies the immutable repaired G2 source, compiles the frozen
                12-item/3-seed P1 plan, runs a separately authorized four-item hardware
                benchmark, and requires a second resource-bound authorization before any
                `kill` split access. It never authorizes P1b or P2 automatically.
                """,
                tag="colab-url",
            ),
            markdown(
                """
                ## Required passes

                Edit **only the next cell**, enable exactly one stage, then run all cells.

                1. Inspect, prepare, and verify the CPU plan.
                2. Authorize the `hardware_dev` benchmark, repeat it until 4 units verify,
                   then aggregate and verify `resource_profile.json`.
                3. Authorize formal P1 against that exact profile.
                4. Run preflight; repeat training until all 396 units are trained.
                5. Only after all training completes, repeat evaluation until all 396 units
                   verify; then aggregate and verify the 18,840-row decision.

                Keep every label, the Drive setting, GPU type, and code revision fixed.
                Leave `TARGET_UNIT_ID` empty in normal operation; use a full ID only to resume
                one exact planned unit. A passing or meaningful result is review eligibility,
                not downstream authorization.
                """,
                tag="instructions",
            ),
            code(CONTROLS, tag="user-controls"),
            code(SETUP, tag="setup"),
            code(PATHS_AND_IDENTITY, tag="paths-and-identity"),
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
