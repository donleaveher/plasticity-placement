from __future__ import annotations

from plasticity_placement.pathmem_consolidation_exec.config import (
    G1C_EXECUTION_RECIPE,
    RopcdExecutionRecipe,
)

PLAN_SCHEMA_VERSION = "pathmem-ropcd-p1-plan-v1"
PLAN_MANIFEST_SCHEMA_VERSION = "pathmem-ropcd-p1-plan-manifest-v1"
G2_HANDOFF_SCHEMA_VERSION = "pathmem-ropcd-g2-to-p1-handoff-v1"
BENCHMARK_AUTHORIZATION_SCHEMA_VERSION = "pathmem-ropcd-p1-benchmark-authorization-v1"
BENCHMARK_MANIFEST_SCHEMA_VERSION = "pathmem-ropcd-p1-benchmark-manifest-v1"
BENCHMARK_PROFILE_SCHEMA_VERSION = "pathmem-ropcd-p1-benchmark-profile-v1"
AUTHORIZATION_SCHEMA_VERSION = "pathmem-ropcd-p1-authorization-v1"
EXECUTION_MANIFEST_SCHEMA_VERSION = "pathmem-ropcd-p1-execution-manifest-v1"
PREFLIGHT_SCHEMA_VERSION = "pathmem-ropcd-p1-preflight-v1"
ROOT_SCHEMA_VERSION = "pathmem-ropcd-p1-root-v1"
TRAINING_SCHEMA_VERSION = "pathmem-ropcd-p1-training-v1"
CHECKPOINT_SCHEMA_VERSION = "pathmem-ropcd-p1-checkpoint-v1"
SUMMARY_SCHEMA_VERSION = "pathmem-ropcd-p1-summary-v1"

P1_ROPCD_RECIPE: RopcdExecutionRecipe = G1C_EXECUTION_RECIPE

OPERATOR_ID = "routed_on_policy_context_distillation_v1"
PHASE = "P1-R-OPCD"
SPLIT = "kill"
HARDWARE_SPLIT = "hardware_dev"
EXPERIMENT_SEED = 20260813
TRAINING_SEEDS = (41, 42, 43)
PANEL_SEED = P1_ROPCD_RECIPE.panel_seed
EXECUTION_ORDER_SEED = P1_ROPCD_RECIPE.execution_order_seed
BOOTSTRAP_SEED = 20260813
BOOTSTRAP_REPLICATES = 10_000

EXPECTED_ITEMS = 12
EXPECTED_ITEM_SEED_CELLS = EXPECTED_ITEMS * len(TRAINING_SEEDS)
CORE_UNITS_PER_CELL = 10
DUPLICATE_UNITS_PER_CELL = 1
EXPECTED_CORE_UNITS = EXPECTED_ITEM_SEED_CELLS * CORE_UNITS_PER_CELL
EXPECTED_DUPLICATE_UNITS = EXPECTED_ITEM_SEED_CELLS * DUPLICATE_UNITS_PER_CELL
EXPECTED_UNITS = EXPECTED_CORE_UNITS + EXPECTED_DUPLICATE_UNITS
EXPECTED_ROOTS = EXPECTED_ITEM_SEED_CELLS
EXPECTED_BENCHMARK_UNITS = 4
EXPECTED_ROWS = 18_840
EXPECTED_TEACHER_ROWS = EXPECTED_UNITS * P1_ROPCD_RECIPE.prompt_pairs_per_unit

EPSILON_JS_NATS = 0.02
CONSEQUENCE_THRESHOLD = 0.10
TECHNICAL_TOLERANCE = 1e-5
TECHNICAL_DEFECT_TOLERANCE_NATS = 0.002

PATH_SUFFIXES = {
    "ABA": ("A1", "B1", "A2"),
    "BAA": ("B1", "A1", "A2"),
    "BAB": ("B1", "A1", "B2"),
    "ABB": ("A1", "B1", "B2"),
}
FINAL_PATHS = tuple(PATH_SUFFIXES)
DUPLICATE_PATHS = FINAL_PATHS
SCORED_LOGICAL_NAMES = {*FINAL_PATHS, "A2", "B2"}

REQUIRED_ARM_COUNTS = {
    "no_memory": 336,
    "p_latest": 1_008,
    "base_before": 2_016,
    "parametric_path": 2_016,
    "parametric_rescore": 2_016,
    "adapter_disabled": 2_016,
    "external_latest": 2_016,
    "icl_history": 2_016,
    "both": 2_016,
    "technical_duplicate": 504,
    "base_unrelated": 1_152,
    "parametric_unrelated": 1_152,
    "path_qualification": 576,
}


def planning_permissions() -> dict[str, bool]:
    return {
        "hardware_dev_benchmark_training_authorized": False,
        "hardware_dev_benchmark_gpu_inference_authorized": False,
        "p1_training_authorized": False,
        "p1_gpu_inference_authorized": False,
        "kill_path_contrast_authorized": False,
        "reserve_access_authorized": False,
        "p1b_authorized": False,
        "p2_authorized": False,
        "learned_router_authorized": False,
        "rl_controller_authorized": False,
        "automatic_hyperparameter_search_authorized": False,
    }


def benchmark_permissions() -> dict[str, bool]:
    permissions = planning_permissions()
    permissions.update(
        {
            "hardware_dev_benchmark_training_authorized": True,
            "hardware_dev_benchmark_gpu_inference_authorized": True,
        }
    )
    return permissions


def execution_permissions() -> dict[str, bool]:
    permissions = planning_permissions()
    permissions.update(
        {
            "p1_training_authorized": True,
            "p1_gpu_inference_authorized": True,
            "kill_path_contrast_authorized": True,
        }
    )
    return permissions
