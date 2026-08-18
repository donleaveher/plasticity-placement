from __future__ import annotations

from plasticity_placement.pathmem_consolidation_exec.config import (
    G1C_EXECUTION_RECIPE,
    RopcdExecutionRecipe,
)

PLAN_SCHEMA_VERSION = "pathmem-ropcd-p0-plan-v1"
PLAN_MANIFEST_SCHEMA_VERSION = "pathmem-ropcd-p0-plan-manifest-v1"
AUTHORIZATION_SCHEMA_VERSION = "pathmem-ropcd-p0-authorization-v1"
EXECUTION_MANIFEST_SCHEMA_VERSION = "pathmem-ropcd-p0-execution-manifest-v1"
PREFLIGHT_SCHEMA_VERSION = "pathmem-ropcd-p0-preflight-v1"
TRAINING_SCHEMA_VERSION = "pathmem-ropcd-p0-training-v1"
CHECKPOINT_SCHEMA_VERSION = "pathmem-ropcd-p0-checkpoint-v1"
SUMMARY_SCHEMA_VERSION = "pathmem-ropcd-p0-summary-v1"

# The old 0.5B engineering model was never qualified under R-OPCD. P0 therefore
# preserves the exact model/operator recipe that passed G1-C.
P0_ROPCD_RECIPE: RopcdExecutionRecipe = G1C_EXECUTION_RECIPE

OPERATOR_ID = "routed_on_policy_context_distillation_v1"
PHASE = "P0-R-OPCD"
SPLIT = "smoke"
EXPERIMENT_SEED = 20260813
TRAINING_SEED = P0_ROPCD_RECIPE.training_seed
EXECUTION_ORDER_SEED = P0_ROPCD_RECIPE.execution_order_seed
PANEL_SEED = P0_ROPCD_RECIPE.panel_seed
EXPECTED_ITEMS = 4
CORE_UNITS_PER_ITEM = 10
DUPLICATE_UNITS_PER_ITEM = 1
EXPECTED_UNITS = EXPECTED_ITEMS * (CORE_UNITS_PER_ITEM + DUPLICATE_UNITS_PER_ITEM)
EXPECTED_ROWS = 2_168
TECHNICAL_TOLERANCE = 1e-5
TECHNICAL_DEFECT_TOLERANCE_NATS = 0.002
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260813

PATH_SUFFIXES = {
    "ABA": ("A1", "B1", "A2"),
    "BAA": ("B1", "A1", "A2"),
    "BAB": ("B1", "A1", "B2"),
    "ABB": ("A1", "B1", "B2"),
}
FINAL_PATHS = tuple(PATH_SUFFIXES)
DUPLICATE_PATH_BY_ITEM_INDEX = ("ABA", "BAA", "BAB", "ABB")
SCORED_LOGICAL_NAMES = {*FINAL_PATHS, "A2", "B2"}


def planning_permissions() -> dict[str, bool]:
    return {
        "p0_training_authorized": False,
        "p0_gpu_inference_authorized": False,
        "smoke_path_contrast_authorized": False,
        "p1_authorized": False,
        "kill_or_reserve_access_authorized": False,
        "learned_router_authorized": False,
        "rl_controller_authorized": False,
        "automatic_hyperparameter_search_authorized": False,
    }


def execution_permissions() -> dict[str, bool]:
    permissions = planning_permissions()
    permissions.update(
        {
            "p0_training_authorized": True,
            "p0_gpu_inference_authorized": True,
            "smoke_path_contrast_authorized": True,
        }
    )
    return permissions
