from __future__ import annotations

from pathlib import Path
from typing import Any

from plasticity_placement.pathmem_consolidation_exec.config import RopcdExecutionRecipe
from plasticity_placement.pathmem_consolidation_exec.training import TrainingSession
from plasticity_placement.pathmem_ropcd_p0.training import (
    TrainingArtifactContract,
    prepare_root_adapter,
    train_p0_unit,
    verify_p0_unit,
    verify_root_adapter,
)
from plasticity_placement.pathmem_ropcd_p1.config import (
    CHECKPOINT_SCHEMA_VERSION,
    P1_ROPCD_RECIPE,
    ROOT_SCHEMA_VERSION,
    TRAINING_SCHEMA_VERSION,
)

P1_ARTIFACT_CONTRACT = TrainingArtifactContract(
    root_schema_version=ROOT_SCHEMA_VERSION,
    training_schema_version=TRAINING_SCHEMA_VERSION,
    checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
    label="R-OPCD P1",
    include_training_seed=True,
)


def prepare_p1_root(
    session: TrainingSession,
    *,
    root: Path,
    run_id: str,
    item_id: str,
    root_id: str,
    root_seed: int,
) -> dict[str, Any]:
    return prepare_root_adapter(
        session,
        root=root,
        run_id=run_id,
        item_id=item_id,
        root_id=root_id,
        root_seed=root_seed,
        contract=P1_ARTIFACT_CONTRACT,
    )


def verify_p1_root(
    root: Path,
    *,
    run_id: str,
    item_id: str,
    root_id: str,
    root_seed: int,
    recipe: RopcdExecutionRecipe = P1_ROPCD_RECIPE,
) -> dict[str, Any]:
    return verify_root_adapter(
        root,
        run_id=run_id,
        item_id=item_id,
        root_id=root_id,
        root_seed=root_seed,
        recipe=recipe,
        contract=P1_ARTIFACT_CONTRACT,
    )


def train_p1_unit(
    session: TrainingSession,
    *,
    unit: dict[str, Any],
    parent_adapter_dir: Path,
    root_adapter_sha256: str,
    run_id: str,
    output_root: Path,
) -> dict[str, Any]:
    return train_p0_unit(
        session,
        unit=unit,
        parent_adapter_dir=parent_adapter_dir,
        root_adapter_sha256=root_adapter_sha256,
        run_id=run_id,
        output_root=output_root,
        training_seed=int(unit["training_seed"]),
        contract=P1_ARTIFACT_CONTRACT,
    )


def verify_p1_unit(
    final_root: Path,
    *,
    unit: dict[str, Any],
    parent_adapter_sha256: str,
    root_adapter_sha256: str,
    run_id: str,
    recipe: RopcdExecutionRecipe = P1_ROPCD_RECIPE,
) -> dict[str, Any]:
    return verify_p0_unit(
        final_root,
        unit=unit,
        parent_adapter_sha256=parent_adapter_sha256,
        root_adapter_sha256=root_adapter_sha256,
        run_id=run_id,
        recipe=recipe,
        training_seed=int(unit["training_seed"]),
        contract=P1_ARTIFACT_CONTRACT,
    )
