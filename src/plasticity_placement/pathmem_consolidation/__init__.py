"""CPU-only PathMem G0-v2 and G1-C consolidation planning."""

from plasticity_placement.pathmem_consolidation.config import (
    G1C_THRESHOLDS,
    OPERATOR_ID,
    build_g0v2_contract,
    evaluate_g1c_gate,
)
from plasticity_placement.pathmem_consolidation.planner import compile_g1c_plan

__all__ = [
    "G1C_THRESHOLDS",
    "OPERATOR_ID",
    "build_g0v2_contract",
    "compile_g1c_plan",
    "evaluate_g1c_gate",
]
