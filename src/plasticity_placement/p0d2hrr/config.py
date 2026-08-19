from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from plasticity_placement.p0d2hcrd.config import (
    EXPECTED_MODEL_NAME,
    EXPECTED_MODEL_REVISION,
)
from plasticity_placement.training.config import LayerBand, LoraTrainingConfig

PILOT_SCHEMA_VERSION = "p0d2hrr-spec-v1"
PILOT_CONFIG_SCHEMA_VERSION = "p0d2hrr-config-v1"
PILOT_STAGE = "route_remediation_lora_pilot"
EXPERIMENT_ID = "P1-ROUTE-REMEDIATION-LORA-PILOT-v1"
DATA_GENERATOR_VERSION = "p0d2hrr-counterfactual-route-bank-v1"
AUTHORIZATION_SCOPE = "single_route_remediation_lora_pilot"
OFFICIAL_EVALUATION_PRECISION = "source_compatible"
TARGET_MODULES = ("q_proj", "v_proj")
TRAIN_GROUP_COUNT = 60
DEV_GROUP_COUNT = 12
EXAMPLES_PER_GROUP = 8
TRAINING_SEED = 20260729

CONDITIONAL_ROUTE_MIN_ACCURACY = 0.75
ROUTE_ONLY_BASELINE_ACCURACY = 0.625
RETRIEVAL_ONLY_MIN_ACCURACY = 1.0
COMBINED_MIN_ACCURACY = 0.9688
MAX_TIE_ERROR_NONFINITE_RATE = 0.0


@dataclass(frozen=True, slots=True)
class RouteDataSpec:
    generator_version: str = DATA_GENERATOR_VERSION
    train_group_count: int = TRAIN_GROUP_COUNT
    dev_group_count: int = DEV_GROUP_COUNT
    examples_per_group: int = EXAMPLES_PER_GROUP
    seed: int = TRAINING_SEED

    def __post_init__(self) -> None:
        if self.generator_version != DATA_GENERATOR_VERSION:
            raise ValueError("route-remediation data generator version changed")
        if (
            self.train_group_count != TRAIN_GROUP_COUNT
            or self.dev_group_count != DEV_GROUP_COUNT
            or self.examples_per_group != EXAMPLES_PER_GROUP
        ):
            raise ValueError("route-remediation train/dev matrix is frozen")
        if self.seed != TRAINING_SEED:
            raise ValueError("route-remediation data seed is frozen")

    @property
    def train_example_count(self) -> int:
        return self.train_group_count * self.examples_per_group

    @property
    def dev_example_count(self) -> int:
        return self.dev_group_count * self.examples_per_group


@dataclass(frozen=True, slots=True)
class RouteTrainingSpec:
    layer_band: str = LayerBand.FULL.value
    target_modules: tuple[str, ...] = TARGET_MODULES
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    learning_rate: float = 2e-4
    epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    warmup_ratio: float = 0.03
    max_steps: int = 30
    seed: int = TRAINING_SEED
    use_4bit: bool = True
    gradient_checkpointing: bool = True
    use_chat_template: bool = True
    checkpoint_selection: str = "fixed_final_step"

    def __post_init__(self) -> None:
        expected = (
            LayerBand.FULL.value,
            TARGET_MODULES,
            8,
            16,
            0.0,
            2e-4,
            1,
            4,
            4,
            512,
            0.03,
            30,
            TRAINING_SEED,
            True,
            True,
            True,
            "fixed_final_step",
        )
        observed = (
            self.layer_band,
            self.target_modules,
            self.rank,
            self.alpha,
            self.dropout,
            self.learning_rate,
            self.epochs,
            self.batch_size,
            self.gradient_accumulation_steps,
            self.max_length,
            self.warmup_ratio,
            self.max_steps,
            self.seed,
            self.use_4bit,
            self.gradient_checkpointing,
            self.use_chat_template,
            self.checkpoint_selection,
        )
        if observed != expected:
            raise ValueError(
                "route-remediation LoRA configuration is frozen; "
                "use a new protocol version to change it"
            )

    def to_lora_config(
        self,
        *,
        model_name: str,
        model_revision: str,
        data_path: Path,
        output_dir: Path,
    ) -> LoraTrainingConfig:
        return LoraTrainingConfig(
            model_name=model_name,
            model_revision=model_revision,
            data_path=data_path,
            output_dir=output_dir,
            layer_band=LayerBand(self.layer_band),
            target_modules=self.target_modules,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_length=self.max_length,
            warmup_ratio=self.warmup_ratio,
            max_steps=self.max_steps,
            seed=self.seed,
            use_4bit=self.use_4bit,
            gradient_checkpointing=self.gradient_checkpointing,
            use_chat_template=self.use_chat_template,
        )


@dataclass(frozen=True, slots=True)
class RouteGateSpec:
    conditional_route_min_accuracy: float = CONDITIONAL_ROUTE_MIN_ACCURACY
    route_only_strictly_greater_than: float = ROUTE_ONLY_BASELINE_ACCURACY
    retrieval_only_min_accuracy: float = RETRIEVAL_ONLY_MIN_ACCURACY
    combined_min_accuracy: float = COMBINED_MIN_ACCURACY
    max_tie_error_nonfinite_rate: float = MAX_TIE_ERROR_NONFINITE_RATE

    def __post_init__(self) -> None:
        if (
            self.conditional_route_min_accuracy,
            self.route_only_strictly_greater_than,
            self.retrieval_only_min_accuracy,
            self.combined_min_accuracy,
            self.max_tie_error_nonfinite_rate,
        ) != (
            CONDITIONAL_ROUTE_MIN_ACCURACY,
            ROUTE_ONLY_BASELINE_ACCURACY,
            RETRIEVAL_ONLY_MIN_ACCURACY,
            COMBINED_MIN_ACCURACY,
            MAX_TIE_ERROR_NONFINITE_RATE,
        ):
            raise ValueError(
                "route-remediation decision thresholds are frozen; "
                "use a new protocol version to change them"
            )


@dataclass(frozen=True, slots=True)
class PilotSpec:
    schema_version: str = PILOT_SCHEMA_VERSION
    experiment_id: str = EXPERIMENT_ID
    model_name: str = EXPECTED_MODEL_NAME
    model_revision: str = EXPECTED_MODEL_REVISION
    data: RouteDataSpec = RouteDataSpec()
    training: RouteTrainingSpec = RouteTrainingSpec()
    gates: RouteGateSpec = RouteGateSpec()
    official_evaluation_precision: str = OFFICIAL_EVALUATION_PRECISION
    allowed_external_evaluations: int = 1
    allows_hyperparameter_search: bool = False
    allows_mappings_per_adapter_scan: bool = False
    allows_rlvr: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != PILOT_SCHEMA_VERSION:
            raise ValueError("unsupported route-remediation spec schema")
        if self.experiment_id != EXPERIMENT_ID:
            raise ValueError("route-remediation experiment ID changed")
        if self.model_name != EXPECTED_MODEL_NAME or self.model_revision != EXPECTED_MODEL_REVISION:
            raise ValueError("route-remediation model identity changed")
        if self.official_evaluation_precision != OFFICIAL_EVALUATION_PRECISION:
            raise ValueError("official evaluation precision policy changed")
        if self.allowed_external_evaluations != 1:
            raise ValueError("pilot permits exactly one locked external evaluation")
        if (
            self.allows_hyperparameter_search
            or self.allows_mappings_per_adapter_scan
            or self.allows_rlvr
        ):
            raise ValueError("pilot scope cannot authorize scans or RLVR")

    @classmethod
    def from_path(cls, path: Path) -> PilotSpec:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PilotSpec:
        if not isinstance(payload, dict):
            raise ValueError("pilot spec must be a JSON object")
        allowed = {
            "schema_version",
            "experiment_id",
            "model_name",
            "model_revision",
            "data",
            "training",
            "gates",
            "official_evaluation_precision",
            "allowed_external_evaluations",
            "allows_hyperparameter_search",
            "allows_mappings_per_adapter_scan",
            "allows_rlvr",
        }
        extras = set(payload) - allowed
        if extras:
            raise ValueError(f"pilot spec contains unsupported fields: {sorted(extras)}")
        data = RouteDataSpec(**payload.get("data", {}))
        training_payload = dict(payload.get("training", {}))
        if "target_modules" in training_payload:
            training_payload["target_modules"] = tuple(training_payload["target_modules"])
        training = RouteTrainingSpec(**training_payload)
        gates = RouteGateSpec(**payload.get("gates", {}))
        return cls(
            **{
                key: value
                for key, value in payload.items()
                if key not in {"data", "training", "gates"}
            },
            data=data,
            training=training,
            gates=gates,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["training"]["target_modules"] = list(self.training.target_modules)
        return payload


@dataclass(frozen=True, slots=True)
class ResolvedPilotConfig:
    output_dir: Path
    source_manifest_path: Path
    code_sha256: str
    source_run_id: str
    source_manifest_sha256: str
    source_summary_sha256: str
    source_candidate_token_audit_sha256: str
    source_raw_results_sha256: str
    spec: PilotSpec
    spec_sha256: str
    train_data_sha256: str
    dev_data_sha256: str
    data_audit_sha256: str
    dev_token_audit_sha256: str
    forced_choice_token_audit_sha256: str
    crd_bank_audit_sha256: str
    crd_token_audit_sha256: str
    source_evaluation_precision: str

    def __post_init__(self) -> None:
        hashes = {
            "code_sha256": self.code_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_summary_sha256": self.source_summary_sha256,
            "source_candidate_token_audit_sha256": (self.source_candidate_token_audit_sha256),
            "source_raw_results_sha256": self.source_raw_results_sha256,
            "spec_sha256": self.spec_sha256,
            "train_data_sha256": self.train_data_sha256,
            "dev_data_sha256": self.dev_data_sha256,
            "data_audit_sha256": self.data_audit_sha256,
            "dev_token_audit_sha256": self.dev_token_audit_sha256,
            "forced_choice_token_audit_sha256": (self.forced_choice_token_audit_sha256),
            "crd_bank_audit_sha256": self.crd_bank_audit_sha256,
            "crd_token_audit_sha256": self.crd_token_audit_sha256,
        }
        for name, value in hashes.items():
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{name} must be a SHA-256 digest")
        if not self.source_run_id or not self.source_evaluation_precision:
            raise ValueError("resolved pilot source identity is incomplete")

    @property
    def preregistration_sha256(self) -> str:
        from plasticity_placement.p0d2hrr.io import json_hash

        return json_hash(self.identity_dict(include_preregistration_hash=False))

    def identity_dict(
        self,
        *,
        include_preregistration_hash: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "schema_version": PILOT_CONFIG_SCHEMA_VERSION,
            "stage": PILOT_STAGE,
            "code_sha256": self.code_sha256,
            "source_run_id": self.source_run_id,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_summary_sha256": self.source_summary_sha256,
            "source_candidate_token_audit_sha256": (self.source_candidate_token_audit_sha256),
            "source_raw_results_sha256": self.source_raw_results_sha256,
            "spec": self.spec.to_dict(),
            "spec_sha256": self.spec_sha256,
            "train_data_sha256": self.train_data_sha256,
            "dev_data_sha256": self.dev_data_sha256,
            "data_audit_sha256": self.data_audit_sha256,
            "dev_token_audit_sha256": self.dev_token_audit_sha256,
            "forced_choice_token_audit_sha256": (self.forced_choice_token_audit_sha256),
            "crd_bank_audit_sha256": self.crd_bank_audit_sha256,
            "crd_token_audit_sha256": self.crd_token_audit_sha256,
            "source_evaluation_precision": self.source_evaluation_precision,
            "authorization_scope": AUTHORIZATION_SCOPE,
            "training_run_limit": 1,
            "locked_external_evaluation_limit": 1,
            "training_complexity_review_eligible": False,
            "automatic_training_started": False,
            "automatic_narrow_scan_started": False,
            "automatic_rlvr_started": False,
        }
        if include_preregistration_hash:
            payload["preregistration_sha256"] = self.preregistration_sha256
        return payload
