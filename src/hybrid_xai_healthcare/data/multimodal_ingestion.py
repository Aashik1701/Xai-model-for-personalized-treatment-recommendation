"""Dataset-specific ingestion helpers for multimodal healthcare corpora."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

from .data_loader import DataLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for dataset splitting."""

    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    stratify: bool = True

    @classmethod
    def from_mapping(
        cls, config: Optional[Dict[str, Union[int, float, bool]]]
    ) -> "SplitConfig":
        if config is None:
            return cls()
        return cls(
            test_size=float(config.get("test_size", cls.test_size)),
            validation_size=float(config.get("validation_size", cls.validation_size)),
            random_state=int(config.get("random_state", cls.random_state)),
            stratify=bool(config.get("stratify", cls.stratify)),
        )


@dataclass(frozen=True)
class DatasetIngestionSpec:
    """Static contract for a processed dataset."""

    key: str
    processed_path: str
    target_column: Optional[str]
    modality: str
    required_columns: Sequence[str]
    id_column: Optional[str] = None


MULTIMODAL_DATASETS: Dict[str, DatasetIngestionSpec] = {
    "synthea_longitudinal": DatasetIngestionSpec(
        key="synthea_longitudinal",
        processed_path="data/processed/synthea_longitudinal_processed.csv",
        target_column="target",
        modality="temporal",
        required_columns=(
            "patient_id",
            "target",
            "n_encounters",
            "n_conditions",
            "n_medications",
            "longitudinal_complexity_score",
            "dominant_condition_domain",
            "temporal_event_summary",
        ),
        id_column="patient_id",
    ),
    "skin_cancer_imaging": DatasetIngestionSpec(
        key="skin_cancer_imaging",
        processed_path="data/processed/skin_cancer_imaging_processed.csv",
        target_column="target",
        modality="metadata",
        required_columns=(
            "image_path",
            "lesion_label",
            "lesion_icd10_code",
            "lesion_risk_category",
            "is_malignant",
            "target",
        ),
        id_column="image_path",
    ),
    "medical_question_pairs": DatasetIngestionSpec(
        key="medical_question_pairs",
        processed_path="data/processed/medical_question_pairs_processed.csv",
        target_column="is_duplicate",
        modality="text",
        required_columns=(
            "question_primary",
            "question_secondary",
            "primary_concept_domain",
            "secondary_concept_domain",
            "concept_alignment",
            "has_ontology_match",
            "shared_concept_domain",
            "concept_pair_summary",
        ),
        id_column="pair_group_id",
    ),
    "drug_reviews": DatasetIngestionSpec(
        key="drug_reviews",
        processed_path="data/processed/drug_reviews_processed.csv",
        target_column="rating",
        modality="text",
        required_columns=(
            "review_id",
            "drug_name",
            "condition",
            "effectiveness_score",
            "side_effect_severity",
            "drug_rxnorm_code",
            "condition_concept_domain",
            "rating",
        ),
        id_column="review_id",
    ),
}


def available_multimodal_datasets() -> Sequence[str]:
    """Return dataset keys that are supported by the ingestion helpers."""

    return tuple(MULTIMODAL_DATASETS.keys())


def resolve_dataset_spec(dataset_key: str) -> DatasetIngestionSpec:
    """Return ingestion spec for a dataset key."""

    try:
        return MULTIMODAL_DATASETS[dataset_key]
    except KeyError as exc:
        raise KeyError(f"Unknown multimodal dataset '{dataset_key}'") from exc


def _resolve_base_dir(base_dir: Optional[Union[str, Path]]) -> Path:
    if base_dir is None:
        return Path.cwd()
    return Path(base_dir)


def load_multimodal_dataset(
    dataset_key: str,
    *,
    base_dir: Optional[Union[str, Path]] = None,
    split_config: Optional[
        Union[SplitConfig, Dict[str, Union[int, float, bool]]]
    ] = None,
    validate: bool = True,
    additional_required: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Load a processed multimodal dataset and optionally generate splits."""

    spec = resolve_dataset_spec(dataset_key)
    root = _resolve_base_dir(base_dir)
    file_path = (root / spec.processed_path).resolve()

    loader = DataLoader(config={"dataset": dataset_key, "modality": spec.modality})
    data = loader.load_data(file_path)

    validation_report = None
    if validate:
        required_columns: Sequence[str]
        if additional_required:
            required_columns = tuple(
                dict.fromkeys((*spec.required_columns, *additional_required))
            )
        else:
            required_columns = spec.required_columns
        validation_report = loader.validate_data(
            required_columns=list(required_columns) if required_columns else None,
            target_column=spec.target_column,
        )

    split_cfg = split_config
    if isinstance(split_cfg, dict):
        split_cfg = SplitConfig.from_mapping(split_cfg)
    elif split_cfg is None:
        split_cfg = SplitConfig()

    splits = None
    if spec.target_column and split_cfg.test_size > 0:
        splits = loader.split_data(
            target_column=spec.target_column,
            test_size=split_cfg.test_size,
            validation_size=split_cfg.validation_size,
            random_state=split_cfg.random_state,
            stratify=split_cfg.stratify,
        )

    logger.info(
        "Loaded multimodal dataset '%s' from %s (rows=%s, cols=%s)",
        dataset_key,
        file_path,
        data.shape[0],
        data.shape[1],
    )

    return {
        "data": data,
        "metadata": loader.metadata,
        "validation": validation_report,
        "splits": splits,
        "spec": spec,
    }
