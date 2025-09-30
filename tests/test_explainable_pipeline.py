"""Tests for the explainable ensemble pipeline."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import]

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_xai_healthcare.pipeline.explainable_pipeline import (  # noqa: E402
    ExplainableEnsemblePipeline,
)  # type: ignore[import]


def _make_sample_dataframe(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {
        "age": rng.integers(20, 80, size=rows),
        "systolic_bp": rng.normal(120, 15, size=rows),
        "diastolic_bp": rng.normal(80, 10, size=rows),
        "bmi": rng.normal(27, 4, size=rows),
        "gender": rng.choice(["M", "F"], size=rows),
    }
    df = pd.DataFrame(data)
    risk = (
        (df["age"] > 50).astype(int)
        + (df["systolic_bp"] > 130).astype(int)
        + (df["bmi"] > 30).astype(int)
    )
    df["risk_level"] = np.where(risk > 1, "high", "low")
    return df


def test_pipeline_fit_predict_and_persistence(tmp_path: Path) -> None:
    df = _make_sample_dataframe()
    pipeline = ExplainableEnsemblePipeline(
        random_state=0,
        background_sample=8,
        top_k=3,
    )

    metrics = pipeline.fit_from_dataframe(df, target_column="risk_level")
    assert "validation" in metrics
    assert "class_weight" in metrics

    feature_frame = df.drop(columns=["risk_level"]).head(5)
    result = pipeline.predict(feature_frame)
    assert len(result["predictions"]) == len(feature_frame)
    assert result["classes"], "Classes should be recorded after fitting"

    if "explanations" in result:
        assert len(result["explanations"]) == len(feature_frame)

    artifact_path = pipeline.save(tmp_path)
    assert artifact_path.exists()

    loaded = ExplainableEnsemblePipeline.load(artifact_path)
    loaded_result = loaded.predict(feature_frame)
    assert len(loaded_result["predictions"]) == len(feature_frame)
