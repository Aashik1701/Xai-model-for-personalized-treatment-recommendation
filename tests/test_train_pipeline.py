"""Integration-style tests for `scripts/train_ensembles.py`."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.train_ensembles as train_ensembles  # type: ignore  # noqa: E402


class DummyMLflow:
    """Lightweight MLflow stub capturing params and metrics."""

    def __init__(self) -> None:
        self.params: dict[str, object] = {}
        self.metrics: dict[str, float] = {}
        self.uri: str | None = None
        self._run_id = "run-123"

    def set_tracking_uri(self, uri: str) -> None:
        self.uri = uri

    class _RunContext:
        def __init__(self, outer: "DummyMLflow", run_name: str | None) -> None:
            self.outer = outer
            self.run_name = run_name

        def __enter__(self) -> "DummyMLflow._RunContext":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            return False

    def start_run(self, run_name: str | None = None) -> "DummyMLflow._RunContext":
        return DummyMLflow._RunContext(self, run_name)

    def log_param(self, key: str, value: object) -> None:
        self.params[key] = value

    def log_metric(self, key: str, value: float) -> None:
        self.metrics[key] = float(value)

    def active_run(self) -> SimpleNamespace:
        return SimpleNamespace(info=SimpleNamespace(run_id=self._run_id))


def _write_dataset(tmp_path: Path, minority_ratio: float = 0.2) -> Path:
    rng = np.random.default_rng(42)
    total = 200
    minority = int(total * minority_ratio)
    majority = total - minority
    features_majority = rng.normal(0, 1, size=(majority, 3))
    features_minority = rng.normal(2, 1.2, size=(minority, 3))
    X = np.vstack([features_majority, features_minority])
    y = np.array([0] * majority + [1] * minority)
    df = pd.DataFrame(X, columns=["feat1", "feat2", "feat3"])
    df["target"] = y
    csv_path = tmp_path / "toy_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _write_config(tmp_path: Path, csv_path: Path, use_smote: bool) -> Path:
    config = {
        "default_dataset": "toy",
        "datasets": {
            "toy": {
                "raw_path": str(csv_path),
                "processed_path": str(tmp_path / "toy_processed.csv"),
                "target_column": "target",
                "data_type": "tabular",
                "overrides": {"use_smote": use_smote},
            }
        },
        "preprocessing": {
            "tabular": {
                "numeric_imputation": "median",
                "categorical_imputation": "most_frequent",
                "class_imbalance": {
                    "enable_auto_weight": True,
                    "imbalance_ratio_threshold": 3.0,
                    "save_processed_default": False,
                },
            }
        },
        "split_config": {
            "test_split": 0.2,
            "validation_split": 0.0,
            "random_state": 42,
            "stratify": True,
        },
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return cfg_path


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_training_pipeline_mlflow_logging(tmp_path, monkeypatch):
    csv_path = _write_dataset(tmp_path)
    cfg_path = _write_config(tmp_path, csv_path, use_smote=True)

    dummy_mlflow = DummyMLflow()
    monkeypatch.setattr(train_ensembles, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(train_ensembles, "mlflow", dummy_mlflow)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ensembles.py",
            "--dataset",
            "toy",
            "--mlflow",
        ],
    )

    train_ensembles.main()

    assert dummy_mlflow.params.get("smote_requested") is True

    assert dummy_mlflow.params.get("smote_applied") is True
    # Check per-class metrics logged
    assert any(k.startswith("voting_precision_0") for k in dummy_mlflow.metrics)
    assert "voting_recall_1" in dummy_mlflow.metrics
    assert "voting_roc_auc" in dummy_mlflow.metrics


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_smote_cli_changes_distribution(tmp_path, monkeypatch, caplog):
    csv_path = _write_dataset(tmp_path)
    cfg_path = _write_config(tmp_path, csv_path, use_smote=False)

    dummy_mlflow = DummyMLflow()
    monkeypatch.setattr(train_ensembles, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(train_ensembles, "mlflow", dummy_mlflow)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_ensembles.py",
            "--dataset",
            "toy",
            "--smote",
        ],
    )
    caplog.set_level("INFO", logger="train_ensembles")

    train_ensembles.main()

    smote_logs = [
        rec.getMessage()
        for rec in caplog.records
        if "SMOTE attempt" in rec.getMessage()
    ]
    assert smote_logs, "Expected SMOTE log entry"
    before_segment = smote_logs[0].split("before=")[1].split(" after=")[0]
    after_segment = smote_logs[0].split("after=")[1]
    assert before_segment != after_segment
    # No MLflow logging requested, so parameters need not be set
