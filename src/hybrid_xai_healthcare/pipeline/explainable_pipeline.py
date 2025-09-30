"""Explainable ensemble training and inference pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from hybrid_xai_healthcare.data.preprocessor import HealthcareDataPreprocessor
from hybrid_xai_healthcare.models.ensemble import (
    StackingEnsemble,
    VotingEnsemble,
)
from hybrid_xai_healthcare.utils.imbalance import (
    apply_smote,
    decide_class_weight,
)

try:  # pragma: no cover - optional dependency
    import shap  # type: ignore
except Exception:  # noqa: BLE001
    shap = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class PipelineArtifacts:
    """Container persisted by :meth:`ExplainableEnsemblePipeline.save`."""

    preprocessor: HealthcareDataPreprocessor
    model: VotingEnsemble | StackingEnsemble
    model_name: str
    class_labels: List[Any]
    feature_names: List[str]
    shap_background: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExplainableEnsemblePipeline:
    """Bundles preprocessing, ensemble training, and explainability."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        background_sample: int = 20,
        top_k: int = 5,
    ) -> None:
        self.config = config or {}
        self.random_state = random_state
        self.background_sample = max(1, background_sample)
        self.top_k = max(1, top_k)

        tabular_cfg = (
            self.config.get("preprocessing", {}).get("tabular", {})
            if isinstance(self.config, dict)
            else {}
        )
        self.preprocessor = HealthcareDataPreprocessor(tabular_cfg)

        self.voting = self._build_voting_model()
        self.stacking = self._build_stacking_model()
        self.best_model: VotingEnsemble | StackingEnsemble | None = None
        self.best_model_name: str = ""
        self.class_labels: List[Any] = []
        self.feature_names: List[str] = []
        self.background_data: Optional[np.ndarray] = None
        self._explainer: Any = None
        self.training_metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Model builders and configuration helpers
    # ------------------------------------------------------------------
    def _build_voting_model(self) -> VotingEnsemble:
        estimators = {
            "rf": RandomForestClassifier(
                n_estimators=120,
                random_state=self.random_state,
            ),
            "gb": GradientBoostingClassifier(random_state=self.random_state),
            "lr": LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=self.random_state,
            ),
        }
        return VotingEnsemble(estimators=estimators, voting="soft")

    def _build_stacking_model(self) -> StackingEnsemble:
        base_estimators = {
            "rf": RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
            ),
            "gb": GradientBoostingClassifier(random_state=self.random_state),
        }
        meta = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=self.random_state,
        )
        return StackingEnsemble(
            base_estimators=base_estimators,
            meta_estimator=meta,
            n_folds=3,
            random_state=self.random_state,
        )

    def _set_class_weight(
        self,
        class_weight: Optional[str | Dict[str, float]],
    ) -> None:
        for estimator in self.voting.estimators.values():
            if hasattr(estimator, "class_weight"):
                try:
                    estimator.set_params(class_weight=class_weight)
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "Estimator %s does not support class_weight",
                        estimator,
                    )
        for estimator in self.stacking.base_estimators.values():
            if hasattr(estimator, "class_weight"):
                try:
                    estimator.set_params(class_weight=class_weight)
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "Estimator %s does not support class_weight",
                        estimator,
                    )
        if hasattr(self.stacking.meta_estimator, "class_weight"):
            try:
                self.stacking.meta_estimator.set_params(
                    class_weight=class_weight,
                )
            except Exception:  # noqa: BLE001
                logger.debug("Meta estimator does not support class_weight")

    def _apply_smote(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series, bool, str]:
        X_res, y_res, applied, message = apply_smote(X_train, y_train)
        if applied:
            if not isinstance(X_res, pd.DataFrame):
                X_res = pd.DataFrame(X_res, columns=X_train.columns)
            else:
                X_res = X_res.reset_index(drop=True)
            if not isinstance(y_res, pd.Series):
                y_res = pd.Series(y_res, name=y_train.name or "target")
            else:
                y_res = y_res.reset_index(drop=True)
        return X_res, y_res, applied, message

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_from_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        stratify: bool = True,
        class_weight: Optional[str | Dict[str, float]] = "auto",
        use_smote: bool = False,
    ) -> Dict[str, Any]:
        """High-level helper to split data, train, and compute metrics."""

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")

        X = df.drop(columns=[target_column])
        y = df[target_column]
        stratify_y = y if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_y,
        )

        cw_value: Optional[str | Dict[str, float]] = None
        imbalance_ratio = 1.0
        if class_weight == "auto":
            imbalance_cfg = (
                self.config.get("preprocessing", {})
                .get("tabular", {})
                .get("class_imbalance", {})
                if isinstance(self.config, dict)
                else {}
            )
            ratio_threshold = float(imbalance_cfg.get("imbalance_ratio_threshold", 3.0))
            cw_value, imbalance_ratio = decide_class_weight(
                y_train,
                ratio_threshold,
            )
        elif class_weight is not None:
            cw_value = class_weight

        self._set_class_weight(cw_value)

        smote_applied = False
        smote_message = "disabled"
        if use_smote:
            X_train, y_train, smote_applied, smote_message = self._apply_smote(
                X_train,
                y_train,
            )

        self.fit(X_train, y_train, X_val, y_val)

        validation_metrics = self._evaluate_model(
            self.best_model_name,
            self.best_model,
            self.preprocessor.transform(X_val),
            y_val,
        )

        self.training_metrics = {
            "validation": validation_metrics,
            "class_weight": cw_value or "none",
            "imbalance_ratio": float(imbalance_ratio),
            "smote_applied": bool(smote_applied),
            "smote_message": smote_message,
        }
        return self.training_metrics

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> "ExplainableEnsemblePipeline":
        """Fit preprocessing and ensemble models."""

        logger.info(
            "Fitting explainable ensemble pipeline on %s samples",
            len(X_train),
        )
        X_train_proc = self.preprocessor.fit_transform(X_train)
        self.feature_names = list(self.preprocessor.get_feature_names_out())

        y_array = (
            y_train.to_numpy()
            if isinstance(y_train, pd.Series)
            else np.asarray(y_train)
        )
        self.voting.set_feature_names(self.feature_names)
        self.voting.fit(X_train_proc, y_array)
        self.stacking.set_feature_names(self.feature_names)
        self.stacking.fit(X_train_proc, y_array)

        if X_val is not None and y_val is not None:
            X_val_proc = self.preprocessor.transform(X_val)
            metrics_v = self._evaluate_model(
                "voting",
                self.voting,
                X_val_proc,
                y_val,
            )
            metrics_s = self._evaluate_model(
                "stacking",
                self.stacking,
                X_val_proc,
                y_val,
            )
        else:
            metrics_v = self._evaluate_model(
                "voting",
                self.voting,
                X_train_proc,
                y_train,
            )
            metrics_s = self._evaluate_model(
                "stacking",
                self.stacking,
                X_train_proc,
                y_train,
            )

        self.best_model, self.best_model_name = self._choose_model(
            metrics_v,
            metrics_s,
        )
        self.class_labels = list(getattr(self.best_model, "classes_", []))

        background = np.asarray(X_train_proc)
        if background.shape[0] > self.background_sample:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(
                background.shape[0],
                size=self.background_sample,
                replace=False,
            )
            background = background[idx]
        self.background_data = background
        self._init_explainer()
        return self

    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        if self.best_model is None:
            raise RuntimeError("Pipeline not fitted")

        X_proc = self.preprocessor.transform(X)
        preds = self.best_model.predict(X_proc)
        proba = None
        try:
            proba = self.best_model.predict_proba(X_proc)
        except Exception:  # noqa: BLE001
            proba = None

        preds_list = preds.tolist() if isinstance(preds, np.ndarray) else list(preds)
        result: Dict[str, Any] = {
            "predictions": preds_list,
            "classes": list(self.class_labels),
            "probabilities": (
                proba.tolist() if isinstance(proba, np.ndarray) else None
            ),
        }

        explanations = self._generate_explanations(X_proc, preds_list, proba)
        if explanations:
            result["explanations"] = explanations
        return result

    def save(self, output_dir: Path) -> Path:
        if self.best_model is None or self.background_data is None:
            raise RuntimeError("Pipeline not trained; cannot save")

        output_dir.mkdir(parents=True, exist_ok=True)
        artifact = PipelineArtifacts(
            preprocessor=self.preprocessor,
            model=self.best_model,
            model_name=self.best_model_name,
            class_labels=self.class_labels,
            feature_names=self.feature_names,
            shap_background=self.background_data,
            metadata={
                "config": self.config,
                "metrics": self.training_metrics,
            },
        )
        artifact_path = output_dir / "explainable_ensemble.joblib"
        joblib.dump(artifact, artifact_path)
        logger.info("Pipeline artifact saved to %s", artifact_path)
        return artifact_path

    @classmethod
    def load(cls, artifact_path: Path) -> "ExplainableEnsemblePipeline":
        payload: PipelineArtifacts = joblib.load(artifact_path)
        pipeline = cls(config=payload.metadata.get("config", {}))
        pipeline.preprocessor = payload.preprocessor
        pipeline.best_model = payload.model
        pipeline.best_model_name = payload.model_name
        pipeline.class_labels = payload.class_labels
        pipeline.feature_names = payload.feature_names
        pipeline.background_data = payload.shap_background
        pipeline.training_metrics = payload.metadata.get("metrics", {})
        pipeline._explainer = None
        pipeline._init_explainer()
        return pipeline

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _evaluate_model(
        self,
        name: str,
        model: VotingEnsemble | StackingEnsemble | None,
        X: np.ndarray,
        y: Any,
    ) -> Dict[str, float]:
        if model is None:
            return {"accuracy": 0.0, "f1": 0.0, "roc_auc": 0.0}
        preds = model.predict(X)
        proba = None
        try:
            proba = model.predict_proba(X)
        except Exception:  # noqa: BLE001
            proba = None
        if isinstance(y, pd.Series):
            y_array = y.to_numpy()
        else:
            y_array = np.asarray(y)
        if y_array.ndim > 1:
            y_array = y_array.ravel()
        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y, preds)),
            "f1": float(f1_score(y, preds, average="weighted", zero_division=0)),
            "roc_auc": 0.0,
        }
        if proba is not None and proba.shape[1] == 2:
            try:
                le = LabelEncoder()
                y_binary = le.fit_transform(y_array)
                classes = list(getattr(model, "classes_", le.classes_))
                positive_label = (
                    le.classes_[1] if len(le.classes_) > 1 else le.classes_[0]
                )
                if classes and positive_label in classes:
                    positive_idx = classes.index(positive_label)
                else:
                    positive_idx = 1 if proba.shape[1] > 1 else 0
                metrics["roc_auc"] = float(
                    roc_auc_score(y_binary, proba[:, positive_idx])
                )
            except Exception:  # noqa: BLE001
                metrics["roc_auc"] = 0.0
        logger.info("%s metrics: %s", name, metrics)
        return metrics

    def _choose_model(
        self,
        metrics_v: Dict[str, float],
        metrics_s: Dict[str, float],
    ) -> Tuple[VotingEnsemble | StackingEnsemble, str]:
        score_v = metrics_v.get("roc_auc", metrics_v.get("f1", 0.0))
        score_s = metrics_s.get("roc_auc", metrics_s.get("f1", 0.0))
        if score_s > score_v:
            logger.info(
                "Stacking ensemble selected (score %.4f > %.4f)",
                score_s,
                score_v,
            )
            return self.stacking, "stacking"
        logger.info(
            "Voting ensemble selected (score %.4f >= %.4f)",
            score_v,
            score_s,
        )
        return self.voting, "voting"

    def _init_explainer(self) -> None:
        if shap is None:
            logger.warning("SHAP not installed; explanations will be unavailable")
            self._explainer = None
            return
        if self.best_model is None or self.background_data is None:
            logger.warning("Cannot initialize explainer before training completes")
            self._explainer = None
            return
        try:
            self._explainer = shap.Explainer(
                self.best_model.predict_proba,
                self.background_data,
                algorithm="auto",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to create SHAP explainer: %s", exc)
            self._explainer = None

    def _generate_explanations(
        self,
        X_proc: np.ndarray,
        preds: List[Any],
        proba: Optional[np.ndarray],
    ) -> List[Dict[str, Any]]:
        if shap is None or self._explainer is None:
            return []
        try:
            explanation = self._explainer(X_proc)
            values = explanation.values
            base_values = explanation.base_values
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unable to compute SHAP explanations: %s", exc)
            return []

        results: List[Dict[str, Any]] = []
        num_classes = len(self.class_labels) if self.class_labels else 1
        for idx, label in enumerate(preds):
            class_idx = 0
            if self.class_labels:
                try:
                    class_idx = self.class_labels.index(label)
                except ValueError:
                    class_idx = 0
                try:
                    contribs = self._slice_values(
                        values,
                        idx,
                        class_idx,
                        num_classes,
                    )
                except ValueError as exc:  # noqa: BLE001
                    logger.warning("Unexpected SHAP values shape: %s", exc)
                    contribs = np.zeros(len(self.feature_names), dtype=float)
            top_features = self._format_top_features(contribs)
            base_val = self._slice_base_value(
                base_values,
                idx,
                class_idx,
                num_classes,
            )
            entry: Dict[str, Any] = {
                "predicted_class": label,
                "top_features": top_features,
            }
            if base_val is not None:
                entry["base_value"] = base_val
            if proba is not None:
                entry["probabilities"] = {
                    cls: float(proba[idx, col])
                    for col, cls in enumerate(self.class_labels)
                }
            results.append(entry)
        return results

    def _slice_values(
        self,
        values: Any,
        sample_idx: int,
        class_idx: int,
        num_classes: int,
    ) -> np.ndarray:
        arr = values
        if isinstance(arr, list):
            arr = arr[class_idx if class_idx < len(arr) else 0]
        arr = np.asarray(arr)
        if arr.ndim == 3:  # (samples, classes, features)
            if arr.shape[1] == num_classes:
                return np.asarray(arr[sample_idx, class_idx, :], dtype=float)
            if arr.shape[0] == num_classes:  # (classes, samples, features)
                return np.asarray(arr[class_idx, sample_idx, :], dtype=float)
        if arr.ndim == 2:
            return np.asarray(arr[sample_idx], dtype=float)
        if arr.ndim == 1:
            return np.asarray(arr, dtype=float)
        raise ValueError("Unexpected SHAP values shape")

    def _slice_base_value(
        self,
        base_values: Any,
        sample_idx: int,
        class_idx: int,
        num_classes: int,
    ) -> Optional[float]:
        arr = np.asarray(base_values)
        if arr.ndim == 2:
            if arr.shape[1] == num_classes:
                return float(arr[sample_idx, class_idx])
            if arr.shape[0] == num_classes:
                return float(arr[class_idx, sample_idx])
        if arr.ndim == 1:
            return float(arr[sample_idx])
        if np.isscalar(arr):
            return float(arr)
        return None

    def _format_top_features(
        self,
        contribs: np.ndarray,
    ) -> List[Dict[str, float]]:
        contribs = np.asarray(contribs)
        order = np.argsort(np.abs(contribs))[::-1][: self.top_k]
        formatted: List[Dict[str, float]] = []
        for idx in order:
            feature_name = (
                str(self.feature_names[idx])
                if idx < len(self.feature_names)
                else f"feature_{idx}"
            )
            formatted.append(
                {
                    "feature": feature_name,
                    "contribution": float(contribs[idx]),
                }
            )
        return formatted
