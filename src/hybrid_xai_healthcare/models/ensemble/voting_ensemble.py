"""Voting ensemble implementation for hybrid healthcare models."""
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from ..base_model import BaseHybridModel
from sklearn.base import ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

class VotingEnsemble(BaseHybridModel):
    """
    Voting ensemble model combining multiple base learners.
    Supports hard and soft voting strategies.
    """
    
    def __init__(self,
                 estimators: Dict[str, ClassifierMixin],
                 voting: str = "soft",
                 weights: Optional[List[float]] = None,
                 config: Optional[Dict[str, Any]] = None,
                 random_state: int = 42):
        super().__init__(model_name="voting_ensemble", config=config or {}, random_state=random_state)
        if voting not in ("soft", "hard"):
            raise ValueError("voting must be 'soft' or 'hard'")
        self.voting = voting
        self.estimators = {name: clone(model) for name, model in estimators.items()}
        self.weights = weights
        self.le = LabelEncoder()
        self.classes_ = None
        self.fitted_estimators_ = {}
    
    def fit(self, X, y, validation_data=None, **kwargs):
        X_arr = self.validate_input(X)
        y_arr = np.array(y)
        self.le.fit(y_arr)
        self.classes_ = self.le.classes_
        y_encoded = self.le.transform(y_arr)
        logger.info(f"Fitting VotingEnsemble with {len(self.estimators)} estimators")
        for name, est in self.estimators.items():
            logger.debug(f"Training estimator: {name}")
            est.fit(X_arr, y_encoded)
            self.fitted_estimators_[name] = est
        self.is_fitted = True
        return self
    
    def _predict_proba_each(self, X):
        X_arr = self.validate_input(X)
        probas = []
        for name, est in self.fitted_estimators_.items():
            if hasattr(est, "predict_proba"):
                p = est.predict_proba(X_arr)
            else:
                # Fallback: derive probabilities from decision_function or predictions
                if hasattr(est, "decision_function"):
                    scores = est.decision_function(X_arr)
                    # convert to probabilities using softmax for multiclass
                    if scores.ndim == 1:
                        scores = np.vstack([-scores, scores]).T
                    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
                    p = exp_scores / exp_scores.sum(axis=1, keepdims=True)
                else:
                    preds = est.predict(X_arr)
                    p = np.zeros((len(preds), len(self.classes_)))
                    for i, c in enumerate(preds):
                        class_idx = np.where(self.classes_ == self.le.inverse_transform([c])[0])[0]
                        if len(class_idx):
                            p[i, class_idx[0]] = 1.0
            # Ensure column alignment
            if p.shape[1] != len(self.classes_):
                # attempt to align if missing
                aligned = np.zeros((p.shape[0], len(self.classes_)))
                if getattr(est, 'classes_', None) is not None:
                    for j, cls in enumerate(est.classes_):
                        cls_idx = np.where(self.classes_ == cls)[0]
                        if len(cls_idx):
                            aligned[:, cls_idx[0]] = p[:, j]
                    p = aligned
            probas.append(p)
        return probas
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        probas = self._predict_proba_each(X)
        weights = self.weights if self.weights is not None else [1.0]*len(probas)
        weights = np.array(weights) / np.sum(weights)
        stacked = np.stack(probas, axis=0)  # (n_estimators, n_samples, n_classes)
        weighted = (stacked * weights[:, None, None]).sum(axis=0)
        return weighted
    
    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X_arr = self.validate_input(X)
        if self.voting == "soft":
            probs = self.predict_proba(X_arr)
            preds_idx = probs.argmax(axis=1)
        else:  # hard voting
            votes = []
            for name, est in self.fitted_estimators_.items():
                preds = est.predict(X_arr)
                votes.append(preds)
            votes = np.vstack(votes)  # (n_estimators, n_samples)
            # majority vote
            preds_idx = []
            for col in votes.T:
                vals, counts = np.unique(col, return_counts=True)
                preds_idx.append(vals[counts.argmax()])
            preds_idx = np.array(preds_idx)
        return self.le.inverse_transform(preds_idx)
    
    def get_feature_importance(self) -> Dict[str, float]:
        # Aggregate feature importances if available
        importances = {}
        counts = {}
        for name, est in self.fitted_estimators_.items():
            est_importance = None
            if hasattr(est, 'feature_importances_'):
                est_importance = est.feature_importances_
            elif hasattr(est, 'coef_'):
                coef = est.coef_
                if coef.ndim > 1:
                    est_importance = np.mean(np.abs(coef), axis=0)
                else:
                    est_importance = np.abs(coef)
            if est_importance is not None and self.feature_names is not None:
                for fname, val in zip(self.feature_names, est_importance):
                    importances[fname] = importances.get(fname, 0.0) + float(val)
                    counts[fname] = counts.get(fname, 0) + 1
        if not importances:
            return {}
        # Average
        for k in importances:
            importances[k] /= counts[k]
        # Normalize
        total = sum(importances.values())
        if total > 0:
            for k in importances:
                importances[k] /= total
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
