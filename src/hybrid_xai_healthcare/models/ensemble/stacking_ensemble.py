"""Stacking ensemble implementation for hybrid healthcare models."""
from typing import Dict, Any, Optional, List
import numpy as np
from ..base_model import BaseHybridModel
from sklearn.base import ClassifierMixin, clone
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

class StackingEnsemble(BaseHybridModel):
    """
    Stacking ensemble with out-of-fold predictions for meta-learner training.
    """
    def __init__(self,
                 base_estimators: Dict[str, ClassifierMixin],
                 meta_estimator: ClassifierMixin,
                 n_folds: int = 5,
                 use_proba: bool = True,
                 config: Optional[Dict[str, Any]] = None,
                 random_state: int = 42):
        super().__init__(model_name="stacking_ensemble", config=config or {}, random_state=random_state)
        self.base_estimators = {name: clone(model) for name, model in base_estimators.items()}
        self.meta_estimator = clone(meta_estimator)
        self.n_folds = n_folds
        self.use_proba = use_proba
        self.le = LabelEncoder()
        self.classes_ = None
        self.fitted_base_estimators_ = {}
    
    def fit(self, X, y, validation_data=None, **kwargs):
        X_arr = self.validate_input(X)
        y_arr = np.array(y)
        self.le.fit(y_arr)
        self.classes_ = self.le.classes_
        y_encoded = self.le.transform(y_arr)
        n_samples = X_arr.shape[0]
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        # Determine meta feature size
        example_est = next(iter(self.base_estimators.values()))
        # We'll discover number of classes from label encoder
        n_classes = len(self.classes_)
        meta_features = []
        oof_matrix = np.zeros((n_samples, len(self.base_estimators) * (n_classes if self.use_proba else 1)))
        logger.info(f"Training stacking ensemble with {len(self.base_estimators)} base estimators and {self.n_folds}-fold OOF")
        for fold, (train_idx, hold_idx) in enumerate(kf.split(X_arr, y_encoded)):
            logger.debug(f"Fold {fold+1}/{self.n_folds}")
            X_train, X_hold = X_arr[train_idx], X_arr[hold_idx]
            y_train = y_encoded[train_idx]
            col_offset = 0
            for est_idx, (name, est) in enumerate(self.base_estimators.items()):
                est_clone = clone(est)
                est_clone.fit(X_train, y_train)
                # store last fitted version for final refit later
                self.fitted_base_estimators_[name] = est_clone
                if self.use_proba and hasattr(est_clone, 'predict_proba'):
                    probs = est_clone.predict_proba(X_hold)
                    if probs.shape[1] != n_classes:
                        # align columns
                        aligned = np.zeros((probs.shape[0], n_classes))
                        if getattr(est_clone, 'classes_', None) is not None:
                            for j, cls in enumerate(est_clone.classes_):
                                cls_idx = np.where(self.classes_ == cls)[0]
                                if len(cls_idx):
                                    aligned[:, cls_idx[0]] = probs[:, j]
                        probs = aligned
                    start = est_idx * (n_classes if self.use_proba else 1)
                    oof_matrix[hold_idx, start:start + n_classes] = probs
                else:
                    preds = est_clone.predict(X_hold)
                    start = est_idx * (n_classes if self.use_proba else 1)
                    oof_matrix[hold_idx, start:start + 1] = preds.reshape(-1, 1)
        # Train meta-estimator
        self.meta_estimator.fit(oof_matrix, y_encoded)
        # Refit base estimators on full data
        for name, est in self.base_estimators.items():
            est_full = clone(est)
            est_full.fit(X_arr, y_encoded)
            self.fitted_base_estimators_[name] = est_full
        self.is_fitted = True
        return self
    
    def _build_meta_features(self, X):
        X_arr = self.validate_input(X)
        n_samples = X_arr.shape[0]
        n_classes = len(self.classes_)
        meta_mat = np.zeros((n_samples, len(self.base_estimators) * (n_classes if self.use_proba else 1)))
        for est_idx, (name, est) in enumerate(self.fitted_base_estimators_.items()):
            if self.use_proba and hasattr(est, 'predict_proba'):
                probs = est.predict_proba(X_arr)
                if probs.shape[1] != n_classes:
                    aligned = np.zeros((probs.shape[0], n_classes))
                    if getattr(est, 'classes_', None) is not None:
                        for j, cls in enumerate(est.classes_):
                            cls_idx = np.where(self.classes_ == cls)[0]
                            if len(cls_idx):
                                aligned[:, cls_idx[0]] = probs[:, j]
                    probs = aligned
                start = est_idx * (n_classes if self.use_proba else 1)
                meta_mat[:, start:start + n_classes] = probs
            else:
                preds = est.predict(X_arr)
                start = est_idx * (n_classes if self.use_proba else 1)
                meta_mat[:, start:start + 1] = preds.reshape(-1, 1)
        return meta_mat
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        meta_features = self._build_meta_features(X)
        if hasattr(self.meta_estimator, 'predict_proba'):
            probs = self.meta_estimator.predict_proba(meta_features)
            # align classes if needed
            if probs.shape[1] != len(self.classes_):
                aligned = np.zeros((probs.shape[0], len(self.classes_)))
                if getattr(self.meta_estimator, 'classes_', None) is not None:
                    for j, cls in enumerate(self.meta_estimator.classes_):
                        cls_idx = np.where(self.classes_ == cls)[0]
                        if len(cls_idx):
                            aligned[:, cls_idx[0]] = probs[:, j]
                probs = aligned
            return probs
        else:
            # fallback: generate pseudo probabilities
            preds = self.meta_estimator.predict(meta_features)
            probs = np.zeros((len(preds), len(self.classes_)))
            for i, p in enumerate(preds):
                cls_idx = p if isinstance(p, int) else np.where(self.classes_ == p)[0][0]
                probs[i, cls_idx] = 1.0
            return probs
    
    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        probs = self.predict_proba(X)
        preds_idx = probs.argmax(axis=1)
        return self.le.inverse_transform(preds_idx)
    
    def get_feature_importance(self) -> Dict[str, float]:
        # Approximate: average base learner importances (like Voting)
        importances = {}
        counts = {}
        for name, est in self.fitted_base_estimators_.items():
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
        for k in importances:
            importances[k] /= counts[k]
        total = sum(importances.values())
        if total > 0:
            for k in importances:
                importances[k] /= total
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
