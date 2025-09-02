"""Basic tests for ensemble models."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from hybrid_xai_healthcare.models.ensemble import VotingEnsemble, StackingEnsemble
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def test_voting_ensemble():
    X, y = make_classification(n_samples=120, n_features=10, n_informative=6, n_classes=3, random_state=42)
    estimators = {
        'rf': RandomForestClassifier(n_estimators=10, random_state=42),
        'gb': GradientBoostingClassifier(random_state=42)
    }
    model = VotingEnsemble(estimators=estimators, voting='soft')
    model.fit(X, y)
    proba = model.predict_proba(X[:5])
    assert proba.shape == (5, len(model.classes_))
    preds = model.predict(X[:5])
    assert len(preds) == 5

def test_stacking_ensemble():
    X, y = make_classification(n_samples=150, n_features=12, n_informative=7, n_classes=3, random_state=0)
    base_estimators = {
        'rf': RandomForestClassifier(n_estimators=15, random_state=0),
        'gb': GradientBoostingClassifier(random_state=0)
    }
    meta = LogisticRegression(max_iter=300, random_state=0)
    model = StackingEnsemble(base_estimators=base_estimators, meta_estimator=meta, n_folds=3)
    model.fit(X, y)
    proba = model.predict_proba(X[:6])
    assert proba.shape == (6, len(model.classes_))
    preds = model.predict(X[:6])
    assert len(preds) == 6
