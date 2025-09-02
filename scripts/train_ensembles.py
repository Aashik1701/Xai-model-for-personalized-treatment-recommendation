"""Train and evaluate ensemble models on synthetic healthcare data."""
import sys
from pathlib import Path
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from hybrid_xai_healthcare.data.data_loader import DataLoader
from hybrid_xai_healthcare.models.ensemble import VotingEnsemble, StackingEnsemble

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_ensembles")

def load_dataset():
    dl = DataLoader()
    df = dl.load_data('data/raw/synthetic_healthcare_data.csv')
    split = dl.split_data(target_column='treatment_outcome', test_size=0.2, validation_size=0.0)
    return split

def build_voting():
    estimators = {
        'rf': RandomForestClassifier(n_estimators=50, random_state=42),
        'gb': GradientBoostingClassifier(random_state=42)
    }
    return VotingEnsemble(estimators=estimators, voting='soft')

def build_stacking():
    base_estimators = {
        'rf': RandomForestClassifier(n_estimators=40, random_state=42),
        'gb': GradientBoostingClassifier(random_state=42)
    }
    meta = LogisticRegression(max_iter=500, random_state=42)
    return StackingEnsemble(base_estimators=base_estimators, meta_estimator=meta, n_folds=3)

def main():
    split = load_dataset()
    X_train, X_test = split['X_train'], split['X_test']
    y_train, y_test = split['y_train'], split['y_test']

    # Basic preprocessing: one-hot encode categorical columns
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        logger.info(f"One-hot encoding categorical columns: {cat_cols}")
        full = pd.concat([X_train, X_test], axis=0)
        full_enc = pd.get_dummies(full, columns=cat_cols, drop_first=True)
        X_train = full_enc.iloc[:len(X_train)].reset_index(drop=True)
        X_test = full_enc.iloc[len(X_train):].reset_index(drop=True)

    logger.info('Training VotingEnsemble...')
    voting = build_voting()
    voting.set_feature_names(list(X_train.columns))
    voting.fit(X_train, y_train)
    preds_v = voting.predict(X_test)
    logger.info('\nVotingEnsemble Report:\n' + classification_report(y_test, preds_v))

    logger.info('Training StackingEnsemble...')
    stacking = build_stacking()
    stacking.set_feature_names(list(X_train.columns))
    stacking.fit(X_train, y_train)
    preds_s = stacking.predict(X_test)
    logger.info('\nStackingEnsemble Report:\n' + classification_report(y_test, preds_s))

    # Feature importance (if available)
    vi = voting.get_feature_importance()
    if vi:
        top5 = list(vi.items())[:5]
        logger.info(f"Voting top5 importance: {top5}")
    si = stacking.get_feature_importance()
    if si:
        top5s = list(si.items())[:5]
        logger.info(f"Stacking top5 importance: {top5s}")

if __name__ == '__main__':
    main()
