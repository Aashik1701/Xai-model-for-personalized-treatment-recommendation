"""Train and evaluate ensemble models on a chosen dataset.

Adds CLI flag --dataset to pick from entries in config/data_config.yaml.
Supports simple per-dataset overrides:
    - target_transform: "collapse_2_to_4" (heart disease multi -> binary)
    - column_imputation: {col: strategy}
"""

import sys
from pathlib import Path
import logging
import argparse
from typing import Any, Dict, Tuple
import yaml  # type: ignore
import pandas as pd  # type: ignore
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from collections import Counter
import mlflow  # type: ignore
from mlflow_manager import MLflowManager
from hybrid_xai_healthcare.utils.imbalance import (
    decide_class_weight,
    apply_smote,
)

# Add src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_xai_healthcare.data.data_loader import DataLoader
from hybrid_xai_healthcare.models.ensemble import (
    VotingEnsemble,
    StackingEnsemble,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_ensembles")

CONFIG_PATH = Path("config/data_config.yaml")


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, "r") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def apply_overrides(
    df: pd.DataFrame,
    overrides: dict,  # type: ignore
    target_col: str,
) -> pd.DataFrame:
    if not overrides:
        return df
    # Target transform
    transform = None
    if isinstance(overrides, dict):
        transform = overrides.get("target_transform")
    if transform == "collapse_2_to_4" and target_col in df.columns:
        df[target_col] = df[target_col].apply(lambda v: 0 if v == 0 else 1)
    # Column imputations (simple) median / most_frequent
    col_impute = overrides.get("column_imputation", {})
    for col, strat in col_impute.items():
        if col in df.columns:
            if strat == "median":
                df[col] = df[col].fillna(df[col].median())
            elif strat in ("most_frequent", "mode"):
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    # Drop columns
    drops = overrides.get("drop_columns", [])
    for col in drops:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df


def load_dataset(dataset_key: str):  # type: ignore
    cfg = load_config()
    ds_cfg = cfg["datasets"][dataset_key]
    raw_path = ds_cfg["raw_path"]
    target_col = ds_cfg["target_column"]
    dl = DataLoader()
    df = dl.load_data(raw_path)
    df = apply_overrides(df, ds_cfg.get("overrides"), target_col)
    dl.data = df  # ensure modified df stored
    split = dl.split_data(
        target_column=target_col,
        test_size=0.2,
        validation_size=0.0,
    )
    return split, target_col


def basic_impute(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Simple median/mode imputation using training-set statistics.

    Uses preprocessing.tabular numeric_imputation & categorical_imputation.
    """
    prep = cfg.get("preprocessing", {}).get("tabular", {})
    num_strategy = prep.get("numeric_imputation", "median")
    cat_strategy = prep.get("categorical_imputation", "most_frequent")

    num_cols = X_train.select_dtypes(include=["number"]).columns
    cat_cols = X_train.select_dtypes(include=["object"]).columns

    # Numeric
    for c in num_cols:
        if num_strategy == "median":
            val = X_train[c].median()
        elif num_strategy == "mean":
            val = X_train[c].mean()
        else:
            val = X_train[c].median()
        X_train[c] = X_train[c].fillna(val)
        if c in X_test.columns:
            X_test[c] = X_test[c].fillna(val)

    # Categorical
    for c in cat_cols:
        if cat_strategy in ("most_frequent", "mode"):
            if X_train[c].mode().shape[0] > 0:
                val = X_train[c].mode().iloc[0]
            else:
                val = "missing"
        else:
            val = "missing"
        X_train[c] = X_train[c].fillna(val)
        if c in X_test.columns:
            X_test[c] = X_test[c].fillna(val)

    # Fallback: remaining NaNs -> 0 for numeric, 'missing' for categorical
    if X_train.isna().any().any():
        for c in X_train.columns:
            if X_train[c].isna().any():
                if c in num_cols:
                    X_train[c] = X_train[c].fillna(0)
                else:
                    X_train[c] = X_train[c].fillna("missing")
    if X_test.isna().any().any():
        for c in X_test.columns:
            if X_test[c].isna().any():
                if c in num_cols:
                    X_test[c] = X_test[c].fillna(0)
                else:
                    X_test[c] = X_test[c].fillna("missing")
    return X_train, X_test


def build_voting():  # type: ignore
    estimators = {
        "rf": RandomForestClassifier(n_estimators=50, random_state=42),
        "gb": GradientBoostingClassifier(random_state=42),
    }
    return VotingEnsemble(estimators=estimators, voting="soft")


def build_stacking():  # type: ignore
    base_estimators = {
        "rf": RandomForestClassifier(n_estimators=40, random_state=42),
        "gb": GradientBoostingClassifier(random_state=42),
    }
    meta = LogisticRegression(max_iter=500, random_state=42)
    return StackingEnsemble(
        base_estimators=base_estimators,
        meta_estimator=meta,
        n_folds=3,
    )


def main():  # type: ignore
    parser = argparse.ArgumentParser(description="Train ensembles on selected dataset")
    parser.add_argument(
        "--dataset",
        "-d",
        default=None,
        help="Dataset key (defaults to config.default_dataset)",
    )
    parser.add_argument(
        "--save-processed",
        action="store_true",
        help=(
            "Save processed train/test feature matrices to dataset "
            "processed_path prefix."
        ),
    )
    parser.add_argument(
        "--no-auto-weight",
        action="store_true",
        help="Disable automatic class weight balancing regardless of config.",
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Apply SMOTE oversampling to training set after encoding.",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE even if enabled via dataset configuration.",
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log parameters and metrics to MLflow (local file store).",
    )
    args = parser.parse_args()

    cfg = load_config()
    dataset_key = args.dataset or cfg.get("default_dataset")
    if dataset_key not in cfg["datasets"]:
        raise SystemExit(f"Dataset '{dataset_key}' not found in config")

    logger.info(f"Using dataset: {dataset_key}")
    ds_cfg = cfg["datasets"][dataset_key]
    overrides_cfg: Dict[str, Any] = {}
    if isinstance(ds_cfg, dict):
        overrides_cfg = ds_cfg.get("overrides", {}) or {}
    split, target_col = load_dataset(dataset_key)
    X_train, X_test = split["X_train"], split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]

    # Impute before encoding
    X_train, X_test = basic_impute(X_train, X_test, cfg)

    # Basic preprocessing: one-hot encode categorical columns
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        logger.info(f"One-hot encoding categorical columns: {cat_cols}")
        n_train = X_train.shape[0]
        combined = pd.concat([X_train, X_test], axis=0)
        encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
        X_train = encoded.iloc[:n_train].reset_index(drop=True)
        X_test = encoded.iloc[n_train:].reset_index(drop=True)
    else:
        logger.info("No categorical columns detected; skipping encoding.")
        # Ensure copy to avoid accidental view issues
        X_train = X_train.copy()
        X_test = X_test.copy()

    # Safety check: remain no NaNs
    if X_train.isna().any().any() or X_test.isna().any().any():
        bad_train = [c for c in X_train.columns if X_train[c].isna().any()]
        bad_test = [c for c in X_test.columns if X_test[c].isna().any()]
        logger.warning(
            "Remaining NaNs after imputation. train_cols=%s test_cols=%s",
            bad_train,
            bad_test,
        )

    # Optional class imbalance handling
    tab_cfg = cfg.get("preprocessing", {}).get("tabular", {}) or {}
    imb_cfg: Dict[str, Any] = {}
    if isinstance(tab_cfg, dict):
        imb_cfg = tab_cfg.get("class_imbalance", {}) or {}
    class_weight = None
    imbalance_ratio = 1.0
    if (
        not args.no_auto_weight
        and isinstance(imb_cfg, dict)
        and imb_cfg.get("enable_auto_weight")
    ):
        ratio_thresh = imb_cfg.get("imbalance_ratio_threshold", 3.0)
        class_weight, imbalance_ratio = decide_class_weight(y_train, ratio_thresh)
        if class_weight:
            logger.info(
                "Applying class_weight='balanced' (ratio %.2f >= " "threshold %.2f)",
                imbalance_ratio,
                ratio_thresh,
            )
        else:
            logger.info(
                "Class imbalance ratio %.2f below threshold %.2f (no " "weighting)",
                imbalance_ratio,
                ratio_thresh,
            )

    # Optional SMOTE (after encoding so features all numeric)
    smote_config_default = bool(overrides_cfg.get("use_smote"))
    smote_flag = bool(args.smote or smote_config_default)
    if args.no_smote:
        smote_flag = False
    smote_applied = False
    smote_message = "not requested"
    if smote_flag and smote_config_default and not args.smote:
        logger.info("SMOTE enabled via dataset override.")
    if smote_flag:
        feature_columns = list(X_train.columns)
        target_dtype = y_train.dtype
        before_counts = Counter(y_train.tolist())
        X_res, y_res, sm_applied, sm_msg = apply_smote(X_train, y_train)
        if sm_applied:
            smote_applied = True
            smote_message = sm_msg
            if not isinstance(X_res, pd.DataFrame):
                X_train = pd.DataFrame(X_res, columns=feature_columns)
            else:
                X_train = X_res
            if isinstance(y_res, pd.Series):
                y_train = y_res.reset_index(drop=True)
            else:
                y_train = pd.Series(y_res)
                try:
                    y_train = y_train.astype(target_dtype)
                except Exception:  # noqa: BLE001
                    pass
        else:
            smote_message = sm_msg
        after_counts = Counter(y_train.tolist())
        logger.info(
            "SMOTE attempt: %s | before=%s after=%s",
            smote_message,
            before_counts,
            after_counts,
        )

    logger.info("Training VotingEnsemble...")
    voting = build_voting()
    if class_weight:
        for name, est in voting.estimators.items():
            if hasattr(est, "class_weight"):
                try:
                    est.set_params(class_weight=class_weight)
                except Exception:  # noqa: BLE001
                    pass
    voting.set_feature_names(list(X_train.columns))
    voting.fit(X_train, y_train)
    preds_v = voting.predict(X_test)
    y_test_list = list(y_test)
    logger.info(
        "Voting ensemble report: shape=%s classes=%d",
        preds_v.shape,
        len(set(y_test_list)),
    )
    report_v_text = classification_report(y_test_list, preds_v)
    logger.info("\n%s", report_v_text)
    report_v_dict = classification_report(y_test_list, preds_v, output_dict=True)
    proba_v = None
    roc_auc_v = None
    try:
        proba_v = voting.predict_proba(X_test)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Voting predict_proba failed: %s", exc)
    if (
        proba_v is not None
        and len(set(y_test_list)) == 2
        and getattr(voting, "classes_", None) is not None
    ):
        classes_v = list(voting.classes_)
        if len(classes_v) == 2:
            pos_cls_v = classes_v[1]
            pos_idx_v = classes_v.index(pos_cls_v)
            y_binary_v = [1 if val == pos_cls_v else 0 for val in y_test_list]
            try:
                roc_auc_v = roc_auc_score(y_binary_v, proba_v[:, pos_idx_v])
                logger.info("Voting ROC-AUC: %.4f", roc_auc_v)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Voting ROC-AUC computation failed: %s", exc)

    logger.info("Training StackingEnsemble...")
    stacking = build_stacking()
    if class_weight:
        for name, est in stacking.base_estimators.items():
            if hasattr(est, "class_weight"):
                try:
                    est.set_params(class_weight=class_weight)
                except Exception:  # noqa: BLE001
                    pass
    stacking.set_feature_names(list(X_train.columns))
    stacking.fit(X_train, y_train)
    preds_s = stacking.predict(X_test)
    logger.info(
        "Stacking ensemble report: shape=%s classes=%d",
        preds_s.shape,
        len(set(y_test_list)),
    )
    report_s_text = classification_report(y_test_list, preds_s)
    logger.info("\n%s", report_s_text)
    report_s_dict = classification_report(y_test_list, preds_s, output_dict=True)
    proba_s = None
    roc_auc_s = None
    try:
        proba_s = stacking.predict_proba(X_test)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Stacking predict_proba failed: %s", exc)
    if (
        proba_s is not None
        and len(set(y_test_list)) == 2
        and getattr(stacking, "classes_", None) is not None
    ):
        classes_s = list(stacking.classes_)
        if len(classes_s) == 2:
            pos_cls_s = classes_s[1]
            pos_idx_s = classes_s.index(pos_cls_s)
            y_binary_s = [1 if val == pos_cls_s else 0 for val in y_test_list]
            try:
                roc_auc_s = roc_auc_score(y_binary_s, proba_s[:, pos_idx_s])
                logger.info("Stacking ROC-AUC: %.4f", roc_auc_s)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Stacking ROC-AUC computation failed: %s", exc)

    # Feature importance (if available)
    vi = voting.get_feature_importance()
    if vi:
        top5 = list(vi.items())[:5]
        logger.info("Voting top5 importance: %s", top5)
    si = stacking.get_feature_importance()
    if si:
        top5s = list(si.items())[:5]
        logger.info("Stacking top5 importance: %s", top5s)

    # Optionally save processed datasets
    if args.save_processed or imb_cfg.get("save_processed_default"):
        ds_cfg = cfg["datasets"][dataset_key]
        processed_path = Path(
            ds_cfg.get(
                "processed_path",
                f"data/processed/{dataset_key}_processed.csv",
            )
        )
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        # Build combined dataframe with target for reproducibility
        proc_train = X_train.copy()
        proc_train[target_col] = y_train.values
        proc_test = X_test.copy()
        proc_test[target_col] = y_test.values
        proc_train.to_csv(
            processed_path.with_name(processed_path.stem + "_train.csv"),
            index=False,
        )
        proc_test.to_csv(
            processed_path.with_name(processed_path.stem + "_test.csv"),
            index=False,
        )
        logger.info(
            "Saved processed train/test to %s and %s",
            processed_path.with_name(processed_path.stem + "_train.csv"),
            processed_path.with_name(processed_path.stem + "_test.csv"),
        )

    # Enhanced MLflow logging with model registry
    if args.mlflow:
        try:
            # Initialize MLflow manager
            mlflow_manager = MLflowManager()

            # Start run with enhanced metadata
            with mlflow_manager.start_run(
                run_name=f"ensemble_{dataset_key}",
                tags={
                    "model_type": "ensemble",
                    "dataset": dataset_key,
                    "imbalance_handling": "smote" if smote_applied else "class_weight",
                },
            ):
                # Log dataset information
                cfg = load_config()
                dataset_config = cfg["datasets"][dataset_key]
                mlflow_manager.log_dataset_info(dataset_key, dataset_config)

                # Log preprocessing parameters
                mlflow_manager.log_preprocessing_params(cfg.get("preprocessing", {}))

                # Log training parameters
                mlflow_manager.log_training_params(
                    class_weight=class_weight,
                    smote_applied=smote_applied,
                    imbalance_ratio=imbalance_ratio,
                    n_features=X_train.shape[1],
                    n_train_samples=X_train.shape[0],
                    n_test_samples=X_test.shape[0],
                )

                # Log model performance metrics
                def _log_report(prefix: str, report: Dict[str, Any]) -> None:
                    metrics_dict = {}
                    for label, metrics in report.items():
                        if label == "accuracy":
                            metrics_dict["accuracy"] = float(metrics)
                            continue
                        safe_label = (
                            str(label)
                            .replace(" ", "_")
                            .replace("/", "_")
                            .replace("-", "_")
                        )
                        for metric_name in ("precision", "recall", "f1-score"):
                            if (
                                metric_name in metrics
                                and metrics[metric_name] is not None
                            ):
                                safe_metric = metric_name.replace("-", "_")
                                key = f"{safe_metric}_{safe_label}"
                                metrics_dict[key] = float(metrics[metric_name])

                    mlflow_manager.log_model_metrics(metrics_dict, prefix)

                _log_report("voting", report_v_dict)
                _log_report("stacking", report_s_dict)

                # Log ROC AUC scores
                if roc_auc_v is not None:
                    mlflow_manager.log_model_metrics(
                        {"roc_auc": float(roc_auc_v)}, "voting"
                    )
                if roc_auc_s is not None:
                    mlflow_manager.log_model_metrics(
                        {"roc_auc": float(roc_auc_s)}, "stacking"
                    )
                # Log feature importance for top features
                if vi:
                    feature_importance = {}
                    for rank, (fname, val) in enumerate(list(vi.items())[:10], 1):
                        safe_fname = str(fname).replace(" ", "_")
                        feature_importance[f"feat_{rank}_{safe_fname}"] = float(val)
                    mlflow_manager.log_model_metrics(feature_importance, "voting")

                if si:
                    feature_importance = {}
                    for rank, (fname, val) in enumerate(list(si.items())[:10], 1):
                        safe_fname = str(fname).replace(" ", "_")
                        feature_importance[f"feat_{rank}_{safe_fname}"] = float(val)
                    mlflow_manager.log_model_metrics(feature_importance, "stacking")

                # Log models to MLflow with model registry
                try:
                    # Prepare input example
                    input_example = None
                    if len(X_test) > 0:
                        input_example = X_test.head(1).to_dict("records")[0]

                    # Log voting ensemble
                    voting_uri = mlflow_manager.log_ensemble_model(
                        model=voting,
                        model_name="voting_ensemble",
                        input_example=input_example,
                    )

                    # Log stacking ensemble
                    stacking_uri = mlflow_manager.log_ensemble_model(
                        model=stacking,
                        model_name="stacking_ensemble",
                        input_example=input_example,
                    )

                    # Register models if performance is good enough
                    if roc_auc_v and roc_auc_v > 0.7:
                        description_v = (
                            f"Voting ensemble trained on {dataset_key} "
                            f"with ROC-AUC: {roc_auc_v:.3f}"
                        )
                        mlflow_manager.register_model(
                            model_uri=voting_uri,
                            model_name=f"healthcare_voting_ensemble_{dataset_key}",
                            stage="Staging",
                            description=description_v,
                        )

                    if roc_auc_s and roc_auc_s > 0.7:
                        description_s = (
                            f"Stacking ensemble trained on {dataset_key} "
                            f"with ROC-AUC: {roc_auc_s:.3f}"
                        )
                        mlflow_manager.register_model(
                            model_uri=stacking_uri,
                            model_name=f"healthcare_stacking_ensemble_{dataset_key}",
                            stage="Staging",
                            description=description_s,
                        )

                except Exception as model_logging_error:
                    logger.warning(f"Model logging failed: {model_logging_error}")

                logger.info(
                    "MLflow run completed with run_id=%s",
                    mlflow.active_run().info.run_id,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("MLflow logging failed: %s", e)


if __name__ == "__main__":
    main()
