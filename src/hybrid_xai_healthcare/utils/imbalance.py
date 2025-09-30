"""Utilities for handling class imbalance (class weighting & resampling)."""

from __future__ import annotations

from collections import Counter
from typing import Tuple, Optional, Iterable, Any

try:  # optional dependency
    from imblearn.over_sampling import SMOTE  # type: ignore
except Exception:  # pragma: no cover
    SMOTE = None  # type: ignore


def decide_class_weight(
    y: Iterable[Any], ratio_threshold: float
) -> Tuple[Optional[str], float]:
    """Decide whether to apply class_weight='balanced'.

    Returns (class_weight_value_or_None, ratio) where ratio =
    majority / minority.
    For multi-class, ratio uses max(count) / min(count).
    """
    counts = Counter(y)
    if len(counts) < 2:
        return None, 1.0
    majority = max(counts.values())
    minority = min(counts.values())
    ratio = majority / max(minority, 1)
    if ratio >= ratio_threshold:
        return "balanced", ratio
    return None, ratio


def apply_smote(
    X: Any,
    y: Iterable[Any],
    k_neighbors: int = 5,
    random_state: int = 42,
) -> Tuple[Any, Iterable[Any], bool, str]:  # pragma: no cover
    """Apply SMOTE if available; gracefully fallback if dependency missing.

    Returns (X_res, y_res, applied_flag, message)
    """
    if SMOTE is None:
        return X, y, False, "imblearn not installed"
    # Adjust k_neighbors if minority class too small
    from collections import Counter as C

    cls_counts = C(y)
    min_count = min(cls_counts.values()) if len(cls_counts) > 1 else 0
    effective_k = min(k_neighbors, max(min_count - 1, 1))
    try:
        sm = SMOTE(k_neighbors=effective_k, random_state=random_state)
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res, True, f"SMOTE applied (k={effective_k})"
    except Exception as e:  # pragma: no cover
        return X, y, False, f"SMOTE failed: {e}"
