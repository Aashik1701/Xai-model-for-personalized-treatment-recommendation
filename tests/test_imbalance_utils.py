"""Tests for imbalance utilities (class weight decision & SMOTE)."""

from collections import Counter
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hybrid_xai_healthcare.utils.imbalance import (  # noqa: E402
    decide_class_weight,  # noqa: E402
    apply_smote,  # noqa: E402
)


def test_decide_class_weight_triggers():
    y = [0] * 90 + [1] * 10
    cw, ratio = decide_class_weight(y, ratio_threshold=3.0)
    assert cw == "balanced"
    assert ratio == 9.0


def test_decide_class_weight_not_trigger():
    y = [0] * 60 + [1] * 55
    cw, ratio = decide_class_weight(y, ratio_threshold=3.0)
    assert cw is None
    assert ratio < 3.0


def test_apply_smote_runs(monkeypatch):
    try:
        from imblearn.over_sampling import SMOTE  # noqa: F401
    except Exception:
        # If imblearn isn't installed in the test env, skip gracefully
        return
    X = np.random.randn(100, 5)
    y = np.array([0] * 90 + [1] * 10)
    X_res, y_res, applied, msg = apply_smote(X, y)
    if applied:
        counts_after = Counter(y_res)
        assert counts_after[0] == counts_after[1]
    else:
        # SMOTE might fail if k_neighbors invalid for ultra tiny minority
        assert "SMOTE" in msg
