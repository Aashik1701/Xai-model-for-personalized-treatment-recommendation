"""Quick test to verify which models and datasets we can actually load"""

import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# List all RandomForest models
models_dir = ROOT / "models" / "trained_real_datasets"
rf_models = sorted(
    [
        f.stem.replace("_RandomForest", "")
        for f in models_dir.glob("*_RandomForest.joblib")
    ]
)

print("Available RandomForest Models:")
print("=" * 60)
for i, model in enumerate(rf_models[:10], 1):  # Show first 10
    print(f"{i:2d}. {model}")
    model_path = models_dir / f"{model}_RandomForest.joblib"
    try:
        m = joblib.load(model_path)
        print(f"    ✓ Model loads successfully (type: {type(m).__name__})")
    except Exception as e:
        print(f"    ✗ Error: {e}")

print("\n" + "=" * 60)
print(f"Total models: {len(rf_models)}")
