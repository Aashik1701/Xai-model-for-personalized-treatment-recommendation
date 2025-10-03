#!/usr/bin/env python3
"""
End-to-End Deployment Validation Script

Validates the complete deployment by:
1. Checking MLflow registry
2. Loading all production models
3. Making test predictions
4. Measuring performance
5. Generating validation report

Usage:
    python scripts/validate_deployment.py
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Configuration
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

PRODUCTION_MODELS = [
    "SyntheaLongitudinal",
    "ISICSkinCancer",
    "DrugReviews",
    "BreastCancer",
    "HeartDisease",
]

# Test data configurations
TEST_CONFIGS = {
    "HeartDisease": {"n_features": 13, "n_samples": 100},
    "BreastCancer": {"n_features": 30, "n_samples": 100},
    "SyntheaLongitudinal": {"n_features": 31, "n_samples": 50},
    "ISICSkinCancer": {"n_features": 14, "n_samples": 50},
    "DrugReviews": {"n_features": 13, "n_samples": 50},
}


class DeploymentValidator:
    """Validates deployment readiness"""

    def __init__(self):
        self.client = MlflowClient()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "mlflow_status": {},
            "model_loading": {},
            "predictions": {},
            "performance": {},
            "overall": {},
        }
        self.errors = []

    def validate_mlflow_registry(self) -> bool:
        """Validate MLflow registry is accessible"""
        print("\n" + "=" * 80)
        print("1. VALIDATING MLFLOW REGISTRY")
        print("=" * 80)

        try:
            models = self.client.search_registered_models()
            print(f"✓ MLflow registry accessible")
            print(f"✓ Found {len(models)} registered models")

            self.results["mlflow_status"]["accessible"] = True
            self.results["mlflow_status"]["total_models"] = len(models)

            # Check production models
            model_names = [m.name for m in models]
            missing = [m for m in PRODUCTION_MODELS if m not in model_names]

            if missing:
                print(f"✗ Missing models: {missing}")
                self.errors.append(f"Missing models: {missing}")
                self.results["mlflow_status"]["missing_models"] = missing
                return False

            print(f"✓ All {len(PRODUCTION_MODELS)} production models found")
            self.results["mlflow_status"]["all_models_present"] = True

            # Check Production stage
            for model_name in PRODUCTION_MODELS:
                versions = self.client.search_model_versions(f"name='{model_name}'")
                production_versions = [
                    v for v in versions if v.current_stage == "Production"
                ]

                if not production_versions:
                    print(f"✗ {model_name} has no Production version")
                    self.errors.append(f"{model_name} missing Production version")
                else:
                    print(
                        f"  ✓ {model_name}: v{production_versions[0].version} (Production)"
                    )

            return len(self.errors) == 0

        except Exception as e:
            print(f"✗ MLflow registry error: {e}")
            self.errors.append(f"MLflow error: {e}")
            self.results["mlflow_status"]["accessible"] = False
            self.results["mlflow_status"]["error"] = str(e)
            return False

    def validate_model_loading(self) -> bool:
        """Validate all models can be loaded"""
        print("\n" + "=" * 80)
        print("2. VALIDATING MODEL LOADING")
        print("=" * 80)

        all_loaded = True
        load_times = {}

        for model_name in PRODUCTION_MODELS:
            model_uri = f"models:/{model_name}/Production"

            try:
                print(f"\nLoading {model_name}...")
                start_time = time.time()
                model = mlflow.pyfunc.load_model(model_uri)
                load_time = time.time() - start_time

                load_times[model_name] = load_time

                print(f"  ✓ Loaded in {load_time:.2f}s")

                self.results["model_loading"][model_name] = {
                    "loaded": True,
                    "load_time_seconds": load_time,
                }

                # Check load time
                if load_time > 5.0:
                    print(f"  ⚠ Load time exceeds 5s threshold")
                    self.results["model_loading"][model_name]["slow_load"] = True

            except Exception as e:
                print(f"  ✗ Failed to load: {e}")
                self.errors.append(f"Failed to load {model_name}: {e}")
                self.results["model_loading"][model_name] = {
                    "loaded": False,
                    "error": str(e),
                }
                all_loaded = False

        # Summary
        if load_times:
            avg_load_time = sum(load_times.values()) / len(load_times)
            max_load_time = max(load_times.values())

            print(f"\n{'─'*80}")
            print(f"Load Time Summary:")
            print(f"  Average: {avg_load_time:.2f}s")
            print(f"  Maximum: {max_load_time:.2f}s")

            self.results["performance"]["avg_load_time"] = avg_load_time
            self.results["performance"]["max_load_time"] = max_load_time

            if avg_load_time < 3.0:
                print(f"  ✓ Average load time meets target (<3s)")
            else:
                print(f"  ⚠ Average load time exceeds target (>3s)")

        return all_loaded

    def validate_predictions(self) -> bool:
        """Validate predictions work correctly"""
        print("\n" + "=" * 80)
        print("3. VALIDATING PREDICTIONS")
        print("=" * 80)

        all_predictions_ok = True

        for model_name in PRODUCTION_MODELS:
            print(f"\nTesting {model_name}...")

            if model_name not in TEST_CONFIGS:
                print(f"  ⚠ No test config, skipping")
                continue

            config = TEST_CONFIGS[model_name]

            try:
                # Load model
                model_uri = f"models:/{model_name}/Production"
                model = mlflow.pyfunc.load_model(model_uri)

                # Generate test data
                n_features = config["n_features"]
                n_samples = config["n_samples"]

                test_data = pd.DataFrame(
                    np.random.randn(n_samples, n_features),
                    columns=[f"feature_{i}" for i in range(n_features)],
                )

                # Make predictions
                start_time = time.time()
                predictions = model.predict(test_data)
                inference_time = (time.time() - start_time) * 1000  # ms

                # Validate predictions
                assert (
                    len(predictions) == n_samples
                ), f"Expected {n_samples} predictions, got {len(predictions)}"

                # Check for NaN
                if isinstance(predictions, np.ndarray):
                    nan_count = np.isnan(predictions).sum()
                    if nan_count > 0:
                        print(f"  ✗ {nan_count} NaN predictions")
                        all_predictions_ok = False
                        self.errors.append(f"{model_name}: {nan_count} NaN predictions")

                per_sample_time = inference_time / n_samples

                print(f"  ✓ {n_samples} predictions successful")
                print(f"  ✓ Total inference time: {inference_time:.2f}ms")
                print(f"  ✓ Per-sample time: {per_sample_time:.2f}ms")

                self.results["predictions"][model_name] = {
                    "success": True,
                    "n_samples": n_samples,
                    "inference_time_ms": inference_time,
                    "per_sample_ms": per_sample_time,
                }

                # Check performance target
                if per_sample_time > 1.0:  # 1ms per sample
                    print(f"  ⚠ Per-sample time exceeds 1ms")
                    self.results["predictions"][model_name]["slow_inference"] = True

            except Exception as e:
                print(f"  ✗ Prediction failed: {e}")
                self.errors.append(f"{model_name} prediction failed: {e}")
                self.results["predictions"][model_name] = {
                    "success": False,
                    "error": str(e),
                }
                all_predictions_ok = False

        return all_predictions_ok

    def validate_batch_performance(self) -> bool:
        """Validate batch prediction performance"""
        print("\n" + "=" * 80)
        print("4. VALIDATING BATCH PERFORMANCE")
        print("=" * 80)

        # Test with a larger batch
        model_name = "HeartDisease"
        batch_size = 1000

        try:
            print(f"\nTesting {model_name} with {batch_size} samples...")

            model_uri = f"models:/{model_name}/Production"
            model = mlflow.pyfunc.load_model(model_uri)

            # Generate test data
            test_data = pd.DataFrame(
                np.random.randn(batch_size, 13),
                columns=[f"feature_{i}" for i in range(13)],
            )

            # Make predictions
            start_time = time.time()
            predictions = model.predict(test_data)
            total_time = (time.time() - start_time) * 1000  # ms

            per_sample_time = total_time / batch_size
            throughput = batch_size / (total_time / 1000)  # samples/sec

            print(f"  ✓ {batch_size} predictions in {total_time:.2f}ms")
            print(f"  ✓ {per_sample_time:.3f}ms per sample")
            print(f"  ✓ Throughput: {throughput:.0f} samples/sec")

            self.results["performance"]["batch_test"] = {
                "batch_size": batch_size,
                "total_time_ms": total_time,
                "per_sample_ms": per_sample_time,
                "throughput_samples_per_sec": throughput,
            }

            # Check targets
            if per_sample_time < 0.1:  # 0.1ms per sample in batch
                print(f"  ✓ Excellent batch performance")
            elif per_sample_time < 1.0:
                print(f"  ✓ Good batch performance")
            else:
                print(f"  ⚠ Batch performance could be improved")

            return True

        except Exception as e:
            print(f"  ✗ Batch test failed: {e}")
            self.errors.append(f"Batch test failed: {e}")
            self.results["performance"]["batch_test"] = {
                "success": False,
                "error": str(e),
            }
            return False

    def generate_report(self) -> Tuple[bool, Dict]:
        """Generate validation report"""
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        # Overall assessment
        passed = len(self.errors) == 0

        print(f"\nStatus: {'✅ PASSED' if passed else '❌ FAILED'}")
        print(f"Total Errors: {len(self.errors)}")

        if self.errors:
            print("\nErrors:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")

        # Success metrics
        print(f"\n{'─'*80}")
        print("Metrics:")

        if "mlflow_status" in self.results:
            total = self.results["mlflow_status"].get("total_models", 0)
            print(f"  • Models Registered: {total}")

        loaded = sum(
            1 for m in self.results["model_loading"].values() if m.get("loaded", False)
        )
        print(f"  • Models Loaded: {loaded}/{len(PRODUCTION_MODELS)}")

        pred_success = sum(
            1 for m in self.results["predictions"].values() if m.get("success", False)
        )
        print(f"  • Prediction Tests Passed: {pred_success}/{len(PRODUCTION_MODELS)}")

        if "performance" in self.results:
            perf = self.results["performance"]
            if "avg_load_time" in perf:
                print(f"  • Avg Load Time: {perf['avg_load_time']:.2f}s")
            if (
                "batch_test" in perf
                and "throughput_samples_per_sec" in perf["batch_test"]
            ):
                print(
                    f"  • Throughput: {perf['batch_test']['throughput_samples_per_sec']:.0f} samples/sec"
                )

        # Overall results
        self.results["overall"] = {
            "passed": passed,
            "total_errors": len(self.errors),
            "errors": self.errors,
            "models_loaded": loaded,
            "predictions_successful": pred_success,
        }

        # Save report
        report_dir = Path("reports/deployment_validation")
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"validation_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n📄 Report saved to: {report_file}")

        return passed, self.results

    def run_full_validation(self) -> bool:
        """Run complete validation suite"""
        print("\n" + "=" * 80)
        print("DEPLOYMENT VALIDATION")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"MLflow URI: {MLFLOW_TRACKING_URI}")
        print(f"Models to Validate: {len(PRODUCTION_MODELS)}")

        # Run all validations
        mlflow_ok = self.validate_mlflow_registry()
        loading_ok = self.validate_model_loading()
        predictions_ok = self.validate_predictions()
        batch_ok = self.validate_batch_performance()

        # Generate report
        passed, results = self.generate_report()

        print("\n" + "=" * 80)
        if passed:
            print("✅ DEPLOYMENT VALIDATION PASSED")
            print("=" * 80)
            print("\n🎉 All systems ready for production!")
        else:
            print("❌ DEPLOYMENT VALIDATION FAILED")
            print("=" * 80)
            print("\n⚠️  Please fix errors before deploying to production")

        return passed


def main():
    """Main entry point"""
    validator = DeploymentValidator()

    try:
        passed = validator.run_full_validation()
        sys.exit(0 if passed else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n❌ Validation failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
