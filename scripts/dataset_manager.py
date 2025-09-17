"""
Dataset Management Utility for Healthcare XAI Project

This script helps download, organize, and validate datasets from the catalog.
"""

import pandas as pd
from pathlib import Path
import logging
import yaml
from typing import Dict, Optional
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages dataset downloads, organization, and validation."""

    def __init__(self, config_path: str = "config/data_config.yaml"):
        """Initialize with configuration."""
        self.config_path = Path(config_path)
        self.data_root = Path("data")
        self.config = self._load_config()
        self.catalog = self._load_catalog()

    def _load_config(self) -> Dict:
        """Load data configuration."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def _load_catalog(self) -> pd.DataFrame:
        """Load dataset catalog."""
        try:
            catalog_path = self.data_root / "Xai_datasets.csv"
            return pd.read_csv(catalog_path)
        except Exception as e:
            logger.error(f"Error loading catalog: {e}")
            return pd.DataFrame()

    def list_available_datasets(self) -> pd.DataFrame:
        """List all available datasets from catalog."""
        if self.catalog.empty:
            logger.warning("No catalog loaded")
            return pd.DataFrame()

        print("\n🩺 Available Healthcare Datasets:")
        print("=" * 60)

        for idx, row in self.catalog.iterrows():
            size_color = (
                "🟢" if "< 1" in row["Size"] else "🟡" if "MB" in row["Size"] else "🔴"
            )
            xai_suitability = "⭐" * min(3, len(row["Best For XAI"].split()))

            print(f"{idx+1:2d}. {row['Dataset Name']}")
            print(f"    {size_color} Size: {row['Size']} | Samples: {row['Samples']}")
            print(f"    🎯 Best for: {row['Best For XAI']} {xai_suitability}")
            print(f"    💻 Requirements: {row['Laptop Specs Needed']}")
            print(f"    ⏱️  Training Time: {row['Training Time']}")
            print(f"    🔗 URL: {row['Download Link']}")
            print()

        return self.catalog

    def recommend_datasets(
        self,
        max_size: str = "1 GB",
        max_training_time: str = "6 hours",
        data_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Recommend datasets based on constraints."""
        if self.catalog.empty:
            return pd.DataFrame()

        # Filter by size (simple heuristic)
        size_filter = self.catalog["Size"].str.contains("< 1|MB", case=False, na=False)
        if "GB" in max_size:
            gb_limit = float(max_size.split()[0])
            size_filter |= (
                self.catalog["Size"]
                .str.extract(r"(\d+).*GB")[0]
                .astype(float, errors="coerce")
                <= gb_limit
            )

        # Filter by training time (simple heuristic)
        time_filter = True
        if "hour" in max_training_time:
            hour_limit = float(max_training_time.split()[0])
            time_filter = self.catalog["Training Time"].str.contains(
                "min", case=False, na=False
            ) | (
                self.catalog["Training Time"]
                .str.extract(r"(\d+).*hour")[0]
                .astype(float, errors="coerce")
                <= hour_limit
            )

        # Filter by data type if specified
        type_filter = True
        if data_type:
            type_filter = self.catalog["Data Type"].str.contains(
                data_type, case=False, na=False
            )

        recommended = self.catalog[size_filter & time_filter & type_filter]

        print(
            f"\n💡 Recommended datasets (max size: {max_size}, max time: {max_training_time}):"
        )
        print("=" * 60)

        for idx, row in recommended.iterrows():
            print(f"✅ {row['Dataset Name']} ({row['Size']}, {row['Training Time']})")

        return recommended

    def organize_dataset(
        self, dataset_name: str, file_path: str, dataset_type: str = "tabular"
    ) -> str:
        """Organize a downloaded dataset into proper folder structure."""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {file_path}")

            # Determine target directory based on size and type
            file_size_mb = source_path.stat().st_size / (1024 * 1024)

            if file_size_mb > 100:  # Large files go to external
                if dataset_type == "image":
                    target_dir = self.data_root / "external" / "images"
                elif dataset_type == "text":
                    target_dir = self.data_root / "external" / "text"
                else:
                    target_dir = self.data_root / "external"
            else:  # Small files go to raw
                target_dir = self.data_root / "raw"

            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)

            # Generate target filename
            clean_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
            target_path = target_dir / f"{clean_name}{source_path.suffix}"

            # Move or copy file
            if source_path.parent != target_dir:
                import shutil

                shutil.copy2(source_path, target_path)
                logger.info(f"Dataset organized: {target_path}")

            # Update config if needed
            self._update_dataset_config(clean_name, str(target_path), dataset_type)

            return str(target_path)

        except Exception as e:
            logger.error(f"Error organizing dataset: {e}")
            raise

    def _update_dataset_config(
        self, dataset_name: str, file_path: str, dataset_type: str
    ):
        """Update data config with new dataset."""
        try:
            if dataset_name not in self.config.get("datasets", {}):
                # Add new dataset to config
                new_dataset = {
                    "raw_path": file_path,
                    "processed_path": f"data/processed/{dataset_name}_processed.csv",
                    "target_column": "target",  # Default, user should update
                    "data_type": dataset_type,
                }

                if "datasets" not in self.config:
                    self.config["datasets"] = {}
                self.config["datasets"][dataset_name] = new_dataset

                # Save updated config
                with open(self.config_path, "w") as f:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)

                logger.info(f"Added {dataset_name} to data config")

        except Exception as e:
            logger.error(f"Error updating config: {e}")

    def validate_dataset(self, dataset_name: str) -> Dict:
        """Validate a dataset."""
        try:
            dataset_config = self.config.get("datasets", {}).get(dataset_name)
            if not dataset_config:
                return {"valid": False, "error": "Dataset not in config"}

            file_path = Path(dataset_config["raw_path"])
            if not file_path.exists():
                return {"valid": False, "error": f"File not found: {file_path}"}

            # Basic validation
            if dataset_config["data_type"] == "tabular":
                df = pd.read_csv(file_path)

                validation_result = {
                    "valid": True,
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "missing_values": df.isnull().sum().sum(),
                    "target_column_exists": dataset_config.get("target_column")
                    in df.columns,
                    "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                }

                logger.info(f"✅ Dataset {dataset_name} validation passed")
                return validation_result

            return {"valid": True, "message": "Basic file existence check passed"}

        except Exception as e:
            logger.error(f"Error validating dataset {dataset_name}: {e}")
            return {"valid": False, "error": str(e)}

    def create_data_summary(self) -> Dict:
        """Create summary of all organized datasets."""
        summary = {
            "total_datasets": 0,
            "by_type": {},
            "by_location": {},
            "total_size_mb": 0,
            "datasets": {},
        }

        try:
            for dataset_name, config in self.config.get("datasets", {}).items():
                file_path = Path(config["raw_path"])
                if file_path.exists():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    data_type = config.get("data_type", "unknown")
                    location = "external" if "external" in str(file_path) else "raw"

                    summary["total_datasets"] += 1
                    summary["total_size_mb"] += size_mb
                    summary["by_type"][data_type] = (
                        summary["by_type"].get(data_type, 0) + 1
                    )
                    summary["by_location"][location] = (
                        summary["by_location"].get(location, 0) + 1
                    )

                    summary["datasets"][dataset_name] = {
                        "path": str(file_path),
                        "size_mb": round(size_mb, 2),
                        "type": data_type,
                        "location": location,
                    }

            return summary

        except Exception as e:
            logger.error(f"Error creating summary: {e}")
            return summary


def main():
    """Main CLI interface for dataset management."""
    manager = DatasetManager()

    print("🏥 Healthcare XAI Dataset Manager")
    print("=" * 40)

    while True:
        print("\nOptions:")
        print("1. List available datasets")
        print("2. Get recommendations")
        print("3. Organize a downloaded dataset")
        print("4. Validate dataset")
        print("5. View data summary")
        print("6. Exit")

        choice = input("\nEnter your choice (1-6): ").strip()

        if choice == "1":
            manager.list_available_datasets()

        elif choice == "2":
            max_size = input("Max size (e.g., '1 GB', '500 MB'): ").strip() or "1 GB"
            max_time = (
                input("Max training time (e.g., '6 hours', '1 hour'): ").strip()
                or "6 hours"
            )
            data_type = (
                input("Data type (tabular/image/text, or leave empty): ").strip()
                or None
            )
            manager.recommend_datasets(max_size, max_time, data_type)

        elif choice == "3":
            dataset_name = input("Dataset name: ").strip()
            file_path = input("Path to downloaded file: ").strip()
            dataset_type = (
                input("Dataset type (tabular/image/text): ").strip() or "tabular"
            )

            if dataset_name and file_path:
                try:
                    organized_path = manager.organize_dataset(
                        dataset_name, file_path, dataset_type
                    )
                    print(f"✅ Dataset organized at: {organized_path}")
                except Exception as e:
                    print(f"❌ Error: {e}")

        elif choice == "4":
            dataset_name = input("Dataset name to validate: ").strip()
            if dataset_name:
                result = manager.validate_dataset(dataset_name)
                if result["valid"]:
                    print("✅ Validation passed")
                    for key, value in result.items():
                        if key != "valid":
                            print(f"  {key}: {value}")
                else:
                    print(
                        f"❌ Validation failed: {result.get('error', 'Unknown error')}"
                    )

        elif choice == "5":
            summary = manager.create_data_summary()
            print(f"\n📊 Data Summary:")
            print(f"Total datasets: {summary['total_datasets']}")
            print(f"Total size: {summary['total_size_mb']:.1f} MB")
            print(f"By type: {summary['by_type']}")
            print(f"By location: {summary['by_location']}")

            if summary["datasets"]:
                print("\nDataset details:")
                for name, info in summary["datasets"].items():
                    print(
                        f"  {name}: {info['size_mb']} MB ({info['type']}, {info['location']})"
                    )

        elif choice == "6":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()
