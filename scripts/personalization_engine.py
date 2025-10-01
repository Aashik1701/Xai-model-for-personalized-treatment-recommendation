"""
Personalization Engine for Healthcare Treatment Recommendations

This module provides patient similarity analysis, cohort identification,
and personalized treatment recommendations based on clinical data.
"""

import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)
from sklearn.neighbors import NearestNeighbors
import joblib
import json
from datetime import datetime, timedelta
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonalizationEngine:
    """
    Comprehensive personalization engine for healthcare treatment recommendations.
    """

    def __init__(self, config_path: str = "config/personalization_config.yaml"):
        """Initialize the personalization engine."""
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Initialize components
        self.scaler = StandardScaler()
        self.pca = None
        self.clustering_model = None
        self.similarity_cache = {}

        # Patient data and embeddings
        self.patient_data = None
        self.patient_embeddings = None
        self.cohort_assignments = None

        # Output directory
        self.output_dir = Path(self.config["output"]["base_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("🏥 Personalization Engine initialized")
        logger.info(f"Configuration loaded from: {self.config_path}")
        logger.info(f"Output directory: {self.output_dir}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            # Return default config
            return {
                "similarity": {
                    "distance_metrics": [{"name": "euclidean", "weight": 1.0}]
                },
                "cohorts": {"n_clusters": 5},
                "output": {"base_dir": "reports/personalization"},
            }

    def load_patient_data(
        self, data_path: str, patient_id_col: str = "patient_id"
    ) -> None:
        """Load patient data for personalization."""
        try:
            if isinstance(data_path, str):
                data_path = Path(data_path)

            if data_path.suffix == ".csv":
                self.patient_data = pd.read_csv(data_path)
            elif data_path.suffix == ".joblib":
                self.patient_data = joblib.load(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path.suffix}")

            # Ensure patient ID column exists
            if patient_id_col not in self.patient_data.columns:
                self.patient_data[patient_id_col] = range(len(self.patient_data))

            logger.info(f"✅ Loaded patient data: {self.patient_data.shape}")
            logger.info(f"Columns: {list(self.patient_data.columns)}")

        except Exception as e:
            logger.error(f"Failed to load patient data: {e}")
            raise

    def create_patient_embeddings(
        self, features: Optional[List[str]] = None
    ) -> np.ndarray:
        """Create patient embeddings for similarity calculation."""
        if self.patient_data is None:
            raise ValueError("Patient data not loaded. Call load_patient_data() first.")

        try:
            # Select features for embedding
            if features is None:
                # Use numeric columns only
                numeric_cols = self.patient_data.select_dtypes(
                    include=[np.number]
                ).columns
                features = [col for col in numeric_cols if col != "patient_id"]

            logger.info(f"Creating embeddings using features: {features}")

            # Extract feature data
            X = self.patient_data[features].copy()

            # Handle missing values
            X = X.fillna(X.median())

            # Standardize features
            X_scaled = self.scaler.fit_transform(X)

            # Apply dimensionality reduction if configured
            embedding_method = self.config.get("similarity", {}).get(
                "embedding_method", "clinical_features"
            )

            if embedding_method == "pca":
                n_components = min(10, X_scaled.shape[1])
                self.pca = PCA(n_components=n_components, random_state=42)
                X_embedded = self.pca.fit_transform(X_scaled)
                logger.info(f"PCA embedding: {X_embedded.shape}")
            else:
                # Use scaled clinical features directly
                X_embedded = X_scaled
                logger.info(f"Clinical features embedding: {X_embedded.shape}")

            self.patient_embeddings = X_embedded
            self.feature_names = features

            logger.info("✅ Patient embeddings created successfully")
            return X_embedded

        except Exception as e:
            logger.error(f"Failed to create patient embeddings: {e}")
            raise

    def calculate_patient_similarity(
        self, patient_id_1: Union[int, str], patient_id_2: Union[int, str]
    ) -> float:
        """Calculate similarity between two patients."""
        if self.patient_embeddings is None:
            raise ValueError(
                "Patient embeddings not created. Call create_patient_embeddings() first."
            )

        try:
            # Get patient indices
            if isinstance(patient_id_1, str):
                idx_1 = self.patient_data[
                    self.patient_data["patient_id"] == patient_id_1
                ].index[0]
            else:
                idx_1 = patient_id_1

            if isinstance(patient_id_2, str):
                idx_2 = self.patient_data[
                    self.patient_data["patient_id"] == patient_id_2
                ].index[0]
            else:
                idx_2 = patient_id_2

            # Get embeddings
            emb_1 = self.patient_embeddings[idx_1].reshape(1, -1)
            emb_2 = self.patient_embeddings[idx_2].reshape(1, -1)

            # Calculate weighted similarity using configured metrics
            total_similarity = 0.0
            total_weight = 0.0

            distance_metrics = self.config.get("similarity", {}).get(
                "distance_metrics", [{"name": "euclidean", "weight": 1.0}]
            )

            for metric_config in distance_metrics:
                metric_name = metric_config["name"]
                weight = metric_config["weight"]

                if metric_name == "cosine":
                    similarity = cosine_similarity(emb_1, emb_2)[0, 0]
                elif metric_name == "euclidean":
                    distance = euclidean_distances(emb_1, emb_2)[0, 0]
                    # Convert distance to similarity (0-1 range)
                    max_distance = np.sqrt(emb_1.shape[1])  # Maximum possible distance
                    similarity = 1 - (distance / max_distance)
                elif metric_name == "manhattan":
                    distance = manhattan_distances(emb_1, emb_2)[0, 0]
                    # Convert distance to similarity
                    max_distance = emb_1.shape[1]  # Maximum possible Manhattan distance
                    similarity = 1 - (distance / max_distance)
                else:
                    # Default to cosine similarity
                    similarity = cosine_similarity(emb_1, emb_2)[0, 0]

                total_similarity += similarity * weight
                total_weight += weight

            final_similarity = (
                total_similarity / total_weight if total_weight > 0 else 0.0
            )
            return max(0.0, min(1.0, final_similarity))  # Clamp to [0, 1]

        except Exception as e:
            logger.error(
                f"Failed to calculate similarity between {patient_id_1} and {patient_id_2}: {e}"
            )
            return 0.0

    def find_similar_patients(
        self, patient_id: Union[int, str], k: int = 10, min_similarity: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Find k most similar patients to the given patient."""
        if self.patient_embeddings is None:
            raise ValueError(
                "Patient embeddings not created. Call create_patient_embeddings() first."
            )

        try:
            # Get patient index
            patient_mask = self.patient_data["patient_id"] == patient_id
            if not patient_mask.any():
                logger.error(f"Patient {patient_id} not found in dataset")
                return []

            patient_idx = self.patient_data[patient_mask].index[0]

            # Get patient embedding
            query_embedding = self.patient_embeddings[patient_idx].reshape(1, -1)

            # Calculate similarities to all other patients
            similarities = []
            for i in range(len(self.patient_embeddings)):
                if i != patient_idx:
                    sim = self.calculate_patient_similarity(patient_idx, i)
                    if sim >= min_similarity:
                        patient_data = self.patient_data.iloc[i]
                        similarities.append(
                            {
                                "patient_id": patient_data.get("patient_id", i),
                                "similarity": sim,
                                "patient_data": patient_data.to_dict(),
                            }
                        )

            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)

            logger.info(
                f"Found {len(similarities)} similar patients for patient {patient_id}"
            )
            return similarities[:k]

        except Exception as e:
            logger.error(f"Failed to find similar patients for {patient_id}: {e}")
            return []

    def create_patient_cohorts(
        self, n_clusters: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create patient cohorts using clustering."""
        if self.patient_embeddings is None:
            raise ValueError(
                "Patient embeddings not created. Call create_patient_embeddings() first."
            )

        try:
            if n_clusters is None:
                n_clusters = self.config.get("cohorts", {}).get("n_clusters", 5)

            clustering_method = self.config.get("cohorts", {}).get(
                "clustering_method", "kmeans"
            )

            if clustering_method == "kmeans":
                self.clustering_model = KMeans(
                    n_clusters=n_clusters, random_state=42, n_init=10
                )
                cluster_labels = self.clustering_model.fit_predict(
                    self.patient_embeddings
                )
            elif clustering_method == "dbscan":
                self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = self.clustering_model.fit_predict(
                    self.patient_embeddings
                )
                n_clusters = len(set(cluster_labels)) - (
                    1 if -1 in cluster_labels else 0
                )
            else:
                # Default to KMeans
                self.clustering_model = KMeans(
                    n_clusters=n_clusters, random_state=42, n_init=10
                )
                cluster_labels = self.clustering_model.fit_predict(
                    self.patient_embeddings
                )

            self.cohort_assignments = cluster_labels

            # Analyze cohorts
            cohort_analysis = self._analyze_cohorts(cluster_labels)

            logger.info(
                f"✅ Created {n_clusters} patient cohorts using {clustering_method}"
            )
            return cohort_analysis

        except Exception as e:
            logger.error(f"Failed to create patient cohorts: {e}")
            return {}

    def _analyze_cohorts(self, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze characteristics of patient cohorts."""
        cohort_analysis = {
            "n_cohorts": len(set(cluster_labels)),
            "cohort_sizes": {},
            "cohort_characteristics": {},
        }

        # Add cluster labels to patient data
        patient_data_with_cohorts = self.patient_data.copy()
        patient_data_with_cohorts["cohort"] = cluster_labels

        for cohort_id in set(cluster_labels):
            if cohort_id == -1:  # DBSCAN noise points
                continue

            cohort_patients = patient_data_with_cohorts[
                patient_data_with_cohorts["cohort"] == cohort_id
            ]
            cohort_size = len(cohort_patients)

            cohort_analysis["cohort_sizes"][f"cohort_{cohort_id}"] = cohort_size

            # Calculate cohort characteristics
            characteristics = {}

            # Numeric features - mean and std
            numeric_cols = cohort_patients.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ["patient_id", "cohort"]:
                    characteristics[f"{col}_mean"] = float(cohort_patients[col].mean())
                    characteristics[f"{col}_std"] = float(cohort_patients[col].std())

            # Categorical features - mode
            categorical_cols = cohort_patients.select_dtypes(include=["object"]).columns
            for col in categorical_cols:
                if col != "patient_id":
                    mode_value = cohort_patients[col].mode()
                    characteristics[f"{col}_mode"] = (
                        mode_value.iloc[0] if len(mode_value) > 0 else "unknown"
                    )

            cohort_analysis["cohort_characteristics"][
                f"cohort_{cohort_id}"
            ] = characteristics

        return cohort_analysis

    def generate_treatment_recommendations(
        self,
        patient_id: Union[int, str],
        treatment_history: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Generate personalized treatment recommendations for a patient."""
        try:
            # Find similar patients
            similar_patients = self.find_similar_patients(
                patient_id,
                k=self.config.get("recommendations", {}).get("k_neighbors", 10),
            )

            if not similar_patients:
                logger.warning(f"No similar patients found for {patient_id}")
                return {
                    "patient_id": patient_id,
                    "recommendations": [],
                    "confidence": "low",
                    "reasoning": "No similar patients found for personalized recommendations",
                }

            # Simulate treatment recommendations based on similar patients
            # In a real system, this would use actual treatment outcomes
            recommendations = self._simulate_treatment_recommendations(
                patient_id, similar_patients
            )

            return recommendations

        except Exception as e:
            logger.error(
                f"Failed to generate treatment recommendations for {patient_id}: {e}"
            )
            return {
                "patient_id": patient_id,
                "recommendations": [],
                "confidence": "low",
                "error": str(e),
            }

    def _simulate_treatment_recommendations(
        self, patient_id: Union[int, str], similar_patients: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Simulate treatment recommendations based on similar patients."""
        # Get patient data
        patient_mask = self.patient_data["patient_id"] == patient_id
        if not patient_mask.any():
            raise ValueError(f"Patient {patient_id} not found in dataset")
        patient_data = self.patient_data[patient_mask].iloc[0]

        # Simulate treatments based on patient characteristics and similar patients
        treatments = []

        # Get patient's target/diagnosis if available
        patient_diagnosis = patient_data.get("target", 0)

        # Analyze similar patients' outcomes for treatment recommendations
        similar_outcomes = [
            p["patient_data"].get("target", 0) for p in similar_patients
        ]
        high_risk_similar = sum(1 for outcome in similar_outcomes if outcome == 1)
        similarity_based_confidence = len(similar_patients) / 10.0  # Scale to 0-1

        # Breast cancer specific treatments based on features
        if "area_worst" in patient_data:
            area_worst = patient_data["area_worst"]
            if area_worst > 1000:  # Large tumor area
                treatments.append(
                    {
                        "treatment": "Neoadjuvant chemotherapy",
                        "confidence": 0.85,
                        "reasoning": f"Large tumor area ({area_worst:.1f}) suggests aggressive treatment",
                    }
                )

        if "concave_points_worst" in patient_data:
            concave_points = patient_data["concave_points_worst"]
            if concave_points > 0.15:  # High concave points
                treatments.append(
                    {
                        "treatment": "Surgical consultation for wide excision",
                        "confidence": 0.80,
                        "reasoning": f"High concave points ({concave_points:.3f}) indicate irregular borders",
                    }
                )

        if "radius_worst" in patient_data:
            radius_worst = patient_data["radius_worst"]
            if radius_worst > 20:  # Large radius
                treatments.append(
                    {
                        "treatment": "Mastectomy evaluation",
                        "confidence": 0.75,
                        "reasoning": f"Large tumor radius ({radius_worst:.1f}) may require extensive surgery",
                    }
                )

        # Heart disease specific treatments
        if "trestbps" in patient_data:  # Resting blood pressure
            bp = patient_data["trestbps"]
            if bp > 140:  # Hypertension
                treatments.append(
                    {
                        "treatment": "ACE inhibitor therapy",
                        "confidence": 0.85,
                        "reasoning": f"Hypertension detected (BP: {bp})",
                    }
                )

        if "chol" in patient_data:  # Cholesterol
            chol = patient_data["chol"]
            if chol > 240:  # High cholesterol
                treatments.append(
                    {
                        "treatment": "Statin therapy",
                        "confidence": 0.80,
                        "reasoning": f"High cholesterol level ({chol} mg/dL)",
                    }
                )

        if "thalach" in patient_data:  # Max heart rate
            thalach = patient_data["thalach"]
            if thalach < 120:  # Low exercise capacity
                treatments.append(
                    {
                        "treatment": "Cardiac rehabilitation",
                        "confidence": 0.75,
                        "reasoning": f"Low max heart rate ({thalach}) suggests reduced capacity",
                    }
                )

        if "cp" in patient_data:  # Chest pain type
            cp = patient_data["cp"]
            if cp in [1, 2]:  # Typical or atypical angina
                treatments.append(
                    {
                        "treatment": "Coronary angiography",
                        "confidence": 0.80,
                        "reasoning": f"Chest pain type {cp} suggests cardiac evaluation needed",
                    }
                )

        # Risk-based recommendations from similar patients
        if (
            high_risk_similar > len(similar_patients) * 0.6
        ):  # >60% similar patients high risk
            treatments.append(
                {
                    "treatment": "Oncology referral",
                    "confidence": min(0.90, 0.70 + similarity_based_confidence),
                    "reasoning": f"{high_risk_similar}/{len(similar_patients)} similar patients are high-risk",
                }
            )
            treatments.append(
                {
                    "treatment": "Genetic counseling",
                    "confidence": 0.70,
                    "reasoning": "High-risk profile similar to other patients",
                }
            )
        else:
            treatments.append(
                {
                    "treatment": "Regular monitoring",
                    "confidence": 0.75,
                    "reasoning": f"Lower risk profile based on {len(similar_patients)} similar patients",
                }
            )

        # General supportive care
        treatments.append(
            {
                "treatment": "Patient education and support",
                "confidence": 0.80,
                "reasoning": "Evidence-based patient care standard",
            }
        )

        # Calculate overall confidence
        if len(similar_patients) >= 5:
            overall_confidence = "high"
        elif len(similar_patients) >= 3:
            overall_confidence = "medium"
        else:
            overall_confidence = "low"

        return {
            "patient_id": patient_id,
            "recommendations": treatments,
            "confidence": overall_confidence,
            "similar_patients_count": len(similar_patients),
            "reasoning": f"Based on analysis of {len(similar_patients)} similar patients",
        }

    def generate_personalization_report(
        self, patient_id: Union[int, str]
    ) -> Dict[str, Any]:
        """Generate comprehensive personalization report for a patient."""
        try:
            # Get patient data
            patient_mask = self.patient_data["patient_id"] == patient_id
            if not patient_mask.any():
                raise ValueError(f"Patient {patient_id} not found in dataset")
            patient_data = self.patient_data[patient_mask].iloc[0]
            patient_idx = self.patient_data[patient_mask].index[0]

            # Find similar patients
            similar_patients = self.find_similar_patients(patient_id, k=5)

            # Get cohort assignment
            cohort_id = None
            if self.cohort_assignments is not None:
                cohort_id = int(self.cohort_assignments[patient_idx])

            # Generate treatment recommendations
            recommendations = self.generate_treatment_recommendations(patient_id)

            report = {
                "patient_id": patient_id,
                "timestamp": datetime.now().isoformat(),
                "patient_profile": patient_data.to_dict(),
                "cohort_assignment": cohort_id,
                "similar_patients": similar_patients,
                "treatment_recommendations": recommendations,
                "personalization_score": self._calculate_personalization_score(
                    similar_patients
                ),
                "metadata": {
                    "engine_version": "1.0.0",
                    "features_used": getattr(self, "feature_names", []),
                    "embedding_method": self.config.get("similarity", {}).get(
                        "embedding_method", "clinical_features"
                    ),
                },
            }

            # Save report
            report_path = (
                self.output_dir
                / f"personalization_report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"✅ Personalization report generated: {report_path}")
            return report

        except Exception as e:
            logger.error(
                f"Failed to generate personalization report for {patient_id}: {e}"
            )
            return {"patient_id": patient_id, "error": str(e)}

    def _calculate_personalization_score(
        self, similar_patients: List[Dict[str, Any]]
    ) -> float:
        """Calculate a personalization score based on similarity quality."""
        if not similar_patients:
            return 0.0

        # Calculate average similarity
        avg_similarity = np.mean([p["similarity"] for p in similar_patients])

        # Calculate score based on number of similar patients and average similarity
        num_patients = len(similar_patients)

        # Score components
        similarity_score = avg_similarity  # 0-1
        quantity_score = min(1.0, num_patients / 10)  # 0-1, max at 10 patients

        # Weighted combination
        personalization_score = 0.7 * similarity_score + 0.3 * quantity_score

        return round(personalization_score, 3)

    def save_model(self, model_path: str) -> None:
        """Save the personalization engine model."""
        try:
            model_data = {
                "config": self.config,
                "scaler": self.scaler,
                "pca": self.pca,
                "clustering_model": self.clustering_model,
                "patient_embeddings": self.patient_embeddings,
                "cohort_assignments": self.cohort_assignments,
                "feature_names": getattr(self, "feature_names", []),
                "timestamp": datetime.now().isoformat(),
            }

            joblib.dump(model_data, model_path)
            logger.info(f"✅ Personalization engine saved: {model_path}")

        except Exception as e:
            logger.error(f"Failed to save personalization engine: {e}")
            raise

    def load_model(self, model_path: str) -> None:
        """Load a pre-trained personalization engine model."""
        try:
            model_data = joblib.load(model_path)

            self.config = model_data.get("config", self.config)
            self.scaler = model_data.get("scaler", StandardScaler())
            self.pca = model_data.get("pca")
            self.clustering_model = model_data.get("clustering_model")
            self.patient_embeddings = model_data.get("patient_embeddings")
            self.cohort_assignments = model_data.get("cohort_assignments")
            self.feature_names = model_data.get("feature_names", [])

            logger.info(f"✅ Personalization engine loaded: {model_path}")

        except Exception as e:
            logger.error(f"Failed to load personalization engine: {e}")
            raise


def main():
    """Example usage of the PersonalizationEngine."""
    # Initialize engine
    engine = PersonalizationEngine()

    # Load sample data (would be real patient data in production)
    print("Loading sample patient data...")
    data_path = "data/raw/synthetic_healthcare_data.csv"

    if not Path(data_path).exists():
        print(f"Sample data not found at {data_path}")
        print("Please run the data generation script first:")
        print("python scripts/generate_sample_data.py")
        return

    engine.load_patient_data(data_path)

    # Create patient embeddings
    print("Creating patient embeddings...")
    engine.create_patient_embeddings()

    # Create patient cohorts
    print("Creating patient cohorts...")
    cohort_analysis = engine.create_patient_cohorts()
    print(f"Created {cohort_analysis.get('n_cohorts', 0)} cohorts")

    # Find similar patients for first patient
    patient_id = engine.patient_data.iloc[0]["patient_id"]
    print(f"\nFinding similar patients for patient {patient_id}...")
    similar = engine.find_similar_patients(patient_id, k=5)
    print(f"Found {len(similar)} similar patients")

    # Generate treatment recommendations
    print(f"\nGenerating treatment recommendations for patient {patient_id}...")
    recommendations = engine.generate_treatment_recommendations(patient_id)
    print(
        f"Generated {len(recommendations.get('recommendations', []))} recommendations"
    )

    # Generate full personalization report
    print(f"\nGenerating personalization report for patient {patient_id}...")
    report = engine.generate_personalization_report(patient_id)

    print(f"\n🎉 Personalization analysis complete!")
    print(f"Reports saved to: {engine.output_dir}")


if __name__ == "__main__":
    main()
