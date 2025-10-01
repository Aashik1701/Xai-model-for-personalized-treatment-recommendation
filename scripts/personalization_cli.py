"""
Command-line interface for the Personalization Engine

This script provides an easy-to-use CLI for running personalization analysis.
"""

import argparse
import sys
import logging
from pathlib import Path

# Add the scripts directory to the path to import personalization_engine
sys.path.append(str(Path(__file__).parent))
from personalization_engine import PersonalizationEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_full_analysis(args):
    """Run complete personalization analysis."""
    try:
        print("🏥 Starting Personalization Analysis")
        print("=" * 50)

        # Initialize engine
        engine = PersonalizationEngine(args.config)

        # Load data
        print(f"📊 Loading patient data from: {args.data}")
        engine.load_patient_data(args.data)
        print(f"✅ Loaded {len(engine.patient_data)} patient records")

        # Create embeddings
        print("🔍 Creating patient embeddings...")
        features = args.features.split(",") if args.features else None
        engine.create_patient_embeddings(features)
        print(
            f"✅ Created embeddings with {engine.patient_embeddings.shape[1]} dimensions"
        )

        # Create cohorts
        print("👥 Creating patient cohorts...")
        cohort_analysis = engine.create_patient_cohorts(args.n_clusters)
        print(f"✅ Created {cohort_analysis.get('n_cohorts', 0)} cohorts")

        # Analyze specific patients if provided
        if args.patient_ids:
            patient_ids = args.patient_ids.split(",")

            for patient_id in patient_ids:
                patient_id = patient_id.strip()
                print(f"\n🔬 Analyzing patient: {patient_id}")

                # Find similar patients
                similar = engine.find_similar_patients(patient_id, k=args.k_similar)
                print(f"   Found {len(similar)} similar patients")

                # Generate recommendations
                recommendations = engine.generate_treatment_recommendations(patient_id)
                print(
                    f"   Generated {len(recommendations.get('recommendations', []))} treatment recommendations"
                )

                # Generate full report
                report = engine.generate_personalization_report(patient_id)
                print(
                    f"   Report saved with personalization score: {report.get('personalization_score', 0)}"
                )

        # Save model if requested
        if args.save_model:
            print(f"\n💾 Saving model to: {args.save_model}")
            engine.save_model(args.save_model)
            print("✅ Model saved successfully")

        print(f"\n🎉 Personalization analysis complete!")
        print(f"📁 Reports saved to: {engine.output_dir}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


def run_similarity_analysis(args):
    """Run patient similarity analysis only."""
    try:
        print("🔍 Starting Similarity Analysis")
        print("=" * 40)

        # Initialize engine
        engine = PersonalizationEngine(args.config)

        # Load data
        engine.load_patient_data(args.data)

        # Create embeddings
        features = args.features.split(",") if args.features else None
        engine.create_patient_embeddings(features)

        # Analyze similarity for specified patients
        patient_ids = args.patient_ids.split(",")

        for patient_id in patient_ids:
            patient_id = patient_id.strip()
            # Convert to int if possible
            try:
                patient_id = int(patient_id)
            except ValueError:
                pass
            print(f"\n👤 Patient {patient_id} - Similar Patients:")

            similar = engine.find_similar_patients(
                patient_id, k=args.k_similar, min_similarity=args.min_similarity
            )

            if not similar:
                print("   No similar patients found")
                continue

            for i, sim_patient in enumerate(similar, 1):
                print(
                    f"   {i}. Patient {sim_patient['patient_id']} "
                    f"(similarity: {sim_patient['similarity']:.3f})"
                )

        print("\n✅ Similarity analysis complete!")

    except Exception as e:
        logger.error(f"Similarity analysis failed: {e}")
        sys.exit(1)


def run_cohort_analysis(args):
    """Run patient cohort analysis only."""
    try:
        print("👥 Starting Cohort Analysis")
        print("=" * 35)

        # Initialize engine
        engine = PersonalizationEngine(args.config)

        # Load data
        engine.load_patient_data(args.data)

        # Create embeddings
        features = args.features.split(",") if args.features else None
        engine.create_patient_embeddings(features)

        # Create cohorts
        cohort_analysis = engine.create_patient_cohorts(args.n_clusters)

        print(f"\n📊 Cohort Analysis Results:")
        print(f"   Total cohorts: {cohort_analysis.get('n_cohorts', 0)}")

        # Display cohort sizes
        cohort_sizes = cohort_analysis.get("cohort_sizes", {})
        for cohort_name, size in cohort_sizes.items():
            print(f"   {cohort_name}: {size} patients")

        print("\n✅ Cohort analysis complete!")

    except Exception as e:
        logger.error(f"Cohort analysis failed: {e}")
        sys.exit(1)


def run_recommendations(args):
    """Generate treatment recommendations for specific patients."""
    try:
        print("💊 Starting Treatment Recommendation Analysis")
        print("=" * 50)

        # Initialize engine
        engine = PersonalizationEngine(args.config)

        # Load data
        engine.load_patient_data(args.data)

        # Create embeddings
        features = args.features.split(",") if args.features else None
        engine.create_patient_embeddings(features)

        # Generate recommendations for specified patients
        patient_ids = args.patient_ids.split(",")

        for patient_id in patient_ids:
            patient_id = patient_id.strip()
            # Convert to int if possible
            try:
                patient_id = int(patient_id)
            except ValueError:
                pass
            print(f"\n👤 Treatment Recommendations for Patient {patient_id}:")

            recommendations = engine.generate_treatment_recommendations(patient_id)

            confidence = recommendations.get("confidence", "unknown")
            print(f"   Confidence Level: {confidence}")
            print(f"   Based on: {recommendations.get('reasoning', 'N/A')}")

            treatments = recommendations.get("recommendations", [])
            if treatments:
                print("   Recommended Treatments:")
                for i, treatment in enumerate(treatments, 1):
                    print(
                        f"   {i}. {treatment['treatment']} "
                        f"(confidence: {treatment['confidence']:.2f})"
                    )
                    print(f"      Reasoning: {treatment['reasoning']}")
            else:
                print("   No specific treatment recommendations available")

        print("\n✅ Treatment recommendation analysis complete!")

    except Exception as e:
        logger.error(f"Treatment recommendation failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Personalization Engine CLI for Healthcare Treatment Recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis for specific patients
  python personalization_cli.py full-analysis --data data/patients.csv --patient-ids "1,2,3"
  
  # Similarity analysis only
  python personalization_cli.py similarity --data data/patients.csv --patient-ids "1,5" --k-similar 10
  
  # Cohort analysis with custom clusters
  python personalization_cli.py cohorts --data data/patients.csv --n-clusters 12
  
  # Treatment recommendations
  python personalization_cli.py recommendations --data data/patients.csv --patient-ids "1,2"
        """,
    )

    # Common arguments
    parser.add_argument(
        "--config",
        default="config/personalization_config.yaml",
        help="Path to personalization configuration file",
    )
    parser.add_argument("--data", required=True, help="Path to patient data CSV file")
    parser.add_argument(
        "--features",
        help="Comma-separated list of features to use (default: all numeric features)",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Full analysis command
    full_parser = subparsers.add_parser(
        "full-analysis", help="Run complete personalization analysis"
    )
    full_parser.add_argument(
        "--patient-ids", help="Comma-separated patient IDs to analyze"
    )
    full_parser.add_argument(
        "--n-clusters", type=int, default=8, help="Number of cohorts to create"
    )
    full_parser.add_argument(
        "--k-similar", type=int, default=5, help="Number of similar patients to find"
    )
    full_parser.add_argument(
        "--save-model", help="Path to save the trained personalization model"
    )

    # Similarity analysis command
    sim_parser = subparsers.add_parser(
        "similarity", help="Run patient similarity analysis"
    )
    sim_parser.add_argument(
        "--patient-ids", required=True, help="Comma-separated patient IDs to analyze"
    )
    sim_parser.add_argument(
        "--k-similar", type=int, default=10, help="Number of similar patients to find"
    )
    sim_parser.add_argument(
        "--min-similarity", type=float, default=0.3, help="Minimum similarity threshold"
    )

    # Cohort analysis command
    cohort_parser = subparsers.add_parser("cohorts", help="Run patient cohort analysis")
    cohort_parser.add_argument(
        "--n-clusters", type=int, default=8, help="Number of cohorts to create"
    )

    # Treatment recommendations command
    rec_parser = subparsers.add_parser(
        "recommendations", help="Generate treatment recommendations"
    )
    rec_parser.add_argument(
        "--patient-ids", required=True, help="Comma-separated patient IDs to analyze"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Validate data path
    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)

    # Run the appropriate command
    if args.command == "full-analysis":
        run_full_analysis(args)
    elif args.command == "similarity":
        run_similarity_analysis(args)
    elif args.command == "cohorts":
        run_cohort_analysis(args)
    elif args.command == "recommendations":
        run_recommendations(args)


if __name__ == "__main__":
    main()
