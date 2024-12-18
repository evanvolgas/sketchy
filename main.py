import warnings
import numpy as np
import pandas as pd
from sketchy import AnomalyDetector, DatasetComparator
from sklearn.model_selection import train_test_split


def prepare_embeddings(df, categorical_cols):
    """Optimized embedding preparation using vectorized operations"""
    # Preallocate the output dataframe to avoid copies
    df_encoded = df.copy()

    # Batch process categorical columns using a dictionary for mapping
    encoding_maps = {
        col: pd.Series(pd.factorize(df[col])[0], index=df.index) for col in categorical_cols
    }

    # Vectorized assignment of encoded values
    for col, values in encoding_maps.items():
        df_encoded[f"{col}_encoded"] = values

    return df_encoded


def analyze_datasets(dataset_a, dataset_b, categorical_cols, numerical_cols):
    """Analyze datasets with optimized processing"""
    # First encode both datasets
    dataset_a_encoded = prepare_embeddings(dataset_a, categorical_cols)
    dataset_b_encoded = prepare_embeddings(dataset_b, categorical_cols)

    # Create embedding columns list after encoding
    embedding_cols = numerical_cols + [f"{col}_encoded" for col in categorical_cols]

    # Initialize detector with optimized parameters
    detector = AnomalyDetector(categorical_cols=categorical_cols, numerical_cols=numerical_cols)

    # Fit detector with optimized parameters
    detector.fit(
        dataset_a_encoded,
        embedding_cols=embedding_cols,
        contamination=0.1,
        dim_reduction="pca",
    )

    # Detect anomalies
    anomalies = detector.detect_anomalies(
        dataset_b_encoded, embedding_cols=embedding_cols, threshold=0.1
    )

    return anomalies


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")

    try:
        # Load and clean data
        df = pd.read_csv(
            "~/Desktop/test.csv",
            sep=",",
        )
        df = df.fillna(0)

        # Split into two datasets for comparison
        dataset_a, dataset_b = train_test_split(df, test_size=0.5, random_state=42)

        # Identify column types
        categorical_cols = dataset_a.select_dtypes("object").columns.tolist()
        numerical_cols = dataset_a.select_dtypes(include=np.number).columns.tolist()

        # Print dataset info for debugging
        print(f"\nDataset Info:")
        print(f"Categorical columns: {categorical_cols}")
        print(f"Numerical columns: {numerical_cols}")
        print(f"Dataset A shape: {dataset_a.shape}")
        print(f"Dataset B shape: {dataset_b.shape}")

        # Create encoded datasets
        dataset_a_encoded = prepare_embeddings(dataset_a, categorical_cols)
        dataset_b_encoded = prepare_embeddings(dataset_b, categorical_cols)

        # Define embedding columns
        embedding_cols = numerical_cols + [f"{col}_encoded" for col in categorical_cols]

        # Initialize comparator with debug info
        print("\nInitializing DatasetComparator...")
        comparator = DatasetComparator(
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
        )

        # Compare datasets with error handling
        print("\nComparing datasets...")
        results = comparator.compare_datasets(
            dataset_a_encoded, dataset_b_encoded, embedding_cols=embedding_cols, threshold=0.05
        )

        # Check if results contain error
        if "error" in results:
            print(f"\nError during comparison: {results['error']}")

        # Print the formatted anomalies
        if results["anomalies"]["global_distribution"]:
            print("Global Distribution Anomalies:")
            for anomaly in results["anomalies"]["global_distribution"]:
                print(f"- {anomaly}")
            print()

        if results["anomalies"]["group_distribution"]:
            print("Group Distribution Anomalies:")
            for anomaly in results["anomalies"]["group_distribution"]:
                print(f"- {anomaly}")
            print()

        if results["anomalies"]["point_anomalies"]:
            print("Point-level Anomalies (showing first 5):")
            for anomaly in results["anomalies"]["point_anomalies"][:5]:
                print(f"- {anomaly}")

        # Print comparison results with proper error handling
        print("\nDistribution Differences (Statistical Tests):")
        if results.get("statistical_tests"):
            for test_name, test_result in results["statistical_tests"].items():
                print(f"\n{test_name}:")
                if isinstance(test_result, dict):
                    if "error" in test_result:
                        print(f"- Error: {test_result['error']}")
                    else:
                        for key, value in test_result.items():
                            print(f"- {key}: {value}")
                else:
                    print("- Invalid test result format")
        else:
            print("No statistical test results available")

        print("\nShape Comparison:")
        if results.get("shape_comparison"):
            shape_comp = results["shape_comparison"]
            if "row_counts" in shape_comp:
                row_counts = shape_comp["row_counts"]
                print(f"Row count difference: {row_counts.get('difference_pct', 'N/A'):.1f}%")
        else:
            print("No shape comparison results available")

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
        print("Stack trace:")
        import traceback

        traceback.print_exc()
