import warnings

import numpy as np
import pandas as pd
from sketchy import AnomalyDetector, DatasetComparator
from sklearn.model_selection import train_test_split


# Encode categorical variables
def prepare_embeddings(df, categorical_cols):
    df_encoded = df.copy()
    for col in categorical_cols:
        df_encoded[f"{col}_encoded"] = pd.factorize(df_encoded[col])[0]
    return df_encoded


warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")

if __name__ == "__main__":
    # Load and clean data
    df = pd.read_csv(
        "~/Desktop/OnlineRetail.csv",
        # "~/Desktop/test.csv",
        sep=",",
    )
    df = df.replace({np.nan: 0})

    # Split into two datasets for comparison
    dataset_a, dataset_b = train_test_split(df, test_size=0.5, random_state=42)

    # Identify column types
    categorical_cols = df.select_dtypes("object").columns.to_list()
    numerical_cols = df.select_dtypes(include=np.number).columns.to_list()

    # Create embedding columns
    dataset_a_encoded = prepare_embeddings(dataset_a, categorical_cols)
    dataset_b_encoded = prepare_embeddings(dataset_b, categorical_cols)

    # Define embedding columns for analysis
    embedding_cols = numerical_cols + [f"{col}_encoded" for col in categorical_cols]

    print("Part 1: Using AnomalyDetector")
    print("-" * 50)

    # Initialize and fit detector on baseline data
    detector = AnomalyDetector(categorical_cols=categorical_cols, numerical_cols=numerical_cols)
    detector.fit(dataset_a_encoded, embedding_cols=embedding_cols, contamination=0.1)

    # Detect anomalies in test data
    anomalies = detector.detect_anomalies(
        dataset_b_encoded, embedding_cols=embedding_cols, threshold=0.1
    )

    # Print anomaly results
    print("\nGlobal Distribution Anomalies:")
    for anomaly in anomalies["global_distribution"]:
        print(f"- {anomaly}")

    print("\nGroup Distribution Anomalies:")
    for anomaly in anomalies["group_distribution"]:
        print(f"- {anomaly}")

    print("\nPoint-level Anomalies (showing first 5):")
    for anomaly in anomalies["point_anomalies"][:5]:
        print(f"- {anomaly}")

    print("\nPart 2: Using DatasetComparator")
    print("-" * 50)

    # Initialize comparator
    comparator = DatasetComparator(categorical_cols=categorical_cols, numerical_cols=numerical_cols)

    # Compare datasets
    results = comparator.compare_datasets(
        dataset_a_encoded, dataset_b_encoded, embedding_cols=embedding_cols
    )

    # Print comparison results
    print("\nDistribution Differences (Statistical Tests):")
    for test_name, test_result in results["statistical_tests"].items():
        print(f"\n{test_name}:")
        if "error" in test_result:
            print(f"- Error: {test_result['error']}")
        else:
            try:
                print(f"- p-value: {test_result.get('p_value', 'N/A'):.4f}")
                print(f"- Significant difference: {test_result.get('significant', 'N/A')}")
            except (TypeError, ValueError):
                print(f"- p-value: {test_result.get('p_value', 'N/A')}")
                print(f"- Significant difference: {test_result.get('significant', 'N/A')}")

    print("\nShape Comparison:")
    row_counts = results["shape_comparison"]["row_counts"]
    print(f"Row count difference: {row_counts['difference_pct']:.1f}%")

    # Print numerical column differences
    for col in numerical_cols:
        if col in results["shape_comparison"]:
            col_stats = results["shape_comparison"][col]
            if "error" in col_stats:
                print(f"\n{col} differences:")
                print(f"- Error: {col_stats['error']}")
            else:
                print(f"\n{col} differences:")
                print(f"- Mean: {col_stats['mean_difference_pct']:.1f}%")
                print(f"- Standard deviation: {col_stats['std_difference_pct']:.1f}%")
                print(f"- Median: {col_stats['quartile_differences']['q2_difference_pct']:.1f}%")

    # Print anomalies
    print("\nAnomalies detected in dataset A compared to B:")
    for anomaly_type, anomalies in results["anomalies"].items():
        if anomalies:
            print(f"\n{anomaly_type}:")
            for anomaly in anomalies[:5]:  # Show first 5 anomalies of each type
                print(f"- {anomaly}")
