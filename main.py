import numpy as np
import pandas as pd
from sketchy import AnomalyDetector, DatasetComparator
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any
import concurrent.futures


# FILE = "~/Desktop/test.csv"
FILE = "~/Desktop/OnlineRetail.csv"


def prepare_embeddings(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """
    Optimize categorical encoding using vectorized operations and memory efficiency.
    """
    # Create a copy only of the required columns to reduce memory usage
    needed_cols = categorical_cols + df.select_dtypes(include=np.number).columns.tolist()
    df_encoded = df[needed_cols].copy()

    # Vectorized encoding using dictionary mapping
    encodings = {}
    for col in categorical_cols:
        # Use categorical dtype to reduce memory usage
        df_encoded[col] = df_encoded[col].astype("category")
        encodings[f"{col}_encoded"] = df_encoded[col].cat.codes

    # Bulk assignment of encoded columns
    df_encoded = df_encoded.assign(**encodings)
    return df_encoded


def process_anomalies(anomalies: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Process anomalies in parallel for faster processing of large datasets.
    """
    processed_anomalies = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for anomaly_type, items in anomalies.items():
            if isinstance(items, list):
                future = executor.submit(lambda x: [str(item) for item in x[:5]], items)
                futures.append((anomaly_type, future))

        processed_anomalies = {anomaly_type: future.result() for anomaly_type, future in futures}

    return processed_anomalies


def analyze_numerical_column(col: str, stats: Dict[str, Any]) -> Dict[str, str]:
    """
    Analyze a single numerical column's statistics.
    """
    if "error" in stats:
        return {"column": col, "error": f"Error: {stats['error']}"}

    return {
        "column": col,
        "stats": {
            "mean": f"{stats['mean_difference_pct']:.1f}%",
            "std": f"{stats['std_difference_pct']:.1f}%",
            "median": f"{stats['quartile_differences']['q2_difference_pct']:.1f}%",
        },
    }


def main():
    # Load data more efficiently with specific dtypes
    dtypes = {
        # Add your column dtypes here for faster loading
        # example: 'column_name': 'category' or 'float32'
    }

    df = pd.read_csv(
        FILE,
        sep=",",
        dtype=dtypes,
        na_values=["nan", "NULL", ""],  # Specify known NA values
        low_memory=False,
    )

    # Optimize memory usage
    df = df.replace({np.nan: 0}).copy()

    # Identify column types once
    categorical_cols = df.select_dtypes("object").columns.tolist()
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Split datasets
    dataset_a, dataset_b = train_test_split(df, test_size=0.5, random_state=42)
    del df  # Free memory

    # Prepare embeddings in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(prepare_embeddings, dataset, categorical_cols)
            for dataset in (dataset_a, dataset_b)
        ]
        dataset_a_encoded, dataset_b_encoded = [f.result() for f in futures]

    # Define embedding columns
    embedding_cols = numerical_cols + [f"{col}_encoded" for col in categorical_cols]

    print("Part 1: Using AnomalyDetector")
    print("-" * 50)

    # Initialize and fit detector
    detector = AnomalyDetector(categorical_cols=categorical_cols, numerical_cols=numerical_cols)
    detector.fit(dataset_a_encoded, embedding_cols=embedding_cols, contamination=0.1)

    # Detect anomalies
    anomalies = detector.detect_anomalies(
        dataset_b_encoded, embedding_cols=embedding_cols, threshold=0.1
    )

    # Process and print anomalies efficiently
    processed_anomalies = process_anomalies(anomalies)
    for category, items in processed_anomalies.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for item in items:
            print(f"- {item}")

    print("\nPart 2: Using DatasetComparator")
    print("-" * 50)

    # Initialize and compare
    comparator = DatasetComparator(categorical_cols=categorical_cols, numerical_cols=numerical_cols)
    results = comparator.compare_datasets(
        dataset_a_encoded, dataset_b_encoded, embedding_cols=embedding_cols
    )

    # Print statistical tests
    print("\nDistribution Differences (Statistical Tests):")
    for test_name, test_result in results["statistical_tests"].items():
        print(f"\n{test_name}:")
        if "error" in test_result:
            print(f"- Error: {test_result['error']}")
        else:
            try:
                p_value = test_result.get("p_value", "N/A")
                print(f"- p-value: {p_value:.4f if isinstance(p_value, float) else p_value}")
                print(f"- Significant difference: {test_result.get('significant', 'N/A')}")
            except (TypeError, ValueError):
                print(f"- p-value: {test_result.get('p_value', 'N/A')}")
                print(f"- Significant difference: {test_result.get('significant', 'N/A')}")

    # Print shape comparison
    print("\nShape Comparison:")
    row_counts = results["shape_comparison"]["row_counts"]
    print(f"Row count difference: {row_counts['difference_pct']:.1f}%")

    # Process numerical columns in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                analyze_numerical_column,
                col,
                results["shape_comparison"].get(col, {"error": "No data"}),
            )
            for col in numerical_cols
        ]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f"\n{result['column']} differences:")
            if "error" in result:
                print(result["error"])
            else:
                for metric, value in result["stats"].items():
                    print(f"- {metric.title()}: {value}")

    # Print anomalies
    print("\nAnomalies detected in dataset A compared to B:")
    processed_anomalies = process_anomalies(results["anomalies"])
    for anomaly_type, anomalies in processed_anomalies.items():
        if anomalies:
            print(f"\n{anomaly_type}:")
            for anomaly in anomalies:
                print(f"- {anomaly}")


if __name__ == "__main__":
    main()
