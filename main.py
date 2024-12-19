from sketchy import AnomalyDetector, analyze_and_print_results
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    np.random.seed(42)

    # Create baseline data
    base_data = pd.read_csv("~/Desktop/OnlineRetail.csv")
    test_data = pd.read_csv("~/Desktop/test.csv")

    # Print column names to see what's available
    print("Base data columns:", base_data.columns.tolist())
    print("\nTest data columns:", test_data.columns.tolist())

    # Before proceeding, verify columns exist and select appropriate numeric columns
    numeric_columns = base_data.select_dtypes(include=[np.number]).columns.tolist()
    print("\nAvailable numeric columns:", numeric_columns)

    detector = AnomalyDetector(
        method="isolation_forest",
        dim_reduction="pca",
        n_components=2,
        contamination=0.1,
        use_gpu=True,
    )

    detector.fit(base_data[numeric_columns])

    analyze_and_print_results(
        base_data=base_data,
        test_data=test_data,
        detector=detector,
        numeric_columns=numeric_columns,
        group_by="Country",  # Make sure this column exists too
        threshold=10.0,
    )


if __name__ == "__main__":
    main()
