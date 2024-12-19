from sketchy import AnomalyDetector, analyze_and_print_results
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    np.random.seed(42)

    # Load data and print initial counts by country
    base_data = pd.read_csv("~/Desktop/OnlineRetail.csv")
    test_data = pd.read_csv("~/Desktop/test.csv")

    print("\nBase data country distribution:")
    print(base_data["Country"].value_counts())
    print("\nTest data country distribution:")
    print(test_data["Country"].value_counts())

    def add_business_metrics(df):
        """Add business-relevant metrics"""
        df = df.copy()

        # Transaction-based metrics
        df["ItemsPerInvoice"] = df.groupby("InvoiceNo")["Quantity"].transform("sum")
        df["UniqueItemsPerInvoice"] = df.groupby("InvoiceNo")["StockCode"].transform("nunique")
        df["InvoiceValue"] = df.groupby("InvoiceNo")["TotalValue"].transform("sum")

        # Product-based metrics
        df["ProductOrderCount"] = df.groupby("StockCode")["InvoiceNo"].transform("nunique")
        df["ProductTotalQuantity"] = df.groupby("StockCode")["Quantity"].transform("sum")

        # Price-quantity relationships
        epsilon = 1e-10
        df["Price_Quantity_Ratio"] = df["UnitPrice"] / (df["Quantity"].abs() + epsilon)
        df["Value_Density"] = df["TotalValue"] / (df["ItemsPerInvoice"] + epsilon)

        return df

    def preprocess_data(df):
        print(f"\nPreprocessing {len(df)} records...")
        df = df.copy()

        # Convert InvoiceDate to datetime
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

        # Basic time features
        df["Hour"] = df["InvoiceDate"].dt.hour
        df["Minute"] = df["InvoiceDate"].dt.minute

        # Calculate total value (absolute value for negative quantities)
        df["TotalValue"] = df["Quantity"].abs() * df["UnitPrice"]

        # Add business metrics
        df = add_business_metrics(df)

        # Handle negative values before log transform
        log_columns = [
            "Quantity",
            "UnitPrice",
            "TotalValue",
            "ItemsPerInvoice",
            "ProductOrderCount",
            "ProductTotalQuantity",
        ]

        for col in log_columns:
            # Take log of absolute value and preserve sign
            df[f"Log{col}"] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

        return df

    base_data = preprocess_data(base_data)
    test_data = preprocess_data(test_data)

    # Find common countries between datasets
    common_countries = set(base_data["Country"].unique()) & set(test_data["Country"].unique())
    print(f"\nCommon countries between datasets: {len(common_countries)}")
    print(
        "Countries only in base data:",
        set(base_data["Country"].unique()) - set(test_data["Country"].unique()),
    )
    print(
        "Countries only in test data:",
        set(test_data["Country"].unique()) - set(base_data["Country"].unique()),
    )

    # Filter to only common countries
    base_data = base_data[base_data["Country"].isin(common_countries)]
    test_data = test_data[test_data["Country"].isin(common_countries)]

    # Select features for anomaly detection
    feature_columns = [
        # Raw metrics
        "Quantity",
        "UnitPrice",
        "TotalValue",
        "ItemsPerInvoice",
        "UniqueItemsPerInvoice",
        "InvoiceValue",
        "ProductOrderCount",
        "ProductTotalQuantity",
        "Price_Quantity_Ratio",
        "Value_Density",
        # Log-transformed metrics
        "LogQuantity",
        "LogUnitPrice",
        "LogTotalValue",
        "LogItemsPerInvoice",
        "LogProductOrderCount",
        "LogProductTotalQuantity",
        # Time features
        "Hour",
        "Minute",
    ]

    # Scale features
    scaler = RobustScaler()
    base_scaled = pd.DataFrame(
        scaler.fit_transform(base_data[feature_columns]),
        columns=feature_columns,
        index=base_data.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_data[feature_columns]), columns=feature_columns, index=test_data.index
    )

    # Add back grouping columns
    base_scaled["Country"] = base_data["Country"]
    test_scaled["Country"] = test_data["Country"]

    detector = AnomalyDetector(
        method="both", dim_reduction="pca", n_components=2, contamination=0.1, use_gpu=True
    )

    print("\nFitting anomaly detector...")
    detector.fit(base_scaled[feature_columns])

    print("\nAnalyzing results...")
    analyze_and_print_results(
        base_data=base_scaled,
        test_data=test_scaled,
        detector=detector,
        numeric_columns=feature_columns,
        group_by="Country",
        threshold=10.0,
    )


if __name__ == "__main__":
    main()
