from sketchy import AnomalyDetector, analyze_and_print_results
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    np.random.seed(42)

    # Load data and preprocess
    base_data = pd.read_csv("~/Desktop/OnlineRetail.csv")
    test_data = pd.read_csv("~/Desktop/different.csv")

    def add_business_metrics(df):
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
        df = df.copy()

        # Convert InvoiceDate to datetime
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        # Basic time features
        df["Hour"] = df["InvoiceDate"].dt.hour
        df["Minute"] = df["InvoiceDate"].dt.minute

        # Calculate total value
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

    # Define features for anomaly detection
    feature_columns = [
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
        "LogQuantity",
        "LogUnitPrice",
        "LogTotalValue",
        "LogItemsPerInvoice",
        "LogProductOrderCount",
        "LogProductTotalQuantity",
        "Hour",
        "Minute",
    ]

    # Scale features for anomaly detection only
    scaler = RobustScaler()
    base_scaled = scaler.fit_transform(base_data[feature_columns])
    test_scaled = scaler.transform(test_data[feature_columns])

    detector = AnomalyDetector(
        method="isolation_forest",
        dim_reduction="pca",
        n_components=2,
        contamination=0.1,
        use_gpu=True,
    )

    print("\nFitting anomaly detector...")
    detector.fit(base_scaled)

    print("\nAnalyzing results...")
    # Use unscaled data for analysis
    analyze_and_print_results(
        base_data=base_data,
        test_data=test_data,
        detector=detector,
        numeric_columns=feature_columns,
        group_by="Country",
        threshold=10.0,
    )


if __name__ == "__main__":
    main()
