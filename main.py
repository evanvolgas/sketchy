from sketchy import AnomalyDetector, analyze_and_print_results
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    np.random.seed(42)

    # Load and preprocess data
    base_data = pd.read_csv("~/Desktop/OnlineRetail.csv")
    test_data = pd.read_csv("~/Desktop/different.csv")

    # Print country distributions first
    print("\nBase data countries:")
    print(base_data["Country"].value_counts())
    print("\nTest data countries:")
    print(test_data["Country"].value_counts())

    # Find common countries
    common_countries = set(base_data["Country"]) & set(test_data["Country"])
    print(f"\nCommon countries: {len(common_countries)}")
    print(f"Countries only in base: {set(base_data['Country']) - set(test_data['Country'])}")
    print(f"Countries only in test: {set(test_data['Country']) - set(base_data['Country'])}")

    # Filter to only include common countries
    base_data = base_data[base_data["Country"].isin(common_countries)]
    test_data = test_data[test_data["Country"].isin(common_countries)]

    def preprocess_data(df):
        df = df.copy()
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["Hour"] = df["InvoiceDate"].dt.hour
        df["Minute"] = df["InvoiceDate"].dt.minute
        df["TotalValue"] = df["Quantity"] * df["UnitPrice"]
        df["ItemsPerInvoice"] = df.groupby("InvoiceNo")["Quantity"].transform("sum")
        df["UniqueItemsPerInvoice"] = df.groupby("InvoiceNo")["StockCode"].transform("nunique")
        df["InvoiceValue"] = df.groupby("InvoiceNo")["TotalValue"].transform("sum")
        return df

    base_data = preprocess_data(base_data)
    test_data = preprocess_data(test_data)

    feature_columns = [
        "Quantity",
        "UnitPrice",
        "TotalValue",
        "ItemsPerInvoice",
        "UniqueItemsPerInvoice",
        "InvoiceValue",
        "Hour",
        "Minute",
    ]

    # Scale features for anomaly detection
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

    # Get anomaly predictions and scores
    predictions, if_scores, ee_scores = detector.predict(test_scaled)
    anomaly_indices = np.where(predictions == -1)[0]

    print("\nPoint-level Anomalies (showing first 5):")
    for idx in anomaly_indices[:5]:
        anomaly_row = test_data.iloc[idx]
        print(f"\nAnomalous transaction {idx}:")
        print(f"Invoice: {anomaly_row['InvoiceNo']}")
        print(f"Country: {anomaly_row['Country']}")
        print(f"Time: {anomaly_row['InvoiceDate']}")
        print(f"StockCode: {anomaly_row['StockCode']}")
        print(f"Description: {anomaly_row['Description']}")
        print("Key metrics:")
        for col in feature_columns:
            print(f"- {col}: {anomaly_row[col]:.2f}")
        if if_scores is not None:
            print(f"Isolation Forest Score: {if_scores[idx]:.2f}")
        if ee_scores is not None:
            print(f"Elliptic Envelope Score: {ee_scores[idx]:.2f}")

    print("\nAnalyzing distribution results...")
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
