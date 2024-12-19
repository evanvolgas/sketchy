import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
import umap
from typing import Optional, Union, Literal, Dict, List, Tuple  # Added Tuple here
import multiprocessing
from dataclasses import dataclass
import torch
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DistributionMetrics:
    """Stores distribution metrics for a dataset or subset"""

    median: float
    mean: float
    std: float
    count: int
    min: float
    max: float


class AnomalyDetector:
    def __init__(
        self,
        method: Literal["isolation_forest", "elliptic_envelope", "both"] = "isolation_forest",
        dim_reduction: Optional[Literal["pca", "umap", "pca+umap"]] = None,
        n_components: int = 2,
        random_state: int = 42,
        n_jobs: int = -1,
        contamination: float = 0.1,
        use_gpu: bool = False,
    ):
        """
        Initialize the anomaly detector

        Args:
            method: Detection method to use
            dim_reduction: Dimensionality reduction method
            n_components: Number of components for dim reduction
            random_state: Random state for reproducibility
            n_jobs: Number of jobs for parallel processing (-1 for all cores)
            contamination: Expected proportion of outliers in the dataset
            use_gpu: Whether to use GPU acceleration when available
        """
        self.method = method
        self.dim_reduction = dim_reduction
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        self.contamination = contamination
        self.use_gpu = use_gpu

        # Initialize models
        self.isolation_forest = None
        self.elliptic_envelope = None
        self.pca = None
        self.umap_reducer = None

        # Device selection for GPU operations
        if self.use_gpu:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = "cpu"

        logger.info(f"Using device: {self.device}")

    def _prepare_data(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert input data to numpy array and handle missing values"""
        if isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include=[np.number]).to_numpy()
        return np.nan_to_num(X, nan=0)

    def _to_device(self, X: np.ndarray) -> Union[torch.Tensor, np.ndarray]:
        """Convert numpy array to tensor on appropriate device if GPU is enabled"""
        if self.use_gpu and self.device != "cpu":
            return torch.tensor(X, device=self.device, dtype=torch.float32)
        return X

    def _to_numpy(self, X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert tensor to numpy array if necessary"""
        if isinstance(X, torch.Tensor):
            return X.cpu().numpy()
        return X

    def _reduce_dimensions(self, X: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction if specified"""
        if self.dim_reduction is None:
            return X

        X = self._to_numpy(X)  # Ensure we're working with numpy array

        if "pca" in self.dim_reduction:
            if self.pca is None:
                self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
                X = self.pca.fit_transform(X)
            else:
                X = self.pca.transform(X)

        if "umap" in self.dim_reduction:
            if self.umap_reducer is None:
                self.umap_reducer = umap.UMAP(
                    n_components=self.n_components,
                    random_state=self.random_state,
                    n_jobs=1,  # Avoid warning about n_jobs with random_state
                )
                X = self.umap_reducer.fit_transform(X)
            else:
                X = self.umap_reducer.transform(X)

        return X

    def fit(self, X: Union[pd.DataFrame, np.ndarray]):
        """Fit the anomaly detection model(s)"""
        X = self._prepare_data(X)
        X = self._reduce_dimensions(X)
        X = self._to_numpy(X)  # Ensure numpy array for sklearn

        if self.method in ["isolation_forest", "both"]:
            self.isolation_forest = IsolationForest(
                n_jobs=self.n_jobs, random_state=self.random_state, contamination=self.contamination
            )
            self.isolation_forest.fit(X)

        if self.method in ["elliptic_envelope", "both"]:
            self.elliptic_envelope = EllipticEnvelope(
                random_state=self.random_state, contamination=self.contamination
            )
            self.elliptic_envelope.fit(X)

        return self

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict anomalies and return scores"""
        X = self._prepare_data(X)
        X = self._reduce_dimensions(X)
        X = self._to_numpy(X)

        if_scores = None
        ee_scores = None

        if self.method == "both":
            if_scores = self.isolation_forest.score_samples(X)
            ee_scores = self.elliptic_envelope.score_samples(X)
            results = np.where((if_scores < 0) | (ee_scores < 0), -1, 1)
        elif self.method == "isolation_forest":
            if_scores = self.isolation_forest.score_samples(X)
            results = np.where(if_scores < 0, -1, 1)
        else:
            ee_scores = self.elliptic_envelope.score_samples(X)
            results = np.where(ee_scores < 0, -1, 1)

        return results, if_scores, ee_scores


def process_csv_file(
    filepath: Union[str, Path],
    detector: AnomalyDetector,
    numeric_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Process a CSV file for anomaly detection

    Args:
        filepath: Path to CSV file
        detector: Fitted AnomalyDetector instance
        numeric_columns: List of numeric columns to use for detection

    Returns:
        DataFrame with anomaly predictions
    """
    try:
        # Read CSV in chunks for memory efficiency
        chunk_size = 10000
        chunks = []

        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            if numeric_columns is not None:
                chunk = chunk[numeric_columns]
            predictions = detector.predict(chunk)
            chunk["is_anomaly"] = predictions == -1
            chunks.append(chunk)

        return pd.concat(chunks)

    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")
        raise


def analyze_and_print_results(
    base_data: pd.DataFrame,
    test_data: pd.DataFrame,
    detector: AnomalyDetector,
    numeric_columns: List[str],
    group_by: Optional[str] = None,
    threshold: float = 10.0,
) -> None:
    """Analyze and print results in requested format"""

    def safe_pct_change(new_val: float, base_val: float) -> float:
        """Calculate percentage change safely handling zero values"""
        if base_val == 0:
            if new_val == 0:
                return 0
            # Return large value to indicate significant change from zero
            return 999.9 if new_val > 0 else -999.9
        return ((new_val - base_val) / base_val) * 100

    # Global distribution analysis
    print("Global Distribution Anomalies:")
    record_change = safe_pct_change(len(test_data), len(base_data))
    if abs(record_change) > threshold:
        print(f"- Unusual number of records: {len(test_data)} vs baseline {len(base_data)}")

    for col in numeric_columns:
        base_median = base_data[col].median()
        test_median = test_data[col].median()
        change = safe_pct_change(test_median, base_median)
        if abs(change) > threshold:
            print(f"- Unusual {col} median: {test_median:.2f} vs baseline {base_median:.2f}")

    # Group distribution analysis
    if group_by:
        print("\nGroup Distribution Anomalies:")
        for group in base_data[group_by].unique():
            base_group = base_data[base_data[group_by] == group]
            test_group = test_data[test_data[group_by] == group]

            for col in numeric_columns:
                base_median = base_group[col].median()
                test_median = test_group[col].median()
                change = safe_pct_change(test_median, base_median)
                if abs(change) > threshold:
                    print(
                        f"- Group {group}: Unusual {col} median: {test_median:.2f} vs baseline {base_median:.2f}"
                    )


def process_csv_file(
    filepath: Union[str, Path],
    detector: AnomalyDetector,
    numeric_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    try:
        chunk_size = 10000
        chunks = []

        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            if numeric_columns is not None:
                # Ensure we keep all columns but only use numeric ones for prediction
                predictions, _, _ = detector.predict(chunk[numeric_columns])
                chunk["is_anomaly"] = predictions == -1
            chunks.append(chunk)

        return pd.concat(chunks)

    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")
        raise
