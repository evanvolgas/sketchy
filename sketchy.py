from dataclasses import dataclass
import pandas as pd
import numpy as np
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
import umap
from typing import Optional, Dict, List, Tuple, Union
from functools import lru_cache
from sklearn.utils import resample
import concurrent.futures
import warnings
from contextlib import contextmanager


@dataclass
class NumericalStats:
    median: float
    mean: float
    std: float
    min: float
    max: float
    distribution_params: Tuple[float, float]


@dataclass
class GroupStats:
    n_rows: int
    numerical: Dict[str, NumericalStats]


class AnomalyDetector:
    def __init__(
        self,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
        device: str = "auto",
        n_jobs: Optional[int] = -1,
    ):
        """
        Initialize the anomaly detector with GPU support and parallel processing.

        Parameters:
        -----------
        categorical_cols : list, optional
            Columns to treat as categorical
        numerical_cols : list, optional
            Columns to treat as numerical
        device : str
            Device for torch computations: 'auto', 'cpu', 'cuda', or 'mps'
        n_jobs : int or None
            Number of parallel jobs (-1 for all cores, None for system default)
        """
        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []

        # Fix n_jobs handling
        self.n_jobs = None if n_jobs is None or n_jobs <= 0 else n_jobs

        # Initialize models and stats
        self.group_stats: Dict[str, GroupStats] = {}
        self.global_stats: Dict = {}
        self.pca: Optional[PCA] = None
        self.umap_reducer: Optional[umap.UMAP] = None
        self.isolation_forest: Optional[IsolationForest] = None
        self.elliptic_envelope: Optional[EllipticEnvelope] = None
        self.scaler: Optional[StandardScaler] = None

        # Set up device for torch computations
        self.device = self._get_device(device)

        # Set up thread pool for parallel processing
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs)

    def _get_device(self, device: str) -> str:
        """Determine the appropriate compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    @contextmanager
    def _device_context(self, data: np.ndarray) -> torch.Tensor:
        """Context manager for handling device-specific tensor operations."""
        try:
            tensor = torch.tensor(data, device=self.device, dtype=torch.float32)
            yield tensor
        finally:
            if self.device != "cpu":
                torch.cuda.empty_cache() if self.device == "cuda" else None

    def _parallel_compute_stats(self, df: pd.DataFrame, col: str) -> NumericalStats:
        """Compute numerical statistics in parallel."""
        series = df[col]
        return NumericalStats(
            median=series.median(),
            mean=series.mean(),
            std=series.std(),
            min=series.min(),
            max=series.max(),
            distribution_params=stats.norm.fit(series),
        )

    def _vectorized_compute_global_stats(self, df: pd.DataFrame) -> None:
        """Compute global statistics using vectorized operations."""
        numerical_stats = {}

        # Compute numerical stats in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_col = {
                executor.submit(self._parallel_compute_stats, df, col): col
                for col in self.numerical_cols
            }

            for future in concurrent.futures.as_completed(future_to_col):
                col = future_to_col[future]
                numerical_stats[col] = future.result()

        # Vectorized computation for categorical columns
        categorical_stats = {
            col: df[col].value_counts(normalize=True).to_dict() for col in self.categorical_cols
        }

        self.global_stats = {
            "n_rows": len(df),
            "numerical": numerical_stats,
            "categorical": categorical_stats,
        }

    @lru_cache(maxsize=128)
    def _cached_stats_computation(self, data_key: bytes) -> Dict:
        """Cache statistical computations for repeated operations."""
        data = np.frombuffer(data_key, dtype=np.float64)
        return {
            "median": float(np.median(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
        }

    def fit(
        self,
        df: pd.DataFrame,
        embedding_cols: Optional[List[str]] = None,
        contamination: float = 0.1,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        dim_reduction: str = "pca+umap",
        n_components: int = 2,
        pca_components: Optional[int] = None,
        max_samples: int = 10000,
        batch_size: int = 1000,
    ) -> None:
        """
        Fit the anomaly detector with performance optimizations.

        New Parameters:
        --------------
        batch_size : int
            Size of batches for processing large datasets
        """
        # Sample large datasets
        if len(df) > max_samples:
            df = resample(df, n_samples=max_samples, random_state=42)

        # Compute statistics in parallel
        self._vectorized_compute_global_stats(df)

        if embedding_cols:
            # Initialize scaler and transform data
            self.scaler = StandardScaler()
            embedding_data = self.scaler.fit_transform(df[embedding_cols])

            # Process embedding data in batches for GPU
            with self._device_context(embedding_data) as tensor_data:
                if dim_reduction in ["pca+umap", "pca"]:
                    # Perform PCA on GPU if available
                    n_components = pca_components or min(tensor_data.shape)
                    self.pca = PCA(n_components=n_components, random_state=42)

                    # Process in batches for large datasets
                    pca_results = []
                    for i in range(0, len(tensor_data), batch_size):
                        batch = tensor_data[i : i + batch_size]
                        batch_result = self.pca.fit_transform(batch.cpu().numpy())
                        pca_results.append(batch_result)

                    embedding_data = np.vstack(pca_results)

                    print(
                        f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}"
                    )

                if dim_reduction in ["pca+umap", "umap"]:
                    self.umap_reducer = umap.UMAP(
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=n_components,
                        metric="cosine",
                        random_state=42,
                    )
                    embedding_data = self.umap_reducer.fit_transform(embedding_data)

            # Fit outlier detection models
            self.isolation_forest = IsolationForest(
                contamination=contamination, random_state=42, n_jobs=self.n_jobs
            ).fit(embedding_data)

            self.elliptic_envelope = EllipticEnvelope(
                contamination=contamination, random_state=42
            ).fit(embedding_data)

    def _batch_transform(self, data: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """Transform data in batches to handle memory constraints."""
        results = []
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            if self.pca is not None:
                batch = self.pca.transform(batch)
            if self.umap_reducer is not None:
                batch = self.umap_reducer.transform(batch)
            results.append(batch)
        return np.vstack(results)

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        embedding_cols: Optional[List[str]] = None,
        threshold: float = 0.1,
        batch_size: int = 1000,
    ) -> Dict:
        """
        Detect anomalies using parallel processing and GPU acceleration.

        Parameters:
        -----------
        df : pd.DataFrame
            Data to analyze
        embedding_cols : list, optional
            Columns to use for embedding analysis
        threshold : float
            Threshold for considering something anomalous
        batch_size : int
            Size of batches for processing large datasets

        Returns:
        --------
        dict
            Dictionary containing different types of detected anomalies
        """
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit tasks for parallel execution
            futures.append(executor.submit(self._check_global_distribution, df))
            futures.append(executor.submit(self._check_group_distribution, df, threshold))
            if embedding_cols:
                futures.append(
                    executor.submit(self._check_point_anomalies, df, embedding_cols, batch_size)
                )

        # Collect results
        results = [future.result() for future in futures]

        return {
            "global_distribution": results[0],
            "group_distribution": results[1],
            "point_anomalies": results[2] if embedding_cols else [],
        }

    def _check_global_distribution(self, df: pd.DataFrame) -> List[str]:
        """Check for global distribution anomalies using vectorized operations."""
        anomalies = []

        # Check row count
        row_diff = abs(len(df) - self.global_stats["n_rows"]) / self.global_stats["n_rows"]
        if row_diff > 0.2:
            anomalies.append(
                f"Unusual number of records: {len(df)} vs baseline {self.global_stats['n_rows']}"
            )

        # Check numerical distributions using vectorized operations
        for col in self.numerical_cols:
            curr_stats = self._parallel_compute_stats(df, col)
            baseline_stats = self.global_stats["numerical"][col]

            # Vectorized comparison
            if abs((curr_stats.median - baseline_stats.median) / baseline_stats.median) > 0.1:
                anomalies.append(
                    f"Unusual {col} median: {curr_stats.median:.2f} vs baseline {baseline_stats.median:.2f}"
                )

        # Check categorical distributions using vectorized operations
        for col in self.categorical_cols:
            curr_dist = df[col].value_counts(normalize=True)
            baseline_dist = pd.Series(self.global_stats["categorical"][col])

            # Align distributions
            aligned_curr = curr_dist.reindex(baseline_dist.index, fill_value=0)

            try:
                # Vectorized chi-square test
                chi2, p_value = stats.chisquare(aligned_curr, baseline_dist)
                if p_value < 0.05:
                    anomalies.append(f"Unusual distribution in {col} (p-value: {p_value:.4f})")
            except ValueError as e:
                anomalies.append(f"Unable to compare distributions for {col}: {str(e)}")

        return anomalies

    def _check_point_anomalies(
        self, df: pd.DataFrame, embedding_cols: List[str], batch_size: int = 1000
    ) -> List[str]:
        """Detect point-level anomalies using GPU acceleration."""
        if not (self.pca is not None or self.umap_reducer is not None) or not embedding_cols:
            return []

        try:
            # Transform data in batches
            embedding_data = self.scaler.transform(df[embedding_cols])
            transformed_data = self._batch_transform(embedding_data, batch_size)

            # Get anomaly scores using GPU if available
            with self._device_context(transformed_data) as tensor_data:
                if_scores = self.isolation_forest.score_samples(tensor_data.cpu().numpy())
                ee_scores = self.elliptic_envelope.score_samples(tensor_data.cpu().numpy())

            # Vectorized anomaly detection
            anomaly_mask = (if_scores < -0.5) & (ee_scores < -3)
            anomaly_indices = np.where(anomaly_mask)[0]

            return [
                f"Row {idx}: Unusual point in embedding space "
                f"(IF score: {if_scores[idx]:.2f}, EE score: {ee_scores[idx]:.2f})"
                for idx in anomaly_indices
            ]

        except Exception as e:
            return [f"Error detecting point anomalies: {str(e)}"]

    def _check_group_distribution(self, df: pd.DataFrame, threshold: float) -> List[str]:
        """Check for group-level distribution anomalies using parallel processing."""
        anomalies = []

        def process_group(group_key: str, baseline_stats: GroupStats) -> List[str]:
            group_anomalies = []
            conditions = group_key.split("_")
            group_df = df.copy()

            # Apply group conditions
            for col, val in zip(self.categorical_cols[: len(conditions)], conditions):
                group_df = group_df[group_df[col] == val]

            if len(group_df) == 0:
                return group_anomalies

            # Check numerical variables using vectorized operations
            for col in self.numerical_cols:
                curr_stats = self._parallel_compute_stats(group_df, col)
                baseline_median = baseline_stats.numerical[col].median

                if baseline_median != 0:
                    pct_change = (curr_stats.median - baseline_median) / baseline_median
                    if abs(pct_change) > threshold:
                        group_anomalies.append(
                            f"Group {group_key}: Unusual {col} median: "
                            f"{curr_stats.median:.2f} vs baseline {baseline_median:.2f}"
                        )

            return group_anomalies

        # Process groups in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_group = {
                executor.submit(process_group, group_key, baseline_stats): group_key
                for group_key, baseline_stats in self.group_stats.items()
            }

            for future in concurrent.futures.as_completed(future_to_group):
                anomalies.extend(future.result())

        return anomalies


from dataclasses import dataclass, field  # Added field import


@dataclass
class DatasetComparator:
    """
    A class to compare and analyze datasets with optimized performance.
    Supports GPU acceleration, vectorized operations, and sampling for large datasets.
    """

    categorical_cols: Optional[List[str]] = field(default_factory=list)
    numerical_cols: Optional[List[str]] = field(default_factory=list)
    sample_size: int = 100000
    use_gpu: bool = True
    n_jobs: Optional[int] = -1

    def __post_init__(self):
        # Ensure lists are initialized properly
        self.categorical_cols = self.categorical_cols or []
        self.numerical_cols = self.numerical_cols or []

        # Fix n_jobs handling
        self.n_jobs = None if self.n_jobs is None or self.n_jobs <= 0 else self.n_jobs

        # Set device
        self.device = self._get_device()

        # Initialize detector with safe parameters
        self.detector = AnomalyDetector(
            categorical_cols=self.categorical_cols,
            numerical_cols=self.numerical_cols,
            device=self.device,
            n_jobs=self.n_jobs,
        )

        # Initialize thread pool with safe number of workers
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs)

    def _get_device(self) -> str:
        """Determine the appropriate compute device."""
        if self.use_gpu:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    @contextmanager
    def _device_context(self, data: np.ndarray) -> torch.Tensor:
        """Context manager for handling device-specific tensor operations."""
        try:
            tensor = torch.tensor(data, device=self.device, dtype=torch.float32)
            yield tensor
        finally:
            if self.device != "cpu":
                torch.cuda.empty_cache() if self.device == "cuda" else None

    def compare_datasets(
        self,
        dataset_a: pd.DataFrame,
        dataset_b: pd.DataFrame,
        embedding_cols: Optional[List[str]] = None,
        threshold: float = 0.1,
    ) -> Dict:
        """
        Compare two datasets for distributional differences.

        Parameters:
        -----------
        dataset_a : pd.DataFrame
            First dataset
        dataset_b : pd.DataFrame
            Second dataset (baseline)
        embedding_cols : list
            Columns to use for dimensional analysis
        threshold : float
            Threshold for considering differences significant

        Returns:
        --------
        dict
            Dictionary containing comparison results
        """
        # Fit detector on dataset B (baseline)
        self.detector.fit(dataset_b, embedding_cols=embedding_cols)

        # Get anomalies in dataset A compared to B
        anomalies = self.detector.detect_anomalies(
            dataset_a, embedding_cols=embedding_cols, threshold=threshold
        )

        # Perform additional statistical tests in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_stats = executor.submit(self._perform_statistical_tests, dataset_a, dataset_b)
            future_shapes = executor.submit(self._compare_shapes, dataset_a, dataset_b)

            statistical_tests = future_stats.result()
            shape_comparison = future_shapes.result()

        return {
            "anomalies": anomalies,
            "statistical_tests": statistical_tests,
            "shape_comparison": shape_comparison,
        }

    def _perform_statistical_tests(self, dataset_a: pd.DataFrame, dataset_b: pd.DataFrame) -> Dict:
        """Perform statistical tests to compare distributions."""
        results = {}

        def process_numerical_column(col: str) -> Dict:
            try:
                # Use GPU if available for numerical computations
                if self.device != "cpu":
                    with self._device_context(
                        dataset_a[col].fillna(0).values
                    ) as values_a, self._device_context(
                        dataset_b[col].fillna(0).values
                    ) as values_b:
                        values_a = values_a.cpu().numpy()
                        values_b = values_b.cpu().numpy()
                else:
                    values_a = dataset_a[col].fillna(0).values
                    values_b = dataset_b[col].fillna(0).values

                ks_statistic, p_value = stats.ks_2samp(values_a, values_b)
                return {
                    "statistic": ks_statistic,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                }
            except Exception as e:
                return {"error": f"Failed to perform KS test: {str(e)}"}

        def process_categorical_column(col: str) -> Dict:
            try:
                # Get value counts
                counts_a = dataset_a[col].value_counts()
                counts_b = dataset_b[col].value_counts()

                # Align categories
                all_categories = sorted(set(counts_a.index) | set(counts_b.index))
                counts_a = counts_a.reindex(all_categories, fill_value=0)
                counts_b = counts_b.reindex(all_categories, fill_value=0)

                # Normalize to get proportions
                props_a = counts_a / counts_a.sum()
                props_b = counts_b / counts_b.sum()

                # Chi-square test
                chi2, p_value = stats.chisquare(props_a, props_b)
                return {
                    "statistic": chi2,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                }
            except Exception as e:
                return {"error": f"Failed to perform chi-square test: {str(e)}"}

        # Process columns in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit numerical column tests
            future_to_num_col = {
                executor.submit(process_numerical_column, col): col for col in self.numerical_cols
            }

            # Submit categorical column tests
            future_to_cat_col = {
                executor.submit(process_categorical_column, col): col
                for col in self.categorical_cols
            }

            # Collect numerical results
            for future in concurrent.futures.as_completed(future_to_num_col):
                col = future_to_num_col[future]
                results[f"{col}_ks_test"] = future.result()

            # Collect categorical results
            for future in concurrent.futures.as_completed(future_to_cat_col):
                col = future_to_cat_col[future]
                results[f"{col}_chi2_test"] = future.result()

        return results

    def _compare_shapes(self, dataset_a: pd.DataFrame, dataset_b: pd.DataFrame) -> Dict:
        """Compare overall shapes and characteristics of datasets."""
        shape_info = {}

        # Compare basic shapes
        shape_info["row_counts"] = {
            "dataset_a": len(dataset_a),
            "dataset_b": len(dataset_b),
            "difference_pct": (
                (len(dataset_a) - len(dataset_b)) / len(dataset_b) * 100
                if len(dataset_b) > 0
                else float("inf")
            ),
        }

        def process_numerical_column(col: str) -> Dict:
            try:
                # Use GPU if available for numerical computations
                if self.device != "cpu":
                    with self._device_context(
                        dataset_a[col].values
                    ) as values_a, self._device_context(dataset_b[col].values) as values_b:
                        stats_a = {
                            "mean": float(torch.mean(values_a)),
                            "std": float(torch.std(values_a)),
                            "min": float(torch.min(values_a)),
                            "max": float(torch.max(values_a)),
                            "25%": float(torch.quantile(values_a, 0.25)),
                            "50%": float(torch.quantile(values_a, 0.50)),
                            "75%": float(torch.quantile(values_a, 0.75)),
                        }
                        stats_b = {
                            "mean": float(torch.mean(values_b)),
                            "std": float(torch.std(values_b)),
                            "min": float(torch.min(values_b)),
                            "max": float(torch.max(values_b)),
                            "25%": float(torch.quantile(values_b, 0.25)),
                            "50%": float(torch.quantile(values_b, 0.50)),
                            "75%": float(torch.quantile(values_b, 0.75)),
                        }
                else:
                    stats_a = dataset_a[col].describe()
                    stats_b = dataset_b[col].describe()

                return {
                    "mean_difference_pct": (
                        (stats_a["mean"] - stats_b["mean"]) / stats_b["mean"] * 100
                        if stats_b["mean"] != 0
                        else float("inf")
                    ),
                    "std_difference_pct": (
                        (stats_a["std"] - stats_b["std"]) / stats_b["std"] * 100
                        if stats_b["std"] != 0
                        else float("inf")
                    ),
                    "range_difference": {
                        "min_difference_pct": (
                            (stats_a["min"] - stats_b["min"]) / stats_b["min"] * 100
                            if stats_b["min"] != 0
                            else float("inf")
                        ),
                        "max_difference_pct": (
                            (stats_a["max"] - stats_b["max"]) / stats_b["max"] * 100
                            if stats_b["max"] != 0
                            else float("inf")
                        ),
                    },
                    "quartile_differences": {
                        "q1_difference_pct": (
                            (stats_a["25%"] - stats_b["25%"]) / stats_b["25%"] * 100
                            if stats_b["25%"] != 0
                            else float("inf")
                        ),
                        "q2_difference_pct": (
                            (stats_a["50%"] - stats_b["50%"]) / stats_b["50%"] * 100
                            if stats_b["50%"] != 0
                            else float("inf")
                        ),
                        "q3_difference_pct": (
                            (stats_a["75%"] - stats_b["75%"]) / stats_b["75%"] * 100
                            if stats_b["75%"] != 0
                            else float("inf")
                        ),
                    },
                }
            except Exception as e:
                return {"error": f"Failed to compare shapes: {str(e)}"}

        # Process numerical columns in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_col = {
                executor.submit(process_numerical_column, col): col for col in self.numerical_cols
            }

            for future in concurrent.futures.as_completed(future_to_col):
                col = future_to_col[future]
                shape_info[col] = future.result()

        return shape_info
