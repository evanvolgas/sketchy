import concurrent.futures
import multiprocessing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import umap
from scipy import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")


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

        # Vectorized computation for categorical columns with proper normalization
        categorical_stats = {}
        for col in self.categorical_cols:
            try:
                value_counts = df[col].value_counts()
                proportions = value_counts / value_counts.sum()
                categorical_stats[col] = proportions.to_dict()
            except Exception as e:
                print(f"Warning: Failed to compute stats for {col}: {str(e)}")
                categorical_stats[col] = {}

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

        # Check categorical distributions
        for col in self.categorical_cols:
            try:
                curr_dist = df[col].value_counts(normalize=True)
                baseline_dist = pd.Series(self.global_stats["categorical"][col])

                # Get all unique categories
                all_categories = pd.Index(set(curr_dist.index) | set(baseline_dist.index))

                # Align and normalize distributions
                curr_dist = curr_dist.reindex(all_categories, fill_value=0)
                baseline_dist = baseline_dist.reindex(all_categories, fill_value=0)

                # Renormalize after alignment
                curr_dist = curr_dist / curr_dist.sum()
                baseline_dist = baseline_dist / baseline_dist.sum()

                # Calculate Jensen-Shannon divergence
                m = 0.5 * (curr_dist + baseline_dist)
                jsd = 0.5 * (
                    (curr_dist * np.log(curr_dist / m + 1e-10)).sum()
                    + (baseline_dist * np.log(baseline_dist / m + 1e-10)).sum()
                )

                if jsd > 0.1:  # Threshold can be adjusted
                    anomalies.append(f"Unusual distribution in {col} (JSD: {jsd:.4f})")

            except Exception as e:
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


@dataclass
class DatasetComparator:
    categorical_cols: Optional[List[str]] = field(default_factory=list)
    numerical_cols: Optional[List[str]] = field(default_factory=list)
    sample_size: int = 100000
    use_gpu: bool = True
    n_jobs: Optional[int] = -1

    def __post_init__(self):
        # Ensure lists are initialized properly
        self.categorical_cols = self.categorical_cols or []
        self.numerical_cols = self.numerical_cols or []

        # Optimize number of jobs based on CPU cores and task type
        if self.n_jobs is None or self.n_jobs <= 0:
            self.n_jobs = min(32, multiprocessing.cpu_count())  # Cap at 32 for stability

        # Set device
        self.device = self._get_device()

        # Initialize detector with optimized parameters
        self.detector = AnomalyDetector(
            categorical_cols=self.categorical_cols,
            numerical_cols=self.numerical_cols,
            device=self.device,
            n_jobs=self.n_jobs,
        )

        # Initialize separate pools for I/O and CPU-bound operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs)
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=max(1, self.n_jobs // 2)
        )

    def _get_device(self) -> str:
        """Determine the appropriate compute device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def fast_distribution_compare(
        self, values_a: np.ndarray, values_b: np.ndarray, n_bins: int = 100
    ) -> Dict:
        """
        Performs a fast distribution comparison using histogram binning and
        the Chi-square test. Much faster than KS test for large datasets.
        """
        # Convert to float32 for memory efficiency
        values_a = values_a.astype(np.float32)
        values_b = values_b.astype(np.float32)

        # Calculate range for binning
        min_val = min(np.min(values_a), np.min(values_b))
        max_val = max(np.max(values_a), np.max(values_b))

        # Create histograms with identical bins
        hist_a, bins = np.histogram(values_a, bins=n_bins, range=(min_val, max_val), density=True)
        hist_b, _ = np.histogram(values_b, bins=bins, density=True)

        # Add small constant to avoid division by zero
        hist_a = hist_a + 1e-10
        hist_b = hist_b + 1e-10

        # Calculate chi-square statistic
        chi2_stat = np.sum((hist_a - hist_b) ** 2 / (hist_a + hist_b))

        # Calculate degrees of freedom (number of bins - 1)
        df = n_bins - 1

        # Calculate p-value
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)

        return {
            "statistic": float(chi2_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
        }

    def _perform_statistical_tests(self, dataset_a: pd.DataFrame, dataset_b: pd.DataFrame) -> Dict:
        """Perform statistical tests to compare distributions."""
        results = {}

        # Pre-compute and cache value counts for categorical columns
        cat_counts_a = {
            col: dataset_a[col].fillna("MISSING").value_counts() for col in self.categorical_cols
        }
        cat_counts_b = {
            col: dataset_b[col].fillna("MISSING").value_counts() for col in self.categorical_cols
        }

        def process_categorical_column(col: str) -> Dict:
            try:
                counts_a = cat_counts_a[col]
                counts_b = cat_counts_b[col]

                # Get frequencies
                total_a = counts_a.sum()
                total_b = counts_b.sum()

                # Get all unique categories
                all_categories = pd.Index(set(counts_a.index) | set(counts_b.index))

                # Calculate frequencies with fill value 0
                freq_a = (counts_a.reindex(all_categories, fill_value=0) / total_a).values
                freq_b = (counts_b.reindex(all_categories, fill_value=0) / total_b).values

                # For high cardinality columns (like IDs), do a different comparison
                unique_ratio = len(all_categories) / max(total_a, total_b)

                if unique_ratio > 0.1:  # If more than 10% unique values
                    overlap_ratio = len(set(counts_a.index) & set(counts_b.index)) / len(
                        all_categories
                    )
                    return {
                        "type": "high_cardinality",
                        "unique_ratio": float(unique_ratio),
                        "category_overlap": float(overlap_ratio),
                        "n_categories": len(all_categories),
                        "n_categories_a": len(counts_a),
                        "n_categories_b": len(counts_b),
                        "significant": overlap_ratio
                        < 0.9,  # Consider significant if overlap is low
                    }

                # For normal categorical columns, use Jensen-Shannon divergence
                # This is more stable than chi-square for comparing distributions
                m = 0.5 * (freq_a + freq_b)
                jsd = 0.5 * (
                    np.sum(freq_a * np.log(freq_a / m + 1e-10))
                    + np.sum(freq_b * np.log(freq_b / m + 1e-10))
                )

                return {
                    "type": "categorical",
                    "distance": float(jsd),
                    "significant": jsd > 0.1,  # Threshold can be adjusted
                    "n_categories": len(all_categories),
                    "n_unique_a": len(counts_a),
                    "n_unique_b": len(counts_b),
                }

            except Exception as e:
                return {"error": f"Failed to compare distributions: {str(e)}"}

        def process_numerical_column(col: str) -> Dict:
            try:
                values_a = dataset_a[col].fillna(0).astype(np.float32).values
                values_b = dataset_b[col].fillna(0).astype(np.float32).values

                return self.fast_distribution_compare(values_a, values_b)

            except Exception as e:
                return {"error": f"Failed to perform distribution test: {str(e)}"}

        # Process all columns using a single thread pool
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                # Submit numerical columns
                num_futures = {
                    executor.submit(process_numerical_column, col): (col, "ks_test")
                    for col in self.numerical_cols
                }

                # Submit categorical columns
                cat_futures = {
                    executor.submit(process_categorical_column, col): (col, "chi2_test")
                    for col in self.categorical_cols
                }

                # Combine all futures
                all_futures = {**num_futures, **cat_futures}

                # Collect results
                for future in concurrent.futures.as_completed(all_futures):
                    col, test_type = all_futures[future]
                    try:
                        results[f"{col}_{test_type}"] = future.result()
                    except Exception as e:
                        results[f"{col}_{test_type}"] = {"error": str(e)}

        except Exception as e:
            print(f"Error in thread pool execution: {str(e)}")
            return {}

        return results  # Make sure this is returned

    def _compare_shapes(self, dataset_a: pd.DataFrame, dataset_b: pd.DataFrame) -> Dict:
        """Compare dataset shapes efficiently."""
        # Pre-compute memory usage stats
        memory_usage_a = dataset_a.memory_usage(deep=True).sum()
        memory_usage_b = dataset_b.memory_usage(deep=True).sum()

        # Calculate row differences
        row_diff = len(dataset_a) - len(dataset_b)
        row_diff_pct = (row_diff / len(dataset_b)) * 100 if len(dataset_b) > 0 else float("inf")

        return {
            "row_counts": {
                "dataset_a": len(dataset_a),
                "dataset_b": len(dataset_b),
                "difference": row_diff,
                "difference_pct": row_diff_pct,
            },
            "column_counts": {
                "dataset_a": len(dataset_a.columns),
                "dataset_b": len(dataset_b.columns),
                "difference": len(dataset_a.columns) - len(dataset_b.columns),
            },
            "memory_usage": {
                "dataset_a": memory_usage_a,
                "dataset_b": memory_usage_b,
                "ratio": memory_usage_a / memory_usage_b if memory_usage_b > 0 else float("inf"),
            },
        }

    def compare_datasets(
        self,
        dataset_a: pd.DataFrame,
        dataset_b: pd.DataFrame,
        embedding_cols: Optional[List[str]] = None,
        threshold: float = 0.1,
    ) -> Dict:
        """Compare datasets with optimized performance."""
        # Optimize memory usage by converting to efficient dtypes
        for col in self.numerical_cols:
            dataset_a[col] = dataset_a[col].astype(np.float32)
            dataset_b[col] = dataset_b[col].astype(np.float32)

        # Handle dataset size differences efficiently
        if len(dataset_a) > len(dataset_b) * 1.5:
            dataset_a = dataset_a.sample(n=len(dataset_b), random_state=42)
        elif len(dataset_b) > len(dataset_a) * 1.5:
            dataset_b = dataset_b.sample(n=len(dataset_a), random_state=42)

        # Fit detector on dataset B (baseline)
        self.detector.fit(dataset_b, embedding_cols=embedding_cols)

        # Use thread pool for parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit tasks
            future_anomalies = executor.submit(
                self.detector.detect_anomalies,
                dataset_a,
                embedding_cols=embedding_cols,
                threshold=threshold,
            )
            future_stats = executor.submit(self._perform_statistical_tests, dataset_a, dataset_b)
            future_shapes = executor.submit(self._compare_shapes, dataset_a, dataset_b)

            # Collect results
            return {
                "anomalies": future_anomalies.result(),
                "statistical_tests": future_stats.result(),
                "shape_comparison": future_shapes.result(),
            }
