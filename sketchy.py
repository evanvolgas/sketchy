import concurrent.futures
import multiprocessing
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import umap
from scipy import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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
        """[Previous docstring remains the same]"""
        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []
        self.n_jobs = None if n_jobs is None or n_jobs <= 0 else n_jobs
        self.group_stats: Dict[str, GroupStats] = {}
        self.global_stats: Dict = {}
        self.pca: Optional[PCA] = None
        self.umap_reducer: Optional[umap.UMAP] = None
        self.isolation_forest: Optional[IsolationForest] = None
        self.elliptic_envelope: Optional[EllipticEnvelope] = None
        self.scaler: Optional[StandardScaler] = None
        self.device = self._get_device(device)
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
        batch_size: int = 1000,
    ) -> None:
        """
        Fit the anomaly detector using the full dataset.

        Parameters:
        -----------
        df : pd.DataFrame
            Full dataset to analyze
        [Other parameters remain the same]
        """
        # Compute statistics in parallel for the full dataset
        self._vectorized_compute_global_stats(df)

        if embedding_cols:
            # Initialize scaler and transform data
            self.scaler = StandardScaler()
            embedding_data = self.scaler.fit_transform(df[embedding_cols])

            # Process embedding data in batches for GPU
            with self._device_context(embedding_data) as tensor_data:
                if dim_reduction in ["pca+umap", "pca"]:
                    # Perform PCA on full dataset using batches
                    n_components = pca_components or min(tensor_data.shape)
                    self.pca = PCA(n_components=n_components, random_state=42)

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
                    # Use UMAP on full dataset
                    self.umap_reducer = umap.UMAP(
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=n_components,
                        metric="cosine",
                        random_state=42,
                    )
                    embedding_data = self.umap_reducer.fit_transform(embedding_data)

            # Fit outlier detection models on full dataset
            self.isolation_forest = IsolationForest(
                contamination=contamination, random_state=42, n_jobs=self.n_jobs
            ).fit(embedding_data)

            self.elliptic_envelope = EllipticEnvelope(
                contamination=contamination,
                random_state=42,
                support_fraction=0.99,
                assume_centered=True,
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

    def _check_global_distribution(self, df: pd.DataFrame) -> List[str]:
        """Check for global distribution anomalies using vectorized operations."""
        anomalies = []

        # Check row count
        row_diff = abs(len(df) - self.global_stats["n_rows"])
        if row_diff > 0:
            anomalies.append(
                f"Unusual number of records: {len(df)} vs baseline {self.global_stats['n_rows']}"
            )

        # Check numerical distributions using vectorized operations
        for col in self.numerical_cols:
            curr_stats = self._parallel_compute_stats(df, col)
            baseline_stats = self.global_stats["numerical"][col]

            # Compare medians with formatted output
            if abs((curr_stats.median - baseline_stats.median) / baseline_stats.median) > 0.1:
                anomalies.append(
                    f"Unusual {col} median: {curr_stats.median:.2f} vs baseline {baseline_stats.median:.2f}"
                )

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

        def process_group(group_key: str, group_df: pd.DataFrame) -> List[str]:
            group_anomalies = []

            # Check numerical variables
            for col in self.numerical_cols:
                curr_stats = self._parallel_compute_stats(group_df, col)
                baseline_stats = self.group_stats[group_key].numerical[col]

                if (
                    abs((curr_stats.median - baseline_stats.median) / baseline_stats.median)
                    > threshold
                ):
                    # Format with group name and exact values
                    group_anomalies.append(
                        f"Group {group_key}: Unusual {col} median: "
                        f"{curr_stats.median:.2f} vs baseline {baseline_stats.median:.2f}"
                    )

            return group_anomalies

        # Process each group
        for col in self.categorical_cols:
            for group_name, group_data in df.groupby(col):
                if group_name in self.group_stats:
                    anomalies.extend(process_group(group_name, group_data))

        return anomalies

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        embedding_cols: Optional[List[str]] = None,
        threshold: float = 0.1,
        batch_size: int = 1000,
    ) -> Dict:
        """Detect anomalies with improved formatting."""
        results = {
            "global_distribution": self._check_global_distribution(df),
            "group_distribution": self._check_group_distribution(df, threshold),
            "point_anomalies": [],
        }

        # Add point anomalies if embedding cols are provided
        if embedding_cols:
            point_results = self._check_point_anomalies(df, embedding_cols, batch_size)
            if point_results:
                results["point_anomalies"] = point_results

        return results


@dataclass
class DatasetComparator:
    categorical_cols: Optional[List[str]] = field(default_factory=list)
    numerical_cols: Optional[List[str]] = field(default_factory=list)
    use_gpu: bool = True
    n_jobs: Optional[int] = -1

    def __post_init__(self):
        # Initialize lists
        self.categorical_cols = self.categorical_cols or []
        self.numerical_cols = self.numerical_cols or []

        # Optimize number of jobs
        if self.n_jobs is None or self.n_jobs <= 0:
            self.n_jobs = min(32, multiprocessing.cpu_count())

        # Set device and initialize components
        self.device = self._get_device()
        self.detector = AnomalyDetector(
            categorical_cols=self.categorical_cols,
            numerical_cols=self.numerical_cols,
            device=self.device,
            n_jobs=self.n_jobs,
        )
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
        self,
        values_a: np.ndarray,
        values_b: np.ndarray,
        batch_size: int = 10000,
        n_bins: int = 100,
        eps: float = 1e-10,
    ) -> Dict:
        """
        Performs a memory-efficient distribution comparison using histogram binning and
        the Chi-square test. Processes data in batches for large arrays.

        Parameters:
        -----------
        values_a: np.ndarray
            First array of values to compare
        values_b: np.ndarray
            Second array of values to compare
        batch_size: int
            Size of batches for processing large arrays
        n_bins: int
            Number of bins for histogram
        eps: float
            Small constant to avoid division by zero

        Returns:
        --------
        Dict containing:
            - statistic: Chi-square statistic
            - p_value: P-value from chi-square test
            - significant: Boolean indicating if difference is significant (p < 0.05)
            - kl_divergence: Kullback-Leibler divergence between distributions
        """
        try:
            # Convert to float32 for memory efficiency
            values_a = np.asarray(values_a, dtype=np.float32)
            values_b = np.asarray(values_b, dtype=np.float32)

            # Handle empty arrays
            if len(values_a) == 0 or len(values_b) == 0:
                return {
                    "error": "One or both arrays are empty",
                    "statistic": None,
                    "p_value": None,
                    "significant": None,
                }

            # Calculate global range for binning
            min_val = min(
                np.nanmin(values_a) if len(values_a) > 0 else np.inf,
                np.nanmin(values_b) if len(values_b) > 0 else np.inf,
            )
            max_val = max(
                np.nanmax(values_a) if len(values_a) > 0 else -np.inf,
                np.nanmax(values_b) if len(values_b) > 0 else -np.inf,
            )

            # Handle case where all values are the same
            if min_val == max_val:
                return {
                    "statistic": 0.0,
                    "p_value": 1.0,
                    "significant": False,
                    "kl_divergence": 0.0,
                }

            # Initialize histograms
            hist_a = np.zeros(n_bins, dtype=np.float32)
            hist_b = np.zeros(n_bins, dtype=np.float32)

            # Process array A in batches
            for i in range(0, len(values_a), batch_size):
                batch = values_a[i : i + batch_size]
                batch_hist, _ = np.histogram(batch, bins=n_bins, range=(min_val, max_val))
                hist_a += batch_hist

            # Process array B in batches
            for i in range(0, len(values_b), batch_size):
                batch = values_b[i : i + batch_size]
                batch_hist, _ = np.histogram(batch, bins=n_bins, range=(min_val, max_val))
                hist_b += batch_hist

            # Normalize histograms
            hist_a = hist_a / (np.sum(hist_a) + eps)
            hist_b = hist_b / (np.sum(hist_b) + eps)

            # Add small constant to avoid division by zero
            hist_a = hist_a + eps
            hist_b = hist_b + eps

            # Calculate chi-square statistic
            chi2_stat = np.sum((hist_a - hist_b) ** 2 / (hist_a + hist_b))

            # Calculate degrees of freedom (number of bins - 1)
            df = n_bins - 1

            # Calculate p-value
            p_value = 1 - stats.chi2.cdf(chi2_stat, df)

            # Calculate KL divergence
            kl_div = np.sum(hist_a * np.log(hist_a / hist_b))

            return {
                "statistic": float(chi2_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "kl_divergence": float(kl_div),
            }

        except Exception as e:
            return {
                "error": str(e),
                "statistic": None,
                "p_value": None,
                "significant": None,
                "kl_divergence": None,
            }

    def _perform_statistical_tests(self, dataset_a: pd.DataFrame, dataset_b: pd.DataFrame) -> Dict:
        """Perform statistical tests to compare distributions."""
        results = {}

        # Pre-compute value counts for categorical columns
        cat_counts_a = {
            col: dataset_a[col].fillna("MISSING").value_counts() for col in self.categorical_cols
        }
        cat_counts_b = {
            col: dataset_b[col].fillna("MISSING").value_counts() for col in self.categorical_cols
        }

        # Compare numerical distributions using vectorized operations
        for col in self.numerical_cols:
            try:
                # Use the fast distribution compare method for efficiency
                comparison = self.fast_distribution_compare(
                    dataset_a[col].values, dataset_b[col].values
                )
                results[col] = comparison
            except Exception as e:
                results[col] = {"error": str(e)}

        # Compare categorical distributions using chi-square test
        for col in self.categorical_cols:
            try:
                # Get all unique categories
                all_categories = pd.Index(
                    set(cat_counts_a[col].index) | set(cat_counts_b[col].index)
                )

                # Align distributions and fill missing values with 0
                counts_a = cat_counts_a[col].reindex(all_categories, fill_value=0)
                counts_b = cat_counts_b[col].reindex(all_categories, fill_value=0)

                # Perform chi-square test
                chi2, p_value = stats.chi2_contingency([counts_a, counts_b])[:2]

                results[col] = {
                    "statistic": float(chi2),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }
            except Exception as e:
                results[col] = {"error": str(e)}

        return results

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

    def format_comparison_results(
        self,
        dataset_a: pd.DataFrame,
        dataset_b: pd.DataFrame,
        results: Dict,
        max_point_anomalies: int = 5,
    ) -> str:
        """Format comparison results into a human-readable string."""
        output = []

        # Format Global Distribution Anomalies
        if results.get("anomalies", {}).get("global_distribution"):
            output.append("Global Distribution Anomalies:")
            for anomaly in results["anomalies"]["global_distribution"]:
                output.append(f"- {anomaly}")

        # Format Group Distribution Anomalies
        if results.get("anomalies", {}).get("group_distribution"):
            output.append("\nGroup Distribution Anomalies:")
            for anomaly in results["anomalies"]["group_distribution"]:
                output.append(f"- {anomaly}")

        # Format Point-level Anomalies
        if results.get("anomalies", {}).get("point_anomalies"):
            point_anomalies = results["anomalies"]["point_anomalies"][:max_point_anomalies]
            output.append(f"\nPoint-level Anomalies (showing first {max_point_anomalies}):")
            for anomaly in point_anomalies:
                output.append(f"- {anomaly}")

        # Format Statistical Comparison
        if results.get("statistical_tests"):
            output.append("\nComparison of key statistics:")
            output.append("Baseline Data:")
            baseline_stats = pd.DataFrame(
                {
                    col: {
                        "count": len(dataset_b),
                        "mean": dataset_b[col].mean(),
                        "std": dataset_b[col].std(),
                        "min": dataset_b[col].min(),
                        "25%": dataset_b[col].quantile(0.25),
                        "50%": dataset_b[col].quantile(0.50),
                        "75%": dataset_b[col].quantile(0.75),
                        "max": dataset_b[col].max(),
                    }
                    for col in self.numerical_cols
                }
            )
            output.append(baseline_stats.to_string())

            output.append("\nTest Data:")
            test_stats = pd.DataFrame(
                {
                    col: {
                        "count": len(dataset_a),
                        "mean": dataset_a[col].mean(),
                        "std": dataset_a[col].std(),
                        "min": dataset_a[col].min(),
                        "25%": dataset_a[col].quantile(0.25),
                        "50%": dataset_a[col].quantile(0.50),
                        "75%": dataset_a[col].quantile(0.75),
                        "max": dataset_a[col].max(),
                    }
                    for col in self.numerical_cols
                }
            )
            output.append(test_stats.to_string())

        return "\n".join(output)

    def compare_datasets(
        self,
        dataset_a: pd.DataFrame,
        dataset_b: pd.DataFrame,
        embedding_cols: Optional[List[str]] = None,
        threshold: float = 0.1,
    ) -> Dict:  # Keep returning Dict
        """Compare full datasets and return results dictionary."""
        try:
            # Optimize memory usage
            for col in self.numerical_cols:
                dataset_a[col] = dataset_a[col].astype(np.float32)
                dataset_b[col] = dataset_b[col].astype(np.float32)

            # Fit detector on full dataset B (baseline)
            self.detector.fit(dataset_b, embedding_cols=embedding_cols, dim_reduction="pca")

            # Initialize result futures
            futures = {}

            # Use thread pool for parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                # Submit all tasks
                futures["anomalies"] = executor.submit(
                    self.detector.detect_anomalies,
                    dataset_a,
                    embedding_cols=embedding_cols,
                    threshold=threshold,
                )
                futures["stats"] = executor.submit(
                    self._perform_statistical_tests, dataset_a, dataset_b
                )
                futures["shapes"] = executor.submit(self._compare_shapes, dataset_a, dataset_b)

                # Collect results with proper error handling
                results = {
                    "anomalies": futures["anomalies"].result(),
                    "statistical_tests": futures["stats"].result(),
                    "shape_comparison": futures["shapes"].result(),
                }

            return results

        except Exception as e:
            return {
                "error": str(e),
                "anomalies": {},
                "statistical_tests": {},
                "shape_comparison": {},
            }
