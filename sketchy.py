import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import umap
from scipy import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="force_all_finite")
np.seterr(all="raise")


@dataclass
class GroupStats:
    n_rows: int
    numerical: Dict[str, Dict[str, float]]


@dataclass
class GlobalStats:
    n_rows: int
    numerical: Dict[str, Dict[str, Union[float, tuple]]]
    categorical: Dict[str, Dict[str, float]]


class AnomalyDetector:
    def __init__(
        self,
        categorical_cols: Optional[List[str]] = None,
        numerical_cols: Optional[List[str]] = None,
    ):
        self.categorical_cols = categorical_cols or []
        self.numerical_cols = numerical_cols or []
        self.group_stats: Dict[str, GroupStats] = {}
        self.global_stats: Optional[GlobalStats] = None
        self.pca: Optional[PCA] = None
        self.umap_reducer: Optional[umap.UMAP] = None
        self.isolation_forest: Optional[IsolationForest] = None
        self.elliptic_envelope: Optional[EllipticEnvelope] = None
        self.scaler: Optional[StandardScaler] = None

    def _compute_numerical_stats(self, data: pd.Series) -> Dict[str, float]:
        if len(data) == 0:
            return {
                "median": 0,
                "mean": 0,
                "std": 0,
                "min": 0,
                "max": 0,
                "distribution_params": (0, 1),
            }

        # Single pass through data
        data_np = np.asarray(data)
        mean = np.mean(data_np)
        std = np.std(data_np)
        return {
            "median": np.median(data_np),
            "mean": mean,
            "std": std,
            "min": np.min(data_np),
            "max": np.max(data_np),
            "distribution_params": (mean, std),  # Reuse computed values instead of stats.norm.fit
        }

    def _compute_global_stats(self, df: pd.DataFrame) -> None:
        """Compute global statistics using vectorized operations."""
        numerical_stats = {
            col: self._compute_numerical_stats(df[col]) for col in self.numerical_cols
        }

        categorical_stats = {
            col: df[col].value_counts(normalize=True).to_dict() for col in self.categorical_cols
        }

        self.global_stats = GlobalStats(
            n_rows=len(df), numerical=numerical_stats, categorical=categorical_stats
        )

    def _compute_group_stats(self, df: pd.DataFrame) -> None:
        """Compute group statistics using optimized groupby operations."""
        for col1 in self.categorical_cols:
            groups = df.groupby(col1, observed=True)
            self._store_group_stats(groups, [col1])

            # Use numpy's triu_indices for efficient pair generation
            pairs = np.triu_indices(len(self.categorical_cols), k=1)[0]
            for i in pairs:
                col2 = self.categorical_cols[i]
                if col1 < col2:
                    groups = df.groupby([col1, col2], observed=True)
                    self._store_group_stats(groups, [col1, col2])

    def _store_group_stats(self, groups, group_cols: List[str]) -> None:
        """Store group statistics using vectorized operations."""
        for group_name, group_df in groups:
            key = "_".join(
                str(x) for x in (group_name if isinstance(group_name, tuple) else [group_name])
            )

            numerical_stats = {
                col: {**self._compute_numerical_stats(group_df[col])} for col in self.numerical_cols
            }

            self.group_stats[key] = GroupStats(n_rows=len(group_df), numerical=numerical_stats)

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
    ) -> None:
        """Fit the anomaly detector using parallel processing where possible."""
        self._compute_global_stats(df)
        self._compute_group_stats(df)

        if embedding_cols:
            self.scaler = StandardScaler()
            embedding_data = self.scaler.fit_transform(df[embedding_cols])

            if dim_reduction in ["pca+umap", "pca"]:
                pca_components = pca_components or 100
                self.pca = PCA(n_components=pca_components, random_state=42)
                embedding_data = self.pca.fit_transform(embedding_data)

                cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
                print(f"PCA explained variance ratio: {cumulative_variance[-1]:.3f}")

            if dim_reduction in ["pca+umap", "umap"]:
                self.umap_reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    n_components=n_components,
                    metric="cosine",
                    random_state=42,
                )
                embedding_data = self.umap_reducer.fit_transform(embedding_data)

            if dim_reduction == "pca":
                self.pca = PCA(n_components=n_components, random_state=42)
                embedding_data = self.pca.fit_transform(embedding_data)

            # Fit outlier detection models in parallel
            self.isolation_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_jobs=-1,
            ).fit(embedding_data)

            self.elliptic_envelope = EllipticEnvelope(
                contamination=contamination, random_state=42
            ).fit(embedding_data)

    def _check_group_distribution(self, df: pd.DataFrame, threshold: float) -> List[str]:
        """Check for anomalies within specific demographic groups with zero handling."""
        anomalies = []

        for group_key, baseline_stats in self.group_stats.items():
            conditions = group_key.split("_")

            # Create mask for group filtering
            mask = pd.Series(True, index=df.index)
            for col, val in zip(self.categorical_cols[: len(conditions)], conditions):
                mask &= df[col] == val

            group_df = df[mask]

            if len(group_df) == 0:
                continue

            # Vectorized numerical comparisons with zero handling
            for col in self.numerical_cols:
                curr_median = group_df[col].median() if not group_df[col].empty else 0
                baseline_median = baseline_stats.numerical[col]["median"]

                # Use small epsilon for zero comparison
                if abs(baseline_median) > 1e-10:
                    pct_change = abs((curr_median - baseline_median) / baseline_median)

                    if pct_change > threshold:
                        anomalies.append(
                            f"Group {group_key}: Unusual {col} median: "
                            f"{curr_median:.2f} vs baseline {baseline_median:.2f}"
                        )
                else:
                    # If baseline is zero, check if current value is significantly different from zero
                    if abs(curr_median) > 1e-10:
                        anomalies.append(
                            f"Group {group_key}: Unusual {col} median: "
                            f"{curr_median:.2f} vs zero baseline"
                        )

        return anomalies

    def detect_anomalies(
        self, df: pd.DataFrame, embedding_cols: Optional[List[str]] = None, threshold: float = 0.1
    ) -> Dict[str, List[str]]:
        """Detect anomalies using vectorized operations."""
        return {
            "global_distribution": self._check_global_distribution(df),
            "group_distribution": self._check_group_distribution(df, threshold),
            "point_anomalies": (
                self._check_point_anomalies(df, embedding_cols) if embedding_cols else []
            ),
        }

    def _check_global_distribution(self, df: pd.DataFrame) -> List[str]:
        """Check global distribution using vectorized operations with zero handling."""
        anomalies = []

        # Check row count only if baseline is not zero
        if self.global_stats.n_rows > 0:
            row_ratio = abs(len(df) - self.global_stats.n_rows) / self.global_stats.n_rows
            if row_ratio > 0.2:
                anomalies.append(
                    f"Unusual number of records: {len(df)} vs baseline {self.global_stats.n_rows}"
                )

        # Vectorized numerical checks with zero handling
        for col in self.numerical_cols:
            curr_median = df[col].median() if not df[col].empty else 0

        # Calculate percentage change only if baseline is not zero
        baseline_median = self.global_stats.numerical[col]["median"]
        if abs(baseline_median) < 1e-10:  # Use epsilon comparison
            if abs(curr_median) > 1e-10:
                anomalies.append(f"Unusual {col} median: {curr_median:.2f} vs zero baseline")
        else:
            pct_change = abs((curr_median - baseline_median) / baseline_median)

        # Vectorized categorical checks with zero handling
        for col in self.categorical_cols:
            curr_dist = df[col].value_counts(normalize=True)
            baseline_dist = pd.Series(self.global_stats.categorical[col])

            # Ensure both distributions sum to 1 and handle zeros
            curr_dist = curr_dist.fillna(0)
            baseline_dist = baseline_dist.fillna(0)

            # Align distributions
            aligned_curr = curr_dist.reindex(baseline_dist.index, fill_value=0)

            # Only perform chi-square test if we have valid data
            if len(aligned_curr) > 0 and len(baseline_dist) > 0:
                try:
                    # Add small epsilon to avoid zero frequencies
                    epsilon = 1e-10
                    aligned_curr = aligned_curr + epsilon
                    baseline_dist = baseline_dist + epsilon

                    # Normalize after adding epsilon
                    aligned_curr = aligned_curr / aligned_curr.sum()
                    baseline_dist = baseline_dist / baseline_dist.sum()

                    _, p_value = stats.chisquare(aligned_curr, baseline_dist)
                    if p_value < 0.05:
                        anomalies.append(f"Unusual distribution in {col} (p-value: {p_value:.4f})")
                except ValueError as e:
                    anomalies.append(f"Unable to compare distributions for {col}: {str(e)}")

        return anomalies

    def _check_point_anomalies(self, df: pd.DataFrame, embedding_cols: List[str]) -> List[str]:
        """Check point anomalies using vectorized operations."""
        if not (self.pca is not None or self.umap_reducer is not None) or not embedding_cols:
            return []

        try:
            embedding_data = self.scaler.transform(df[embedding_cols])
            transformed_data = self._transform_embedding_data(embedding_data)

            # Vectorized anomaly detection
            if_scores = self.isolation_forest.score_samples(transformed_data)
            ee_scores = self.elliptic_envelope.score_samples(transformed_data)

            # Vectorized condition checking
            anomaly_mask = (if_scores < -0.5) & (ee_scores < -3)
            anomaly_indices = np.where(anomaly_mask)[0]

            return [
                f"Row {idx}: Unusual point in embedding space "
                f"(IF score: {if_scores[idx]:.2f}, EE score: {ee_scores[idx]:.2f})"
                for idx in anomaly_indices
            ]

        except Exception as e:
            return [f"Error detecting point anomalies: {str(e)}"]

    def _transform_embedding_data(self, embedding_data: np.ndarray) -> np.ndarray:
        """Transform data using fitted dimensionality reduction models."""
        if self.pca is not None:
            embedding_data = self.pca.transform(embedding_data)
        if self.umap_reducer is not None:
            embedding_data = self.umap_reducer.transform(embedding_data)
        return embedding_data


@dataclass
class DatasetComparator:
    categorical_cols: List[str] = field(default_factory=list)
    numerical_cols: List[str] = field(default_factory=list)
    _cached_stats: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.detector = AnomalyDetector(self.categorical_cols, self.numerical_cols)

    def compare_datasets(
        self,
        dataset_a: pd.DataFrame,
        dataset_b: pd.DataFrame,
        embedding_cols: Optional[List[str]] = None,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """Compare two datasets for distributional differences."""
        # Clear cache for new comparison
        self._cached_stats.clear()

        # Fit detector on dataset B (baseline)
        self.detector.fit(dataset_b, embedding_cols=embedding_cols)

        # Get anomalies in dataset A compared to B
        anomalies = self.detector.detect_anomalies(
            dataset_a, embedding_cols=embedding_cols, threshold=threshold
        )

        return {
            "anomalies": anomalies,
            "statistical_tests": self._perform_statistical_tests(dataset_a, dataset_b),
            "shape_comparison": self._compare_shapes(dataset_a, dataset_b),
        }

    @staticmethod
    @lru_cache(maxsize=128)
    def _calculate_distribution_stats(values: Tuple[float, ...]) -> pd.Series:
        """Calculate and cache distribution statistics."""
        return pd.Series(values).describe()

    def _perform_statistical_tests(
        self, dataset_a: pd.DataFrame, dataset_b: pd.DataFrame
    ) -> Dict[str, Any]:
        """Perform statistical tests using vectorized operations."""
        results = {}

        # Vectorized KS test for numerical columns
        for col in self.numerical_cols:
            try:
                # Convert to numpy arrays for faster computation
                a_values = (x for x in dataset_a[col].fillna(0))
                b_values = (x for x in dataset_b[col].fillna(0))

                ks_statistic, p_value = stats.ks_2samp(a_values, b_values)
                results[f"{col}_ks_test"] = {
                    "statistic": float(ks_statistic),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }
            except Exception as e:
                results[f"{col}_ks_test"] = {"error": str(e)}

        # Vectorized chi-square test for categorical columns
        for col in self.categorical_cols:
            try:
                # Use numpy operations for faster computation
                counts_a = dataset_a[col].value_counts()
                counts_b = dataset_b[col].value_counts()

                # Align categories using numpy operations
                all_categories = np.union1d(counts_a.index, counts_b.index)
                props_a = counts_a.reindex(all_categories, fill_value=0) / len(dataset_a)
                props_b = counts_b.reindex(all_categories, fill_value=0) / len(dataset_b)

                chi2, p_value = stats.chisquare(props_a, props_b)
                results[f"{col}_chi2_test"] = {
                    "statistic": float(chi2),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }
            except Exception as e:
                results[f"{col}_chi2_test"] = {"error": str(e)}

        return results

    def _compare_shapes(self, dataset_a: pd.DataFrame, dataset_b: pd.DataFrame) -> Dict[str, Any]:
        """Compare shapes using vectorized operations."""
        shape_info = {
            "row_counts": {
                "dataset_a": len(dataset_a),
                "dataset_b": len(dataset_b),
                "difference_pct": self._safe_percentage(
                    len(dataset_a) - len(dataset_b), len(dataset_b)
                ),
            }
        }

        # Vectorized comparison for numerical columns
        for col in self.numerical_cols:
            try:
                # Use cached stats
                stats_a = self._calculate_distribution_stats(tuple(dataset_a[col].dropna()))
                stats_b = self._calculate_distribution_stats(tuple(dataset_b[col].dropna()))

                shape_info[col] = {
                    "mean_difference_pct": self._safe_percentage(
                        stats_a["mean"] - stats_b["mean"], stats_b["mean"]
                    ),
                    "std_difference_pct": self._safe_percentage(
                        stats_a["std"] - stats_b["std"], stats_b["std"]
                    ),
                    "range_difference": {
                        "min_difference_pct": self._safe_percentage(
                            stats_a["min"] - stats_b["min"], stats_b["min"]
                        ),
                        "max_difference_pct": self._safe_percentage(
                            stats_a["max"] - stats_b["max"], stats_b["max"]
                        ),
                    },
                    "quartile_differences": {
                        f"q{i}_difference_pct": self._safe_percentage(
                            stats_a[f"{q}%"] - stats_b[f"{q}%"], stats_b[f"{q}%"]
                        )
                        for i, q in enumerate([25, 50, 75], 1)
                    },
                }
            except Exception as e:
                shape_info[col] = {"error": str(e)}

        return shape_info

    @staticmethod
    def _safe_percentage(difference: float, base: float, default: float = float("inf")) -> float:
        """Safely calculate percentage difference with better zero handling."""
        try:
            if abs(base) < 1e-10:  # Use small epsilon instead of exact zero
                return default
            return difference / base * 100
        except (ZeroDivisionError, FloatingPointError):
            return default
