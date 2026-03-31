#!/usr/bin/env python3
"""
Unified MultiModal ML Pipeline for Audio, Video, and BioSig (ECG/EMG/RSP) data.

This script provides a unified interface to run ML classification on different
modalities with standardized settings. Modalities include:
- Audio
- Video
- ECG 
- EMG 
- RSP 

Settings (fixed as requested):
- Cross-validation: GroupKFold only
- Advance Level: nested CV + feature selection + regularization + HPO
- HPO search: GridSearchCV only
- Feature selection: Mann-Whitney + RFE
- Pipeline includes ConstantFilter for all modalities
- Models: Logistic Regression, SVC, KNN, Random Forest, AdaBoost, Decision Tree, XGBoost

Phase name standardization:
- training / training_pos -> training_pos
- coping / training_neg -> training_neg  
- emotion_induction_1 / induction1 -> induction1
- emotion_induction_2 / induction2 -> induction2
- latency -> latency


Author: Yasaman Moradi Fard
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from scipy.stats import mannwhitneyu, skew, kurtosis
from numpy.fft import rfft, rfftfreq
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
)
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

RANDOM_STATE = 42
POS_LABEL = 1  # Depressed
NEG_LABEL = 0  # Healthy

# =============================================================================
# Phase name standardization mapping
# =============================================================================
PHASE_NAME_MAPPING = {
    # Audio phases
    "training_pos": "training_pos",
    "training_neg": "training_neg",
    "induction1": "induction1",
    "induction2": "induction2",
    # Video phases (same as Audio)
    "latency": "latency",
    # BioSig phases
    "training": "training_pos",
    "coping": "training_neg",
    "emotion_induction_1": "induction1",
    "emotion_induction_2": "induction2",
}

STANDARD_PHASES = ["training_pos", "training_neg", "induction1", "induction2", "latency"]


def standardize_phase_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize phase names across all modalities."""
    if "phase" in df.columns:
        df = df.copy()
        df["phase"] = df["phase"].map(lambda x: PHASE_NAME_MAPPING.get(x, x))
    return df


# =============================================================================
# Aggregation Classes for Each Modality
# =============================================================================

class AudioAggregation:
    """
    Aggregation for Audio (OpenSmile) data.
    
    Computes 19 time-series statistics per feature:
    mean, std, min, max, skew, kurt, range, entropy, rate_of_change,
    peaks_count, median, pctl_25, pctl_75, iqr, slope, intercept,
    linear_error, coeff_var, mean_peak_dist
    """
    
    @staticmethod
    def compute_entropy(series: pd.Series, bins: int = 10) -> float:
        """Compute Shannon entropy of a numeric series via histogram."""
        arr = series.to_numpy(dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return 0.0
        hist, _ = np.histogram(arr, bins=bins, density=True)
        hist = hist[hist > 0]
        if hist.size == 0:
            return 0.0
        return float(-np.sum(hist * np.log2(hist)))
    
    @staticmethod
    def aggregate_group_stats(dataframe: pd.DataFrame, numeric_cols: List[str], bins: int = 10) -> pd.Series:
        """Compute all statistics for one group."""
        aggregated_features: Dict[str, float] = {}
        
        for col in numeric_cols:
            series = pd.to_numeric(dataframe[col], errors="coerce").dropna()
            if series.empty:
                continue
            
            # Basic stats
            aggregated_features[f"{col}__mean"] = float(series.mean())
            aggregated_features[f"{col}__std"] = float(series.std())
            aggregated_features[f"{col}__min"] = float(series.min())
            aggregated_features[f"{col}__max"] = float(series.max())
            
            # Shape of distribution
            aggregated_features[f"{col}__skew"] = float(series.skew())
            aggregated_features[f"{col}__kurt"] = float(series.kurt())
            aggregated_features[f"{col}__range"] = float(series.max() - series.min())
            aggregated_features[f"{col}__entropy"] = float(AudioAggregation.compute_entropy(series, bins=bins))
            
            # Dynamics
            aggregated_features[f"{col}__rate_of_change"] = float(series.diff().abs().mean())
            
            # Peaks
            peak_mask = (series.shift(1) < series) & (series.shift(-1) < series)
            aggregated_features[f"{col}__peaks_count"] = float(peak_mask.sum())
            
            # Robust statistics
            aggregated_features[f"{col}__median"] = float(series.median())
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            aggregated_features[f"{col}__pctl_25"] = q25
            aggregated_features[f"{col}__pctl_75"] = q75
            aggregated_features[f"{col}__iqr"] = q75 - q25
            
            # Linear trend
            x = np.arange(len(series))
            if len(series) > 1:
                slope, intercept = np.polyfit(x, series.values, 1)
                trend = np.polyval([slope, intercept], x)
                residuals = series.values - trend
                linear_error = residuals.std()
            else:
                slope, intercept, linear_error = 0.0, float(series.iloc[0]), 0.0
            aggregated_features[f"{col}__slope"] = float(slope)
            aggregated_features[f"{col}__intercept"] = float(intercept)
            aggregated_features[f"{col}__linear_error"] = float(linear_error)
            
            # Coefficient of variation
            mean_val = float(series.mean())
            aggregated_features[f"{col}__coeff_var"] = (
                float(series.std() / mean_val) if mean_val != 0 else 0.0
            )
            
            # Mean distance between peaks
            peak_indices = series[peak_mask].index.to_numpy()
            if peak_indices.size > 1:
                aggregated_features[f"{col}__mean_peak_dist"] = float(np.diff(peak_indices).mean())
            else:
                aggregated_features[f"{col}__mean_peak_dist"] = 0.0
        
        return pd.Series(aggregated_features)
    
    @staticmethod
    def aggregate_by_ID(df: pd.DataFrame) -> pd.DataFrame:
        """
        Method 1: Aggregate all phases together per participant.
        One vector per ID - all phases combined.
        """
        group_cols = [c for c in ["ID", "condition", "label"] if c in df.columns]
        
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in group_cols
        ]
        
        if not numeric_cols:
            raise ValueError("No numeric feature columns found for audio aggregation.")
        
        grouped = df[group_cols + numeric_cols].groupby(group_cols, sort=False)
        
        rows = []
        for key, g in grouped:
            if isinstance(key, tuple):
                row = {"ID": key[0]}
                if len(key) > 1 and "condition" in df.columns:
                    row["condition"] = key[1] if len(key) > 1 else g["condition"].iloc[0]
                if len(key) > 2 and "label" in df.columns:
                    row["label"] = key[2] if len(key) > 2 else g["label"].iloc[0]
            else:
                row = {"ID": key}
                if "condition" in g.columns:
                    row["condition"] = g["condition"].iloc[0]
                if "label" in g.columns:
                    row["label"] = g["label"].iloc[0]
            
            stats = AudioAggregation.aggregate_group_stats(g, numeric_cols)
            row.update(stats.to_dict())
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def aggregate_by_phase(df: pd.DataFrame) -> pd.DataFrame:
        """
        Method 2: Aggregate by phase.
        One vector per phase per participant.
        """
        group_cols = [c for c in ["ID", "condition", "phase", "label"] if c in df.columns]
        
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in group_cols
        ]
        
        if not numeric_cols:
            raise ValueError("No numeric feature columns found for audio aggregation.")
        
        grouped = df[group_cols + numeric_cols].groupby(["ID", "phase"], sort=False)
        
        rows = []
        for (pid, phase), g in grouped:
            row = {"ID": pid, "phase": phase}
            
            # Get metadata
            if "condition" in g.columns:
                row["condition"] = g["condition"].iloc[0]
            if "label" in g.columns:
                row["label"] = g["label"].iloc[0]
            
            # Compute aggregated stats
            stats = AudioAggregation.aggregate_group_stats(g, numeric_cols)
            row.update(stats.to_dict())
            rows.append(row)
        
        return pd.DataFrame(rows)

class VideoAggregation:
    """
    Aggregation for Video (OpenDBM) data.
    
    Computes 4 statistics per feature: mean, min, max, std
    Applied only for training phases; other phases already have 1 row per participant.
    """
    
    @staticmethod
    def aggregate_by_ID(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate all phases together per participant (one row per ID).
        Computes mean, std, min, max for each numeric feature.
        Keeps condition (and label) from the data; when condition was filtered
        upstream, all rows per ID share the same condition; when condition is
        'all', each ID keeps its condition from the first occurrence.
        """
        if 'ID' not in df.columns:
            raise ValueError("Expected 'ID' to be present for grouping.")

        meta_cols = ['ID', 'label', 'condition']
        numeric_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in meta_cols and col not in ['Aufgabe', 'phase']
        ]
        if not numeric_cols:
            raise ValueError("No numeric feature columns found for aggregation.")

        # Group by ID only → one row per participant; aggregate features
        agg_dict = {col: ['mean', 'std', 'min', 'max'] for col in numeric_cols}
        grouped = df.groupby('ID', sort=False)[numeric_cols].agg(agg_dict).reset_index()

        # Flatten MultiIndex: (col, 'mean') → 'col_mean'; keep 'ID' as-is (tuple ('ID','') → 'ID')
        if isinstance(grouped.columns, pd.MultiIndex):
            grouped.columns = [
                f"{c[0]}_{c[1]}" if isinstance(c, tuple) and len(c) == 2 and c[1] else (c[0] if isinstance(c, tuple) else c)
                for c in grouped.columns
            ]

        # Attach condition and label (first value per ID)
        id_meta = df[['ID']].drop_duplicates(subset=['ID'])
        for col in ['condition', 'label']:
            if col in df.columns:
                first_per_id = df.groupby('ID', sort=False)[col].first().reset_index()
                grouped = grouped.merge(first_per_id, on='ID', how='left')

        return grouped
    
    @staticmethod
    def aggregate_by_phase(df: pd.DataFrame) -> pd.DataFrame:
        """
        Method 2: Aggregate by phase.
        One vector per phase per participant.
        For training phases with multiple aufgabe, aggregate across aufgabe.
        """
        meta_cols = ['ID', 'label', 'condition', 'phase']
        group_cols = [c for c in meta_cols if c in df.columns]
        
        if 'ID' not in group_cols or 'phase' not in group_cols:
            raise ValueError("Expected 'ID' and 'phase' to be present for grouping.")
        
        numeric_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in group_cols and col not in ['Aufgabe']
        ]
        
        if not numeric_cols:
            raise ValueError("No numeric feature columns found for aggregation.")
        
        agg_dict = {col: ['mean', 'std', 'min', 'max'] for col in numeric_cols}
        grouped = df.groupby(group_cols)[numeric_cols].agg(agg_dict).reset_index()
        
        return grouped

class BioSigAggregation:
    """
    Aggregation for BioSig (EMG/ECG/RSP) data.
    
    Computes 10 specialized biosignal statistics per feature:
    mean, std, skew, kurt, rms, zcr (zero-crossing rate), medfreq (median frequency),
    specent (spectral entropy), hjorth_mob (Hjorth mobility), hjorth_comp (Hjorth complexity)
    """
    
    @staticmethod
    def _zero_cross_rate(x: np.ndarray) -> float:
        """Compute zero crossing rate."""
        x = np.asarray(x)
        x = x[~np.isnan(x)]
        if x.size <= 1:
            return np.nan
        return np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))
    
    @staticmethod
    def _median_frequency(x: np.ndarray, fs: float = 1.0) -> float:
        """Compute median frequency from power spectral density."""
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        if x.size < 4:
            return np.nan
        X = np.abs(rfft(x))
        freqs = rfftfreq(x.size, d=1.0/fs)
        psd = X**2
        cs = np.cumsum(psd)
        half = cs[-1] / 2.0
        idx = np.searchsorted(cs, half)
        return float(freqs[idx]) if idx < len(freqs) else np.nan
    
    @staticmethod
    def _spectral_entropy(x: np.ndarray) -> float:
        """Compute spectral entropy."""
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        if x.size < 4:
            return np.nan
        X = np.abs(rfft(x))
        psd = X**2
        s = psd.sum()
        if s <= 0:
            return np.nan
        p = psd / s
        p = p[p > 0]
        return float(-np.sum(p * np.log2(p)) / np.log2(len(p)))
    
    @staticmethod
    def _hjorth_mobility_complexity(x: np.ndarray) -> Tuple[float, float]:
        """Compute Hjorth mobility and complexity parameters."""
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        if x.size < 3:
            return (np.nan, np.nan)
        dx = np.diff(x)
        ddx = np.diff(dx)
        var_x, var_dx, var_ddx = np.var(x), np.var(dx), np.var(ddx)
        if var_x <= 0 or var_dx <= 0:
            return (np.nan, np.nan)
        mobility = math.sqrt(var_dx / var_x)
        complexity = math.sqrt(var_ddx / var_dx) / mobility if mobility > 0 else np.nan
        return float(mobility), float(complexity)
    
    @staticmethod
    def _rms(x: np.ndarray) -> float:
        """Compute root mean square."""
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        return float(np.sqrt(np.mean(x**2))) if x.size else np.nan
    
    @staticmethod
    def _compute_stats(g: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, float]:
        """Helper method to compute aggregation statistics for a group."""
        AGG_FUNCS = {
            "mean": np.nanmean,
            "std": np.nanstd,
            "skew": lambda a: skew(a, nan_policy="omit"),
            "kurt": lambda a: kurtosis(a, nan_policy="omit", fisher=True),
            "rms": BioSigAggregation._rms,
            "zcr": BioSigAggregation._zero_cross_rate,
            "medfreq": BioSigAggregation._median_frequency,
            "specent": BioSigAggregation._spectral_entropy,
            "hjorth_mob": lambda a: BioSigAggregation._hjorth_mobility_complexity(a)[0],
            "hjorth_comp": lambda a: BioSigAggregation._hjorth_mobility_complexity(a)[1],
        }
        
        row = {}
        arr = g[numeric_cols].to_numpy(dtype=float)
        for j, col in enumerate(numeric_cols):
            x = arr[:, j]
            for name, fn in AGG_FUNCS.items():
                try:
                    row[f"{col}__{name}"] = fn(x)
                except Exception:
                    row[f"{col}__{name}"] = np.nan
        return row
    
    @staticmethod
    def aggregate_by_ID(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Method 1: Aggregate all phases together per participant.
        One vector per ID - all phases combined.
        """
        rows = []
        for pid, g in df.groupby("ID", sort=False):
            row = {"ID": pid}
            if "condition" in df.columns:
                row["condition"] = g["condition"].iloc[0]
            if "label" in df.columns:
                row["label"] = g["label"].iloc[0]
            
            stats = BioSigAggregation._compute_stats(g, numeric_cols)
            row.update(stats)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def aggregate_by_phase(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Method 2: Aggregate by phase.
        One vector per phase per participant.
        """
        rows = []
        for (pid, ph), g in df.groupby(["ID", "phase"], sort=False):
            row = {"ID": pid, "phase": ph}
            if "condition" in df.columns:
                row["condition"] = g["condition"].iloc[0]
            if "label" in df.columns:
                row["label"] = g["label"].iloc[0]
            
            stats = BioSigAggregation._compute_stats(g, numeric_cols)
            row.update(stats)
            rows.append(row)
        
        return pd.DataFrame(rows)

# =============================================================================
# Preprocessing filters (outer fold: fit on train, transform train/test)
# =============================================================================

class ConstantFeatureFilter(BaseEstimator, TransformerMixin):
    """Remove features with constant values (learned from training data only)."""
    def __init__(self, mode="pragmatic"):
        self.mode = mode
        self.features_to_keep_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        kept = []
        for col in X.columns:
            if self.mode == "pragmatic":
                is_const = (X[col].nunique(dropna=True) <= 1)
            else:
                is_const = (X[col].nunique(dropna=False) <= 1)
            if not is_const:
                kept.append(col)
        self.features_to_keep_ = kept
        return self

    def transform(self, X):
        if self.features_to_keep_ is None:
            raise ValueError("ConstantFeatureFilter has not been fitted yet.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.features_to_keep_]


class NaNFeatureFilter(BaseEstimator, TransformerMixin):
    """Drop features with more than threshold proportion of NaN; rest are kept for imputation."""
    def __init__(self, threshold=0.69):
        self.threshold = threshold
        self.features_to_keep_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        n_samples = len(X)
        nan_ratios = X.isna().sum() / n_samples
        self.features_to_keep_ = nan_ratios[nan_ratios <= self.threshold].index.tolist()
        return self

    def transform(self, X):
        if self.features_to_keep_ is None:
            raise ValueError("NaNFeatureFilter has not been fitted yet.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.features_to_keep_]


class MannWhitneySelector(BaseEstimator, TransformerMixin):
    """Select top N features by Mann-Whitney U test p-value."""
    def __init__(self, n_features_to_select=25):
        self.n_features_to_select = n_features_to_select
        self.selected_features_ = None  # indices

    def fit(self, X, y):
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        pvals = []
        for i in range(X_arr.shape[1]):
            col = X_arr[:, i]
            c0 = col[y == 0]
            c1 = col[y == 1]
            c0 = c0[~np.isnan(c0)]
            c1 = c1[~np.isnan(c1)]
            if len(c0) > 0 and len(c1) > 0:
                try:
                    _, p = mannwhitneyu(c0, c1, alternative="two-sided")
                except Exception:
                    p = 1.0
            else:
                p = 1.0
            pvals.append((i, p))
        pvals.sort(key=lambda x: x[1])
        k = min(self.n_features_to_select, len(pvals))
        self.selected_features_ = np.array([idx for idx, _ in pvals[:k]])
        return self

    def transform(self, X):
        if self.selected_features_ is None:
            return X
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.selected_features_]
        return X[:, self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.selected_features_
        return [input_features[i] for i in self.selected_features_]


# =============================================================================
# Data Loading Functions for Each Modality
# =============================================================================

def normalize_id(pid):
    """Convert participant IDs to a standardized string format."""
    if pd.isna(pid) or (isinstance(pid, float) and math.isnan(pid)):
        return np.nan
    try:
        pid_int = int(float(pid))
        if pid_int < 100:
            return str(pid_int).zfill(3)
        else:
            return str(pid_int)
    except ValueError:
        return np.nan


def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize column names for XGBoost compatibility.
    Handles MultiIndex columns (e.g. from groupby.agg) by flattening to strings.
    """
    def sanitize_name(name: str) -> str:
        name = str(name)
        name = name.replace('[', '_').replace(']', '_').replace('<', '_')
        name = name.replace('>', '_').replace('|', '_').replace('&', '_')
        return name

    def flatten_col(col) -> str:
        """Convert column (possibly tuple from MultiIndex) to a flat string."""
        if isinstance(col, tuple):
            parts = [str(p) for p in col if p]
            return '_'.join(parts) if parts else sanitize_name(str(col))
        return str(col)

    df = df.copy()
    df.columns = [sanitize_name(flatten_col(col)) for col in df.columns]
    return df


def map_labels_to_binary(y: pd.Series) -> pd.Series:
    """Standardize labels to {0,1}, with Depressed=1, Healthy=0."""
    mapping = {
        'Depressed': 1, 'depressed': 1, 'DEPRESSED': 1, 1: 1, True: 1,
        'Healthy': 0, 'healthy': 0, 'HEALTHY': 0, 0: 0, False: 0
    }
    return y.map(lambda v: mapping.get(v, v)).astype(int)


def load_audio_data(
    condition: str,
    phases: List[str],
    opensmile_data_dir: Path,
    aggregation_method: str = "by_phase",
) -> pd.DataFrame:
    """
    Load Audio data from pre-aggregated CSV files.
    - by_phase: Loads Audio_data_aggregated_by_phase.csv; filters by phases and condition (if condition != 'all').
    - by_ID: Loads Audio_data_aggregated_by_ID.csv; filters by condition if condition != 'all', otherwise uses all data.
    Audio has no latency phase (latency is excluded when filtering phases).
    
    Args:
        condition: Condition filter ('CR', 'ADK', 'CRADK', 'SHAM', 'all').
                   If one specific condition: filter to that condition.
                   If 'all': no condition filter.
        phases: List of phases to load (standard names). Used when aggregation_method is 'by_phase'.
        opensmile_data_dir: Directory containing the aggregated CSV files, or path to a CSV file.
        aggregation_method: 'by_ID' or 'by_phase'.
            - by_ID: Load Audio_data_aggregated_by_ID.csv. Filter by condition if not 'all'; else feed all data.
            - by_phase: Load Audio_data_aggregated_by_phase.csv. Filter by selected phase(s) and by condition if not 'all'.
    
    Returns:
        DataFrame with aggregated audio features (no in-code aggregation; data is pre-aggregated).
    """
    opensmile_path = Path(opensmile_data_dir)
    if opensmile_path.is_file() and str(opensmile_path).lower().endswith(".csv"):
        csv_path = opensmile_path
    else:
        # Choose file by aggregation method
        if aggregation_method == "by_ID":
            csv_path = opensmile_path / "Audio_data_aggregated_by_ID.csv"
        elif aggregation_method == "by_phase":
            csv_path = opensmile_path / "Audio_data_aggregated_by_phase.csv"
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Audio data CSV not found: {csv_path}. "
            f"Place Audio_data_aggregated_by_ID.csv and/or Audio_data_aggregated_by_phase.csv "
            "in the OpenSmile data directory or pass a CSV path as --opensmile_data_dir."
        )

    print(f"[INFO] Loading audio data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    # Drop duplicate columns (e.g. name.1, name.2)
    cols_to_drop = [
        c for c in df.columns
        if isinstance(c, str) and "." in c and c.rsplit(".", 1)[-1].isdigit()
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Require key columns: ID and condition always; phase only for by_phase
    for col in ["ID", "condition"]:
        if col not in df.columns:
            raise ValueError(
                f"Audio CSV must contain column '{col}'. Found: {list(df.columns)[:20]}..."
            )
    if aggregation_method == "by_phase" and "phase" not in df.columns:
        raise ValueError(
            f"Audio_data_aggregated_by_phase.csv must contain column 'phase'. Found: {list(df.columns)[:20]}..."
        )
    df["ID"] = df["ID"].apply(normalize_id)

    # Build label from diagnose if present
    diag_col = None
    for cand in ["diagnose", "Diagnose"]:
        if cand in df.columns:
            diag_col = cand
            break
    if diag_col is not None:
        df["label"] = map_labels_to_binary(df[diag_col])
        df = df.drop(columns=[diag_col], errors="ignore")
    elif "label" not in df.columns:
        raise ValueError("Audio CSV must contain 'diagnose' or 'label' for binary labels.")

    # Standardize phase names for consistent filtering (only if phase column exists)
    df = standardize_phase_names(df.copy())

    # --- Apply filters (data is pre-aggregated; no in-code aggregation) ---
    if aggregation_method == "by_ID":
        # Filter by condition if not 'all'; otherwise use all data
        if condition.strip().lower() != "all":
            df = df[df["condition"].astype(str).str.strip().str.upper() == condition.strip().upper()].copy()
        if df.empty:
            raise ValueError(f"No audio data after filtering condition={condition}")
        combined_df = df
    elif aggregation_method == "by_phase":
        # Filter by selected phase(s) (exclude latency for audio)
        phases_clean = [p for p in phases if str(p).strip().lower() != "latency"]
        if not phases_clean:
            raise ValueError("No phases left for audio after excluding latency. Choose at least one of: training_pos, training_neg, induction1, induction2.")
        df = df[df["phase"].astype(str).isin(phases_clean)].copy()
        # Filter by condition if not 'all'
        if condition.strip().lower() != "all":
            df = df[df["condition"].astype(str).str.strip().str.upper() == condition.strip().upper()].copy()
        if df.empty:
            raise ValueError(f"No audio data after filtering condition={condition}, phases={phases_clean}")
        combined_df = df
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    # Data is already aggregated in the CSV; no call to AudioAggregation
    print(f"[INFO] Audio data loaded (pre-aggregated, method={aggregation_method})")
    aggregated_df = standardize_phase_names(combined_df.copy())
    aggregated_df = sanitize_feature_names(aggregated_df)

    # Add modality prefix to feature columns
    audio_meta = ["ID", "label", "phase", "condition", "aufgabe", "Aufgabe"]
    feature_cols = [c for c in aggregated_df.columns if c not in audio_meta]
    rename_map = {c: f"Audio_{c}" for c in feature_cols}
    aggregated_df = aggregated_df.rename(columns=rename_map)

    print(f"[INFO] Audio data: {aggregated_df.shape[0]} rows, {aggregated_df.shape[1]} columns")
    print(f"[INFO] Unique participants: {aggregated_df['ID'].nunique()}")
    print("-----------------------------------------\n")

    return aggregated_df


def load_video_data(
    condition: str,
    phases: List[str],
    opendbm_data_dir: Path,
    aggregation_method: str = "by_phase",
) -> pd.DataFrame:
    """
    Load Video (OpenDBM) data for given condition and phases.
    
    Aggregation behavior (same as BioSig/Audio):
    - by_phase: Load only the selected phase(s) (typically one in --phases). Filter by condition
      if condition != 'all'. Result: one vector per participant for that phase (in that condition,
      or from all conditions if condition='all').
    - by_ID: Load all phase files and aggregate all phases per participant. Filter only by condition:
      if one condition, keep that condition and aggregate all phases per participant; if condition='all',
      no condition filter, one vector per participant from all conditions.
    
    Args:
        condition: Condition filter ('CR', 'ADK', 'CRADK', 'SHAM', 'all')
        phases: List of phases to load (used only when aggregation_method is 'by_phase')
        opendbm_data_dir: Directory containing OpenDBM CSV files
        aggregation_method: One of 'by_ID', 'by_phase'
    
    Returns:
        DataFrame with aggregated video features
    """
    # Map standard phase names to CSV filenames
    phase_to_file = {
        "training_pos": "positive_training_orig_without_norm-1.csv",
        "training_neg": "negative_training_orig_without_norm-1.csv",
        "induction1": "emotion_induction_1_orig_without_norm_augmented.csv",
        "induction2": "emotion_induction_2_orig_without_norm_augmented.csv",
        "latency": "latency_sony_final.csv",
    }
    
    # When by_ID we load all phases (aggregate all phases per participant); when by_phase we load only --phases
    if aggregation_method == "by_ID":
        phases_to_load = list(phase_to_file.keys())
    else:
        phases_to_load = phases
        if not phases_to_load:
            raise ValueError("For aggregation_method='by_phase' you must pass at least one phase in --phases.")

    all_dfs: List[pd.DataFrame] = []

    for phase in phases_to_load:
        if phase not in phase_to_file:
            if aggregation_method == "by_phase":
                print(f"[WARN] Unknown phase '{phase}' for video data, skipping.")
            continue

        csv_path = opendbm_data_dir / phase_to_file[phase]
        if not csv_path.exists():
            print(f"[WARN] Video CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        print(f"[INFO] Loaded {csv_path.name} with shape {df.shape}")

        # Normalize condition column
        if 'condition' in df.columns:
            df['condition'] = df['condition'].astype(str).str.split('(', n=1).str[0].str.strip()

        # Filter by condition (one condition → filter; "all" → no filter)
        if condition.strip().lower() != "all":
            if 'condition' not in df.columns:
                raise ValueError(f"CSV has no 'condition' column to filter by '{condition}'")
            df = df[df['condition'].astype(str).str.strip().str.upper() == condition.strip().upper()].copy()

        # Add phase column
        df['phase'] = phase

        all_dfs.append(df)

    if not all_dfs:
        raise ValueError(
            f"No video data found for condition={condition}, "
            f"phases_to_load={phases_to_load if aggregation_method == 'by_ID' else phases}."
        )
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Clean column names
    combined_df.columns = [str(c).replace(" ", "_") for c in combined_df.columns]
    
    # Normalize ID column (zero-pad to 3 digits)
    if "ID" in combined_df.columns:
        combined_df["ID"] = combined_df["ID"].apply(normalize_id)
    
    # Drop derivative columns
    deriv_cols = [c for c in combined_df.columns if c.endswith('_deriv')]
    if deriv_cols:
        combined_df = combined_df.drop(columns=deriv_cols)
    
    # Preselect video features: only keep allowed AU, expressivity, gaze, pose columns
    au_numbers = [1, 4, 6, 7, 12, 14, 15, 20, 25, 26]
    au_prefixes = []
    for n in au_numbers:
        au_prefixes.append(f"fac_AU{n:02d}int")
        au_prefixes.append(f"fac_AU{n:02d}pres")
    expressivity_prefixes = [
        "fac_asymmas", "fac_paiin", "fac_com", "fac_neg", "fac_pos",
        "fac_hapin", "fac_sad", "fac_sur", "fac_fea", "fac_ang",
        "fac_dis", "fac_conin",
    ]
    gaze_prefixes = ["gaze_angle_x", "gaze_angle_y"]
    pose_prefixes = ["pose_Rx", "pose_Ry", "pose_Rz"]
    allowed_prefixes = tuple(au_prefixes + expressivity_prefixes + gaze_prefixes + pose_prefixes)
    
    meta_cols = [c for c in ["ID", "phase", "condition", "label", "Aufgabe"] if c in combined_df.columns]
    feature_cols = [c for c in combined_df.columns if c not in meta_cols]
    selected_cols = []
    for col in feature_cols:
        if "_deriv" in col:
            continue
        if "gaze_0_" in col or "pose_T" in col:
            continue
        if col.startswith(allowed_prefixes):
            selected_cols.append(col)
    
    if len(selected_cols) == 0:
        raise ValueError(
            "No video feature columns matched the allowed prefixes (AU, expressivity, gaze_angle, pose_R). "
            "Check that your OpenDBM column names match (e.g. fac_AU01int, fac_AU01pres, gaze_angle_x, pose_Rx)."
        )
    
    combined_df = combined_df[meta_cols + selected_cols].copy()
    print(f"[INFO] Video preselection: {len(feature_cols)} -> {len(selected_cols)} feature columns (allowed: AU, expressivity, gaze_angle, pose_R)")
    
    if "label" not in combined_df.columns:
        raise ValueError(
            "Video data must include a 'label' column for fusion. "
            "Ensure your OpenDBM CSVs contain a diagnosis/label column."
        )
    
    # Aggregate based on method
    print(f"[INFO] Aggregating video features using method: {aggregation_method}...")
    if aggregation_method == "by_ID":
        aggregated_df = VideoAggregation.aggregate_by_ID(combined_df)
    elif aggregation_method == "by_phase":
        aggregated_df = VideoAggregation.aggregate_by_phase(combined_df)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    aggregated_df = standardize_phase_names(aggregated_df)
    aggregated_df = sanitize_feature_names(aggregated_df)

    # Add modality prefix to feature columns
    meta_cols = ["ID", "phase", "condition", "label", "Aufgabe"]
    feature_cols = [c for c in aggregated_df.columns if c not in meta_cols]
    rename_map = {c: f"Video_{c}" for c in feature_cols}
    aggregated_df = aggregated_df.rename(columns=rename_map)

    print(f"[INFO] Video data: {aggregated_df.shape[0]} rows, {aggregated_df.shape[1]} columns")
    print(f"[INFO] Unique participants: {aggregated_df['ID'].nunique()}")
    print("-----------------------------------------\n")
    
    return aggregated_df


def load_biosig_data(
    data_type: str,
    minutes: int,
    condition: str,
    phases: List[str],
    aggregation_method: str = "by_phase",
) -> pd.DataFrame:
    """
    Load BioSig (EMG/ECG/RSP) data for given parameters.
    
    EMG and condition (no include_masseter argument):
    - condition == 'ADK': load EMG from Including_Masseter CSV only.
    - condition == 'all': load both Including_Masseter and Excluding_Masseter; drop columns
      starting with 'emg_m_' from the Including file, then concatenate the two datasets.
    - condition in ('CR', 'CRADK', 'SHAM', ...): load EMG from Excluding_Masseter only.
    
    Aggregation behavior:
    - by_phase: Expect typically one phase in --phases. Filter to that phase, then by condition
      if condition != 'all'. Result: one vector per participant for that phase.
    - by_ID: One vector per participant from all phases. Filter only by condition when not 'all'.
    
    Args:
        data_type: 'EMG', 'ECG', or 'RSP'
        minutes: Window size in minutes (1, 3, 5)
        condition: Condition filter ('CR', 'ADK', 'CRADK', 'SHAM', 'all')
        phases: List of phases (used only when aggregation_method is 'by_phase')
        aggregation_method: One of 'by_ID', 'by_phase'
    
    Returns:
        DataFrame with aggregated biosignal features
    """
    base = "/home/vault/empkins/tpD/D02/Students/Yasaman/1_BioSig_data/feature_extracted_data"
    dtype = data_type.upper()

    if dtype == "EMG":
        cond_upper = condition.strip().upper()
        path_inc = Path(f"{base}/EMG/Including_Masseter/EMG_{minutes}_Minute_Inc_Masseter_phase_assigned.csv")
        path_exc = Path(f"{base}/EMG/Excluding_Masseter/EMG_{minutes}_Minute_Exc_Masseter_phase_assigned.csv")
        if cond_upper == "ADK":
            if not path_inc.exists():
                raise FileNotFoundError(f"EMG (Including Masseter) file not found: {path_inc}")
            print(f"\n[INFO] Loading {dtype} data (ADK → Including_Masseter)")
            df = pd.read_csv(path_inc)
        elif cond_upper == "ALL":
            if not path_inc.exists():
                raise FileNotFoundError(f"EMG Including_Masseter file not found: {path_inc}")
            if not path_exc.exists():
                raise FileNotFoundError(f"EMG Excluding_Masseter file not found: {path_exc}")
            print(f"\n[INFO] Loading {dtype} data (all → Including_Masseter + Excluding_Masseter, dropping emg_m_* from Inc)")
            df_inc = pd.read_csv(path_inc)
            emg_m_cols = [c for c in df_inc.columns if str(c).startswith("emg_m_")]
            if emg_m_cols:
                df_inc = df_inc.drop(columns=emg_m_cols)
            df_exc = pd.read_csv(path_exc)
            # Align columns: use only columns present in both so concat has no extra NaNs
            common_cols = [c for c in df_inc.columns if c in df_exc.columns]
            df_inc = df_inc[[c for c in df_inc.columns if c in common_cols]]
            df_exc = df_exc[[c for c in df_exc.columns if c in common_cols]]
            df = pd.concat([df_inc, df_exc], ignore_index=True)
        else:
            if not path_exc.exists():
                raise FileNotFoundError(f"EMG (Excluding Masseter) file not found: {path_exc}")
            print(f"\n[INFO] Loading {dtype} data (condition {condition} → Excluding_Masseter)")
            df = pd.read_csv(path_exc)
    else:
        data_path = f"{base}/{dtype}/{dtype}_{minutes}_Minute_phase_assigned.csv"
        data_file = Path(data_path)
        if not data_file.exists():
            raise FileNotFoundError(f"BioSig data file not found: {data_path}")
        print(f"\n[INFO] Loading {dtype} data")
        df = pd.read_csv(data_file)


    # Normalize ID
    if "ID" in df.columns:
        df["ID"] = df["ID"].apply(normalize_id)

    
    # Map BioSig phase names to standard names
    if "phase" in df.columns:
        df = standardize_phase_names(df)

    # Filter by condition (for both methods: one condition → filter; "all" → no filter)
    if condition.lower().strip() != "all" and "condition" in df.columns:
        df = df[df["condition"].astype(str).str.strip().str.upper() == condition.strip().upper()].copy()

    # Filter by phases only when aggregating by_phase (one vector per participant per selected phase).
    # When aggregating by_ID we do not filter by phases: we keep all phases and aggregate per participant.
    if aggregation_method == "by_phase":
        if phases and "phase" in df.columns:
            df = df[df["phase"].isin(phases)].copy()
        elif not phases and "phase" in df.columns:
            raise ValueError("For aggregation_method='by_phase' you must pass at least one phase in --phases.")

    # Handle label column
    if "Diagnose" in df.columns:
        df["label"] = map_labels_to_binary(df["Diagnose"])
        df = df.drop(columns=["Diagnose"], errors="ignore")
   
    elif "diagnose" in df.columns:
        df["label"] = map_labels_to_binary(df["diagnose"])
        df = df.drop(columns=["diagnose"], errors="ignore")
    elif "label" not in df.columns:
        raise ValueError("No label column found in BioSig data.")
    
    # Clean column names
    df.columns = [str(c).replace(" ", "_") for c in df.columns]
    
    # Identify numeric feature columns
    meta_cols = ["ID", "phase", "time", "Window_Index", "condition", "gender", "age",
                 "previous_depression_diagnosis", "including_masseter", "Including_masseter",
                 "EMG_measuring", "ECG_measuring", "RSP_measuring", "PHQ8-Score", "PHQ9-Score"]
    
    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in meta_cols + ["label"]
    ]
   
    # Aggregate based on method
    print(f"[INFO] Aggregating BioSig features using method: {aggregation_method}...")
    if aggregation_method == "by_ID":
        aggregated_df = BioSigAggregation.aggregate_by_ID(df, numeric_cols)
    elif aggregation_method == "by_phase":
        aggregated_df = BioSigAggregation.aggregate_by_phase(df, numeric_cols)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    # Empty result can happen when condition+phase filter leaves no rows (e.g. EMG ADK + training_pos)
    if aggregated_df.empty or "ID" not in aggregated_df.columns:########################
        raise ValueError(
            f"No {dtype} data after aggregation for condition={condition!r}, phases={phases!r}. "
            "The filtered dataframe had no rows (or no groups). Check that the BioSig CSV has rows "
            f"with condition '{condition}' and phases {phases} for this modality."
        )############################

    # Merge back metadata (only for methods that preserve phase info)
    if aggregation_method != "by_ID" and "condition" in df.columns:
        if "phase" in aggregated_df.columns:
            cond_map = df.groupby(["ID", "phase"])["condition"].first().to_dict()
            aggregated_df["condition"] = aggregated_df.apply(
                lambda row: cond_map.get((row["ID"], row.get("phase", None)), None), axis=1
            )
        else:
            cond_map = df.groupby("ID")["condition"].first().to_dict() ###########
            aggregated_df["condition"] = aggregated_df["ID"].map(cond_map) ############
    
    if "label" in df.columns:
        if aggregation_method != "by_ID" and "phase" in aggregated_df.columns:
            label_map = df.groupby(["ID", "phase"])["label"].first().to_dict()
            aggregated_df["label"] = aggregated_df.apply(
                lambda row: label_map.get((row["ID"], row.get("phase", None)), None), axis=1
            )
        else:
            label_map = df.groupby("ID")["label"].first().to_dict()
            aggregated_df["label"] = aggregated_df["ID"].map(label_map)
        aggregated_df = aggregated_df.dropna(subset=["label"])
        aggregated_df["label"] = aggregated_df["label"].astype(int)
    
    aggregated_df = sanitize_feature_names(aggregated_df)

    # Add modality prefix to feature columns
    biosig_meta = ["ID", "label", "phase", "condition", "Aufgabe"]
    feature_cols = [c for c in aggregated_df.columns if c not in biosig_meta]
    rename_map = {c: f"{dtype}_{c}" for c in feature_cols}
    aggregated_df = aggregated_df.rename(columns=rename_map)

    print(f"[INFO] BioSig data: {aggregated_df.shape[0]} rows, {aggregated_df.shape[1]} columns")
    print(f"[INFO] Unique participants: {aggregated_df['ID'].nunique()}")
    print("-----------------------------------------\n")
    
    return aggregated_df


def load_text_data(text_csv_path: Path, condition: str = "all") -> pd.DataFrame:
    """
    Load Text modality features from a pre-computed CSV.
    
    Expected format:
        - Column 'patient_id' (participant ID, not zero-padded)
        - Column 'Condition' (treatment condition: CR, ADK, CRADK, SHAM, ...)
        - Column 'Diagnose' (diagnosis label, e.g. 'Depressed' / 'Healthy')
        - 14 feature columns (all numeric)
    
    Processing:
        - Zero-pad 'patient_id' to 3 digits and rename to 'ID'
        - Optionally filter rows by 'Condition' (if condition != 'all')
        - Map 'Diagnose' to binary 'label' using map_labels_to_binary
        - No aggregation by phase (one row per ID)
    """
    if not text_csv_path.exists():
        raise FileNotFoundError(f"Text data CSV not found: {text_csv_path}")
    
    df = pd.read_csv(text_csv_path)
    # Basic column checks
    required_cols = ["patient_id", "condition", "Diagnose"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Text CSV must contain columns {required_cols}, missing: {missing}")
 
    # Zero-pad patient_id and rename to ID
    df = df.copy()
    df["ID"] = df["patient_id"].apply(normalize_id).astype(str)
    df = df.drop(columns=["patient_id"])
    
    # Normalize / filter by condition
    df["condition"] = df["condition"].astype(str).str.strip()
    if condition.lower() != "all":
        df = df[df["condition"] == condition].copy()
        if df.empty:
            raise ValueError(f"No text rows found for condition '{condition}' in {text_csv_path}")
    
    # Map Diagnose -> binary label
    df["label"] = map_labels_to_binary(df["Diagnose"])
    
    # Treat all remaining non-ID / non-label / non-condition columns as numeric features
    feature_cols = [
        c for c in df.columns
        if c not in ["ID", "label", "condition", "Diagnose"]
    ]
    if len(feature_cols) != 14:
        print(f"[WARN] Text data: expected 14 feature columns, found {len(feature_cols)}.")
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df = df.drop(columns=["Diagnose"])

    # Add modality prefix to feature columns
    rename_map = {c: f"Text_{c}" for c in feature_cols}
    df = df.rename(columns=rename_map)

    print(f"[INFO] Text data: {df.shape[0]} rows, {len(feature_cols)} feature columns")
    print(f"[INFO] Unique participants (text): {df['ID'].nunique()}")
    if condition.lower() != "all":
        print(f"[INFO] Condition filter applied to text modality: {condition}")
    print("-----------------------------------------\n")
    return df


# =============================================================================
# Classifier Models
# =============================================================================

def get_classifier_models() -> Dict[str, object]:
    """
    Return classifiers 
    """
    models: Dict[str, object] = {
        "LogisticRegression": LogisticRegression(
            C=0.1,
            max_iter=100000,
            class_weight="balanced",
            solver="saga",
            random_state=RANDOM_STATE,
        ),
        "SVC_RBF": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=100000,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, class_weight="balanced"),
            n_estimators=100,
            learning_rate=0.5,
            random_state=RANDOM_STATE,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
        ),
    }
    
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=2,
            random_state=RANDOM_STATE,
        )
    
    return models


def get_param_grids_clf_only() -> Dict[str, Dict]:
    """Classifier-only param grids (no RFE n). RFE n is chosen by a for loop over 1..n_features."""
    grids = {
        "LogisticRegression": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        "SVC_RBF": {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", "auto", 0.01, 0.1]},
        "RandomForest": {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 5, None]},
        "AdaBoost": {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.1, 0.5, 1.0]},
        "DecisionTree": {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 2, 4]},
        "KNN": {"clf__n_neighbors": [3, 5, 7, 9]},
    }
    if HAS_XGB:
        grids["XGBoost"] = {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 4, 5],
            "clf__learning_rate": [0.01, 0.05, 0.1],
        }
    return grids


# =============================================================================
# Fusion pipelines and helpers
# =============================================================================

def build_preprocess_pipeline(nan_threshold: float = 0.69) -> Pipeline:
    """Outer fold: ConstantFilter -> NaNFilter (drop >70% NaN) -> Imputer -> Scaler. Fit on train, transform train/test."""
    return Pipeline([
        ("constant_filter", ConstantFeatureFilter(mode="pragmatic")),
        ("nan_filter", NaNFeatureFilter(threshold=nan_threshold)),
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])


def build_inner_pipeline_early_fusion(clf, rfe_n: int) -> Pipeline:
    """Early fusion inner: RFE(n) on concatenated features -> classifier. MW per modality done separately.
    RFE always uses DecisionTreeClassifier for consistent, fast feature selection across all final classifiers."""
    rfe_estimator = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    return Pipeline([
        ("rfe_selector", RFE(estimator=rfe_estimator, n_features_to_select=rfe_n, step=0.1)),
        ("clf", clf),
    ])


def extract_selected_features_early_fusion(pipe: Pipeline, feature_names: List[str]) -> List[str]:
    """Extract selected feature names from early fusion pipeline (RFE only)."""
    if "rfe_selector" not in pipe.named_steps:
        return list(feature_names)
    rfe = pipe.named_steps["rfe_selector"]
    if hasattr(rfe, "support_"):
        return [f for f, s in zip(feature_names, rfe.support_) if s]
    return list(feature_names)


# =============================================================================
# Evaluation and Plotting Functions
# =============================================================================

def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive metrics."""
    y_pred = model.predict(X_test)
    
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)),
        "sensitivity": float(recall_score(y_test, y_pred, pos_label=POS_LABEL, zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    
    # ROC-AUC
    y_pred_proba = None
    try:
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
        elif hasattr(model, "decision_function"):
            y_pred_proba = model.decision_function(X_test)
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
        else:
            metrics["roc_auc"] = float("nan")
    except Exception:
        metrics["roc_auc"] = float("nan")
    
    # Confusion matrix
    try:
        cm = confusion_matrix(y_test, y_pred, labels=[NEG_LABEL, POS_LABEL])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["tn"] = int(tn)
            metrics["fp"] = int(fp)
            metrics["fn"] = int(fn)
            metrics["tp"] = int(tp)
            metrics["cm"] = [[int(tn), int(fp)], [int(fn), int(tp)]]
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        else:
            metrics["tn"] = metrics["fp"] = metrics["fn"] = metrics["tp"] = 0
            metrics["specificity"] = float("nan")
            metrics["cm"] = [[0, 0], [0, 0]]
    except Exception:
        metrics["tn"] = metrics["fp"] = metrics["fn"] = metrics["tp"] = 0
        metrics["specificity"] = float("nan")
        metrics["cm"] = [[0, 0], [0, 0]]
    
    metrics["y_true"] = y_test.tolist()
    metrics["y_pred"] = y_pred.tolist()
    metrics["y_pred_proba"] = y_pred_proba.tolist() if y_pred_proba is not None else None
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str,
                          normalization: str, save_path: Path) -> None:
    """Plot and save confusion matrix."""
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"[WARN] Empty arrays provided for confusion matrix for {model_name}")
        return
    
    cm = confusion_matrix(y_true, y_pred, labels=[NEG_LABEL, POS_LABEL])
    
    if cm.size == 0 or cm.shape != (2, 2):
        print(f"[WARN] Invalid confusion matrix shape {cm.shape} for {model_name}")
        return
    
    plt.figure(figsize=(8, 6))

    # Compute overall metrics for the title
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted")
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Healthy", "Depressed"],
                    yticklabels=["Healthy", "Depressed"])
    else:
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar()
        plt.xticks([0, 1], ["Healthy", "Depressed"])
        plt.yticks([0, 1], ["Healthy", "Depressed"])
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(f"Confusion Matrix\n{model_name} | (Acc: {acc:.3f}, F1-weighted: {f1_w:.3f})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved confusion matrix to {save_path}")
    print("-----------------------------------------\n")

def plot_roc_curves(roc_data: Dict[str, Tuple[List[float], List[float]]],
                    normalization: str, save_path: Path) -> None:
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for idx, (model_name, (y_true_list, y_proba_list)) in enumerate(roc_data.items()):
        if not y_true_list or not y_proba_list:
            continue
        
        y_true = np.array(y_true_list)
        y_proba = np.array(y_proba_list)
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                     label=f'{model_name} (AUC = {roc_auc:.3f})')
        except Exception:
            continue
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - All Models\n(Normalization: {normalization})')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved ROC curves to {save_path}")
    print("-----------------------------------------\n")


# =============================================================================
# Main Function
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="MultiModal ML: early fusion")

    p.add_argument("--run_type", default="main", choices=["main", "test_main"])
    p.add_argument("--fusion_modalities", nargs="+", default=["audio", "video", "ecg", "emg", "rsp"],
                   help="Modalities to use in early fusion. E.g. 'ecg emg' or 'audio video ecg emg rsp text'")
    p.add_argument("--condition", type=str, default="all", help="Condition filter: CR, ADK, CRADK, SHAM, all") # CR, ADK, CRADK, SHAM, all
    p.add_argument("--phases", nargs="+", default=["training_neg"], help="Phases to load (standard names)") # training_pos, training_neg, induction1, induction2, latency, all
    p.add_argument("--minutes", type=int, default=3, help="BioSig window minutes (1, 3, 5)")
    # Audio-specific arguments
    p.add_argument(
        "--opensmile_data_dir",
        type=str,
        default="/home/vault/empkins/tpD/D02/Students/Yasaman/3_Audio_data/Data/OpenSmile_data",
        help="Directory containing Audio_data.csv (all participants), or path to Audio_data.csv (for audio modality)",
    )
    # Video-specific arguments
    p.add_argument(
        "--opendbm_data_dir",
        type=str,
        default="/home/vault/empkins/tpD/D02/processed_data/processed_openDBM_functional",
        help="Directory with OpenDBM CSVs (for video modality)",
    )
    # Text modality (pre-computed features CSV; no preprocessing / Mann-Whitney)
    p.add_argument(
        "--text_data_path",
        type=str,
        default="/home/vault/empkins/tpD/D02/Students/Yasaman/4_Text_data/Data/text_features_3370.csv", # 3370 had the best results
        help="Path to text modality CSV (patient_id + numeric features). Used when 'text' is in --fusion_modalities",
    )
    p.add_argument(
        "--aggregation_method",
        type=str,
        default="by_ID",
        choices=["by_phase", "by_ID"],
        help=(
            "Aggregation method for modalities. Text has no phase, but can be combined with either "
            "method; IDs are aligned appropriately."
        ),
    )
    p.add_argument("--classifiers", nargs="*", default=None,
                   help="Classifier names to run (default: all). E.g. LogisticRegression SVC_RBF") # LogisticRegression, SVC_RBF, KNN, RandomForest, DecisionTree, XGBoost, AdaBoost
    p.add_argument("--outer_folds", type=int, default=5, help="Outer GroupKFold splits") # 5
    p.add_argument("--inner_folds", type=int, default=3, help="Inner GroupKFold splits for HPO") # 3
    p.add_argument("--nan_threshold", type=float, default=0.69,
                   help="Drop features with more than this fraction of NaN (rest imputed)") # 0.69
    p.add_argument("--mannwhitney_per_modality", type=int, default=20, help="Top K features per modality (Mann-Whitney) before concat") # 20
    p.add_argument("--output_dir", type=str, default="/home/vault/empkins/tpD/D02/Students/Yasaman/5_MultiModal_ML/Early Fusion Classification Results", help="Output directory (default: Early_Fusion_* under script dir)")
    args = p.parse_args()
    return args


def main(cfg) -> None:
    """
    Early fusion: load all modalities, outer fold = preprocessing per modality; inner = 20 features per modality (MW),
    concatenate, then for loop over all possible n_features (RFE) + HPO; pick best n and hyperparameters; evaluate on test.
    """
    # ============================
    # DATA PREPARATION / LOADING
    # ============================
    # Load all modalities from cfg and align by sample key (inlined former _load_and_align_modalities).
    print("-----------------------------------------\n")
    print("Data preparation / loading")
    print("-----------------------------------------\n")
    modalities = [m.strip().lower() for m in cfg.fusion_modalities]
    condition = cfg.condition
    phases = cfg.phases if isinstance(cfg.phases, list) else [cfg.phases] # a list of phases or only 'all'
    aggregation_method = cfg.aggregation_method # phase or ID
    opensmile_dir = Path(cfg.opensmile_data_dir) 
    opendbm_dir = Path(cfg.opendbm_data_dir)
    text_path = Path(getattr(cfg, "text_data_path", ""))
    print("Dashboard:")
    print(f"Modalities: {modalities}")
    print(f"Condition: {condition}")
    print(f"Phases: {phases}")
    print(f"Aggregation method: {aggregation_method}")
    print(f"OpenSmile data directory: {opensmile_dir}")
    print(f"OpenDBM data directory: {opendbm_dir}")
    print(f"Text data path: {text_path}")
    print(f"Outer folds: {cfg.outer_folds}")
    print(f"Inner folds: {cfg.inner_folds}")
    print(f"Nan threshold: {cfg.nan_threshold}")
    print(f"Mann-Whitney per modality: {cfg.mannwhitney_per_modality}")
    print(f"Output directory: {cfg.output_dir}")
    print("-----------------------------------------\n")

    # Load data for each modality and align by sample key:
    modality_data: Dict[str, Dict] = {}

    for mod in modalities:
        if mod == "text":
            df = load_text_data(text_path, condition=condition)
            
        elif mod == "audio":
            df = load_audio_data(condition, phases, opensmile_dir, aggregation_method=aggregation_method)
            
        elif mod == "video":
            df = load_video_data(condition, phases, opendbm_data_dir=opendbm_dir, aggregation_method=aggregation_method)
            
        elif mod in ("ecg", "emg", "rsp"):
            dtype = {"ecg": "ECG", "emg": "EMG", "rsp": "RSP"}[mod]
            df = load_biosig_data(
                dtype,
                cfg.minutes,
                condition,
                phases,
                aggregation_method=aggregation_method,
            )
        else:
            raise ValueError(f"Unknown modality: {mod}")

        drop_cols = [c for c in ["label", "ID", "phase", "condition", "aufgabe", "Aufgabe"] if c in df.columns]
        X = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
        modality_data[mod] = {
            "X": X,
            "y": df["label"].to_numpy(dtype=int),
            "groups": df["ID"].astype(str).to_numpy(),
            "df": df,
        }

    # Find common IDs across all modalities (simple intersection)
    common_IDs = set(modality_data[modalities[0]]["df"]["ID"].astype(str))
    for mod in modalities[1:]:
        ids_mod = set(modality_data[mod]["df"]["ID"].astype(str))
        common_IDs &= ids_mod
    common_IDs = sorted(common_IDs)
    print(f"Number of common IDs: {len(common_IDs)}")
    if not common_IDs:
        raise ValueError("No common samples across modalities")

    # Re-align modalities: filter to common IDs, sort by ID, one row per participant
    # Horizontal concatenation keyed by ID (same row order across modalities)
    meta_cols = ["label", "ID", "phase", "condition", "aufgabe", "Aufgabe"]
    for mod in modalities:
        d = modality_data[mod]
        df_mod = d["df"].copy()
        df_mod["ID"] = df_mod["ID"].astype(str)
        mask = df_mod["ID"].isin(common_IDs)
        df_mod = df_mod[mask].copy()
        # If multiple rows per ID (e.g. by_phase with multiple phases), take first per ID
        if df_mod.duplicated(subset="ID").any():
            df_mod = df_mod.groupby("ID", sort=True).first().reset_index()
        df_mod = df_mod.set_index("ID").loc[common_IDs].reset_index()
        drop_cols = [c for c in meta_cols if c in df_mod.columns]
        X_mod = df_mod.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
        d["X_aligned"] = X_mod
        d["y_aligned"] = df_mod["label"].to_numpy(dtype=int)
        d["groups_aligned"] = df_mod["ID"].to_numpy()

    # ============================
    # GLOBAL LABELS / GROUPS SETUP
    # ============================
    print("-----------------------------------------\n")
    print("Global labels / groups setup")
    print("-----------------------------------------\n")
    ref_mod = modalities[0]
    y = modality_data[ref_mod]["y_aligned"]
    groups = modality_data[ref_mod]["groups_aligned"]
    outer_folds = cfg.outer_folds
    inner_folds = cfg.inner_folds
    nan_threshold = cfg.nan_threshold
    mw_per_mod = getattr(cfg, "mannwhitney_per_modality", 20)
    phases = cfg.phases if isinstance(cfg.phases, list) else [cfg.phases]

    mod_str = "_".join(sorted(modalities))
    phases_str = "_".join(phases)
    agg_suffix = "byPhase" if cfg.aggregation_method == "by_phase" else "by_ID"
    base_dir = Path(cfg.output_dir) if cfg.output_dir else Path(__file__).resolve().parent
    out_dir = base_dir / f"Early_Fusion_{mod_str}_{cfg.condition}_{phases_str}_{agg_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"early_fusion_{mod_str}_{cfg.condition}_{phases_str}"

    n_outer = outer_folds
    if len(np.unique(groups)) < n_outer:
        raise ValueError(f"Not enough groups for {n_outer}-fold GroupKFold")

    # ============================
    # MODEL SETUP (CLASSIFIERS + PARAM GRIDS)
    # ============================
    print("-----------------------------------------\n")
    print("Model setup")
    print("-----------------------------------------\n")
    models = get_classifier_models()
    if cfg.classifiers:
        models = {k: v for k, v in models.items() if k in cfg.classifiers}
    if not models:
        raise ValueError("No valid classifiers selected")
    param_grids_clf = get_param_grids_clf_only()

    # Create one subfolder per classifier (e.g. KNN, RandomForest, LogisticRegression)
    for model_name in models:
        (out_dir / model_name).mkdir(parents=True, exist_ok=True)

    # ============================
    # START OUTER FOLD (GroupKFold CV)
    # ============================

    kf_outer = GroupKFold(n_splits=n_outer)
    all_results: List[Dict] = []
    selected_features_all: Dict[int, Dict[str, List[str]]] = {}
    cm_accum: Dict[str, Tuple[List[int], List[int]]] = {}
    roc_accum: Dict[str, Tuple[List[int], List[float]]] = {}  # (y_true, y_proba) per model for ROC
    # Store per-fold confusion matrices per model (tp, tn, fp, fn)
    fold_confusions: Dict[int, Dict[str, Dict[str, int]]] = {}

    for fold_idx, (train_idx, test_idx) in enumerate(kf_outer.split(y, y, groups=groups), start=1):
        print(f"\n[INFO] Early fusion outer fold {fold_idx}/{n_outer}")
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]

        # ============================
        # OUTER-FOLD DATA PREPARATION / PREPROCESSING
        # (fit preprocessing on train, transform train & test, per modality)
        # ============================

        X_train_per_mod: Dict[str, pd.DataFrame] = {}
        X_test_per_mod: Dict[str, pd.DataFrame] = {}
        for mod in modalities:
            X_raw = modality_data[mod]["X_aligned"]
            X_train_raw = X_raw.iloc[train_idx].copy()
            X_test_raw = X_raw.iloc[test_idx].copy()
            if mod == "text":
                # Text: no preprocessing (no filters, imputation, scaling); use as-is
                X_train_per_mod[mod] = X_train_raw.copy()
                X_test_per_mod[mod] = X_test_raw.copy()
                # Set index to IDs to ensure proper alignment during concatenation
                X_train_per_mod[mod].index = groups_train
                X_test_per_mod[mod].index = groups_test
            else:
                preprocess = build_preprocess_pipeline(nan_threshold=nan_threshold)
                preprocess.fit(X_train_raw)
                cols = preprocess.named_steps["nan_filter"].features_to_keep_
                X_train_per_mod[mod] = pd.DataFrame(preprocess.transform(X_train_raw), columns=cols)
                X_test_per_mod[mod] = pd.DataFrame(preprocess.transform(X_test_raw), columns=cols)
                # Set index to IDs to ensure proper alignment during concatenation
                X_train_per_mod[mod].index = groups_train
                X_test_per_mod[mod].index = groups_test

        # ============================
        # INNER-FOLD DATA PREPARATION (per-modality MW + concatenation)
        # ============================

        X_train_cat_list: List[pd.DataFrame] = []
        X_test_cat_list: List[pd.DataFrame] = []
        selected_per_mod: Dict[str, List[str]] = {}
        for mod in modalities:
            X_tr = X_train_per_mod[mod]
            X_te = X_test_per_mod[mod]
            if mod == "text":
                names = list(X_tr.columns)
                selected_per_mod[mod] = names
                X_train_cat_list.append(X_tr.copy())
                X_test_cat_list.append(X_te.copy())
            else:
                mw = MannWhitneySelector(n_features_to_select=min(mw_per_mod, X_tr.shape[1]))
                mw.fit(X_tr, y_train)
                names = mw.get_feature_names_out(list(X_tr.columns))
                selected_per_mod[mod] = names
                X_tr_sel = X_tr[names].copy()
                X_te_sel = X_te[names].copy()
                X_train_cat_list.append(X_tr_sel)
                X_test_cat_list.append(X_te_sel)
        X_train_cat = pd.concat(X_train_cat_list, axis=1)
        X_test_cat = pd.concat(X_test_cat_list, axis=1)
        feature_names_cat = list(X_train_cat.columns)
        n_max_rfe = X_train_cat.shape[1]
        if n_max_rfe < 1:
            raise ValueError(
                "No features after per-modality selection and concatenation. "
                "Check data loading, preprocessing (nan_threshold), and modality inputs."
            )

        selected_features_all[fold_idx] = {"per_modality": selected_per_mod}

        # ============================
        # START INNER FOLD (nested GroupKFold for HPO + RFE)
        # ============================

        cv_inner = GroupKFold(n_splits=inner_folds)
        for model_name, clf in models.items():
            print("-----------------------------------------\n")
            print(f"  [CLASSIFIER] {model_name}")
            
            best_inner_score = -1.0
            best_pipe = None
            grid_clf = param_grids_clf.get(model_name, {})

            # Fix for KNN: filter out n_neighbors values that exceed available training samples
            if model_name == "KNN" and "clf__n_neighbors" in grid_clf:
                # Calculate minimum training samples across all CV folds
                min_train_samples = float('inf')
                for train_idx, _ in cv_inner.split(X_train_cat, y_train, groups=groups_train):
                    min_train_samples = min(min_train_samples, len(train_idx))
                
                # Filter n_neighbors to only include values <= min_train_samples
                original_n_neighbors = grid_clf["clf__n_neighbors"]
                filtered_n_neighbors = [n for n in original_n_neighbors if n <= min_train_samples]
                
                if not filtered_n_neighbors:
                    # If all values are too large, use at least 1 (minimum valid value)
                    filtered_n_neighbors = [min(1, min_train_samples)]
                    print(f"  Warning: All n_neighbors values too large. Using {filtered_n_neighbors}")
                else:
                    print(f"  Adjusted n_neighbors from {original_n_neighbors} to {filtered_n_neighbors} "
                          f"(min_train_samples={min_train_samples})")
                
                grid_clf = grid_clf.copy()
                grid_clf["clf__n_neighbors"] = filtered_n_neighbors

            # ============================
            # MODEL TRAINING + HPO (inner loop over RFE sizes + GridSearchCV / CV)
            # ============================
            # Create list of RFE n values: 1, then 5, 10, 15, ..., up to n_max_rfe (step=5)
            rfe_n_values = sorted(set([1] + list(range(5, n_max_rfe + 1, 5)) + [n_max_rfe]))
            for n_rfe in rfe_n_values:
                inner_pipe = build_inner_pipeline_early_fusion(clone(clf), rfe_n=n_rfe)

                search = GridSearchCV(
                    inner_pipe, grid_clf, cv=cv_inner, scoring="f1_weighted", refit=True,
                    n_jobs=1, error_score="raise",
                )
                search.fit(X_train_cat, y_train, groups=groups_train)
                if search.best_score_ > best_inner_score:
                    best_inner_score = search.best_score_
                    best_pipe = search.best_estimator_

            if best_pipe is None:
                best_pipe = build_inner_pipeline_early_fusion(clone(clf), rfe_n=1)
                best_pipe.fit(X_train_cat, y_train)

            sel_names = extract_selected_features_early_fusion(best_pipe, feature_names_cat)
            selected_features_all[fold_idx][model_name] = sel_names

            # ============================
            # MODEL EVALUATION ON OUTER TEST SET
            # ============================
            test_metrics = evaluate_model(best_pipe, X_test_cat, y_test)
            all_results.append({
                "fold": fold_idx, "model": model_name,
                "f1_weighted": test_metrics["f1_weighted"],
                "accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "roc_auc": test_metrics["roc_auc"],
            })

            # Store per-fold confusion matrix components for this model
            fold_confusions.setdefault(fold_idx, {})
            fold_confusions[fold_idx][model_name] = {
                "tp": int(test_metrics.get("tp", 0)),
                "tn": int(test_metrics.get("tn", 0)),
                "fp": int(test_metrics.get("fp", 0)),
                "fn": int(test_metrics.get("fn", 0)),
            }

            if model_name not in cm_accum:
                cm_accum[model_name] = (test_metrics["y_true"], test_metrics["y_pred"])
            else:
                cm_accum[model_name] = (cm_accum[model_name][0] + test_metrics["y_true"], cm_accum[model_name][1] + test_metrics["y_pred"])

            if test_metrics.get("y_pred_proba") is not None:
                if model_name not in roc_accum:
                    roc_accum[model_name] = ([], [])
                roc_accum[model_name][0].extend(test_metrics["y_true"])
                roc_accum[model_name][1].extend(test_metrics["y_pred_proba"])

            print(f"    [F1-weighted]: {test_metrics['f1_weighted']:.3f}, [Num_selected features]: {len(sel_names)}")


        # ============================
        # SAVE SELECTED FEATURES (PER FOLD, PER MODEL)
        # ============================

        for model_name in models:
            model_dir = out_dir / model_name
            feat_path = model_dir / f"selected_features_fold{fold_idx}.json"
            # selected_features_all already contains only JSON-serializable objects (lists, dicts, strings, ints),
            # so we can save it directly without an extra helper function.
            to_save = {
                "per_modality": selected_features_all[fold_idx]["per_modality"],
                "selected_features": selected_features_all[fold_idx].get(model_name, []),
            "confusion_matrix": fold_confusions.get(fold_idx, {}).get(model_name, {}),
            }
            with open(feat_path, "w") as f:
                json.dump(to_save, f, indent=2)
        print(f"[INFO] Saved selected features per model for fold {fold_idx}")
        print("-----------------------------------------\n")

    # ============================
    # AGGREGATE & SAVE FINAL RESULTS (ALL FOLDS / MODELS)
    # ============================

    results_df = pd.DataFrame(all_results)
    # Save full results (all folds, all models) at root
    results_df.to_csv(out_dir / "all_results.csv", index=False)
    summary = results_df.groupby("model").agg(
        f1_weighted_mean=("f1_weighted", "mean"), f1_weighted_std=("f1_weighted", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
    ).reset_index()
    # Save complete selected features (all folds, all models) at root; use string keys for JSON
    selected_features_all_json = {str(k): v for k, v in selected_features_all.items()}
    with open(out_dir / "selected_features_all_folds.json", "w") as f:
        json.dump(selected_features_all_json, f, indent=2)
    # Write per-classifier outputs into each model's subfolder (summary, results, confusion matrix, ROC)
    for model_name in models:
        model_dir = out_dir / model_name
        model_results = results_df[results_df["model"] == model_name]
        model_summary = summary[summary["model"] == model_name]
        model_summary.to_csv(model_dir / "summary.csv", index=False)
        model_results.to_csv(model_dir / "results.csv", index=False)
        # Single-row final_results.csv summarizing metrics across folds for this model
        if not model_summary.empty:
            ms = model_summary.iloc[0]
            # Class balance based on all aligned samples (same for all models)
            n_pos = int((y == POS_LABEL).sum())
            n_neg = int((y == NEG_LABEL).sum())
            class_balance_str = f"{n_pos}/{n_neg}"
            final_row = pd.DataFrame([{
                "model": model_name,
                "class balance (1/0)": class_balance_str,
                "f1_weigted_avg": float(ms["f1_weighted_mean"]),
                "f1_weighted_std": float(ms["f1_weighted_std"]),
                "accuracy_avg": float(ms["accuracy_mean"]),
                "accuracy_std": float(ms["accuracy_std"]),
                "precision_avg": float(ms["precision_mean"]),
                "precision_std": float(ms["precision_std"]),
                "sensitivity_avg": float(ms["recall_mean"]),
                "sensitivity_std": float(ms["recall_std"]),
                "recall_avg": float(ms["recall_mean"]),
                "recall_std": float(ms["recall_std"]),
                "roc_auc_avg": float(ms["roc_auc_mean"]),
                "roc_auc_std": float(ms["roc_auc_std"]),
            }])
            final_row.to_csv(model_dir / "final_results.csv", index=False)
        # Confusion matrix (aggregated over all outer folds) — save for every model
        y_true_cm = np.array(cm_accum[model_name][0])
        y_pred_cm = np.array(cm_accum[model_name][1])
        plot_confusion_matrix(
            y_true_cm, y_pred_cm, model_name,
            normalization="count", save_path=model_dir / "confusion_matrix.png",
        )
        # ROC curve (aggregated over all outer folds) — save for every model
        roc_path = model_dir / "roc_curve.png"
        if model_name in roc_accum and roc_accum[model_name][0] and roc_accum[model_name][1]:
            plot_roc_curves(
                {model_name: (roc_accum[model_name][0], roc_accum[model_name][1])},
                normalization="early_fusion", save_path=roc_path,
            )
        else:
            # Placeholder when model has no probability estimates (e.g. no predict_proba)
            plt.figure(figsize=(6, 5))
            plt.text(0.5, 0.5, "ROC curve not available\n(model does not provide probability estimates)",
                     ha="center", va="center", fontsize=12, wrap=True)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis("off")
            plt.title(f"ROC — {model_name}")
            plt.tight_layout()
            plt.savefig(roc_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[INFO] Saved ROC placeholder to {roc_path}")
        print("-----------------------------------------\n")
    # Config at root
    cfg_dict = {k: getattr(cfg, k) for k in ["fusion_modalities", "condition", "phases", "minutes", "outer_folds", "inner_folds", "nan_threshold", "mannwhitney_per_modality", "aggregation_method", "text_data_path"] if hasattr(cfg, k)}
    # Add global class balance information based on aligned labels
    n_pos_total = int((y == POS_LABEL).sum())
    n_neg_total = int((y == NEG_LABEL).sum())
    class_balance_ratio_total = float(n_pos_total / n_neg_total) if n_neg_total > 0 else float("nan")
    cfg_dict["class_balance"] = {
        "n_pos": n_pos_total,
        "n_neg": n_neg_total,
        "ratio_1_over_0": class_balance_ratio_total,
    }
    for k, v in list(cfg_dict.items()):
        if isinstance(v, (Path, set)):
            cfg_dict[k] = str(v)
        elif isinstance(v, np.ndarray):
            cfg_dict[k] = v.tolist()
    with open(out_dir / "args.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)
    print(f"[INFO] Early fusion done. Summary:\n{summary}")
    print(f"[INFO] Results and selected features saved under {out_dir} (one subfolder per classifier: {', '.join(models)})")
    print("-----------------------------------------\n")



def test_main(cfg=None):
    """Print config dashboard (and optionally run full pipeline). Use cfg if provided, else parse args."""
    if cfg is None:
        cfg = parse_args()

    modalities = [m.strip().lower() for m in cfg.fusion_modalities]
    condition = cfg.condition
    phases = cfg.phases if isinstance(cfg.phases, list) else [cfg.phases] # a list of phases or only 'all'
    aggregation_method = cfg.aggregation_method # phase or ID
    opensmile_dir = Path(cfg.opensmile_data_dir) 
    opendbm_dir = Path(cfg.opendbm_data_dir)
    text_path = Path(getattr(cfg, "text_data_path", ""))
    print("Dashboard:")
    print(f"Modalities: {modalities}")
    print(f"Condition: {condition}")
    print(f"Phases: {phases}")
    print(f"Aggregation method: {aggregation_method}")
    print(f"OpenSmile data directory: {opensmile_dir}")
    print(f"OpenDBM data directory: {opendbm_dir}")
    print(f"Text data path: {text_path}")
    print(f"Outer folds: {cfg.outer_folds}")
    print(f"Inner folds: {cfg.inner_folds}")
    print(f"Nan threshold: {cfg.nan_threshold}")
    print(f"Mann-Whitney per modality: {cfg.mannwhitney_per_modality}")
    print(f"Output directory: {cfg.output_dir}")
    print("-----------------------------------------\n")


     # Load data for each modality and align by sample key:
    modality_data: Dict[str, Dict] = {}

    for mod in modalities:
        if mod == "text":
            df = load_text_data(text_path, condition=condition)
        elif mod == "audio":
            df = load_audio_data(condition, phases, opensmile_dir, aggregation_method=aggregation_method)
        elif mod == "video":
            df = load_video_data(condition, phases, opendbm_data_dir=opendbm_dir, aggregation_method=aggregation_method)
        elif mod in ("ecg", "emg", "rsp"):
            dtype = {"ecg": "ECG", "emg": "EMG", "rsp": "RSP"}[mod]
            df = load_biosig_data(
                dtype,
                cfg.minutes,
                condition,
                phases,
                aggregation_method=aggregation_method,
            )
        else:
            raise ValueError(f"Unknown modality: {mod}")

        drop_cols = [c for c in ["label", "ID", "phase", "condition", "aufgabe", "Aufgabe"] if c in df.columns]
        X = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
        modality_data[mod] = {
            "X": X,
            "y": df["label"].to_numpy(dtype=int),
            "ID": df["ID"].astype(str).to_numpy(),
            "df": df,
        }

    for mod in modalities:

        d = modality_data[mod]
        print(f"Modality: {mod}")
        print(f"Labels: {d['y']}")
        print(f"IDs: {d['ID']}")
        print(f"DataFrame shape: {d['df'].shape}")
        print(f"X Shape: {d['X'].shape}")
        # all columns except X:
        print(f"list of features: {[c for c in d["df"].columns if c not in d["X"].columns]}") 
        print("-----------------------------------------\n")

    # Please write a code here to find common IDs across all modalities
    common_IDs_1 = set(modality_data[modalities[0]]["df"]["ID"].astype(str))
    for mod in modalities[1:]:
        ids_mod = set(modality_data[mod]["df"]["ID"].astype(str))
        common_IDs_1 &= ids_mod
    common_IDs_1 = sorted(common_IDs_1)
    print(f"Number of common IDs_1: {len(common_IDs_1)}")
    if not common_IDs_1:
        raise ValueError("No common samples across modalities")

    # Find common IDs across all modalities (simple intersection)
    common_IDs = set(modality_data[modalities[0]]["df"]["ID"].astype(str))
    for mod in modalities[1:]:
        ids_mod = set(modality_data[mod]["df"]["ID"].astype(str))
        common_IDs &= ids_mod
    common_IDs = sorted(common_IDs)
    print(f"Number of common IDs: {len(common_IDs)}")
    if not common_IDs:
        raise ValueError("No common samples across modalities")

    # Re-align modalities: filter to common IDs, sort by ID, one row per participant
    # Horizontal concatenation keyed by ID (same row order across modalities)
    meta_cols = ["label", "ID", "phase", "condition", "aufgabe", "Aufgabe"]
    for mod in modalities:
        d = modality_data[mod]
        df_mod = d["df"].copy()
        df_mod["ID"] = df_mod["ID"].astype(str)
        mask = df_mod["ID"].isin(common_IDs)
        df_mod = df_mod[mask].copy()
        # If multiple rows per ID (e.g. by_phase with multiple phases), take first per ID
        if df_mod.duplicated(subset="ID").any():
            df_mod = df_mod.groupby("ID", sort=True).first().reset_index()
        df_mod = df_mod.set_index("ID").loc[common_IDs].reset_index()
        drop_cols = [c for c in meta_cols if c in df_mod.columns]
        X_mod = df_mod.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
        d["X_aligned"] = X_mod
        d["y_aligned"] = df_mod["label"].to_numpy(dtype=int)
        d["groups_aligned"] = df_mod["ID"].to_numpy()

    

    """
    # ============================
    # GLOBAL LABELS / GROUPS SETUP
    # ============================
    print("-----------------------------------------\n")
    print("Global labels / groups setup")
    print("-----------------------------------------\n")
    ref_mod = modalities_with_phase[0] if modalities_with_phase else modalities[0]
    y = modality_data[ref_mod]["y_aligned"]
    groups = modality_data[ref_mod]["groups_aligned"]
    outer_folds = cfg.outer_folds
    inner_folds = cfg.inner_folds
    nan_threshold = cfg.nan_threshold
    mw_per_mod = getattr(cfg, "mannwhitney_per_modality", 20)
    phases = cfg.phases if isinstance(cfg.phases, list) else [cfg.phases]

    mod_str = "_".join(sorted(modalities))
    phases_str = "_".join(phases)
    agg_suffix = "byPhase" if cfg.aggregation_method == "by_phase" else "by_ID"
    out_dir = Path(cfg.output_dir) if cfg.output_dir else Path(__file__).resolve().parent / f"Early_Fusion_{mod_str}_{cfg.condition}_{phases_str}_{agg_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"early_fusion_{mod_str}_{cfg.condition}_{phases_str}"

    n_outer = outer_folds
    if len(np.unique(groups)) < n_outer:
        raise ValueError(f"Not enough groups for {n_outer}-fold GroupKFold")

    # ============================
    # MODEL SETUP (CLASSIFIERS + PARAM GRIDS)
    # ============================
    print("-----------------------------------------\n")
    print("Model setup")
    print("-----------------------------------------\n")
    models = get_classifier_models()
    if cfg.classifiers:
        models = {k: v for k, v in models.items() if k in cfg.classifiers}
    if not models:
        raise ValueError("No valid classifiers selected")
    param_grids_clf = get_param_grids_clf_only()

    # Create one subfolder per classifier (e.g. KNN, RandomForest, LogisticRegression)
    for model_name in models:
        (out_dir / model_name).mkdir(parents=True, exist_ok=True)

    # ============================
    # START OUTER FOLD (GroupKFold CV)
    # ============================

    kf_outer = GroupKFold(n_splits=n_outer)
    all_results: List[Dict] = []
    selected_features_all: Dict[int, Dict[str, List[str]]] = {}
    cm_accum: Dict[str, Tuple[List[int], List[int]]] = {}
    roc_accum: Dict[str, Tuple[List[int], List[float]]] = {}  # (y_true, y_proba) per model for ROC

    for fold_idx, (train_idx, test_idx) in enumerate(kf_outer.split(y, y, groups=groups), start=1):
        print(f"\n[INFO] Early fusion outer fold {fold_idx}/{n_outer}")
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

    """


if __name__ == "__main__":
    cfg = parse_args()
    if cfg.run_type == "test_main":
        test_main(cfg)
    else:
        main(cfg)