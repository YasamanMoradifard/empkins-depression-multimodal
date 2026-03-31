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


def build_inner_pipeline_late_fusion(clf, mw_k: int = 25, rfe_n: int = 10) -> Pipeline:
    """Inner CV: Mann-Whitney top mw_k -> RFE select rfe_n -> classifier. rfe_n tuned via GridSearchCV.
    RFE always uses DecisionTreeClassifier for consistent, fast feature selection across all final classifiers."""
    rfe_estimator = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
    return Pipeline([
        ("mw_selector", MannWhitneySelector(n_features_to_select=mw_k)),
        ("rfe_selector", RFE(estimator=rfe_estimator, n_features_to_select=rfe_n, step=1)),
        ("clf", clf),
    ])


def build_full_late_fusion_pipeline(
    nan_threshold: float,
    mw_k: int,
    rfe_n: int,
    clf: object,
) -> Pipeline:
    """Full late fusion pipeline: preprocess (ConstantFilter, NaN, impute, scale) + Mann-Whitney(mw_k) + RFE(rfe_n) + classifier. For use with run_late_fusion_cv."""
    preprocess = build_preprocess_pipeline(nan_threshold=nan_threshold)
    inner = build_inner_pipeline_late_fusion(clone(clf), mw_k=mw_k, rfe_n=rfe_n)
    steps = list(preprocess.named_steps.items()) + list(inner.named_steps.items())
    return Pipeline(steps=steps)


def extract_selected_features_late_fusion(pipe: Pipeline, feature_names: List[str]) -> List[str]:
    """Extract final selected feature names after MW + RFE from a fitted inner pipeline."""
    names = list(feature_names)
    if "mw_selector" in pipe.named_steps:
        mw = pipe.named_steps["mw_selector"]
        if hasattr(mw, "selected_features_") and mw.selected_features_ is not None:
            names = [names[i] for i in mw.selected_features_ if i < len(names)]
    if "rfe_selector" in pipe.named_steps:
        rfe = pipe.named_steps["rfe_selector"]
        if hasattr(rfe, "support_"):
            names = [f for f, s in zip(names, rfe.support_) if s]
    return names


# =============================================================================
# Late fusion CV (standalone API): align, weights, OOF, main CV
# =============================================================================


def align_proba_to_classes(
    proba: np.ndarray,
    estimator_classes: np.ndarray,
    all_classes: np.ndarray,
) -> np.ndarray:
    """
    Reorder or pad predict_proba columns to match a global class order.
    Missing classes in estimator_classes get zeros. Returns (n_samples, len(all_classes)).
    """
    num_samples = proba.shape[0]
    num_global_classes = len(all_classes)
    out = np.zeros((num_samples, num_global_classes), dtype=proba.dtype)
    # Map estimator class label -> column index in proba
    class_to_estimator_idx = {c: i for i, c in enumerate(estimator_classes)}
    for global_idx, class_label in enumerate(all_classes):
        if class_label in class_to_estimator_idx:
            estimator_col = class_to_estimator_idx[class_label]
            out[:, global_idx] = proba[:, estimator_col]
    return out


def compute_fusion_weights(
    f1_scores: Dict[str, float],
    gamma: float,
    epsilon: float,
    ) -> Dict[str, float]:
    """
    Performance-proportional fusion weights without shrinkage.
    raw_weight[m] = max(epsilon, f1[m])^gamma; weights = raw / sum(raw).
    Returns a dict of weights that sum to 1.
    """
    modality_names = list(f1_scores.keys())
    # Raw weights: max(epsilon, f1)^gamma per modality
    raw_weights = []
    for mod in modality_names:
        score = max(epsilon, f1_scores[mod])
        raw_weights.append(score ** gamma)
    raw_sum = sum(raw_weights)
    if raw_sum == 0:
        # Fall back to uniform weights if all scores are zero
        uniform = 1.0 / len(modality_names)
        return {mod: uniform for mod in modality_names}
    # Normalized performance-proportional weights
    weights = {mod: raw_weights[i] / raw_sum for i, mod in enumerate(modality_names)}
    return weights


def _get_train_subset(
    X: Union[pd.DataFrame, np.ndarray],
    indices: np.ndarray,
) -> Union[pd.DataFrame, np.ndarray]:
    """Return X subset by indices (DataFrame or array)."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[indices]
    return X[indices]


def _cap_knn_n_neighbors(estimator, n_samples: int) -> None:
    """
    Cap KNN n_neighbors to not exceed n_samples (avoids ValueError when fitting
    KNN on small inner folds). Mirrors the fix in MultiModal_Early_Classification.
    """
    if n_samples <= 0:
        return
    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            _cap_knn_n_neighbors(step, n_samples)
        return
    if hasattr(estimator, "estimator") and hasattr(estimator, "param_grid"):
        # GridSearchCV: cap inner estimator and filter param_grid
        _cap_knn_n_neighbors(estimator.estimator, n_samples)
        param_grid = estimator.param_grid
        if isinstance(param_grid, dict) and "clf__n_neighbors" in param_grid:
            orig = param_grid["clf__n_neighbors"]
            filtered = [n for n in orig if n <= n_samples]
            if not filtered:
                filtered = [min(1, n_samples)]
            param_grid["clf__n_neighbors"] = filtered
        return
    if isinstance(estimator, KNeighborsClassifier):
        if estimator.n_neighbors > n_samples:
            estimator.n_neighbors = min(estimator.n_neighbors, max(1, n_samples))


def get_oof_probabilities(
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    estimator: object,
    inner_splits: int,
    all_classes: np.ndarray,
) -> np.ndarray:
    """
    Run GroupKFold(inner_splits) on train_indices only; for each inner fold fit on
    inner-train, predict_proba on inner-val, align to all_classes, fill OOF array.
    Returns array of shape (len(train_indices), len(all_classes)).
    """
    num_train = len(train_indices)
    num_classes = len(all_classes)
    oof_proba = np.zeros((num_train, num_classes), dtype=np.float64)
    oof_proba[:] = np.nan

    X_train = _get_train_subset(X, train_indices)
    y_train = y[train_indices]
    groups_train = groups[train_indices]

    inner_cv = GroupKFold(n_splits=inner_splits)
    splits = list(inner_cv.split(X_train, y_train, groups_train))

    for fold_idx, (inner_train_idx, inner_val_idx) in enumerate(splits):
        # Get inner train/val data
        X_inner_train = _get_train_subset(X_train, inner_train_idx)
        y_inner_train = y_train[inner_train_idx]
        X_inner_val = _get_train_subset(X_train, inner_val_idx)

        cloned = clone(estimator)
        n_inner_train = len(y_inner_train)
        _cap_knn_n_neighbors(cloned, n_inner_train)
        cloned.fit(X_inner_train, y_inner_train)

        if hasattr(cloned, "predict_proba"):
            proba_val = cloned.predict_proba(X_inner_val)
            estimator_classes = np.asarray(cloned.classes_)
            proba_aligned = align_proba_to_classes(proba_val, estimator_classes, all_classes)
        else:
            pred_val = cloned.predict(X_inner_val)
            proba_aligned = np.zeros((len(inner_val_idx), num_classes))
            for j, c in enumerate(all_classes):
                proba_aligned[:, j] = (pred_val == c).astype(float)

        for local_row, global_row in enumerate(inner_val_idx):
            oof_proba[global_row, :] = proba_aligned[local_row, :]

    return oof_proba


def run_late_fusion_cv(
    X_by_modality: Dict[str, Union[pd.DataFrame, np.ndarray]],
    y: np.ndarray,
    groups: np.ndarray,
    build_estimator_fn: Callable[[str], object],
    outer_splits: int = 5,
    inner_splits: int = 3,
    gamma: float = 2.0,
    epsilon: float = 1e-4,
) -> Dict:
    """
    Late fusion CV (binary only: depressed vs healthy): outer GroupKFold, OOF probabilities
    per modality on outer-train, fusion weights from OOF F1, threshold tune on fused OOF,
    evaluate fused on outer-test. build_estimator_fn(modality_name) returns an unfitted
    sklearn-like estimator with fit and predict_proba.
    """
    # === STEP 0: Setup ===
    all_classes = np.unique(y)
    num_classes = len(all_classes)
    modality_names = list(X_by_modality.keys())
    num_samples = len(y)
    print(f"[Late fusion] samples={num_samples}, classes={all_classes.tolist()}, modalities={modality_names} (binary)")

    outer_cv = GroupKFold(n_splits=outer_splits)
    per_fold: List[Dict] = []
    all_f1: List[float] = []
    all_accuracy: List[float] = []
    all_sensitivity: List[float] = []
    all_specificity: List[float] = []
    all_precision: List[float] = []
    all_recall: List[float] = []
    all_roc_auc: List[float] = []
    all_best_threshold: List[float] = []

    for fold_index, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(np.zeros(num_samples), y, groups=groups)
    ):
        print(f"=== Outer Fold {fold_index + 1}/{outer_splits} ===")
        num_outer_train = len(outer_train_idx)
        num_outer_test = len(outer_test_idx)
        groups_train = groups[outer_train_idx]
        groups_test = groups[outer_test_idx]
        num_unique_groups_train = len(np.unique(groups_train))
        num_unique_groups_test = len(np.unique(groups_test))
        print(f"  Train samples: {num_outer_train}, test samples: {num_outer_test}")
        print(f"  Unique groups (train): {num_unique_groups_train}, (test): {num_unique_groups_test}")

        # Binary: positive class (e.g. depressed) and its column index
        pos_label = int(all_classes[-1])
        pos_col = np.where(all_classes == pos_label)[0][0]

        # === STEP 2: Inner loop to get OOF probabilities per modality ===
        oof_probs_by_modality: Dict[str, np.ndarray] = {}
        unimodal_oof_f1: Dict[str, float] = {}
        unimodal_oof_metrics: Dict[str, Dict] = {}
        y_outer_train = y[outer_train_idx]
        for mod in modality_names:
            X_mod = X_by_modality[mod]
            estimator = build_estimator_fn(mod)
            oof_proba = get_oof_probabilities(
                X_mod, y, groups, outer_train_idx, estimator, inner_splits, all_classes
            )
            oof_probs_by_modality[mod] = oof_proba
            # OOF predictions (binary: threshold 0.5 on positive-class prob)
            oof_pred_labels = np.where(oof_proba[:, pos_col] >= 0.5, pos_label, all_classes[0])
            # OOF metrics: F1-weighted, Accuracy, Precision, Recall, Sensitivity, confusion matrix
            f1_w = float(f1_score(y_outer_train, oof_pred_labels, average="weighted", zero_division=0))
            unimodal_oof_f1[mod] = f1_w
            acc = float(accuracy_score(y_outer_train, oof_pred_labels))
            prec = float(precision_score(y_outer_train, oof_pred_labels, pos_label=pos_label, zero_division=0))
            rec = float(recall_score(y_outer_train, oof_pred_labels, pos_label=pos_label, zero_division=0))
            sens = rec  # sensitivity = recall for positive class
            cm_oof = confusion_matrix(y_outer_train, oof_pred_labels, labels=all_classes)
            unimodal_oof_metrics[mod] = {
                "f1_weighted": f1_w,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "sensitivity": sens,
                "confusion_matrix": cm_oof.tolist(),
            }
            print(f"  {mod}: inner CV done, OOF F1-weighted={f1_w:.4f} Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} Sens={sens:.4f}")

        # === STEP 3: Compute fusion weights (no shrinkage) ===
        weights_final = compute_fusion_weights(
            unimodal_oof_f1, gamma=gamma, epsilon=epsilon
        )
        print("  fusion weights (rounded):", {k: round(v, 3) for k, v in weights_final.items()})

        # === STEP 4: Use fixed decision threshold (no OOF tuning) ===
        # We keep the learned modality weights, but apply a fixed 0.5 threshold
        # on the fused positive-class probability for classification.
        print(f"  Using fixed decision threshold: 0.5")

        # === STEP 5: Evaluate on outer test ===
        test_probs_by_modality: Dict[str, np.ndarray] = {}
        selected_features_this_fold: Dict[str, List[str]] = {}
        for mod in modality_names:
            X_mod = X_by_modality[mod]
            X_outer_train = _get_train_subset(X_mod, outer_train_idx)
            X_outer_test = _get_train_subset(X_mod, outer_test_idx)
            y_outer_train = y[outer_train_idx]
            estimator = build_estimator_fn(mod)
            fitted = clone(estimator)
            fitted.fit(X_outer_train, y_outer_train)
            # Optionally extract RFE-selected feature names for this modality (if pipeline has mw + rfe)
            if hasattr(fitted, "named_steps") and "nan_filter" in fitted.named_steps and "mw_selector" in fitted.named_steps and "rfe_selector" in fitted.named_steps:
                try:
                    feature_names_after_preprocess = list(fitted.named_steps["nan_filter"].features_to_keep_)
                    selected_features_this_fold[mod] = extract_selected_features_late_fusion(fitted, feature_names_after_preprocess)
                except Exception:
                    selected_features_this_fold[mod] = []
            if hasattr(fitted, "predict_proba"):
                proba_test = fitted.predict_proba(X_outer_test)
                est_classes = np.asarray(fitted.classes_)
                test_probs_by_modality[mod] = align_proba_to_classes(
                    proba_test, est_classes, all_classes
                )
            else:
                pred_test = fitted.predict(X_outer_test)
                arr = np.zeros((len(outer_test_idx), num_classes))
                for j, c in enumerate(all_classes):
                    arr[:, j] = (pred_test == c).astype(float)
                test_probs_by_modality[mod] = arr

        fused_test_proba = np.zeros((num_outer_test, num_classes))
        for mod in modality_names:
            fused_test_proba += weights_final[mod] * test_probs_by_modality[mod]

        probs_pos = fused_test_proba[:, pos_col]
        fused_pred = np.where(probs_pos >= 0.5, pos_label, all_classes[0])

        y_outer_test = y[outer_test_idx]
        fused_f1 = float(f1_score(y_outer_test, fused_pred, average="weighted", zero_division=0))
        fused_acc = float(accuracy_score(y_outer_test, fused_pred))
        fused_prec = float(precision_score(y_outer_test, fused_pred, pos_label=pos_label, zero_division=0))
        fused_rec = float(recall_score(y_outer_test, fused_pred, pos_label=pos_label, zero_division=0))
        cm = confusion_matrix(y_outer_test, fused_pred, labels=all_classes)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
            fused_sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            fused_spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        else:
            fused_sens = fused_rec  # sensitivity = recall for positive class
            fused_spec = float("nan")
        try:
            fused_roc = float(roc_auc_score(y_outer_test, probs_pos))
        except Exception:
            fused_roc = float("nan")
        print(f"  Fused test F1-weighted = {fused_f1:.4f}")
        all_f1.append(fused_f1)
        all_accuracy.append(fused_acc)
        all_sensitivity.append(fused_sens)
        all_specificity.append(fused_spec)
        all_precision.append(fused_prec)
        all_recall.append(fused_rec)
        all_roc_auc.append(fused_roc)

        # === STEP 6: Store results for this fold ===
        fold_result: Dict = {
            "fold": fold_index,
            "unimodal_oof_f1": dict(unimodal_oof_f1),
            "unimodal_oof_metrics": {mod: dict(metrics) for mod, metrics in unimodal_oof_metrics.items()},
            "weights_final": dict(weights_final),
            "fused_test_f1_weighted": fused_f1,
            "fused_test_accuracy": fused_acc,
            "fused_test_sensitivity": fused_sens,
            "fused_test_specificity": fused_spec,
            "fused_test_precision": fused_prec,
            "fused_test_recall": fused_rec,
            "fused_test_roc_auc": fused_roc,
            "confusion_matrix": cm.tolist(),
            "y_true": y_outer_test.tolist(),
            "y_pred": fused_pred.tolist(),
            "y_proba": probs_pos.tolist(),
        }
        if selected_features_this_fold:
            fold_result["selected_features"] = dict(selected_features_this_fold)
        per_fold.append(fold_result)

    # Summary across folds (use nanmean/nanstd for metrics that may be NaN in some folds)
    f1_mean = float(np.mean(all_f1))
    f1_std = float(np.std(all_f1))
    acc_mean = float(np.mean(all_accuracy))
    acc_std = float(np.std(all_accuracy))
    sens_mean = float(np.nanmean(all_sensitivity))
    sens_std = float(np.nanstd(all_sensitivity))
    spec_mean = float(np.nanmean(all_specificity))
    spec_std = float(np.nanstd(all_specificity))
    prec_mean = float(np.nanmean(all_precision))
    prec_std = float(np.nanstd(all_precision))
    rec_mean = float(np.nanmean(all_recall))
    rec_std = float(np.nanstd(all_recall))
    roc_auc_mean = float(np.nanmean(all_roc_auc))
    roc_auc_std = float(np.nanstd(all_roc_auc))

    print("\n--- Summary ---")
    print("  Metric         | Mean    | Std")
    print("  ---------------|---------|--------")
    print(f"  F1-weighted    | {f1_mean:.4f}  | {f1_std:.4f}")
    print(f"  Accuracy       | {acc_mean:.4f}  | {acc_std:.4f}")
    print(f"  Sensitivity    | {sens_mean:.4f}  | {sens_std:.4f}")
    print(f"  Specificity    | {spec_mean:.4f}  | {spec_std:.4f}")
    print(f"  Precision      | {prec_mean:.4f}  | {prec_std:.4f}")
    print(f"  Recall         | {rec_mean:.4f}  | {rec_std:.4f}")
    print(f"  ROC-AUC        | {roc_auc_mean:.4f}  | {roc_auc_std:.4f}")

    return {
        "per_fold": per_fold,
        "aggregate": {
            "f1_weighted_mean": f1_mean,
            "f1_weighted_std": f1_std,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "sensitivity_mean": sens_mean,
            "sensitivity_std": sens_std,
            "specificity_mean": spec_mean,
            "specificity_std": spec_std,
            "precision_mean": prec_mean,
            "precision_std": prec_std,
            "recall_mean": rec_mean,
            "recall_std": rec_std,
            "roc_auc_mean": roc_auc_mean,
            "roc_auc_std": roc_auc_std,
        },
    }

# =============================================================================
# Evaluation and Plotting Functions
# =============================================================================

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> Dict[str, float]:
    """
    Compute comprehensive metrics from pre-computed predictions (for late fusion
    where we have predictions but no single model).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[NEG_LABEL, POS_LABEL])
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
    metrics["y_true"] = y_true.tolist()
    metrics["y_pred"] = y_pred.tolist()
    metrics["y_pred_proba"] = y_proba.tolist() if y_proba is not None else None
    return metrics


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
# Main Function (Late Fusion Only)
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="MultiModal ML: late fusion")
    p.add_argument(
        "--fusion_modalities",
        nargs="+",
        choices=["audio", "video", "ecg", "emg", "rsp", "text"],
        help="Modalities to use in late fusion. Valid: audio, video, ecg, emg, rsp, text",
    )

    p.add_argument("--condition", type=str, default="all", help="Condition filter: CR, ADK, CRADK, SHAM, all") # CR, ADK, CRADK, SHAM, all
    p.add_argument("--phases", nargs="+", default=["training_neg"], help="Phases to load (standard names)") # training_pos, training_neg, testing_pos, testing_neg
    p.add_argument("--minutes", type=int, default=3, help="BioSig window minutes (1, 3, 5)") # 1, 3, 5

    # Audio-specific arguments
    p.add_argument(
        "--opensmile_data_dir",
        type=str,
        default="/home/vault/empkins/tpD/D02/Students/Yasaman/3_Audio_data/Data/OpenSmile_data",
        help="Directory with OpenSmile CSVs (for audio modality)",
    )
    # Video-specific arguments
    p.add_argument(
        "--opendbm_data_dir",
        type=str,
        default="/home/vault/empkins/tpD/D02/processed_data/processed_openDBM_functional",
        help="Directory with OpenDBM CSVs (for video modality)",
    )
    # Text-specific arguments
    p.add_argument(
        "--text_data_csv",
        type=str,
        default="/home/vault/empkins/tpD/D02/Students/Yasaman/4_Text_data/Data/text_features_3370.csv", # 3370 had the best results
        help="CSV file with text features (used when modality 'text' is selected; aggregation by ID only).",
    )

    p.add_argument(
        "--aggregation_method",
        type=str,
        default="by_phase",
        choices=["by_phase", "by_ID"],
        help="Aggregation method for modalities",
    )

    p.add_argument("--classifiers", nargs="*", default='LogisticRegression') # LogisticRegression, SVC_RBF, KNN, RandomForest, DecisionTree, XGBoost, AdaBoost
    p.add_argument("--outer_folds", type=int, default=5, help="Outer GroupKFold splits")
    p.add_argument("--inner_folds", type=int, default=3, help="Inner GroupKFold splits for HPO")

    p.add_argument("--nan_threshold", type=float, default=0.69,
                   help="Drop features with more than this fraction of NaN (rest imputed)")
    p.add_argument("--mannwhitney_k", type=int, default=50,
                   help="Late fusion: top K from Mann-Whitney (default 25), then RFE selects rfe_n")
    p.add_argument("--rfe_n", type=int, default=25, help="Late fusion: number of features to select by RFE after Mann-Whitney")

    p.add_argument("--output_dir", type=str, default="/home/vault/empkins/tpD/D02/Students/Yasaman/5_MultiModal_ML/Late Fusion Classification Results", help="Output directory (default: Late_Fusion_* under script dir)")
    args = p.parse_args()
    return args


def main(cfg) -> None:
    """
    Late fusion via run_late_fusion_cv (standalone API): load modalities, prefix feature names
    (Video_, Audio_, ECG_, EMG_, RSP_, Text_), align by participant ID (one row per ID), then
    run_late_fusion_cv with weighted probability fusion (fixed 0.5 decision threshold).
    Pipeline per modality: preprocess + Mann-Whitney(mw_k) + RFE(rfe_n) + classifier.
    Saves aggregate summary, per-fold results, and RFE-selected features per fold in JSON/CSV.
    """
    modalities = [m.strip().lower() for m in cfg.fusion_modalities]
    outer_folds = cfg.outer_folds
    inner_folds = cfg.inner_folds
    nan_threshold = cfg.nan_threshold
    mw_k = cfg.mannwhitney_k
    rfe_n = getattr(cfg, "rfe_n", 10)
    opensmile_dir = Path(cfg.opensmile_data_dir)
    opendbm_dir = Path(cfg.opendbm_data_dir)
    condition = cfg.condition
    phases = cfg.phases if isinstance(cfg.phases, list) else [cfg.phases]
    aggregation_method = cfg.aggregation_method

    has_text = "text" in modalities
    base_modalities = [m for m in modalities if m != "text"]

    if not base_modalities:
        raise ValueError("At least one base modality (audio, video, ecg, emg, rsp) is required. Text modality alone is not supported.")

    mod_str = "_".join(sorted(modalities))
    phases_str = "_".join(phases)
    agg_suffix = "byPhase" if aggregation_method == "by_phase" else "by_ID"
    base_dir = Path(cfg.output_dir) if cfg.output_dir else Path(__file__).resolve().parent
    out_dir = base_dir / f"Late_Fusion_{mod_str}_{cfg.condition}_{phases_str}_{agg_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"late_fusion_{mod_str}_{cfg.condition}_{phases_str}"

    # Load modality data (audio, video, biosignals)
    modality_data: Dict[str, Dict] = {}
    for mod in base_modalities:
        if mod == "audio":
            df = load_audio_data(condition, phases, opensmile_dir, aggregation_method=aggregation_method)
        elif mod == "video":
            df = load_video_data(condition, phases, opendbm_dir, aggregation_method=aggregation_method)
        elif mod in ("ecg", "emg", "rsp"):
            dtype = {"ecg": "ECG", "emg": "EMG", "rsp": "RSP"}[mod]
            df = load_biosig_data(dtype, cfg.minutes, condition, phases, aggregation_method=aggregation_method)
        else:
            raise ValueError(f"Unknown modality: {mod}")
        drop_cols = [c for c in ["label", "ID", "phase", "condition", "aufgabe", "Aufgabe"] if c in df.columns]
        X = df.drop(columns=drop_cols, errors="ignore").select_dtypes(include=[np.number])
        y = df["label"].to_numpy(dtype=int)
        groups = df["ID"].astype(str).to_numpy()
        modality_data[mod] = {"X": X, "y": y, "groups": groups, "df": df}

    # Load text modality (by ID only, no aggregation), then merge labels/condition from first base modality
    if has_text:
        text_path = Path(cfg.text_data_csv)
        df_text = load_text_data(text_path, condition=condition)
        label_source_mod = base_modalities[0]
        base_df = modality_data[label_source_mod]["df"]
        if "ID" not in base_df.columns or "label" not in base_df.columns:
            raise ValueError(f"Base modality '{label_source_mod}' must contain 'ID' and 'label' columns to attach labels to text data.")
        label_cols = ["ID", "label"]
        if "condition" in base_df.columns:
            label_cols.append("condition")
        label_df = base_df[label_cols].drop_duplicates(subset=["ID"])
        # Drop label/condition from text before merge to avoid duplicate columns; use base modality's labels
        df_text = df_text.drop(columns=[c for c in ["label", "condition"] if c in df_text.columns], errors="ignore")
        df_text = df_text.merge(label_df, on="ID", how="inner")
        if df_text.empty:
            raise ValueError("No overlapping participants between text data and base modalities.")
        drop_cols_text = [c for c in ["label", "ID", "phase", "condition", "aufgabe", "Aufgabe"] if c in df_text.columns]
        X_text = df_text.drop(columns=drop_cols_text, errors="ignore").select_dtypes(include=[np.number])
        y_text = df_text["label"].to_numpy(dtype=int)
        groups_text = df_text["ID"].astype(str).to_numpy()
        modality_data["text"] = {"X": X_text, "y": y_text, "groups": groups_text, "df": df_text}

    effective_modalities = base_modalities + (["text"] if has_text else [])

    # Align by participant ID (one row per ID across all modalities)
    common_IDs = set(modality_data[effective_modalities[0]]["df"]["ID"].astype(str))
    for mod in effective_modalities[1:]:
        ids_mod = set(modality_data[mod]["df"]["ID"].astype(str))
        common_IDs &= ids_mod
    common_IDs = sorted(common_IDs)
    if not common_IDs:
        raise ValueError("No common samples across modalities")

    # Prefix for feature names per modality (e.g. Video_featureName, RSP_featureName)
    modality_prefix: Dict[str, str] = {"video": "Video", "audio": "Audio", "ecg": "ECG", "emg": "EMG", "rsp": "RSP", "text": "Text"}

    meta_cols = ["label", "ID", "phase", "condition", "aufgabe", "Aufgabe"]
    for mod in effective_modalities:
        d = modality_data[mod]
        df_mod = d["df"].copy()
        df_mod["ID"] = df_mod["ID"].astype(str)
        mask = df_mod["ID"].isin(common_IDs)
        df_mod = df_mod[mask].copy()
        # If multiple rows per ID remain (e.g. by_phase outputs), collapse to one row per ID
        if df_mod.duplicated(subset="ID").any():
            df_mod = df_mod.groupby("ID", sort=True).first().reset_index()
        df_mod = df_mod.set_index("ID").loc[common_IDs].reset_index()

        X = df_mod.drop(columns=[c for c in meta_cols if c in df_mod.columns], errors="ignore").select_dtypes(include=[np.number])
        prefix = modality_prefix.get(mod, mod.capitalize())
        X.columns = [f"{prefix}_{c}" for c in X.columns]
        d["X_aligned"] = X
        d["y_aligned"] = df_mod["label"].to_numpy(dtype=int)
        d["groups_aligned"] = df_mod["ID"].astype(str).to_numpy()

    # Use first modality as reference for y and groups
    y = modality_data[effective_modalities[0]]["y_aligned"]
    groups = modality_data[effective_modalities[0]]["groups_aligned"]
    if len(np.unique(groups)) < outer_folds:
        raise ValueError(f"Not enough groups for {outer_folds}-fold GroupKFold")

    X_by_modality = {mod: modality_data[mod]["X_aligned"] for mod in effective_modalities}

    # Prepare classifiers to run
    models = get_classifier_models()
    if cfg.classifiers:
        # cfg.classifiers may be a single string or list; normalize to list
        if isinstance(cfg.classifiers, str):
            requested = [cfg.classifiers]
        else:
            requested = cfg.classifiers
        models = {k: v for k, v in models.items() if k in requested}
    if not models:
        raise ValueError("No valid classifiers selected")

    # Save run configuration once at the top-level directory
    cfg_dict = {k: getattr(cfg, k) for k in ["fusion_modalities", "condition", "phases", "minutes", "outer_folds", "inner_folds", "nan_threshold", "mannwhitney_k", "rfe_n", "aggregation_method", "include_masseter"] if hasattr(cfg, k)}
    for k, v in list(cfg_dict.items()):
        if isinstance(v, (Path, set)):
            cfg_dict[k] = str(v)
        elif isinstance(v, np.ndarray):
            cfg_dict[k] = v.tolist()
    with open(out_dir / f"{base_name}_args.json", "w") as fp:
        json.dump(cfg_dict, fp, indent=2)

    # Run late fusion separately for each classifier, storing results in subfolders
    for model_name, clf_base in models.items():
        clf = clf_base

        def build_estimator_fn(modality_name: str) -> Pipeline:
            # Special case: Text modality – no ConstantFilter/NaNFilter/MW, only RFE over 14 features with step=4
            if modality_name == "text":
                rfe_estimator = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
                text_pipe = Pipeline([
                    ("rfe_selector", RFE(estimator=rfe_estimator, step=4)),
                    ("clf", clone(clf)),
                ])
                param_grid_text = {
                    "rfe_selector__n_features_to_select": [4, 8, 12, 14],
                }
                return GridSearchCV(
                    text_pipe,
                    param_grid_text,
                    cv=inner_folds,
                    scoring="f1_weighted",
                    refit=True,
                    n_jobs=1,
                    error_score="raise",
                )

            # Default: full late-fusion pipeline with preprocessing + MW + RFE
            return build_full_late_fusion_pipeline(
                nan_threshold=nan_threshold,
                mw_k=mw_k,
                rfe_n=rfe_n,
                clf=clone(clf),
            )

        clf_dir = out_dir / model_name
        clf_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] === Classifier: {model_name} ===")
        print(f"[INFO] Late fusion modalities: {effective_modalities} (feature names prefixed: Video_, Audio_, ECG_, EMG_, RSP_, Text_)")
        print(f"[INFO] Pipeline (non-text): preprocess + Mann-Whitney({mw_k}) + RFE({rfe_n}); classifier: {model_name}")
        if has_text:
            print(f"[INFO] Text pipeline: RFE(step=4) with GridSearch over n_features_to_select=[4, 8, 12, 14]; classifier: {model_name}")

        result = run_late_fusion_cv(
            X_by_modality=X_by_modality,
            y=y,
            groups=groups,
            build_estimator_fn=build_estimator_fn,
            outer_splits=outer_folds,
            inner_splits=inner_folds,
        )

        agg = result["aggregate"]

        # ------------------------------------------------------------------
        # Per-fold results (results.csv)
        # ------------------------------------------------------------------
        per_fold_rows = []
        for f in result["per_fold"]:
            # Confusion matrix per fold (tn, fp, fn, tp)
            cm_f = np.asarray(f["confusion_matrix"])
            if cm_f.shape == (2, 2):
                tn_f, fp_f, fn_f, tp_f = int(cm_f[0, 0]), int(cm_f[0, 1]), int(cm_f[1, 0]), int(cm_f[1, 1])
            else:
                tn_f = fp_f = fn_f = tp_f = 0

            # Selected feature count (total across modalities) if available
            sel_count_total = None
            if "selected_features" in f and isinstance(f["selected_features"], dict):
                sel_count_total = sum(len(v) for v in f["selected_features"].values())

            per_fold_rows.append(
                {
                    "fold": f["fold"],
                    "model": model_name,
                    "f1_weighted": f["fused_test_f1_weighted"],
                    "accuracy": f["fused_test_accuracy"],
                    "sensitivity": f.get("fused_test_sensitivity"),
                    "specificity": f.get("fused_test_specificity"),
                    "precision": f.get("fused_test_precision"),
                    "recall": f.get("fused_test_recall"),
                    "roc_auc": f.get("fused_test_roc_auc"),
                    "tn": tn_f,
                    "fp": fp_f,
                    "fn": fn_f,
                    "tp": tp_f,
                    "selected_features_total": sel_count_total,
                }
            )

        pd.DataFrame(per_fold_rows).to_csv(clf_dir / "results.csv", index=False)

        # ------------------------------------------------------------------
        # Final summary (final_results.csv)
        # ------------------------------------------------------------------
        # Aggregate confusion matrix across folds
        tn_sum = fp_sum = fn_sum = tp_sum = 0
        for f in result["per_fold"]:
            cm_f = np.asarray(f["confusion_matrix"])
            if cm_f.shape == (2, 2):
                tn_sum += int(cm_f[0, 0])
                fp_sum += int(cm_f[0, 1])
                fn_sum += int(cm_f[1, 0])
                tp_sum += int(cm_f[1, 1])

        total = tn_sum + fp_sum + fn_sum + tp_sum
        if total > 0:
            acc_agg = (tp_sum + tn_sum) / total
            sens_agg = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else np.nan
            spec_agg = tn_sum / (tn_sum + fp_sum) if (tn_sum + fp_sum) > 0 else np.nan
        else:
            acc_agg = sens_agg = spec_agg = np.nan

        # Class balance (global, based on all y)
        unique_labels, label_counts = np.unique(y, return_counts=True)
        class_counts = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, label_counts)}
        n_samples = int(label_counts.sum()) if label_counts.size > 0 else 0
        n_neg = class_counts.get(NEG_LABEL, 0)
        n_pos = class_counts.get(POS_LABEL, 0)
        pct_neg = n_neg / n_samples if n_samples > 0 else np.nan
        pct_pos = n_pos / n_samples if n_samples > 0 else np.nan

        final_row = {
            "model": model_name,
            # Aggregated means/stds from run_late_fusion_cv
            "f1_weighted_mean": agg["f1_weighted_mean"],
            "f1_weighted_std": agg["f1_weighted_std"],
            "accuracy_mean": agg["accuracy_mean"],
            "accuracy_std": agg["accuracy_std"],
            "sensitivity_mean": agg["sensitivity_mean"],
            "sensitivity_std": agg["sensitivity_std"],
            "specificity_mean": agg["specificity_mean"],
            "specificity_std": agg["specificity_std"],
            "precision_mean": agg["precision_mean"],
            "precision_std": agg["precision_std"],
            "recall_mean": agg["recall_mean"],
            "recall_std": agg["recall_std"],
            "roc_auc_mean": agg["roc_auc_mean"],
            "roc_auc_std": agg["roc_auc_std"],
            # Aggregated confusion matrix across folds
            "tn_sum": tn_sum,
            "fp_sum": fp_sum,
            "fn_sum": fn_sum,
            "tp_sum": tp_sum,
            "accuracy_agg_from_cm": acc_agg,
            "sensitivity_agg_from_cm": sens_agg,
            "specificity_agg_from_cm": spec_agg,
            # Class balance information
            "n_samples": n_samples,
            "n_neg": n_neg,
            "n_pos": n_pos,
            "pct_neg": pct_neg,
            "pct_pos": pct_pos,
        }

        final_df = pd.DataFrame([final_row])
        final_df.to_csv(clf_dir / "final_results.csv", index=False)

        # ------------------------------------------------------------------
        # Evaluate, plot confusion matrix, and plot ROC curves (aggregated over folds)
        # ------------------------------------------------------------------
        y_true_all: List[float] = []
        y_pred_all: List[float] = []
        y_proba_all: List[float] = []
        for f in result["per_fold"]:
            y_true_all.extend(f["y_true"])
            y_pred_all.extend(f["y_pred"])
            y_proba_all.extend(f["y_proba"])
        y_true_arr = np.array(y_true_all)
        y_pred_arr = np.array(y_pred_all)
        y_proba_arr = np.array(y_proba_all) if y_proba_all else None
        normalization = getattr(cfg, "aggregation_method", "late_fusion")
        fusion_label = f"Late_Fusion_{model_name}"
        test_metrics = evaluate_predictions(y_true_arr, y_pred_arr, y_proba_arr)
        print(f"[INFO] Evaluated {fusion_label} (aggregated over folds): F1={test_metrics['f1_weighted']:.3f}, Acc={test_metrics['accuracy']:.3f}, AUC={test_metrics['roc_auc']:.3f}")
        plot_confusion_matrix(
            y_true_arr, y_pred_arr, fusion_label,
            normalization=normalization, save_path=clf_dir / "confusion_matrix.png",
        )
        if y_proba_arr is not None and len(y_proba_arr) > 0:
            plot_roc_curves(
                {fusion_label: (y_true_all, y_proba_all)},
                normalization=normalization, save_path=clf_dir / "roc_curve.png",
            )
        else:
            plt.figure(figsize=(6, 5))
            plt.text(0.5, 0.5, "ROC curve not available\n(no probability estimates)",
                     ha="center", va="center", fontsize=12, wrap=True)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis("off")
            plt.title(f"ROC — {fusion_label}")
            plt.tight_layout()
            plt.savefig(clf_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[INFO] Saved ROC placeholder to {clf_dir / 'roc_curve.png'}")

        for f in result["per_fold"]:
            fold_idx = f["fold"]
            to_save = {
                "unimodal_oof_f1": f["unimodal_oof_f1"],
                "unimodal_oof_metrics": f["unimodal_oof_metrics"],
                "weights_final": f["weights_final"],
            }
            if "selected_features" in f:
                to_save["selected_features"] = f["selected_features"]
            feat_path = clf_dir / f"{base_name}_selected_features_fold{fold_idx}.json"
            with open(feat_path, "w") as fp:
                json.dump(to_save, fp, indent=2)
            print(f"[INFO] Saved selected features (RFE per modality) for {model_name} to {feat_path}")

        print(f"[INFO] Late fusion done for classifier {model_name}. Summary:\n{final_df}")
        print(f"[INFO] Results and selected features saved under {clf_dir}")


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)