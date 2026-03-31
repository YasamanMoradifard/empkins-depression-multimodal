#!/usr/bin/env python3
"""
Unified MultiModal ML Pipeline for Audio, Video, and BioSig (ECG/EMG/RSP) data.

This script provides a unified interface to run ML regression on different
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
- Feature selection: Correlation + RFE
- Pipeline includes ConstantFilter for all modalities
- Models: Linear Regression, Random Forest, AdaBoost, Decision Tree, KNN, XGBoost

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
import re
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

from scipy.optimize import minimize
from scipy.stats import skew, kurtosis
from numpy.fft import rfft, rfftfreq
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

RANDOM_STATE = 42
PHQ9_SCORE_DATA_PATH = Path("/home/vault/empkins/tpD/D02/Students/Yasaman/0_RCT_Data_Info/result/Data_D02_info.csv")

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


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """Select top N features by absolute correlation with continuous target (regression)."""
    def __init__(self, n_features_to_select=25):
        self.n_features_to_select = n_features_to_select
        self.selected_features_ = None  # indices

    def fit(self, X, y):
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = np.asarray(y).ravel()
        valid = ~(np.isnan(y_arr) | np.isnan(X_arr).any(axis=1))
        X_clean = X_arr[valid]
        y_clean = y_arr[valid]
        if X_clean.size == 0 or y_clean.size == 0:
            self.selected_features_ = np.arange(min(self.n_features_to_select, X_arr.shape[1]))
            return self
        corrs = []
        for i in range(X_clean.shape[1]):
            col = X_clean[:, i]
            valid_col = ~np.isnan(col)
            if valid_col.sum() < 2:
                corrs.append((i, 0.0))
            else:
                c = np.corrcoef(col[valid_col], y_clean[valid_col])[0, 1]
                corrs.append((i, abs(c) if not np.isnan(c) else 0.0))
        corrs.sort(key=lambda x: x[1], reverse=True)
        k = min(self.n_features_to_select, len(corrs))
        self.selected_features_ = np.array([idx for idx, _ in corrs[:k]])
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
    """Convert participant IDs to a standardized string format (zero-padded to 3 digits)."""
    if pd.isna(pid) or (isinstance(pid, float) and math.isnan(pid)):
        return np.nan
    try:
        pid_int = int(float(pid))
        return str(pid_int).zfill(3)
    except ValueError:
        return np.nan


def load_phq9_scores(phq9_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load PHQ9-Score from Data_D02_info.csv with zero-padded ID for merging.
    Returns DataFrame with columns: ID (str, zero-padded), label (PHQ9-Score).
    """
    path = phq9_path or PHQ9_SCORE_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"PHQ9 data not found: {path}")
    df = pd.read_csv(path)
    if "ID" not in df.columns or "PHQ9-Score" not in df.columns:
        raise ValueError(
            f"PHQ9 CSV must contain 'ID' and 'PHQ9-Score'. Found: {list(df.columns)[:15]}"
        )
    df = df[["ID", "PHQ9-Score"]].copy()
    df["ID"] = df["ID"].apply(normalize_id)
    df = df.rename(columns={"PHQ9-Score": "label"})
    df = df.dropna(subset=["ID", "label"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    return df


def merge_phq9_scores(df: pd.DataFrame, phq9_scores: pd.DataFrame) -> pd.DataFrame:
    """Merge PHQ9 scores into dataframe by ID. ID must be zero-padded in both."""
    df = df.copy()
    df["ID"] = df["ID"].astype(str)
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    phq9_scores = phq9_scores.copy()
    phq9_scores["ID"] = phq9_scores["ID"].astype(str)
    merged = df.merge(phq9_scores[["ID", "label"]], on="ID", how="inner")
    merged = merged.dropna(subset=["label"])
    return merged


def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize column names for XGBoost compatibility.
    Handles MultiIndex columns (e.g. from groupby.agg) by flattening to strings.
    XGBoost requires valid Python identifiers: must start with letter/underscore, 
    contain only letters, digits, underscores, and be non-empty.
    """
    def sanitize_name(name: str) -> str:
        name = str(name)
        # Replace problematic characters
        name = name.replace('[', '_').replace(']', '_').replace('<', '_')
        name = name.replace('>', '_').replace('|', '_').replace('&', '_')
        name = name.replace(' ', '_').replace('-', '_').replace('.', '_')
        name = name.replace('(', '_').replace(')', '_').replace(',', '_')
        name = name.replace(':', '_').replace(';', '_').replace('=', '_')
        name = name.replace('+', '_').replace('*', '_').replace('/', '_')
        name = name.replace('\\', '_').replace('"', '_').replace("'", '_')
        # Remove consecutive underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        # Ensure it starts with a letter or underscore (XGBoost requirement)
        if name and not (name[0].isalpha() or name[0] == '_'):
            name = 'f_' + name
        # If empty after sanitization, use a default name
        if not name or name == '':
            name = 'feature'
        # Ensure it's a valid Python identifier
        if not name.isidentifier():
            # Replace invalid characters and ensure it starts correctly
            name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
            if name and not (name[0].isalpha() or name[0] == '_'):
                name = 'f_' + name
            if not name:
                name = 'feature'
        return name

    def flatten_col(col) -> str:
        """Convert column (possibly tuple from MultiIndex) to a flat string."""
        if isinstance(col, tuple):
            parts = [str(p) for p in col if p]
            return '_'.join(parts) if parts else sanitize_name(str(col))
        return str(col)

    df = df.copy()
    sanitized_cols = []
    seen_names = {}
    for col in df.columns:
        base_name = sanitize_name(flatten_col(col))
        # Handle duplicate names
        if base_name in seen_names:
            seen_names[base_name] += 1
            base_name = f"{base_name}_{seen_names[base_name]}"
        else:
            seen_names[base_name] = 0
        sanitized_cols.append(base_name)
    df.columns = sanitized_cols
    return df


def load_audio_data(
    condition: str,
    phases: List[str],
    opensmile_data_dir: Path,
    aggregation_method: str = "by_phase",
    phq9_path: Optional[Path] = None,
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

    # Merge PHQ9-Score as label (regression target)
    phq9_scores = load_phq9_scores(phq9_path)
    aggregated_df = merge_phq9_scores(aggregated_df, phq9_scores)

    print(f"[INFO] Audio data: {aggregated_df.shape[0]} rows, {aggregated_df.shape[1]} columns")
    print(f"[INFO] Unique participants: {aggregated_df['ID'].nunique()}")
    print("-----------------------------------------\n")

    return aggregated_df


def load_video_data(
    condition: str,
    phases: List[str],
    opendbm_data_dir: Path,
    aggregation_method: str = "by_phase",
    phq9_path: Optional[Path] = None,
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
    
    # Merge PHQ9-Score as label (regression target) before aggregation
    phq9_scores = load_phq9_scores(phq9_path)
    combined_df = merge_phq9_scores(combined_df, phq9_scores)
    
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

    # Merge PHQ9-Score as label (regression target)
    phq9_scores = load_phq9_scores(PHQ9_SCORE_DATA_PATH)
    aggregated_df = merge_phq9_scores(aggregated_df, phq9_scores)

    # PHQ9 already merged before aggregation; no need to merge again
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
    phq9_path: Optional[Path] = None,
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

    # Merge PHQ9-Score as label (regression target) before aggregation
    phq9_scores = load_phq9_scores(phq9_path)
    df = merge_phq9_scores(df, phq9_scores)

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
    
    aggregated_df = sanitize_feature_names(aggregated_df)

    # Add modality prefix to feature columns (label already from PHQ9 merge before aggregation)
    biosig_meta = ["ID", "label", "phase", "condition", "Aufgabe"]
    feature_cols = [c for c in aggregated_df.columns if c not in biosig_meta]
    rename_map = {c: f"{dtype}_{c}" for c in feature_cols}
    aggregated_df = aggregated_df.rename(columns=rename_map)

    print(f"[INFO] BioSig data: {aggregated_df.shape[0]} rows, {aggregated_df.shape[1]} columns")
    print(f"[INFO] Unique participants: {aggregated_df['ID'].nunique()}")
    print("-----------------------------------------\n")
    
    return aggregated_df


def load_text_data(text_csv_path: Path, condition: str = "all", phq9_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load Text modality features from a pre-computed CSV.
    
    Expected format:
        - Column 'patient_id' (participant ID, not zero-padded)
        - Column 'condition' (treatment condition: CR, ADK, CRADK, SHAM, ...)
        - Numeric feature columns
    
    Processing:
        - Zero-pad 'patient_id' to 3 digits and rename to 'ID'
        - Optionally filter rows by 'condition' (if condition != 'all')
        - Merge PHQ9-Score as label (regression target)
    """
    if not text_csv_path.exists():
        raise FileNotFoundError(f"Text data CSV not found: {text_csv_path}")
    
    df = pd.read_csv(text_csv_path)
    # Basic column checks
    if "patient_id" not in df.columns:
        raise ValueError(f"Text CSV must contain 'patient_id'. Found: {list(df.columns)[:10]}")
 
    # Zero-pad patient_id and rename to ID
    df = df.copy()
    df["ID"] = df["patient_id"].apply(normalize_id).astype(str)
    df = df.drop(columns=["patient_id"], errors="ignore")
    
    # Normalize / filter by condition
    if "condition" in df.columns:
        df["condition"] = df["condition"].astype(str).str.strip()
        if condition.lower() != "all":
            df = df[df["condition"] == condition].copy()
            if df.empty:
                raise ValueError(f"No text rows found for condition '{condition}' in {text_csv_path}")
    
    # Treat all remaining non-ID / non-condition columns as numeric features
    meta_cols = ["ID", "condition", "Diagnose"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df = df.drop(columns=[c for c in ["Diagnose"] if c in df.columns], errors="ignore")

    # Merge PHQ9-Score as label (regression target)
    phq9_scores = load_phq9_scores(phq9_path)
    df = merge_phq9_scores(df, phq9_scores)

    # Add modality prefix to feature columns
    feature_cols = [c for c in df.columns if c not in ["ID", "label", "condition"]]
    rename_map = {c: f"Text_{c}" for c in feature_cols}
    df = df.rename(columns=rename_map)

    print(f"[INFO] Text data: {df.shape[0]} rows, {len(feature_cols)} feature columns")
    print(f"[INFO] Unique participants (text): {df['ID'].nunique()}")
    if condition.lower() != "all":
        print(f"[INFO] Condition filter applied to text modality: {condition}")
    print("-----------------------------------------\n")
    return df


# =============================================================================
# Regression Models
# =============================================================================

def get_regression_models() -> Dict[str, object]:
    """Return regression models for PHQ9-Score prediction."""
    models: Dict[str, object] = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=2,
            n_jobs=2,
            random_state=RANDOM_STATE,
        ),
        "AdaBoost": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=1),
            n_estimators=100,
            learning_rate=0.5,
            random_state=RANDOM_STATE,
        ),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=4,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsRegressor(
            n_neighbors=7,
            weights="distance",
        ),
    }
    
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            tree_method='hist',  # Memory-efficient method, faster than 'exact'
            n_jobs=1,  # Avoid oversubscription in nested CV
            verbosity=0,  # Suppress XGBoost output to reduce memory/log overhead
            enable_categorical=False,  # Explicitly disable categorical features
            max_bin=256,  # Reduce memory usage
        )
    
    return models


def get_param_grids_reg_only() -> Dict[str, Dict]:
    """Regression param grids (no RFE n). RFE n is chosen by a for loop. Use reg__ prefix for pipeline."""
    grids = {
        "LinearRegression": {},
        "RandomForest": {"reg__n_estimators": [100, 200], "reg__max_depth": [3, 5, None]},
        "AdaBoost": {"reg__n_estimators": [50, 100, 200], "reg__learning_rate": [0.1, 0.5, 1.0]},
        "DecisionTree": {"reg__max_depth": [3, 5, None], "reg__min_samples_leaf": [1, 2, 4]},
        "KNN": {"reg__n_neighbors": [3, 5, 7, 9]},
    }
    if HAS_XGB:
        grids["XGBoost"] = {
            "reg__n_estimators": [100, 200],
            "reg__max_depth": [3, 4, 5],
            "reg__learning_rate": [0.01, 0.05, 0.1],
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


def build_inner_pipeline_late_fusion_regression(reg, corr_k: int = 25, rfe_n: int = 10) -> Pipeline:
    """Inner CV: CorrelationSelector top corr_k -> RFE select rfe_n -> regressor.
    RFE uses DecisionTreeRegressor for consistent, fast feature selection across all regression models."""
    rfe_estimator = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)
    return Pipeline([
        ("corr_selector", CorrelationSelector(n_features_to_select=corr_k)),
        ("rfe_selector", RFE(estimator=rfe_estimator, n_features_to_select=rfe_n, step=1)),
        ("reg", reg),
    ])


def build_full_late_fusion_pipeline(
    nan_threshold: float,
    corr_k: int,
    rfe_n: int,
    reg: object,
) -> Pipeline:
    """Full late fusion pipeline: preprocess (ConstantFilter, NaN, impute, scale) + CorrelationSelector(corr_k) + RFE(rfe_n) + regressor. For use with run_late_fusion_regression_cv."""
    preprocess = build_preprocess_pipeline(nan_threshold=nan_threshold)
    inner = build_inner_pipeline_late_fusion_regression(clone(reg), corr_k=corr_k, rfe_n=rfe_n)
    steps = list(preprocess.named_steps.items()) + list(inner.named_steps.items())
    return Pipeline(steps=steps)


def extract_selected_features_late_fusion(pipe: Pipeline, feature_names: List[str]) -> List[str]:
    """Extract final selected feature names after CorrelationSelector + RFE from a fitted inner pipeline."""
    names = list(feature_names)
    if "corr_selector" in pipe.named_steps:
        corr = pipe.named_steps["corr_selector"]
        if hasattr(corr, "selected_features_") and corr.selected_features_ is not None:
            names = [names[i] for i in corr.selected_features_ if i < len(names)]
    if "rfe_selector" in pipe.named_steps:
        rfe = pipe.named_steps["rfe_selector"]
        if hasattr(rfe, "support_"):
            names = [f for f, s in zip(names, rfe.support_) if s]
    return names


# =============================================================================
# Late fusion CV (standalone API): weights, OOF, main CV for REGRESSION
# =============================================================================


def _get_train_subset(
    X: Union[pd.DataFrame, np.ndarray],
    indices: np.ndarray,
) -> Union[pd.DataFrame, np.ndarray]:
    """Return X subset by indices (DataFrame or array)."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[indices]
    return X[indices]


def compute_fusion_weights_regression(
    rmse_by_modality: Dict[str, float],
    weight_method: str = "optimize",
    preds_by_modality: Optional[Dict[str, np.ndarray]] = None,
    y_val: Optional[np.ndarray] = None,
    epsilon: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute fusion weights for regression based on modality performance.

    Method A (inverse_rmse): weight = (1/RMSE) / sum(1/RMSE). Better modalities (lower RMSE) get higher weights.
    Method B (optimize): Use scipy.optimize.minimize to find optimal weights that minimize MSE of fused predictions
    on validation set. Constraints: weights sum to 1, each weight in [0, 1].

    Args:
        rmse_by_modality: Dict mapping modality name -> RMSE on validation set.
        weight_method: 'inverse_rmse' or 'optimize'.
        preds_by_modality: For 'optimize' method: dict of modality -> validation predictions (n_samples,).
        y_val: For 'optimize' method: true validation targets.
        epsilon: Small value to avoid division by zero in inverse_rmse.

    Returns:
        Dict of modality -> weight (sum to 1).
    """
    modality_names = list(rmse_by_modality.keys())
    if not modality_names:
        return {}

    if weight_method == "inverse_rmse":
        raw_weights = []
        for mod in modality_names:
            rmse = max(epsilon, rmse_by_modality[mod])
            raw_weights.append(1.0 / rmse)
        raw_sum = sum(raw_weights)
        if raw_sum <= 0:
            uniform = 1.0 / len(modality_names)
            return {mod: uniform for mod in modality_names}
        return {mod: raw_weights[i] / raw_sum for i, mod in enumerate(modality_names)}

    if weight_method == "optimize" and preds_by_modality is not None and y_val is not None:
        preds_arr = np.column_stack([preds_by_modality[mod] for mod in modality_names])
        n_mod = len(modality_names)
        y_val = np.asarray(y_val).ravel()

        def mse_loss(w: np.ndarray) -> float:
            fused = preds_arr @ w
            return float(np.mean((y_val - fused) ** 2))

        # Constraints: sum(w)=1, 0 <= w_i <= 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * n_mod
        x0 = np.ones(n_mod) / n_mod
        res = minimize(mse_loss, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = res.x
        weights = np.clip(weights, 0, 1)
        weights = weights / weights.sum()
        return {mod: float(weights[i]) for i, mod in enumerate(modality_names)}

    # Fallback to inverse_rmse
    raw_weights = []
    for mod in modality_names:
        rmse = max(epsilon, rmse_by_modality[mod])
        raw_weights.append(1.0 / rmse)
    raw_sum = sum(raw_weights)
    if raw_sum <= 0:
        uniform = 1.0 / len(modality_names)
        return {mod: uniform for mod in modality_names}
    return {mod: raw_weights[i] / raw_sum for i, mod in enumerate(modality_names)}


class WeightedLateFusion:
    """
    Weighted late fusion for multimodal regression.

    Trains separate models per modality, then combines predictions using learned weights.
    Weights can be computed via inverse RMSE or optimization.

    Attributes:
        models_: Dict mapping modality -> fitted estimator.
        weights_: Dict mapping modality -> fusion weight.
        modality_names_: List of modality names.
    """

    def __init__(
        self,
        models_by_modality: Dict[str, object],
        weight_method: str = "optimize",
    ):
        """
        Args:
            models_by_modality: Dict of modality name -> unfitted sklearn-like regressor.
            weight_method: 'inverse_rmse' or 'optimize'.
        """
        self.models_by_modality = models_by_modality
        self.weight_method = weight_method
        self.models_: Dict[str, object] = {}
        self.weights_: Dict[str, float] = {}
        self.modality_names_ = list(models_by_modality.keys())

    def fit(
        self,
        X_by_modality: Dict[str, Union[pd.DataFrame, np.ndarray]],
        y_train: np.ndarray,
        X_val_by_modality: Optional[Dict[str, Union[pd.DataFrame, np.ndarray]]] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "WeightedLateFusion":
        """
        Train individual models and compute fusion weights.

        If X_val_by_modality and y_val are provided, weights are computed on validation set.
        Otherwise, weights default to inverse_rmse using training-set predictions (less ideal).
        """
        n = len(y_train)
        for mod in self.modality_names_:
            X = X_by_modality[mod]
            if isinstance(X, pd.DataFrame):
                X = X.values if hasattr(X, "values") else np.asarray(X)
            if len(X) != n:
                raise ValueError(f"Modality {mod}: expected {n} samples, got {len(X)}")
            model = clone(self.models_by_modality[mod])
            model.fit(X, y_train)
            self.models_[mod] = model

        # Get predictions for weight computation
        preds_train = {}
        for mod in self.modality_names_:
            X = X_by_modality[mod]
            preds_train[mod] = np.asarray(self.models_[mod].predict(X)).ravel()

        rmse_train = {}
        for mod in self.modality_names_:
            rmse_train[mod] = float(np.sqrt(np.mean((y_train - preds_train[mod]) ** 2)))

        if X_val_by_modality is not None and y_val is not None:
            preds_val = {}
            for mod in self.modality_names_:
                Xv = X_val_by_modality[mod]
                preds_val[mod] = np.asarray(self.models_[mod].predict(Xv)).ravel()
            self.weights_ = compute_fusion_weights_regression(
                rmse_train,
                weight_method=self.weight_method,
                preds_by_modality=preds_val,
                y_val=y_val,
            )
        else:
            self.weights_ = compute_fusion_weights_regression(
                rmse_train, weight_method=self.weight_method
            )
        return self

    def predict(
        self,
        X_by_modality: Dict[str, Union[pd.DataFrame, np.ndarray]],
    ) -> np.ndarray:
        """Make fused predictions on new data."""
        preds = []
        for mod in self.modality_names_:
            X = X_by_modality[mod]
            p = np.asarray(self.models_[mod].predict(X)).ravel()
            preds.append(self.weights_[mod] * p)
        return np.sum(preds, axis=0)

    def get_weights(self) -> Dict[str, float]:
        """Return the learned weights for each modality."""
        return dict(self.weights_)


def _ensure_xgboost_compatible(X, y=None):
    """Ensure data is compatible with XGBoost: convert to numpy array, handle NaNs, ensure float32/float64."""
    if isinstance(X, pd.DataFrame):
        # Ensure feature names are sanitized (should already be done, but double-check)
        X = X.copy()
        # Convert to numpy array
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
    
    # Ensure float dtype for XGBoost compatibility
    if X_arr.dtype not in [np.float32, np.float64]:
        X_arr = X_arr.astype(np.float64)
    
    # Check for infinite values
    if np.any(np.isinf(X_arr)):
        print("[WARNING] Found infinite values in X, replacing with NaN")
        X_arr = np.where(np.isinf(X_arr), np.nan, X_arr)
    
    if y is not None:
        y_arr = np.asarray(y)
        if y_arr.dtype not in [np.float32, np.float64]:
            y_arr = y_arr.astype(np.float64)
        if np.any(np.isinf(y_arr)):
            print("[WARNING] Found infinite values in y, replacing with NaN")
            y_arr = np.where(np.isinf(y_arr), np.nan, y_arr)
        return X_arr, y_arr
    
    return X_arr


def get_oof_predictions(
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    estimator: object,
    inner_splits: int,
) -> np.ndarray:
    """
    Run GroupKFold(inner_splits) on train_indices only; for each inner fold fit on
    inner-train, predict on inner-val, fill OOF array. Returns array of shape (len(train_indices),).
    """
    num_train = len(train_indices)
    oof_pred = np.full(num_train, np.nan, dtype=np.float64)

    X_train = _get_train_subset(X, train_indices)
    y_train = y[train_indices]
    groups_train = groups[train_indices]

    inner_cv = GroupKFold(n_splits=inner_splits)
    splits = list(inner_cv.split(X_train, y_train, groups_train))

    for inner_train_idx, inner_val_idx in splits:
        X_inner_train = _get_train_subset(X_train, inner_train_idx)
        y_inner_train = y_train[inner_train_idx]
        X_inner_val = _get_train_subset(X_train, inner_val_idx)

        cloned = clone(estimator)
        
        # Ensure XGBoost compatibility if using XGBoost
        is_xgb = HAS_XGB and (hasattr(cloned, 'named_steps') and 
                              'reg' in cloned.named_steps and 
                              'XGB' in str(type(cloned.named_steps['reg'])).upper())
        if not is_xgb:
            # Check if it's a GridSearchCV wrapper
            if hasattr(cloned, 'estimator'):
                est = cloned.estimator
                if hasattr(est, 'named_steps') and 'reg' in est.named_steps:
                    reg = est.named_steps['reg']
                    is_xgb = HAS_XGB and 'XGB' in str(type(reg)).upper()
        
        try:
            if is_xgb:
                # Ensure data compatibility for XGBoost
                X_inner_train, y_inner_train = _ensure_xgboost_compatible(X_inner_train, y_inner_train)
                X_inner_val = _ensure_xgboost_compatible(X_inner_val)
            
            cloned.fit(X_inner_train, y_inner_train)
            pred_val = np.asarray(cloned.predict(X_inner_val)).ravel()

            for local_idx, global_idx in enumerate(inner_val_idx):
                oof_pred[global_idx] = pred_val[local_idx]
        except Exception as e:
            print(f"[ERROR] Failed to fit/predict in OOF loop: {e}")
            print(f"[ERROR] XGBoost compatibility check: {is_xgb}")
            print(f"[ERROR] X shape: {X_inner_train.shape if hasattr(X_inner_train, 'shape') else 'unknown'}")
            print(f"[ERROR] y shape: {y_inner_train.shape if hasattr(y_inner_train, 'shape') else 'unknown'}")
            # Continue with NaN predictions for this fold
            continue

    return oof_pred


def run_late_fusion_regression_cv(
    X_by_modality: Dict[str, Union[pd.DataFrame, np.ndarray]],
    y: np.ndarray,
    groups: np.ndarray,
    build_estimator_fn: Callable[[str], object],
    outer_splits: int = 5,
    inner_splits: int = 3,
    weight_method: str = "optimize",
) -> Dict:
    """
    Nested CV for weighted late fusion regression. Outer GroupKFold for performance estimation,
    inner loop for OOF predictions per modality. Fusion weights computed from OOF RMSE
    (inverse_rmse or optimize). build_estimator_fn(modality_name) returns unfitted sklearn-like
    regressor with fit and predict.
    """
    modality_names = list(X_by_modality.keys())
    num_samples = len(y)
    y = np.asarray(y).ravel().astype(np.float64)
    print(f"[Late fusion regression] samples={num_samples}, modalities={modality_names}, weight_method={weight_method}")

    outer_cv = GroupKFold(n_splits=outer_splits)
    per_fold: List[Dict] = []
    all_rmse: List[float] = []
    all_mae: List[float] = []
    all_r2: List[float] = []
    all_mape: List[float] = []

    for fold_index, (outer_train_idx, outer_test_idx) in enumerate(
        outer_cv.split(np.zeros(num_samples), y, groups=groups)
    ):
        print(f"=== Outer Fold {fold_index + 1}/{outer_splits} ===")
        num_outer_train = len(outer_train_idx)
        num_outer_test = len(outer_test_idx)
        y_outer_train = y[outer_train_idx]
        y_outer_test = y[outer_test_idx]
        print(f"  Train samples: {num_outer_train}, test samples: {num_outer_test}")

        # === Inner loop: OOF predictions per modality ===
        oof_preds_by_modality: Dict[str, np.ndarray] = {}
        unimodal_oof_rmse: Dict[str, float] = {}
        unimodal_oof_metrics: Dict[str, Dict] = {}
        for mod in modality_names:
            X_mod = X_by_modality[mod]
            estimator = build_estimator_fn(mod)
            oof_pred = get_oof_predictions(
                X_mod, y, groups, outer_train_idx, estimator, inner_splits
            )
            valid = ~np.isnan(oof_pred)
            if valid.sum() == 0:
                rmse_oof = float("inf")
                mae_oof = float("inf")
                r2_oof = float("-inf")
            else:
                rmse_oof = float(np.sqrt(np.mean((y_outer_train[valid] - oof_pred[valid]) ** 2)))
                mae_oof = float(mean_absolute_error(y_outer_train[valid], oof_pred[valid]))
                r2_oof = float(r2_score(y_outer_train[valid], oof_pred[valid]))
            unimodal_oof_rmse[mod] = rmse_oof
            unimodal_oof_metrics[mod] = {"rmse": rmse_oof, "mae": mae_oof, "r2": r2_oof}
            oof_preds_by_modality[mod] = oof_pred
            print(f"  {mod}: OOF RMSE={rmse_oof:.4f} MAE={mae_oof:.4f} R²={r2_oof:.4f}")

        # === Compute fusion weights (inverse_rmse from OOF, or optimize on OOF) ===
        preds_for_weights = {mod: oof_preds_by_modality[mod] for mod in modality_names}
        weights_final = compute_fusion_weights_regression(
            unimodal_oof_rmse,
            weight_method=weight_method,
            preds_by_modality=preds_for_weights,
            y_val=y_outer_train,
        )
        print("  fusion weights (rounded):", {k: round(v, 3) for k, v in weights_final.items()})

        # === Evaluate on outer test ===
        test_preds_by_modality: Dict[str, np.ndarray] = {}
        selected_features_this_fold: Dict[str, List[str]] = {}
        fitted_pipelines_this_fold: Dict[str, object] = {}
        for mod in modality_names:
            X_mod = X_by_modality[mod]
            X_outer_train = _get_train_subset(X_mod, outer_train_idx)
            X_outer_test = _get_train_subset(X_mod, outer_test_idx)
            estimator = build_estimator_fn(mod)
            fitted = clone(estimator)
            
            # Check if using XGBoost
            is_xgb = HAS_XGB and (hasattr(fitted, 'named_steps') and 
                                  'reg' in fitted.named_steps and 
                                  'XGB' in str(type(fitted.named_steps['reg'])).upper())
            if not is_xgb and hasattr(fitted, 'estimator'):
                est = fitted.estimator
                if hasattr(est, 'named_steps') and 'reg' in est.named_steps:
                    reg = est.named_steps['reg']
                    is_xgb = HAS_XGB and 'XGB' in str(type(reg)).upper()
            
            try:
                if is_xgb:
                    # Ensure data compatibility for XGBoost
                    X_outer_train, y_outer_train_clean = _ensure_xgboost_compatible(X_outer_train, y_outer_train)
                    X_outer_test = _ensure_xgboost_compatible(X_outer_test)
                else:
                    y_outer_train_clean = y_outer_train
                
                fitted.fit(X_outer_train, y_outer_train_clean)
                pred_test = np.asarray(fitted.predict(X_outer_test)).ravel()
                test_preds_by_modality[mod] = pred_test
                fitted_pipelines_this_fold[mod] = fitted
            except Exception as e:
                print(f"[ERROR] Failed to fit {mod} model in outer fold: {e}")
                print(f"[ERROR] XGBoost: {is_xgb}, Train shape: {X_outer_train.shape if hasattr(X_outer_train, 'shape') else 'unknown'}")
                # Use mean prediction as fallback
                pred_test = np.full(len(outer_test_idx), np.nanmean(y_outer_train))
                test_preds_by_modality[mod] = pred_test
                fitted_pipelines_this_fold[mod] = None
            # Extract selected features if pipeline has corr_selector + rfe_selector
            if hasattr(fitted, "named_steps") and "nan_filter" in fitted.named_steps and "corr_selector" in fitted.named_steps and "rfe_selector" in fitted.named_steps:
                try:
                    feature_names_after_preprocess = list(fitted.named_steps["nan_filter"].features_to_keep_)
                    selected_features_this_fold[mod] = extract_selected_features_late_fusion(fitted, feature_names_after_preprocess)
                except Exception:
                    selected_features_this_fold[mod] = []

        fused_pred = np.sum(
            [weights_final[mod] * test_preds_by_modality[mod] for mod in modality_names],
            axis=0,
        )

        fused_rmse = float(np.sqrt(mean_squared_error(y_outer_test, fused_pred)))
        fused_mae = float(mean_absolute_error(y_outer_test, fused_pred))
        fused_r2 = float(r2_score(y_outer_test, fused_pred))
        fused_mape = _mape(y_outer_test, fused_pred)
        print(f"  Fused test RMSE={fused_rmse:.4f} MAE={fused_mae:.4f} R²={fused_r2:.4f} MAPE={fused_mape:.2f}%")

        all_rmse.append(fused_rmse)
        all_mae.append(fused_mae)
        all_r2.append(fused_r2)
        all_mape.append(fused_mape)

        fold_result: Dict = {
            "fold": fold_index,
            "unimodal_oof_rmse": dict(unimodal_oof_rmse),
            "unimodal_oof_metrics": {mod: dict(metrics) for mod, metrics in unimodal_oof_metrics.items()},
            "weights_final": dict(weights_final),
            "fused_test_rmse": fused_rmse,
            "fused_test_mae": fused_mae,
            "fused_test_r2": fused_r2,
            "fused_test_mape": fused_mape,
            "y_true": y_outer_test.tolist(),
            "y_pred": fused_pred.tolist(),
        }
        if selected_features_this_fold:
            fold_result["selected_features"] = dict(selected_features_this_fold)
        # Store fitted pipelines from last fold for feature importance plot
        if fold_index == outer_splits - 1:
            fold_result["fitted_pipelines"] = fitted_pipelines_this_fold
            fold_result["selected_features_for_importance"] = dict(selected_features_this_fold)
        per_fold.append(fold_result)

    rmse_mean = float(np.mean(all_rmse))
    rmse_std = float(np.std(all_rmse))
    mae_mean = float(np.mean(all_mae))
    mae_std = float(np.std(all_mae))
    r2_mean = float(np.mean(all_r2))
    r2_std = float(np.std(all_r2))
    mape_mean = float(np.nanmean(all_mape))
    mape_std = float(np.nanstd(all_mape))

    print("\n--- Summary ---")
    print("  Metric  | Mean    | Std")
    print("  --------|---------|--------")
    print(f"  RMSE    | {rmse_mean:.4f}  | {rmse_std:.4f}")
    print(f"  MAE     | {mae_mean:.4f}  | {mae_std:.4f}")
    print(f"  R²      | {r2_mean:.4f}  | {r2_std:.4f}")
    print(f"  MAPE(%) | {mape_mean:.2f}  | {mape_std:.2f}")

    return {
        "per_fold": per_fold,
        "aggregate": {
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "r2_mean": r2_mean,
            "r2_std": r2_std,
            "mape_mean": mape_mean,
            "mape_std": mape_std,
        },
    }

# =============================================================================
# Evaluation and Plotting Functions
# =============================================================================

def _adjusted_r2(r2: float, n: int, p: int) -> float:
    """Adjusted R² = 1 - (1-R²)*(n-1)/(n-p-1)."""
    if n <= p + 1:
        return float("nan")
    return float(1 - (1 - r2) * (n - 1) / (n - p - 1))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Avoid division by zero."""
    mask = np.abs(y_true) > 1e-10
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics: MAE, RMSE, MAPE, R², Adjusted R²."""
    y_pred = model.predict(X_test)
    y_true = np.asarray(y_test).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n_samples = len(y_true)
    n_features = X_test.shape[1] if hasattr(X_test, 'shape') else 0

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = _mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = _adjusted_r2(r2, n_samples, n_features)

    metrics: Dict[str, float] = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
        "Adjusted_R2": float(adj_r2),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }
    return metrics


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, model_name: str,
                             save_path: Path, metrics: Optional[Dict[str, float]] = None) -> None:
    """Scatter plot: predicted vs actual. Points should cluster around 45° line."""
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"[WARN] Empty arrays for predicted vs actual: {model_name}")
        return
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", lw=2, label="Perfect prediction (45°)")
    plt.xlabel("Actual PHQ9-Score")
    plt.ylabel("Predicted PHQ9-Score")
    title = f"Predicted vs Actual — {model_name}"
    if metrics:
        r2 = metrics.get("R2", float("nan"))
        mae = metrics.get("MAE", float("nan"))
        title += f"\nR²={r2:.3f}, MAE={mae:.3f}"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved predicted vs actual to {save_path}")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: Path) -> None:
    """Residual plot: residuals vs predicted. Random scatter around zero is ideal."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return
    residuals = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5)
    plt.axhline(0, color="r", linestyle="--", lw=2)
    plt.xlabel("Predicted PHQ9-Score")
    plt.ylabel("Residual (Actual − Predicted)")
    plt.title(f"Residual Plot — {model_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved residual plot to {save_path}")


def plot_residual_distribution(residuals: np.ndarray, model_name: str, save_path: Path) -> None:
    """Histogram + Q-Q plot of residuals to check normality."""
    if len(residuals) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(residuals, bins=15, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Residual Distribution — {model_name}")
    axes[0].axvline(0, color="r", linestyle="--")
    try:
        from scipy import stats as scipy_stats
        scipy_stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q Plot (Normal)")
    except Exception:
        axes[1].text(0.5, 0.5, "Q-Q plot unavailable", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved residual distribution to {save_path}")


# PHQ9 severity bins (standard): 0-4 Minimal, 5-9 Mild, 10-14 Moderate, 15-19 Mod. severe, 20-27 Severe
PHQ9_BINS = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 27)]
PHQ9_BIN_LABELS = ["0-4 (Minimal)", "5-9 (Mild)", "10-14 (Moderate)", "15-19 (Mod. severe)", "20-27 (Severe)"]


def plot_error_by_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path,
    bins: Optional[List[Tuple[int, int]]] = None,
    bin_labels: Optional[List[str]] = None,
) -> None:
    """Bar chart of mean/median absolute error by PHQ9 severity bins (from y_true)."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return
    bins = bins or PHQ9_BINS
    bin_labels = bin_labels or PHQ9_BIN_LABELS
    abs_errors = np.abs(np.asarray(y_true).ravel() - np.asarray(y_pred).ravel())
    y_true_flat = np.asarray(y_true).ravel()
    mae_per_bin: List[float] = []
    median_ae_per_bin: List[float] = []
    used_labels: List[str] = []
    for (lo, hi), label in zip(bins, bin_labels):
        mask = (y_true_flat >= lo) & (y_true_flat <= hi)
        if mask.sum() == 0:
            continue
        mae_per_bin.append(float(np.mean(abs_errors[mask])))
        median_ae_per_bin.append(float(np.median(abs_errors[mask])))
        used_labels.append(f"{label}\n(n={mask.sum()})")
    if not used_labels:
        return
    x = np.arange(len(used_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, mae_per_bin, width, label="Mean AE", color="#3498db")
    ax.bar(x + width / 2, median_ae_per_bin, width, label="Median AE", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(used_labels, rotation=15, ha="right")
    ax.set_ylabel("Absolute Error")
    ax.set_title(f"Error by PHQ9 Severity Bin — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved error by bins to {save_path}")


def plot_oof_dashboard(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Path,
    fusion_type: str = "Late",
) -> None:
    """Single OOF dashboard figure: Pred vs True, Residuals, Residual dist, Error by bins (from concatenated outer-test predictions)."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_true - y_pred
    agg_metrics = {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle(f"OOF Dashboard — {model_name} ({fusion_type} Fusion)\nR²={agg_metrics['R2']:.3f}, MAE={agg_metrics['MAE']:.3f}, RMSE={agg_metrics['RMSE']:.3f}", fontsize=11)

    # (1) Predicted vs True + 45° line
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidth=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", lw=2, label="Perfect (45°)")
    ax.set_xlabel("Actual PHQ9-Score")
    ax.set_ylabel("Predicted PHQ9-Score")
    ax.set_title("Predicted vs Actual (OOF)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) Residuals vs Predicted
    ax = axes[0, 1]
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidth=0.5)
    ax.axhline(0, color="r", linestyle="--", lw=2)
    ax.set_xlabel("Predicted PHQ9-Score")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title("Residuals vs Predicted")
    ax.grid(True, alpha=0.3)

    # (3) Residual distribution (histogram)
    ax_hist = axes[1, 0]
    ax_hist.hist(residuals, bins=15, edgecolor="black", alpha=0.7)
    ax_hist.set_xlabel("Residual")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("Residual Distribution")
    ax_hist.axvline(0, color="r", linestyle="--")
    ax_hist.grid(True, alpha=0.3)

    # (4) Q-Q plot
    try:
        from scipy import stats as scipy_stats
        scipy_stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("Q-Q Plot (Normal)")
    except Exception:
        axes[1, 1].text(0.5, 0.5, "Q-Q plot unavailable", ha="center", va="center", transform=axes[1, 1].transAxes)
    axes[1, 1].grid(True, alpha=0.3)

    # (5) Error by PHQ9 severity bins
    ax_bins = axes[2, 0]
    axes[2, 1].axis("off")
    abs_errors = np.abs(residuals)
    mae_per_bin = []
    median_ae_per_bin = []
    used_labels = []
    for (lo, hi), label in zip(PHQ9_BINS, PHQ9_BIN_LABELS):
        mask = (y_true >= lo) & (y_true <= hi)
        if mask.sum() == 0:
            continue
        mae_per_bin.append(float(np.mean(abs_errors[mask])))
        median_ae_per_bin.append(float(np.median(abs_errors[mask])))
        used_labels.append(f"{label}\n(n={mask.sum()})")
    if used_labels:
        x = np.arange(len(used_labels))
        width = 0.35
        ax_bins.bar(x - width / 2, mae_per_bin, width, label="Mean AE", color="#3498db")
        ax_bins.bar(x + width / 2, median_ae_per_bin, width, label="Median AE", color="#e74c3c")
        ax_bins.set_xticks(x)
        ax_bins.set_xticklabels(used_labels, rotation=15, ha="right")
        ax_bins.set_ylabel("Absolute Error")
        ax_bins.set_title("Error by PHQ9 Severity Bin")
        ax_bins.legend()
        ax_bins.grid(True, alpha=0.3, axis="y")
    else:
        ax_bins.text(0.5, 0.5, "No data in bins", ha="center", va="center", transform=ax_bins.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved OOF dashboard to {save_path}")


def plot_rmse_comparison(
    unimodal_rmse: Dict[str, float],
    fused_rmse: float,
    model_name: str,
    save_path: Path,
) -> None:
    """Bar chart comparing RMSE across individual modalities and fused model."""
    labels = list(unimodal_rmse.keys()) + ["Fused"]
    values = [unimodal_rmse[m] for m in unimodal_rmse] + [fused_rmse]
    colors = ["#3498db"] * len(unimodal_rmse) + ["#e74c3c"]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(labels)), values, color=colors)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("RMSE")
    plt.title(f"RMSE Comparison — {model_name} (Late Fusion)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved RMSE comparison plot to {save_path}")


def plot_feature_importance(pipe: Pipeline, feature_names: List[str], model_name: str,
                            save_path: Path) -> None:
    """Bar chart of feature importance or coefficients."""
    reg = pipe.named_steps.get("reg", None)
    if reg is None:
        # Try final estimator
        reg = pipe.steps[-1][1]
    imp = None
    if hasattr(reg, "coef_"):
        imp = np.abs(reg.coef_).ravel()
    elif hasattr(reg, "feature_importances_"):
        imp = reg.feature_importances_
    else:
        print(f"[INFO] No feature importance available for {model_name}")
        return
    if len(imp) != len(feature_names):
        return
    idx = np.argsort(imp)[::-1][:min(25, len(imp))]
    names = [feature_names[i] for i in idx]
    vals = [imp[i] for i in idx]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(names)), vals, align="center")
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.xlabel("Importance / |Coefficient|")
    plt.title(f"Feature Importance — {model_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved feature importance to {save_path}")


def plot_feature_importance_late_fusion(
    fitted_pipelines: Dict[str, object],
    selected_features_by_mod: Dict[str, List[str]],
    model_name: str,
    save_path: Path,
) -> None:
    """Bar chart of feature importance across modalities (Late Fusion). Combines top features from each modality."""
    all_names: List[str] = []
    all_vals: List[float] = []
    modality_prefix_map = {"video": "Video", "audio": "Audio", "ecg": "ECG", "emg": "EMG", "rsp": "RSP", "text": "Text"}
    for mod, pipe in fitted_pipelines.items():
        sel_names = selected_features_by_mod.get(mod, [])
        if not sel_names:
            continue
        # Handle GridSearchCV (e.g. text modality) -> use best_estimator_
        if hasattr(pipe, "best_estimator_"):
            pipe = pipe.best_estimator_
        reg = getattr(pipe, "named_steps", None)
        if reg is not None:
            reg = reg.get("reg", pipe.steps[-1][1] if hasattr(pipe, "steps") and pipe.steps else None)
        else:
            reg = pipe.steps[-1][1] if hasattr(pipe, "steps") and pipe.steps else None
        if reg is None:
            continue
        imp = None
        if hasattr(reg, "coef_"):
            imp = np.abs(reg.coef_).ravel()
        elif hasattr(reg, "feature_importances_"):
            imp = reg.feature_importances_
        if imp is None or len(imp) != len(sel_names):
            continue
        prefix = modality_prefix_map.get(mod, mod.capitalize())
        idx = np.argsort(imp)[::-1][:min(10, len(imp))]
        for i in idx:
            all_names.append(f"{prefix}_{sel_names[i]}")
            all_vals.append(float(imp[i]))
    if not all_names:
        print(f"[INFO] No feature importance available for {model_name} (Late Fusion)")
        return
    # Take top 25 overall
    top_idx = np.argsort(all_vals)[::-1][:min(25, len(all_vals))]
    names = [all_names[i] for i in top_idx]
    vals = [all_vals[i] for i in top_idx]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(names)), vals, align="center")
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.xlabel("Importance / |Coefficient|")
    plt.title(f"Feature Importance — {model_name} (Late Fusion)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved feature importance to {save_path}")


# =============================================================================
# Main Function (Late Fusion Only) #############################################
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="MultiModal ML: late fusion")
    p.add_argument(
        "--fusion_modalities",
        nargs="+",
        choices=["audio", "video", "ecg", "emg", "rsp", "text"],
        help="Modalities to use in late fusion. Valid: audio, video, ecg, emg, rsp, text",
    )

    p.add_argument("--condition", type=str, default="ADK", help="Condition filter: CR, ADK, CRADK, SHAM, all") # CR, ADK, CRADK, SHAM, all
    p.add_argument("--phases", nargs="+", default=["training_pos"], help="Phases to load (standard names)") # training_pos, training_neg, testing_pos, testing_neg
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

    p.add_argument("--regressors", nargs="*", default=None,
                   help="Regression models: LinearRegression, RandomForest, AdaBoost, DecisionTree, KNN, XGBoost. Default: all")
    p.add_argument("--outer_folds", type=int, default=5, help="Outer GroupKFold splits")
    p.add_argument("--inner_folds", type=int, default=3, help="Inner GroupKFold splits for HPO")

    p.add_argument("--nan_threshold", type=float, default=0.69,
                   help="Drop features with more than this fraction of NaN (rest imputed)")
    p.add_argument("--corr_k", type=int, default=50,
                   help="Late fusion: top K from CorrelationSelector, then RFE selects rfe_n")
    p.add_argument("--rfe_n", type=int, default=25, help="Late fusion: number of features to select by RFE after CorrelationSelector")
    p.add_argument("--weight_method", type=str, default="optimize", choices=["inverse_rmse", "optimize"],
                   help="Fusion weight method: inverse_rmse or optimize")

    p.add_argument("--output_dir", type=str, default="/home/vault/empkins/tpD/D02/Students/Yasaman/5_MultiModal_ML/Late Fusion Regression Results",
                   help="Base output directory. Results saved in: <output_dir>/Late_Fusion_{mod}_{cond}_{phases}_{agg}/ (default: script dir)")
    args = p.parse_args()
    return args


def main(cfg) -> None:
    """
    Late fusion regression via run_late_fusion_regression_cv: load modalities, prefix feature names
    (Video_, Audio_, ECG_, EMG_, RSP_, Text_), align by participant ID (one row per ID), then
    run nested CV with weighted late fusion. Weights computed from OOF RMSE (inverse_rmse or optimize).
    Pipeline per modality: preprocess + CorrelationSelector(corr_k) + RFE(rfe_n) + regressor.
    Saves aggregate summary, per-fold results, and RFE-selected features per fold in JSON/CSV.
    """
    modalities = [m.strip().lower() for m in cfg.fusion_modalities]
    outer_folds = cfg.outer_folds
    inner_folds = cfg.inner_folds
    nan_threshold = cfg.nan_threshold
    corr_k = getattr(cfg, "corr_k", 50)
    rfe_n = getattr(cfg, "rfe_n", 10)
    weight_method = getattr(cfg, "weight_method", "optimize")
    opensmile_dir = Path(cfg.opensmile_data_dir)
    opendbm_dir = Path(cfg.opendbm_data_dir)
    condition = cfg.condition
    phases = cfg.phases if isinstance(cfg.phases, list) else [cfg.phases]
    aggregation_method = cfg.aggregation_method

    has_text = "text" in modalities
    base_modalities = [m for m in modalities if m != "text"]

    if has_text and not base_modalities:
        raise ValueError("Text modality requires at least one other modality to provide labels.")

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
        y = df["label"].to_numpy(dtype=np.float64)
        groups = df["ID"].astype(str).to_numpy()
        modality_data[mod] = {"X": X, "y": y, "groups": groups, "df": df}

    # Load text modality (by ID only, no aggregation), then merge labels/condition from first base modality
    if has_text:
        text_path = Path(cfg.text_data_csv)
        df_text = load_text_data(text_path)
        label_source_mod = base_modalities[0]
        base_df = modality_data[label_source_mod]["df"]
        if "ID" not in base_df.columns or "label" not in base_df.columns:
            raise ValueError(f"Base modality '{label_source_mod}' must contain 'ID' and 'label' columns to attach labels to text data.")
        label_cols = ["ID", "label"]
        if "condition" in base_df.columns:
            label_cols.append("condition")
        label_df = base_df[label_cols].drop_duplicates(subset=["ID"])
        # Drop overlapping cols from df_text so merge yields single label/condition from base modality
        drop_before_merge = [c for c in ["label", "condition"] if c in df_text.columns and c in label_df.columns]
        if drop_before_merge:
            df_text = df_text.drop(columns=drop_before_merge)
        df_text = df_text.merge(label_df, on="ID", how="inner")
        if df_text.empty:
            raise ValueError("No overlapping participants between text data and base modalities.")
        drop_cols_text = [c for c in ["label", "ID", "phase", "condition", "aufgabe", "Aufgabe"] if c in df_text.columns]
        X_text = df_text.drop(columns=drop_cols_text, errors="ignore").select_dtypes(include=[np.number])
        y_text = df_text["label"].to_numpy(dtype=np.float64)
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
        d["y_aligned"] = df_mod["label"].to_numpy(dtype=np.float64)
        d["groups_aligned"] = df_mod["ID"].astype(str).to_numpy()

    # Use first modality as reference for y and groups
    y = modality_data[effective_modalities[0]]["y_aligned"]
    groups = modality_data[effective_modalities[0]]["groups_aligned"]
    if len(np.unique(groups)) < outer_folds:
        raise ValueError(f"Not enough groups for {outer_folds}-fold GroupKFold")

    X_by_modality = {mod: modality_data[mod]["X_aligned"] for mod in effective_modalities}

    # Prepare regressors to run
    models = get_regression_models()
    if cfg.regressors:
        # cfg.regressors may be a single string or list; normalize to list
        if isinstance(cfg.regressors, str):
            requested = [cfg.regressors]
        else:
            requested = list(cfg.regressors)
        models = {k: v for k, v in models.items() if k in requested}
    if not models:
        raise ValueError("No valid regressors selected")

    # Save run configuration once at the top-level directory
    cfg_dict = {k: getattr(cfg, k) for k in ["fusion_modalities", "condition", "phases", "minutes", "outer_folds", "inner_folds", "nan_threshold", "corr_k", "rfe_n", "aggregation_method", "weight_method"] if hasattr(cfg, k)}
    cfg_dict["target"] = "PHQ9-Score"
    for k, v in list(cfg_dict.items()):
        if isinstance(v, (Path, set)):
            cfg_dict[k] = str(v)
        elif isinstance(v, np.ndarray):
            cfg_dict[k] = v.tolist()
    with open(out_dir / f"{base_name}_args.json", "w") as fp:
        json.dump(cfg_dict, fp, indent=2)

    # Run late fusion separately for each regressor, storing results in subfolders
    for model_name, reg_base in models.items():
        reg = reg_base

        def _make_build_fn(r):
            def build_estimator_fn(modality_name: str) -> Pipeline:
                # Special case: Text modality – no ConstantFilter/NaNFilter, only RFE
                if modality_name == "text":
                    rfe_estimator = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE)
                    text_pipe = Pipeline([
                        ("rfe_selector", RFE(estimator=rfe_estimator, step=4)),
                        ("reg", clone(r)),
                    ])
                    param_grid_text = {
                        "rfe_selector__n_features_to_select": [4, 8, 12, 14],
                    }
                    return GridSearchCV(
                        text_pipe,
                        param_grid_text,
                        cv=inner_folds,
                        scoring="neg_mean_squared_error",
                        refit=True,
                        n_jobs=1,
                        error_score="raise",
                    )

                # Default: full late-fusion pipeline with preprocessing + CorrelationSelector + RFE
                return build_full_late_fusion_pipeline(
                    nan_threshold=nan_threshold,
                    corr_k=corr_k,
                    rfe_n=rfe_n,
                    reg=clone(r),
                )
            return build_estimator_fn

        build_estimator_fn = _make_build_fn(reg)

        model_dir = out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] === Regressor: {model_name} ===")
        print(f"[INFO] Late fusion modalities: {effective_modalities} (feature names prefixed: Video_, Audio_, ECG_, EMG_, RSP_, Text_)")
        print(f"[INFO] Pipeline (non-text): preprocess + CorrelationSelector({corr_k}) + RFE({rfe_n}); regressor: {model_name}")
        if has_text:
            print(f"[INFO] Text pipeline: RFE(step=4) with GridSearch over n_features_to_select=[4, 8, 12, 14]; regressor: {model_name}")

        result = run_late_fusion_regression_cv(
            X_by_modality=X_by_modality,
            y=y,
            groups=groups,
            build_estimator_fn=build_estimator_fn,
            outer_splits=outer_folds,
            inner_splits=inner_folds,
            weight_method=weight_method,
        )

        agg = result["aggregate"]

        # ------------------------------------------------------------------
        # Per-fold results (results.csv)
        # ------------------------------------------------------------------
        per_fold_rows = []
        for f in result["per_fold"]:
            sel_count_total = None
            if "selected_features" in f and isinstance(f["selected_features"], dict):
                sel_count_total = sum(len(v) for v in f["selected_features"].values())

            per_fold_rows.append(
                {
                    "fold": f["fold"],
                    "model": model_name,
                    "rmse": f["fused_test_rmse"],
                    "mae": f["fused_test_mae"],
                    "r2": f["fused_test_r2"],
                    "mape": f["fused_test_mape"],
                    "selected_features_total": sel_count_total,
                }
            )

        pd.DataFrame(per_fold_rows).to_csv(model_dir / "results.csv", index=False)

        # ------------------------------------------------------------------
        # Final summary (final_results.csv)
        # ------------------------------------------------------------------
        n_samples = len(y)
        final_row = {
            "model": model_name,
            "rmse_mean": agg["rmse_mean"],
            "rmse_std": agg["rmse_std"],
            "mae_mean": agg["mae_mean"],
            "mae_std": agg["mae_std"],
            "r2_mean": agg["r2_mean"],
            "r2_std": agg["r2_std"],
            "mape_mean": agg["mape_mean"],
            "mape_std": agg["mape_std"],
            "n_samples": n_samples,
        }

        final_df = pd.DataFrame([final_row])
        final_df.to_csv(model_dir / "final_results.csv", index=False)

        # Bar chart: RMSE comparison (mean unimodal OOF RMSE and fused test RMSE across folds)
        unimodal_rmse_mean = {}
        for mod in effective_modalities:
            vals = [f["unimodal_oof_rmse"][mod] for f in result["per_fold"]]
            unimodal_rmse_mean[mod] = float(np.mean(vals))
        plot_rmse_comparison(
            unimodal_rmse_mean,
            agg["rmse_mean"],
            model_name,
            model_dir / f"{base_name}_rmse_comparison.png",
        )

        # Regression plots (OOF = concatenated outer-test predictions, one per sample)
        y_true_all: List[float] = []
        y_pred_all: List[float] = []
        for f in result["per_fold"]:
            y_true_all.extend(f.get("y_true", []))
            y_pred_all.extend(f.get("y_pred", []))
        y_true_plot = np.array(y_true_all)
        y_pred_plot = np.array(y_pred_all)
        if len(y_true_plot) > 0 and len(y_pred_plot) > 0:
            agg_metrics = {
                "MAE": float(mean_absolute_error(y_true_plot, y_pred_plot)),
                "RMSE": float(np.sqrt(mean_squared_error(y_true_plot, y_pred_plot))),
                "R2": float(r2_score(y_true_plot, y_pred_plot)),
            }
            # OOF dashboard (single figure for thesis/paper)
            plot_oof_dashboard(
                y_true_plot, y_pred_plot, model_name,
                save_path=model_dir / "oof_dashboard.png",
                fusion_type="Late",
            )
            plot_predicted_vs_actual(
                y_true_plot, y_pred_plot, model_name,
                save_path=model_dir / "predicted_vs_actual.png",
                metrics=agg_metrics,
            )
            plot_residuals(y_true_plot, y_pred_plot, model_name,
                          save_path=model_dir / "residual_plot.png")
            residuals = y_true_plot - y_pred_plot
            plot_residual_distribution(residuals, model_name,
                                      save_path=model_dir / "residual_distribution.png")
            plot_error_by_bins(y_true_plot, y_pred_plot, model_name,
                              save_path=model_dir / "error_by_bins.png")

        # Feature importance (from last fold's fitted pipelines per modality)
        last_fold = next((f for f in result["per_fold"] if "fitted_pipelines" in f), None)
        if last_fold and "fitted_pipelines" in last_fold and "selected_features_for_importance" in last_fold:
            plot_feature_importance_late_fusion(
                last_fold["fitted_pipelines"],
                last_fold["selected_features_for_importance"],
                model_name,
                save_path=model_dir / "feature_importance.png",
            )

        for f in result["per_fold"]:
            fold_idx = f["fold"]
            to_save = {
                "unimodal_oof_rmse": f["unimodal_oof_rmse"],
                "unimodal_oof_metrics": f["unimodal_oof_metrics"],
                "weights_final": f["weights_final"],
            }
            if "selected_features" in f:
                to_save["selected_features"] = f["selected_features"]
            feat_path = model_dir / f"{base_name}_selected_features_fold{fold_idx}.json"
            with open(feat_path, "w") as fp:
                json.dump(to_save, fp, indent=2)
            print(f"[INFO] Saved selected features (RFE per modality) for {model_name} to {feat_path}")

        print(f"[INFO] Late fusion regression done for {model_name}. RMSE={agg['rmse_mean']:.4f}±{agg['rmse_std']:.4f}, R²={agg['r2_mean']:.4f}±{agg['r2_std']:.4f}")
        print(f"[INFO] Results and selected features saved under {model_dir}")


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)